import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
from datasets import semantic_kitti
from models.segformer.model import SegFormer
import utils

parser = argparse.ArgumentParser("Train on semantic kitti")
parser.add_argument("--config",
                    default='./configs/semantic_kitti.yaml',
                    type=str)
args = parser.parse_args()


def run_val(model, val_loader, n_iter, writer):
    print("\nRunning validation")
    model.eval()
    inference_time = []
    eval_metric = utils.eval_seg.iouEval(n_classes=20, device="cuda:0", ignore=0)
    with torch.no_grad():
        for step, items in tqdm(enumerate(val_loader)):
            images = items["points_proj"].cuda(0, non_blocking=True)
            labels = items["labels"].long().cuda(0, non_blocking=True)
            py = items["py"].float().cuda(0, non_blocking=True)
            px = items["px"].float().cuda(0, non_blocking=True)
            points = items["points"].float().cuda(0, non_blocking=True)

            torch.cuda.synchronize()
            time_start = time.time()

            images = utils.completion.nearest_completion(images)
            predictions = model(images, px, py, points)

            torch.cuda.synchronize()
            time_end = time.time()
            inference_time.append(time_end - time_start)

            _, predictions_argmax = torch.max(predictions, 1)
            eval_metric.addBatch(predictions_argmax, labels)

        miou, ious = eval_metric.getIoU()
        print(f"Iteration {n_iter} , mIou {miou}")
        print(f"Per class Ious {ious}")
        print(f"FPS:{len(inference_time) / sum(inference_time)}")
        writer.add_scalar("val/mIoU", miou, n_iter)
    return miou


def train(cfg):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model = SegFormer(backbone="mit_b1")
    if cfg["pretrained_path"] is not None:
        state_dict = torch.load(cfg["pretrained_path"], map_location='cpu')["state_dict"]
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict, )

    device = torch.device('cuda:0')
    writer = SummaryWriter(log_dir=cfg["save_dir"], flush_secs=20)
    model.to(device)

    # print(model.parameters)
    train_dataset = semantic_kitti.SemanticKitti(
        Path(cfg["dataset_dir"]) / "dataset/sequences", "train",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        num_workers=4,
        drop_last=True,
        shuffle=True,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=semantic_kitti.SemanticKitti(
            Path(cfg["dataset_dir"]) / "dataset/sequences", "val",
        ),
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    loss_fn = utils.ohem.OhemCrossEntropy(ignore_index=0, thresh=0.9, min_kept=40000).to(device)
    lovasz_loss = utils.lovasz_loss.Lovasz_softmax(ignore=0).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate_init"], betas=(0.9, 0.999), weight_decay=cfg["weight_decay"]
    )

    scheduler = utils.cosine_schedule.CosineAnnealingWarmUpRestarts(
        optimizer, T_0=cfg["epoch"] * len(train_loader), T_mult=1, eta_max=cfg["learning_rate_max"],
        T_up=len(train_loader) * cfg["warmup_epoch"], gamma=1.0
    )

    if cfg["checkpoint_path"] is not None:
        checkpoint = torch.load(cfg["checkpoint_path"], map_location="cuda")
        model.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"]),
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        n_iter = checkpoint["n_iter"]
        best_mIoU = checkpoint["best_mIoU"]
    else:
        start_epoch = 0
        n_iter = 0
        best_mIoU = 0

    for epoch in range(start_epoch + 1, cfg["epoch"] + 1):
        model.train()
        for step, items in enumerate(train_loader):
            images = items["points_proj"].to(device, non_blocking=True)
            labels = items["labels"].long().to(device, non_blocking=True)
            py = items["py"].float().to(device, non_blocking=True)
            px = items["px"].float().to(device, non_blocking=True)
            points = items["points"].float().to(device, non_blocking=True)

            images = utils.completion.nearest_completion(images)
            predictions, aux2, aux3, aux4 = model(images, px, py, points)
            aux_loss = \
                loss_fn(aux2, labels) + lovasz_loss(aux2.softmax(1), labels) + \
                loss_fn(aux3, labels) + lovasz_loss(aux3.softmax(1), labels) + \
                loss_fn(aux4, labels) + lovasz_loss(aux4.softmax(1), labels)
            main_loss = loss_fn(predictions, labels) + lovasz_loss(predictions.softmax(1), labels)
            loss = main_loss + aux_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

            print(
                f"Epoch: {epoch} Iteration: {step} / {len(train_loader)} Loss: {loss.item()}", end='\r'
            )
            writer.add_scalar("loss/train", loss.item(), n_iter)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], n_iter)
            n_iter += 1
            scheduler.step()

        mIoU = run_val(model, val_loader, n_iter, writer)
        if mIoU > best_mIoU:
            best_mIoU = mIoU
            torch.save(
                model.state_dict(), Path(cfg["save_dir"]) / f"best.pth"
            )
        if epoch > 90:
            save_name = str(epoch) + ".pth"
            torch.save(
                model.state_dict(), Path(cfg["save_dir"]) / save_name
            )
        torch.save(
            {
                'epoch': epoch,
                'n_iter': n_iter,
                'best_mIoU': best_mIoU,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            },
            Path(cfg["save_dir"]) / "last.pth"
        )
        print(f"best_IoU:{best_mIoU}")


def main() -> None:
    cfg = yaml.safe_load(open(args.config, 'r'))
    train(cfg=cfg)


if __name__ == "__main__":
    main()
