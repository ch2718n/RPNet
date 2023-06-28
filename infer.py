import argparse
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from datasets.semantic_kitti import (
    SemanticKitti,
    class_names,
    map_inv,
    splits,
)
import utils
from models.segformer.model import SegFormer

parser = argparse.ArgumentParser("Run lidar bug inference")
parser.add_argument("--checkpoint-path",
                    default="./checkpoints/best.pth", type=Path)
parser.add_argument("--output-path",
                    default="./checkpoints/", type=Path)
parser.add_argument("--semantic-kitti-dir",
                    default="../SemanticKITTI/dataset/sequences/", type=Path)
parser.add_argument("--split", default="val", type=str)

args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SemanticKitti(args.semantic_kitti_dir, args.split, num_vote=1)
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
    )
    for seq in splits[args.split]:
        seq_dir = args.output_path / "sequences" / f"{seq:0>2}" / "predictions"
        seq_dir.mkdir(parents=True, exist_ok=True)
    model = SegFormer(backbone="mit_b1")
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
    print("Running validation")
    model.eval()
    eval_metric = utils.eval_seg.iouEval(n_classes=20, device=device, ignore=0)
    inference_time = []
    with torch.no_grad():
        for step, items in tqdm(enumerate(loader), total=len(loader)):
            images = items["points_proj"].cuda(0, non_blocking=True)
            labels = items["labels"].long().cuda(0, non_blocking=True)
            py = items["py"].float().cuda(0, non_blocking=True)
            px = items["px"].float().cuda(0, non_blocking=True)
            points = items["points"].float().cuda(0, non_blocking=True)

            torch.cuda.synchronize()
            time_start = time.time()

            if dataset.num_vote > 1:
                images = images[0]
                px = px[0]
                py = py[0]
                points = points[0]

            images = utils.completion.nearest_completion(images)
            predictions = model(images, px, py, points)

            predictions = torch.mean(predictions, dim=0, keepdim=True)
            _, predictions_argmax = torch.max(predictions, 1)

            torch.cuda.synchronize()
            time_end = time.time()
            inference_time.append(time_end - time_start)

            eval_metric.addBatch(predictions_argmax, labels)

            predictions_points = predictions_argmax.cpu().numpy()
            predictions_points = np.vectorize(map_inv.get)(predictions_points).astype(
                np.uint32
            )
            seq, sweep = items["seq"][0], items["sweep"][0]
            out_file = (
                args.output_path
                / "sequences"
                / f"{seq}"
                / "predictions"
                / f"{sweep}.label"
            )

            predictions_points.tofile(out_file.as_posix())

        miou, ious = eval_metric.getIoU()
        print(f"mIou {miou}")
        print("Mean time: ", sum(inference_time) / len(inference_time))
        print("FPS: ", len(inference_time) / sum(inference_time))
        for class_name, iou in zip(class_names, ious):
            print(f"{class_name}: {iou}")


if __name__ == "__main__":
    main()
