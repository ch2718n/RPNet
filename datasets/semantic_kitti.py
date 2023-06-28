from pathlib import Path

import numpy as np
import torch

splits = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    "val": [8],
    "test": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}


class SemanticKitti(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path, split: str, num_vote=1) -> None:
        self.split = split
        self.seqs = splits[split]
        self.dataset_dir = dataset_dir
        self.sweeps = []
        self.num_vote = num_vote
        self.lt_classes = {1: 0.3, 2: 1, 3: 1, 4: 0.3, 5: 0.5, 6: 1, 7: 1, 8: 1,
                           16: 0.3, 18: 0.3, 19: 0.5}
        self.nlt_classes = {9: 0.24, 11: 0.18, 13: 0.15, 15: 0.34, 17: 0.09}
        self.means = [-0.0957, 0.5058, -1.0653, 0.2750, 11.7939]
        self.std = [12.2258, 8.9893, 0.8299, 0.1456, 9.8732]
        self.proj_size = [64, 2048]
        self.trans_std = [0.1, 0.1, 0.1]
        self.rotate_aug = 1
        self.flip_aug = 1
        self.scale_aug = 1
        self.transform_aug = 1
        self.mix_aug = 1
        self.shift_aug = 1
        self.max_points = 120000

        for seq in self.seqs:
            seq_str = f"{seq:0>2}"
            seq_path = dataset_dir / seq_str / "velodyne"
            for sweep in seq_path.iterdir():
                self.sweeps.append((seq_str, sweep.stem))

    def get_points(self, index):
        seq, sweep = self.sweeps[index]
        sweep_file = self.dataset_dir / seq / "velodyne" / f"{sweep}.bin"
        points = np.fromfile(sweep_file.as_posix(), dtype=np.float32)
        points = points.reshape((-1, 4))
        if self.split != "test":
            labels_file = self.dataset_dir / seq / "labels" / f"{sweep}.label"
            labels = np.fromfile(labels_file.as_posix(), dtype=np.int32)
            labels = labels.reshape((-1))
            labels &= 0xFFFF
            labels = np.vectorize(learning_map.get)(labels)
        else:
            labels = np.zeros((points.shape[0],))

        return points, labels, seq, sweep

    def do_range_projection(self, points):
        # laser parameters
        fov_up = 3 / 180.0 * np.pi  # field of view up in rad
        fov_down = -25 / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]
        depth = points[:, 4]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_size[1]  # in [0.0, W]
        proj_y *= self.proj_size[0]  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_size[1] - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        # random shift
        if (self.split == "train") | (self.num_vote > 1):
            if np.random.random() < self.shift_aug:
                shift = np.random.randint(0, self.proj_size[1])
                proj_x = (proj_x + shift) % self.proj_size[1]
        px = np.copy(proj_x)  # store a copy in original order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_size[0] - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        py = np.copy(proj_y)  # store a copy in original order

        proj_points = np.zeros((self.proj_size[0], self.proj_size[1], 5))
        proj_points[proj_y, proj_x] = points

        return proj_points, py, px

    def augmentation(self, points, labels, points_mix, labels_mix):
        if np.random.random() < self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            points[:, :2] = np.dot(points[:, :2], j)
        if np.random.random() < self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                points[:, 0] = -points[:, 0]
            elif flip_type == 2:
                points[:, 1] = -points[:, 1]
            elif flip_type == 3:
                points[:, :2] = -points[:, :2]
        if np.random.random() < self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            points[:, 0] = noise_scale * points[:, 0]
            points[:, 1] = noise_scale * points[:, 1]
        if np.random.random() < self.transform_aug:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T
            points[:, 0:3] += noise_translate

        if np.random.random() < self.mix_aug:
            for paste_class, v in self.lt_classes.items():
                if np.random.random() < v:
                    points_paste = points_mix[np.where(labels_mix == paste_class)]
                    labels_paste = labels_mix[np.where(labels_mix == paste_class)]
                    points = np.concatenate([points, points_paste], axis=0)
                    labels = np.concatenate([labels, labels_paste], axis=0)
        while points.shape[0] > self.max_points:
            delete_class = np.random.choice(list(self.nlt_classes.keys()),
                                            p=list(self.nlt_classes.values()))
            delete_index = np.where(labels == delete_class)
            if delete_index[0].shape[0] > (points.shape[0] - self.max_points):
                delete_index = np.random.choice(list(delete_index[0]), size=(points.shape[0] - self.max_points))

            points = np.delete(points, delete_index, axis=0)
            labels = np.delete(labels, delete_index, axis=0)
        return points, labels

    def transform(self, points, points_proj, labels, py, px):
        py = 2 * (py / self.proj_size[0] - 0.5)
        px = 2 * (px / self.proj_size[1] - 0.5)

        if self.split == "train":
            if px.shape[0] < self.max_points:
                pad_len = self.max_points - px.shape[0]
                px = np.hstack([px, np.zeros((pad_len,))])
                py = np.hstack([py, np.zeros((pad_len,))])
                labels = np.hstack([labels, np.zeros((pad_len,))])
                points = np.vstack([points, np.zeros((pad_len, 5))])
            labels = np.hstack([labels, np.ones((0,))])
            points = np.vstack([points, np.zeros((0, 5))])

        # normalize
        mask_range = points_proj[:, :, 4] != 0
        mask_points = points[:, 4] != 0
        for i in range(5):
            points_proj[:, :, i] = (points_proj[:, :, i] - self.means[i]) / self.std[i] * mask_range
            points[:, i] = (points[:, i] - self.means[i]) / self.std[i] * mask_points

        return points, points_proj, labels, py, px

    def __getitem__(self, index):
        points, labels, seq, sweep = self.get_points(index)
        if self.split == "train":
            index_mix = np.random.randint(0, len(self.sweeps))
            points_mix, label_mix, _, _ = self.get_points(index_mix)
            points, labels = self.augmentation(points, labels, points_mix, label_mix)

        if self.num_vote == 1:
            depth = np.linalg.norm(points[:, :3], 2, axis=1)
            points = np.concatenate((points, depth[:, np.newaxis]), axis=1)
            points_proj, py, px = self.do_range_projection(points)
            points, points_proj, labels, py, px = self.transform(points, points_proj, labels, py, px)
            px = px[np.newaxis, :]
            py = py[np.newaxis, :]
            labels = labels[np.newaxis, :]
            points_proj = points_proj.transpose(2, 0, 1).astype(np.float32)
        else:
            points_proj_vote = []
            px_vote = []
            py_vote = []
            points_vote = []
            for i in range(self.num_vote):
                depth0 = np.linalg.norm(points[:, :3], 2, axis=1)
                points0 = np.concatenate((points, depth0[:, np.newaxis]), axis=1)
                points_proj0, py0, px0 = self.do_range_projection(points0)
                points0, points_proj0, labels, py0, px0 = self.transform(points0, points_proj0, labels, py0, px0)

                points_vote.append(np.expand_dims(points0, axis=0))
                points_proj_vote.append(np.expand_dims(points_proj0.transpose(2, 0, 1).astype(np.float32), axis=0))
                px_vote.append(np.expand_dims(px0[np.newaxis, :], axis=0))
                py_vote.append(np.expand_dims(py0[np.newaxis, :], axis=0))

            points_proj = np.concatenate(points_proj_vote, axis=0)
            points = np.concatenate(points_vote, axis=0)
            labels = labels[np.newaxis, :]
            px = np.concatenate(px_vote, axis=0)
            py = np.concatenate(py_vote, axis=0)

        res = {
            "points_proj": points_proj,
            "px": px,
            "py": py,
            "points": points,
            "labels": labels,
        }
        if self.split in ["test", "val"]:
            res["seq"] = seq
            res["sweep"] = sweep
        return res

    def __len__(self):
        return len(self.sweeps)


learning_map = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,  # "car"
    11: 2,  # "bicycle"
    13: 5,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,  # "motorcycle"
    16: 5,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,  # "truck"
    20: 5,  # "other-vehicle"
    30: 6,  # "person"
    31: 7,  # "bicyclist"
    32: 8,  # "motorcyclist"
    40: 9,  # "road"
    44: 10,  # "parking"
    48: 11,  # "sidewalk"
    49: 12,  # "other-ground"
    50: 13,  # "building"
    51: 14,  # "fence"
    52: 0,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,  # "lane-marking" to "road" ---------------------------------mapped
    70: 15,  # "vegetation"
    71: 16,  # "trunk"
    72: 17,  # "terrain"
    80: 18,  # "pole"
    81: 19,  # "traffic-sign"
    99: 0,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,  # "moving-car" to "car" ------------------------------------mapped
    253: 7,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,  # "moving-person" to "person" ------------------------------mapped
    255: 8,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,  # "moving-truck" to "truck" --------------------------------mapped
    259: 5,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}
class_names = [
    "unlabeled",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]
map_inv = {
    0: 0,
    1: 10,  # "car"
    2: 11,  # "bicycle"
    3: 15,  # "motorcycle"
    4: 18,  # "truck"
    5: 20,  # "other-vehicle"
    6: 30,  # "person"
    7: 31,  # "bicyclist"
    8: 32,  # "motorcyclist"
    9: 40,  # "road"
    10: 44,  # "parking"
    11: 48,  # "sidewalk"
    12: 49,  # "other-ground"
    13: 50,  # "building"
    14: 51,  # "fence"
    15: 70,  # "vegetation"
    16: 71,  # "trunk"
    17: 72,  # "terrain"
    18: 80,  # "pole"
    19: 81,  # "traffic-sign
}

color_map = {  # bgr
    0: [0, 0, 0],
    1: [0, 0, 255],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [245, 150, 100],
    256: [255, 0, 0],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0],
}

train_color_map = {i: color_map[j] for i, j in map_inv.items()}
