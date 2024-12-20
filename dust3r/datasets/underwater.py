# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed arkitscenes
# dataset at https://github.com/apple/ARKitScenes - Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License https://github.com/apple/ARKitScenes/tree/main?tab=readme-ov-file#license
# See datasets_preprocess/preprocess_arkitscenes.py
# --------------------------------------------------------
import os.path as osp
import cv2
import numpy as np
import torch

from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy.spatial import cKDTree


class UnderWaterDataset(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        if split == "train":
            self.split = "train"
        elif split == "test":
            self.split = "test"
        else:
            raise ValueError("")

        self.intrinsic_cameras_path = osp.join(self.ROOT, 'sparse_txt', 'cameras.txt')
        self.cameras_path = osp.join(self.ROOT, 'sparse_txt', 'images.txt')
        self.images_path = osp.join(self.ROOT, 'images')
        self.depthmap_path = osp.join(self.ROOT, 'stereo', 'depth_maps')

        self.intrinsic_camera = self._load_intrinsic_camera_params(self.intrinsic_cameras_path)
        self.image_poses = self._load_image_poses(self.cameras_path)

        if split == "train":
            self.split_data = self._get_split_data(train=True)
        elif split == "test":
            self.split_data = self._get_split_data(train=False)
        else:
            raise ValueError("split must be either 'train' or 'test'")

        # print(self.split_data)

        self.selected_pairs = self._create_pairs_indexes()

    def _get_split_data(self, train=True):
        """
        Split the data into training and testing subsets (80/20 split)
        """
        # image_ids = list(self.image_poses.keys())
        image_ids = sorted(self.image_poses.keys())

        split_index = int(0.8 * len(image_ids))
        if train:
            return image_ids[:split_index]  # First 80% for training
        else:
            return image_ids[split_index:]  # Remaining 20% for testing

    def _load_intrinsic_camera_params(self, cameras_path):
        camera_params = {}
        with open(cameras_path, 'r') as f:
            lines = f.readlines()

        for line in lines[3:]:  # Skip the header
            if line.strip():
                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                fx = float(parts[4])
                fy = float(parts[5])
                cx = float(parts[6])
                cy = float(parts[7])
                camera_params[camera_id] = {
                    "model": model,
                    "width": width,
                    "height": height,
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy
                }
        return camera_params

    def _load_image_poses(self, images_path):
        image_poses = {}
        with open(images_path, 'r') as f:
            lines = f.readlines()

        for line in lines[4::2]:  # Skip the header
            if line.strip():
                parts = line.split()
                image_id = int(parts[0]) - 1
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                image_name = parts[9]
                image_poses[image_id] = {
                    "rotation": [qw, qx, qy, qz],
                    "translation": [tx, ty, tz],
                    "camera_id": camera_id,
                    "image_name": image_name
                }
        return image_poses

    def __len__(self):
        return len(self.selected_pairs)

    def _quaternion_to_rotation_matrix(self, q0, q1, q2, q3):
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        rot_matrix = np.array([[r00, r01, r02],
                               [r10, r11, r12],
                               [r20, r21, r22]])

        return rot_matrix

    def _extrinsics_matrix(self, qw, qx, qy, qz, tx, ty, tz):
        R = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = [tx, ty, tz]

        return extrinsics

    def _intrinsic_matrix(self, fx, fy, cx, cy):
        K = np.array([
            [fx,  0,  cx],
            [ 0, fy,  cy],
            [ 0,  0,   1]
        ])
        return K

    def _get_one_view(self, idx, indexes):
        # print('idxxxxxxxxx', idx)
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            # assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)

        idx_current = idx
        # print('idx_current', idx_current)

        image_id = indexes[idx_current]
        # print('image_id', image_id)
        pose = self.image_poses.get(image_id)
        if not pose:
            raise ValueError(f"image id {image_id} not found in images.txt")

        camera_id = pose["camera_id"]
        intrinsics = self.intrinsic_camera.get(camera_id)
        if not intrinsics:
            raise ValueError(f"camera id {camera_id} not found in cameras.txt")

        qw, qx, qy, qz = pose["rotation"]
        tx, ty, tz = pose["translation"]
        camera_pose = self._extrinsics_matrix(qw, qx, qy, qz, tx, ty, tz)

        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']

        camera_intrinsics = self._intrinsic_matrix(fx, fy, cx, cy)

        image_name = pose["image_name"]
        rgb_image_path = osp.join(self.images_path, image_name)
        depthmap_path = osp.join(self.depthmap_path, f"{image_name}.geometric.bin")

        rgb_image = imread_cv2(rgb_image_path)
        if rgb_image is None:
            raise FileNotFoundError(f"RGB image {image_name} not found at {rgb_image_path}")

        with open(depthmap_path, "rb") as fid:
            width, height, channels = np.genfromtxt(
                fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
            )
            fid.seek(0)
            num_delimiter = 0
            byte = fid.read(1)
            while True:
                if byte == b"&":
                    num_delimiter += 1
                    if num_delimiter >= 3:
                        break
                byte = fid.read(1)
            array = np.fromfile(fid, np.float32)
        array = array.reshape((width, height, channels), order="F")
        depthmap = np.transpose(array, (1, 0, 2)).squeeze()

        if depthmap is None:
            raise FileNotFoundError(f"Depth map for {image_name} not found at {depthmap_path}")

        depthmap[~np.isfinite(depthmap)] = 0

        rgb_image, depthmap, camera_intrinsics = self._crop_resize_if_necessary(
            rgb_image, depthmap, camera_intrinsics, resolution, rng=self._rng, info=image_name)

        # print('image_name', image_name)

        return dict(
            img=rgb_image,
            depthmap=depthmap,
            camera_pose=camera_pose.astype(np.float32),
            camera_intrinsics=camera_intrinsics.astype(np.float32),
            dataset='UnderWaterDataset',
            label=self.ROOT,
            instance=image_name
        )

    def compute_iou(self, pointcloud1, pointcloud2, iou_threshold=0.75, pixels_count=40000, distance_threshold=1):
        if len(pointcloud1) > pixels_count:
            pointcloud1 = pointcloud1[np.random.choice(len(pointcloud1), pixels_count, replace=False)]
        if len(pointcloud2) > pixels_count:
            pointcloud2 = pointcloud2[np.random.choice(len(pointcloud2), pixels_count, replace=False)]

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pointcloud2)
        distances, indices = nbrs.kneighbors(pointcloud1)
        intersection1 = np.count_nonzero(distances.flatten() < distance_threshold)

        # nbrs = NearestNeighbors(n_neighbors=1, algorithm = 'kd_tree').fit(pointcloud1)
        # distances, indices = nbrs.kneighbors(pointcloud2)
        # intersection2 = np.count_nonzero(distances.flatten() < distance_threshold)

        # iou = min(intersection1 / pixels_count, intersection2 / pixels_count)
        iou = intersection1 / len(pointcloud1)

        if iou > iou_threshold:
            return 0.0

        return intersection1

    def simulate_occlusion(self, point_cloud, radius):
        """
        Simulates occlusion by attaching a ball of fixed radius to each 3D point.
        Points occluded by another point within the radius are removed.
        """
        if len(point_cloud.shape) == 3:
            point_cloud = point_cloud.reshape(-1, 3)
        return point_cloud
        # tree = cKDTree(point_cloud)
        #
        # indices = tree.query_ball_tree(tree, radius)
        #
        # visible_mask = np.full(len(point_cloud), True, dtype=bool)
        # for i, neighbors in enumerate(indices):
        #     if visible_mask[i]:
        #         for neighbor in neighbors[1:]:
        #             visible_mask[neighbor] = False
        #
        # visible_points = point_cloud[visible_mask]
        # return visible_points

    def compute_iou_with_occlusion(self, pc1, pc2, iou_threshold, radius=0.05):
        """
        Computes the IoU between two 3D point clouds with artificial occlusion.
        """
        visible_pc1 = self.simulate_occlusion(pc1, radius)
        visible_pc2 = self.simulate_occlusion(pc2, radius)

        return self.compute_iou(visible_pc1, visible_pc2, iou_threshold)

    def quality_pair_score(self, iou, alpha):
        """
        Compute the quality pair score s = IoU × 4 cos(α)(1 - cos(α))
        """
        angle_term = 4 * np.cos(alpha) * (1 - np.cos(alpha))
        return iou * angle_term if angle_term > 0 else 0

    def select_best_pairs(self, pairs_indexes, iou_threshold=0.85, score_threshold=0.001):
        """
        Select the best image pairs using a greedy algorithm
        """
        pair_scores = []

        # print('pairs_indexes', pairs_indexes)

        for pair_ij in tqdm(pairs_indexes):
            # if iter_cur % 100 == 0:
            #     print(f'pairs selection: {iter_cur} out of {len(pairs_indexes)}')
            i, j = pair_ij
            img1 = self._get_one_view(i, self.split_data)
            img2 = self._get_one_view(j, self.split_data)

            pts3d1, _ = depthmap_to_absolute_camera_coordinates(**img1)
            pts3d2, _ = depthmap_to_absolute_camera_coordinates(**img2)

            # print('i and j', i, j)
            iou = self.compute_iou_with_occlusion(pts3d1, pts3d2, iou_threshold)
            if iou == 0:
                continue

            pose1, pose2 = img1['camera_pose'], img2['camera_pose']
            rotation_diff = pose1[:3, :3].T @ pose2[:3, :3]  # Relative rotation matrix
            alpha = np.arccos(np.clip((np.trace(rotation_diff) - 1) / 2, -1, 1))  # Angle in radians

            score = self.quality_pair_score(iou, alpha)
            if score > score_threshold:
                pair_scores.append((score, alpha, self.split_data[i], self.split_data[j]))

        pair_scores.sort(key=lambda x: x[0], reverse=True)
        return pair_scores

    def _create_pairs_indexes(self, pairs_per_image=30, step=2, image_threshold=50):
        n = len(self.split_data)

        pairs = []

        for i in range(n):
            next_indices = list(range(i + 1, min(i + pairs_per_image + 1, n), step))
            # next_indices = list(range(i + 1, min(i + 2, n)))
            pairs.extend([[i, j] for j in next_indices])
            # pairs.extend([[self.split_data[i], self.split_data[j]] for j in next_indices])

        pair_scores = self.select_best_pairs(pairs)
        selected_pairs = []
        used_images = defaultdict(int)

        for score, alpha, i, j in pair_scores:
            if used_images[i] > image_threshold or used_images[j] > image_threshold:
                continue

            selected_pairs.append((i, j, score, alpha))
            used_images[i] += 1
            used_images[j] += 1

        print('LEN selected_pairs', len(selected_pairs))

        # for i in range(n - 1):
        #     if (self.split_data[i], self.split_data[i + 1]) not in selected_pairs:
        #         selected_pairs.append((self.split_data[i], self.split_data[i + 1]))

        # print('LEN selected_pairs with adjacent frames', len(selected_pairs))

        return selected_pairs


    def _get_views(self, idx, resolution, rng):
        views = []

        for idx_current in self.selected_pairs[idx][:2]:
            image_id = idx_current
            pose = self.image_poses.get(image_id)
            if not pose:
                raise ValueError(f"image id {image_id} not found in images.txt")

            camera_id = pose["camera_id"]
            intrinsics = self.intrinsic_camera.get(camera_id)
            if not intrinsics:
                raise ValueError(f"camera id {camera_id} not found in cameras.txt")

            qw, qx, qy, qz = pose["rotation"]
            tx, ty, tz = pose["translation"]
            camera_pose = self._extrinsics_matrix(qw, qx, qy, qz, tx, ty, tz)

            fx = intrinsics['fx']
            fy = intrinsics['fy']
            cx = intrinsics['cx']
            cy = intrinsics['cy']

            camera_intrinsics = self._intrinsic_matrix(fx, fy, cx, cy)

            image_name = pose["image_name"]
            rgb_image_path = osp.join(self.images_path, image_name)
            depthmap_path = osp.join(self.depthmap_path, f"{image_name}.geometric.bin")

            rgb_image = imread_cv2(rgb_image_path)
            if rgb_image is None:
                raise FileNotFoundError(f"RGB image {image_name} not found at {rgb_image_path}")

            with open(depthmap_path, "rb") as fid:
                width, height, channels = np.genfromtxt(
                    fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
                )
                fid.seek(0)
                num_delimiter = 0
                byte = fid.read(1)
                while True:
                    if byte == b"&":
                        num_delimiter += 1
                        if num_delimiter >= 3:
                            break
                    byte = fid.read(1)
                array = np.fromfile(fid, np.float32)
            array = array.reshape((width, height, channels), order="F")
            depthmap = np.transpose(array, (1, 0, 2)).squeeze()

            if depthmap is None:
                raise FileNotFoundError(f"Depth map for {image_name} not found at {depthmap_path}")

            depthmap[~np.isfinite(depthmap)] = 0

            rgb_image, depthmap, camera_intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, camera_intrinsics, resolution, rng=rng, info=image_name)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=camera_intrinsics.astype(np.float32),
                dataset='UnderWaterDataset',
                label=self.ROOT,
                instance=image_name
            ))

        return views

if __name__ == "__main__":
    train_dataset = UnderWaterDataset(split='train', ROOT="/home/aleksandra/dense_glomap_output", resolution=(224, 224))
    test_dataset = UnderWaterDataset(split='test', ROOT="/home/aleksandra/dense_glomap_output", resolution=(224, 224))

    print(len(train_dataset), len(test_dataset))

    for idx in range(len(train_dataset)):
        views = train_dataset[idx]
        print(views)
        break

    for idx in range(len(test_dataset)):
        views = test_dataset[idx]
        print(views)
        break