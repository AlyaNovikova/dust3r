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

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class UnderWaterDataset2(BaseStereoViewDataset):
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

    def _get_split_data(self, train=True):
        """
        Split the data into training and testing subsets (80/20 split)
        """
        image_ids = list(self.image_poses.keys())

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
                image_id = int(parts[0])
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
        return len(self.split_data) - 1

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

    def _get_views(self, idx, resolution, rng):
        views = []

        for idx_current in [idx, idx + 1]:
            image_id = self.split_data[idx_current]
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