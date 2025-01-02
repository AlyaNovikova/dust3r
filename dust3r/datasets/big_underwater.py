# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed arkitscenes
# dataset at https://github.com/apple/ARKitScenes - Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License https://github.com/apple/ARKitScenes/tree/main?tab=readme-ov-file#license
# See datasets_preprocess/preprocess_arkitscenes.py
# --------------------------------------------------------
import os.path as osp
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.datasets import UnderWaterDataset

class MergedUnderWaterDataset(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        if split == "train":
            self.split = "train"
        elif split == "test":
            self.split = "test"
        else:
            raise ValueError("")

        self.root1 = self.ROOT
        self.root2 = self.ROOT + "_2"
        self.root3 = self.ROOT + "_3"

        print("ARGS", *args)
        print("KWARGS", **kwargs)

        self.dataset3 = UnderWaterDataset(*args, split=self.split, ROOT=self.root3, **kwargs)
        self.dataset1 = UnderWaterDataset(*args, split=self.split, ROOT=self.root1, **kwargs)
        self.dataset2 = UnderWaterDataset(*args, split=self.split, ROOT=self.root2, **kwargs)


    def __len__(self):
        return len(self.dataset1) + len(self.dataset2) + len(self.dataset3)

    def _get_views(self, idx, resolution, rng):
        if idx < len(self.dataset1):
            return self.dataset1._get_views(idx, resolution, rng)
        if idx < len(self.dataset1) + len(self.dataset2):
            return self.dataset2._get_views(idx - len(self.dataset1), resolution, rng)
        else:
            return self.dataset3._get_views(idx - len(self.dataset1) - len(self.dataset2), resolution, rng)

if __name__ == "__main__":
    train_dataset = MergedUnderWaterDataset(split='train', ROOT="/home/aleksandra/dense_glomap_output", resolution=(224, 224))
    test_dataset = MergedUnderWaterDataset(split='test', ROOT="/home/aleksandra/dense_glomap_output", resolution=(224, 224))

    print(len(train_dataset), len(test_dataset))

    print(len(train_dataset.dataset1), len(test_dataset.dataset1))
    print(len(train_dataset.dataset2), len(test_dataset.dataset2))
    print(len(train_dataset.dataset3), len(test_dataset.dataset3))