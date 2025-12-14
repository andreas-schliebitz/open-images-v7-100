#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pytest
import numpy as np

from tqdm import tqdm
from torch.utils.data import Subset
from torchvision.transforms import v2
from open_images_v7_100.dataset import OpenImagesV7


class TestOpenImagesV7:
    @classmethod
    def setup_class(cls) -> None:
        cls.root = "./datasets/open-images-v7-100"
        cls.train_perc = 0.7
        cls.val_perc = 0.1
        cls.test_perc = 0.2
        cls.resize_dim = 224
        cls.random_seed = 42
        cls.max_num_augments = 3
        cls.num_classes = 100
        cls.samples_per_class = 345
        cls.image_dtype = torch.float32
        cls.dataset_sizes = [34500, 58649, 82798, 106947]
        cls.train_split_sizes = [24149, 48298, 72447, 96596]
        cls.val_split_size = 3451
        cls.test_split_size = 6900

    def test_open_images_v7(self) -> None:
        for num_augments in range(self.max_num_augments + 1):
            dataset_size = 0
            for split in ("train", "val", "test"):
                print(
                    f"Running tests for dataset split '{split}' and num_augments={num_augments}."
                )
                dataset_split = OpenImagesV7(
                    root=self.root,
                    split=split,
                    split_lengths=(self.train_perc, self.val_perc, self.test_perc),
                    random_seed=self.random_seed,
                    num_augments=num_augments,
                    transform=v2.Compose(
                        [
                            v2.Resize(size=(self.resize_dim, self.resize_dim)),
                            v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(
                                mean=[0.46805658, 0.43819318, 0.40574419],
                                std=[0.22731383, 0.22057558, 0.22091592],
                            ),
                        ]
                    ),
                )

                assert isinstance(dataset_split.dataset, Subset)
                assert len(dataset_split.classes) == self.num_classes
                assert np.unique(dataset_split.targets).size == self.num_classes

                split_size = len(dataset_split)
                match split:
                    case "train":
                        assert split_size == self.train_split_sizes[num_augments]
                    case "val":
                        assert split_size == self.val_split_size
                    case "test":
                        assert split_size == self.test_split_size

                start, end = (
                    (split_size - 100, split_size) if split == "train" else (0, 100)
                )

                for image, label in tqdm(Subset(dataset_split, range(start, end))):
                    assert isinstance(image, torch.Tensor) and isinstance(label, int)
                    assert image.ndim == 3 and image.dtype == self.image_dtype

                dataset_size += split_size

            assert dataset_size == self.dataset_sizes[num_augments]


if __name__ == "__main__":
    pytest.main()
