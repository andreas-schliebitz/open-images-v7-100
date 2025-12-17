import torch
import numpy as np

from typing import Callable
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, Dataset
from sklearn.model_selection import train_test_split


class OpenImagesV7(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        split_lengths: tuple[int, int, int] = (0.7, 0.1, 0.2),
        random_seed: int = 42,
        num_augments: int = 0,
        augmentation_transform: Callable = T.AutoAugment(
            policy=T.AutoAugmentPolicy.IMAGENET
        ),
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        assert split in {
            "train",
            "val",
            "test",
        }, f"The value of 'split' has to be either 'train', 'val' or 'test'. Got split='{split}'."

        assert (
            abs(sum(split_lengths) - 1.0) <= 1e-6
        ), "Sum of train, validation and test percentages must equal 1."

        assert (
            num_augments >= 0
        ), f"Number of augmentations has to be non-negative. Got num_augments={num_augments}."

        self.root = root
        self.split = split
        self.split_lengths = split_lengths
        self.random_seed = random_seed
        self.num_augments = num_augments
        self.augmentation_transform = augmentation_transform
        self.transform = transform
        self.target_transform = target_transform

        if self.num_augments > 0 and self.split != "train":
            print(
                f"WARNING: Refusing to augment samples in non-train split '{split}'. Got num_augments={num_augments} > 0."
            )
            self.num_augments = 0

        self.dataset, self.targets, self.classes = self._create_dataset_split(
            dataset=ImageFolder(root=root)
        )

    def _create_dataset_split(
        self, dataset: ImageFolder
    ) -> tuple[Subset, list[int], list[str]]:
        # Get indices of full ImageFolder dataset
        indices = np.arange(len(dataset))

        # Get class indices for stratification
        targets = np.array(dataset.targets)

        # Unpack split percentages
        train_perc, val_perc, test_perc = self.split_lengths

        # 1. Split dataset into train+val (left) and test indices (right)
        # while stratifying over the class indices
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_perc,
            stratify=targets,
            shuffle=True,
            random_state=self.random_seed,
        )

        # 2. Split the train+val indices into train (left) and val indices (right)
        # while stratifying over train+val subset of the entire dataset
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_perc / (train_perc + val_perc),
            shuffle=True,
            stratify=targets[train_val_indices],
            random_state=self.random_seed,
        )

        match self.split:
            case "train":
                split_indices = train_indices
            case "val":
                split_indices = val_indices
            case "test":
                split_indices = test_indices
            case _:
                raise ValueError(f"Unsupported split '{self.split}'.")

        split_indices = split_indices.tolist()
        dataset_split = Subset(dataset=dataset, indices=split_indices)
        split_targets = targets[split_indices]

        num_classes_full = np.unique(targets).size
        num_classes_split = np.unique(split_targets).size

        assert (
            num_classes_split == num_classes_full
        ), f"Stratified sampling of split '{self.split}' did not yield a dataset containing all {num_classes_full} classes, only {num_classes_split}."

        return dataset_split, split_targets.tolist(), dataset.classes

    def __len__(self) -> int:
        # Note that num_augments=0 returns len(self.dataset))
        return len(self.dataset) * (self.num_augments + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # Assume num_augments=2 and dataset_size=2
        # -> dataset_size=6 [index: 0,1,2,3,4,5]
        # -> index=0: First original image
        # -> index=1: #1 augmentation of first image
        # -> index=2: #2 augmentation of first image
        # -> index=3: Second original image
        # -> index=4: #1 augmentation of second image
        # -> index=5: #2 augmentation of second image
        # Note that: index % (num_augments + 1) == 0 -> original image

        # Note that num_augments=0 will set augment_image=False
        augment_group_size = self.num_augments + 1
        augment_index = index % augment_group_size
        augment_image = augment_index != 0

        image, label = self.dataset[index // augment_group_size]

        if image.mode != "RGB":
            image = image.convert("RGB")

        if augment_image:
            image = self.augmentation_transform(image)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
