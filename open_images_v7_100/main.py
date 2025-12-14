#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dotenv import load_dotenv

load_dotenv()


import math
import torch
import random
import fiftyone as fo
import fiftyone.zoo as foz

from pathlib import Path
from pprint import pprint
from fiftyone import ViewField as F
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from open_images_v7_100.args import get_args
from open_images_v7_100.utils import (
    seed_everything,
    write_dataset_info,
    sample_count_for_num_classes,
    calculate_normalization_constants,
    calculate_mean_sample_resolution,
)

if __name__ == "__main__":

    ARGS = get_args()
    print("===== Arguments =====")
    args_dict = vars(ARGS)
    pprint(args_dict)

    seed_everything(seed=ARGS.random_seed)
    dataset_info = {"args": args_dict}

    FULL_DATASET_SIZE = 1743042 + 41620 + 125436
    DATASET_BASE_NAME = f"open-images-v7-{ARGS.num_classes}"

    # Load all splits of the entire Open Images v7 dataset
    # See: https://docs.voxel51.com/dataset_zoo/datasets/open_images_v7.html
    full_dataset = foz.load_zoo_dataset(
        "open-images-v7",
        dataset_name="open-images-v7-full",
        splits=("train", "validation", "test"),
        label_types="classifications",
        num_workers=ARGS.num_workers,
    )
    assert (
        full_dataset.count() == FULL_DATASET_SIZE
    ), f"Expected exactly {FULL_DATASET_SIZE} samples in full Open Images V7 dataset (actual size: {full_dataset.count()})"

    # Remove all samples with multiple labels
    single_class_dataset = full_dataset.match(
        (F("positive_labels.classifications").exists())
        & (F("positive_labels.classifications").length() == 1)
    )

    print(
        "Creating single class classification field 'ground_truth' from 'positive_labels'..."
    )

    # Re-assign single label of type 'fo.Classification' to new 'label' field
    for sample in single_class_dataset.iter_samples(
        progress=True, autosave=True, batch_size=ARGS.iter_batch_size
    ):
        sample["ground_truth"] = sample.positive_labels.classifications[0]

    # Count how many samples each class has
    class_counts = single_class_dataset.count_values("ground_truth.label")

    required_sample_count = sample_count_for_num_classes(class_counts, ARGS.num_classes)
    print(
        f"A perfectly balanced dataset with {ARGS.num_classes} classes can have at most {required_sample_count} samples per class.",
    )

    # Calculate augmentation factor necessary to reach at least --samples-per-class
    num_augments = (
        0
        if required_sample_count >= ARGS.samples_per_class
        else math.ceil(ARGS.samples_per_class / required_sample_count) - 1
    )
    total_samples_per_class = (num_augments + 1) * required_sample_count
    augmented_dataset_size = ARGS.num_classes * total_samples_per_class
    dataset_info["num_augments"] = num_augments

    print(
        f"Each original sample has to be augmented {num_augments} additional times after sampling for a total of {total_samples_per_class} samples per class and a dataset size of {augmented_dataset_size}."
    )

    # Select all classes with at least 'required_sample_count' samples
    sample_classes = [
        cls for cls, count in class_counts.items() if count >= required_sample_count
    ]

    assert (
        len(sample_classes) >= ARGS.num_classes
    ), f"Insufficient number of classes ({len(sample_classes)}) in dataset have at least {required_sample_count} samples."

    # Randomly sample --num-classes from eligible classes
    random_classes = random.sample(sample_classes, ARGS.num_classes)

    # Create empty sampled dataset
    sampled_dataset = fo.Dataset(name=DATASET_BASE_NAME)
    for random_class in random_classes:
        print(f"Sampling {required_sample_count} samples for class '{random_class}'...")
        # For each random class, randomly select --samples-per-class samples from main dataset
        # See: https://docs.voxel51.com/user_guide/using_views.html#random-sampling
        class_samples = single_class_dataset.match(
            F("ground_truth.label") == random_class
        ).take(required_sample_count, seed=ARGS.random_seed)

        # Add class samples to sampled dataset
        sampled_dataset.add_samples(class_samples)

    # Ensure the final dataset is of expected size
    sampled_dataset_size = ARGS.num_classes * ARGS.samples_per_class
    assert (
        sampled_dataset.count() == sampled_dataset_size
    ), f"Expected exactly {sampled_dataset_size} samples in final OpenImagesV7-{ARGS.num_classes} dataset (actual size: {sampled_dataset.count()})"

    # Export dataset to PyTorch compatible image dataset
    # See: https://docs.voxel51.com/user_guide/export_datasets.html#image-classification-dir-tree
    DATASET_EXPORT_DIR = str(
        Path(ARGS.export_dir).expanduser().resolve().joinpath(DATASET_BASE_NAME)
    )
    print(
        f"Exporting to PyTorch ImageFolder compatible dataset: '{DATASET_EXPORT_DIR}'..."
    )
    sampled_dataset.export(
        export_dir=DATASET_EXPORT_DIR,
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        label_field="ground_truth",
    )

    # Calculate normalization vectors if requested
    if ARGS.calculate_normalization_constants:
        img_size = ARGS.normalization_resize_dim
        print(
            f"Calculating normalization vectors (RGB mean and standard deviation) for dataset after {img_size}x{img_size}px resize..."
        )
        dataset = ImageFolder(
            root=DATASET_EXPORT_DIR,
            transform=v2.Compose(
                [
                    v2.Resize(size=(img_size, img_size)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            ),
        )
        mean, std = calculate_normalization_constants(dataset)
        dataset_info |= {"mean": mean, "std": std}

    if ARGS.calculate_dataset_resolution:
        sampled_dataset.compute_metadata(num_workers=ARGS.num_workers)

        print("Calculating dataset's mean sample resolution...")
        mean_width, mean_height = calculate_mean_sample_resolution(
            dataset=sampled_dataset
        )

        dataset_info |= {
            "avg_width": mean_width,
            "avg_height": mean_height,
        }

    dataset_info |= {
        "num_samples": sampled_dataset.count(),
        "classes": sampled_dataset.count_values("ground_truth.label"),
    }
    pprint(dataset_info)
    write_dataset_info(filepath=f"{DATASET_BASE_NAME}.json", info=dataset_info)
