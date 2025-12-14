import argparse

from argparse import Namespace


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the OpenImagesV7-100 image classification dataset."
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="./datasets",
        help="Directory path to export PyTorch ImageFolder compatible dataset to.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=100,
        help="Number of classes to include in dataset.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=345,
        help="Number of samples per class to include in dataset.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for selecting classes and downloading dataset samples.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads.",
    )
    parser.add_argument(
        "--iter-batch-size",
        type=int,
        default=1000,
        help="Number of samples to batch while iterating over datasets.",
    )
    parser.add_argument(
        "--calculate-normalization-constants",
        default=None,
        action="store_const",
        const=True,
        help="Calculate normalization vectors (RGB mean and stdev) for a resized version of the dataset (s. --normalization-resize-dim).",
    )
    parser.add_argument(
        "--calculate-dataset-resolution",
        default=None,
        action="store_const",
        const=True,
        help="Calculate average sample resolution of the dataset.",
    )
    parser.add_argument(
        "--normalization-resize-dim",
        type=int,
        default=224,
        help="Square resize dimensions (width and height) for samples to calculate normalization constants for.",
    )

    return parser.parse_args()
