import json
import torch
import random
import numpy as np
import fiftyone as fo

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset


def write_dataset_info(filepath: Path | str, info: dict) -> None:
    with open(filepath, mode="w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=4)


def sample_count_for_num_classes(
    class_counts: dict[str, int], num_classes: int = 100
) -> int | None:
    counts = list(class_counts.values())
    if len(counts) < num_classes:
        return None
    counts.sort(reverse=True)
    return counts[num_classes - 1]


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def calculate_normalization_constants(
    dataset: Dataset,
) -> tuple[list[float], list[float]]:
    sample_stats = []
    for sample, _ in tqdm(dataset):
        mean = sample.mean(axis=(1, 2))
        std = sample.std(axis=(1, 2))
        stats = torch.cat((mean, std))
        sample_stats.append(stats)
    mean_std = torch.stack(sample_stats).mean(dim=0, dtype=float)
    mean, std = mean_std[:3].tolist(), mean_std[-3:].tolist()
    return mean, std


def calculate_mean_sample_resolution(dataset: fo.Dataset) -> tuple[float, float]:
    widths, heights = [], []
    for sample in dataset.iter_samples(progress=True):
        widths.append(sample.metadata.width)
        heights.append(sample.metadata.height)

    mean_width = np.mean(widths)
    mean_height = np.mean(heights)
    return float(mean_width), float(mean_height)
