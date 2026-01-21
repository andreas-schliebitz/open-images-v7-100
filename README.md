# The OpenImagesV7-100 Dataset

**Authors**: Andreas Schliebitz [1], Heiko Tapken [1] and Martin Atzmueller [2]

_[1] Faculty of Engineering and Computer Science, Osnabrück University of Applied Sciences, Osnabrück Germany_

_[2] Semantic Information Systems Group, Osnabrück University and Germany Research Center for AI (DFKI)_

## Abstract

This image classification dataset is a **subset** of Google's latest [**Open Images V7**](https://storage.googleapis.com/openimages/web/index.html) dataset ([600 classes, ~1.9 mio. annotated samples](https://docs.voxel51.com/dataset_zoo/datasets/open_images_v7.html)) consisting of **100** randomly sampled classes with exactly **345** samples each. Using our PyTorch compatible [dataset class implementation](./open_images_v7_100/dataset.py), each **training** sample _can_ be augmented an arbitrary amount of times using [`AutoAugment`](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.AutoAugment.html) ([Cubuk et al.](https://doi.org/10.48550/arXiv.1805.09501)). Selecting `num_augments=3` will yield a train split comparable to ImageNet100 in size. No augmentations are applied by default. We also [publish](./datasets/open-images-v7-100.json) **normalization vectors** (RGB mean and standard deviation) for our dataset as well as other metadata like **average image resolution** and **class names**.

## Sampling and Augmentation

For any given class number (e. g. 100), our sampling strategy dynamically determines the number of required (and present) samples in the dataset. This allows us to produce **perfectly balanced** classes regardless of class count, provided that enough samples are present in each class to satisfy the requested sample count. Using the **optional** augmentation feature of our `OpenImagesV7` [dataset class](./open_images_v7_100/dataset.py), the number of training samples can be increased as desired. This enables us to easily reproduce the **train split** size of [ImageNet100](https://github.com/HobbitLong/CMC/tree/master).

This dataset is intended to be used in experiments involving transfer learning or fine tuning (convolutional) neural networks where flexible **dataset size**, **class balance** and **disjointness to ImageNet** are important. Note that we only include images in our dataset that are **unambiguously labeled** to contain **exactly one class** per sample.

As of December 2025 and to the best of our knowledge, this repository is the only source for obtaining a high quality and fully reproducible 100 class subset of the Open Images V7 dataset for image classification that is comparable to ImageNet100 in class diversity and balance.

## Getting Started

Dataset acquisition, augmentation and sampling are based on the `fiftyone` and `torchvision` Python libraries. Note that `fiftyone` is the [recommended tool](https://docs.voxel51.com/dataset_zoo/datasets/open_images_v7.html) for interacting with the Open Images V7 dataset.

### Installation

This project can be installed in classic Python virtual environments via `pip` as well as with the [Poetry](https://python-poetry.org/docs/) dependency manager. The following guide uses the classic `venv` approach:

1. Create a Python virtual environment:

    ```bash
    python3 -m venv venv
    ```

2. Activate the virtual environment:

    ```bash
    source venv/bin/activate
    ```

3. Update `pip` and install Python dependencies:

    ```bash
    ./install.sh
    ```

### Setup

1. Navigate into the source directory:

    ```bash
    cd open_images_v7_100
    ```

2. Create a custom `.env` from `.env.example`:

    ```bash
    cp .env.example .env
    ```

    You _can_ modify FiftyOne's [environment variables](https://docs.voxel51.com/user_guide/config.html#configuration-options) by editing this file as needed.

    **Optional**: In some environments, FiftyOne can fail to start a local MongoDB instance. In that case, specify an external instance via `FIFTYONE_DATABASE_URI`:

    ```text
    mongodb://[username:password@]host[:port]
    ```

    **Hint**: If your external MongoDB uses TLS or has only an admin user, you may have to add the following query parameters to your database URI:

    ```text
    ?tls=true&authSource=admin
    ```

## Dataset Generation

### OpenImagesV7-100

The original **OpenImagesV7-100** dataset can be generated via the following command:

```bash
python3 main.py \
    --export-dir ./datasets \
    --num-classes 100 \
    --samples-per-class 345 \
    --iter-batch-size 1000 \
    --random-seed 42 \
    --num-workers 8 \
    --calculate-dataset-resolution \
    --calculate-normalization-constants \
    --normalization-resize-dim 224
```

### OpenImagesV7-30

If you want to increase the number of samples per class to be comparable to ImageNet100 (~1300 samples per class) without augmentation, you can generate **OpenImagesV7-30** at the cost of class diversity. This subset of Open Images V7 consists of **30 randomly sampled classes** with exactly **1414** samples each:

```bash
python3 main.py \
    --export-dir ./datasets \
    --num-classes 30 \
    --samples-per-class 1414 \
    --iter-batch-size 1000 \
    --random-seed 42 \
    --num-workers 8 \
    --calculate-dataset-resolution \
    --calculate-normalization-constants \
    --normalization-resize-dim 224
```

More information on **sampling different class counts** can be found [below](#sampling-fewer-classes).

## Usage

The code snippet below illustrates loading and [normalization](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#i-just-want-to-do-image-classification) of our dataset's **train split** using the provided [`OpenImagesV7`](./open_images_v7_100/dataset.py) [`Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) subclass. We do not dictate any specific splits, meaning you are free to subdivide the OpenImagesV7-100 dataset as you wish. For **reproducibility reasons**, we do however _recommend_ using the **default** `split_lengths` of `(0.7, 0.1, 0.2)` for [stratified](https://scikit-learn.org/stable/modules/cross_validation.html#stratification) train, validation and test splits with `random_seed` set to `42`:

```python
import torch
from torchvision.transforms import v2
from torch.utils.data import random_split
from open_images_v7_100.dataset import OpenImagesV7

train_split = OpenImagesV7(
    root="./datasets/open-images-v7-100",
    split="train",
    split_lengths=(0.7, 0.1, 0.2),
    random_seed=42,
    num_augments=0,
    transform=v2.Compose(
        [
            v2.Resize(size=(224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.46805658, 0.43819318, 0.40574419],
                std=[0.22731383, 0.22057558, 0.22091592],
            ),
        ]
    ),
)

for image, label in train_split:
    assert isinstance(image, torch.Tensor) and isinstance(label, int)
    assert len(train_split.classes) == 100
```

We also provide **unit tests** for our `OpenImagesV7` dataset class which can be executed once the `open-images-v7-100` dataset has been generated. Simply adjust the [`cls.root`](./tests/test_open_images_v7.py) attribute in the test class and run the `pytest` command from within the project's root directory.

**Note**: As with ImageNet, the published normalization vectors are calculated over the **entire** dataset and only apply for `num_augments=0` after all dataset samples have been resized to **224x224px**.

## Advantages over ImageNet100

* **Disjointness to ImageNet**: Most model weights offered by popular computer vision libraries like [Torchvision](https://docs.pytorch.org/vision/stable/index.html) are obtained through supervised pre-training of a randomly initialized model on the ImageNet-1K dataset (s. `DEFAULT`, `IMAGENET1K_V1`, `IMAGENET1K_V2` of [ResNet50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)). Depending on experimental design, it can be problematic to train such a pre-initialized model on ImageNet100 as it is a subset of ImageNet-1K. In these cases it is quite possible, that some interpretations of the resulting metrics or training characteristics (i. e. model convergence) may get invalidated. This problem can be alleviated by using our datasets (i. e. OpenImagesV7 with 100 or [fewer classes](#sampling-fewer-classes)) instead.

* **Sample resolution**: With an average native sample resolution of 964x800px, our OpenImagesV7-100 dataset is 325% larger in pixel area (resolution) than ImageNet-1K ([469x387px](https://image-net.org/challenges/LSVRC/2014/index)) and by extension ImageNet100. This allows for experiments with up to 800x800px sample resolution where on average visual sample quality is maintained due to slight down- and not heavy upscaling as it would be the case with ImageNet100.

* **Sampling flexibility**: Using our sampling method and the source code published in this repository, users can create **any balanced subset** of the Open Images V7 classification dataset with an arbitrary number of classes. Note that in our `OpenImagesV7` [dataset implementation](./open_images_v7_100/dataset.py), augmentation is only applied if `num_augments > 0`. The default augmentation method is `AutoAugment` with the `IMAGENET` [augmentation policy](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.AutoAugmentPolicy.html#torchvision.transforms.AutoAugmentPolicy). You _can_ use any other augmentation strategy like [`AugMix`](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.AugMix.html) or [`RandAugment`](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandAugment.html) through the `augmentation_transform` parameter.

## Sampling fewer Classes

The following smaller datasets can also be sampled with balanced classes:

| # Classes | Samples per Class |
|---------|--------------------|
| 10      | 4073               |
| 20      | 2307               |
| 30      | 1414               |
| 40      | 924                |
| 50      | 748                |
| 60      | 556                |
| 70      | 463                |
| 80      | 412                |
| 90      | 373                |
| **100** | **345**            |

Simply pass the desired combination of `--num-classes=<# Classes>` and `--samples-per-class=<Samples per Class>` to `main.py` as seen [above](#dataset-generation). Note that sample sizes are computed to be maximal per class, i. e. there is no larger sample size per class that would still allow for a **perfectly balanced** dataset with exactly `# Classes`.

## Citation

If you use this software in your work, please cite this GitHub repository:

```text
@software{OpenImagesV7-100,
    author = {Schliebitz, Andreas and Tapken, Heiko and Atzmueller, Martin},
    license = {Apache-2.0},
    month = dec,
    title = {{The OpenImagesV7-100 Dataset}},
    url = {https://github.com/andreas-schliebitz/open-images-v7-100},
    version = {0.1.0},
    year = {2025}
}
```
