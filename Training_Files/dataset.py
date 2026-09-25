"""
dataset.py

PyTorch Dataset for the road-severity classification task, reading
from a manifest CSV produced by build_manifest.py / split_manifest.py.

Changes vs. the original:
  * output is (INPUT_HEIGHT x INPUT_WIDTH), keeping the wide aspect of the
    ROI strip instead of squashing it into a square
  * JPEG draft mode makes decoding huge images (e.g. Norway ~3650x2044)
    much faster without hurting the final resolution
  * no rotation / blur augmentation (both smear the fine crack texture)
"""

import csv
import math

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


class BottomCrop:
    """
    Crops the bottom `fraction` of the image (the immediate road plane),
    dropping sky/horizon/buildings. Applied before resize in both training
    and inference so the two pipelines see the same geometry. The labels in
    manifest.csv are computed on this same region (see build_manifest.py).
    """

    def __init__(self, fraction=config.ROI_BOTTOM_FRACTION):
        self.fraction = fraction

    def __call__(self, img):
        w, h = img.size
        top = int(h * (1 - self.fraction))
        return img.crop((0, top, w, h))


_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


def get_transforms(train: bool):
    size = (config.INPUT_HEIGHT, config.INPUT_WIDTH)
    if train:
        return transforms.Compose([
            BottomCrop(),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.ToTensor(),
            _NORMALIZE,
        ])
    return transforms.Compose([
        BottomCrop(),
        transforms.Resize(size),
        transforms.ToTensor(),
        _NORMALIZE,
    ])


def _load_image(path):
    img = Image.open(path)
    # JPEG draft: let the decoder downscale by 1/2, 1/4, 1/8 while the
    # cropped strip stays at least as large as the model input.
    need_w = config.INPUT_WIDTH
    need_h = math.ceil(config.INPUT_HEIGHT / config.ROI_BOTTOM_FRACTION)
    if img.format == "JPEG":
        img.draft("RGB", (need_w, need_h))
    return img.convert("RGB")


class RoadSeverityDataset(Dataset):
    def __init__(self, csv_path, train: bool):
        with open(csv_path, newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = self.transform(_load_image(row["image_path"]))
        label = config.CLASS_TO_IDX[row["label"]]
        return image, label

    def class_counts(self):
        counts = [0] * len(config.CLASSES)
        for row in self.rows:
            counts[config.CLASS_TO_IDX[row["label"]]] += 1
        return counts

    def labels(self):
        return [config.CLASS_TO_IDX[r["label"]] for r in self.rows]

    def countries(self):
        return [r.get("country", "?") for r in self.rows]

    def ambiguous_mask(self):
        return [int(r.get("ambiguous", 0) or 0) for r in self.rows]
