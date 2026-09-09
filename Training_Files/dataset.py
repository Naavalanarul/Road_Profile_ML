"""
dataset.py

PyTorch Dataset for the road-severity classification task, reading
from a manifest CSV produced by build_manifest.py / split_manifest.py.
"""

import csv

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


class BottomCrop:
    """
    Crops the bottom `fraction` of the image (the immediate road plane),
    dropping sky/horizon/buildings that the CNN could otherwise overfit
    to instead of asphalt texture. Applied before resize, in both
    training and inference, so the two pipelines see the same geometry.
    """

    def __init__(self, fraction=config.ROI_BOTTOM_FRACTION):
        self.fraction = fraction

    def __call__(self, img):
        w, h = img.size
        top = int(h * (1 - self.fraction))
        return img.crop((0, top, w, h))


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            BottomCrop(),
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        BottomCrop(),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class RoadSeverityDataset(Dataset):
    def __init__(self, csv_path, train: bool):
        with open(csv_path, newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = config.CLASS_TO_IDX[row["label"]]
        return image, label

    def class_counts(self):
        counts = [0] * len(config.CLASSES)
        for row in self.rows:
            counts[config.CLASS_TO_IDX[row["label"]]] += 1
        return counts