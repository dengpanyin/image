from __future__ import annotations

from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_val_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class MultiLabelCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        label_names: list[str],
        image_root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.label_names = label_names
        self.image_root = Path(image_root)
        self.transform = transform

        missing = [c for c in ["image_path", *label_names] if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        image_path = self.image_root / str(row["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        labels = row[self.label_names].astype(np.float32).to_numpy()
        return {
            "image": image,
            "labels": torch.from_numpy(labels),
            "image_path": str(row["image_path"]),
        }


def compute_pos_weight(train_csv: str | Path, label_names: list[str], cap: float = 10.0) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    weights = []
    for name in label_names:
        pos = df[name].sum()
        neg = len(df) - pos
        if pos <= 0:
            weights.append(1.0)
            continue
        w = float(neg / pos)
        weights.append(min(w, cap))
    return torch.tensor(weights, dtype=torch.float32)
