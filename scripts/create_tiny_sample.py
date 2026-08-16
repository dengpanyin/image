#!/usr/bin/env python3
"""Create a tiny local sample dataset to smoke-test training."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]

LABELS = [
    "fire_smoke",
    "lane_blocked",
    "debris",
    "door_open_abnormal",
    "wet_floor",
]


def make_image(kind: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    w, h = 320, 240
    img = Image.new("RGB", (w, h), (rng.randint(40, 90),) * 3)
    draw = ImageDraw.Draw(img)

    if kind == "smoke":
        for _ in range(25):
            x, y = rng.randint(0, w), rng.randint(0, h // 2)
            r = rng.randint(20, 70)
            color = (rng.randint(120, 200),) * 3
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
    elif kind == "blocked":
        draw.rectangle((40, 100, w - 40, h - 40), fill=(120, 90, 60))
    elif kind == "debris":
        for _ in range(8):
            x, y = rng.randint(20, w - 60), rng.randint(h // 2, h - 30)
            draw.rectangle((x, y, x + 40, y + 25), fill=(100, 80, 50))
    elif kind == "normal":
        draw.rectangle((0, h - 40, w, h), fill=(90, 90, 90))
    else:
        draw.line((0, h - 40, w, h - 40), fill=(180, 180, 180), width=3)

    return img


def labels_for(kind: str) -> dict[str, int]:
    mapping = {
        "smoke": {"fire_smoke": 1},
        "blocked": {"lane_blocked": 1},
        "debris": {"debris": 1},
        "wet": {"wet_floor": 1},
        "door": {"door_open_abnormal": 1},
        "normal": {},
    }
    row = {name: 0 for name in LABELS}
    row.update(mapping.get(kind, {}))
    return row


def build_split(split: str, specs: list[tuple[str, str]]) -> pd.DataFrame:
    image_root = ROOT / "data" / "images" / split
    image_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (kind, name) in enumerate(specs):
        rel = f"{split}/{name}"
        img = make_image(kind, seed=hash((split, name)) % 10_000)
        img.save(image_root / name, quality=90)
        row = {"image_path": rel}
        row.update(labels_for(kind))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    train_specs = [
        ("smoke", "001_smoke.jpg"),
        ("smoke", "002_smoke.jpg"),
        ("normal", "003_normal.jpg"),
        ("normal", "004_normal.jpg"),
        ("blocked", "005_blocked.jpg"),
        ("debris", "006_debris.jpg"),
        ("wet", "007_wet.jpg"),
        ("door", "008_door.jpg"),
    ]
    val_specs = [
        ("smoke", "001_smoke.jpg"),
        ("normal", "002_normal.jpg"),
        ("blocked", "003_blocked.jpg"),
    ]

    splits = ROOT / "data" / "splits"
    splits.mkdir(parents=True, exist_ok=True)

    train_df = build_split("train", train_specs)
    val_df = build_split("val", val_specs)
    train_df.to_csv(splits / "train.csv", index=False)
    val_df.to_csv(splits / "val.csv", index=False)

    print(f"Created {len(train_df)} train + {len(val_df)} val samples")
    print(f"Images: {ROOT / 'data' / 'images'}")
    print(f"CSV: {splits / 'train.csv'}, {splits / 'val.csv'}")


if __name__ == "__main__":
    main()
