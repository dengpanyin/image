#!/usr/bin/env python3
"""Download Home-fire sample dataset and convert to project CSV format."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_URL = (
    "https://github.com/PengBo0/Home-fire-dataset/releases/download/v1.0.0/test.zip"
)
LABEL_NAMES = [
    "fire_smoke",
    "lane_blocked",
    "debris",
    "door_open_abnormal",
    "wet_floor",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Using cached archive: {dest}")
        return

    print(f"Downloading {url}")
    print(f"  -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded {dest.stat().st_size / 1_048_576:.1f} MB")


def extract(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted"
    if marker.exists():
        print(f"Using cached extract: {extract_dir}")
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok", encoding="utf-8")
    print(f"Extracted to {extract_dir}")


def find_image_label_pairs(root: Path) -> list[tuple[Path, Path | None]]:
    pairs: list[tuple[Path, Path | None]] = []
    for image_path in root.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            candidate = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            label_path = candidate if candidate.exists() else None
        pairs.append((image_path, label_path))
    return pairs


def yolo_has_fire_smoke(label_path: Path | None) -> int:
    if label_path is None or not label_path.exists():
        return 0
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    # Any YOLO box (fire or smoke) -> positive for multi-label fire_smoke
    return 1


def build_rows(pairs: list[tuple[Path, Path | None]], split_prefix: str) -> list[dict]:
    rows = []
    for idx, (image_path, label_path) in enumerate(pairs):
        rel_name = f"{split_prefix}/{idx:05d}{image_path.suffix.lower()}"
        row = {"image_path": rel_name, "source_path": str(image_path)}
        row["fire_smoke"] = yolo_has_fire_smoke(label_path)
        for name in LABEL_NAMES:
            if name not in row:
                row[name] = 0
        rows.append(row)
    return rows


def materialize_images(rows: list[dict], image_root: Path) -> None:
    image_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        dst = image_root / row["image_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["source_path"], dst)


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df = df[["image_path", *LABEL_NAMES]]
    df.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--max-train", type=int, default=400)
    parser.add_argument("--max-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw" / "home_fire"
    archive = raw_dir / "test.zip"
    extract_dir = raw_dir / "extracted"
    image_root = ROOT / "data" / "images"
    splits_dir = ROOT / "data" / "splits"

    download(args.url, archive)
    extract(archive, extract_dir)

    pairs = find_image_label_pairs(extract_dir)
    if not pairs:
        raise RuntimeError(f"No images found under {extract_dir}")

    random.seed(args.seed)
    random.shuffle(pairs)

    pos = [p for p in pairs if yolo_has_fire_smoke(p[1])]
    neg = [p for p in pairs if not yolo_has_fire_smoke(p[1])]
    print(f"Found {len(pairs)} images: {len(pos)} fire/smoke, {len(neg)} negative")

    total = min(len(pairs), args.max_train + args.max_val)
    pos_ratio = len(pos) / max(len(pairs), 1)
    n_pos = min(len(pos), max(1, int(total * pos_ratio)))
    n_neg = min(len(neg), total - n_pos)
    selected = pos[:n_pos] + neg[:n_neg]
    random.shuffle(selected)

    n_val = min(args.max_val, max(1, len(selected) // 5))
    val_pairs = selected[:n_val]
    train_pairs = selected[n_val:]

    train_rows = build_rows(train_pairs, "train")
    val_rows = build_rows(val_pairs, "val")

    if splits_dir.joinpath("train.csv").exists() and splits_dir.joinpath("val.csv").exists():
        for p in image_root.glob("train/*"):
            p.unlink()
        for p in image_root.glob("val/*"):
            p.unlink()

    materialize_images(train_rows, image_root)
    materialize_images(val_rows, image_root)
    write_csv(train_rows, splits_dir / "train.csv")
    write_csv(val_rows, splits_dir / "val.csv")

    print(f"Wrote {len(train_rows)} train + {len(val_rows)} val rows")
    print(f"CSV: {splits_dir / 'train.csv'}")
    print(f"Images: {image_root}")


if __name__ == "__main__":
    main()
