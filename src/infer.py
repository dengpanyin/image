from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import build_val_transform
from device_utils import pick_device
from model import DINOv2MultiLabelClassifier


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    config: dict | None = None,
) -> tuple[DINOv2MultiLabelClassifier, list[str], dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_names: list[str] = checkpoint["label_names"]
    resolved_config = checkpoint.get("config", config)
    if resolved_config is None:
        raise ValueError("Checkpoint missing `config`; pass --config explicitly.")

    model = DINOv2MultiLabelClassifier(
        backbone_name=resolved_config["model"]["name"],
        num_classes=len(label_names),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, label_names, resolved_config


@torch.no_grad()
def predict_image(
    model: DINOv2MultiLabelClassifier,
    image_path: str | Path,
    transform,
    device: torch.device,
    threshold: float,
) -> dict[str, float | bool]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return {
        "probabilities": probs,
        "threshold": threshold,
    }


def format_result(label_names: list[str], probs: np.ndarray, threshold: float) -> dict:
    result = {}
    for name, prob in zip(label_names, probs):
        result[name] = {
            "prob": float(prob),
            "hit": bool(prob >= threshold),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DINOv2 multi-label inference on a single image")
    parser.add_argument("--checkpoint", default="outputs/dinov2_multilabel/final_model.pt")
    parser.add_argument("--config", default=None, help="Optional if checkpoint contains config")
    parser.add_argument("--image", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Using device: {device}")
    config = load_config(args.config) if args.config else None
    model, label_names, config = load_model(args.checkpoint, device, config=config)
    threshold = args.threshold if args.threshold is not None else config["inference"]["threshold"]
    transform = build_val_transform(config["data"]["img_size"])

    out = predict_image(model, args.image, transform, device, threshold)
    result = format_result(label_names, out["probabilities"], threshold)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
