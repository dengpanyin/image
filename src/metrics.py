from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score


@torch.no_grad()
def predict_probs(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].cpu().numpy()
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int32)
    per_class = {}
    ap_scores = []

    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        yhat = y_pred[:, i]
        if yt.sum() == 0:
            ap = 0.0
        else:
            ap = float(average_precision_score(yt, yp))
        ap_scores.append(ap)

        tp = int(((yhat == 1) & (yt == 1)).sum())
        fp = int(((yhat == 1) & (yt == 0)).sum())
        fn = int(((yhat == 0) & (yt == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[name] = {
            "ap": ap,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(yt, yhat, zero_division=0),
            "support_pos": int(yt.sum()),
        }

    return {
        "mAP": float(np.mean(ap_scores)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "threshold": threshold,
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
