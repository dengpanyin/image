from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import MultiLabelCSVDataset, build_train_transform, build_val_transform, compute_pos_weight
from device_utils import pick_device
from metrics import compute_metrics, predict_probs, save_metrics
from model import DINOv2MultiLabelClassifier


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> float:
    model.train(train)
    total_loss = 0.0
    count = 0

    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        count += batch_size
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(count, 1)


def train_stage(
    model: DINOv2MultiLabelClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: torch.Tensor,
    device: torch.device,
    epochs: int,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    output_dir: Path,
    stage_name: str,
    label_names: list[str],
    threshold: float,
) -> DINOv2MultiLabelClassifier:
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    param_groups = model.trainable_parameter_groups(head_lr=head_lr, backbone_lr=backbone_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    best_map = -1.0
    best_path = output_dir / f"best_{stage_name}.pt"

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, None, device, train=False)
        probs, labels = predict_probs(model, val_loader, device)
        metrics = compute_metrics(labels, probs, label_names, threshold=threshold)
        mAP = metrics["mAP"]

        print(
            f"[{stage_name}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} mAP={mAP:.4f}"
        )

        if mAP >= best_map:
            best_map = mAP
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_names": label_names,
                    "metrics": metrics,
                    "stage": stage_name,
                    "epoch": epoch,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint for {stage_name}: mAP={best_map:.4f} -> {best_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 for multi-label classification (V1)")
    parser.add_argument("--config", default="configs/dinov2_multilabel.yaml")
    parser.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    parser.add_argument("--skip-stage1", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    label_names: list[str] = cfg["labels"]
    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    print(f"Using device: {device}")
    img_size = cfg["data"]["img_size"]
    threshold = cfg["inference"]["threshold"]

    train_ds = MultiLabelCSVDataset(
        csv_path=cfg["data"]["train_csv"],
        label_names=label_names,
        image_root=cfg["data"]["image_root"],
        transform=build_train_transform(img_size),
    )
    val_ds = MultiLabelCSVDataset(
        csv_path=cfg["data"]["val_csv"],
        label_names=label_names,
        image_root=cfg["data"]["image_root"],
        transform=build_val_transform(img_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    pos_weight = compute_pos_weight(
        cfg["data"]["train_csv"],
        label_names,
        cap=cfg["train"]["pos_weight_cap"],
    )

    model = DINOv2MultiLabelClassifier(
        backbone_name=cfg["model"]["name"],
        num_classes=len(label_names),
    ).to(device)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    if not args.skip_stage1:
        print("Stage 1: freeze backbone, train classification head")
        model.freeze_backbone()
        model = train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_weight=pos_weight,
            device=device,
            epochs=cfg["train"]["stage1_epochs"],
            head_lr=cfg["train"]["head_lr"],
            backbone_lr=cfg["train"]["backbone_lr"],
            weight_decay=cfg["train"]["weight_decay"],
            output_dir=output_dir,
            stage_name="stage1",
            label_names=label_names,
            threshold=threshold,
        )

    print(f"Stage 2: unfreeze last {cfg['model']['unfreeze_last_n_blocks']} blocks")
    model.unfreeze_last_n_blocks(cfg["model"]["unfreeze_last_n_blocks"])
    model = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        pos_weight=pos_weight,
        device=device,
        epochs=cfg["train"]["stage2_epochs"],
        head_lr=cfg["train"]["head_lr"],
        backbone_lr=cfg["train"]["backbone_lr"],
        weight_decay=cfg["train"]["weight_decay"],
        output_dir=output_dir,
        stage_name="stage2",
        label_names=label_names,
        threshold=threshold,
    )

    probs, labels = predict_probs(model, val_loader, device)
    final_metrics = compute_metrics(labels, probs, label_names, threshold=threshold)
    save_metrics(final_metrics, output_dir / "val_metrics.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_names": label_names,
            "config": cfg,
            "metrics": final_metrics,
        },
        output_dir / "final_model.pt",
    )
    print(f"Training done. mAP={final_metrics['mAP']:.4f}. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
