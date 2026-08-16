# Indoor Danger Multi-Label Classification

Fine-tune an image classifier for **multi-label indoor hazard detection** (one image can have multiple labels).

This repo currently ships a **V1 single-frame** pipeline:

- CSV-based multi-label dataset
- Backbone + linear classification head
- Two-stage training (freeze backbone, then partial fine-tune)
- Single-image inference

Future work (not implemented here yet): YOLO branch, TCN temporal models, zone rules.

---

## Requirements

- Python 3.10+
- macOS (Apple Silicon), Linux, or Windows
- Optional: CUDA GPU or Apple Silicon (MPS)

---

## Quick Start

### 1. Create virtual environment

```bash
cd classification
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create tiny sample data

This generates **11 synthetic images** so you can smoke-test the pipeline without downloading a dataset:

```bash
python3 scripts/create_tiny_sample.py
```

Outputs:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/images/train/`
- `data/images/val/`

These images are **not real photos**. They only verify that training/inference code runs.

### 3. Train

**MacBook / offline smoke test** (uses small `simple_cnn`, no model download):

```bash
python3 src/train.py --config configs/dinov2_multilabel_mac.yaml --device auto
```

**Default / GPU config** (uses DINOv2 ViT-B, requires network on first run):

```bash
python3 src/train.py --config configs/dinov2_multilabel.yaml --device auto
```

Training artifacts are saved to the directory configured in YAML, for example:

- `outputs/dinov2_multilabel_mac/final_model.pt`
- `outputs/dinov2_multilabel_mac/val_metrics.json`

### 4. Run inference

```bash
python3 src/infer.py \
  --checkpoint outputs/dinov2_multilabel_mac/final_model.pt \
  --image data/images/train/001_smoke.jpg \
  --device auto
```

Example output:

```json
{
  "fire_smoke": {"prob": 0.53, "hit": true},
  "lane_blocked": {"prob": 0.50, "hit": true}
}
```

---

## Project Layout

```text
classification/
├── configs/
│   ├── dinov2_multilabel.yaml       # Default DINOv2 ViT-B config
│   └── dinov2_multilabel_mac.yaml   # Mac/offline-friendly config
├── data/
│   ├── images/                      # Image files
│   └── splits/                      # train.csv / val.csv
├── scripts/
│   ├── create_tiny_sample.py        # Generate 11 synthetic samples
│   ├── download_sample_dataset.py   # Download real fire/smoke subset
│   └── train.sh
├── src/
│   ├── dataset.py                   # CSV dataset + transforms
│   ├── model.py                     # Backbone + multi-label head
│   ├── train.py                     # Two-stage fine-tuning
│   ├── infer.py                     # Single-image inference
│   ├── metrics.py                   # mAP / precision / recall
│   └── device_utils.py              # auto | mps | cuda | cpu
└── outputs/                         # Checkpoints and metrics
```

---

## Dataset Format

`train.csv` and `val.csv` use one row per image:

```csv
image_path,fire_smoke,lane_blocked,debris,door_open_abnormal,wet_floor
train/001_smoke.jpg,1,0,0,0,0
train/003_normal.jpg,0,0,0,0,0
```

Rules:

- `image_path` is relative to `data.image_root` in the config
- Labels are `0` or `1`
- One image may have multiple `1` values

See `data/splits/train.csv.example` for a template.

---

## Configuration

Edit the YAML config before training.

| Key | Description |
|-----|-------------|
| `model.name` | Backbone: `simple_cnn`, `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14` |
| `model.unfreeze_last_n_blocks` | How many ViT blocks to unfreeze in stage 2 |
| `labels` | Class names used in CSV and model output |
| `data.train_csv` / `data.val_csv` | Annotation files |
| `data.image_root` | Root folder for images |
| `data.img_size` | Input resize/pad size |
| `train.batch_size` | Batch size |
| `train.stage1_epochs` | Epochs with frozen backbone |
| `train.stage2_epochs` | Epochs with partial backbone fine-tune |
| `inference.threshold` | Default sigmoid threshold |

---

## Backbone Options

| Name | Use case |
|------|----------|
| `simple_cnn` | Offline smoke test, MacBook, no download |
| `dinov2_vits14` | Small ViT, good for MacBook when hub works |
| `dinov2_vitb14` | Default quality/speed balance |
| `dinov2_vitl14` | Highest quality, needs more GPU memory |

First-time DINOv2 runs download weights via `torch.hub` from GitHub. If download fails, use `simple_cnn` first, then switch back later.

---

## Device Selection

`--device auto` picks:

1. CUDA if available
2. else Apple MPS
3. else CPU

You can also force a device:

```bash
python3 src/train.py --config configs/dinov2_multilabel_mac.yaml --device mps
python3 src/train.py --config configs/dinov2_multilabel_mac.yaml --device cpu
```

---

## Training Stages

Training runs in two stages:

1. **Stage 1**: freeze backbone, train classification head only
2. **Stage 2**: unfreeze last N backbone blocks, fine-tune with smaller LR

Skip stage 1 if needed:

```bash
python3 src/train.py --config configs/dinov2_multilabel_mac.yaml --skip-stage1
```

---

## Optional: Download Real Sample Data

To replace synthetic data with a small real indoor fire/smoke subset:

```bash
python3 scripts/download_sample_dataset.py --max-train 200 --max-val 50
```

This downloads the [Home-fire dataset](https://github.com/PengBo0/Home-fire-dataset) test split from GitHub, converts YOLO labels to this repo's CSV format, and writes:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/images/...`

Requires a stable network connection. The archive is several hundred MB.

---

## Tips for MacBook Air

Use `configs/dinov2_multilabel_mac.yaml`:

- `model.name: simple_cnn` for offline testing
- `model.name: dinov2_vits14` when GitHub/torch.hub is reachable
- `img_size: 224`
- `batch_size: 2` if you run out of memory
- `num_workers: 0`

---

## Common Commands

```bash
# Regenerate synthetic sample data
python3 scripts/create_tiny_sample.py

# Train (Mac/offline)
python3 src/train.py --config configs/dinov2_multilabel_mac.yaml --device auto

# Train (DINOv2 ViT-B)
python3 src/train.py --config configs/dinov2_multilabel.yaml --device auto

# Inference
python3 src/infer.py \
  --checkpoint outputs/dinov2_multilabel_mac/final_model.pt \
  --image path/to/image.jpg \
  --device auto \
  --threshold 0.5
```

---

## Notes

- Synthetic sample data is only for verifying the code path.
- For production, train on real labeled indoor monitoring data.
- Person-related hazards (helmet, fall, intrusion) should eventually use a detection/tracking branch, not this whole-image classifier alone.
