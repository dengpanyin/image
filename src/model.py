from __future__ import annotations

import torch
import torch.nn as nn


DINOv2_HUB_NAMES = {
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_vitb14": "dinov2_vitb14",
    "dinov2_vitl14": "dinov2_vitl14",
}


class SimpleCNNBackbone(nn.Module):
    """Tiny CNN for offline smoke tests when DINOv2 cannot be downloaded."""

    embed_dim = 128

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.blocks = nn.ModuleList([self.features[4], self.features[7]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


def load_backbone(name: str) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNNBackbone()
    if name not in DINOv2_HUB_NAMES:
        raise ValueError(
            f"Unsupported backbone: {name}. Choose from simple_cnn or {list(DINOv2_HUB_NAMES)}"
        )
    return torch.hub.load("facebookresearch/dinov2", DINOv2_HUB_NAMES[name])


class DINOv2MultiLabelClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = load_backbone(backbone_name)
        embed_dim = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        if not hasattr(self.backbone, "blocks"):
            return
        n = min(n, len(self.backbone.blocks))
        for block in self.backbone.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    def trainable_parameter_groups(self, head_lr: float, backbone_lr: float) -> list[dict]:
        head_params = list(self.head.parameters())
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": head_lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        return groups
