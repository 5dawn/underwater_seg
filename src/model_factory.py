from __future__ import annotations

import torch

from model_pointnet import PointNetSeg
from model_pointnet2 import PointNet2Seg


def get_model_name(cfg: dict) -> str:
    return str(cfg.get("model", {}).get("name", "pointnet")).lower()


def build_model(cfg: dict, input_channels: int, num_classes: int) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    model_name = get_model_name(cfg)
    dropout = float(model_cfg.get("dropout", 0.3))

    if model_name == "pointnet":
        return PointNetSeg(input_channels=input_channels, num_classes=num_classes, dropout=dropout)
    if model_name in {"pointnet2", "pointnet++"}:
        return PointNet2Seg(input_channels=input_channels, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unsupported model.name: {model_name}")
