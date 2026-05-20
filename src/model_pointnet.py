from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointNetSeg(nn.Module):
    """A compact PointNet segmentation baseline for binary point labels."""

    def __init__(self, input_channels: int = 3, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.local_mlp = nn.Sequential(
            ConvBNReLU(input_channels, 64),
            ConvBNReLU(64, 64),
            ConvBNReLU(64, 64),
        )
        self.global_mlp = nn.Sequential(
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 1024),
        )
        self.seg_head = nn.Sequential(
            ConvBNReLU(64 + 1024, 512),
            nn.Dropout(dropout),
            ConvBNReLU(512, 256),
            nn.Dropout(dropout),
            ConvBNReLU(256, 128),
            nn.Conv1d(128, num_classes, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-point logits.

        Args:
            features: Tensor with shape [B, C, N].
        """
        local_features = self.local_mlp(features)
        global_features = self.global_mlp(local_features)
        global_features = torch.max(global_features, dim=2, keepdim=True).values
        global_features = global_features.expand(-1, -1, local_features.shape[2])
        fused = torch.cat([local_features, global_features], dim=1)
        return self.seg_head(fused)
