from __future__ import annotations

import torch
from torch import nn


def knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Return k-nearest-neighbor indices from xyz with shape [B, 3, N]."""
    xyz_t = xyz.transpose(1, 2).contiguous()
    num_points = xyz_t.shape[1]
    if num_points == 1:
        return torch.zeros(xyz_t.shape[0], 1, 1, dtype=torch.long, device=xyz.device)
    k = min(int(k), num_points - 1)
    dists = torch.cdist(xyz_t, xyz_t)
    return dists.topk(k=k + 1, dim=-1, largest=False).indices[:, :, 1:]


def index_neighbors(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather neighbor features.

    Args:
        features: Tensor with shape [B, C, N].
        idx: Neighbor indices with shape [B, N, K].
    """
    batch_size, channels, num_points = features.shape
    k = idx.shape[-1]
    features_t = features.transpose(1, 2).contiguous()
    batch_idx = torch.arange(batch_size, device=features.device).view(batch_size, 1, 1)
    neighbors = features_t[batch_idx, idx]
    return neighbors.permute(0, 3, 1, 2).contiguous().view(batch_size, channels, num_points, k)


def edge_features(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Build EdgeConv features concat(x_i, x_j - x_i)."""
    neighbors = index_neighbors(features, idx)
    central = features.unsqueeze(-1).expand_as(neighbors)
    return torch.cat([central, neighbors - central], dim=1)


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        edges = edge_features(features, idx)
        return self.net(edges).max(dim=-1).values


class DGCNNSeg(nn.Module):
    """Lightweight DGCNN segmentation baseline with a PointNet-compatible interface."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3,
        k: int = 20,
        emb_channels: int = 512,
    ) -> None:
        super().__init__()
        if input_channels < 3:
            raise ValueError("DGCNNSeg expects the first three input channels to be xyz coordinates.")
        self.k = int(k)
        self.edge1 = EdgeConvBlock(input_channels, 64)
        self.edge2 = EdgeConvBlock(64, 64)
        self.edge3 = EdgeConvBlock(64, 128)
        local_channels = 64 + 64 + 128
        self.global_mlp = nn.Sequential(
            nn.Conv1d(local_channels, emb_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.seg_head = nn.Sequential(
            nn.Conv1d(local_channels + emb_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-point logits for features with shape [B, C, N]."""
        idx = knn_indices(features[:, :3, :], self.k)
        x1 = self.edge1(features, idx)
        x2 = self.edge2(x1, idx)
        x3 = self.edge3(x2, idx)
        local = torch.cat([x1, x2, x3], dim=1)
        global_features = self.global_mlp(local).max(dim=2, keepdim=True).values
        global_features = global_features.expand(-1, -1, local.shape[2])
        return self.seg_head(torch.cat([local, global_features], dim=1))
