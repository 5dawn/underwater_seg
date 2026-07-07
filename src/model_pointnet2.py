from __future__ import annotations

import torch
from torch import nn


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate squared distances between two point sets.

    Args:
        src: Source points with shape [B, N, C].
        dst: Target points with shape [B, M, C].
    """
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by batched indices."""
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling on xyz with shape [B, N, 3]."""
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    npoint = min(int(npoint), num_points)
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), 1e10, device=device)
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1).indices
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Find local neighborhoods around sampled centroids."""
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    sample_count = min(int(nsample), num_points)
    group_idx = torch.arange(num_points, dtype=torch.long, device=device).view(1, 1, num_points)
    group_idx = group_idx.repeat(batch_size, new_xyz.shape[1], 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = num_points
    group_idx = group_idx.sort(dim=-1).values[:, :, :sample_count]
    group_first = group_idx[:, :, 0:1].repeat(1, 1, sample_count)
    group_idx[group_idx == num_points] = group_first[group_idx == num_points]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample centroids and group neighboring point features."""
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channels: int, mlp: list[int]) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []
        last_channels = in_channels
        for out_channels in mlp:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(last_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            last_channels = out_channels
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply set abstraction.

        Args:
            xyz: Point coordinates with shape [B, 3, N].
            points: Point features with shape [B, C, N].
        """
        xyz_t = xyz.transpose(1, 2).contiguous()
        points_t = points.transpose(1, 2).contiguous() if points is not None else None
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz_t, points_t)
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, dim=2).values
        return new_xyz.transpose(1, 2).contiguous(), new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channels: int, mlp: list[int]) -> None:
        super().__init__()
        layers = []
        last_channels = in_channels
        for out_channels in mlp:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(last_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            last_channels = out_channels
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: torch.Tensor | None,
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate features from xyz2 onto xyz1."""
        xyz1_t = xyz1.transpose(1, 2).contiguous()
        xyz2_t = xyz2.transpose(1, 2).contiguous()
        points2_t = points2.transpose(1, 2).contiguous()

        if xyz2_t.shape[1] == 1:
            interpolated_points = points2_t.repeat(1, xyz1_t.shape[1], 1)
        else:
            dists = square_distance(xyz1_t, xyz2_t)
            k = min(3, xyz2_t.shape[1])
            dists, idx = dists.topk(k, dim=-1, largest=False, sorted=True)
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2_t, idx) * weight[:, :, :, None], dim=2)

        interpolated_points = interpolated_points.transpose(1, 2).contiguous()
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points
        return self.mlp(new_points)


class PointNet2Seg(nn.Module):
    """PointNet++ segmentation model with a PointNet-compatible interface."""

    def __init__(self, input_channels: int = 3, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        if input_channels < 3:
            raise ValueError("PointNet2Seg expects the first three input channels to be xyz coordinates.")

        self.sa1 = PointNetSetAbstraction(
            npoint=1024,
            radius=0.1,
            nsample=32,
            in_channels=input_channels + 3,
            mlp=[64, 64, 128],
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256,
            radius=0.2,
            nsample=32,
            in_channels=128 + 3,
            mlp=[128, 128, 256],
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=32,
            in_channels=256 + 3,
            mlp=[256, 256, 512],
        )
        self.fp3 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 128, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + input_channels, [128, 128, 128])
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-point logits for features with shape [B, C, N]."""
        xyz = features[:, :3, :]
        l0_points = features

        l1_xyz, l1_points = self.sa1(xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, l0_points, l1_points)
        return self.classifier(l0_points)
