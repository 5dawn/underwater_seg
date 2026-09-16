from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VoxelizedSample:
    coord: torch.Tensor
    feat: torch.Tensor
    grid_coord: torch.Tensor
    inverse: torch.Tensor
    counts: torch.Tensor


def voxelize_xyz(points: torch.Tensor, voxel_size: float) -> VoxelizedSample:
    """Label-free, per-sample mean aggregation with an exact inverse map."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    grid = torch.floor(points / voxel_size).to(torch.long)
    unique, inverse, counts = torch.unique(
        grid, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    sums = points.new_zeros((len(unique), 3))
    sums.index_add_(0, inverse, points)
    means = sums / counts.to(points.dtype).unsqueeze(1)
    # Official PTv3 serialization expects non-negative integer grid coordinates.
    # Translation does not change voxel membership or the inverse mapping.
    serialized_grid = unique - unique.min(dim=0).values
    return VoxelizedSample(means, means, serialized_grid, inverse, counts)


def voxel_diagnostics(points: torch.Tensor, voxel_size: float) -> dict[str, float | int]:
    voxels = voxelize_xyz(points, voxel_size)
    counts = voxels.counts.float()
    quantiles = torch.quantile(counts, counts.new_tensor([0.5, 0.9, 0.99]))
    return {
        "input_points": int(len(points)),
        "valid_voxels": int(len(counts)),
        "unique_voxel_ratio": float(len(counts) / max(len(points), 1)),
        "single_point_voxel_ratio": float((counts == 1).float().mean()),
        "points_per_voxel_mean": float(counts.mean()),
        "points_per_voxel_median": float(quantiles[0]),
        "points_per_voxel_p90": float(quantiles[1]),
        "points_per_voxel_p99": float(quantiles[2]),
        "points_per_voxel_max": int(counts.max()),
    }


def choose_voxel_size(rows: list[dict], initial: float = 0.005) -> tuple[float, str]:
    """Apply the frozen, single-correction engineering rule."""
    if not rows:
        raise ValueError("at least one diagnostic row is required")
    ratios = torch.tensor([row["unique_voxel_ratio"] for row in rows])
    singles = torch.tensor([row["single_point_voxel_ratio"] for row in rows])
    ratio_median = float(ratios.median())
    single_median = float(singles.median())
    if ratio_median < 0.25:
        return 0.0025, f"unique voxel ratio median {ratio_median:.6f} < 0.25"
    if ratio_median > 0.95 and single_median > 0.90:
        return 0.01, (
            f"unique voxel ratio median {ratio_median:.6f} > 0.95 and "
            f"single-point voxel ratio median {single_median:.6f} > 0.90"
        )
    return initial, "neither frozen correction trigger fired"


class UnderwaterLocalFeatureRefine(nn.Module):
    def __init__(self, channels: int, radius: float, k: int = 16, chunk_size: int = 512,
                 zero_difference: bool = False, zero_geometry: bool = False):
        super().__init__()
        if radius <= 0 or k <= 0 or chunk_size <= 0:
            raise ValueError("radius, k, and chunk_size must be positive")
        self.channels = int(channels)
        self.radius = float(radius)
        self.k = int(k)
        self.chunk_size = int(chunk_size)
        self.zero_difference = bool(zero_difference)
        self.zero_geometry = bool(zero_geometry)
        self.last_diagnostics: dict[str, float | int] = {}
        hidden = max(channels // 4, 16)
        self.w1 = nn.Linear(2 * channels + 7, hidden)
        self.act = nn.GELU()
        self.w2 = nn.Linear(hidden, channels)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or points.ndim != 3:
            raise ValueError("features and points must be [B, N, C] and [B, N, 3]")
        if features.shape[:2] != points.shape[:2] or features.shape[2] != self.channels:
            raise ValueError("feature/point shapes do not match the configured channels")
        residuals = []
        neighbor_counts = []
        for batch_idx in range(features.shape[0]):
            residual, counts = self._sample_residual(features[batch_idx], points[batch_idx])
            residuals.append(residual)
            neighbor_counts.append(counts)
        stacked = torch.stack(residuals, dim=0)
        all_counts = torch.cat(neighbor_counts).float()
        self.last_diagnostics = {
            "neighbor_count_min": int(all_counts.min().item()),
            "neighbor_count_mean": float(all_counts.mean().item()),
            "neighbor_count_max": int(all_counts.max().item()),
            "fraction_below_three": float((all_counts < 3).float().mean().item()),
            "residual_rms": float(stacked.detach().float().square().mean().sqrt().item()),
        }
        return features + stacked

    def _sample_residual(self, feat: torch.Tensor, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(xyz)
        output = feat.new_zeros(feat.shape)
        all_counts = torch.zeros(n, dtype=torch.long, device=xyz.device)
        if n <= 1:
            return output, all_counts
        take = min(self.k, n - 1)
        for start in range(0, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            distances = torch.cdist(xyz[start:stop], xyz)
            rows = torch.arange(stop - start, device=xyz.device)
            cols = torch.arange(start, stop, device=xyz.device)
            distances[rows, cols] = torch.inf
            values, indices = torch.topk(distances, take, largest=False, sorted=True)
            valid = values <= self.radius
            count = valid.sum(dim=1)
            all_counts[start:stop] = count
            mask = valid.unsqueeze(-1)
            neighbor_feat = feat[indices]
            denom = count.clamp_min(1).to(feat.dtype).unsqueeze(1)
            mean_feat = (neighbor_feat * mask).sum(dim=1) / denom
            difference = feat[start:stop] - mean_feat
            if self.zero_difference:
                difference = torch.zeros_like(difference)

            relative = (xyz[indices] - xyz[start:stop, None, :]) / (self.radius + 1e-8)
            mean_rel = (relative * mask).sum(dim=1) / denom
            centered = relative - mean_rel[:, None, :]
            variance = (centered.square() * mask).sum(dim=1) / denom
            std_rel = torch.sqrt(variance.clamp_min(0) + 1e-12)
            ratio = (count.to(feat.dtype) / self.k).unsqueeze(1)
            geometry = torch.cat((mean_rel, std_rel, ratio), dim=1)
            if self.zero_geometry:
                geometry = torch.zeros_like(geometry)
            hidden = self.act(self.w1(torch.cat((feat[start:stop], difference, geometry), dim=1)))
            residual = self.w2(hidden)
            output[start:stop] = residual * (count >= 3).to(feat.dtype).unsqueeze(1)
        return output, all_counts
