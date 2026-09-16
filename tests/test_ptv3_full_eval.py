from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptv3_full_eval import evaluate_ptv3_full_cloud, spatial_cover


class CoordinateClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_lengths: list[int] = []

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        self.seen_lengths.extend([points.shape[-1]] * points.shape[0])
        score = points[:, 0]
        return torch.stack((-score, score), dim=1)


def save_cloud(path: Path, points: np.ndarray) -> None:
    labels = (points[:, 0] > 0).astype(np.int64)
    np.savez(path, points=points.astype(np.float32), labels=labels)


def test_tail_blocks_run_at_actual_length_and_cover_exactly_five(tmp_path: Path):
    rng = np.random.default_rng(7)
    points = rng.normal(size=(4103, 3)).astype(np.float32)
    cloud = tmp_path / "rack_cloud.npz"
    save_cloud(cloud, points)
    model = CoordinateClassifier()
    result = evaluate_ptv3_full_cloud(model, [cloud], torch.device("cpu"), tmp_path / "votes",
                                      votes=5, block_points=4096, batch_size=4)
    assert 7 in model.seen_lengths
    assert result["per_file"][0]["coverage_min"] == 5
    assert result["per_file"][0]["coverage_max"] == 5


def test_morton_ties_and_evaluation_are_row_order_independent(tmp_path: Path):
    # Duplicated quantized locations force Morton ties; XYZ is the explicit tie-breaker.
    points = np.asarray([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0)
                         for z in (-1.0, 1.0)] * 3, dtype=np.float32)
    rng = np.random.default_rng(99)
    shuffled = rng.permutation(len(points))
    for vote in range(5):
        ordered_xyz = points[np.concatenate(spatial_cover(points, 7, vote))]
        shuffled_xyz = points[shuffled][np.concatenate(spatial_cover(points[shuffled], 7, vote))]
        assert np.array_equal(ordered_xyz, shuffled_xyz)
    first = tmp_path / "a4noise_first.npz"
    second = tmp_path / "a4noise_second.npz"
    save_cloud(first, points)
    save_cloud(second, points[shuffled])
    model = CoordinateClassifier()
    a = evaluate_ptv3_full_cloud(model, [first], torch.device("cpu"), tmp_path / "a", block_points=7)
    b = evaluate_ptv3_full_cloud(model, [second], torch.device("cpu"), tmp_path / "b", block_points=7)
    assert a["confusion_matrix"] == b["confusion_matrix"]
