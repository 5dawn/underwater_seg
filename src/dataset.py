from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class UnderwaterSegDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        num_points: int = 4096,
        augment: bool = False,
        augment_cfg: dict | None = None,
        sample_mode: str = "random",
        foreground_ratio: float = 0.25,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.augment_cfg = augment_cfg or {}
        self.sample_mode = sample_mode
        self.foreground_ratio = foreground_ratio
        self.files = sorted((self.root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.root / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        path = self.files[idx]
        data = np.load(path)
        points = data["points"].astype(np.float32)
        features = data["features"].astype(np.float32)
        labels = data["labels"].astype(np.int64)

        choice = self._sample_indices(labels)
        points = points[choice]
        features = features[choice]
        labels = labels[choice]

        if self.augment:
            points, features = self._augment(points, features)

        return {
            "points": torch.from_numpy(points),
            "features": torch.from_numpy(features).transpose(0, 1),
            "labels": torch.from_numpy(labels),
            "path": str(path),
        }

    def _sample_indices(self, labels: np.ndarray) -> np.ndarray:
        n = len(labels)
        if self.sample_mode == "foreground_balanced":
            return self._foreground_balanced_indices(labels)
        if self.sample_mode != "random":
            raise ValueError(f"Unsupported sample_mode: {self.sample_mode}")
        replace = n < self.num_points
        return np.random.choice(n, self.num_points, replace=replace)

    def _foreground_balanced_indices(self, labels: np.ndarray) -> np.ndarray:
        n = len(labels)
        foreground = np.flatnonzero(labels == 1)
        target_fg = int(round(self.num_points * self.foreground_ratio)) if len(foreground) else 0
        target_fg = min(target_fg, self.num_points)

        parts: list[np.ndarray] = []
        if target_fg > 0:
            parts.append(np.random.choice(foreground, target_fg, replace=len(foreground) < target_fg))

        remaining = self.num_points - target_fg
        if remaining > 0:
            replace = n < remaining
            parts.append(np.random.choice(n, remaining, replace=replace))

        choice = np.concatenate(parts) if parts else np.random.choice(n, self.num_points, replace=n < self.num_points)
        np.random.shuffle(choice)
        return choice

    def _augment(self, points: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.augment_cfg
        xyz = points.copy()

        if cfg.get("rotate_z", True):
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            xyz = xyz @ rot.T

        scale = np.random.uniform(cfg.get("scale_min", 1.0), cfg.get("scale_max", 1.0))
        xyz *= scale

        shift_std = cfg.get("shift_std", 0.0)
        if shift_std > 0:
            xyz += np.random.normal(0, shift_std, size=(1, 3)).astype(np.float32)

        jitter_std = cfg.get("jitter_std", 0.0)
        if jitter_std > 0:
            clip = cfg.get("jitter_clip", 0.02)
            xyz += np.clip(np.random.normal(0, jitter_std, xyz.shape), -clip, clip).astype(np.float32)

        augmented = features.copy()
        augmented[:, :3] = xyz

        dropout = cfg.get("point_dropout", 0.0)
        if dropout > 0:
            mask = np.random.rand(len(augmented)) < dropout
            if mask.any():
                keep = np.where(~mask)[0]
                if len(keep) > 0:
                    replacement = np.random.choice(keep, size=mask.sum(), replace=True)
                    xyz[mask] = xyz[replacement]
                    augmented[mask] = augmented[replacement]

        return xyz.astype(np.float32), augmented.astype(np.float32)


def infer_input_channels(root: str | Path, split: str = "train") -> int:
    first = next(iter(sorted((Path(root) / split).glob("*.npz"))), None)
    if first is None:
        raise FileNotFoundError(f"No .npz files found in {Path(root) / split}")
    with np.load(first) as data:
        return int(data["features"].shape[1])


def compute_class_weights(root: str | Path, split: str = "train", num_classes: int = 2) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for path in sorted((Path(root) / split).glob("*.npz")):
        labels = np.load(path)["labels"].astype(np.int64)
        counts += np.bincount(labels[(labels >= 0) & (labels < num_classes)], minlength=num_classes)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
