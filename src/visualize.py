from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from device import select_device
from model_factory import build_model


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_vote(vote_logits: np.ndarray, vote_counts: np.ndarray, indices: np.ndarray, logits: torch.Tensor) -> None:
    logits_np = logits.squeeze(0).transpose(0, 1).detach().cpu().numpy()
    np.add.at(vote_logits, indices, logits_np)
    np.add.at(vote_counts, indices, 1)


def predict_labels(
    npz_path: Path,
    checkpoint_path: Path,
    cfg: dict,
    requested_device: str = "auto",
    num_votes: int = 10,
    num_points: int | None = None,
) -> np.ndarray:
    device = select_device(requested_device)
    data = np.load(npz_path)
    features_np = data["features"].astype(np.float32)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_channels = int(checkpoint.get("input_channels", features_np.shape[1]))
    num_classes = int(checkpoint.get("num_classes", cfg["model"]["num_classes"]))
    num_points = int(num_points or cfg["data"].get("num_points", 4096))

    model_cfg = dict(checkpoint.get("config", cfg))
    model_cfg["model"] = dict(model_cfg.get("model", cfg.get("model", {})))
    model_cfg["model"]["name"] = checkpoint.get("model_name", model_cfg["model"].get("name", "pointnet"))
    model = build_model(model_cfg, input_channels=input_channels, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    n = len(features_np)
    rng = np.random.default_rng(cfg.get("seed", 42))
    vote_logits = np.zeros((n, num_classes), dtype=np.float64)
    vote_counts = np.zeros(n, dtype=np.int64)

    with torch.no_grad():
        for _ in range(max(int(num_votes), 1)):
            indices = rng.choice(n, num_points, replace=n < num_points)
            features = torch.from_numpy(features_np[indices]).transpose(0, 1).unsqueeze(0).to(device)
            logits = model(features)
            add_vote(vote_logits, vote_counts, indices, logits)

        uncovered = np.flatnonzero(vote_counts == 0)
        for start in range(0, len(uncovered), num_points):
            indices = uncovered[start : start + num_points]
            if len(indices) == 0:
                continue
            features = torch.from_numpy(features_np[indices]).transpose(0, 1).unsqueeze(0).to(device)
            logits = model(features)
            add_vote(vote_logits, vote_counts, indices, logits)

    mean_logits = vote_logits / np.maximum(vote_counts[:, None], 1)
    return mean_logits.argmax(axis=1).astype(np.int64)


def colorize(labels: np.ndarray, mode: str) -> np.ndarray:
    colors = np.zeros((len(labels), 3), dtype=np.float32)
    if mode == "pred":
        colors[:] = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        colors[labels == 1] = np.array([1.0, 0.15, 0.05], dtype=np.float32)
    else:
        colors[:] = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        colors[labels == -1] = np.array([0.1, 0.2, 0.9], dtype=np.float32)
        colors[labels == 1] = np.array([0.1, 0.8, 0.25], dtype=np.float32)
    return colors


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize labels or model predictions for one .npz sample.")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--config", default="configs/pointnet.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save-ply", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--num-votes", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=None)
    args = parser.parse_args()

    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Visualization requires open3d. Install with: pip install open3d") from exc

    npz_path = Path(args.npz)
    cfg = load_config(args.config)
    data = np.load(npz_path)
    points = data["points"].astype(np.float32)
    labels = data["labels"].astype(np.int64)

    if args.checkpoint:
        labels = predict_labels(npz_path, Path(args.checkpoint), cfg, args.device, args.num_votes, args.num_points)
        mode = "pred"
    else:
        mode = "gt"

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colorize(labels, mode))

    if args.save_ply:
        out = Path(args.save_ply)
        out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out), cloud)
        print(f"Saved {out}")
    else:
        o3d.visualization.draw_geometries([cloud], window_name=f"{npz_path.name} {mode}")


if __name__ == "__main__":
    main()
