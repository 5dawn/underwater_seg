from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from model_pointnet import PointNetSeg


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict_labels(npz_path: Path, checkpoint_path: Path, cfg: dict) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(npz_path)
    features_np = data["features"].astype(np.float32)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_channels = int(checkpoint.get("input_channels", features_np.shape[1]))
    num_classes = int(checkpoint.get("num_classes", cfg["model"]["num_classes"]))

    model = PointNetSeg(
        input_channels=input_channels,
        num_classes=num_classes,
        dropout=float(cfg["model"].get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(features_np).transpose(0, 1).unsqueeze(0).to(device)
        logits = model(features)
        return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)


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
        labels = predict_labels(npz_path, Path(args.checkpoint), cfg)
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
