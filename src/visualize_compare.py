from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def colorize_gt(labels: np.ndarray) -> np.ndarray:
    colors = np.full((len(labels), 3), [0.45, 0.45, 0.45], dtype=np.float32)
    colors[labels == -1] = np.array([0.1, 0.2, 0.9], dtype=np.float32)
    colors[labels == 1] = np.array([0.1, 0.8, 0.25], dtype=np.float32)
    return colors


def colorize_error(gt_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    colors = np.full((len(gt_labels), 3), [0.45, 0.45, 0.45], dtype=np.float32)
    valid = gt_labels != -1
    true_positive = valid & (gt_labels == 1) & (pred_labels == 1)
    false_negative = valid & (gt_labels == 1) & (pred_labels == 0)
    false_positive = valid & (gt_labels == 0) & (pred_labels == 1)

    colors[gt_labels == -1] = np.array([0.1, 0.2, 0.9], dtype=np.float32)
    colors[true_positive] = np.array([0.1, 0.8, 0.25], dtype=np.float32)
    colors[false_negative] = np.array([1.0, 0.05, 0.05], dtype=np.float32)
    colors[false_positive] = np.array([1.0, 0.85, 0.05], dtype=np.float32)
    return colors


def error_counts(gt_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, int]:
    valid = gt_labels != -1
    return {
        "tp": int(np.count_nonzero(valid & (gt_labels == 1) & (pred_labels == 1))),
        "fn": int(np.count_nonzero(valid & (gt_labels == 1) & (pred_labels == 0))),
        "fp": int(np.count_nonzero(valid & (gt_labels == 0) & (pred_labels == 1))),
        "tn": int(np.count_nonzero(valid & (gt_labels == 0) & (pred_labels == 0))),
        "ignore": int(np.count_nonzero(gt_labels == -1)),
    }


def rotation_matrix_xyz(degrees: list[float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(degrees, dtype=np.float32))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def transform_points(
    points: np.ndarray,
    base_offset_x: float,
    translate: list[float],
    rotate_degrees: list[float],
) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    rotated = (points - center) @ rotation_matrix_xyz(rotate_degrees).T + center
    shifted = rotated + np.asarray(translate, dtype=np.float32).reshape(1, 3)
    shifted[:, 0] += base_offset_x
    return shifted.astype(np.float32)


def make_cloud(points: np.ndarray, colors: np.ndarray):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Visualization requires open3d. Install with: pip install open3d") from exc

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def label_anchor(points: np.ndarray) -> np.ndarray:
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    anchor = np.array(
        [
            (xyz_min[0] + xyz_max[0]) * 0.5,
            xyz_min[1] - 0.08 * max(float(xyz_max[1] - xyz_min[1]), 1.0),
            xyz_max[2],
        ],
        dtype=np.float32,
    )
    return anchor


def make_label(text: str, anchor: np.ndarray, color: tuple[float, float, float], size: float):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Visualization requires open3d. Install with: pip install open3d") from exc

    if not hasattr(o3d.t.geometry.TriangleMesh, "create_text"):
        return None

    label = o3d.t.geometry.TriangleMesh.create_text(text, depth=0.01).to_legacy()
    label.scale(size, center=(0, 0, 0))
    label.paint_uniform_color(color)
    label.translate(anchor)
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GT and one experiment error map side by side.")
    parser.add_argument(
        "--npz",
        default="data/processed/test/10-48-56Merged clouds.npz",
        help="Processed .npz point cloud to visualize.",
    )
    parser.add_argument("--exp-name", default="exp_g", help="Experiment label shown in the Open3D window.")
    parser.add_argument("--config", "--config-d", dest="config", default="configs/pointnet_exp_g.yaml")
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-d",
        dest="checkpoint",
        default="checkpoints/pointnet_exp_g_best.pth",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--num-votes", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--save-ply", default=None, help="Optional path to save the combined comparison cloud.")
    parser.add_argument("--no-labels", action="store_true", help="Do not draw text labels above the two clouds.")
    parser.add_argument("--gt-translate", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--exp-translate",
        "--d-translate",
        dest="exp_translate",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--gt-rotate", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("RX", "RY", "RZ"))
    parser.add_argument(
        "--exp-rotate",
        "--d-rotate",
        dest="exp_rotate",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("RX", "RY", "RZ"),
    )
    args = parser.parse_args()

    from visualize import load_config, predict_labels

    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Visualization requires open3d. Install with: pip install open3d") from exc

    npz_path = Path(args.npz)
    data = np.load(npz_path)
    points = data["points"].astype(np.float32)
    gt_labels = data["labels"].astype(np.int64)

    cfg = load_config(args.config)
    pred = predict_labels(
        npz_path=npz_path,
        checkpoint_path=Path(args.checkpoint),
        cfg=cfg,
        requested_device=args.device,
        num_votes=args.num_votes,
        num_points=args.num_points,
    )

    extent_x = float(points[:, 0].max() - points[:, 0].min())
    spacing = max(extent_x * 1.35, 2.2)
    offsets = [-spacing * 0.5, spacing * 0.5]
    gt_points = transform_points(points, offsets[0], args.gt_translate, args.gt_rotate)
    exp_points = transform_points(points, offsets[1], args.exp_translate, args.exp_rotate)

    geometries = [
        make_cloud(gt_points, colorize_gt(gt_labels)),
        make_cloud(exp_points, colorize_error(gt_labels, pred)),
    ]

    if not args.no_labels:
        text_size = max(spacing * 0.045, 0.08)
        labels = [
            make_label("GT", label_anchor(gt_points), (0.1, 0.8, 0.25), text_size),
            make_label(f"{args.exp_name} error", label_anchor(exp_points), (1.0, 0.85, 0.05), text_size),
        ]
        geometries.extend(label for label in labels if label is not None)

    if args.save_ply:
        combined = geometries[0]
        for geometry in geometries[1:2]:
            combined += geometry
        out = Path(args.save_ply)
        out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out), combined)
        print(f"Saved comparison cloud: {out}")
        return

    counts = error_counts(gt_labels, pred)
    print("Colors: GT submarine=green, GT ignore=blue.")
    print("Error map: TP=green, FN=red, FP=yellow, TN=gray, ignore=blue.")
    print(f"Experiment: {args.exp_name}; checkpoint: {args.checkpoint}")
    print(f"Counts: {counts}")
    o3d.visualization.draw_geometries(geometries, window_name=f"{npz_path.name} GT vs {args.exp_name} error")


if __name__ == "__main__":
    main()
