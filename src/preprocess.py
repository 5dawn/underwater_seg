from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABEL_FIELDS = ("label", "labels", "class", "class_id", "seg_label", "semantic", "scalar_Label")
INTENSITY_FIELDS = ("intensity", "Intensity", "reflectance", "reflectivity", "scalar_Intensity")
VALID_LABELS = {-1, 0, 1}
LABEL_NAMES = {-1: "ignore", 0: "background", 1: "submarine"}


def read_ply(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("Reading PLY scalar fields requires plyfile. Install with: pip install plyfile") from exc

    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    extras: dict[str, np.ndarray] = {}
    labels = None
    for name in names:
        if name in INTENSITY_FIELDS:
            extras["intensity"] = np.asarray(vertex[name], dtype=np.float32)
        if name in LABEL_FIELDS:
            labels = np.asarray(vertex[name])
    return points, extras, labels


def read_pcd_ascii(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    header: list[str] = []
    data_start = None
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, line in enumerate(raw):
        header.append(line.strip())
        if line.upper().startswith("DATA"):
            if "ascii" not in line.lower():
                raise ValueError("Only ASCII PCD files can be parsed for custom fields")
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("Invalid PCD: missing DATA line")

    fields = []
    for line in header:
        if line.upper().startswith("FIELDS"):
            fields = line.split()[1:]
            break
    if not fields:
        raise ValueError("Invalid PCD: missing FIELDS line")

    arr = np.loadtxt(raw[data_start:], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    field_to_col = {name: idx for idx, name in enumerate(fields)}
    points = np.stack([arr[:, field_to_col["x"]], arr[:, field_to_col["y"]], arr[:, field_to_col["z"]]], axis=1)

    extras: dict[str, np.ndarray] = {}
    labels = None
    for name in INTENSITY_FIELDS:
        if name in field_to_col:
            extras["intensity"] = arr[:, field_to_col[name]].astype(np.float32)
            break
    for name in LABEL_FIELDS:
        if name in field_to_col:
            labels = arr[:, field_to_col[name]]
            break
    return points.astype(np.float32), extras, labels


def read_with_open3d(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], None]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Reading this point cloud requires open3d. Install with: pip install open3d") from exc

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    extras: dict[str, np.ndarray] = {}
    if len(cloud.colors) == len(points):
        extras["intensity"] = np.asarray(cloud.colors, dtype=np.float32).mean(axis=1)
    return points, extras, None


def read_point_cloud(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return read_ply(path)
    if suffix == ".pcd":
        try:
            return read_pcd_ascii(path)
        except Exception:
            return read_with_open3d(path)
    return read_with_open3d(path)


def load_sidecar_labels(path: Path, n: int) -> np.ndarray | None:
    candidates = [
        path.with_name(f"{path.stem}_labels.npy"),
        path.with_name(f"{path.stem}_labels.txt"),
        path.with_name(f"{path.stem}.labels.npy"),
        path.with_name(f"{path.stem}.labels.txt"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        labels = np.load(candidate) if candidate.suffix == ".npy" else np.loadtxt(candidate)
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != n:
            raise ValueError(f"Label sidecar length mismatch for {path}: {len(labels)} vs {n}")
        return labels
    return None


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale <= 0:
        scale = 1.0
    return (centered / scale).astype(np.float32), centroid.astype(np.float32), scale


def validate_labels(labels: np.ndarray, path: Path) -> None:
    labels = np.asarray(labels)
    if not np.all(np.isfinite(labels)):
        raise ValueError(f"Invalid non-finite labels in {path}. Allowed labels are -1, 0, 1.")
    if not np.all(np.isclose(labels, np.round(labels))):
        invalid = np.unique(labels[~np.isclose(labels, np.round(labels))]).tolist()
        raise ValueError(f"Non-integer labels in {path}: {invalid}. Allowed labels are -1, 0, 1.")
    unique = set(np.unique(labels.astype(np.int64)).tolist())
    invalid = sorted(unique - VALID_LABELS)
    if invalid:
        raise ValueError(f"Invalid labels in {path}: {invalid}. Allowed labels are -1, 0, 1.")


def label_statistics(labels: np.ndarray) -> dict[str, dict[str, float | int]]:
    total = int(len(labels))
    stats: dict[str, dict[str, float | int]] = {}
    for label, name in LABEL_NAMES.items():
        count = int(np.count_nonzero(labels == label))
        stats[name] = {
            "count": count,
            "ratio": float(count / total) if total > 0 else 0.0,
        }
    return stats


def empty_label_counts() -> dict[str, int]:
    return {name: 0 for name in LABEL_NAMES.values()}


def add_label_counts(target: dict[str, int], labels: np.ndarray) -> None:
    for label, name in LABEL_NAMES.items():
        target[name] += int(np.count_nonzero(labels == label))


def counts_with_ratios(counts: dict[str, int]) -> dict[str, dict[str, float | int]]:
    total = sum(counts.values())
    return {
        name: {
            "count": int(count),
            "ratio": float(count / total) if total > 0 else 0.0,
        }
        for name, count in counts.items()
    }


def build_features(
    normalized_points: np.ndarray,
    raw_points: np.ndarray,
    extras: dict[str, np.ndarray],
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    actual_names: list[str] = []
    if "xyz" in feature_names:
        features.append(normalized_points.astype(np.float32))
        actual_names.extend(["x", "y", "z"])
    if "intensity" in feature_names:
        intensity = extras.get("intensity")
        if intensity is None:
            intensity = np.zeros(len(points), dtype=np.float32)
        intensity = intensity.astype(np.float32).reshape(-1, 1)
        std = float(intensity.std())
        if std > 0:
            intensity = (intensity - float(intensity.mean())) / std
        features.append(intensity)
        actual_names.append("intensity")
    if "range" in feature_names:
        ranges = np.linalg.norm(raw_points, axis=1, keepdims=True).astype(np.float32)
        max_range = float(ranges.max())
        if max_range > 0:
            ranges /= max_range
        features.append(ranges)
        actual_names.append("range")
    if not features:
        raise ValueError("At least one feature group must be selected")
    return np.concatenate(features, axis=1).astype(np.float32), actual_names


def split_files(files: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    files = [files[i] for i in order]
    n_train = int(round(len(files) * train_ratio))
    n_val = int(round(len(files) * val_ratio))
    return {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }


def process_file(
    path: Path,
    out_path: Path,
    feature_names: list[str],
    allow_unlabeled: bool,
) -> tuple[list[str], dict[str, dict[str, float | int]]]:
    points, extras, labels = read_point_cloud(path)
    if len(points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    if labels is None:
        labels = load_sidecar_labels(path, len(points))
    if labels is None:
        if not allow_unlabeled:
            raise ValueError(f"No labels found for {path}. Add label field or sidecar labels.")
        labels = np.zeros(len(points), dtype=np.int64)

    validate_labels(labels, path)
    labels = labels.astype(np.int64)
    norm_points, centroid, scale = normalize_points(points)
    features, actual_names = build_features(norm_points, points, extras, feature_names)
    stats = label_statistics(labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        points=norm_points,
        features=features,
        labels=labels.astype(np.int64),
        centroid=centroid,
        scale=np.asarray(scale, dtype=np.float32),
        source=str(path),
    )
    return actual_names, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw point clouds to normalized .npz samples.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--features", default="xyz", help="Comma-separated groups: xyz,intensity,range")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-unlabeled", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    feature_names = [x.strip() for x in args.features.split(",") if x.strip()]
    files = sorted([p for p in raw_dir.rglob("*") if p.suffix.lower() in {".ply", ".pcd"}])
    if not files:
        raise FileNotFoundError(f"No .ply or .pcd files found in {raw_dir}")

    splits = split_files(files, args.train_ratio, args.val_ratio, args.seed)
    actual_feature_names: list[str] | None = None
    counts: dict[str, int] = {}
    label_counts: dict[str, dict[str, int]] = {}
    file_label_stats: dict[str, dict[str, dict[str, float | int]]] = {}
    for split, split_files_ in splits.items():
        counts[split] = len(split_files_)
        label_counts[split] = empty_label_counts()
        file_label_stats[split] = {}
        for path in split_files_:
            names, stats = process_file(
                path=path,
                out_path=out_dir / split / f"{path.stem}.npz",
                feature_names=feature_names,
                allow_unlabeled=args.allow_unlabeled,
            )
            actual_feature_names = names
            file_label_stats[split][path.name] = stats
            for label, name in LABEL_NAMES.items():
                label_counts[split][name] += int(stats[name]["count"])

    metadata = {
        "feature_names": actual_feature_names,
        "feature_behavior": {
            "points": "Saved points are centered and scaled normalized xyz coordinates.",
            "range": "When requested, range is computed from raw input coordinates before centering/scaling, then divided by the per-cloud max range.",
        },
        "label_mapping": {"-1": "ignore", "0": "background", "1": "submarine"},
        "class_names": ["background", "submarine"],
        "counts": counts,
        "label_statistics": {split: counts_with_ratios(split_counts) for split, split_counts in label_counts.items()},
        "file_label_statistics": file_label_stats,
        "source_dir": str(raw_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
