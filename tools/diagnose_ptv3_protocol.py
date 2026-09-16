from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptv3_protocol import choose_voxel_size, voxel_diagnostics  # noqa: E402


def fixed_indices(n: int, count: int, seed: int, file_index: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, file_index]))
    return rng.choice(n, count, replace=n < count)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def aggregate(rows: list[dict]) -> dict:
    keys = [key for key in rows[0] if key not in {"file", "scene"}]
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "median": float(np.median([row[key] for row in rows])),
            "p90": percentile(np.asarray([row[key] for row in rows]), 90),
            "p99": percentile(np.asarray([row[key] for row in rows]), 99),
            "min": float(np.min([row[key] for row in rows])),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen train-only PTv3 voxel/radius diagnostics")
    parser.add_argument("--processed-dir", default="data/processed_label2")
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/ptv3/diagnostics_initial.json")
    args = parser.parse_args()
    files = sorted((ROOT / args.processed_dir / "train").glob("*.npz"))
    if len(files) != 45:
        raise RuntimeError(f"expected exactly 45 train files, found {len(files)}")
    rows, d16 = [], []
    for file_index, path in enumerate(files):
        with np.load(path) as data:
            points = data["points"].astype(np.float32)
        selected = points[fixed_indices(len(points), args.num_points, args.seed, file_index)]
        row = voxel_diagnostics(torch.from_numpy(selected), args.voxel_size)
        row.update(file=path.name, scene=path.name.split("_", 1)[0])
        rows.append(row)
        # Query 17 because the first returned point is the point itself.
        distances, _ = cKDTree(selected).query(selected, k=17, workers=1)
        d16.extend(distances[:, 16].tolist())
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)
    selected_size, reason = choose_voxel_size(rows, args.voxel_size)
    report = {
        "scope": "train only; fixed 4096-point samples; labels unused",
        "seed": args.seed,
        "initial_voxel_size": args.voxel_size,
        "selected_voxel_size": selected_size,
        "selection_reason": reason,
        "radius_p95_d16": percentile(np.asarray(d16), 95),
        "overall": aggregate(rows),
        "per_scene": {scene: aggregate(items) for scene, items in sorted(by_scene.items())},
        "per_file": rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["selected_voxel_size", "selection_reason", "radius_p95_d16"]}, indent=2))


if __name__ == "__main__":
    main()
