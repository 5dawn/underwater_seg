from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABELS = {
    -1: "ignore",
    0: "background",
    1: "submarine",
}


def file_stats(path: Path) -> dict[str, int | float | str]:
    data = np.load(path)
    labels = data["labels"].astype(np.int64)
    total = int(len(labels))
    ignore = int(np.count_nonzero(labels == -1))
    background = int(np.count_nonzero(labels == 0))
    submarine = int(np.count_nonzero(labels == 1))
    known = ignore + background + submarine
    if known != total:
        invalid = sorted(set(np.unique(labels).astype(np.int64).tolist()) - set(LABELS))
        raise ValueError(f"Invalid labels in {path}: {invalid}. Allowed labels are -1, 0, 1.")
    return {
        "file": str(path),
        "total": total,
        "ignore": ignore,
        "background": background,
        "submarine": submarine,
        "ignore_ratio": float(ignore / total) if total > 0 else 0.0,
        "submarine_ratio": float(submarine / total) if total > 0 else 0.0,
    }


def summarize(rows: list[dict[str, int | float | str]]) -> dict[str, int | float]:
    total = int(sum(int(row["total"]) for row in rows))
    ignore = int(sum(int(row["ignore"]) for row in rows))
    background = int(sum(int(row["background"]) for row in rows))
    submarine = int(sum(int(row["submarine"]) for row in rows))
    return {
        "files": len(rows),
        "total": total,
        "ignore": ignore,
        "background": background,
        "submarine": submarine,
        "ignore_ratio": float(ignore / total) if total > 0 else 0.0,
        "background_ratio": float(background / total) if total > 0 else 0.0,
        "submarine_ratio": float(submarine / total) if total > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect labels in processed .npz point cloud samples.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--ignore-threshold", type=float, default=0.5)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    report: dict[str, object] = {
        "processed_dir": str(processed_dir),
        "ignore_threshold": args.ignore_threshold,
        "splits": {},
        "zero_submarine_files": [],
        "high_ignore_files": [],
    }

    for split in ("train", "val", "test"):
        files = sorted((processed_dir / split).glob("*.npz"))
        rows = [file_stats(path) for path in files]
        zero_submarine = [row["file"] for row in rows if int(row["submarine"]) == 0]
        high_ignore = [row["file"] for row in rows if float(row["ignore_ratio"]) >= args.ignore_threshold]
        report["splits"][split] = {
            "summary": summarize(rows),
            "files": rows,
        }
        report["zero_submarine_files"].extend(zero_submarine)
        report["high_ignore_files"].extend(high_ignore)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    for split, split_report in report["splits"].items():
        summary = split_report["summary"]
        print(
            f"[{split}] files={summary['files']} total={summary['total']} "
            f"ignore={summary['ignore']} ({summary['ignore_ratio']:.4f}) "
            f"background={summary['background']} ({summary['background_ratio']:.4f}) "
            f"submarine={summary['submarine']} ({summary['submarine_ratio']:.4f})"
        )
        for row in split_report["files"]:
            print(
                f"  {Path(row['file']).name}: total={row['total']} "
                f"ignore={row['ignore']} background={row['background']} "
                f"submarine={row['submarine']} submarine_ratio={row['submarine_ratio']:.4f}"
            )

    print("\nZero-submarine samples:")
    zero_files = report["zero_submarine_files"]
    if zero_files:
        for path in zero_files:
            print(f"  {path}")
    else:
        print("  none")

    print(f"\nHigh-ignore samples (ignore_ratio >= {args.ignore_threshold:.2f}):")
    high_ignore_files = report["high_ignore_files"]
    if high_ignore_files:
        for path in high_ignore_files:
            print(f"  {path}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
