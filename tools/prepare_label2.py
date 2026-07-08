from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


NAME_RE = re.compile(
    r"^(?P<run>R\d+)-(?P<distance>D\d+)-(?P<pose>P\d+)-(?P<scene>S\d+)-(?P<laser>A\d+)\.ply$",
    re.IGNORECASE,
)

LABEL_FIELDS = (
    "label",
    "labels",
    "class",
    "class_id",
    "seg_label",
    "semantic",
    "scalar_Label",
    "scalar_label",
)


def read_ply_header(path: Path) -> tuple[list[str], int]:
    lines: list[str] = []
    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Invalid PLY without end_header: {path}")
            line = raw.decode("ascii", errors="replace").rstrip("\r\n")
            lines.append(line)
            if line == "end_header":
                return lines, f.tell()


def parse_header(lines: list[str]) -> dict:
    if not lines or lines[0] != "ply":
        raise ValueError("Not a PLY file")
    fmt = None
    vertex_count = None
    vertex_props: list[tuple[str, str]] = []
    in_vertex = False
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts[0] == "property" and len(parts) == 3:
            vertex_props.append((parts[1], parts[2]))
    if fmt != "binary_little_endian":
        raise ValueError(f"Only binary_little_endian PLY is supported here, got {fmt}")
    if vertex_count is None:
        raise ValueError("No vertex element in PLY")
    return {"format": fmt, "vertex_count": vertex_count, "vertex_props": vertex_props}


def dtype_for_props(props: list[tuple[str, str]]) -> np.dtype:
    type_map = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int16": "<i2",
        "uint16": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "int32": "<i4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    fields = []
    for typ, name in props:
        if typ not in type_map:
            raise ValueError(f"Unsupported PLY property type: {typ}")
        fields.append((name, type_map[typ]))
    return np.dtype(fields)


def label_stats(path: Path) -> dict:
    header, offset = read_ply_header(path)
    meta = parse_header(header)
    props = meta["vertex_props"]
    names = [name for _, name in props]
    label_field = next((name for name in LABEL_FIELDS if name in names), None)
    if label_field is None:
        return {
            "vertex_count": meta["vertex_count"],
            "label_field": None,
            "labels": {},
            "valid": False,
            "error": "missing label field",
        }
    arr = np.memmap(path, dtype=dtype_for_props(props), mode="r", offset=offset, shape=(meta["vertex_count"],))
    labels = np.asarray(arr[label_field])
    rounded = np.rint(labels).astype(np.int64)
    counts = Counter(rounded.tolist())
    valid = set(counts).issubset({-1, 0, 1}) and np.allclose(labels, rounded)
    return {
        "vertex_count": int(meta["vertex_count"]),
        "label_field": label_field,
        "labels": {str(k): int(v) for k, v in sorted(counts.items())},
        "valid": bool(valid),
        "error": None if valid else "labels must be integer values in {-1, 0, 1}",
    }


def parse_group(path: Path, root: Path) -> dict:
    rel = path.relative_to(root)
    m = NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected file name: {rel}")
    group = m.groupdict()
    if rel.parts[0] == "C":
        group["condition"] = "C_pool_bottom"
    elif rel.parts[0] == "A4":
        group["condition"] = "A4_strong_laser_noise"
    else:
        group["condition"] = "S1_A1_normal"
    return group


def unique_name(group: dict, original: str) -> str:
    prefix = {
        "S1_A1_normal": "normal",
        "C_pool_bottom": "poolbottom",
        "A4_strong_laser_noise": "a4noise",
    }[group["condition"]]
    return f"{prefix}_{original}"


def choose_split(group: dict) -> str:
    condition = group["condition"]
    d = group["distance"]
    p = group["pose"]
    if condition == "A4_strong_laser_noise":
        if d == "D1" and p in {"P1", "P2"}:
            return "train"
        if d == "D2" and p == "P1":
            return "val"
        return "test"
    if condition == "C_pool_bottom":
        if d == "D3" and p in {"P3", "P4", "P5"}:
            return "test"
        if (d, p) in {("D2", "P5"), ("D3", "P2")}:
            return "val"
        return "train"
    if d == "D3" and p in {"P4", "P5"}:
        return "test"
    if (d, p) in {("D2", "P5"), ("D3", "P2")}:
        return "val"
    return "train"


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "exists"
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and stage label2 point clouds for training.")
    parser.add_argument("--label-dir", default="label2")
    parser.add_argument("--raw-out", default="data/raw_label2")
    parser.add_argument("--split-out", default="splits_label2.json")
    parser.add_argument("--report-out", default="label2_group_report.json")
    args = parser.parse_args()

    root = Path(args.label_dir)
    raw_out = Path(args.raw_out)
    files = sorted(root.rglob("*.ply"))
    if not files:
        raise FileNotFoundError(f"No .ply files found in {root}")

    records = []
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    grouped_counts: dict[str, Counter] = defaultdict(Counter)
    total_labels = Counter()
    link_modes = Counter()

    for path in files:
        group = parse_group(path, root)
        stats = label_stats(path)
        staged_name = unique_name(group, path.name)
        split = choose_split(group)
        mode = link_or_copy(path, raw_out / staged_name)
        link_modes[mode] += 1
        splits[split].append(staged_name)
        for label, count in stats["labels"].items():
            total_labels[label] += count
            grouped_counts[group["condition"]][label] += count
        records.append(
            {
                "source": str(path),
                "staged": str(raw_out / staged_name),
                "staged_name": staged_name,
                "split": split,
                **group,
                **stats,
            }
        )

    for split in splits:
        splits[split] = sorted(splits[split])

    report = {
        "source_dir": str(root),
        "raw_out": str(raw_out),
        "split_out": args.split_out,
        "file_count": len(records),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "condition_counts": dict(Counter(r["condition"] for r in records)),
        "distance_counts": dict(Counter(r["distance"] for r in records)),
        "pose_counts": dict(Counter(r["pose"] for r in records)),
        "laser_counts": dict(Counter(r["laser"] for r in records)),
        "label_totals": dict(sorted(total_labels.items())),
        "label_totals_by_condition": {
            condition: dict(sorted(counter.items())) for condition, counter in grouped_counts.items()
        },
        "stage_modes": dict(link_modes),
        "invalid_files": [r for r in records if not r["valid"]],
        "records": records,
    }
    Path(args.split_out).write_text(json.dumps(splits, indent=2), encoding="utf-8")
    Path(args.report_out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
