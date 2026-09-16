from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptv3_protocol import voxelize_xyz


SMOKE_DIR = Path(os.environ.get("PTV3_SMOKE_DIR", ""))


def report(name: str) -> dict:
    if not os.environ.get("PTV3_SMOKE_DIR"):
        pytest.skip("set PTV3_SMOKE_DIR after running the separate CUDA smoke scripts")
    path = SMOKE_DIR / name
    if not path.exists():
        pytest.skip(f"missing separate CUDA smoke artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def test_official_ptv3_forward():
    data = report("official_smoke.json")
    assert data["finite_output"] and data["output_feature_shape"][1] == 64


def test_official_ptv3_backward():
    data = report("official_smoke.json")
    assert data["finite_gradients"] and data["loss"] > 0


def test_ptv3_adapter_shape():
    assert report("adapter_smoke.json")["logits_shape"] == [4, 2, 4096]


def test_ptv3_inverse_mapping():
    xyz = torch.tensor([[0.001, 0, 0], [0.002, 0, 0], [0.02, 0, 0]])
    item = voxelize_xyz(xyz, 0.01)
    assert item.inverse.tolist() == [0, 0, 1]


def test_ptv3_mixed_batch():
    data = report("adapter_smoke.json")
    assert len(data["voxel_counts"]) == 4 and all(x <= 4096 for x in data["voxel_counts"])


def test_ptv3_ignore_label():
    data = report("real_data_smoke.json")["real_batch_step"]
    assert data["ignore_points"] > 0 and data["finite_gradients"]


def test_ptv3_checkpoint():
    data = report("adapter_smoke.json")
    assert data["strict_checkpoint_reload"]
    assert data["reload_max_abs_logit_difference"] == 0.0


def test_ptv3_validation_consistency():
    train_val = report("val_smoke.json")
    independent = report("val_smoke_independent.json")
    assert train_val == independent
    assert len(train_val["per_file"]) == 9
    assert all(x["coverage_min"] == x["coverage_max"] == 5 for x in train_val["per_file"])
