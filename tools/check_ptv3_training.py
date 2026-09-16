from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


def max_state_difference(a: dict, b: dict) -> float:
    values = []
    for key in a:
        if torch.is_tensor(a[key]):
            values.append(float((a[key].detach().cpu() - b[key].detach().cpu()).abs().max()))
    return max(values, default=0.0)


def train_step(model, optimizer, points, labels) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(points), labels, ignore_index=-1)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--common-init", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "tools"))
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    official = Path(cfg["official_source"]["local_path"])
    sys.path[:0] = [str(project / "src"), str(official.parent)]
    from PointTransformerV3.model import PointTransformerV3
    from train_ptv3 import (PTv3TrainDataset, batch_fingerprint, build_model, capture_rng,
                            restore_rng, seed_all)

    device = torch.device("cuda")
    root = project / cfg["data"]["processed_dir"]
    files = sorted((root / "train").glob("*.npz"))
    dataset = PTv3TrainDataset(files[:4], 1024, 2, int(cfg["seed"]), cfg["augment"])
    dataset.set_epoch(0)
    sample_a = [dataset[i] for i in (0, 1)]
    sample_b = [dataset[i] for i in (2, 3)]
    stack = lambda rows, key: torch.stack([row[key] for row in rows]).to(device)
    x1, y1 = stack(sample_a, "points"), stack(sample_a, "labels")
    x2, y2 = stack(sample_b, "points"), stack(sample_b, "labels")

    seed_all(987654)
    before = capture_rng()
    baseline = build_model(cfg, "baseline", PointTransformerV3, Path(args.common_init), device)
    refine = build_model(cfg, "refine", PointTransformerV3, Path(args.common_init), device)
    after = capture_rng()
    rng_equal = (
        before["python"] == after["python"] and
        np.array_equal(before["numpy"][1], after["numpy"][1]) and
        torch.equal(before["torch_cpu"], after["torch_cpu"]) and
        all(torch.equal(a, b) for a, b in zip(before["torch_cuda"], after["torch_cuda"]))
    )
    baseline.eval(); refine.eval()
    evaluation_state = capture_rng()
    with torch.no_grad():
        baseline_logits = baseline(x1)
        restore_rng(evaluation_state)
        refine_logits = refine(x1)
        zero_init_difference = float((baseline_logits - refine_logits).abs().max())

    refine.train()
    refine.zero_grad(set_to_none=True)
    F.cross_entropy(refine(x1), y1, ignore_index=-1).backward()
    first_w2_grad = float(refine.refine.w2.weight.grad.abs().sum())
    first_w1_grad = float(refine.refine.w1.weight.grad.abs().sum())
    with torch.no_grad():
        refine.refine.w2.weight.add_(refine.refine.w2.weight.grad, alpha=-1e-3)
    refine.zero_grad(set_to_none=True)
    F.cross_entropy(refine(x2), y2, ignore_index=-1).backward()
    second_w1_grad = float(refine.refine.w1.weight.grad.abs().sum())
    neighbor_diagnostics = deepcopy(refine.refine.last_diagnostics)
    del refine
    torch.cuda.empty_cache()

    continuous = build_model(cfg, "baseline", PointTransformerV3, Path(args.common_init), device)
    opt_cont = torch.optim.AdamW(continuous.parameters(), lr=1e-3, weight_decay=0.05)
    seed_all(777)
    first_loss = train_step(continuous, opt_cont, x1, y1)
    saved = {"model": deepcopy(continuous.state_dict()), "optimizer": deepcopy(opt_cont.state_dict()),
             "rng": capture_rng()}
    continuous_loss = train_step(continuous, opt_cont, x2, y2)
    continuous_state = deepcopy(continuous.state_dict())
    del continuous, opt_cont
    torch.cuda.empty_cache()

    resumed = build_model(cfg, "baseline", PointTransformerV3, Path(args.common_init), device)
    opt_resumed = torch.optim.AdamW(resumed.parameters(), lr=1e-3, weight_decay=0.05)
    resumed.load_state_dict(saved["model"], strict=True)
    opt_resumed.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    resumed_loss = train_step(resumed, opt_resumed, x2, y2)
    resume_parameter_max_abs = max_state_difference(continuous_state, resumed.state_dict())

    # Epoch-boundary sampler state: completing an iterator consumes one complete
    # permutation. Restoring that state must reproduce the next epoch's first batch.
    small = PTv3TrainDataset(files[:4], 64, 2, int(cfg["seed"]), cfg["augment"])
    generator = torch.Generator().manual_seed(20045)
    loader = DataLoader(small, batch_size=2, shuffle=True, num_workers=0, generator=generator)
    small.set_epoch(0)
    list(loader)
    epoch_boundary_state = generator.get_state().clone()
    small.set_epoch(1)
    expected_batch = next(iter(loader))
    restored_generator = torch.Generator(); restored_generator.set_state(epoch_boundary_state)
    restored_loader = DataLoader(small, batch_size=2, shuffle=True, num_workers=0,
                                 generator=restored_generator)
    actual_batch = next(iter(restored_loader))
    expected_fingerprint = batch_fingerprint(expected_batch)
    actual_fingerprint = batch_fingerprint(actual_batch)

    result = {
        "global_rng_unchanged_by_model_initialization": rng_equal,
        "zero_initialized_refine_logit_max_abs": zero_init_difference,
        "refine_first_backward_w2_grad_abs_sum": first_w2_grad,
        "refine_first_backward_w1_grad_abs_sum": first_w1_grad,
        "refine_second_backward_w1_grad_abs_sum": second_w1_grad,
        "refine_neighbor_diagnostics": neighbor_diagnostics,
        "resume_first_loss": first_loss,
        "continuous_next_loss": continuous_loss,
        "resumed_next_loss": resumed_loss,
        "resume_loss_abs_difference": abs(continuous_loss - resumed_loss),
        "resume_parameter_max_abs_difference": resume_parameter_max_abs,
        "next_epoch_batch_fingerprint_expected": expected_fingerprint,
        "next_epoch_batch_fingerprint_resumed": actual_fingerprint,
        "next_epoch_batch_equal": expected_fingerprint == actual_fingerprint,
    }
    checks = [rng_equal, zero_init_difference == 0.0, first_w2_grad > 0,
              first_w1_grad == 0.0, second_w1_grad > 0,
              abs(continuous_loss - resumed_loss) <= 1e-7,
              resume_parameter_max_abs <= 1e-7, expected_fingerprint == actual_fingerprint]
    result["passed"] = all(checks)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
