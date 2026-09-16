from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def backbone(cls):
    return cls(
        in_channels=3,
        enc_patch_size=(128,) * 5,
        dec_patch_size=(128,) * 4,
        enable_flash=False,
        enable_rpe=False,
        upcast_attention=True,
        upcast_softmax=True,
    )


def fixed_batch(files: list[Path], n: int, seed: int, device: torch.device):
    points, labels = [], []
    for i, path in enumerate(files):
        with np.load(path) as data:
            xyz = data["points"].astype(np.float32)
            target = data["labels"].astype(np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        index = rng.choice(len(xyz), n, replace=len(xyz) < n)
        points.append(torch.from_numpy(xyz[index]))
        labels.append(torch.from_numpy(target[index]))
    return torch.stack(points).transpose(1, 2).to(device), torch.stack(labels).to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--official-parent", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    project = Path(args.project)
    sys.path[:0] = [str(project / "src"), args.official_parent]
    from model_ptv3 import OfficialPTv3Seg
    from ptv3_protocol import voxelize_xyz
    from PointTransformerV3.model import PointTransformerV3

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    files = sorted((project / "data/processed_label2/train").glob("*.npz"))
    if len(files) != 45:
        raise RuntimeError(f"expected 45 train files, got {len(files)}")

    inputs, labels = fixed_batch(files[:4], 4096, 42, device)
    voxel_counts = [len(voxelize_xyz(x.T, 0.005).coord) for x in inputs]
    model = OfficialPTv3Seg(backbone(PointTransformerV3), 64, 2, 0.005).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t = time.perf_counter(); logits = model(inputs); torch.cuda.synchronize(); fwd = time.perf_counter() - t
    loss = F.cross_entropy(logits, labels, ignore_index=-1)
    torch.cuda.synchronize(); t = time.perf_counter(); loss.backward(); torch.cuda.synchronize(); bwd = time.perf_counter() - t
    gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    t = time.perf_counter(); opt.step(); torch.cuda.synchronize(); step = time.perf_counter() - t
    real_step = {
        "files": [x.name for x in files[:4]], "input_shape": list(inputs.shape),
        "logits_shape": list(logits.shape), "voxel_counts": voxel_counts,
        "unique_voxel_ratios": [x / 4096 for x in voxel_counts],
        "ignore_points": int((labels == -1).sum()), "loss": float(loss.detach()),
        "finite_logits": bool(torch.isfinite(logits).all()), "finite_gradients": bool(gradients_finite),
        "forward_seconds": fwd, "backward_seconds": bwd, "optimizer_step_seconds": step,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    del model, opt, inputs, labels, logits, loss
    torch.cuda.empty_cache()

    fit_x, fit_y = fixed_batch(files[:2], 1024, 4242, device)
    fit_model = OfficialPTv3Seg(backbone(PointTransformerV3), 64, 2, 0.005).to(device).train()
    fit_opt = torch.optim.AdamW(fit_model.parameters(), lr=1e-3, weight_decay=0.05)
    losses = []
    for _ in range(10):
        fit_opt.zero_grad(set_to_none=True)
        fit_logits = fit_model(fit_x)
        fit_loss = F.cross_entropy(fit_logits, fit_y, ignore_index=-1)
        fit_loss.backward()
        fit_opt.step()
        losses.append(float(fit_loss.detach()))
    sanity_fit = {
        "files": [x.name for x in files[:2]], "steps": len(losses), "losses": losses,
        "initial_loss": losses[0], "final_loss": losses[-1],
        "relative_decrease": (losses[0] - losses[-1]) / losses[0],
        "loss_decreased": losses[-1] < losses[0] * 0.8,
    }
    report = {"real_batch_step": real_step, "sanity_fit": sanity_fit}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
