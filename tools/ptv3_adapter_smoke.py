from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sync() -> None:
    torch.cuda.synchronize()


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def make_backbone(cls):
    return cls(
        in_channels=3,
        enc_patch_size=(128, 128, 128, 128, 128),
        dec_patch_size=(128, 128, 128, 128),
        enable_flash=False,
        enable_rpe=False,
        upcast_attention=True,
        upcast_softmax=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--official-parent", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.project) / "src"))
    sys.path.insert(0, args.official_parent)
    from model_ptv3 import OfficialPTv3Seg
    from ptv3_protocol import voxelize_xyz
    from PointTransformerV3.model import PointTransformerV3

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda")
    batch_size, num_points = 4, 4096
    # Normalized XYZ only; clustered coordinates exercise actual voxel merging.
    inputs = (torch.randn(batch_size, 3, num_points, device=device) * 0.15).clamp(-1, 1)
    labels = torch.randint(0, 2, (batch_size, num_points), device=device)
    labels[:, ::31] = -1
    voxel_counts = [len(voxelize_xyz(x.T, 0.005).coord) for x in inputs]
    model = OfficialPTv3Seg(make_backbone(PointTransformerV3), 64, 2, 0.005).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    torch.cuda.reset_peak_memory_stats()
    sync(); t = time.perf_counter(); logits = model(inputs); sync(); forward_s = time.perf_counter() - t
    assert logits.shape == (batch_size, 2, num_points)
    loss = F.cross_entropy(logits, labels, ignore_index=-1)
    t = time.perf_counter(); loss.backward(); sync(); backward_s = time.perf_counter() - t
    finite_gradients = all(x.grad is None or torch.isfinite(x.grad).all() for x in model.parameters())
    t = time.perf_counter(); optimizer.step(); scheduler.step(); sync(); step_s = time.perf_counter() - t
    model.eval()
    eval_rng = capture_rng()
    with torch.no_grad():
        reference = model(inputs)
    checkpoint_path = Path(args.out).with_name("adapter_checkpoint_smoke.pth")
    checkpoint = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": 1,
        "best_metric": 0.0, "config": {"voxel_size": 0.005, "refine": False},
        "rng_state": capture_rng(),
    }
    torch.save(checkpoint, checkpoint_path)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    restored = OfficialPTv3Seg(make_backbone(PointTransformerV3), 64, 2, 0.005).to(device).eval()
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored.load_state_dict(loaded["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=0.05)
    restored_optimizer.load_state_dict(loaded["optimizer"])
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=10)
    restored_scheduler.load_state_dict(loaded["scheduler"])
    restore_rng(eval_rng)
    with torch.no_grad():
        reloaded = restored(inputs)
    report = {
        "input_shape": list(inputs.shape), "logits_shape": list(logits.shape),
        "voxel_counts": voxel_counts, "voxel_size": 0.005,
        "ignore_points": int((labels == -1).sum()), "loss": float(loss.detach()),
        "finite_logits": bool(torch.isfinite(logits).all()),
        "finite_gradients": bool(finite_gradients),
        "parameter_count": sum(p.numel() for p in restored.parameters()),
        "forward_seconds": forward_s, "backward_seconds": backward_s,
        "optimizer_step_seconds": step_s,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "strict_checkpoint_reload": True,
        "reload_max_abs_logit_difference": float((reference - reloaded).abs().max()),
        "eval_rng_restored_before_comparison": True,
        "checkpoint_fields": sorted(loaded),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
