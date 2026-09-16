from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-parent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--points", type=int, default=4096)
    args = parser.parse_args()
    sys.path.insert(0, args.official_parent)
    from PointTransformerV3.model import PointTransformerV3

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official PTv3 smoke")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    model_config = {
        "in_channels": 3,
        "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
        "stride": (2, 2, 2, 2),
        "enc_depths": (2, 2, 2, 6, 2),
        "enc_channels": (32, 64, 128, 256, 512),
        "enc_num_head": (2, 4, 8, 16, 32),
        "enc_patch_size": (128, 128, 128, 128, 128),
        "dec_depths": (2, 2, 2, 2),
        "dec_channels": (64, 64, 128, 256),
        "dec_num_head": (4, 4, 8, 16),
        "dec_patch_size": (128, 128, 128, 128),
        "enable_flash": False,
        "enable_rpe": False,
        "upcast_attention": True,
        "upcast_softmax": True,
        "cls_mode": False,
    }
    model = PointTransformerV3(**model_config).to(device).train()
    parameter_count = sum(p.numel() for p in model.parameters())
    n = args.batch_size * args.points
    # Unique-ish integer voxels with normalized XYZ as the only input feature.
    grid = torch.randint(0, 256, (n, 3), device=device, dtype=torch.int32)
    coord = grid.float() / 255.0
    batch = torch.arange(args.batch_size, device=device).repeat_interleave(args.points)
    offset = torch.arange(1, args.batch_size + 1, device=device) * args.points
    payload = {"coord": coord, "grid_coord": grid, "feat": coord.clone(), "batch": batch, "offset": offset}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    torch.cuda.reset_peak_memory_stats()
    synchronize()
    start = time.perf_counter()
    output = model(payload)
    synchronize()
    forward_s = time.perf_counter() - start
    feature = output.feat
    if feature.shape != (n, 64):
        raise AssertionError(f"unexpected decoder feature shape: {tuple(feature.shape)}")
    loss = feature.float().square().mean()
    start = time.perf_counter()
    loss.backward()
    synchronize()
    backward_s = time.perf_counter() - start
    finite_gradients = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    start = time.perf_counter()
    optimizer.step()
    synchronize()
    step_s = time.perf_counter() - start
    report = {
        "model": "official PointTransformerV3 detached implementation",
        "input_feature": "normalized XYZ only",
        "batch_size": args.batch_size,
        "points_per_sample": args.points,
        "total_points": n,
        "output_feature_shape": list(feature.shape),
        "decoder_feature_dim": int(feature.shape[1]),
        "parameter_count": parameter_count,
        "loss": float(loss.detach()),
        "finite_output": bool(torch.isfinite(feature).all()),
        "finite_gradients": bool(finite_gradients),
        "forward_seconds": forward_s,
        "backward_seconds": backward_s,
        "optimizer_step_seconds": step_s,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model_config": {key: list(value) if isinstance(value, tuple) else value for key, value in model_config.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
