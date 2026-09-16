from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [x.clone() for x in torch.cuda.get_rng_state_all()],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda"]])


def indexed_rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, index, 0x50545633]))


class PTv3TrainDataset(Dataset):
    def __init__(self, files: list[Path], num_points: int, samples_per_cloud: int,
                 seed: int, augment: dict[str, Any]):
        self.files = files
        self.num_points = num_points
        self.samples_per_cloud = samples_per_cloud
        self.seed = seed
        self.augment = augment
        self.epoch = 0
        self.cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.files) * self.samples_per_cloud

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index % len(self.files)]
        if path not in self.cache:
            with np.load(path) as data:
                self.cache[path] = (data["points"].astype(np.float32), data["labels"].astype(np.int64))
        source_xyz, source_labels = self.cache[path]
        rng = indexed_rng(self.seed, self.epoch, index)
        selected = rng.choice(len(source_xyz), self.num_points, replace=len(source_xyz) < self.num_points)
        xyz = source_xyz[selected].copy()
        labels = source_labels[selected].copy()
        theta = rng.uniform(0.0, 2.0 * np.pi) if self.augment.get("rotate_z", True) else 0.0
        c, s = np.cos(theta), np.sin(theta)
        xyz = xyz @ np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32).T
        scale = float(rng.uniform(self.augment.get("scale_min", 1.0), self.augment.get("scale_max", 1.0)))
        xyz *= scale
        jitter_std = float(self.augment.get("jitter_std", 0.0))
        if jitter_std:
            clip = float(self.augment.get("jitter_clip", 0.02))
            xyz += np.clip(rng.normal(0, jitter_std, xyz.shape), -clip, clip).astype(np.float32)
        dropout = float(self.augment.get("point_dropout", 0.0))
        if dropout:
            mask = rng.random(len(xyz)) < dropout
            keep = np.flatnonzero(~mask)
            if mask.any() and len(keep):
                replacement = rng.choice(keep, int(mask.sum()), replace=True)
                xyz[mask] = xyz[replacement]
                labels[mask] = labels[replacement]
                selected[mask] = selected[replacement]
        return {
            "points": torch.from_numpy(xyz).transpose(0, 1),
            "labels": torch.from_numpy(labels),
            "sample_indices": torch.from_numpy(selected.astype(np.int64, copy=False)),
            "path": str(path),
            "scale": scale,
        }


def backbone_factory(point_transformer_cls):
    return point_transformer_cls(
        in_channels=3,
        enc_patch_size=(128,) * 5,
        dec_patch_size=(128,) * 4,
        enable_flash=False,
        enable_rpe=False,
        upcast_attention=True,
        upcast_softmax=True,
    )


def build_model(cfg: dict, variant: str, point_transformer_cls, common_path: Path, device: torch.device):
    from model_ptv3 import OfficialPTv3Seg, isolated_rng
    from ptv3_protocol import UnderwaterLocalFeatureRefine

    seed = int(cfg["seed"])
    model_cfg = cfg["model"]
    with isolated_rng(seed):
        model = OfficialPTv3Seg(
            backbone_factory(point_transformer_cls),
            int(model_cfg["decoder_channels"]),
            int(model_cfg["num_classes"]),
            float(cfg["voxel"]["selected_size"]),
        )
    if not common_path.exists():
        common_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": deepcopy(model.state_dict()), "seed": seed}, common_path)
    common = torch.load(common_path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(common, strict=True)
    if variant == "refine":
        with isolated_rng(seed + 1_000_003):
            model.refine = UnderwaterLocalFeatureRefine(
                int(model_cfg["decoder_channels"]),
                float(cfg["radius"]["selected_p95_d16"]),
                int(cfg["radius"]["k"]),
                int(cfg["radius"]["query_chunk_size"]),
                bool(model_cfg.get("refine_zero_D", False)),
                bool(model_cfg.get("refine_zero_G", False)),
            )
    return model.to(device)


def lr_at_step(base_lr: float, step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps <= 1 or step < warmup_steps:
        fraction = step / max(warmup_steps - 1, 1)
        factor = 0.1 + 0.9 * fraction
    else:
        fraction = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
        factor = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(fraction, 1.0)))
    return base_lr * factor


def batch_fingerprint(batch: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(batch["points"].numpy().tobytes())
    digest.update(batch["labels"].numpy().tobytes())
    digest.update(batch["sample_indices"].numpy().tobytes())
    for path in batch["path"]:
        digest.update(path.encode("utf-8"))
    return digest.hexdigest()


def save_checkpoint(path: Path, model, optimizer, next_epoch: int, global_step: int,
                    loader_generator: torch.Generator, history: list[dict], cfg: dict, variant: str) -> None:
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "next_epoch": next_epoch, "global_step": global_step,
        "rng": capture_rng(), "loader_generator": loader_generator.get_state(),
        "history": history, "config": cfg, "variant": variant,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("baseline", "refine"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--common-init", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--eval-only")
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    official = Path(cfg["official_source"]["local_path"])
    sys.path[:0] = [str(project / "src"), str(official.parent)]
    from PointTransformerV3.model import PointTransformerV3
    from ptv3_full_eval import evaluate_ptv3_full_cloud

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(cfg["data"]["processed_dir"])
    if not data_root.is_absolute():
        data_root = project / data_root
    train_files = sorted((data_root / "train").glob("*.npz"))
    val_files = sorted((data_root / "val").glob("*.npz"))
    expected = cfg["data"]["expected_counts"]
    if len(train_files) != int(expected["train"]) or len(val_files) != int(expected["val"]):
        raise RuntimeError(f"split mismatch: train={len(train_files)}, val={len(val_files)}")
    manifest = {
        "train": [{"file": p.name, "sha256": sha256(p)} for p in train_files],
        "val": [{"file": p.name, "sha256": sha256(p)} for p in val_files],
        "historical_test_used": False,
    }
    json_dump(output / "data_manifest.json", manifest)
    effective = deepcopy(cfg)
    effective["run"] = {"variant": args.variant, "output": str(output.resolve())}
    json_dump(output / "effective_config.json", effective)

    device = torch.device("cuda")
    seed_all(int(cfg["seed"]))
    common_path = Path(args.common_init)
    model = build_model(cfg, args.variant, PointTransformerV3, common_path, device)
    parameter_counts = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "refine": sum(p.numel() for p in model.refine.parameters()) if model.refine is not None else 0,
    }
    json_dump(output / "parameter_counts.json", parameter_counts)

    if args.eval_only:
        checkpoint = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        result = evaluate_ptv3_full_cloud(
            model, val_files, device, output / "independent_vote_counts", save_predictions=True
        )
        json_dump(output / "independent_evaluation.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    train_cfg = cfg["train"]
    epochs = int(args.max_epochs or train_cfg["epochs"])
    dataset = PTv3TrainDataset(
        train_files, int(cfg["data"]["num_points"]), int(train_cfg["samples_per_cloud"]),
        int(cfg["seed"]), cfg["augment"],
    )
    loader_generator = torch.Generator().manual_seed(int(cfg["seed"]) + 20_003)
    loader = DataLoader(dataset, batch_size=int(train_cfg["batch_size"]), shuffle=True,
                        num_workers=0, drop_last=False, generator=loader_generator)
    if len(loader) != int(train_cfg["updates_per_epoch"]):
        raise RuntimeError(f"expected {train_cfg['updates_per_epoch']} updates, got {len(loader)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    start_epoch = 0
    global_step = 0
    history: list[dict] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["variant"] != args.variant:
            raise RuntimeError("resume variant mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["next_epoch"])
        global_step = int(checkpoint["global_step"])
        history = checkpoint["history"]
        restore_rng(checkpoint["rng"])
        loader_generator.set_state(checkpoint["loader_generator"].cpu())

    base_lr = float(train_cfg["learning_rate"])
    warmup_steps = len(loader)
    total_steps = int(train_cfg["epochs"]) * len(loader)
    best_iou = max((x["validation"]["foreground_iou"] for x in history), default=-1.0)
    audit: list[dict] = []
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        model.train()
        losses = []
        scales = []
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            if batch_index < 2:
                audit.append({"epoch": epoch, "batch": batch_index, "fingerprint": batch_fingerprint(batch)})
            lr = lr_at_step(base_lr, global_step, warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            points = batch["points"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            scales.extend(float(x) for x in batch["scale"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(points)
            loss = F.cross_entropy(logits, labels, ignore_index=int(cfg["data"]["ignore_index"]))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"non-finite gradient at step {global_step}")
            optimizer.step()
            losses.append(float(loss.detach()))
            global_step += 1
        train_seconds = time.perf_counter() - started
        validation = evaluate_ptv3_full_cloud(
            model, val_files, device, output / f"epoch_{epoch + 1:02d}_vote_counts",
            votes=int(cfg["evaluation"]["votes"]), block_points=int(cfg["data"]["num_points"]),
            batch_size=int(train_cfg["batch_size"]), eval_seed=int(cfg["eval_seed"]),
        )
        row = {
            "epoch": epoch + 1, "global_step": global_step,
            "train_loss_mean": float(np.mean(losses)), "train_loss_last": losses[-1],
            "lr_last": optimizer.param_groups[0]["lr"], "train_seconds": train_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "augmentation_scale_min": min(scales), "augmentation_scale_max": max(scales),
            "fixed_radius": float(cfg["radius"]["selected_p95_d16"]),
            "equivalent_radius_over_scale_min": float(cfg["radius"]["selected_p95_d16"]) / max(scales),
            "equivalent_radius_over_scale_max": float(cfg["radius"]["selected_p95_d16"]) / min(scales),
            "validation": validation,
        }
        if model.refine is not None:
            row["refine_diagnostics_last_batch"] = model.refine.last_diagnostics
        history.append(row)
        json_dump(output / "history.json", history)
        json_dump(output / "batch_audit.json", audit)
        save_checkpoint(output / "last.pth", model, optimizer, epoch + 1, global_step,
                        loader_generator, history, cfg, args.variant)
        current_iou = validation["foreground_iou"]
        if current_iou > best_iou:
            best_iou = current_iou
            save_checkpoint(output / "best.pth", model, optimizer, epoch + 1, global_step,
                            loader_generator, history, cfg, args.variant)
            json_dump(output / "best_validation.json", validation)
        print(json.dumps({"epoch": epoch + 1, "loss": row["train_loss_mean"],
                          "foreground_iou": current_iou, "background_fpr": validation["background_fpr"]}))


if __name__ == "__main__":
    main()
