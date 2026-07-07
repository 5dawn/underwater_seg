from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from dataset import infer_input_channels
from device import select_device
from metrics import confusion_matrix, metrics_from_confusion
from model_factory import build_model


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_indices(n: int, num_points: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n, num_points, replace=n < num_points)


def forward_logits(model: torch.nn.Module, features_np: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    features = torch.from_numpy(features_np[indices].astype(np.float32)).transpose(0, 1).unsqueeze(0).to(device)
    return model(features)


def add_vote(vote_logits: np.ndarray, vote_counts: np.ndarray, indices: np.ndarray, logits: torch.Tensor) -> None:
    logits_np = logits.squeeze(0).transpose(0, 1).detach().cpu().numpy()
    np.add.at(vote_logits, indices, logits_np)
    np.add.at(vote_counts, indices, 1)


def evaluate_file(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    num_classes: int,
    num_points: int,
    num_votes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int, dict[str, float | list[list[int]] | str]]:
    data = np.load(path)
    features_np = data["features"].astype(np.float32)
    labels_np = data["labels"].astype(np.int64)
    n = len(labels_np)
    if n == 0:
        raise ValueError(f"Empty sample: {path}")

    vote_logits = np.zeros((n, num_classes), dtype=np.float64)
    vote_counts = np.zeros(n, dtype=np.int64)
    total_loss = 0.0
    loss_batches = 0

    with torch.no_grad():
        for _ in range(num_votes):
            indices = sample_indices(n, num_points, rng)
            logits = forward_logits(model, features_np, indices, device)
            labels = torch.from_numpy(labels_np[indices]).unsqueeze(0).to(device)
            if (labels != -1).any():
                total_loss += float(F.cross_entropy(logits, labels, ignore_index=-1).item())
                loss_batches += 1
            add_vote(vote_logits, vote_counts, indices, logits)

        uncovered = np.flatnonzero(vote_counts == 0)
        for start in range(0, len(uncovered), num_points):
            indices = uncovered[start : start + num_points]
            if len(indices) == 0:
                continue
            logits = forward_logits(model, features_np, indices, device)
            labels = torch.from_numpy(labels_np[indices]).unsqueeze(0).to(device)
            if (labels != -1).any():
                total_loss += float(F.cross_entropy(logits, labels, ignore_index=-1).item())
                loss_batches += 1
            add_vote(vote_logits, vote_counts, indices, logits)

    mean_logits = vote_logits / np.maximum(vote_counts[:, None], 1)
    pred = mean_logits.argmax(axis=1).astype(np.int64)
    cm = confusion_matrix(pred, labels_np, num_classes)
    metrics = metrics_from_confusion(cm)
    metrics["loss"] = total_loss / max(loss_batches, 1)
    metrics["file"] = path.name
    metrics["confusion_matrix"] = cm.tolist()
    return cm, total_loss, loss_batches, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PointNet checkpoint.")
    parser.add_argument("--config", default="configs/pointnet.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/pointnet_best.pth")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-votes", type=int, default=10)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    processed_dir = cfg["data"]["processed_dir"]
    if "input_channels" in checkpoint:
        input_channels = int(checkpoint["input_channels"])
    else:
        input_channels = infer_input_channels(processed_dir, split=args.split)
    num_classes = int(checkpoint.get("num_classes", cfg["model"]["num_classes"]))
    model_cfg = dict(checkpoint.get("config", cfg))
    model_cfg["model"] = dict(model_cfg.get("model", cfg.get("model", {})))
    model_cfg["model"]["name"] = checkpoint.get("model_name", model_cfg["model"].get("name", "pointnet"))
    model = build_model(model_cfg, input_channels=input_channels, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    files = sorted((Path(processed_dir) / args.split).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {Path(processed_dir) / args.split}")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    loss_batches = 0
    per_file_metrics = []
    rng = np.random.default_rng(cfg.get("seed", 42))
    num_votes = max(int(args.num_votes), 1)
    for path in tqdm(files, desc=f"Eval {args.split}"):
        file_cm, file_loss, file_loss_batches, file_metrics = evaluate_file(
            model=model,
            path=path,
            device=device,
            num_classes=num_classes,
            num_points=int(cfg["data"]["num_points"]),
            num_votes=num_votes,
            rng=rng,
        )
        cm += file_cm
        total_loss += file_loss
        loss_batches += file_loss_batches
        per_file_metrics.append(file_metrics)

    metrics = metrics_from_confusion(cm)
    metrics["loss"] = total_loss / max(loss_batches, 1)
    metrics["split"] = args.split
    metrics["num_votes"] = num_votes
    metrics["checkpoint"] = args.checkpoint
    metrics["confusion_matrix"] = cm.tolist()
    metrics["per_file"] = per_file_metrics

    text = json.dumps(metrics, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
