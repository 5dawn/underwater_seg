from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UnderwaterSegDataset, compute_class_weights, infer_input_channels
from metrics import confusion_matrix, metrics_from_confusion
from model_pointnet import PointNetSeg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_loader(cfg: dict, split: str, train: bool) -> DataLoader:
    dataset = UnderwaterSegDataset(
        root=cfg["data"]["processed_dir"],
        split=split,
        num_points=cfg["data"]["num_points"],
        augment=train and cfg.get("augment", {}).get("enabled", False),
        augment_cfg=cfg.get("augment", {}),
        sample_mode=cfg["data"].get("sample_mode", "random") if train else "random",
        foreground_ratio=float(cfg["data"].get("foreground_ratio", 0.25)),
    )
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=train and len(dataset) >= cfg["train"]["batch_size"],
    )


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            logits = model(features)
            if not (labels != -1).any():
                continue
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            total_loss += float(loss.item())
            batches += 1
            pred = logits.argmax(dim=1).cpu().numpy()
            target = labels.cpu().numpy()
            cm += confusion_matrix(pred, target, num_classes)
    result = metrics_from_confusion(cm)
    result["loss"] = total_loss / max(batches, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PointNet for underwater target segmentation.")
    parser.add_argument("--config", default="configs/pointnet.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processed_dir = cfg["data"]["processed_dir"]
    input_channels_cfg = cfg["model"].get("input_channels", "auto")
    input_channels = infer_input_channels(processed_dir) if input_channels_cfg == "auto" else int(input_channels_cfg)
    num_classes = int(cfg["model"]["num_classes"])

    model = PointNetSeg(
        input_channels=input_channels,
        num_classes=num_classes,
        dropout=float(cfg["model"].get("dropout", 0.3)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    start_epoch = 1
    best_score = -1.0
    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -1.0))

    class_weights = None
    if cfg["train"].get("use_class_weights", True):
        class_weights = compute_class_weights(processed_dir, num_classes=num_classes).to(device)
        print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    train_loader = make_loader(cfg, "train", train=True)
    val_loader = make_loader(cfg, "val", train=False)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(cfg["train"]["log_dir"])
    except Exception:
        writer = None

    best_metric = cfg["train"].get("best_metric", "submarine_iou")
    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            logits = model(features)
            if not (labels != -1).any():
                continue
            loss = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=-1)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        score = float(val_metrics.get(best_metric, val_metrics["mIoU"]))

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)

        is_best = score > best_score
        if is_best:
            best_score = score
            best_path = checkpoint_dir / "pointnet_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_score": best_score,
                    "config": cfg,
                    "input_channels": input_channels,
                    "num_classes": num_classes,
                },
                best_path,
            )

        last_path = checkpoint_dir / "pointnet_last.pth"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": best_score,
                "config": cfg,
                "input_channels": input_channels,
                "num_classes": num_classes,
            },
            last_path,
        )

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val": val_metrics,
                    "best_score": best_score,
                    "saved_best": is_best,
                },
                indent=2,
            )
        )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
