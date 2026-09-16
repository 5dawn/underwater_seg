from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from metrics import confusion_matrix, metrics_from_confusion


def _capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _binary_metrics(cm: np.ndarray) -> dict:
    result = metrics_from_confusion(cm)
    result.setdefault("foreground_iou", result["class_1_iou"])
    result.setdefault("foreground_precision", result["class_1_precision"])
    result.setdefault("foreground_recall", result["class_1_recall"])
    result.setdefault("foreground_f1", result["class_1_f1"])
    result["TN"], result["FP"] = int(cm[0, 0]), int(cm[0, 1])
    result["FN"], result["TP"] = int(cm[1, 0]), int(cm[1, 1])
    result["background_fpr"] = float(cm[0, 1] / max(cm[0].sum(), 1))
    return result


def _morton_codes(points: np.ndarray, vote: int) -> np.ndarray:
    permutations = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0))
    xyz = points[:, permutations[vote % len(permutations)]].astype(np.float64)
    if vote & 1:
        xyz[:, 0] *= -1
    if vote & 2:
        xyz[:, 1] *= -1
    low, high = xyz.min(0), xyz.max(0)
    scaled = (xyz - low) / np.maximum(high - low, 1e-12)
    grid = np.minimum((scaled * 1023).astype(np.uint64), 1023)
    code = np.zeros(len(grid), dtype=np.uint64)
    for bit in range(10):
        code |= ((grid[:, 0] >> bit) & 1) << (3 * bit)
        code |= ((grid[:, 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((grid[:, 2] >> bit) & 1) << (3 * bit + 2)
    return code


def spatial_cover(points: np.ndarray, block_points: int, vote: int) -> list[np.ndarray]:
    order = np.argsort(_morton_codes(points, vote), kind="stable")
    return [order[start : start + block_points] for start in range(0, len(order), block_points)]


def evaluate_ptv3_full_cloud(
    model: torch.nn.Module,
    files: list[Path],
    device: torch.device,
    out_dir: Path,
    votes: int = 5,
    block_points: int = 4096,
    batch_size: int = 4,
    eval_seed: int = 1001,
) -> dict:
    if votes != 5:
        raise ValueError("the frozen protocol requires exactly 5 complete-cover votes")
    training = model.training
    saved_rng = _capture_rng()
    random.seed(eval_seed); np.random.seed(eval_seed); torch.manual_seed(eval_seed); torch.cuda.manual_seed_all(eval_seed)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    overall = np.zeros((2, 2), dtype=np.int64)
    scene_cms: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((2, 2), dtype=np.int64))
    per_file = []
    try:
        with torch.no_grad():
            for path in files:
                with np.load(path) as data:
                    points = data["points"].astype(np.float32)
                    labels = data["labels"].astype(np.int64)
                logits_sum = np.zeros((len(points), 2), dtype=np.float64)
                counts = np.zeros(len(points), dtype=np.int16)
                tasks: list[tuple[np.ndarray, int]] = []
                for vote in range(votes):
                    for indices in spatial_cover(points, block_points, vote):
                        valid = len(indices)
                        if valid < block_points:
                            pad = np.resize(indices, block_points - valid)
                            indices = np.concatenate((indices, pad))
                        tasks.append((indices, valid))
                for start in range(0, len(tasks), batch_size):
                    group = tasks[start : start + batch_size]
                    batch = torch.from_numpy(np.stack([points[idx] for idx, _ in group])).transpose(1, 2).to(device)
                    output = model(batch).transpose(1, 2).float().cpu().numpy()
                    for row, (indices, valid) in enumerate(group):
                        original = indices[:valid]
                        logits_sum[original] += output[row, :valid]
                        counts[original] += 1
                if counts.min() < votes or counts.max() != votes:
                    raise AssertionError(f"incomplete coverage for {path.name}: {counts.min()}..{counts.max()}")
                prediction = (logits_sum / counts[:, None]).argmax(1)
                cm = confusion_matrix(prediction, labels, 2)
                metrics = _binary_metrics(cm)
                metrics.update(file=path.name, coverage_min=int(counts.min()), coverage_max=int(counts.max()), confusion_matrix=cm.tolist())
                np.save(out_dir / f"{path.stem}_vote_counts.npy", counts)
                per_file.append(metrics)
                overall += cm
                scene_cms[path.name.split("_", 1)[0]] += cm
        result = _binary_metrics(overall)
        result["confusion_matrix"] = overall.tolist()
        result["votes"] = votes
        result["eval_seed"] = eval_seed
        result["raw_argmax"] = True
        result["per_file"] = per_file
        result["per_scene"] = {}
        for scene, cm in sorted(scene_cms.items()):
            item = _binary_metrics(cm); item["confusion_matrix"] = cm.tolist()
            result["per_scene"][scene] = item
        return result
    finally:
        _restore_rng(saved_rng)
        model.train(training)
