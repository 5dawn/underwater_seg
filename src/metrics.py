from __future__ import annotations

import numpy as np


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int = -1) -> np.ndarray:
    """Build a confusion matrix while ignoring uncertain labels."""
    pred = pred.reshape(-1).astype(np.int64)
    target = target.reshape(-1).astype(np.int64)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    encoded = num_classes * target[valid] + pred[valid]
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom_iou = tp + fp + fn

    iou = np.divide(tp, denom_iou, out=np.zeros_like(tp), where=denom_iou > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    total = cm.sum()
    result = {
        "accuracy": float(tp.sum() / total) if total > 0 else 0.0,
        "mIoU": float(iou.mean()) if len(iou) else 0.0,
    }
    for idx in range(len(iou)):
        result[f"class_{idx}_iou"] = float(iou[idx])
        result[f"class_{idx}_precision"] = float(precision[idx])
        result[f"class_{idx}_recall"] = float(recall[idx])
        result[f"class_{idx}_f1"] = float(f1[idx])
    if len(iou) > 1:
        result["submarine_iou"] = result["class_1_iou"]
        result["submarine_precision"] = result["class_1_precision"]
        result["submarine_recall"] = result["class_1_recall"]
        result["submarine_f1"] = result["class_1_f1"]
    return result
