from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def delta(a: dict, b: dict, key: str) -> float:
    return float(b[key] - a[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    baseline_dir = root / "baseline_independent"
    refine_dir = root / "refine_independent"
    baseline = load(baseline_dir / "independent_evaluation.json")
    refine = load(refine_dir / "independent_evaluation.json")
    base_params = load(root / "baseline" / "parameter_counts.json")
    refine_params = load(root / "refine" / "parameter_counts.json")
    base_history = load(root / "baseline" / "history.json")
    refine_history = load(root / "refine" / "history.json")

    gates = {
        "foreground_iou_delta_at_least_0.01": delta(baseline, refine, "foreground_iou") >= 0.01,
        "background_fpr_strictly_lower": refine["background_fpr"] < baseline["background_fpr"],
        "precision_not_lower": refine["foreground_precision"] >= baseline["foreground_precision"],
        "recall_delta_at_least_minus_0.03": delta(baseline, refine, "foreground_recall") >= -0.03,
        "a4noise_iou_delta_at_least_minus_0.01": delta(
            baseline["per_scene"]["a4noise"], refine["per_scene"]["a4noise"], "foreground_iou") >= -0.01,
        "a4noise_recall_delta_at_least_minus_0.03": delta(
            baseline["per_scene"]["a4noise"], refine["per_scene"]["a4noise"], "foreground_recall") >= -0.03,
    }
    comparison = {
        "baseline_best_epoch": int(np.argmax([x["validation"]["foreground_iou"] for x in base_history])) + 1,
        "refine_best_epoch": int(np.argmax([x["validation"]["foreground_iou"] for x in refine_history])) + 1,
        "overall": {}, "per_scene": {}, "parameters": {
            "baseline": base_params, "refine": refine_params,
            "delta_total": refine_params["total"] - base_params["total"],
        },
        "runtime_and_memory": {
            "baseline_train_seconds_total": sum(x["train_seconds"] for x in base_history),
            "refine_train_seconds_total": sum(x["train_seconds"] for x in refine_history),
            "baseline_peak_reserved_bytes": max(x["peak_reserved_bytes"] for x in base_history),
            "refine_peak_reserved_bytes": max(x["peak_reserved_bytes"] for x in refine_history),
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "next_stage": "ablation" if all(gates.values()) else "stop_without_ablation_or_multiseed",
    }
    keys = ("foreground_iou", "foreground_precision", "foreground_recall", "background_fpr")
    for key in keys:
        comparison["overall"][key] = {
            "baseline": baseline[key], "refine": refine[key], "delta": delta(baseline, refine, key)}
    for scene in ("a4noise", "poolbottom", "rack"):
        comparison["per_scene"][scene] = {}
        for key in keys:
            comparison["per_scene"][scene][key] = {
                "baseline": baseline["per_scene"][scene][key],
                "refine": refine["per_scene"][scene][key],
                "delta": delta(baseline["per_scene"][scene], refine["per_scene"][scene], key),
            }
    (root / "pair_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    rows = ["| Metric | Baseline | Refine | Delta |", "|---|---:|---:|---:|"]
    for key, item in comparison["overall"].items():
        rows.append(f"| {key} | {item['baseline']:.6f} | {item['refine']:.6f} | {item['delta']:+.6f} |")
    scene_rows = ["| Scene | IoU A | IoU B | ΔIoU | FPR A | FPR B | ΔFPR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for scene, item in comparison["per_scene"].items():
        iou, fpr = item["foreground_iou"], item["background_fpr"]
        scene_rows.append(f"| {scene} | {iou['baseline']:.6f} | {iou['refine']:.6f} | {iou['delta']:+.6f} | {fpr['baseline']:.6f} | {fpr['refine']:.6f} | {fpr['delta']:+.6f} |")
    gate_rows = [f"- [{'x' if value else ' '}] {name}" for name, value in gates.items()]
    report = "\n".join([
        "# PTv3 baseline / 水下局部特征细化模块：seed 42 正式配对结果", "",
        "本报告仅描述当前 45/9 数据划分与 10 epochs 短训练预算下的 validation 结果。历史 test 未参与。", "",
        "## Overall", "", *rows, "", "## Scenes", "", *scene_rows, "",
        "## Frozen gates", "", *gate_rows, "",
        f"结论：**{'通过，允许进入消融' if comparison['all_gates_passed'] else '未通过，停止消融和多 seed'}**。", "",
        f"Baseline best epoch: {comparison['baseline_best_epoch']}; Refine best epoch: {comparison['refine_best_epoch']}.",
        f"Parameters: {base_params['total']:,} → {refine_params['total']:,} ({comparison['parameters']['delta_total']:+,}).",
        f"Training seconds: {comparison['runtime_and_memory']['baseline_train_seconds_total']:.2f} → {comparison['runtime_and_memory']['refine_train_seconds_total']:.2f}.",
        f"Peak reserved GiB: {comparison['runtime_and_memory']['baseline_peak_reserved_bytes']/2**30:.3f} → {comparison['runtime_and_memory']['refine_peak_reserved_bytes']/2**30:.3f}.",
    ])
    (root / "PAIR_REPORT_ZH.md").write_text(report, encoding="utf-8")

    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    selected = ("a4noise_R1-D2-P1-S1-A4", "poolbottom_R1-D2-P5-S1-A1", "rack_R2-D2-P5-S1-A1")
    for stem in selected:
        with np.load(Path(args.data) / "val" / f"{stem}.npz") as data:
            points = data["points"].astype(np.float32)
            labels = data["labels"].astype(np.int64)
        with np.load(baseline_dir / "independent_vote_counts" / f"{stem}_predictions.npz") as data:
            pred_a = data["prediction"]
        with np.load(refine_dir / "independent_vote_counts" / f"{stem}_predictions.npz") as data:
            pred_b = data["prediction"]
        rng = np.random.default_rng(1001)
        keep = rng.choice(len(points), min(len(points), 60000), replace=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        axes[0].scatter(points[keep, 0], points[keep, 1], c=labels[keep], s=1, cmap="coolwarm", vmin=0, vmax=1)
        axes[0].set_title("Ground truth")
        for ax, pred, title in ((axes[1], pred_a, "Baseline errors"), (axes[2], pred_b, "Refine errors")):
            colors = np.full((len(keep), 4), (0.65, 0.65, 0.65, 0.12))
            fp = (pred[keep] == 1) & (labels[keep] == 0)
            fn = (pred[keep] == 0) & (labels[keep] == 1)
            colors[fp] = (0.9, 0.1, 0.1, 0.9)
            colors[fn] = (0.1, 0.25, 0.95, 0.9)
            ax.scatter(points[keep, 0], points[keep, 1], c=colors, s=1)
            ax.set_title(title + " (red FP, blue FN)")
        for ax in axes:
            ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("X"); ax.set_ylabel("Y")
        fig.suptitle(stem)
        fig.savefig(figure_dir / f"{stem}_fp_fn.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
