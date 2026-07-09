import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("photo") / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "photo"
OUT.mkdir(exist_ok=True)
(OUT / ".mplconfig").mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.figsize": (12.8, 7.2),
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
    }
)

COLORS = {
    "blue": "#2F6FED",
    "cyan": "#18A6A6",
    "green": "#3D9A57",
    "yellow": "#E3A018",
    "red": "#D94B4B",
    "purple": "#7C5CC4",
    "gray": "#6C7280",
    "light": "#E8EDF5",
    "dark": "#263238",
}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def add_value_labels(ax, fmt="{:.0f}", dy=0.01):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for patch in ax.patches:
        h = patch.get_height()
        if math.isnan(h):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            h + span * dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=10,
            color=COLORS["dark"],
        )


def load_json(path):
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_json_stream(path):
    text = (ROOT / path).read_text(encoding="utf-8", errors="ignore")
    dec = json.JSONDecoder()
    out = []
    i = 0
    while i < len(text):
        start = text.find("{", i)
        if start < 0:
            break
        try:
            obj, end = dec.raw_decode(text[start:])
        except json.JSONDecodeError:
            i = start + 1
            continue
        out.append(obj)
        i = start + end
    return [x for x in out if isinstance(x, dict) and "epoch" in x]


group_report = load_json("label2_group_report.json")
exp_a = load_json("runs/label2_pointnet2_exp_a/test_metrics.json")
exp_a_last = load_json("runs/label2_pointnet2_exp_a/test_metrics_last.json")
exp_a_epochs = parse_json_stream("runs/label2_pointnet2_exp_a/train_console.log")
exp_c_epochs = parse_json_stream("runs/label2_pointnet2_exp_c/train_console.log")


def chart_01_pipeline():
    fig, ax = plt.subplots()
    ax.axis("off")
    steps = [
        ("label2 原始点云", "39 个点云\n常规 / 池底 / A4强噪声"),
        ("数据暂存", "统一命名\n硬链接整理"),
        ("预处理", "NPZ 点云与标签\n忽略标签=-1"),
        ("模型训练", "PointNet 冒烟测试\nPointNet++ 实验 A/C"),
        ("模型评估", "全点云投票\n潜艇 IoU/精度/召回/F1"),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, (title, body) in enumerate(steps):
        ax.add_patch(
            plt.Rectangle(
                (xs[i] - 0.085, 0.42),
                0.17,
                0.22,
                facecolor=COLORS["light"],
                edgecolor=COLORS["blue"],
                linewidth=1.5,
            )
        )
        ax.text(xs[i], 0.57, title, ha="center", va="center", fontsize=13, weight="bold")
        ax.text(xs[i], 0.48, body, ha="center", va="center", fontsize=10, color=COLORS["dark"])
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.095, 0.53),
                xytext=(xs[i] + 0.095, 0.53),
                arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=2),
            )
    ax.set_title("近期工作流程：从数据整理到误差诊断", pad=18)
    save(fig, "01_pipeline_overview.png")


def chart_02_condition_counts():
    counts = group_report["condition_counts"]
    labels = ["常规场景", "池底场景", "A4强噪声"]
    vals = [counts["S1_A1_normal"], counts["C_pool_bottom"], counts["A4_strong_laser_noise"]]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, vals, color=[COLORS["blue"], COLORS["cyan"], COLORS["red"]])
    ax.set_title("label2 场景/工况组成")
    ax.set_ylabel("点云文件数量")
    ax.set_ylim(0, max(vals) * 1.25)
    add_value_labels(ax)
    ax.text(0.02, 0.93, "共 39 个 PLY 文件", transform=ax.transAxes, fontsize=12, color=COLORS["gray"])
    save(fig, "02_condition_counts.png")


def chart_03_distance_pose():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2))
    dist = group_report["distance_counts"]
    pose = group_report["pose_counts"]
    axes[0].bar(dist.keys(), dist.values(), color=COLORS["blue"])
    axes[0].set_title("距离分布")
    axes[0].set_ylabel("文件数量")
    axes[0].set_ylim(0, max(dist.values()) * 1.25)
    add_value_labels(axes[0])
    axes[1].bar(pose.keys(), pose.values(), color=COLORS["cyan"])
    axes[1].set_title("姿态分布")
    axes[1].set_ylabel("文件数量")
    axes[1].set_ylim(0, max(pose.values()) * 1.25)
    add_value_labels(axes[1])
    fig.suptitle("label2 距离与姿态覆盖情况", fontsize=18, weight="bold")
    save(fig, "03_distance_pose_distribution.png")


def chart_04_label_totals():
    totals = group_report["label_totals"]
    labels = ["忽略点", "背景", "潜艇"]
    vals = [totals["-1"], totals["0"], totals["1"]]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["green"]]
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        vals,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        startangle=110,
        colors=colors,
        textprops={"fontsize": 11},
    )
    ax.set_title("整体标签分布")
    ax.text(0, -1.25, f"总标注点数：{sum(vals):,}", ha="center", fontsize=12, color=COLORS["gray"])
    save(fig, "04_label_totals_pie.png")


def chart_05_label_by_condition():
    conds = [
        ("常规场景", "S1_A1_normal"),
        ("池底场景", "C_pool_bottom"),
        ("A4强噪声", "A4_strong_laser_noise"),
    ]
    ignore = []
    bg = []
    sub = []
    for _, key in conds:
        d = group_report["label_totals_by_condition"][key]
        total = d["-1"] + d["0"] + d["1"]
        ignore.append(d["-1"] / total * 100)
        bg.append(d["0"] / total * 100)
        sub.append(d["1"] / total * 100)
    x = np.arange(len(conds))
    fig, ax = plt.subplots()
    ax.bar(x, bg, label="背景", color=COLORS["blue"])
    ax.bar(x, sub, bottom=bg, label="潜艇", color=COLORS["green"])
    ax.bar(x, ignore, bottom=np.array(bg) + np.array(sub), label="忽略点", color=COLORS["gray"])
    ax.set_xticks(x, [c[0] for c in conds])
    ax.set_ylim(0, 100)
    ax.set_ylabel("点数占比（%）")
    ax.set_title("不同工况下的标签比例")
    ax.legend(loc="upper right")
    for i, v in enumerate(sub):
        ax.text(i, bg[i] + v / 2, f"潜艇 {v:.1f}%", ha="center", va="center", color="white", fontsize=10)
    save(fig, "05_label_ratio_by_condition.png")


def chart_06_split_counts():
    counts = group_report["split_counts"]
    labels = ["训练集", "验证集", "测试集"]
    keys = ["train", "val", "test"]
    vals = [counts[k] for k in keys]
    fig, ax = plt.subplots()
    ax.bar(labels, vals, color=[COLORS["blue"], COLORS["yellow"], COLORS["red"]])
    ax.set_title("固定数据划分数量")
    ax.set_ylabel("文件数量")
    ax.set_ylim(0, max(vals) * 1.25)
    add_value_labels(ax)
    save(fig, "06_split_counts.png")


def chart_07_split_label_ratios():
    stats = defaultdict(lambda: Counter())
    for rec in group_report["records"]:
        stats[rec["split"]].update(rec["labels"])
    splits = ["train", "val", "test"]
    split_labels = ["训练集", "验证集", "测试集"]
    ignore, bg, sub = [], [], []
    for split in splits:
        d = stats[split]
        total = d["-1"] + d["0"] + d["1"]
        ignore.append(d["-1"] / total * 100)
        bg.append(d["0"] / total * 100)
        sub.append(d["1"] / total * 100)
    x = np.arange(len(splits))
    fig, ax = plt.subplots()
    ax.bar(x, bg, label="背景", color=COLORS["blue"])
    ax.bar(x, sub, bottom=bg, label="潜艇", color=COLORS["green"])
    ax.bar(x, ignore, bottom=np.array(bg) + np.array(sub), label="忽略点", color=COLORS["gray"])
    ax.set_xticks(x, split_labels)
    ax.set_ylabel("点数占比（%）")
    ax.set_ylim(0, 100)
    ax.set_title("不同数据划分下的标签比例")
    ax.legend(loc="upper right")
    for i, v in enumerate(sub):
        ax.text(i, bg[i] + v / 2, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=10)
    save(fig, "07_split_label_ratios.png")


def chart_08_model_roadmap():
    fig, ax = plt.subplots()
    ax.axis("off")
    items = [
        ("PointNet 冒烟测试", "3轮训练\n流程验证通过\n潜艇IoU 0.0051", COLORS["gray"]),
        ("PointNet++ 实验A", "50轮训练\n最佳测试潜艇IoU 0.0982\n高召回、低精度", COLORS["blue"]),
        ("PointNet++ 实验C", "训练中/阶段结果\n降低前景压力\n验证最佳0.1358", COLORS["yellow"]),
        ("DGCNN 实验A", "代码已实现\n尚未正式训练", COLORS["purple"]),
    ]
    y = 0.72
    for i, (title, body, color) in enumerate(items):
        x = 0.12 + i * 0.25
        ax.scatter([x], [y], s=650, color=color, zorder=3)
        ax.text(x, y, str(i + 1), ha="center", va="center", color="white", fontsize=13, weight="bold")
        ax.text(x, 0.48, title, ha="center", va="center", fontsize=13, weight="bold")
        ax.text(x, 0.34, body, ha="center", va="center", fontsize=10)
        if i < len(items) - 1:
            ax.plot([x + 0.045, x + 0.205], [y, y], color=COLORS["light"], lw=6, solid_capstyle="round")
            ax.annotate("", xy=(x + 0.205, y), xytext=(x + 0.16, y), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("模型实验路线图", pad=16)
    save(fig, "08_model_experiment_roadmap.png")


def chart_09_smoke_vs_exp_a():
    metrics = ["准确率", "mIoU", "潜艇IoU"]
    smoke = [0.8735, 0.4392, 0.0051]
    pointnet2 = [exp_a["accuracy"], exp_a["mIoU"], exp_a["submarine_iou"]]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - w / 2, smoke, w, label="PointNet 冒烟测试", color=COLORS["gray"])
    ax.bar(x + w / 2, pointnet2, w, label="PointNet++ 实验A最佳模型", color=COLORS["blue"])
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.0)
    ax.set_title("冒烟基线与 PointNet++ 实验A 对比")
    ax.set_ylabel("指标值")
    ax.legend()
    for i, v in enumerate(smoke):
        ax.text(i - w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(pointnet2):
        ax.text(i + w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    save(fig, "09_smoke_vs_pointnet2_exp_a.png")


def epoch_series(records, metric):
    epochs = [r["epoch"] for r in records]
    vals = [r["val"][metric] for r in records]
    return epochs, vals


def chart_10_exp_a_curve():
    fig, ax = plt.subplots()
    e, iou = epoch_series(exp_a_epochs, "submarine_iou")
    _, prec = epoch_series(exp_a_epochs, "submarine_precision")
    _, rec = epoch_series(exp_a_epochs, "submarine_recall")
    ax.plot(e, iou, label="潜艇IoU", color=COLORS["blue"], lw=2.2)
    ax.plot(e, prec, label="精度", color=COLORS["red"], lw=1.8)
    ax.plot(e, rec, label="召回率", color=COLORS["green"], lw=1.8)
    best_i = int(np.argmax(iou))
    ax.scatter([e[best_i]], [iou[best_i]], color=COLORS["blue"], s=70, zorder=3)
    ax.annotate(f"最佳轮次 {e[best_i]}\nIoU {iou[best_i]:.3f}", xy=(e[best_i], iou[best_i]), xytext=(e[best_i] + 4, iou[best_i] + 0.08), arrowprops=dict(arrowstyle="->"))
    ax.set_title("PointNet++ 实验A 验证集曲线")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("验证指标")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    save(fig, "10_exp_a_validation_curve.png")


def chart_11_best_vs_last():
    metrics = ["accuracy", "mIoU", "submarine_iou", "submarine_precision", "submarine_recall", "submarine_f1"]
    labels = ["准确率", "mIoU", "潜艇IoU", "精度", "召回率", "F1"]
    best = [exp_a[m] for m in metrics]
    last = [exp_a_last[m] for m in metrics]
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots()
    ax.bar(x - w / 2, best, w, label="最佳检查点", color=COLORS["blue"])
    ax.bar(x + w / 2, last, w, label="最终检查点", color=COLORS["yellow"])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("PointNet++ 实验A：最佳与最终检查点测试对比")
    ax.legend()
    save(fig, "11_exp_a_best_vs_last_checkpoint.png")


def chart_12_test_metrics():
    labels = ["准确率", "mIoU", "潜艇IoU", "精度", "召回率", "F1"]
    vals = [exp_a["accuracy"], exp_a["mIoU"], exp_a["submarine_iou"], exp_a["submarine_precision"], exp_a["submarine_recall"], exp_a["submarine_f1"]]
    colors = [COLORS["blue"], COLORS["cyan"], COLORS["purple"], COLORS["red"], COLORS["green"], COLORS["yellow"]]
    fig, ax = plt.subplots()
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_title("PointNet++ 实验A 测试集指标")
    ax.set_ylabel("指标值")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    save(fig, "12_exp_a_test_metrics.png")


def chart_13_precision_recall():
    fig, ax = plt.subplots()
    vals = [exp_a["submarine_precision"], exp_a["submarine_recall"]]
    ax.bar(["精度", "召回率"], vals, color=[COLORS["red"], COLORS["green"]], width=0.45)
    ax.set_ylim(0, 1.05)
    ax.set_title("主要问题：召回率高，但精度低")
    ax.set_ylabel("潜艇类别指标")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.48, "模型能找到大部分潜艇点，\n但同时把大量背景点误判为潜艇。", ha="center", va="center", transform=ax.transAxes, fontsize=14, color=COLORS["dark"])
    save(fig, "13_precision_recall_gap.png")


def chart_14_confusion():
    cm = np.array(exp_a["confusion_matrix"])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("PointNet++ 实验A 测试集混淆矩阵")
    ax.set_xticks([0, 1], ["预测背景", "预测潜艇"])
    ax.set_yticks([0, 1], ["真实背景", "真实潜艇"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() * 0.55 else COLORS["dark"], fontsize=14, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "14_exp_a_confusion_matrix.png")


def chart_15_fp_fn():
    cm = np.array(exp_a["confusion_matrix"])
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    fig, ax = plt.subplots()
    vals = [fp, fn, tp]
    labels = ["误检点", "漏检点", "正确检出点"]
    ax.bar(labels, vals, color=[COLORS["red"], COLORS["yellow"], COLORS["green"]])
    ax.set_title("错误数量分解")
    ax.set_ylabel("点数")
    ax.ticklabel_format(style="plain", axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.03, f"{v:,}", ha="center", fontsize=11)
    save(fig, "15_false_positive_false_negative_counts.png")


def chart_16_per_file_iou():
    files = exp_a["per_file"]
    def zh_file_name(name):
        name = name.replace(".npz", "")
        name = name.replace("a4noise", "A4强噪声")
        name = name.replace("poolbottom", "池底")
        name = name.replace("normal", "常规")
        return name

    data = sorted([(zh_file_name(f["file"]), f["submarine_iou"]) for f in files], key=lambda x: x[1])
    labels = [x[0] for x in data]
    vals = [x[1] for x in data]
    fig, ax = plt.subplots(figsize=(12.8, 8.0))
    colors = [COLORS["red"] if "A4强噪声" in name else COLORS["cyan"] if "池底" in name else COLORS["blue"] for name in labels]
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("潜艇IoU")
    ax.set_title("逐文件测试潜艇IoU（PointNet++ 实验A）")
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    save(fig, "16_per_file_submarine_iou.png")


def chart_17_group_perf():
    groups = {
        "A4强噪声": [f for f in exp_a["per_file"] if f["file"].startswith("a4noise")],
        "池底场景": [f for f in exp_a["per_file"] if f["file"].startswith("poolbottom")],
        "常规D3": [f for f in exp_a["per_file"] if f["file"].startswith("normal")],
    }
    metrics = ["submarine_iou", "submarine_precision", "submarine_recall"]
    labels = ["潜艇IoU", "精度", "召回率"]
    x = np.arange(len(groups))
    w = 0.24
    fig, ax = plt.subplots()
    for i, metric in enumerate(metrics):
        vals = [np.mean([f[metric] for f in files]) for files in groups.values()]
        ax.bar(x + (i - 1) * w, vals, w, label=labels[i], color=[COLORS["blue"], COLORS["red"], COLORS["green"]][i])
    ax.set_xticks(x, groups.keys())
    ax.set_ylim(0, 1.05)
    ax.set_title("困难样本分组表现")
    ax.set_ylabel("平均指标")
    ax.legend()
    save(fig, "17_group_performance_summary.png")


def chart_18_exp_c_curve():
    fig, ax = plt.subplots()
    e, iou = epoch_series(exp_c_epochs, "submarine_iou")
    _, prec = epoch_series(exp_c_epochs, "submarine_precision")
    _, rec = epoch_series(exp_c_epochs, "submarine_recall")
    ax.plot(e, iou, label="潜艇IoU", color=COLORS["blue"], lw=2.2)
    ax.plot(e, prec, label="精度", color=COLORS["red"], lw=1.8)
    ax.plot(e, rec, label="召回率", color=COLORS["green"], lw=1.8)
    best_i = int(np.argmax(iou))
    ax.scatter([e[best_i]], [iou[best_i]], color=COLORS["blue"], s=70)
    ax.annotate(f"最佳轮次 {e[best_i]}\nIoU {iou[best_i]:.3f}", xy=(e[best_i], iou[best_i]), xytext=(e[best_i] + 2, iou[best_i] + 0.1), arrowprops=dict(arrowstyle="->"))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("验证指标")
    ax.set_title("PointNet++ 实验C 阶段性验证曲线")
    ax.legend(loc="upper right")
    save(fig, "18_exp_c_partial_validation_curve.png")


def chart_19_exp_a_vs_c():
    a_best = max(r["val"]["submarine_iou"] for r in exp_a_epochs)
    c_best = max(r["val"]["submarine_iou"] for r in exp_c_epochs)
    fig, ax = plt.subplots()
    ax.bar(["实验A\n前景0.10，权重1:3", "实验C阶段结果\n前景0.05，权重1:1.5"], [a_best, c_best], color=[COLORS["blue"], COLORS["yellow"]], width=0.5)
    ax.set_ylim(0, max(a_best, c_best) * 1.45)
    ax.set_ylabel("最佳验证潜艇IoU")
    ax.set_title("降低前景压力后的早期验证表现")
    for i, v in enumerate([a_best, c_best]):
        ax.text(i, v + max(a_best, c_best) * 0.05, f"{v:.3f}", ha="center", fontsize=13, weight="bold")
    save(fig, "19_exp_a_vs_exp_c_val_best.png")


def chart_20_next_plan():
    fig, ax = plt.subplots()
    ax.axis("off")
    tasks = [
        ("完成实验C", "跑满50轮并补充测试集指标"),
        ("控制误检", "调整类别权重、前景比例与判定阈值"),
        ("可视化诊断", "重点查看A4、池底、常规D3难例"),
        ("对比DGCNN", "使用相同划分和相同指标"),
        ("补充数据", "增加R2、D3/P5、C、A4困难样本"),
    ]
    y0 = 0.78
    for i, (title, body) in enumerate(tasks):
        y = y0 - i * 0.14
        ax.add_patch(plt.Circle((0.12, y), 0.035, color=COLORS["blue"] if i == 0 else COLORS["light"], ec=COLORS["blue"], lw=2))
        ax.text(0.12, y, str(i + 1), ha="center", va="center", fontsize=11, weight="bold", color="white" if i == 0 else COLORS["blue"])
        ax.text(0.19, y + 0.025, title, ha="left", va="center", fontsize=14, weight="bold")
        ax.text(0.19, y - 0.03, body, ha="left", va="center", fontsize=11, color=COLORS["gray"])
        if i < len(tasks) - 1:
            ax.plot([0.12, 0.12], [y - 0.04, y - 0.10], color=COLORS["light"], lw=4)
    ax.set_title("下一阶段实验计划", pad=16)
    save(fig, "20_next_experiment_plan.png")


def main():
    for func in [
        chart_01_pipeline,
        chart_02_condition_counts,
        chart_03_distance_pose,
        chart_04_label_totals,
        chart_05_label_by_condition,
        chart_06_split_counts,
        chart_07_split_label_ratios,
        chart_08_model_roadmap,
        chart_09_smoke_vs_exp_a,
        chart_10_exp_a_curve,
        chart_11_best_vs_last,
        chart_12_test_metrics,
        chart_13_precision_recall,
        chart_14_confusion,
        chart_15_fp_fn,
        chart_16_per_file_iou,
        chart_17_group_perf,
        chart_18_exp_c_curve,
        chart_19_exp_a_vs_c,
        chart_20_next_plan,
    ]:
        func()
    print(f"Wrote charts to {OUT}")
    for path in sorted(OUT.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
