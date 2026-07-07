# underwater_seg 文件夹检查记录

检查日期：2026-07-07

## 顶层目录概览

| 路径 | 大小 | 最近修改 | 判断 |
| --- | ---: | --- | --- |
| `runs/` | 632.10 MB | 2026-06-10 | 实验输出、日志、预测点云，可归档但不建议直接删除 |
| `label/` | 613.71 MB | 2026-05-20 | 原始标注点云，13 个 `.ply`，应保留 |
| `data/` | 435.84 MB | 2026-05-20 | 处理后的数据，13 个 `.npz` 加 metadata，应保留 |
| `.git/` | 276.43 MB | 2026-06-30 | Git 仓库记录 |
| `checkpoints/` | 257.10 MB | 2026-06-10 | 模型权重，每个实验含 best/last，可按实验归档 |
| `label2/` | 222.59 MB | 2026-07-06 | 最近新增/修改的点云标注或数据，共 29 个 `.ply`，应先保留 |
| `src/` | 0.20 MB | 2026-06-10 | 项目源码，包含少量 `__pycache__` 缓存 |
| `configs/` | 0.01 MB | 2026-06-10 | 实验配置，应保留 |

## 数据与标注

- `label/`：包含 `train/`、`val/`、`test/`，共 13 个 `.ply`。
- `data/`：包含 `raw/`、`processed/`，共 13 个 `.npz`、1 个 `metadata.json`。
- `label2/`：根目录 15 个 `.ply`，`label2/C/` 下 14 个 `.ply`，合计 29 个 `.ply`。最近修改时间在 2026-07-01 到 2026-07-06 之间，属于近期数据，不建议清理。

## 实验日志与指标

`runs/` 下主要日志文件是 `train_console.log`、`train_resume_console.log` 和 metrics JSON。日志中的 `NativeCommandError` 出现在 PowerShell 启动包装信息处，后续训练继续运行并写出指标，不像训练中途崩溃。

| 实验 | 指标文件 | accuracy | mIoU | 时间 |
| --- | --- | ---: | ---: | --- |
| `pointnet_exp_d` | test | 0.9831 | 0.7403 | 2026-05-26 |
| `pointnet_exp_d` | val | 0.9789 | 0.7976 | 2026-05-26 |
| `pointnet_exp_g` | test | 0.9560 | 0.5083 | 2026-05-26 |
| `pointnet_exp_i` | test | 0.9682 | 0.5110 | 2026-05-27 |
| `pointnet2_exp_a` | test | 0.9733 | 0.4932 | 2026-06-10 |
| `pointnet2_exp_b` | test | 0.9731 | 0.4881 | 2026-06-10 |
| `pointnet2_smoke` | val | 0.9537 | 0.4778 | 2026-06-10 |
| `pointnet` | test | 0.1458 | 0.0763 | 2026-05-20 |
| `pointnet_cpu_backup_20260520_152339` | test_cpu | 0.0269 | 0.0134 | 2026-05-20 |

当前最值得优先保留和复现实验的是 `pointnet_exp_d`，它的 test mIoU 为 0.7403，val mIoU 为 0.7976。

## 大文件

- `runs/exp_g_error_compare.ply`：86.40 MB，单独的误差对比点云。
- `label/train/*.ply`：最大约 64.82 MB，是原始标注数据。
- `runs/*/pred_10-48-56.ply`：每个约 43.20 MB，是各实验预测输出，累计占用较大。
- `checkpoints/*.pth`：单个约 9.8-10.5 MB，每个实验通常有 `best` 和 `last` 两份。

## 建议整理方案

1. 保留源码与配置：`src/`、`configs/`、`README.md`、`requirements.txt`、`splits_label13.json`。
2. 保留数据：`label/`、`label2/`、`data/`。其中 `label2/` 建议后续补充来源说明和 train/val/test 划分。
3. 保留关键实验：`runs/pointnet_exp_d/`、`checkpoints/pointnet_exp_d_best.pth`、`checkpoints/pointnet_exp_d_last.pth`。
4. 归档一般实验：`pointnet_exp_a/b/c/e/f/g/h/i`、`pointnet2_exp_a/b`、`pointnet2_smoke` 可移入类似 `archive/runs_202605-202606/` 的归档目录。
5. 可清理缓存：`src/__pycache__/` 可删除；`.gitignore` 已包含 `__pycache__/`。
6. 可考虑删除或归档预测点云：`runs/*/pred_10-48-56.ply` 占用较大；如果 metrics 和 best checkpoint 已保留，这些可以压缩归档。

## `.gitignore` 状态

`.gitignore` 已覆盖：

- Python 缓存：`__pycache__/`、`*.py[cod]`
- 实验输出：`runs/*`
- 权重：`checkpoints/*`、`*.pth`
- 数据产物：`data/raw/*`、`data/processed/*`
- 日志：`*.log`

整体看，当前忽略规则是合理的。
