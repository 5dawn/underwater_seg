# label2 分组与跑通记录

检查日期：2026-07-07

## 输入数据

原始目录：`label2/`

共检查到 39 个 `.ply`：

| 条件组 | 数量 | 说明 |
| --- | ---: | --- |
| `S1_A1_normal` | 15 | 根目录下的 `R1-D1~D3-P1~P5-S1-A1` |
| `C_pool_bottom` | 14 | `C/`，目标在池底；缺少 `D3-P5` |
| `A4_strong_laser_noise` | 10 | `A4/`，强激光噪声组；包含 D1/D2 |

所有文件都有可识别标签字段 `scalar_label`，标签值均合法：`-1 ignore`、`0 background`、`1 submarine`。

## 自动分组产物

为避免 `C/` 与根目录文件同名导致预处理覆盖，已生成唯一命名 staging 数据：

- staging 原始数据：`data/raw_label2/`
- 固定划分文件：`splits_label2.json`
- 分组与标签统计：`label2_group_report.json`
- 准备脚本：`tools/prepare_label2.py`

`data/raw_label2/` 中 39 个文件均为硬链接，不额外占用一份完整点云空间。

## 划分策略

当前划分：

| split | 数量 | 用途 |
| --- | ---: | --- |
| train | 23 | 主要包含 normal、C 的 D1/D2，以及少量 A4 D1 |
| val | 5 | 包含 D2/D3、C、少量 A4，用于观察泛化 |
| test | 11 | 主要包含 D3、C 池底难例、A4 强噪声压力测试 |

标签总量：

| split | ignore | background | submarine | submarine ratio |
| --- | ---: | ---: | ---: | ---: |
| train | 211143 | 8435794 | 1271882 | 0.1282 |
| val | 13158 | 1646516 | 87329 | 0.0500 |
| test | 172131 | 6109452 | 403987 | 0.0604 |

验证/测试集潜艇比例较低，是因为远距离与强噪声样本更难，符合压力测试目的。

## 跑通结果

已成功完成：

1. `label2` 标签统计与分组。
2. `data/raw_label2/` staging。
3. `data/processed_label2/` 预处理。
4. `src/inspect_labels.py` 标签检查。
5. 3 epoch smoke 训练。
6. test split 评估。

使用配置：

- `configs/label2_smoke.yaml`
- 模型：PointNet
- 特征：xyz
- 训练：3 epoch，foreground-balanced sampling，class weight `[1.0, 3.0]`
- checkpoint：`checkpoints/label2_smoke_best.pth`

smoke 结果：

| split | accuracy | mIoU | submarine IoU | submarine recall | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| val | 0.8431 | 0.4225 | 0.0019 | 0.0059 | 第 3 epoch |
| test | 0.8735 | 0.4392 | 0.0051 | 0.0104 | 强噪声和 D3 难例占比较高 |

这个 smoke 分数不代表最终能力，只说明训练链路已经跑通。3 epoch 太短，且测试集刻意偏难，模型目前基本仍偏向预测背景。

## 下一步建议

1. 正式训练先用 PointNet 跑 50 epoch，沿用 `foreground_balanced`，观察 submarine IoU 是否稳定提升。
2. 如果仍然预测背景为主，把 `foreground_ratio` 从 `0.10` 提到 `0.20` 或 `0.25`，并尝试 class weight `[1.0, 5.0]`。
3. 暂时不要用 A4 作为主要训练来源；建议保留多数 A4 在 test，用来衡量强激光噪声鲁棒性。
4. 尽快补标一小批 R2，尤其是 `D3/P5/C/A4` 类困难样本，否则现在的评估仍主要是 R1 内泛化。
5. 后续如果要训练正式模型，建议新增 `configs/label2_pointnet_exp_d.yaml`，epochs 50，samples_per_cloud 32，val full-cloud votes 5。
