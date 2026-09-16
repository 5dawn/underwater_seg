# 官方 PTv3 基线与水下局部特征细化模块：实施状态

## 协议状态

- 固定数据划分：45 train / 9 val / 15 historical test。
- 当前阶段禁止使用 historical test 做结构、参数或门槛选择。
- 官方实现来源固定为 Pointcept/PointTransformerV3；必须填写不可变 commit 后才能把运行称为“官方 PTv3 baseline”。
- 体素大小只能按冻结规则最多修正一次；半径只能由 train 固定样本的第 16 近邻距离 P95 确定。
- 第一轮只允许 seed 42 的 A/B 10 epochs 配对实验；条件不通过时不运行消融，消融不通过时不运行多 seed。

## 已实现

- 无标签的逐样本 XYZ 体素均值聚合、逆映射及体素诊断。
- 官方 PTv3 backbone 薄适配层，保持 `[B,3,N] -> [B,2,N]`。
- 恢复原始点顺序后运行的“水下局部特征细化模块”。
- 分块、同样本、排除自身、最多 16 邻点、固定半径查询；少于 3 个有效邻点时残差为零。
- Refine 输出层全零初始化；公共 backbone/classifier 初始化快照与 Refine RNG 隔离。
- Background FPR 指标。
- 冻结配置草案和核心单元测试。
- train-only 固定采样诊断已完成：`voxel_size=0.005` 未触发修正，固定半径为 `0.05208010460155825`。

## 当前状态

2026-09-16 已转到 AutoDL Ubuntu 22.04.4 / RTX 5090 服务器完成官方 PTv3 环境与 baseline smoke。独立 `ptv3` 环境、官方固定 commit、standalone forward/backward、adapter B=4/N=4096、真实数据小样本拟合、checkpoint/resume、9-file val 5-vote 完整覆盖与 15 项测试均通过。historical test 未上传服务器。本轮按协议停止，尚未启动正式 10-epoch baseline。
