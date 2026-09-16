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

2026-09-16 已在 AutoDL Ubuntu 22.04.4 / RTX 5090 服务器完成正式训练接入与 seed 42 配对实验：

- 独立训练入口支持 baseline/refine、配置冻结、数据哈希、公共初始化、best/last、严格恢复和独立评估。
- 真实长度验证尾块、Morton 并列 XYZ 排序、CPU/CUDA RNG 隔离、下一批与下一步恢复一致性检查通过。
- 17 项测试通过；训练内 best 与独立评估逐字节一致；A/B 前两批逐 epoch 指纹完全一致。
- Baseline best 为 epoch 10：foreground IoU `0.221773`，Precision `0.271180`，Recall `0.548990`，Background FPR `0.083746`。
- Refine best 为 epoch 9：foreground IoU `0.280962`，Precision `0.315173`，Recall `0.721323`，Background FPR `0.088961`。
- Refine 的 IoU、Precision、Recall 改善，但 Background FPR 增加 `0.005215`，未满足“严格下降”的冻结门槛。
- 按协议停止消融与多 seed，不放宽门槛。结论仅适用于当前 45/9 划分与 10 epochs 短预算。

正式报告见 `results/ptv3/formal/seed42/PAIR_REPORT_ZH.md`。大型 checkpoint 保留在服务器，路径及 SHA256 见 `results/ptv3/formal/seed42/CHECKPOINT_SHA256.txt`。historical test 未上传、未使用。
