# 官方 PTv3 baseline smoke 报告

## 结论

服务器上的官方 PTv3 baseline 最小集成验证已经通过，具备启动固定协议 10-epoch baseline 的技术条件。本轮按要求停在 smoke 阶段，没有开始正式训练、Refine、消融、多 seed、threshold calibration 或 historical test。

## 环境

- 服务器：AutoDL，Ubuntu 22.04.4 LTS。
- GPU：NVIDIA GeForce RTX 5090，32607 MiB。
- Driver：570.211.01，CUDA compatibility 12.8。
- Conda：独立环境 `/root/miniconda3/envs/ptv3`。
- Python：3.10.21。
- PyTorch：2.8.0+cu128；CUDA 可用。
- torch_scatter：2.1.2+pt28cu128，GPU forward/backward 通过。
- spconv：2.3.8（cu126 wheel，在 CUDA 12.8 driver 上运行），实际 `SubMConv3d` 通过。

## 官方源码

- Repo：`https://github.com/Pointcept/PointTransformerV3`。
- Commit：`3229e9b7de1770c8ad17c316f8e349982de509f8`。
- 获取方式：服务器无法连接 GitHub smart HTTP；从 GitHub 官方 API 固定 commit，并从官方 `codeload.github.com` 下载该 SHA 的 ZIP。
- ZIP SHA256：`b40828efa4d2f45adeb442724406cefae307f17c8c62da3d4bacd091dd0762b4`。
- 未修改官方 `model.py`、serialization 或 encoder/decoder 核心源码。
- 归档目录没有 `.git`，因此没有伪称执行过 `git rev-parse HEAD`；commit 由官方 API 记录。

## 官方实际配置

- encoder depths/channels/heads：`[2,2,2,6,2]` / `[32,64,128,256,512]` / `[2,4,8,16,32]`。
- encoder patch size：全阶段 128。
- decoder depths/channels/heads：`[2,2,2,2]` / `[64,64,128,256]` / `[4,4,8,16]`。
- decoder patch size：全阶段 128。
- `enable_flash=false`，`upcast_attention=true`，`upcast_softmax=true`。
- 最后 decoder feature 维度：64。
- 输入 feature：仅归一化 XYZ；未增加 RGB、反射率、range 或 normal。

## Standalone 官方模型

- B=2，每样本 4096 点；输出 `[8192,64]`。
- 官方 backbone 参数量：46,154,272。
- forward/backward/optimizer：8.7073 / 14.8174 / 0.0635 s（含首次 CUDA/spconv 初始化）。
- 峰值 allocated/reserved：3,662,801,920 / 4,332,716,032 bytes。
- loss、输出与梯度均为有限值。

## Adapter 与真实数据 smoke

- 冻结 `voxel_size=0.005`；Refine 关闭；接口 `[B,3,N] -> [B,2,N]`。
- B=4、N=4096，输出 `[4,2,4096]`。
- 4 个真实训练样本 voxel 数：3563、3670、2889、3202；比例 0.8699、0.8960、0.7053、0.7817。
- ignore 点数 546；loss 0.615379；logits 和梯度均有限。
- forward/backward/optimizer：0.5627 / 0.4887 / 0.0625 s。
- 峰值 allocated/reserved：1,658,022,400 / 1,944,059,904 bytes。
- 总参数量（backbone + Linear(64,2)）：46,154,402。
- checkpoint 包含 model、optimizer、scheduler、epoch、best metric、config 和 RNG state；strict load 通过。
- 固定相同 eval RNG 后，重载前后最大 logits 差为 0。

## 小样本拟合

- 两个固定 train cloud，每个固定采样 1024 点，10 steps。
- loss：1.270482 -> 0.004242，相对下降 99.67%。
- 仅用于验证训练链路，不作为性能结果。

## Validation smoke

- 只上传并读取 45 train / 9 val；服务器不存在 historical test 目录。
- XYZ Morton 空间分块，4096 点/block，5 轮完整覆盖。
- 所有 val 原始点 coverage min=max=5，逐文件 vote counts 已保存。
- raw argmax，`eval_seed=1001`；输出逐文件、overall、分场景与 Background FPR。
- 同 checkpoint/seed 的训练内式调用与独立脚本 JSON 完全一致。
- smoke checkpoint 未正式训练，指标接近全前景预测，不得作为性能结论。

## 测试与下一步

- 原有协议测试 7 项、新增官方集成测试 8 项，合计 `15 passed`。
- 技术条件已具备启动 10 epochs；正式运行前只需冻结本环境/commit/config 并确认 GPU 空闲。
- 本报告不授权或启动正式训练。
