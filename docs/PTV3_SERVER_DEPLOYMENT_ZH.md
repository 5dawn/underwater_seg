# 官方 PTv3 服务器部署与最小集成验证

本文记录官方 Point Transformer V3（PTv3）基线在 Linux GPU 服务器上的可复现部署流程。服务器登录地址、端口、口令以及本机私钥路径不进入仓库。

## 1. 固定范围

- 官方实现：Pointcept/PointTransformerV3。
- 官方源码提交：`3229e9b7de1770c8ad17c316f8e349982de509f8`。
- Python：3.10。
- PyTorch：2.8.0 + CUDA 12.8。
- 数据：45 个训练文件和 9 个验证文件。
- 历史测试集不上传、不运行，也不参与结构、参数或门槛选择。
- 本阶段只验证官方 PTv3 基线；水下局部特征细化模块保持关闭。

完整环境清单、源码校验值和运行结果见
`results/ptv3/environment/baseline_smoke_report.md`。

## 2. 服务器目录

建议将项目和环境放在独立目录中，避免影响服务器上的其他用户或现有环境：

```text
<work-dir>/underwater_seg_ptv3/
├── official/PointTransformerV3/
├── data/train/
├── data/val/
├── configs/
├── src/
├── tests/
├── tools/
└── results/ptv3/
```

Conda 环境名固定为 `ptv3`。不要修改系统 Python、系统 CUDA 或其他用户的 Conda 环境。

## 3. 环境建立

```bash
conda create -n ptv3 python=3.10 -y
conda activate ptv3

python -m pip install +  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 +  --index-url https://download.pytorch.org/whl/cu128

python -m pip install +  numpy scipy pyyaml pytest einops addict timm +  torch-geometric
```

按当前 PyTorch/CUDA 组合安装兼容的 `torch-scatter` 与 `spconv` wheel。安装后先执行最小 CUDA 检查，确认 PyTorch 能识别 GPU，并能完成张量计算。

本次实际版本记录在：

- `results/ptv3/environment/conda_list.txt`
- `results/ptv3/environment/pip_freeze.txt`
- `results/ptv3/environment/environment.txt`
- `results/ptv3/environment/cuda_smoke.json`

## 4. 官方源码固定

优先从官方仓库取得指定提交，并记录远端 URL 与提交号。若服务器网络导致 Git smart HTTP 无法稳定完成，可从 GitHub 官方 codeload 下载该提交的归档；此时必须记录归档 SHA256，且不能把归档目录误称为可执行 `git rev-parse` 的 Git checkout。

本次来源记录：

- `results/ptv3/environment/PTV3_REPO_URL.txt`
- `results/ptv3/environment/PTV3_COMMIT.txt`
- `results/ptv3/environment/PTV3_ARCHIVE_SHA256.txt`

官方模型核心文件保持原样，项目适配逻辑放在 `src/model_ptv3.py`。

## 5. 数据传输规则

只同步训练集和验证集：

```text
data/train/  45 files
data/val/     9 files
```

传输完成后核对文件数量。历史测试集不得出现在服务器项目目录中。任何评估脚本均应显式接收 val 路径，避免通过目录遍历误读 test。

## 6. 执行顺序

在项目根目录、激活 `ptv3` 环境后依次运行：

```bash
python tools/ptv3_official_smoke.py
python tools/ptv3_adapter_smoke.py
python tools/ptv3_real_smoke.py
python tools/ptv3_val_smoke.py
pytest -q tests/test_ptv3_protocol.py tests/test_official_ptv3_integration.py
```

各阶段含义：

1. `ptv3_official_smoke.py`：直接实例化官方 PTv3，检查完整编码器/解码器前向。
2. `ptv3_adapter_smoke.py`：检查项目接口 `[B,3,N] -> [B,2,N]`、体素化、逆映射、损失和梯度。
3. `ptv3_real_smoke.py`：使用真实训练样本完成小样本拟合、严格 checkpoint 重载及固定评估 RNG 复现。
4. `ptv3_val_smoke.py`：对 9 个验证文件执行基于 XYZ 的空间分块、5 次完整覆盖，并保存每点预测次数。
5. Pytest：验证体素逆映射、重复点、混合批次、少邻点、ignore、输入重排、RNG 隔离和 checkpoint 行为。

## 7. 当前验收结果

- 官方完整 PTv3 前向通过，末级 decoder 通道数为 64。
- 批量 `B=4, N=4096` 输出为 `[4,2,4096]`。
- 真实数据前向、反向和有限梯度检查通过。
- 两云小样本拟合损失从 1.270482 降至 0.004242。
- checkpoint 严格重载后，在固定评估状态下 logits 最大差值为 0。
- 9 个验证文件均完成 5 次完整覆盖，每点预测次数恰好为 5。
- 同一 checkpoint 的两次独立评估结果完全一致。
- 15 项必要测试全部通过。

这些结果只证明运行环境、官方基线数据路径和最小集成正确，不代表 10 epochs 已充分收敛，也不构成 Refine 有效性的证据。

## 8. 后续顺序

继续严格遵循：

```text
官方基线验收
→ 体素诊断及至多一次预设修正
→ 冻结共同配置
→ seed 42 的 baseline / Refine 配对训练
→ 达到预设门槛后消融
→ 消融支持后多 seed
```

共同配置发生变化时，旧运行仅保留为准备实验，不与新结果混用。
