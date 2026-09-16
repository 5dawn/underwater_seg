# PTv3 baseline / 水下局部特征细化模块：seed 42 正式配对结果

本报告仅描述当前 45/9 数据划分与 10 epochs 短训练预算下的 validation 结果。历史 test 未参与。

## Overall

| Metric | Baseline | Refine | Delta |
|---|---:|---:|---:|
| foreground_iou | 0.221773 | 0.280962 | +0.059188 |
| foreground_precision | 0.271180 | 0.315173 | +0.043992 |
| foreground_recall | 0.548990 | 0.721323 | +0.172334 |
| background_fpr | 0.083746 | 0.088961 | +0.005215 |

## Scenes

| Scene | IoU A | IoU B | ΔIoU | FPR A | FPR B | ΔFPR |
|---|---:|---:|---:|---:|---:|---:|
| a4noise | 0.113368 | 0.122391 | +0.009023 | 0.070222 | 0.089090 | +0.018869 |
| poolbottom | 0.181699 | 0.158770 | -0.022929 | 0.257715 | 0.309654 | +0.051938 |
| rack | 0.263333 | 0.565294 | +0.301961 | 0.021755 | 0.008235 | -0.013520 |

## Frozen gates

- [x] foreground_iou_delta_at_least_0.01
- [ ] background_fpr_strictly_lower
- [x] precision_not_lower
- [x] recall_delta_at_least_minus_0.03
- [x] a4noise_iou_delta_at_least_minus_0.01
- [x] a4noise_recall_delta_at_least_minus_0.03

结论：**未通过，停止消融和多 seed**。

Baseline best epoch: 10; Refine best epoch: 9.
Parameters: 46,154,402 → 46,157,666 (+3,264).
Training seconds: 348.94 → 459.43.
Peak reserved GiB: 2.541 → 2.525.