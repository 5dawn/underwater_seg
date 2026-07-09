# Experiment Log

Last updated: 2026-07-08

## Current Data State

- Active dataset: `label2`
- Staged raw directory: `data/raw_label2/` (local only, ignored by Git)
- Processed directory: `data/processed_label2/` (local only, ignored by Git)
- Split file committed for reproducibility: `splits_label2.json`
- Label/group manifest committed for review: `label2_group_report.json`
- Split counts: train 23, val 5, test 11
- Label check status: passed
  - no zero-submarine samples
  - no high-ignore samples
  - labels are `-1 ignore`, `0 background`, `1 submarine`

Raw point clouds, processed `.npz` files, checkpoints, and full run logs are not committed to GitHub. They are large local artifacts. GitHub stores code, configs, split definitions, and concise experiment records.

## Network Separation

Each model family has its own config, run directory, and checkpoint prefix.

| Network | Config | Run Dir | Checkpoint Prefix | Status |
| --- | --- | --- | --- | --- |
| PointNet smoke | `configs/label2_smoke.yaml` | `runs/label2_smoke` | `label2_smoke` | completed 3 epoch smoke |
| PointNet++ exp A | `configs/label2_pointnet2_exp_a.yaml` | `runs/label2_pointnet2_exp_a` | `label2_pointnet2_exp_a` | ready for long training |
| PointNet++ exp B | `configs/label2_pointnet2_exp_b.yaml` | `runs/label2_pointnet2_exp_b` | `label2_pointnet2_exp_b` | fallback if exp A remains background-biased |
| DGCNN exp A | `configs/label2_dgcnn_exp_a.yaml` | `runs/label2_dgcnn_exp_a` | `label2_dgcnn_exp_a` | implemented, not yet trained |

Do not reuse run directories between networks. Do not evaluate a checkpoint with a config from another network.

## Key Parameters

PointNet++ exp A:

- model: `pointnet2`
- processed data: `data/processed_label2`
- epochs: 50
- batch size: 4
- points per sample: 4096
- samples per cloud: 32
- sample mode: `foreground_balanced`
- foreground ratio: 0.10
- class weights: `[1.0, 3.0]`
- validation: full-cloud voting, 5 votes
- best metric: `submarine_iou`

PointNet++ exp B:

- same as exp A
- foreground ratio: 0.20
- class weights: `[1.0, 5.0]`

DGCNN exp A:

- model: `dgcnn`
- EdgeConv k: 20
- embedding channels: 512
- same split and training schedule as PointNet++ exp A


## VS Code Entry Point

Open this workspace with the Tool-folder VS Code installation:

```powershell
"E:\Tool\VsCode\Microsoft VS Code\bin\code.cmd" D:\underwater_seg
```

Use `Terminal: Run Task` in VS Code. Run tasks one at a time; do not start multiple long training tasks concurrently.

Recommended task order:

1. `data: inspect label2 labels`
2. `pointnet++: train exp_a`
3. `pointnet++: eval exp_a test`
4. `pointnet++: train exp_b` only if exp A remains background-biased
5. `dgcnn: train exp_a` only after PointNet++ exp A has complete test metrics

## Execution Commands

Label check:

```powershell
E:\condaData\envs\uwseg\python.exe src\inspect_labels.py --processed-dir data\processed_label2
```

PointNet++ exp A training:

```powershell
E:\condaData\envs\uwseg\python.exe src\train.py --config configs\label2_pointnet2_exp_a.yaml
```

PointNet++ exp A test evaluation:

```powershell
E:\condaData\envs\uwseg\python.exe src\eval.py --config configs\label2_pointnet2_exp_a.yaml --checkpoint checkpoints\label2_pointnet2_exp_a_best.pth --split test --num-votes 5 --out runs\label2_pointnet2_exp_a\test_metrics.json
```

PointNet++ exp B training, only if exp A is still strongly background-biased:

```powershell
E:\condaData\envs\uwseg\python.exe src\train.py --config configs\label2_pointnet2_exp_b.yaml
```

DGCNN exp A training, only after PointNet++ exp A has a complete test result:

```powershell
E:\condaData\envs\uwseg\python.exe src\train.py --config configs\label2_dgcnn_exp_a.yaml
```

## Results So Far

PointNet smoke, 3 epochs:

- checkpoint: `checkpoints/label2_smoke_best.pth` (local only)
- test accuracy: 0.8735
- test mIoU: 0.4392
- test submarine IoU: 0.0051
- interpretation: pipeline is valid, but 3 epochs is not enough; model remains mostly background-biased.

PointNet++ exp A:

- previous manual attempt was stopped during epoch 1
- no valid exp A checkpoint has been produced yet
- expected runtime is long because current PointNet++ FPS/ball-query is pure PyTorch

DGCNN exp A:

- implementation compiles
- small forward check passed
- not yet trained

## Next Required Action

Run PointNet++ exp A to completion during a long idle period. After training finishes, run the test evaluation command and append the key metrics here:

- `accuracy`
- `mIoU`
- `submarine_iou`
- `submarine_precision`
- `submarine_recall`
- `submarine_f1`
- worst A4 file
- worst C/pool-bottom file
- worst D3 file

Only after these results exist should exp B or DGCNN be run.
## PointNet++ exp A Completed - 2026-07-08

Training finished for `configs/label2_pointnet2_exp_a.yaml`.

Local artifacts:

- train log: `runs/label2_pointnet2_exp_a/train_console.log`
- train stderr/progress: `runs/label2_pointnet2_exp_a/train_console.err.log`
- best checkpoint: `checkpoints/label2_pointnet2_exp_a_best.pth`
- last checkpoint: `checkpoints/label2_pointnet2_exp_a_last.pth`
- test metrics: `runs/label2_pointnet2_exp_a/test_metrics.json`

Validation summary:

- best validation epoch: 5
- best validation `submarine_iou`: 0.0978
- epoch 50 validation `submarine_iou`: 0.0312
- interpretation: later epochs became conservative on val; best checkpoint is early, not final.

Test summary using best checkpoint:

| metric | value |
| --- | ---: |
| accuracy | 0.4614 |
| mIoU | 0.2630 |
| submarine_iou | 0.0982 |
| submarine_precision | 0.0988 |
| submarine_recall | 0.9458 |
| submarine_f1 | 0.1789 |
| loss | 1.6243 |

Difficult-group summary:

| group | files | mean submarine IoU | mean precision | mean recall | worst file |
| --- | ---: | ---: | ---: | ---: | --- |
| A4 strong laser noise | 7 | 0.0897 | 0.0901 | 0.9218 | `a4noise_R1-D2-P2-S1-A4.npz` |
| C pool-bottom | 2 | 0.1462 | 0.1462 | 1.0000 | `poolbottom_R1-D3-P3-S1-A1.npz` |
| normal D3 | 2 | 0.1281 | 0.1337 | 0.7541 | `normal_R1-D3-P4-S1-A1.npz` |

Decision:

- PointNet++ exp A is substantially better than PointNet smoke (`submarine_iou` 0.0982 vs 0.0051).
- The dominant failure is low precision / many false positives, not missed submarine points.
- Do not run exp B as originally defined without reconsidering it: exp B increases foreground pressure (`foreground_ratio` 0.20, class weight `[1,5]`) and may worsen false positives.
- Next recommended experiment should target precision: reduce false positives, review predicted PLY visualizations, and consider a milder class weight or cleaner validation split before DGCNN comparison.

## PointNet++ exp C Completed - 2026-07-09

Purpose: reduce false positives after exp A by lowering foreground pressure.

Config changes from exp A:

- `foreground_ratio`: 0.05
- `class_weights`: `[1.0, 1.5]`
- run dir: `runs/label2_pointnet2_exp_c/`
- checkpoint prefix: `label2_pointnet2_exp_c`

Local artifacts:

- train log: `runs/label2_pointnet2_exp_c/train_console.log`
- train stderr/progress: `runs/label2_pointnet2_exp_c/train_console.err.log`
- best checkpoint: `checkpoints/label2_pointnet2_exp_c_best.pth`
- last checkpoint: `checkpoints/label2_pointnet2_exp_c_last.pth`
- test metrics: `runs/label2_pointnet2_exp_c/test_metrics.json`

Validation summary:

- training completed all 50 epochs
- best validation epoch: 5
- best validation `submarine_iou`: 0.1358
- epoch 50 validation `submarine_iou`: 0.0309
- interpretation: exp C becomes over-conservative after the early best; later epochs predict too few submarine points.

Test summary using best checkpoint:

| metric | exp A best | exp C best |
| --- | ---: | ---: |
| accuracy | 0.4614 | 0.5389 |
| mIoU | 0.2630 | 0.3112 |
| submarine_iou | 0.0982 | 0.1119 |
| submarine_precision | 0.0988 | 0.1128 |
| submarine_recall | 0.9458 | 0.9370 |
| submarine_f1 | 0.1789 | 0.2013 |
| loss | 1.6243 | 1.1155 |

Difficult-group summary for exp C best:

| group | files | mean submarine IoU | mean precision | mean recall | mean F1 | worst file |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A4 strong laser noise | 7 | 0.1164 | 0.1186 | 0.9071 | 0.2057 | `a4noise_R1-D2-P2-S1-A4.npz` |
| C pool-bottom | 2 | 0.1473 | 0.1482 | 0.9611 | 0.2568 | `poolbottom_R1-D3-P4-S1-A1.npz` |
| normal D3 | 2 | 0.2029 | 0.2172 | 0.7653 | 0.3370 | `normal_R1-D3-P5-S1-A1.npz` |

Decision:

- exp C is modestly better than exp A on test IoU and F1, especially on A4 and normal D3.
- It does not solve the main problem: precision remains low and false positives remain substantial.
- The final checkpoint is not useful; keep selecting by best validation `submarine_iou`.
- Next experiment should not simply reduce foreground pressure further. A better next step is to keep exp C-like mild weights, but make validation cheaper during training and add a precision-oriented loss or post-processing check after inspecting prediction PLYs.
- A stronger GPU would help if CUDA is active, but reducing per-epoch full-cloud 5-vote validation will likely save more time immediately.

## Full label2 Data Refresh - 2026-07-09

The `label2/` directory now contains the completed labeled set and has been restaged/reprocessed.

Condition mapping:

- root `label2/*.ply`: `S1_A1_normal`, staged as `normal_*`, target on stool, 15 files
- `label2/A4/*.ply`: `A4_strong_laser_noise`, staged as `a4noise_*`, target on stool with strong laser noise, 10 files
- `label2/C/*.ply`: `C_pool_bottom`, staged as `poolbottom_*`, target on pool bottom, 14 files
- `label2/J/*.ply`: `J_rack`, staged as `rack_*`, target on rack, R1/R2 x 3 distances x 5 poses, 30 files

Generated artifacts:

- staged raw hardlinks/copies: `data/raw_label2/` (local only)
- processed samples: `data/processed_label2/` (local only)
- split file: `splits_label2.json`
- group/label report: `label2_group_report.json`
- local refresh logs: `runs/label2_data_refresh_2026-07-09/`

Dataset summary:

| item | value |
| --- | ---: |
| total files | 69 |
| train files | 45 |
| val files | 9 |
| test files | 15 |
| ignore labels | 484,284 |
| background labels | 26,752,659 |
| submarine labels | 3,051,607 |

Processed label check:

- train: 45 files, submarine ratio 0.1243, ignore ratio 0.0150
- val: 9 files, submarine ratio 0.0534, ignore ratio 0.0051
- test: 15 files, submarine ratio 0.0592, ignore ratio 0.0222
- zero-submarine samples: none
- high-ignore samples: none

Notes:

- `C_pool_bottom` currently has 14 files; the expected `R1-D3-P5-S1-A1` pool-bottom sample is not present in `label2/C/`.
- The previous PointNet++ exp A/C results were produced on the older 39-file dataset. After this refresh, future experiments should be treated as a new full-label2 dataset run.
- Next recommended training config should point to the same `data/processed_label2`, but use a fresh run directory/checkpoint prefix such as `label2_full_pointnet2_exp_a`.
