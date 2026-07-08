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
