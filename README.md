# Underwater Point Cloud Segmentation Baseline

First-stage baseline for underwater submarine model segmentation:

- Input: `.ply` or `.pcd` point clouds.
- Labels: `-1 ignore`, `0 background`, `1 submarine`.
- Intermediate format: `.npz` with `points`, `features`, `labels`.
- Model: compact PointNet segmentation baseline.

## Environment

```bash
conda create -n uwseg python=3.10
conda activate uwseg
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

If you do not have an NVIDIA GPU, install the CPU version of PyTorch from the official PyTorch instructions.

## Data Layout

Put labeled point clouds under:

```text
data/raw/
```

For the current 13 labeled clouds, keep the first experiment simple:

```text
train: 8-9 clouds
val:   2 clouds
test:  2-3 clouds
```

Put the hardest and most different samples in `test`: farther range, larger view-angle changes, lower point density, turbidity, occlusion, or stronger scattering.

Supported labels:

- PLY/ASCII PCD field names such as `label`, `labels`, `class`, `class_id`, `seg_label`, `semantic`.
- Or sidecar files next to the cloud: `<stem>_labels.npy`, `<stem>_labels.txt`, `<stem>.labels.npy`, `<stem>.labels.txt`.

CloudCompare label convention:

```text
-1 ignore      uncertain boundary points, scattering outliers, occluded or ambiguous points
 0 background  pool floor, pool wall, supports, and clearly non-submarine points
 1 submarine   points clearly on the submarine surface
```

Training and evaluation ignore label `-1` in loss and metrics. Do not encode uncertain points as background, because that teaches the model that ambiguous submarine-boundary returns are negative examples.

## Preprocess

Use xyz only with a reproducible split file:

```bash
python src/preprocess.py --raw-dir data/raw --out-dir data/processed --features xyz --split-file splits_13.txt
```

The split file can be plain text:

```text
train sample_01.ply
train sample_02.ply
val sample_10.ply
test sample_12.ply
```

or JSON:

```json
{
  "train": ["sample_01.ply", "sample_02.ply"],
  "val": ["sample_10.ply", "sample_11.ply"],
  "test": ["sample_12.ply", "sample_13.ply"]
}
```

Every raw cloud must be assigned exactly once. Entries may use filename, stem, or path relative to `data/raw`.

If you intentionally want automatic random splitting, omit `--split-file`:

```bash
python src/preprocess.py --raw-dir data/raw --out-dir data/processed --features xyz
```

Use xyz plus range and intensity:

```bash
python src/preprocess.py --raw-dir data/raw --out-dir data/processed --features xyz,intensity,range --split-file splits_13.txt
```

Preprocessing validates that labels contain only `-1`, `0`, and `1`. It also writes `metadata.json` with label mapping, per-split label statistics, and feature behavior. Saved `points` are normalized xyz coordinates; the optional `range` feature is computed from the raw input coordinates before centering/scaling and then normalized by the per-cloud maximum range.

## Inspect Labels

Before training, inspect the processed labels:

```bash
python src/inspect_labels.py --processed-dir data/processed
```

This reports per-file and per-split counts for ignore/background/submarine points, flags samples with zero submarine points, and flags samples whose ignore ratio is high. The default high-ignore threshold is `0.3`; override it with:

```bash
python src/inspect_labels.py --processed-dir data/processed --ignore-threshold 0.3
```

## Train

The default config is a 30 epoch smoke test with `best_metric: submarine_iou` and `sample_mode: random`:

```bash
python src/train.py --config configs/pointnet.yaml
```

Ignore points may appear in a sampled crop, but they do not contribute to the loss. Foreground-balanced sampling exists in the dataset code and can be enabled later by changing `sample_mode` to `foreground_balanced`; it is not required for the first smoke test.

For the v2 baseline comparison, run these three fixed-split xyz experiments:

```bash
python src/train.py --config configs/pointnet_exp_a.yaml
python src/train.py --config configs/pointnet_exp_b.yaml
python src/train.py --config configs/pointnet_exp_c.yaml
```

- `pointnet_exp_a`: random sampling, no class weight.
- `pointnet_exp_b`: foreground-balanced sampling, no class weight.
- `pointnet_exp_c`: foreground-balanced sampling, mild class weight `[1, 3]`.

Each experiment uses `samples_per_cloud: 32`, so one epoch has more training updates than the first smoke test.

For the v3 comparison, experiments D-G use full-cloud voting validation for checkpoint selection:

```bash
python src/train.py --config configs/pointnet_exp_d.yaml
python src/train.py --config configs/pointnet_exp_e.yaml
python src/train.py --config configs/pointnet_exp_f.yaml
python src/train.py --config configs/pointnet_exp_g.yaml
```

- `pointnet_exp_d`: foreground ratio 0.10, class weight `[1, 3]`.
- `pointnet_exp_e`: foreground ratio 0.15, class weight `[1, 3]`.
- `pointnet_exp_f`: foreground ratio 0.10, class weight `[1, 5]`.
- `pointnet_exp_g`: foreground ratio 0.15, class weight `[1, 5]`.

Best checkpoint:

```text
checkpoints/pointnet_best.pth
```

## Evaluate

```bash
python src/eval.py --config configs/pointnet.yaml --checkpoint checkpoints/pointnet_best.pth --split test --num-votes 10
```

Evaluation uses repeated voting by default. Each cloud is sampled multiple times, logits are accumulated back to the original points, and metrics are computed on the full cloud while ignoring `-1`. The key first-stage metrics are `submarine_iou`, `submarine_precision`, `submarine_recall`, `submarine_f1`, and `mIoU`.

## Split Discipline

Do not randomly mix adjacent frames from the same acquisition sequence across train/val/test. Neighboring frames often share nearly identical geometry, viewpoint, water condition, and labeling artifacts. If adjacent frames leak into the test split, the reported IoU can look good while the model has only memorized one sequence. Prefer splitting by acquisition run, distance, view angle, water turbidity, and submarine pose.

## Visualize

Ground truth:

```bash
python src/visualize.py --npz data/processed/test/example.npz
```

Prediction:

```bash
python src/visualize.py --npz data/processed/test/example.npz --checkpoint checkpoints/pointnet_best.pth
```

Save colored prediction:

```bash
python src/visualize.py --npz data/processed/test/example.npz --checkpoint checkpoints/pointnet_best.pth --save-ply runs/prediction.ply
```
