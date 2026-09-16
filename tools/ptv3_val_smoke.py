from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def backbone(cls):
    return cls(in_channels=3, enc_patch_size=(128,) * 5, dec_patch_size=(128,) * 4,
               enable_flash=False, enable_rpe=False, upcast_attention=True, upcast_softmax=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True); p.add_argument("--official-parent", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--out", required=True)
    args = p.parse_args(); project = Path(args.project)
    sys.path[:0] = [str(project / "src"), args.official_parent]
    from model_ptv3 import OfficialPTv3Seg
    from ptv3_full_eval import evaluate_ptv3_full_cloud
    from PointTransformerV3.model import PointTransformerV3

    device = torch.device("cuda")
    files = sorted((project / "data/processed_label2/val").glob("*.npz"))
    if len(files) != 9: raise RuntimeError(f"expected 9 val files, got {len(files)}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = OfficialPTv3Seg(backbone(PointTransformerV3), 64, 2, 0.005).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    result = evaluate_ptv3_full_cloud(model, files, device, Path(args.out).parent / "vote_counts", eval_seed=1001)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("foreground_iou", "foreground_precision", "foreground_recall", "background_fpr", "votes", "eval_seed")}, indent=2))


if __name__ == "__main__": main()
