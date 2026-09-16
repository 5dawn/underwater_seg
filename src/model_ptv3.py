from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch
from torch import nn

from ptv3_protocol import UnderwaterLocalFeatureRefine, voxelize_xyz


def _extract_features(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict) and "feat" in output:
        return output["feat"]
    if hasattr(output, "feat"):
        return output.feat
    raise TypeError("Official PTv3 output must be a tensor or expose `feat`")


class OfficialPTv3Seg(nn.Module):
    """Thin adapter: official PTv3 core + local inverse map/refine/head."""

    def __init__(
        self,
        backbone: nn.Module,
        decoder_channels: int,
        num_classes: int = 2,
        voxel_size: float = 0.005,
        refine: UnderwaterLocalFeatureRefine | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.voxel_size = float(voxel_size)
        self.refine = refine
        self.classifier = nn.Linear(decoder_channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != 3:
            raise ValueError("PTv3 baseline expects XYZ in [B, 3, N]")
        points = inputs.transpose(1, 2).contiguous()
        voxelized = [voxelize_xyz(sample, self.voxel_size) for sample in points]
        counts = torch.tensor([len(item.coord) for item in voxelized], device=inputs.device)
        offset = counts.cumsum(0).to(torch.int32)
        batch = torch.repeat_interleave(torch.arange(len(points), device=inputs.device), counts)
        payload = {
            "coord": torch.cat([item.coord for item in voxelized]),
            "feat": torch.cat([item.feat for item in voxelized]),
            "grid_coord": torch.cat([item.grid_coord for item in voxelized]),
            "batch": batch,
            "offset": offset,
        }
        voxel_features = _extract_features(self.backbone(payload))
        recovered = []
        cursor = 0
        for item in voxelized:
            recovered.append(voxel_features[cursor : cursor + len(item.coord)][item.inverse])
            cursor += len(item.coord)
        decoded = torch.stack(recovered)
        if self.refine is not None:
            decoded = self.refine(decoded, points)
        return self.classifier(decoded).transpose(1, 2).contiguous()


def paired_models(
    backbone_factory: Callable[[], nn.Module],
    decoder_channels: int,
    voxel_size: float,
    radius: float,
    public_seed: int,
    refine_seed: int,
) -> tuple[OfficialPTv3Seg, OfficialPTv3Seg]:
    """Create A/B from one public snapshot without perturbing the global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(public_seed)
        baseline = OfficialPTv3Seg(backbone_factory(), decoder_channels, voxel_size=voxel_size)
    common = deepcopy(baseline.state_dict())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(refine_seed)
        refined = OfficialPTv3Seg(
            backbone_factory(), decoder_channels, voxel_size=voxel_size,
            refine=UnderwaterLocalFeatureRefine(decoder_channels, radius),
        )
    refined.load_state_dict(common, strict=False)
    return baseline, refined
