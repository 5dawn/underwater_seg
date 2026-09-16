from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_ptv3 import OfficialPTv3Seg, paired_models
from ptv3_protocol import UnderwaterLocalFeatureRefine, choose_voxel_size, voxelize_xyz


class IdentityBackbone(nn.Module):
    def forward(self, payload):
        return payload["feat"]


def test_voxel_inverse_and_duplicate_mean():
    points = torch.tensor([[0.001, 0, 0], [0.002, 0, 0], [0.02, 0, 0]], dtype=torch.float32)
    result = voxelize_xyz(points, 0.01)
    assert result.counts.tolist() == [2, 1]
    assert result.inverse.tolist() == [0, 0, 1]
    assert torch.allclose(result.feat[0], torch.tensor([0.0015, 0, 0]))


def test_mixed_batch_never_crosses_voxelization():
    model = OfficialPTv3Seg(IdentityBackbone(), 3, voxel_size=0.1)
    points = torch.tensor([[[0.01, 0.02], [0, 0], [0, 0]], [[0.01, 0.02], [0, 0], [0, 0]]])
    output = model(points)
    assert output.shape == (2, 2, 2)
    assert torch.allclose(output[0], output[1])


def test_refine_zero_init_identity_and_gradients():
    refine = UnderwaterLocalFeatureRefine(8, radius=10.0, chunk_size=3)
    features = torch.randn(2, 7, 8, requires_grad=True)
    points = torch.randn(2, 7, 3)
    output = refine(features, points)
    assert torch.equal(output, features)
    output.square().sum().backward()
    assert refine.w2.weight.grad is not None
    assert torch.isfinite(refine.w2.weight.grad).all()
    assert refine.w2.weight.grad.abs().sum() > 0
    assert torch.equal(refine.w1.weight.grad, torch.zeros_like(refine.w1.weight.grad))
    with torch.no_grad():
        refine.w2.weight.fill_(0.01)
    refine.zero_grad(set_to_none=True)
    refine(features, points).square().sum().backward()
    assert refine.w1.weight.grad.abs().sum() > 0


def test_fewer_than_three_neighbors_has_zero_residual():
    refine = UnderwaterLocalFeatureRefine(4, radius=0.01)
    nn.init.ones_(refine.w2.weight)
    nn.init.ones_(refine.w2.bias)
    features = torch.randn(1, 3, 4)
    points = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]])
    assert torch.equal(refine(features, points), features)


def test_paired_public_weights_and_global_rng_isolation():
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    a, b = paired_models(IdentityBackbone, 3, 0.01, 1.0, 42, 1042)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert torch.equal(a.classifier.weight, b.classifier.weight)
    assert torch.equal(a.classifier.bias, b.classifier.bias)


def test_frozen_voxel_rule():
    assert choose_voxel_size([{"unique_voxel_ratio": 0.2, "single_point_voxel_ratio": 0.1}])[0] == 0.0025
    assert choose_voxel_size([{"unique_voxel_ratio": 0.96, "single_point_voxel_ratio": 0.91}])[0] == 0.01
    assert choose_voxel_size([{"unique_voxel_ratio": 0.5, "single_point_voxel_ratio": 0.5}])[0] == 0.005


def test_input_reordering_is_equivariant_and_checkpoint_strictly_reloads():
    torch.manual_seed(9)
    model = OfficialPTv3Seg(IdentityBackbone(), 3, voxel_size=0.1)
    points = torch.randn(1, 3, 11)
    permutation = torch.randperm(11)
    original = model(points)
    reordered = model(points[:, :, permutation])
    assert torch.allclose(reordered, original[:, :, permutation], atol=1e-7, rtol=0)
    clone = OfficialPTv3Seg(IdentityBackbone(), 3, voxel_size=0.1)
    clone.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(clone(points), original)
