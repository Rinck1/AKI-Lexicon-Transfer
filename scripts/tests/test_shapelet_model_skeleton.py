"""G1 skeleton: stage pool + model forward on synthetic data (no patient I/O)."""

from __future__ import annotations

import torch

from scripts.utils.shapelet_model import (
    ShapeletRiskModel,
    activation_from_distance,
    stage_softmin_pool,
)
from scripts.utils.shapelet_prep import make_synthetic_batch, stage_start_masks


def test_stage_softmin_empty_and_normalized():
    # 5 starts: stages cover 0-6 but only n_starts=5 so B1 only partial
    dist = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    masks = stage_start_masks(5)
    d_b, t_star = stage_softmin_pool(dist, masks, tau=0.1)
    assert d_b.shape == (1, 4)
    assert t_star.shape == (1, 4)
    assert torch.isfinite(d_b[0, 0])
    # stages with no starts among 0..4 beyond B1 (0-6) — B2 starts at 6 → empty
    assert d_b[0, 1].item() > 100  # large barrier


def test_activation_scheme_b():
    d = torch.tensor([0.0, 1.0, 2.0])
    q = torch.tensor(1.0)
    a = activation_from_distance(d, q, tau_act=0.5)
    assert a[1].item() == torch.sigmoid(torch.tensor(0.0)).item()
    assert a[0] > a[1] > a[2]


def test_model_forward_backward_synthetic():
    torch.manual_seed(0)
    vars_ = ("urine_output_ml",)
    model = ShapeletRiskModel(vars_=vars_, lengths=(6,), k=2, n_static=11, pos_weight=2.0)
    batch = make_synthetic_batch(batch_size=4, n_vars=1, n_static=11)
    out = model(batch["x"], batch["m"], batch["delta"], batch["static"], batch["y"])
    assert out.activations.shape == (4, 2 * 4)  # K=2, 4 stages
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.shapelets[0].grad is not None
    assert torch.isfinite(model.shapelets[0].grad).all()
    assert torch.isfinite(model.alpha_logits[0].grad).all()


def test_set_q_from_distances_median():
    model = ShapeletRiskModel(
        vars_=("urine_output_ml",), lengths=(6,), k=2, n_static=11
    )
    # two templates, four samples each
    d_flat = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )
    model.set_q_from_distances(d_flat, quantile=0.5)
    assert torch.allclose(model.q, torch.tensor([1.5, 25.0]))


def test_head_ablations_static_and_shapelet():
    torch.manual_seed(0)
    vars_ = ("urine_output_ml",)
    batch = make_synthetic_batch(batch_size=4, n_vars=1, n_static=11)
    m_static = ShapeletRiskModel(
        vars_=vars_, lengths=(6,), k=2, n_static=11, head_mode="static"
    )
    out_s = m_static(
        batch["x"], batch["m"], batch["delta"], batch["static"], batch["y"]
    )
    assert torch.isfinite(out_s.loss)
    out_s.loss.backward()
    assert m_static.theta.grad is not None
    assert m_static.shapelets[0].grad is None  # Soft-DTW path unused

    m_sh = ShapeletRiskModel(
        vars_=vars_, lengths=(6,), k=2, n_static=11, head_mode="shapelet"
    )
    out_h = m_sh(batch["x"], batch["m"], batch["delta"], batch["static"], batch["y"])
    assert torch.isfinite(out_h.loss)
    out_h.loss.backward()
    assert m_sh.shapelets[0].grad is not None
    # theta not in loss graph for shapelet-only
    assert m_sh.theta.grad is None
