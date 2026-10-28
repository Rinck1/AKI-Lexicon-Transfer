"""Hard-Concrete / L0 utilities (Louizos et al. 2018 stretch params)."""

from __future__ import annotations

import math

import torch

from scripts.utils.hard_concrete import (
    HC_BETA,
    HC_GAMMA,
    HC_ZETA,
    hard_concrete_l0_prob,
    hard_concrete_sample,
    hard_mask_from_log_alpha,
    masked_pearson,
)


def test_l0_prob_at_default_init_is_about_0_2():
    pi = hard_concrete_l0_prob(torch.tensor([-3.0]))
    offset = HC_BETA * math.log(-HC_GAMMA / HC_ZETA)
    expect = torch.sigmoid(torch.tensor(-3.0 - offset))
    assert torch.allclose(pi, expect, atol=1e-6)
    assert 0.15 < float(pi.item()) < 0.25


def test_eval_sample_is_deterministic_and_in_unit_interval():
    torch.manual_seed(0)
    la = torch.linspace(-6, 6, 13)
    z1 = hard_concrete_sample(la, training=False)
    z2 = hard_concrete_sample(la, training=False)
    assert torch.equal(z1, z2)
    assert torch.all((z1 >= 0) & (z1 <= 1))


def test_train_sample_is_stochastic_and_in_unit_interval():
    torch.manual_seed(1)
    la = torch.zeros(256)
    z = hard_concrete_sample(la, training=True)
    assert torch.all((z >= 0) & (z <= 1))
    assert float(z.std()) > 0.05


def test_hard_mask_is_binary_and_threshold_matches_stretch():
    la = torch.tensor([-3.0, 0.0, 5.0])
    z = hard_mask_from_log_alpha(la)
    assert set(z.tolist()) <= {0.0, 1.0}
    s = torch.sigmoid(la)
    stretched = s * (HC_ZETA - HC_GAMMA) + HC_GAMMA
    assert torch.equal(z, (stretched > 0).to(la.dtype))
    assert float(z[0]) == 0.0
    assert float(z[2]) == 1.0


def test_masked_pearson_identical_opposite_and_tiny_mask():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([2.0, 4.0, 6.0, 8.0])
    mask = torch.ones(4, dtype=torch.bool)
    assert torch.allclose(masked_pearson(a, b, mask), a.new_ones(()), atol=1e-5)
    assert torch.allclose(masked_pearson(a, -b, mask), -a.new_ones(()), atol=1e-5)
    tiny = torch.tensor([True, True, False, False])
    assert float(masked_pearson(a, b, tiny).item()) == 1.0
