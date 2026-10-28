"""Soft-DTW batch vectorization smoke tests."""

from __future__ import annotations

import pytest
import torch

from scripts.utils.shapelet_cost import (
    alpha_from_logits,
    cost_batch_windows,
    cost_for_all_starts_batch,
    pairwise_cost_matrix,
    shapelet_distances,
)
from scripts.utils.soft_dtw import soft_dtw, soft_dtw_batch, soft_dtw_normalized


def _soft_dtw_scalar_reference(cost: torch.Tensor, gamma: float) -> torch.Tensor:
    """Independent scalar dynamic program used only as a correctness oracle."""
    L = cost.shape[0]
    barrier = torch.tensor(1e5, dtype=cost.dtype, device=cost.device)
    rows: list[list[torch.Tensor]] = [
        [barrier.clone() for _ in range(L + 1)] for _ in range(L + 1)
    ]
    rows[0][0] = torch.zeros((), dtype=cost.dtype, device=cost.device)
    for i in range(1, L + 1):
        for j in range(1, L + 1):
            predecessors = torch.stack(
                (rows[i - 1][j - 1], rows[i - 1][j], rows[i][j - 1])
            )
            soft_min = -gamma * torch.logsumexp(-predecessors / gamma, dim=0)
            rows[i][j] = cost[i - 1, j - 1] + soft_min
    return rows[L][L]


def test_soft_dtw_identical_small_distance():
    # zeros cost -> soft-dtw near 0 (not exact due to soft paths)
    L = 4
    c = torch.zeros(L, L)
    d = soft_dtw(c, gamma=0.1)
    assert torch.isfinite(d)
    assert d.item() < 1.0


def test_batch_matches_single():
    torch.manual_seed(0)
    n, L = 5, 6
    c = torch.rand(n, L, L)
    d_batch = soft_dtw_batch(c, gamma=0.1)
    d_reference = torch.stack([_soft_dtw_scalar_reference(c[i], gamma=0.1) for i in range(n)])
    assert torch.allclose(d_batch, d_reference, atol=1e-5)


def test_backward_finite():
    c = torch.rand(8, 6, 6, requires_grad=True)
    d = soft_dtw_batch(c, gamma=0.1).sum()
    d.backward()
    assert c.grad is not None
    assert torch.isfinite(c.grad).all()


def test_normalized_distance_divides_by_length():
    c = torch.rand(3, 6, 6)
    raw = soft_dtw_batch(c, gamma=0.1)
    normalized = soft_dtw_normalized(c, gamma=0.1)
    assert torch.allclose(normalized, raw / 6.0)


def test_batch_windows_apply_minimum_observed_fraction():
    x = torch.zeros(2, 6)
    m = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    delta = torch.zeros_like(x)
    shapelet = torch.zeros(4)
    alpha_logits = torch.zeros(4)
    costs, valid = cost_for_all_starts_batch(
        x, m, delta, shapelet, alpha_logits, min_obs_frac=0.5
    )
    assert costs.shape == (2, 3, 4, 4)
    assert valid[0].all()
    assert not valid[1].any()
    distances = shapelet_distances(costs, valid)
    assert torch.isfinite(distances).all()
    assert torch.all(distances[1] == 1e4)


def test_alpha_has_mean_one():
    alpha = alpha_from_logits(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    assert torch.allclose(alpha.mean(), torch.tensor(1.0))


def test_soft_dtw_rejects_non_float32_and_bad_gamma():
    with pytest.raises(TypeError):
        soft_dtw(torch.zeros(4, 4, dtype=torch.float16))
    with pytest.raises(ValueError):
        soft_dtw(torch.zeros(4, 4), gamma=0.0)


def test_pairwise_and_batch_windows_share_default_warping_band():
    """warping_radius=None must resolve the same way for single-window and batch."""
    torch.manual_seed(1)
    L = 4
    x = torch.randn(L)
    m = torch.ones(L)
    d = torch.zeros(L)
    shapelet = torch.randn(L)
    alpha_logits = torch.zeros(L)
    alpha = alpha_from_logits(alpha_logits)
    single = pairwise_cost_matrix(x, m, d, shapelet, alpha, warping_radius=None)
    batch = cost_batch_windows(
        x.unsqueeze(0),
        m.unsqueeze(0),
        d.unsqueeze(0),
        shapelet,
        alpha_logits,
        warping_radius=None,
    )[0]
    assert torch.allclose(single, batch)
    # Off-band cells use the finite barrier (radius=1 for L=4 → |i-j|>1).
    assert single[0, 3].item() == pytest.approx(1e4)
    assert single[3, 0].item() == pytest.approx(1e4)
