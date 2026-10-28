"""
Missing-aware local costs for Shapelet matching (docs/12 §12.2).

Scale convention (approved):
  alpha = L * softmax(alpha_logits)   # mean weight ≈ 1
  D_used = soft_dtw(C) / L            # comparable across L=4/6/12

Production path (training step core)::

    delta = compute_delta(m)                                 # (B, T)
    costs, valid = cost_for_all_starts_batch(x, m, delta, S, alpha_logits)
    d = shapelet_distances(costs, valid)                     # (B, T-L+1)

API notes
---------
- Batch / sliding APIs take **alpha_logits** and apply ``alpha_from_logits`` inside.
- ``pairwise_cost_matrix`` is low-level and takes **pre-scaled alpha**
  (``alpha_from_logits(...)`` already applied).
- ``warping_radius=None`` always resolves to ``WARPING_RADIUS[L]`` (single-window
  and batch paths are consistent).
- Stage softmin / activation / risk head live in ``shapelet_model`` (not here).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.utils.shapelet_constants import (
    DELTA_MAX,
    GAMMA_SOFTDTW,
    LAMBDA_DELTA,
    LAMBDA_M,
    MIN_OBS_FRAC,
    WARPING_RADIUS,
)
from scripts.utils.soft_dtw import soft_dtw_normalized


def compute_delta(
    mask: torch.Tensor,
    delta_max: float = DELTA_MAX,
) -> torch.Tensor:
    """Hours since last measurement (0 at measured hours).

    Parameters
    ----------
    mask : (..., T)
        1 = measured, 0 = missing.
    delta_max : float
        Cap (default DELTA_MAX=48).

    Returns
    -------
    delta : (..., T)
    """
    m = mask.float()
    T = m.shape[-1]
    device = m.device
    dtype = torch.float32
    last = torch.full(m.shape[:-1], -1, device=device, dtype=torch.long)
    deltas = []
    for t in range(T):
        measured = m[..., t] > 0.5
        d = torch.zeros(m.shape[:-1], device=device, dtype=dtype)
        never = last < 0
        not_m = ~measured
        d = torch.where(
            not_m & never,
            torch.tensor(float(t + 1), device=device, dtype=dtype),
            d,
        )
        d = torch.where(not_m & ~never, (t - last).float().clamp(min=0), d)
        last = torch.where(measured, torch.full_like(last, t), last)
        deltas.append(d)
    return torch.stack(deltas, dim=-1).clamp(max=delta_max)


def _apply_warping_band(cost: torch.Tensor, radius: int) -> torch.Tensor:
    """
    cost: (..., L, L). Set |i-j| > radius cells to large finite barrier.
    """
    L = cost.shape[-1]
    device, dtype = cost.device, cost.dtype
    idx = torch.arange(L, device=device)
    band = (idx[:, None] - idx[None, :]).abs() <= radius
    barrier = torch.tensor(1e4, device=device, dtype=dtype)
    return torch.where(band, cost, barrier)


def _resolve_warping_radius(L: int, warping_radius: int | None) -> int:
    """None → frozen WARPING_RADIUS[L] (default 1 if L missing from map)."""
    if warping_radius is not None:
        return int(warping_radius)
    return int(WARPING_RADIUS.get(int(L), 1))


def pairwise_cost_matrix(
    x_win: torch.Tensor,
    m_win: torch.Tensor,
    d_win: torch.Tensor,
    shapelet: torch.Tensor,
    alpha: torch.Tensor,
    lambda_m: float = LAMBDA_M,
    lambda_delta: float = LAMBDA_DELTA,
    warping_radius: int | None = None,
) -> torch.Tensor:
    """(L,L) local cost for one window vs one shapelet.

    Parameters
    ----------
    x_win, m_win, d_win : (L,)
    shapelet : (L,)
    alpha : (L,)
        **Pre-scaled** weights ``L * softmax(logits)`` (mean ≈ 1).
        Do **not** pass raw logits here — use ``alpha_from_logits`` first,
        or prefer ``cost_batch_windows`` / ``cost_for_all_starts_batch``.
    warping_radius : int or None
        None uses ``WARPING_RADIUS[L]`` (same as batch path).
    """
    L = int(shapelet.shape[0])
    radius = _resolve_warping_radius(L, warping_radius)
    x_safe = torch.where(m_win > 0.5, x_win, torch.zeros_like(x_win))
    xr = x_safe[:, None]
    mr = m_win[:, None].float()
    dr = d_win[:, None].float()
    sj = shapelet[None, :]
    aj = alpha[None, :]
    shape_term = mr * aj * (xr - sj) ** 2
    miss_term = (1.0 - mr) * (
        lambda_m + lambda_delta * torch.log1p(dr.clamp(min=0.0))
    )
    cost = shape_term + miss_term
    return _apply_warping_band(cost, radius)


def alpha_from_logits(alpha_logits: torch.Tensor) -> torch.Tensor:
    """L * softmax so average weight is 1 (scale-stable across lengths)."""
    L = alpha_logits.shape[-1]
    return L * F.softmax(alpha_logits, dim=-1)


def cost_for_all_starts(
    x: torch.Tensor,
    m: torch.Tensor,
    delta: torch.Tensor,
    shapelet: torch.Tensor,
    alpha_logits: torch.Tensor,
    lambda_m: float = LAMBDA_M,
    lambda_delta: float = LAMBDA_DELTA,
    min_obs_frac: float = MIN_OBS_FRAC,
    warping_radius: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sliding-window costs for a single stay (thin wrapper over batch).

    Parameters
    ----------
    x, m, delta : (T,)
    shapelet, alpha_logits : (L,)

    Returns
    -------
    costs : (T-L+1, L, L)
    valid : (T-L+1,) bool — observed fraction ≥ min_obs_frac
    """
    if x.dim() != 1 or m.shape != x.shape or delta.shape != x.shape:
        raise ValueError("x, m, delta must all have shape (T,)")
    costs, valid = cost_for_all_starts_batch(
        x.unsqueeze(0),
        m.unsqueeze(0),
        delta.unsqueeze(0),
        shapelet,
        alpha_logits,
        lambda_m=lambda_m,
        lambda_delta=lambda_delta,
        min_obs_frac=min_obs_frac,
        warping_radius=warping_radius,
    )
    return costs[0], valid[0]


def cost_batch_windows(
    x_batch: torch.Tensor,
    m_batch: torch.Tensor,
    d_batch: torch.Tensor,
    shapelet: torch.Tensor,
    alpha_logits: torch.Tensor,
    lambda_m: float = LAMBDA_M,
    lambda_delta: float = LAMBDA_DELTA,
    warping_radius: int | None = None,
) -> torch.Tensor:
    """Vectorized (N,L,L) costs for already-sliced windows of length L.

    Parameters
    ----------
    x_batch, m_batch, d_batch : (N, L)
    shapelet, alpha_logits : (L,)
        Logits — converted with ``alpha_from_logits`` inside.
    warping_radius : int or None
        None → ``WARPING_RADIUS[L]``.

    Returns
    -------
    cost : (N, L, L)
    """
    if x_batch.dim() != 2 or m_batch.shape != x_batch.shape or d_batch.shape != x_batch.shape:
        raise ValueError("x_batch, m_batch, d_batch must all have shape (N,L)")
    if shapelet.dim() != 1 or alpha_logits.shape != shapelet.shape:
        raise ValueError("shapelet and alpha_logits must both have shape (L,)")
    if x_batch.shape[-1] != shapelet.shape[0]:
        raise ValueError("window length and shapelet length must match")
    radius = _resolve_warping_radius(int(shapelet.shape[0]), warping_radius)
    alpha = alpha_from_logits(alpha_logits)
    x_safe = torch.where(m_batch > 0.5, x_batch, torch.zeros_like(x_batch))
    # (N, L, 1) vs (1, 1, L)
    xr = x_safe[:, :, None]
    mr = m_batch[:, :, None].float()
    dr = d_batch[:, :, None].float()
    sj = shapelet[None, None, :]
    aj = alpha[None, None, :]
    shape_term = mr * aj * (xr - sj) ** 2
    miss_term = (1.0 - mr) * (
        lambda_m + lambda_delta * torch.log1p(dr.clamp(min=0.0))
    )
    cost = shape_term + miss_term
    return _apply_warping_band(cost, radius)


def cost_for_all_starts_batch(
    x: torch.Tensor,
    m: torch.Tensor,
    delta: torch.Tensor,
    shapelet: torch.Tensor,
    alpha_logits: torch.Tensor,
    lambda_m: float = LAMBDA_M,
    lambda_delta: float = LAMBDA_DELTA,
    min_obs_frac: float = MIN_OBS_FRAC,
    warping_radius: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized sliding windows for a stay batch — **main production entry**.

    Parameters
    ----------
    x, m, delta : (B, T)
    shapelet, alpha_logits : (L,)
    min_obs_frac : float
        Window valid iff mean(m_window) ≥ this (default 0.5).

    Returns
    -------
    costs : (B, T-L+1, L, L)
    valid : (B, T-L+1) bool
    """
    if x.dim() != 2 or m.shape != x.shape or delta.shape != x.shape:
        raise ValueError("x, m, delta must all have shape (B,T)")
    L = int(shapelet.shape[0])
    if L > x.shape[-1]:
        raise ValueError(f"shapelet length L={L} exceeds T={x.shape[-1]}")
    x_windows = x.unfold(-1, L, 1)
    m_windows = m.unfold(-1, L, 1)
    d_windows = delta.unfold(-1, L, 1)
    batch_size, n_starts, _ = x_windows.shape
    valid = m_windows.float().mean(dim=-1) >= min_obs_frac
    costs = cost_batch_windows(
        x_windows.reshape(-1, L),
        m_windows.reshape(-1, L),
        d_windows.reshape(-1, L),
        shapelet,
        alpha_logits,
        lambda_m=lambda_m,
        lambda_delta=lambda_delta,
        warping_radius=warping_radius,
    )
    return costs.reshape(batch_size, n_starts, L, L), valid


def shapelet_distances(
    costs: torch.Tensor,
    valid: torch.Tensor,
    gamma: float = GAMMA_SOFTDTW,
    invalid_value: float = 1e4,
) -> torch.Tensor:
    """Soft-DTW / L on cost cubes; invalid windows → ``invalid_value``.

    Parameters
    ----------
    costs : (..., L, L)
    valid : (...,) matching all dims before (L, L)
    gamma : Soft-DTW temperature (default GAMMA_SOFTDTW=0.1)

    Returns
    -------
    distances : (...,)  float32-normalized Soft-DTW
    """
    if costs.dim() < 3 or costs.shape[-1] != costs.shape[-2]:
        raise ValueError("costs must end with (L,L)")
    if valid.shape != costs.shape[:-2]:
        raise ValueError("valid shape must match all cost dimensions before (L,L)")
    L = costs.shape[-1]
    flat = costs.reshape(-1, L, L)
    distances = soft_dtw_normalized(flat, gamma=gamma).reshape(valid.shape)
    return torch.where(valid, distances, torch.full_like(distances, invalid_value))
