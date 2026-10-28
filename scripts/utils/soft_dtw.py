"""
Differentiable Soft-DTW — **batched / vectorized** implementation.

The previous per-matrix Python loop is unusable on RTX 4070 for full training
(~minutes per step). This version runs Soft-DTW over N cost matrices in parallel
on GPU with only L×L sequential steps (L is small: 4/6/12).
"""

from __future__ import annotations

import torch


def softmin3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """Elementwise softmin of three tensors."""
    stacked = torch.stack((a, b, c), dim=0)
    return -gamma * torch.logsumexp(-stacked / gamma, dim=0)


def soft_dtw(cost: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """
    Soft-DTW for one or many cost matrices.

    Parameters
    ----------
    cost : (L, L) or (N, L, L)
    gamma : Soft-DTW temperature > 0

    Returns
    -------
    scalar if (L,L), else (N,)
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if cost.dtype != torch.float32:
        raise TypeError(
            f"Soft-DTW must run in float32, got {cost.dtype}; "
            "disable autocast for the DP block"
        )
    if cost.dim() == 2:
        return soft_dtw_batch(cost.unsqueeze(0), gamma=gamma)[0]
    if cost.dim() == 3:
        return soft_dtw_batch(cost, gamma=gamma)
    raise ValueError(f"cost must be (L,L) or (N,L,L), got {tuple(cost.shape)}")


def soft_dtw_batch(cost: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """
    Batched Soft-DTW.

    Parameters
    ----------
    cost : (N, L, L) local costs
    gamma : temperature

    Returns
    -------
    d : (N,) Soft-DTW values (may be negative for classical Soft-DTW)
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if cost.dtype != torch.float32:
        raise TypeError(
            f"Soft-DTW must run in float32, got {cost.dtype}; "
            "disable autocast for the DP block"
        )
    if cost.dim() != 3 or cost.shape[1] != cost.shape[2]:
        raise ValueError(f"cost must be (N,L,L), got {tuple(cost.shape)}")
    n, L, _ = cost.shape
    device, dtype = cost.device, cost.dtype
    # Large finite barrier (avoid inf for autograd)
    barrier = torch.tensor(1e5, device=device, dtype=dtype)
    R = barrier.expand(n, L + 1, L + 1).clone()
    R[:, 0, 0] = 0.0

    for i in range(1, L + 1):
        for j in range(1, L + 1):
            r0 = R[:, i - 1, j - 1]
            r1 = R[:, i - 1, j]
            r2 = R[:, i, j - 1]
            R[:, i, j] = cost[:, i - 1, j - 1] + softmin3(r0, r1, r2, gamma)
    return R[:, L, L]


def soft_dtw_normalized(cost: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Shapelet distance convention: raw Soft-DTW divided by template length L."""
    if cost.dim() not in (2, 3):
        raise ValueError(f"cost must be (L,L) or (N,L,L), got {tuple(cost.shape)}")
    L = cost.shape[-1]
    return soft_dtw(cost, gamma=gamma) / float(L)


# Backward-compatible name used by earlier drafts
def soft_dtw_batch_starts(cost_btll: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Alias: cost (N,L,L) -> (N,)."""
    return soft_dtw_batch(cost_btll, gamma=gamma)
