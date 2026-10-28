"""Hard-Concrete / L0 gates (Louizos et al., 2018)."""

from __future__ import annotations

import math

import torch

# Standard stretch parameters: support slightly outside [0, 1] then clamp.
HC_BETA = 2.0 / 3.0
HC_GAMMA = -0.1
HC_ZETA = 1.1


def hard_concrete_sample(
    log_alpha: torch.Tensor,
    *,
    training: bool,
    beta: float = HC_BETA,
    gamma: float = HC_GAMMA,
    zeta: float = HC_ZETA,
) -> torch.Tensor:
    """Differentiable sample in train; deterministic stretch at eval (still in [0, 1])."""
    if training:
        u = torch.rand_like(log_alpha).clamp(1e-6, 1.0 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / float(beta))
    else:
        s = torch.sigmoid(log_alpha)
    z = s * (float(zeta) - float(gamma)) + float(gamma)
    return z.clamp(0.0, 1.0)


def hard_concrete_l0_prob(
    log_alpha: torch.Tensor,
    *,
    beta: float = HC_BETA,
    gamma: float = HC_GAMMA,
    zeta: float = HC_ZETA,
) -> torch.Tensor:
    """P(z != 0) under the stretched concrete (complexity / selection probability)."""
    # sigmoid(log_alpha - beta * log(-gamma / zeta))
    offset = float(beta) * math.log(-float(gamma) / float(zeta))
    return torch.sigmoid(log_alpha - offset)


def hard_mask_from_log_alpha(
    log_alpha: torch.Tensor,
    *,
    gamma: float = HC_GAMMA,
    zeta: float = HC_ZETA,
) -> torch.Tensor:
    """Deterministic 0/1 mask used at inference."""
    s = torch.sigmoid(log_alpha)
    stretched = s * (float(zeta) - float(gamma)) + float(gamma)
    return (stretched > 0.0).to(log_alpha.dtype)


def masked_pearson(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Pearson corr over masked entries; 1 if both constant (stable), else in [-1, 1]."""
    m = mask.to(dtype=torch.bool)
    if int(m.sum().item()) < 3:
        return a.new_ones(())
    x = a[m]
    y = b[m]
    x = x - x.mean()
    y = y - y.mean()
    nx = torch.linalg.vector_norm(x)
    ny = torch.linalg.vector_norm(y)
    if float(nx.detach()) < 1e-6 and float(ny.detach()) < 1e-6:
        return a.new_ones(())
    if float(nx.detach()) < 1e-6 or float(ny.detach()) < 1e-6:
        return a.new_zeros(())
    return (x * y).sum() / (nx * ny)
