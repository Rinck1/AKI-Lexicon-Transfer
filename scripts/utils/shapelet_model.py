"""
Shapelet bank + stage pooling + risk head (docs/12 §7–§8).

Learnable: S, alpha_logits, beta, theta, bias.
Frozen hyperparams: gamma, tau_pool, tau_act, q, lambdas (from shapelet_constants).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.utils.shapelet_constants import (
    GAMMA_SOFTDTW,
    LAMBDA_DIV,
    LAMBDA_SP,
    LAMBDA_TIME,
    PILOT_K,
    PILOT_LENGTHS,
    SHAPELET_STATIC_COLS,
    SHAPELET_VARS,
    STAGE_BOUNDS,
    TAU_ACTIVATION,
    TAU_POOL,
    T_HOURS,
)
from scripts.utils.shapelet_cost import cost_for_all_starts_batch, shapelet_distances
from scripts.utils.shapelet_prep import stage_start_masks


def stage_softmin_pool(
    distances: torch.Tensor,
    stage_masks: torch.Tensor,
    tau: float = TAU_POOL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized softmin over starts within each stage.

    Parameters
    ----------
    distances : (B, n_starts)
    stage_masks : (n_stages, n_starts) bool
    tau : pool temperature

    Returns
    -------
    d_stage : (B, n_stages)
        -tau * log(mean_{t in Tb} exp(-D(t)/tau)); empty stage → large distance.
    t_star : (B, n_stages)
        Soft-argmin start index (0 if stage empty).
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    bsz, n_starts = distances.shape
    n_stages = stage_masks.shape[0]
    device, dtype = distances.device, distances.dtype
    large = torch.tensor(1e4, device=device, dtype=dtype)
    d_out = []
    t_out = []
    starts = torch.arange(n_starts, device=device, dtype=dtype)
    for s in range(n_stages):
        mask = stage_masks[s]  # (n_starts,)
        n_tb = int(mask.sum().item())
        if n_tb == 0:
            d_out.append(large.expand(bsz))
            t_out.append(torch.zeros(bsz, device=device, dtype=dtype))
            continue
        # invalid starts outside stage: push to +inf for softmin
        d = distances.masked_fill(~mask.unsqueeze(0), large)
        # normalized softmin: -tau log( mean exp(-d/tau) ) over stage cells only
        # mean over |Tb| using only stage positions
        neg = (-d / tau).masked_fill(~mask.unsqueeze(0), torch.finfo(dtype).min / 4)
        # logsumexp over starts, then subtract log(|Tb|)
        lse = torch.logsumexp(neg, dim=-1)
        d_b = -tau * (lse - torch.log(torch.tensor(float(n_tb), device=device, dtype=dtype)))
        d_out.append(d_b)
        # soft-argmin weights on stage
        w = torch.softmax(neg, dim=-1)  # mass outside stage ~ 0 due to min fill
        t_star = (w * starts.unsqueeze(0)).sum(dim=-1)
        t_out.append(t_star)
    return torch.stack(d_out, dim=-1), torch.stack(t_out, dim=-1)


def activation_from_distance(
    d: torch.Tensor,
    q: torch.Tensor,
    tau_act: float = TAU_ACTIVATION,
) -> torch.Tensor:
    """Scheme B: a = sigmoid((q - d) / tau_act)."""
    return torch.sigmoid((q - d) / tau_act)


@dataclass
class ForwardOut:
    logits: torch.Tensor
    prob: torch.Tensor
    activations: torch.Tensor  # (B, n_act)
    d_stages: torch.Tensor  # (B, n_templates, n_stages)
    t_star: torch.Tensor  # (B, n_templates, n_stages)
    loss_bce: torch.Tensor
    loss_div: torch.Tensor
    loss_time: torch.Tensor
    loss_sp: torch.Tensor
    loss: torch.Tensor


class ShapeletRiskModel(nn.Module):
    """Multi-variable multi-scale shapelet bank + linear risk head."""

    def __init__(
        self,
        vars_: tuple[str, ...] = SHAPELET_VARS,
        lengths: tuple[int, ...] = PILOT_LENGTHS,
        k: int = PILOT_K,
        n_static: int = len(SHAPELET_STATIC_COLS),
        t_hours: int = T_HOURS,
        gamma: float = GAMMA_SOFTDTW,
        tau_pool: float = TAU_POOL,
        tau_act: float = TAU_ACTIVATION,
        pos_weight: float = 1.0,
        q_init: float = 0.0,
        head_mode: str = "combined",
    ):
        super().__init__()
        head_mode = head_mode.lower()
        if head_mode not in ("combined", "static", "shapelet"):
            raise ValueError(
                f"head_mode must be combined|static|shapelet, got {head_mode!r}"
            )
        self.head_mode = head_mode
        self.vars = tuple(vars_)
        self.lengths = tuple(int(x) for x in lengths)
        self.k = int(k)
        self.n_static = int(n_static)
        self.t_hours = int(t_hours)
        self.gamma = float(gamma)
        self.tau_pool = float(tau_pool)
        self.tau_act = float(tau_act)
        self.n_stages = len(STAGE_BOUNDS)

        # templates: list aligned with flat index (var_i, L, k)
        self._keys: list[tuple[str, int, int]] = []
        self.shapelets = nn.ParameterList()
        self.alpha_logits = nn.ParameterList()
        for v in self.vars:
            for L in self.lengths:
                for ki in range(self.k):
                    self._keys.append((v, L, ki))
                    # small random init
                    self.shapelets.append(nn.Parameter(0.1 * torch.randn(L)))
                    self.alpha_logits.append(nn.Parameter(torch.zeros(L)))

        self.n_templates = len(self._keys)
        self.n_act = self.n_templates * self.n_stages
        # q per template (not per stage); buffer — not trained by default
        self.register_buffer("q", torch.full((self.n_templates,), float(q_init)))

        self.beta = nn.Parameter(torch.zeros(self.n_act))
        self.theta = nn.Parameter(torch.zeros(self.n_static))
        self.bias = nn.Parameter(torch.zeros(()))

        pw = torch.tensor([float(pos_weight)], dtype=torch.float32)
        self.register_buffer("pos_weight", pw)

        # cache stage masks per L
        self._stage_mask_cache: dict[int, torch.Tensor] = {}

    def template_keys(self) -> list[tuple[str, int, int]]:
        return list(self._keys)

    def activation_names(self) -> list[str]:
        names = []
        for v, L, ki in self._keys:
            for s_name, _lo, _hi in STAGE_BOUNDS:
                names.append(f"{v}|L{L}|k{ki}|{s_name}")
        return names

    def _stage_masks(self, L: int, device: torch.device) -> torch.Tensor:
        n_starts = self.t_hours - L + 1
        key = L
        if key not in self._stage_mask_cache or self._stage_mask_cache[key].device != device:
            self._stage_mask_cache[key] = stage_start_masks(n_starts, device=device)
        return self._stage_mask_cache[key]

    def set_q_from_distances(self, d_flat: torch.Tensor, quantile: float = 0.5) -> None:
        """Set q from train distance samples. d_flat: (n_templates, n_samples)."""
        if d_flat.dim() != 2 or d_flat.shape[0] != self.n_templates:
            raise ValueError("d_flat must be (n_templates, n_samples)")
        qs = []
        for i in range(self.n_templates):
            row = d_flat[i].float()
            if row.numel() == 0:
                qs.append(torch.zeros((), device=self.q.device, dtype=self.q.dtype))
            else:
                qs.append(torch.quantile(row, quantile))
        self.q.copy_(torch.stack(qs).to(device=self.q.device, dtype=self.q.dtype))

    def freeze_templates(self, freeze: bool = True) -> None:
        """Freeze/unfreeze shapelet S and alpha logits (for q-calibration phase)."""
        for p in self.shapelets.parameters():
            p.requires_grad = not freeze
        for p in self.alpha_logits.parameters():
            p.requires_grad = not freeze

    def head_parameter_groups(self, lr_head: float, weight_decay: float) -> list[dict]:
        """Only risk-head params (β / θ / bias depending on head_mode)."""
        if self.head_mode == "static":
            params = [self.theta, self.bias]
        elif self.head_mode == "shapelet":
            params = [self.beta, self.bias]
        else:
            params = [self.beta, self.theta, self.bias]
        return [{"params": params, "lr": lr_head, "weight_decay": weight_decay}]

    def match_one_template(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        delta: torch.Tensor,
        template_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match one template on one variable channel.

        x,m,delta: (B, T) for that variable.
        Returns d_stage (B, n_stages), t_star (B, n_stages), distances (B, n_starts).
        """
        S = self.shapelets[template_idx]
        a_log = self.alpha_logits[template_idx]
        L = int(S.shape[0])
        costs, valid = cost_for_all_starts_batch(x, m, delta, S, a_log)
        dist = shapelet_distances(costs, valid, gamma=self.gamma)
        masks = self._stage_masks(L, device=x.device)
        d_stage, t_star = stage_softmin_pool(dist, masks, tau=self.tau_pool)
        return d_stage, t_star, dist

    def forward_activations(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x,m,delta: (B, T, V)
        returns a (B, n_act), d_stages (B, n_tmpl, n_stages), t_star same.
        """
        var_index = {v: i for i, v in enumerate(self.vars)}
        d_list = []
        t_list = []
        a_list = []
        for ti, (v, _L, _k) in enumerate(self._keys):
            j = var_index[v]
            d_stage, t_star, _dist = self.match_one_template(
                x[:, :, j], m[:, :, j], delta[:, :, j], ti
            )
            q_i = self.q[ti]
            a_stage = activation_from_distance(d_stage, q_i, tau_act=self.tau_act)
            # empty-stage: d was large → a ~ 0 already; keep
            d_list.append(d_stage)
            t_list.append(t_star)
            a_list.append(a_stage)
        d_stages = torch.stack(d_list, dim=1)
        t_star = torch.stack(t_list, dim=1)
        # flatten (tmpl, stage) → feature dim
        a = torch.stack(a_list, dim=1).reshape(x.shape[0], self.n_act)
        return a, d_stages, t_star

    def diversity_loss(self) -> torch.Tensor:
        """Penalize cosine similarity among k templates of same (var, L)."""
        device = self.shapelets[0].device
        total = torch.zeros((), device=device)
        n_pairs = 0
        # group indices by (var, L)
        groups: dict[tuple[str, int], list[int]] = {}
        for i, (v, L, _k) in enumerate(self._keys):
            groups.setdefault((v, L), []).append(i)
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            mats = [F.normalize(self.shapelets[i], dim=0) for i in idxs]
            for a in range(len(mats)):
                for b in range(a + 1, len(mats)):
                    total = total + (mats[a] * mats[b]).sum().pow(2)
                    n_pairs += 1
        if n_pairs == 0:
            return torch.zeros((), device=device)
        return total / n_pairs

    def time_smooth_loss(self) -> torch.Tensor:
        """Sum_b (beta_{b} - beta_{b-1})^2 over templates."""
        beta = self.beta.view(self.n_templates, self.n_stages)
        if self.n_stages < 2:
            return beta.new_zeros(())
        diff = beta[:, 1:] - beta[:, :-1]
        return diff.pow(2).mean()

    def sparsity_loss(self) -> torch.Tensor:
        return self.beta.abs().mean()

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        delta: torch.Tensor,
        static: torch.Tensor,
        y: torch.Tensor | None = None,
        lambda_div: float = LAMBDA_DIV,
        lambda_time: float = LAMBDA_TIME,
        lambda_sp: float = LAMBDA_SP,
    ) -> ForwardOut:
        bsz = x.shape[0]
        device, dtype = x.device, x.dtype
        # Ablations: static skips Soft-DTW; shapelet skips static linear term.
        if self.head_mode == "static":
            a = torch.zeros(bsz, self.n_act, device=device, dtype=dtype)
            d_stages = torch.zeros(
                bsz, self.n_templates, self.n_stages, device=device, dtype=dtype
            )
            t_star = torch.zeros_like(d_stages)
            logits = self.bias + (static * self.theta).sum(dim=-1)
        else:
            a, d_stages, t_star = self.forward_activations(x, m, delta)
            logits = self.bias + (a * self.beta).sum(dim=-1)
            if self.head_mode == "combined":
                logits = logits + (static * self.theta).sum(dim=-1)
            # head_mode == "shapelet": activations only
        prob = torch.sigmoid(logits)

        if y is None:
            zero = logits.new_zeros(())
            return ForwardOut(
                logits=logits,
                prob=prob,
                activations=a,
                d_stages=d_stages,
                t_star=t_star,
                loss_bce=zero,
                loss_div=zero,
                loss_time=zero,
                loss_sp=zero,
                loss=zero,
            )

        bce = F.binary_cross_entropy_with_logits(
            logits, y.float(), pos_weight=self.pos_weight
        )
        if self.head_mode == "static":
            l_div = logits.new_zeros(())
            l_time = logits.new_zeros(())
            l_sp = logits.new_zeros(())
            loss = bce
        else:
            l_div = self.diversity_loss()
            l_time = self.time_smooth_loss()
            l_sp = self.sparsity_loss()
            loss = bce + lambda_div * l_div + lambda_time * l_time + lambda_sp * l_sp
        return ForwardOut(
            logits=logits,
            prob=prob,
            activations=a,
            d_stages=d_stages,
            t_star=t_star,
            loss_bce=bce,
            loss_div=l_div,
            loss_time=l_time,
            loss_sp=l_sp,
            loss=loss,
        )

    def grouped_parameter_groups(
        self,
        lr_shapelet: float,
        lr_alpha: float,
        lr_head: float,
        weight_decay: float,
    ) -> list[dict]:
        """AdamW param groups (docs/12 §9B); respects head_mode ablation."""
        groups: list[dict] = []
        if self.head_mode != "static":
            groups.append(
                {
                    "params": list(self.shapelets.parameters()),
                    "lr": lr_shapelet,
                    "weight_decay": weight_decay,
                }
            )
            groups.append(
                {
                    "params": list(self.alpha_logits.parameters()),
                    "lr": lr_alpha,
                    "weight_decay": weight_decay,
                }
            )
            head_params = [self.beta, self.bias]
            if self.head_mode == "combined":
                head_params.append(self.theta)
            groups.append(
                {
                    "params": head_params,
                    "lr": lr_head,
                    "weight_decay": weight_decay,
                }
            )
        else:
            groups.append(
                {
                    "params": [self.theta, self.bias],
                    "lr": lr_head,
                    "weight_decay": weight_decay,
                }
            )
        return groups
