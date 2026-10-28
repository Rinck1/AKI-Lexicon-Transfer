"""
Directional / relation-aware GNN risk model (docs/13 §8).

Pure PyTorch — no torch_geometric. Stage B: synthetic forward/backward only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.utils.hard_concrete import (  # noqa: E402
    hard_concrete_l0_prob,
    hard_concrete_sample,
    hard_mask_from_log_alpha,
    masked_pearson,
)
from scripts.utils.graph_constants import (
    A_PROJ_DIM,
    EDGE_ENC_HIDDEN,
    EDGE_ENC_OUT,
    EDGE_GATE_INIT_BIAS,
    EDGE_GATE_JITTER_A,
    EDGE_GATE_JITTER_HOURS,
    EDGE_GATE_LAMBDA_SPARSE,
    EDGE_GATE_LAMBDA_STABLE,
    EDGE_L0_LAMBDA,
    EDGE_L0_LAMBDA_RANK,
    EDGE_L0_LOG_ALPHA_INIT,
    EDGE_TOPK_FRACTION,
    EDGE_TOPK_LAMBDA_RANK,
    EDGE_VALUE_MAX,
    EDGE_POOL_DIM,
    EDGE_TYPE_CO_OCCUR,
    EDGE_TYPE_PRECEDES,
    EMPTY_GRAPH_EMBED_DIM,
    FUSION_DIM,
    GNN_DROPOUT,
    HIDDEN_DIM,
    N_EDGE_ATTR,
    N_GNN_LAYERS,
    N_MESSAGE_RELS,
    N_SPARSE_FEATURES,
    N_STATIC_FEATURES,
    NODE_TYPE_DIM,
    REL_CO_OCCUR,
    REL_FROM_PREDECESSOR,
    REL_FROM_SUCCESSOR,
    RISK_HIDDEN,
    SPARSE_PROJ_DIM,
    STATIC_PROJ_DIM,
)


def require_torch_2() -> None:
    major = int(torch.__version__.split(".")[0])
    if major < 2:
        raise RuntimeError(
            f"docs/13 requires torch>=2.0 for scatter_reduce_; got {torch.__version__}"
        )


@dataclass
class MessageTrace:
    layer: int
    batch_idx: torch.Tensor
    edge_slot: torch.Tensor
    src_node: torch.Tensor
    target_node: torch.Tensor
    relation_type: torch.Tensor
    eta: torch.Tensor
    message_vector_norm: torch.Tensor


@dataclass
class GraphForwardOut:
    logits: torch.Tensor
    prob: torch.Tensor
    h_graph: torch.Tensor
    node_attention: torch.Tensor
    edge_attention: torch.Tensor
    loss_bce: torch.Tensor
    loss: torch.Tensor
    node_contrib: torch.Tensor | None = None
    edge_contrib: torch.Tensor | None = None
    graph_logit: torch.Tensor | None = None
    tabular_logit: torch.Tensor | None = None
    message_attention: torch.Tensor | None = None
    node_contrib_by_layer: torch.Tensor | None = None
    h0: torch.Tensor | None = None
    h1: torch.Tensor | None = None
    h2: torch.Tensor | None = None
    h3: torch.Tensor | None = None
    message_trace: list[MessageTrace] | None = None
    edge_gate: torch.Tensor | None = None
    edge_value: torch.Tensor | None = None
    edge_selection_prob: torch.Tensor | None = None
    loss_sparse: torch.Tensor | None = None
    loss_stable: torch.Tensor | None = None
    mean_gate: torch.Tensor | None = None


def _segment_softmax(
    scores: torch.Tensor,
    segment_ids: torch.Tensor,
    n_segments: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Softmax of scores within segments (stable).

    scores: (M,)
    segment_ids: (M,) long in [0, n_segments)
    mask: (M,) bool — invalid messages get zero weight
    """
    if scores.numel() == 0:
        return scores
    neg_inf = torch.finfo(scores.dtype).min / 4
    s = scores.masked_fill(~mask, neg_inf)
    # max per segment
    max_s = torch.full(
        (n_segments,), neg_inf, device=scores.device, dtype=scores.dtype
    )
    max_s = max_s.scatter_reduce(
        0, segment_ids, s, reduce="amax", include_self=True
    )
    # All-invalid segments stay at -inf; (-inf)-(-inf) would be NaN.
    safe_max = torch.where(torch.isfinite(max_s), max_s, torch.zeros_like(max_s))
    s_shift = s - safe_max[segment_ids]
    exp_s = s_shift.exp().masked_fill(~mask, 0.0)
    denom = torch.zeros(n_segments, device=scores.device, dtype=scores.dtype)
    denom = denom.index_add(0, segment_ids, exp_s)
    denom = denom.clamp_min(1e-12)
    return exp_s / denom[segment_ids]


class EdgeEncoder(nn.Module):
    def __init__(self, in_dim: int = N_EDGE_ATTR, hidden: int = EDGE_ENC_HIDDEN, out: int = EDGE_ENC_OUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.net(edge_attr)


class AttentionMLP(nn.Module):
    def __init__(self, hidden: int = HIDDEN_DIM, edge_dim: int = EDGE_ENC_OUT, n_rel: int = N_MESSAGE_RELS):
        super().__init__()
        # [h_src, h_tgt, e_enc, r_emb]
        in_dim = hidden * 2 + edge_dim + n_rel  # one-hot rel
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        h_src: torch.Tensor,
        h_tgt: torch.Tensor,
        e_enc: torch.Tensor,
        rel_oh: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([h_src, h_tgt, e_enc, rel_oh], dim=-1)
        return self.net(x).squeeze(-1)


class DirectionalGraphLayer(nn.Module):
    """
    One layer of directed / relation-aware message passing (docs/13 §8.4).

    Canonical edges expanded to messages:
      PRECEDES u->w: FROM_PRED (w<-u) + FROM_SUCC (u<-w)
      CO_OCCUR a-b:  CO both directions
    """

    def __init__(
        self,
        hidden: int = HIDDEN_DIM,
        dropout: float = GNN_DROPOUT,
        edge_enc_out: int = EDGE_ENC_OUT,
    ):
        super().__init__()
        self.hidden = hidden
        self.W_self = nn.Linear(hidden, hidden, bias=False)
        self.W_pred = nn.Linear(hidden, hidden, bias=False)
        self.W_succ = nn.Linear(hidden, hidden, bias=False)
        self.W_co = nn.Linear(hidden, hidden, bias=False)
        self.edge_encoder = EdgeEncoder(out=edge_enc_out)
        self.att = AttentionMLP(hidden=hidden, edge_dim=edge_enc_out)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)
        # relation one-hot buffers
        self.register_buffer(
            "rel_eye",
            torch.eye(N_MESSAGE_RELS, dtype=torch.float32),
            persistent=False,
        )

    def _expand_messages(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Expand batch edges to message lists.

        Returns flat tensors (M,):
          batch_idx, src, tgt, rel, valid, edge_slot
          and e_attr (M, 5)
        """
        bsz, emax = edge_src.shape
        device = edge_src.device
        # indices for all padded slots
        b_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, emax)
        e_idx = torch.arange(emax, device=device).unsqueeze(0).expand(bsz, emax)
        flat_b = b_idx.reshape(-1)
        flat_e = e_idx.reshape(-1)
        flat_src = edge_src.reshape(-1)
        flat_dst = edge_dst.reshape(-1)
        flat_type = edge_type.reshape(-1)
        flat_mask = edge_mask.reshape(-1)
        flat_attr = edge_attr.reshape(-1, edge_attr.shape[-1])

        is_pre = flat_mask & (flat_type == EDGE_TYPE_PRECEDES)
        is_co = flat_mask & (flat_type == EDGE_TYPE_CO_OCCUR)

        # PRECEDES: two directed messages
        # pred: src->tgt (target reads predecessor)
        # succ: dst->src as source, target=src (source reads successor)
        msgs_b = []
        msgs_src = []
        msgs_tgt = []
        msgs_rel = []
        msgs_attr = []
        msgs_valid = []
        msgs_e = []

        def _append(mask: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor, rel: int, attr: torch.Tensor):
            if mask.any():
                msgs_b.append(flat_b[mask])
                msgs_e.append(flat_e[mask])
                msgs_src.append(src[mask])
                msgs_tgt.append(tgt[mask])
                msgs_rel.append(
                    torch.full((int(mask.sum().item()),), rel, device=device, dtype=torch.long)
                )
                msgs_attr.append(attr[mask])
                msgs_valid.append(torch.ones(int(mask.sum().item()), device=device, dtype=torch.bool))

        _append(is_pre, flat_src, flat_dst, REL_FROM_PREDECESSOR, flat_attr)
        _append(is_pre, flat_dst, flat_src, REL_FROM_SUCCESSOR, flat_attr)
        _append(is_co, flat_src, flat_dst, REL_CO_OCCUR, flat_attr)
        _append(is_co, flat_dst, flat_src, REL_CO_OCCUR, flat_attr)

        if not msgs_b:
            empty_l = torch.zeros(0, dtype=torch.long, device=device)
            empty_a = torch.zeros(0, flat_attr.shape[-1], dtype=flat_attr.dtype, device=device)
            empty_m = torch.zeros(0, dtype=torch.bool, device=device)
            return empty_l, empty_l, empty_l, empty_l, empty_a, empty_m, empty_l

        return (
            torch.cat(msgs_b),
            torch.cat(msgs_src),
            torch.cat(msgs_tgt),
            torch.cat(msgs_rel),
            torch.cat(msgs_attr, dim=0),
            torch.cat(msgs_valid),
            torch.cat(msgs_e),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
        active_mask: torch.Tensor,
        return_messages: bool = False,
        return_trace: bool = False,
        edge_z: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, MessageTrace]:
        """
        h: (B, N, H)
        edges: (B, Emax, ...)
        active_mask: (B, N)
        """
        bsz, n_nodes, hidden = h.shape
        emax = edge_src.shape[1]
        device = h.device

        b_idx, m_src, m_tgt, m_rel, m_attr, m_valid, m_edge = self._expand_messages(
            edge_src, edge_dst, edge_type, edge_attr, edge_mask
        )

        # self transform for all nodes
        msg_acc = self.W_self(h)  # (B, N, H)
        msg_att = h.new_zeros(bsz, emax)
        empty_l = torch.zeros(0, dtype=torch.long, device=device)
        empty_f = torch.zeros(0, dtype=h.dtype, device=device)
        trace = MessageTrace(
            layer=0,
            batch_idx=empty_l,
            edge_slot=empty_l,
            src_node=empty_l,
            target_node=empty_l,
            relation_type=empty_l,
            eta=empty_f,
            message_vector_norm=empty_f,
        )

        if m_valid.numel() > 0 and m_valid.any():
            # gather h_src, h_tgt
            h_src = h[b_idx, m_src]  # (M, H)
            h_tgt = h[b_idx, m_tgt]
            e_enc = self.edge_encoder(m_attr)  # (M, edge_dim)
            rel_oh = self.rel_eye[m_rel]  # (M, 3)

            scores = self.att(h_src, h_tgt, e_enc, rel_oh)  # (M,)

            # segment: (batch, target, relation) -> linear id
            # n_segments = B * N * N_REL
            n_seg = bsz * n_nodes * N_MESSAGE_RELS
            seg = (
                b_idx * (n_nodes * N_MESSAGE_RELS)
                + m_tgt * N_MESSAGE_RELS
                + m_rel
            )
            # only active targets
            tgt_active = active_mask[b_idx, m_tgt]
            src_active = active_mask[b_idx, m_src]
            valid = m_valid & tgt_active & src_active
            z_m = None
            if edge_z is not None:
                z_m = edge_z[b_idx, m_edge]
                valid = valid & (z_m > 0)
                scores = scores + z_m.clamp_min(1e-6).log()
            eta = _segment_softmax(scores, seg, n_seg, valid)  # (M,)

            # relation-specific transforms of source
            # stack W_pred, W_succ, W_co
            h_pred = self.W_pred(h_src)
            h_succ = self.W_succ(h_src)
            h_co = self.W_co(h_src)
            # select by rel
            h_msg = torch.where(
                (m_rel == REL_FROM_PREDECESSOR).unsqueeze(-1),
                h_pred,
                torch.where(
                    (m_rel == REL_FROM_SUCCESSOR).unsqueeze(-1),
                    h_succ,
                    h_co,
                ),
            )
            weighted = h_msg * eta.unsqueeze(-1)  # (M, H)
            if z_m is not None:
                weighted = weighted * z_m.unsqueeze(-1)

            # scatter-add to (B, N, H)
            flat_tgt = b_idx * n_nodes + m_tgt  # (M,)
            acc_flat = torch.zeros(bsz * n_nodes, hidden, device=device, dtype=h.dtype)
            acc_flat = acc_flat.index_add(0, flat_tgt, weighted)
            msg_acc = msg_acc + acc_flat.view(bsz, n_nodes, hidden)

            if (return_messages or return_trace) and valid.any():
                slot = b_idx * emax + m_edge
                eta_sum = h.new_zeros(bsz * emax)
                eta_cnt = h.new_zeros(bsz * emax)
                eta_sum = eta_sum.index_add(0, slot[valid], eta[valid])
                eta_cnt = eta_cnt.index_add(
                    0, slot[valid], torch.ones_like(eta[valid])
                )
                msg_att = (eta_sum / eta_cnt.clamp_min(1.0)).view(bsz, emax)
            if return_trace:
                keep = valid
                trace = MessageTrace(
                    layer=0,
                    batch_idx=b_idx[keep].long(),
                    edge_slot=m_edge[keep].long(),
                    src_node=m_src[keep].long(),
                    target_node=m_tgt[keep].long(),
                    relation_type=m_rel[keep].long(),
                    eta=eta[keep],
                    message_vector_norm=weighted[keep].norm(dim=-1),
                )

        out = self.norm(h + self.dropout(F.gelu(msg_acc)))
        out = out * active_mask.unsqueeze(-1).to(out.dtype)
        if return_trace:
            return out, msg_att, trace
        if return_messages:
            return out, msg_att
        return out


class NodeAttentionPool(nn.Module):
    def __init__(self, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.W = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(
        self, h: torch.Tensor, active_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h: (B, N, H), active_mask: (B, N)
        returns h_node (B, H), omega (B, N)
        """
        r = self.v(torch.tanh(self.W(h))).squeeze(-1)  # (B, N)
        neg_inf = torch.finfo(r.dtype).min / 4
        r = r.masked_fill(~active_mask, neg_inf)
        # if no active nodes, softmax is uniform over nothing -> zeros later
        has_active = active_mask.any(dim=-1, keepdim=True)
        omega = torch.softmax(r, dim=-1)
        omega = omega * active_mask.to(omega.dtype)
        # re-normalize when some active
        denom = omega.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        omega = torch.where(has_active, omega / denom, torch.zeros_like(omega))
        h_node = (omega.unsqueeze(-1) * h).sum(dim=1)
        return h_node, omega


class EdgeAttentionPool(nn.Module):
    def __init__(
        self,
        hidden: int = HIDDEN_DIM,
        edge_attr_dim: int = N_EDGE_ATTR,
        edge_dim: int = EDGE_POOL_DIM,
        n_edge_types: int = 2,
    ):
        super().__init__()
        self.rel_emb = nn.Embedding(n_edge_types, 8)
        in_dim = hidden * 2 + 8 + edge_attr_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.score = nn.Linear(edge_dim, 1)

    def forward(
        self,
        h: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns h_edge (B, edge_dim), alpha (B, Emax)

        CO_OCCUR endpoints are ordered by node id so storage order is invariant.
        """
        bsz, emax = edge_src.shape
        edge_dim = self.edge_mlp[-1].out_features
        device = h.device
        dtype = h.dtype

        if emax == 0:
            return (
                torch.zeros(bsz, edge_dim, device=device, dtype=dtype),
                torch.zeros(bsz, 0, device=device, dtype=dtype),
            )

        # Canonicalize CO_OCCUR (min_id, max_id) for symmetric pool representation
        is_co = edge_type == EDGE_TYPE_CO_OCCUR
        src_ord = torch.where(is_co, torch.minimum(edge_src, edge_dst), edge_src)
        dst_ord = torch.where(is_co, torch.maximum(edge_src, edge_dst), edge_dst)

        b_idx = torch.arange(bsz, device=device).unsqueeze(1)
        h_s = h[b_idx, src_ord.clamp(min=0)]
        h_d = h[b_idx, dst_ord.clamp(min=0)]
        r = self.rel_emb(edge_type.clamp(min=0, max=1))
        er = self.edge_mlp(torch.cat([h_s, h_d, r, edge_attr], dim=-1))  # (B, E, D)
        scores = self.score(er).squeeze(-1)  # (B, E)
        neg_inf = torch.finfo(scores.dtype).min / 4
        scores = scores.masked_fill(~edge_mask, neg_inf)
        has_e = edge_mask.any(dim=-1, keepdim=True)
        alpha = torch.softmax(scores, dim=-1)
        alpha = alpha * edge_mask.to(alpha.dtype)
        denom = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        alpha = torch.where(has_e, alpha / denom, torch.zeros_like(alpha))
        h_edge = (alpha.unsqueeze(-1) * er).sum(dim=1)
        h_edge = torch.where(has_e, h_edge, torch.zeros_like(h_edge))
        return h_edge, alpha


# Head-ablation arms (docs/18). Same fusion dim; disabled branches → zeros.
BRANCH_MODES: dict[str, dict[str, bool]] = {
    "C": {"use_graph": True, "use_act": True, "use_sparse": True, "use_static": True},
    "full": {"use_graph": True, "use_act": True, "use_sparse": True, "use_static": True},
    "A": {"use_graph": False, "use_act": True, "use_sparse": False, "use_static": True},
    "act_static": {
        "use_graph": False,
        "use_act": True,
        "use_sparse": False,
        "use_static": True,
    },
    "B": {"use_graph": False, "use_act": True, "use_sparse": True, "use_static": True},
    "act_sparse_static": {
        "use_graph": False,
        "use_act": True,
        "use_sparse": True,
        "use_static": True,
    },
    # A+G: graph without the 308-d stage-lab shortcut (2×2 cell opposite B)
    "AG": {"use_graph": True, "use_act": True, "use_sparse": False, "use_static": True},
    "A+G": {"use_graph": True, "use_act": True, "use_sparse": False, "use_static": True},
    "act_static_graph": {
        "use_graph": True,
        "use_act": True,
        "use_sparse": False,
        "use_static": True,
    },
    # Graph + static only: activations enter as node features, not as a 252-d bag
    "G": {"use_graph": True, "use_act": False, "use_sparse": False, "use_static": True},
    "graph_static": {
        "use_graph": True,
        "use_act": False,
        "use_sparse": False,
        "use_static": True,
    },
}

READOUT_MODES: tuple[str, ...] = (
    "pool",
    "additive",
    "additive_ms",
    "additive_gate",
    "additive_l0",
    "additive_topk",
)
ADDITIVE_READOUTS: frozenset[str] = frozenset(
    {"additive", "additive_ms", "additive_gate", "additive_l0", "additive_topk"}
)


def resolve_readout(name: str) -> str:
    key = str(name).strip().lower()
    if key not in READOUT_MODES:
        raise ValueError(f"unknown readout {name!r}; expected one of {READOUT_MODES}")
    return key


def _l1_normalize(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = values.abs() * mask.to(values.dtype)
    denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    has = mask.any(dim=-1, keepdim=True)
    return torch.where(has, w / denom, torch.zeros_like(w))


def edge_budget_k(n_edges: torch.Tensor, fraction: float = EDGE_TOPK_FRACTION) -> torch.Tensor:
    """K=max(1, ceil(fraction E)) for E>0, else 0. n_edges: (B,) long/float."""
    n = n_edges.to(dtype=torch.long)
    ke = torch.ceil(n.to(dtype=torch.float32) * float(fraction)).to(dtype=torch.long).clamp(min=1)
    return torch.where(n > 0, torch.minimum(ke, n), torch.zeros_like(n))


def hard_topk_mask(
    scores: torch.Tensor,
    valid: torch.Tensor,
    *,
    fraction: float = EDGE_TOPK_FRACTION,
) -> torch.Tensor:
    """Per-row hard 0/1 mask of the top-K valid scores. K=max(1, ceil(fraction E))."""
    n_e = valid.sum(dim=-1)
    k = edge_budget_k(n_e, fraction=fraction)
    filled = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    order = filled.argsort(dim=-1, descending=True)
    pos = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
    keep_sorted = pos < k.unsqueeze(1)
    z = torch.zeros_like(scores)
    z.scatter_(1, order, keep_sorted.to(scores.dtype))
    return z * valid.to(scores.dtype)


def resolve_branch_mode(name: str) -> str:
    key = str(name).strip()
    if key not in BRANCH_MODES:
        raise ValueError(f"unknown branch_mode {name!r}; expected one of {sorted(BRANCH_MODES)}")
    return key


class GraphRiskModel(nn.Module):
    """Patient-graph risk model (docs/13 §8)."""

    def __init__(
        self,
        n_nodes: int,
        *,
        n_sparse: int = N_SPARSE_FEATURES,
        n_static: int = N_STATIC_FEATURES,
        node_type_dim: int = NODE_TYPE_DIM,
        hidden: int = HIDDEN_DIM,
        n_layers: int = N_GNN_LAYERS,
        dropout: float = GNN_DROPOUT,
        branch_mode: str = "C",
        residual_graph: bool = False,
        readout: str = "pool",
        use_node_length: bool = False,
    ):
        super().__init__()
        require_torch_2()
        self.n_nodes = int(n_nodes)
        self.hidden = hidden
        self.branch_mode = resolve_branch_mode(branch_mode)
        flags = BRANCH_MODES[self.branch_mode]
        self.use_graph = bool(flags["use_graph"])
        self.use_act = bool(flags["use_act"])
        self.use_sparse = bool(flags["use_sparse"])
        self.use_static = bool(flags["use_static"])
        self.residual_graph = bool(residual_graph)
        self.readout = resolve_readout(readout)
        self.n_gnn_layers = int(n_layers)
        self.n_scale = self.n_gnn_layers + 1
        self.use_node_length = bool(use_node_length)
        if self.residual_graph and self.readout in ADDITIVE_READOUTS:
            raise ValueError("residual_graph and additive readout cannot be combined")
        self.lambda_sparse = float(EDGE_GATE_LAMBDA_SPARSE)
        self.lambda_stable = float(EDGE_GATE_LAMBDA_STABLE)
        self.lambda_l0 = float(EDGE_L0_LAMBDA)
        self.lambda_rank = float(
            EDGE_TOPK_LAMBDA_RANK if self.readout == "additive_topk" else EDGE_L0_LAMBDA_RANK
        )
        self.v_max = float(EDGE_VALUE_MAX)
        self.topk_fraction = float(EDGE_TOPK_FRACTION)
        self.node_emb = nn.Embedding(self.n_nodes, node_type_dim)
        # [E_u, a, t, t_stage] or [E_u, a, t, t_stage, L/12] -> H
        node_in_extra = 4 if self.use_node_length else 3
        self.node_in = nn.Linear(node_type_dim + node_in_extra, hidden)
        self.node_in_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList(
            [DirectionalGraphLayer(hidden=hidden, dropout=dropout) for _ in range(n_layers)]
        )
        self.node_pool = NodeAttentionPool(hidden=hidden)
        self.edge_pool = EdgeAttentionPool(hidden=hidden)
        self.empty_graph_embedding = nn.Parameter(
            torch.zeros(EMPTY_GRAPH_EMBED_DIM)
        )
        nn.init.normal_(self.empty_graph_embedding, std=0.02)

        self.a_proj = nn.Sequential(
            nn.Linear(self.n_nodes, A_PROJ_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sparse_proj = nn.Sequential(
            nn.Linear(n_sparse, SPARSE_PROJ_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, STATIC_PROJ_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(FUSION_DIM, RISK_HIDDEN),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(RISK_HIDDEN, 1),
        )
        # logit = logit_B + gate * delta_graph  (gate init 0); only in residual mode
        if self.residual_graph:
            self.graph_delta_head = nn.Linear(EMPTY_GRAPH_EMBED_DIM, 1)
            self.graph_gate = nn.Parameter(torch.zeros(()))
        else:
            self.graph_delta_head = None
            self.graph_gate = None
        # Additive readout: graph_logit = sum node_contrib + sum edge_contrib
        self.edge_gate = None
        self.edge_value = None
        self.edge_score = None
        self.edge_type_prior = None
        self.edge_delta = None
        if self.readout == "additive":
            self.node_risk = nn.Linear(hidden, 1)
            self.edge_risk = nn.Linear(EDGE_POOL_DIM, 1)
            nn.init.zeros_(self.node_risk.weight)
            nn.init.zeros_(self.node_risk.bias)
            nn.init.zeros_(self.edge_risk.weight)
            nn.init.zeros_(self.edge_risk.bias)
        elif self.readout == "additive_ms":
            self.node_risk = nn.Linear(hidden * self.n_scale, 1, bias=False)
            self.edge_risk = nn.Linear(EDGE_POOL_DIM, 1)
            nn.init.zeros_(self.node_risk.weight)
            nn.init.zeros_(self.edge_risk.weight)
            nn.init.zeros_(self.edge_risk.bias)
        elif self.readout == "additive_gate":
            self.node_risk = nn.Linear(hidden, 1)
            self.edge_risk = None
            self.edge_gate = nn.Linear(EDGE_POOL_DIM, 1)
            self.edge_value = nn.Linear(EDGE_POOL_DIM, 1)
            nn.init.zeros_(self.node_risk.weight)
            nn.init.zeros_(self.node_risk.bias)
            nn.init.zeros_(self.edge_gate.weight)
            nn.init.constant_(self.edge_gate.bias, float(EDGE_GATE_INIT_BIAS))
            nn.init.normal_(self.edge_value.weight, std=0.02)
            nn.init.zeros_(self.edge_value.bias)
        elif self.readout == "additive_l0":
            self.node_risk = nn.Linear(hidden, 1)
            self.edge_risk = None
            n_clinical = self.n_nodes * self.n_nodes * 2
            self.edge_type_prior = nn.Embedding(n_clinical, 1)
            nn.init.constant_(self.edge_type_prior.weight, float(EDGE_L0_LOG_ALPHA_INIT))
            self.edge_delta = nn.Linear(EDGE_POOL_DIM, 1)
            nn.init.zeros_(self.edge_delta.weight)
            nn.init.zeros_(self.edge_delta.bias)
            self.edge_value = nn.Linear(EDGE_POOL_DIM, 1)
            nn.init.zeros_(self.edge_value.weight)
            nn.init.zeros_(self.edge_value.bias)
            nn.init.zeros_(self.node_risk.weight)
            nn.init.zeros_(self.node_risk.bias)
        elif self.readout == "additive_topk":
            self.node_risk = nn.Linear(hidden, 1)
            self.edge_risk = None
            self.edge_score = nn.Linear(EDGE_POOL_DIM, 1)
            self.edge_value = nn.Linear(EDGE_POOL_DIM, 1)
            nn.init.normal_(self.edge_score.weight, std=0.02)
            nn.init.zeros_(self.edge_score.bias)
            nn.init.zeros_(self.edge_value.weight)
            nn.init.zeros_(self.edge_value.bias)
            nn.init.zeros_(self.node_risk.weight)
            nn.init.zeros_(self.node_risk.bias)
        else:
            self.node_risk = None
            self.edge_risk = None

    B_PARAM_PREFIXES: tuple[str, ...] = (
        "a_proj.",
        "sparse_proj.",
        "static_proj.",
        "risk_head.",
    )

    def encode_nodes(
        self,
        a_dense: torch.Tensor,
        node_time: torch.Tensor,
        node_time_within_stage: torch.Tensor,
        active_mask: torch.Tensor,
        node_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, n = a_dense.shape
        ids = torch.arange(n, device=a_dense.device)
        e = self.node_emb(ids).unsqueeze(0).expand(bsz, -1, -1)
        parts = [a_dense, node_time, node_time_within_stage]
        if self.use_node_length:
            nl = node_length
            if nl is None:
                nl = a_dense.new_zeros(a_dense.shape)
            parts.append(nl)
        feat = torch.stack(parts, dim=-1)
        h0 = F.gelu(self.node_in(torch.cat([e, feat], dim=-1)))
        h0 = self.node_in_norm(h0)
        h0 = h0 * active_mask.unsqueeze(-1).to(h0.dtype)
        return h0

    def _run_encoder_states(
        self,
        a: torch.Tensor,
        batch: dict[str, torch.Tensor],
        active: torch.Tensor,
        *,
        return_trace: bool = False,
        edge_z: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[MessageTrace] | None]:
        bsz = a.shape[0]
        emax = batch["edge_mask"].shape[1]
        h = self.encode_nodes(
            a,
            batch["node_time"],
            batch["node_time_within_stage"],
            active,
            node_length=batch.get("node_length"),
        )
        states = [h]
        traces: list[MessageTrace] = []
        msg_att = a.new_zeros(bsz, emax)
        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) - 1
            if return_trace:
                h, msg_att, tr = layer(
                    h,
                    batch["edge_src"],
                    batch["edge_dst"],
                    batch["edge_type"],
                    batch["edge_attr"],
                    batch["edge_mask"],
                    active,
                    return_trace=True,
                    edge_z=edge_z,
                )
                tr.layer = i + 1
                traces.append(tr)
            elif last:
                h, msg_att = layer(
                    h,
                    batch["edge_src"],
                    batch["edge_dst"],
                    batch["edge_type"],
                    batch["edge_attr"],
                    batch["edge_mask"],
                    active,
                    return_messages=True,
                    edge_z=edge_z,
                )
            else:
                h = layer(
                    h,
                    batch["edge_src"],
                    batch["edge_dst"],
                    batch["edge_type"],
                    batch["edge_attr"],
                    batch["edge_mask"],
                    active,
                    edge_z=edge_z,
                )
            states.append(h)
        return states, msg_att, traces if return_trace else None

    def _run_encoder(
        self,
        a: torch.Tensor,
        batch: dict[str, torch.Tensor],
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states, msg_att, _tr = self._run_encoder_states(a, batch, active)
        return states[-1], msg_att

    def _split_multiscale_node_contrib(
        self,
        states: list[torch.Tensor],
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (total B,N), (by_layer B,N,L+1) from bias-free Linear chunks."""
        assert self.node_risk is not None
        h_cat = torch.cat(states, dim=-1)
        total = self.node_risk(h_cat).squeeze(-1) * active.to(h_cat.dtype)
        weight = self.node_risk.weight.view(len(states), self.hidden)
        parts = [
            (h_l * weight[i]).sum(dim=-1) * active.to(h_l.dtype)
            for i, h_l in enumerate(states)
        ]
        by_layer = torch.stack(parts, dim=-1)
        return total, by_layer

    def _edge_repr(
        self,
        h: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        edge_src = batch["edge_src"]
        edge_dst = batch["edge_dst"]
        edge_type = batch["edge_type"]
        is_co = edge_type == EDGE_TYPE_CO_OCCUR
        src_ord = torch.where(is_co, torch.minimum(edge_src, edge_dst), edge_src)
        dst_ord = torch.where(is_co, torch.maximum(edge_src, edge_dst), edge_dst)
        b_idx = torch.arange(h.shape[0], device=h.device).unsqueeze(1)
        h_s = h[b_idx, src_ord.clamp(min=0)]
        h_d = h[b_idx, dst_ord.clamp(min=0)]
        r = self.edge_pool.rel_emb(edge_type.clamp(min=0, max=1))
        return self.edge_pool.edge_mlp(torch.cat([h_s, h_d, r, batch["edge_attr"]], dim=-1))

    def _jitter_node_features(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Same-patient view: small a/t noise, frozen topology."""
        out = dict(batch)
        active = batch["active_mask"].to(batch["a_dense"].dtype)
        da = EDGE_GATE_JITTER_A * torch.randn_like(batch["a_dense"]) * active
        dh = EDGE_GATE_JITTER_HOURS * torch.randn_like(batch["node_time"]) * active
        out["a_dense"] = torch.clamp(batch["a_dense"] + da, 0.0, 1.0)
        out["node_time"] = torch.clamp(batch["node_time"] + dh / 48.0, 0.0, 1.0)
        out["node_time_within_stage"] = torch.clamp(
            batch["node_time_within_stage"] + dh / 12.0, 0.0, 1.0
        )
        return out

    def _edge_gate_and_value(
        self, er: torch.Tensor, edge_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.edge_gate is not None and self.edge_value is not None
        mask = edge_mask.to(er.dtype)
        g = torch.sigmoid(self.edge_gate(er).squeeze(-1)) * mask
        v = self.edge_value(er).squeeze(-1) * mask
        return g, v, g * v

    def _compute_additive_gate_graph(
        self,
        a: torch.Tensor,
        batch: dict[str, torch.Tensor],
        active: torch.Tensor,
        *,
        compute_aux: bool = False,
    ) -> tuple:
        """logit_graph = sum_u c_u + E^{-1/2} sum_e g_e v_e. No edge softmax."""
        assert self.node_risk is not None
        h, msg_att = self._run_encoder(a, batch, active)
        edge_mask = batch["edge_mask"]
        node_contrib = self.node_risk(h).squeeze(-1) * active.to(h.dtype)
        if edge_mask.shape[1] == 0:
            edge_contrib = h.new_zeros(h.shape[0], 0)
            g = h.new_zeros(h.shape[0], 0)
            v = h.new_zeros(h.shape[0], 0)
            er = h.new_zeros(h.shape[0], 0, EDGE_POOL_DIM)
        else:
            er = self._edge_repr(h, batch)
            g, v, edge_contrib = self._edge_gate_and_value(er, edge_mask)
        n_e = edge_mask.to(h.dtype).sum(dim=-1).clamp_min(1.0)
        graph_logit = node_contrib.sum(dim=-1) + n_e.rsqrt() * edge_contrib.sum(dim=-1)
        node_att = _l1_normalize(node_contrib, active)
        # Independent gates: not softmax, not forced to sum to 1.
        edge_att = g
        h_node = (node_att.unsqueeze(-1) * h).sum(dim=1)
        if er.shape[1] == 0:
            h_edge = h.new_zeros(h.shape[0], EDGE_POOL_DIM)
        else:
            w = g
            denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            h_edge = ((w / denom).unsqueeze(-1) * er).sum(dim=1)
        h_graph_raw = torch.cat([h_node, h_edge], dim=-1)
        empty = ~active.any(dim=-1)
        empty_emb = self.empty_graph_embedding.unsqueeze(0).expand(a.shape[0], -1)
        h_graph = torch.where(empty.unsqueeze(-1), empty_emb, h_graph_raw)

        n_valid = edge_mask.to(h.dtype).sum().clamp_min(1.0)
        loss_sparse = g.sum() / n_valid
        loss_stable = h.new_zeros(())
        if (
            compute_aux
            and self.training
            and edge_mask.shape[1] > 0
            and bool(edge_mask.any())
        ):
            batch2 = self._jitter_node_features(batch)
            h2, _ = self._run_encoder(batch2["a_dense"], batch2, active)
            er2 = self._edge_repr(h2, batch2)
            g2, _v2, _c2 = self._edge_gate_and_value(er2, edge_mask)
            loss_stable = (g - g2).abs().sum() / n_valid

        return (
            graph_logit,
            h_graph,
            node_att,
            edge_att,
            node_contrib,
            edge_contrib,
            msg_att,
            g,
            v,
            loss_sparse,
            loss_stable,
        )

    def _clinical_edge_index(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        n = self.n_nodes
        return (
            src.clamp(0, n - 1) * (n * 2)
            + dst.clamp(0, n - 1) * 2
            + edge_type.clamp(0, 1)
        ).long()

    def _l0_log_alpha(
        self, h0: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        assert self.edge_type_prior is not None and self.edge_delta is not None
        er0 = self._edge_repr(h0, batch)
        idx = self._clinical_edge_index(
            batch["edge_src"], batch["edge_dst"], batch["edge_type"]
        )
        alpha = self.edge_type_prior(idx).squeeze(-1)
        delta = self.edge_delta(er0).squeeze(-1)
        return alpha + delta

    def _compute_additive_l0_graph(
        self,
        a: torch.Tensor,
        batch: dict[str, torch.Tensor],
        active: torch.Tensor,
        *,
        compute_aux: bool = False,
    ) -> tuple:
        """Same z_e gates both GNN layers and the direct edge term."""
        assert self.node_risk is not None and self.edge_value is not None
        h0 = self.encode_nodes(
            a,
            batch["node_time"],
            batch["node_time_within_stage"],
            active,
            node_length=batch.get("node_length"),
        )
        edge_mask = batch["edge_mask"]
        log_alpha = self._l0_log_alpha(h0, batch)
        pi = hard_concrete_l0_prob(log_alpha) * edge_mask.to(log_alpha.dtype)
        if self.training:
            z = hard_concrete_sample(log_alpha, training=True)
        else:
            z = hard_mask_from_log_alpha(log_alpha)
        z = z * edge_mask.to(z.dtype)

        states, msg_att, _tr = self._run_encoder_states(
            a, batch, active, edge_z=z
        )
        h = states[-1]
        node_contrib = self.node_risk(h).squeeze(-1) * active.to(h.dtype)
        if edge_mask.shape[1] == 0:
            edge_contrib = h.new_zeros(h.shape[0], 0)
            v = h.new_zeros(h.shape[0], 0)
            er = h.new_zeros(h.shape[0], 0, EDGE_POOL_DIM)
        else:
            er = self._edge_repr(h, batch)
            v = self.v_max * torch.tanh(self.edge_value(er).squeeze(-1))
            v = v * edge_mask.to(v.dtype)
            edge_contrib = z * v
        n_sel = z.sum(dim=-1).clamp_min(1.0)
        graph_logit = node_contrib.sum(dim=-1) + n_sel.rsqrt() * edge_contrib.sum(dim=-1)
        node_att = _l1_normalize(node_contrib, active)
        # Rank / entropy use selection probability, not a softmax over edges.
        edge_att = pi
        h_node = (node_att.unsqueeze(-1) * h).sum(dim=1)
        if er.shape[1] == 0:
            h_edge = h.new_zeros(h.shape[0], EDGE_POOL_DIM)
        else:
            w = z
            denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            h_edge = ((w / denom).unsqueeze(-1) * er).sum(dim=1)
        h_graph_raw = torch.cat([h_node, h_edge], dim=-1)
        empty = ~active.any(dim=-1)
        empty_emb = self.empty_graph_embedding.unsqueeze(0).expand(a.shape[0], -1)
        h_graph = torch.where(empty.unsqueeze(-1), empty_emb, h_graph_raw)

        n_valid = edge_mask.to(h.dtype).sum().clamp_min(1.0)
        loss_sparse = pi.sum() / n_valid
        loss_stable = h.new_zeros(())
        if (
            compute_aux
            and self.training
            and edge_mask.shape[1] > 0
            and bool(edge_mask.any())
        ):
            batch2 = self._jitter_node_features(batch)
            h0b = self.encode_nodes(
                batch2["a_dense"],
                batch2["node_time"],
                batch2["node_time_within_stage"],
                active,
                node_length=batch2.get("node_length"),
            )
            pi2 = hard_concrete_l0_prob(self._l0_log_alpha(h0b, batch2)) * edge_mask.to(
                h.dtype
            )
            corr = masked_pearson(pi, pi2, edge_mask)
            loss_stable = 1.0 - corr

        return (
            graph_logit,
            h_graph,
            node_att,
            edge_att,
            node_contrib,
            edge_contrib,
            msg_att,
            z,
            v,
            loss_sparse,
            loss_stable,
            pi,
        )

    def _compute_additive_topk_graph(
        self,
        a: torch.Tensor,
        batch: dict[str, torch.Tensor],
        active: torch.Tensor,
        *,
        compute_aux: bool = False,
    ) -> tuple:
        """Fixed-budget STE top-k: same z gates both GNN layers and the edge term."""
        assert self.node_risk is not None and self.edge_score is not None
        assert self.edge_value is not None
        h0 = self.encode_nodes(
            a,
            batch["node_time"],
            batch["node_time_within_stage"],
            active,
            node_length=batch.get("node_length"),
        )
        edge_mask = batch["edge_mask"]
        if edge_mask.shape[1] == 0:
            s = h0.new_zeros(h0.shape[0], 0)
        else:
            er0 = self._edge_repr(h0, batch)
            s = self.edge_score(er0).squeeze(-1)
        z_hard = hard_topk_mask(s, edge_mask, fraction=self.topk_fraction)
        if self.training:
            s_st = s.masked_fill(~edge_mask, 0.0)
            z = z_hard + (s_st - s_st.detach()) * edge_mask.to(s.dtype)
        else:
            z = z_hard

        states, msg_att, _tr = self._run_encoder_states(a, batch, active, edge_z=z)
        h = states[-1]
        node_contrib = self.node_risk(h).squeeze(-1) * active.to(h.dtype)
        if edge_mask.shape[1] == 0:
            edge_contrib = h.new_zeros(h.shape[0], 0)
            v = h.new_zeros(h.shape[0], 0)
            er = h.new_zeros(h.shape[0], 0, EDGE_POOL_DIM)
        else:
            er = self._edge_repr(h, batch)
            v = self.v_max * torch.tanh(self.edge_value(er).squeeze(-1))
            v = v * edge_mask.to(v.dtype)
            edge_contrib = z * v
        n_sel = z_hard.sum(dim=-1).clamp_min(1.0)
        graph_logit = node_contrib.sum(dim=-1) + n_sel.rsqrt() * edge_contrib.sum(dim=-1)
        node_att = _l1_normalize(node_contrib, active)
        edge_att = s
        h_node = (node_att.unsqueeze(-1) * h).sum(dim=1)
        if er.shape[1] == 0:
            h_edge = h.new_zeros(h.shape[0], EDGE_POOL_DIM)
        else:
            w = z_hard
            denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            h_edge = ((w / denom).unsqueeze(-1) * er).sum(dim=1)
        h_graph_raw = torch.cat([h_node, h_edge], dim=-1)
        empty = ~active.any(dim=-1)
        empty_emb = self.empty_graph_embedding.unsqueeze(0).expand(a.shape[0], -1)
        h_graph = torch.where(empty.unsqueeze(-1), empty_emb, h_graph_raw)

        loss_sparse = h.new_zeros(())
        loss_stable = h.new_zeros(())
        if (
            compute_aux
            and self.training
            and edge_mask.shape[1] > 0
            and bool(edge_mask.any())
        ):
            batch2 = self._jitter_node_features(batch)
            h0b = self.encode_nodes(
                batch2["a_dense"],
                batch2["node_time"],
                batch2["node_time_within_stage"],
                active,
                node_length=batch2.get("node_length"),
            )
            s2 = self.edge_score(self._edge_repr(h0b, batch2)).squeeze(-1)
            corr = masked_pearson(s, s2, edge_mask)
            loss_stable = 1.0 - corr

        return (
            graph_logit,
            h_graph,
            node_att,
            edge_att,
            node_contrib,
            edge_contrib,
            msg_att,
            z_hard,
            v,
            loss_sparse,
            loss_stable,
            s,
        )

    def _compute_h_graph(
        self,
        a: torch.Tensor,
        batch: dict[str, torch.Tensor],
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, _msg_att = self._run_encoder(a, batch, active)
        h_node, node_att = self.node_pool(h, active)
        h_edge, edge_att = self.edge_pool(
            h,
            batch["edge_src"],
            batch["edge_dst"],
            batch["edge_type"],
            batch["edge_attr"],
            batch["edge_mask"],
        )
        bsz = a.shape[0]
        h_graph_raw = torch.cat([h_node, h_edge], dim=-1)
        empty = ~active.any(dim=-1)
        empty_emb = self.empty_graph_embedding.unsqueeze(0).expand(bsz, -1)
        h_graph = torch.where(empty.unsqueeze(-1), empty_emb, h_graph_raw)
        return h_graph, node_att, edge_att

    def _compute_additive_graph(
        self,
        a: torch.Tensor,
        batch: dict[str, torch.Tensor],
        active: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """graph_logit = sum_i r_i + sum_e c_e; attention is L1(|contrib|)."""
        assert self.node_risk is not None and self.edge_risk is not None
        h, msg_att = self._run_encoder(a, batch, active)
        edge_mask = batch["edge_mask"]
        node_contrib = self.node_risk(h).squeeze(-1) * active.to(h.dtype)
        if edge_mask.shape[1] == 0:
            edge_contrib = h.new_zeros(h.shape[0], 0)
            er = h.new_zeros(h.shape[0], 0, EDGE_POOL_DIM)
        else:
            er = self._edge_repr(h, batch)
            edge_contrib = self.edge_risk(er).squeeze(-1) * edge_mask.to(h.dtype)
        graph_logit = node_contrib.sum(dim=-1) + edge_contrib.sum(dim=-1)
        node_att = _l1_normalize(node_contrib, active)
        edge_att = _l1_normalize(edge_contrib, edge_mask)
        h_node = (node_att.unsqueeze(-1) * h).sum(dim=1)
        if er.shape[1] == 0:
            h_edge = h.new_zeros(h.shape[0], EDGE_POOL_DIM)
        else:
            h_edge = (edge_att.unsqueeze(-1) * er).sum(dim=1)
        h_graph_raw = torch.cat([h_node, h_edge], dim=-1)
        empty = ~active.any(dim=-1)
        empty_emb = self.empty_graph_embedding.unsqueeze(0).expand(a.shape[0], -1)
        h_graph = torch.where(empty.unsqueeze(-1), empty_emb, h_graph_raw)
        return graph_logit, h_graph, node_att, edge_att, node_contrib, edge_contrib, msg_att

    def _compute_additive_ms_graph(
        self,
        a: torch.Tensor,
        batch: dict[str, torch.Tensor],
        active: torch.Tensor,
        *,
        return_trace: bool = False,
    ) -> tuple:
        assert self.node_risk is not None and self.edge_risk is not None
        states, msg_att, traces = self._run_encoder_states(
            a, batch, active, return_trace=return_trace
        )
        h = states[-1]
        edge_mask = batch["edge_mask"]
        node_contrib, by_layer = self._split_multiscale_node_contrib(states, active)
        if edge_mask.shape[1] == 0:
            edge_contrib = h.new_zeros(h.shape[0], 0)
            er = h.new_zeros(h.shape[0], 0, EDGE_POOL_DIM)
        else:
            er = self._edge_repr(h, batch)
            edge_contrib = self.edge_risk(er).squeeze(-1) * edge_mask.to(h.dtype)
        graph_logit = node_contrib.sum(dim=-1) + edge_contrib.sum(dim=-1)
        node_att = _l1_normalize(node_contrib, active)
        edge_att = _l1_normalize(edge_contrib, edge_mask)
        h_node = (node_att.unsqueeze(-1) * h).sum(dim=1)
        if er.shape[1] == 0:
            h_edge = h.new_zeros(h.shape[0], EDGE_POOL_DIM)
        else:
            h_edge = (edge_att.unsqueeze(-1) * er).sum(dim=1)
        h_graph_raw = torch.cat([h_node, h_edge], dim=-1)
        empty = ~active.any(dim=-1)
        empty_emb = self.empty_graph_embedding.unsqueeze(0).expand(a.shape[0], -1)
        h_graph = torch.where(empty.unsqueeze(-1), empty_emb, h_graph_raw)
        return (
            graph_logit,
            h_graph,
            node_att,
            edge_att,
            node_contrib,
            edge_contrib,
            msg_att,
            by_layer,
            states,
            traces,
        )

    def freeze_b_params(self) -> None:
        for name, p in self.named_parameters():
            if any(name.startswith(pref) for pref in self.B_PARAM_PREFIXES):
                p.requires_grad = False

    def load_b_branch(self, state: dict) -> list[str]:
        """Copy act/sparse/static/risk_head weights from a B (or compatible) ckpt."""
        own = self.state_dict()
        loaded: list[str] = []
        for k, v in state.items():
            if any(k.startswith(pref) for pref in self.B_PARAM_PREFIXES) and k in own:
                if own[k].shape != v.shape:
                    raise ValueError(f"B key {k} shape {tuple(v.shape)} != {tuple(own[k].shape)}")
                own[k] = v
                loaded.append(k)
        missing = [
            k
            for k in own
            if any(k.startswith(pref) for pref in self.B_PARAM_PREFIXES) and k not in state
        ]
        if missing:
            raise ValueError(f"B checkpoint missing keys: {missing}")
        self.load_state_dict(own)
        return loaded

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        pos_weight: torch.Tensor | float | None = None,
        compute_loss: bool = True,
        return_trace: bool = False,
    ) -> GraphForwardOut:
        a = batch["a_dense"]
        active = batch["active_mask"]
        bsz = a.shape[0]
        emax = batch["edge_mask"].shape[1]

        a_p = self.a_proj(a)
        if not self.use_act:
            a_p = torch.zeros_like(a_p)
        s_p = self.sparse_proj(batch["sparse"])
        if not self.use_sparse:
            s_p = torch.zeros_like(s_p)
        st_p = self.static_proj(batch["static"])
        if not self.use_static:
            st_p = torch.zeros_like(st_p)

        node_contrib = a.new_zeros(bsz, self.n_nodes)
        edge_contrib = a.new_zeros(bsz, emax)
        graph_logit = a.new_zeros(bsz)
        tabular_logit = None
        msg_att = a.new_zeros(bsz, emax)
        by_layer = None
        states: list[torch.Tensor] | None = None
        traces: list[MessageTrace] | None = None
        edge_gate = a.new_zeros(bsz, emax)
        edge_value = a.new_zeros(bsz, emax)
        edge_selection_prob = a.new_zeros(bsz, emax)
        loss_sparse = a.new_zeros(())
        loss_stable = a.new_zeros(())

        if self.residual_graph:
            # Frozen-B path: graph slot is always zero in the risk head
            z_b = torch.cat(
                [a.new_zeros(bsz, EMPTY_GRAPH_EMBED_DIM), a_p, s_p, st_p], dim=-1
            )
            logit_b = self.risk_head(z_b).squeeze(-1)
            tabular_logit = logit_b
            h_graph, node_att, edge_att = self._compute_h_graph(a, batch, active)
            delta = self.graph_delta_head(h_graph).squeeze(-1)
            graph_logit = self.graph_gate * delta
            logits = logit_b + graph_logit
        elif self.readout == "additive":
            z_tab = torch.cat(
                [a.new_zeros(bsz, EMPTY_GRAPH_EMBED_DIM), a_p, s_p, st_p], dim=-1
            )
            tabular_logit = self.risk_head(z_tab).squeeze(-1)
            if self.use_graph:
                (
                    graph_logit,
                    h_graph,
                    node_att,
                    edge_att,
                    node_contrib,
                    edge_contrib,
                    msg_att,
                ) = self._compute_additive_graph(a, batch, active)
            else:
                h_graph = a.new_zeros(bsz, EMPTY_GRAPH_EMBED_DIM)
                node_att = a.new_zeros(bsz, self.n_nodes)
                edge_att = a.new_zeros(bsz, emax)
            logits = tabular_logit + graph_logit
        elif self.readout == "additive_gate":
            z_tab = torch.cat(
                [a.new_zeros(bsz, EMPTY_GRAPH_EMBED_DIM), a_p, s_p, st_p], dim=-1
            )
            tabular_logit = self.risk_head(z_tab).squeeze(-1)
            if self.use_graph:
                (
                    graph_logit,
                    h_graph,
                    node_att,
                    edge_att,
                    node_contrib,
                    edge_contrib,
                    msg_att,
                    edge_gate,
                    edge_value,
                    loss_sparse,
                    loss_stable,
                ) = self._compute_additive_gate_graph(
                    a, batch, active, compute_aux=compute_loss
                )
            else:
                h_graph = a.new_zeros(bsz, EMPTY_GRAPH_EMBED_DIM)
                node_att = a.new_zeros(bsz, self.n_nodes)
                edge_att = a.new_zeros(bsz, emax)
            logits = tabular_logit + graph_logit
        elif self.readout == "additive_l0":
            z_tab = torch.cat(
                [a.new_zeros(bsz, EMPTY_GRAPH_EMBED_DIM), a_p, s_p, st_p], dim=-1
            )
            tabular_logit = self.risk_head(z_tab).squeeze(-1)
            if self.use_graph:
                (
                    graph_logit,
                    h_graph,
                    node_att,
                    edge_att,
                    node_contrib,
                    edge_contrib,
                    msg_att,
                    edge_gate,
                    edge_value,
                    loss_sparse,
                    loss_stable,
                    edge_selection_prob,
                ) = self._compute_additive_l0_graph(
                    a, batch, active, compute_aux=compute_loss
                )
            else:
                h_graph = a.new_zeros(bsz, EMPTY_GRAPH_EMBED_DIM)
                node_att = a.new_zeros(bsz, self.n_nodes)
                edge_att = a.new_zeros(bsz, emax)
            logits = tabular_logit + graph_logit
        elif self.readout == "additive_topk":
            z_tab = torch.cat(
                [a.new_zeros(bsz, EMPTY_GRAPH_EMBED_DIM), a_p, s_p, st_p], dim=-1
            )
            tabular_logit = self.risk_head(z_tab).squeeze(-1)
            if self.use_graph:
                (
                    graph_logit,
                    h_graph,
                    node_att,
                    edge_att,
                    node_contrib,
                    edge_contrib,
                    msg_att,
                    edge_gate,
                    edge_value,
                    loss_sparse,
                    loss_stable,
                    _score,
                ) = self._compute_additive_topk_graph(
                    a, batch, active, compute_aux=compute_loss
                )
            else:
                h_graph = a.new_zeros(bsz, EMPTY_GRAPH_EMBED_DIM)
                node_att = a.new_zeros(bsz, self.n_nodes)
                edge_att = a.new_zeros(bsz, emax)
            logits = tabular_logit + graph_logit
        elif self.readout == "additive_ms":
            z_tab = torch.cat(
                [a.new_zeros(bsz, EMPTY_GRAPH_EMBED_DIM), a_p, s_p, st_p], dim=-1
            )
            tabular_logit = self.risk_head(z_tab).squeeze(-1)
            if self.use_graph:
                (
                    graph_logit,
                    h_graph,
                    node_att,
                    edge_att,
                    node_contrib,
                    edge_contrib,
                    msg_att,
                    by_layer,
                    states,
                    traces,
                ) = self._compute_additive_ms_graph(
                    a, batch, active, return_trace=return_trace
                )
            else:
                h_graph = a.new_zeros(bsz, EMPTY_GRAPH_EMBED_DIM)
                node_att = a.new_zeros(bsz, self.n_nodes)
                edge_att = a.new_zeros(bsz, emax)
            logits = tabular_logit + graph_logit
        else:
            if self.use_graph:
                h_graph, node_att, edge_att = self._compute_h_graph(a, batch, active)
            else:
                h_graph = a.new_zeros(bsz, EMPTY_GRAPH_EMBED_DIM)
                node_att = a.new_zeros(bsz, self.n_nodes)
                edge_att = a.new_zeros(bsz, emax)
            z = torch.cat([h_graph, a_p, s_p, st_p], dim=-1)
            logits = self.risk_head(z).squeeze(-1)
            tabular_logit = logits
        prob = torch.sigmoid(logits)

        y = batch["y"].float()
        if compute_loss:
            if pos_weight is None:
                loss_bce = F.binary_cross_entropy_with_logits(logits, y)
            else:
                pw = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
                loss_bce = F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pw
                )
        else:
            loss_bce = torch.zeros((), device=logits.device, dtype=logits.dtype)

        if self.readout == "additive_gate" and compute_loss:
            loss = (
                loss_bce
                + self.lambda_sparse * loss_sparse
                + self.lambda_stable * loss_stable
            )
        elif self.readout == "additive_l0" and compute_loss:
            loss = (
                loss_bce
                + self.lambda_l0 * loss_sparse
                + self.lambda_rank * loss_stable
            )
        elif self.readout == "additive_topk" and compute_loss:
            loss = loss_bce + self.lambda_rank * loss_stable
        else:
            loss = loss_bce
        n_valid_e = batch["edge_mask"].to(logits.dtype).sum().clamp_min(1.0)
        mean_gate = edge_gate.sum() / n_valid_e

        hs = [None, None, None, None]
        if states is not None:
            for i, st in enumerate(states[:4]):
                hs[i] = st
        return GraphForwardOut(
            logits=logits,
            prob=prob,
            h_graph=h_graph,
            node_attention=node_att,
            edge_attention=edge_att,
            loss_bce=loss_bce,
            loss=loss,
            node_contrib=node_contrib,
            edge_contrib=edge_contrib,
            graph_logit=graph_logit,
            tabular_logit=tabular_logit,
            message_attention=msg_att,
            node_contrib_by_layer=by_layer,
            h0=hs[0],
            h1=hs[1],
            h2=hs[2],
            h3=hs[3],
            message_trace=traces,
            edge_gate=edge_gate,
            edge_value=edge_value,
            edge_selection_prob=edge_selection_prob,
            loss_sparse=loss_sparse,
            loss_stable=loss_stable,
            mean_gate=mean_gate,
        )
