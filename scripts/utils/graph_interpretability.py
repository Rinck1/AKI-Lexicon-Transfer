"""Pure perturbation helpers for frozen patient-graph explanations.

The helpers deliberately keep ``a_dense``, ``sparse`` and ``static`` intact.
Only the graph topology is changed, so deletion effects measure the conditional
contribution of the GNN branch rather than removing information from every head.
"""

from __future__ import annotations

import math
from collections import Counter

import torch

from scripts.utils.graph_constants import EDGE_TYPE_PRECEDES


def _k_for_count(n_valid: int, fraction: float) -> int:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if n_valid <= 0:
        return 0
    return min(n_valid, max(1, int(math.ceil(n_valid * float(fraction)))))


def attention_topk_mask(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    fraction: float,
) -> torch.Tensor:
    """Select the highest-attention fraction independently in each row."""
    if scores.shape != valid_mask.shape:
        raise ValueError(f"score shape {scores.shape} != valid shape {valid_mask.shape}")
    selected = torch.zeros_like(valid_mask, dtype=torch.bool)
    for i in range(scores.shape[0]):
        ids = torch.nonzero(valid_mask[i], as_tuple=False).flatten()
        k = _k_for_count(int(ids.numel()), fraction)
        if k:
            local = torch.topk(scores[i, ids], k=k, largest=True, sorted=False).indices
            selected[i, ids[local]] = True
    return selected


def _sample_without_replacement(
    candidates: list[int],
    k: int,
    generator: torch.Generator,
) -> list[int]:
    if k <= 0 or not candidates:
        return []
    k = min(int(k), len(candidates))
    perm = torch.randperm(len(candidates), generator=generator)
    return [candidates[int(j)] for j in perm[:k].tolist()]


def matched_random_edge_mask(
    valid_mask: torch.Tensor,
    selected: torch.Tensor,
    edge_type: torch.Tensor,
    *,
    src_stage_id: torch.Tensor | None = None,
    any_low: torch.Tensor | None = None,
    keys: torch.Tensor,
    seed: int,
    repeat: int,
) -> torch.Tensor:
    """Random mask matching selected count, type mix, and optional stage/low-conf.

    Relaxation order when a stratum is short: (type, low, stage) → (type, low)
    → (type,) → any remaining valid edge. Sampling is with replacement across
    strata only in the sense that leftover quota is filled from coarser pools;
    each edge is used at most once.
    """
    if valid_mask.shape != selected.shape or valid_mask.shape != edge_type.shape:
        raise ValueError("valid/selected/type shapes must match")
    if valid_mask.shape[0] != int(keys.numel()):
        raise ValueError("keys must have one value per batch row")
    out = torch.zeros_like(valid_mask, dtype=torch.bool)
    for i in range(valid_mask.shape[0]):
        valid_ids = torch.nonzero(valid_mask[i], as_tuple=False).flatten()
        sel_ids = torch.nonzero(valid_mask[i] & selected[i], as_tuple=False).flatten()
        k = int(sel_ids.numel())
        if k == 0:
            continue
        key = int(keys[i].detach().cpu().item())
        generator = torch.Generator(device="cpu")
        combined = (int(seed) * 1_000_003 + key * 9_176 + int(repeat) * 101) % (2**63 - 1)
        generator.manual_seed(combined)

        def _feat(e: int) -> tuple[int, int, int]:
            t = int(edge_type[i, e].detach().cpu().item())
            low = (
                int(bool(any_low[i, e].detach().cpu().item()))
                if any_low is not None
                else 0
            )
            st = (
                int(src_stage_id[i, e].detach().cpu().item())
                if src_stage_id is not None
                else 0
            )
            return t, low, st

        valid_list = [int(x) for x in valid_ids.detach().cpu().tolist()]
        sel_list = [int(x) for x in sel_ids.detach().cpu().tolist()]
        remaining = set(valid_list)
        chosen: list[int] = []
        for level in (
            lambda t, low, st: (t, low, st),
            lambda t, low, st: (t, low),
            lambda t, low, _st: (t,),
            lambda _t, _low, _st: (),
        ):
            if len(chosen) >= k:
                break
            need: Counter[tuple] = Counter()
            have: Counter[tuple] = Counter()
            for e in sel_list:
                need[level(*_feat(e))] += 1
            for e in chosen:
                have[level(*_feat(e))] += 1
            for key_s, cnt in need.items():
                miss = cnt - have[key_s]
                if miss <= 0:
                    continue
                cand = [e for e in remaining if level(*_feat(e)) == key_s]
                take = _sample_without_replacement(cand, miss, generator)
                chosen.extend(take)
                remaining.difference_update(take)
                if len(chosen) >= k:
                    break
        if len(chosen) < k:
            extra = _sample_without_replacement(sorted(remaining), k - len(chosen), generator)
            chosen.extend(extra)
        for e in chosen[:k]:
            out[i, e] = True
    return out


def mask_jaccard(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-row Jaccard of two boolean masks on valid entries; empty∪empty is NaN."""
    aa = a.to(dtype=torch.bool) & valid
    bb = b.to(dtype=torch.bool) & valid
    union = (aa | bb).sum(dim=1).to(torch.float32)
    inter = (aa & bb).sum(dim=1).to(torch.float32)
    nan = torch.full_like(union, float("nan"))
    return torch.where(union > 0, inter / union, nan)


def random_valid_mask(
    valid_mask: torch.Tensor,
    *,
    fraction: float,
    keys: torch.Tensor,
    seed: int,
    repeat: int,
) -> torch.Tensor:
    """Deterministic per-patient random mask with the same top-k rule."""
    if valid_mask.shape[0] != int(keys.numel()):
        raise ValueError("keys must have one value per batch row")
    selected = torch.zeros_like(valid_mask, dtype=torch.bool)
    for i in range(valid_mask.shape[0]):
        ids = torch.nonzero(valid_mask[i], as_tuple=False).flatten()
        k = _k_for_count(int(ids.numel()), fraction)
        if not k:
            continue
        # Generate on CPU for identical choices on CPU/CUDA.
        key = int(keys[i].detach().cpu().item())
        generator = torch.Generator(device="cpu")
        combined = (int(seed) * 1_000_003 + key * 9_176 + int(repeat) * 101) % (2**63 - 1)
        generator.manual_seed(combined)
        perm = torch.randperm(int(ids.numel()), generator=generator)[:k]
        chosen = ids.detach().cpu()[perm].to(valid_mask.device)
        selected[i, chosen] = True
    return selected


def perturb_nodes(
    batch: dict[str, torch.Tensor],
    selected_nodes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Remove selected nodes and their incident edges from graph topology only."""
    if selected_nodes.shape != batch["active_mask"].shape:
        raise ValueError("selected_nodes shape mismatch")
    out = dict(batch)
    active = batch["active_mask"] & ~selected_nodes
    src = batch["edge_src"].clamp(min=0)
    dst = batch["edge_dst"].clamp(min=0)
    src_removed = selected_nodes.gather(1, src)
    dst_removed = selected_nodes.gather(1, dst)
    out["active_mask"] = active
    out["edge_mask"] = batch["edge_mask"] & ~src_removed & ~dst_removed
    return out


def perturb_edges(
    batch: dict[str, torch.Tensor],
    selected_edges: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Remove selected canonical edges while retaining all nodes and other branches."""
    if selected_edges.shape != batch["edge_mask"].shape:
        raise ValueError("selected_edges shape mismatch")
    out = dict(batch)
    out["edge_mask"] = batch["edge_mask"] & ~selected_edges
    return out


def topk_jaccard(
    base_scores: torch.Tensor,
    new_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    fraction: float,
) -> torch.Tensor:
    """Per-row Jaccard of top-attention sets; empty rows return NaN."""
    base = attention_topk_mask(base_scores, valid_mask, fraction=fraction)
    new = attention_topk_mask(new_scores, valid_mask, fraction=fraction)
    union = (base | new).sum(dim=1).to(torch.float32)
    inter = (base & new).sum(dim=1).to(torch.float32)
    nan = torch.full_like(union, float("nan"))
    return torch.where(union > 0, inter / union, nan)


def enumerate_precedes_paths_batch(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_type: torch.Tensor,
    edge_mask: torch.Tensor,
) -> list[list[tuple[int, int, int, int, int]]]:
    """Per-row 2-hop PRECEDES paths as (edge1, edge2, node1, node2, node3)."""
    src = edge_src.detach().cpu()
    dst = edge_dst.detach().cpu()
    is_pre = (edge_mask & (edge_type == EDGE_TYPE_PRECEDES)).detach().cpu()
    rows: list[list[tuple[int, int, int, int, int]]] = []
    for i in range(src.shape[0]):
        slots = torch.nonzero(is_pre[i], as_tuple=False).flatten().tolist()
        outgoing: dict[int, list[tuple[int, int]]] = {}
        for e in slots:
            outgoing.setdefault(int(src[i, e]), []).append((e, int(dst[i, e])))
        seen: set[tuple[int, int, int]] = set()
        paths: list[tuple[int, int, int, int, int]] = []
        for e1 in slots:
            n1 = int(src[i, e1])
            n2 = int(dst[i, e1])
            for e2, n3 in outgoing.get(n2, []):
                if e2 == e1:
                    continue
                key = (n1, n2, n3)
                if key in seen:
                    continue
                seen.add(key)
                paths.append((e1, e2, n1, n2, n3))
        rows.append(paths)
    return rows


def select_top_positive_indices(scores: list[float], *, fraction: float) -> list[int]:
    """k=ceil(fraction * n) highest scores among those with score > 0."""
    n = len(scores)
    if n == 0:
        return []
    k = _k_for_count(n, fraction)
    ranked = sorted(range(n), key=lambda j: scores[j], reverse=True)
    return [j for j in ranked if scores[j] > 0.0][:k]


def select_random_indices(
    n_items: int,
    k: int,
    *,
    key: int,
    seed: int,
    repeat: int,
) -> list[int]:
    if n_items <= 0 or k <= 0:
        return []
    k = min(int(k), int(n_items))
    generator = torch.Generator(device="cpu")
    combined = (int(seed) * 1_000_003 + int(key) * 9_176 + int(repeat) * 101) % (2**63 - 1)
    generator.manual_seed(combined)
    return torch.randperm(int(n_items), generator=generator)[:k].tolist()


def union_path_edge_mask(
    edge_mask: torch.Tensor,
    paths_by_row: list[list[tuple[int, int, int, int, int]]],
    selected_by_row: list[list[int]],
) -> torch.Tensor:
    """Select the two PRECEDES edges of each chosen path; nodes stay."""
    chosen = torch.zeros_like(edge_mask, dtype=torch.bool)
    for i, (paths, ids) in enumerate(zip(paths_by_row, selected_by_row)):
        for j in ids:
            e1, e2, _n1, _n2, _n3 = paths[j]
            chosen[i, e1] = True
            chosen[i, e2] = True
    return chosen & edge_mask


def path_interaction(delta_path: float, delta_edge1: float, delta_edge2: float) -> float:
    """Synergy: joint deletion minus the sum of single-edge deletions."""
    return float(delta_path) - float(delta_edge1) - float(delta_edge2)


def fold_stage_label(stage: str) -> str:
    return str(stage).split("_", 1)[0]


def fold_var_label(var: str) -> str:
    mapping = {
        "urine_output_ml": "urine",
        "spo2": "SpO2",
        "mbp": "MBP",
        "sbp": "SBP",
        "dbp": "DBP",
        "heart_rate": "HR",
        "respiratory_rate": "RR",
        "creatinine": "Cr",
        "bun": "BUN",
        "lactate": "lactate",
        "ph": "pH",
        "platelet": "Plt",
        "wbc": "WBC",
        "hemoglobin": "Hb",
        "gcs_total": "GCS",
    }
    return mapping.get(str(var), str(var))


def enumerate_simple_k_hop_precedes(
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_type: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    hops: int,
) -> list[list[tuple[int, ...]]]:
    """Simple hop-length PRECEDES paths: no repeated nodes or canonical edges.

    hops=3 yields 4-node chains (u, v, w, x). Internal reverse messages are
    not expanded here; only stored PRECEDES direction is used.
    """
    if hops < 1:
        raise ValueError("hops must be >= 1")
    src = edge_src.detach().cpu()
    dst = edge_dst.detach().cpu()
    is_pre = (edge_mask & (edge_type == EDGE_TYPE_PRECEDES)).detach().cpu()
    rows: list[list[tuple[int, ...]]] = []
    for i in range(src.shape[0]):
        slots = torch.nonzero(is_pre[i], as_tuple=False).flatten().tolist()
        outgoing: dict[int, list[tuple[int, int]]] = {}
        for e in slots:
            outgoing.setdefault(int(src[i, e]), []).append((e, int(dst[i, e])))
        frontier: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
        for e, n2 in ((e, int(dst[i, e])) for e in slots):
            n1 = int(src[i, e])
            if n1 == n2:
                continue
            frontier.append(((n1, n2), (e,), n2))
        paths: list[tuple[int, ...]] = []
        for _ in range(hops - 1):
            nxt: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
            for nodes, edges, tail in frontier:
                for e2, n3 in outgoing.get(tail, []):
                    if e2 in edges or n3 in nodes:
                        continue
                    nxt.append((nodes + (n3,), edges + (e2,), n3))
            frontier = nxt
        paths.extend(nodes for nodes, _e, _t in frontier)
        rows.append(paths)
    return rows


def fold_path_pattern(var1: str, stage1: str, var2: str, stage2: str, var3: str, stage3: str) -> str:
    a = f"{fold_var_label(var1)}({fold_stage_label(stage1)})"
    b = f"{fold_var_label(var2)}({fold_stage_label(stage2)})"
    c = f"{fold_var_label(var3)}({fold_stage_label(stage3)})"
    return f"{a} → {b} → {c}"


def extract_precedes_paths(
    edges: list[dict],
    *,
    min_abs_score: float = 0.0,
    top_k_per_patient: int = 5,
) -> list[dict]:
    """Length-2 PRECEDES chains ranked by |contrib| or |attention|."""
    by_stay: dict[int, list[dict]] = {}
    for rec in edges:
        if rec.get("edge_type") != "PRECEDES":
            continue
        score = rec["contrib"] if rec.get("contrib") is not None and rec["contrib"] == rec["contrib"] else rec.get("attention", 0.0)
        abs_score = abs(float(score))
        if abs_score < float(min_abs_score):
            continue
        item = dict(rec)
        item["_score"] = float(score)
        item["_abs"] = abs_score
        by_stay.setdefault(int(rec["stay_id"]), []).append(item)
    out: list[dict] = []
    for stay, items in by_stay.items():
        by_src: dict[int, list[dict]] = {}
        for rec in items:
            by_src.setdefault(int(rec["src_id"]), []).append(rec)
        seen: set[tuple[int, int, int]] = set()
        stay_rows: list[dict] = []
        for rec in items:
            mid = int(rec["dst_id"])
            for nxt in by_src.get(mid, []):
                key = (int(rec["src_id"]), mid, int(nxt["dst_id"]))
                if key in seen:
                    continue
                seen.add(key)
                stay_rows.append(
                    {
                        "stay_id": stay,
                        "y": float(rec["y"]),
                        "src_id": int(rec["src_id"]),
                        "mid_id": mid,
                        "dst_id": int(nxt["dst_id"]),
                        "src_name": rec["src_name"],
                        "mid_name": rec["dst_name"],
                        "dst_name": nxt["dst_name"],
                        "abs_score": float(rec["_abs"] + nxt["_abs"]),
                        "signed_score": float(rec["_score"] + nxt["_score"]),
                    }
                )
        stay_rows.sort(key=lambda r: r["abs_score"], reverse=True)
        out.extend(stay_rows[: int(top_k_per_patient)])
    return out


def normalized_attention_entropy(
    weights: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Entropy divided by log(n_valid); 0 for one item and NaN for empty rows."""
    w = torch.where(valid_mask, weights.clamp_min(0.0), torch.zeros_like(weights))
    denom = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    w = w / denom
    entropy = -(w * w.clamp_min(1e-12).log()).sum(dim=1)
    n = valid_mask.sum(dim=1)
    scale = torch.where(n > 1, n.to(weights.dtype).log(), torch.ones_like(entropy))
    value = torch.where(n > 1, entropy / scale, torch.zeros_like(entropy))
    return torch.where(n > 0, value, torch.full_like(value, float("nan")))
