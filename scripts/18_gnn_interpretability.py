#!/usr/bin/env python3
"""Frozen-GNN explanation fidelity and local-stability validation.

This stage is validation-only: it loads ``graph_pack_val.pt`` and a frozen
branch-C checkpoint, never trains, and refuses test input. Top-attention node
deletion changes ``active_mask`` plus incident edges while leaving a_dense,
sparse and static untouched. Edge deletion changes only ``edge_mask``. The
result therefore measures the conditional contribution of the GNN branch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.graph_constants import (  # noqa: E402
    EDGE_TYPE_NAMES,
    STAGE_NAMES,
)
from scripts.utils.graph_data import (  # noqa: E402
    GraphPackDataset,
    collate_graph_batch,
)
from scripts.utils.graph_interpretability import (  # noqa: E402
    attention_topk_mask,
    extract_precedes_paths,
    fold_stage_label,
    mask_jaccard,
    matched_random_edge_mask,
    normalized_attention_entropy,
    perturb_edges,
    perturb_nodes,
    random_valid_mask,
    topk_jaccard,
)
from scripts.utils.graph_model import (  # noqa: E402
    GraphRiskModel,
    resolve_branch_mode,
    resolve_readout,
)
from scripts.utils.shapelet_constants import STAGE_BOUNDS  # noqa: E402


DEFAULT_PACK = ROOT / "data/processed/graphs/graph_final_7var_L4_6_12_K3"
DEFAULT_CKPT = ROOT / "data/processed/gnn/gnn_full_final_lr2e4/ckpt_best.pt"
DEFAULT_OUT = ROOT / "data/processed/gnn/gnn_interpretability_lr2e4"
DEFAULT_REPORT = ROOT / "outputs/reports/18_GNN可解释性验证.md"


def _opt_int(value) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
    }


def _bootstrap_mean_ci(x: np.ndarray, *, n_boot: int, seed: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        vals[i] = x[rng.integers(0, len(x), size=len(x))].mean()
    return {
        "estimate": float(x.mean()),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def _bootstrap_metric_difference(
    y: np.ndarray,
    p_random: np.ndarray,
    p_top: np.ndarray,
    *,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """random-deletion metric minus top-deletion metric; positive supports fidelity."""
    fn = average_precision_score if metric == "auprc" else roc_auc_score
    estimate = float(fn(y, p_random) - fn(y, p_top))
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vals.append(float(fn(yb, p_random[idx]) - fn(yb, p_top[idx])))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def _load_val_pack(pack_dir: Path, max_stays: int | None) -> tuple[GraphPackDataset, dict]:
    if "test" in pack_dir.name.lower():
        raise SystemExit("Refusing a path containing 'test'; this stage is validation-only")
    path = pack_dir / "graph_pack_val.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    patients = list(obj["patients"])
    if max_stays is not None:
        patients = patients[: int(max_stays)]
    return GraphPackDataset(patients), dict(obj.get("meta") or {})


def _load_model(ckpt_path: Path, n_nodes: int, device: torch.device) -> tuple[GraphRiskModel, dict]:
    if "test" in ckpt_path.name.lower():
        raise SystemExit("Refusing a checkpoint path containing 'test'")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = dict(ckpt.get("meta") or {})
    branch = resolve_branch_mode(meta.get("branch_mode") or "C")
    readout = resolve_readout(meta.get("readout") or "pool")
    n_layers = int(meta.get("n_layers") or 2)
    model = GraphRiskModel(
        n_nodes=n_nodes,
        branch_mode=branch,
        readout=readout,
        n_layers=n_layers,
        use_node_length=bool(meta.get("use_node_length", False)),
    ).to(device)
    state = ckpt.get("best_state") or ckpt["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def _load_low_conf_lookup(pack_dir: Path) -> dict[tuple[int, int], bool]:
    """(stay_id, node_id) → family_low_confidence from the events pack, if present."""
    events_dir = None
    manifest = pack_dir / "graph_manifest.json"
    if manifest.is_file():
        try:
            events_dir = Path(json.loads(manifest.read_text(encoding="utf-8")).get("events_dir") or "")
        except (OSError, json.JSONDecodeError, TypeError):
            events_dir = None
    if events_dir is None or not events_dir.is_dir():
        return {}
    act = events_dir / "activations_long.parquet"
    if not act.is_file():
        return {}
    ev = pd.read_parquet(act)
    if "family_low_confidence" not in ev.columns:
        return {}
    if "split" in ev.columns:
        ev = ev[ev["split"] == "val"]
    stay_col = "ICUSTAY_ID" if "ICUSTAY_ID" in ev.columns else "stay_id"
    out: dict[tuple[int, int], bool] = {}
    for rec in ev[[stay_col, "node_id", "family_low_confidence"]].itertuples(index=False):
        out[(int(rec[0]), int(rec[1]))] = bool(rec[2])
    return out


def _edge_match_features(
    batch: dict[str, torch.Tensor],
    *,
    stage_id_by_node: dict[int, int],
    low_conf_lookup: dict[tuple[int, int], bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (src_stage_id, any_low) on the edge grid."""
    bsz, emax = batch["edge_mask"].shape
    src_stage = torch.zeros(bsz, emax, dtype=torch.long, device=batch["edge_mask"].device)
    any_low = torch.zeros(bsz, emax, dtype=torch.bool, device=batch["edge_mask"].device)
    stays = batch["stay_id"].detach().cpu().tolist()
    src = batch["edge_src"].detach().cpu()
    dst = batch["edge_dst"].detach().cpu()
    mask = batch["edge_mask"].detach().cpu()
    for i in range(bsz):
        stay = int(stays[i])
        for e in torch.nonzero(mask[i], as_tuple=False).flatten().tolist():
            sid = int(src[i, e].item())
            did = int(dst[i, e].item())
            src_stage[i, e] = int(stage_id_by_node.get(sid, 0))
            if low_conf_lookup:
                any_low[i, e] = bool(
                    low_conf_lookup.get((stay, sid), False)
                    or low_conf_lookup.get((stay, did), False)
                )
    return src_stage, any_low


def _stage_width_vector(node_dictionary: list[dict]) -> torch.Tensor:
    widths = {name: float(hi - lo) for name, lo, hi in STAGE_BOUNDS}
    return torch.tensor([widths[str(row["stage"])] for row in node_dictionary], dtype=torch.float32)


def _jitter_graph_inputs(
    batch: dict[str, torch.Tensor],
    *,
    stage_widths: torch.Tensor,
    seed: int,
    batch_index: int,
    activation_sd: float,
    time_jitter_hours: float,
) -> dict[str, torch.Tensor]:
    """Small feature perturbation with fixed topology for local stability only."""
    rng = np.random.default_rng(int(seed) + int(batch_index) * 10_007)
    shape = tuple(batch["a_dense"].shape)
    da = torch.as_tensor(
        rng.normal(0.0, activation_sd, size=shape),
        device=batch["a_dense"].device,
        dtype=batch["a_dense"].dtype,
    )
    dh = torch.as_tensor(
        rng.uniform(-time_jitter_hours, time_jitter_hours, size=shape),
        device=batch["a_dense"].device,
        dtype=batch["a_dense"].dtype,
    )
    active = batch["active_mask"].to(batch["a_dense"].dtype)
    out = dict(batch)
    out["a_dense"] = torch.clamp(batch["a_dense"] + da * active, 0.0, 1.0)
    out["node_time"] = torch.clamp(
        batch["node_time"] + (dh / 48.0) * active,
        0.0,
        1.0,
    )
    widths = stage_widths.to(batch["a_dense"].device).unsqueeze(0)
    out["node_time_within_stage"] = torch.clamp(
        batch["node_time_within_stage"] + (dh / widths) * active,
        0.0,
        1.0,
    )
    return out


def _aggregate_node_attention(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["node_id", "var", "L", "k", "stage"]
    base = (
        df.groupby(keys, observed=True)
        .agg(
            patient_support=("stay_id", "nunique"),
            death_support=("y", "sum"),
            mean_attention=("attention", "mean"),
            mean_activation=("activation", "mean"),
            mean_time_h=("time_h", "mean"),
            top1_count=("rank", lambda x: int((x == 1).sum())),
            top_fraction_count=("is_top_fraction", "sum"),
        )
        .reset_index()
    )
    for label, name in [(1.0, "death"), (0.0, "survival")]:
        part = (
            df[df["y"] == label]
            .groupby(keys, observed=True)["attention"]
            .mean()
            .rename(f"mean_attention_{name}")
            .reset_index()
        )
        base = base.merge(part, on=keys, how="left")
    base["attention_death_minus_survival"] = (
        base["mean_attention_death"] - base["mean_attention_survival"]
    )
    return base.sort_values(["mean_attention", "patient_support"], ascending=False)


def _aggregate_edge_attention(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "src_id",
        "src_name",
        "dst_id",
        "dst_name",
        "edge_type",
    ]
    base = (
        df.groupby(keys, observed=True)
        .agg(
            patient_support=("stay_id", "nunique"),
            death_support=("y", "sum"),
            mean_attention=("attention", "mean"),
            mean_delta_t_h=("delta_t_h", "mean"),
            mean_activation_product=("activation_product", "mean"),
            top1_count=("rank", lambda x: int((x == 1).sum())),
            top_fraction_count=("is_top_fraction", "sum"),
            **(
                {
                    "mean_selection_probability": ("selection_probability", "mean"),
                    "mean_hard_mask": ("hard_mask", "mean"),
                    "mean_signed_value": ("signed_value", "mean"),
                    "selected_count": ("hard_mask", "sum"),
                }
                if "hard_mask" in df.columns
                else {}
            ),
        )
        .reset_index()
    )
    for label, name in [(1.0, "death"), (0.0, "survival")]:
        part = (
            df[df["y"] == label]
            .groupby(keys, observed=True)["attention"]
            .mean()
            .rename(f"mean_attention_{name}")
            .reset_index()
        )
        base = base.merge(part, on=keys, how="left")
    base["attention_death_minus_survival"] = (
        base["mean_attention_death"] - base["mean_attention_survival"]
    )
    return base.sort_values(["mean_attention", "patient_support"], ascending=False)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)
    pack_dir = Path(args.graph_pack)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, pack_meta = _load_val_pack(pack_dir, args.max_stays)
    if not len(dataset):
        raise RuntimeError("empty validation dataset")
    node_dictionary = json.loads((pack_dir / "node_dictionary.json").read_text(encoding="utf-8"))
    if len(node_dictionary) != dataset.n_nodes:
        raise ValueError("node dictionary and graph pack disagree")
    node_by_id = {int(x["node_id"]): x for x in node_dictionary}
    stage_widths = _stage_width_vector(node_dictionary)
    model, ckpt = _load_model(ckpt_path, dataset.n_nodes, device)
    readout = str(dict(ckpt.get("meta") or {}).get("readout") or "pool")
    low_conf_lookup = _load_low_conf_lookup(pack_dir)
    stage_id_by_node = {}
    for nid, nd in node_by_id.items():
        folded = fold_stage_label(str(nd.get("stage") or ""))
        if "stage_idx" in nd and nd["stage_idx"] is not None:
            stage_id_by_node[nid] = int(nd["stage_idx"])
        else:
            stage_id_by_node[nid] = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}.get(folded, 0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_batch,
    )

    rows: dict[str, list[float | int]] = {
        k: []
        for k in [
            "stay_id", "subject_id", "y", "n_nodes", "n_edges",
            "base_logit", "base_prob", "node_top_logit", "node_top_prob",
            "edge_top_logit", "edge_top_prob", "node_random_logit",
            "node_random_prob", "edge_random_logit", "edge_random_prob",
            "node_top_abs_delta_logit", "node_random_abs_delta_logit",
            "node_top_signed_delta_logit", "node_random_signed_delta_logit",
            "edge_top_abs_delta_logit", "edge_random_abs_delta_logit",
            "edge_top_signed_delta_logit", "edge_random_signed_delta_logit",
            "node_attention_entropy", "edge_attention_entropy",
            "node_jitter_jaccard", "edge_jitter_jaccard",
            "jitter_abs_delta_prob",
            "n_selected", "mean_selection_prob",
            "edge_jitter_jaccard_nonempty",
        ]
    }
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(tqdm(loader, desc="interpret val", unit="batch")):
            batch = {k: v.to(device) for k, v in cpu_batch.items()}
            base = model(batch, compute_loss=False)
            top_nodes = attention_topk_mask(
                base.node_attention, batch["active_mask"], fraction=args.top_fraction
            )
            if readout == "additive_topk" and base.edge_gate is not None:
                # Delete the precomputed hard top-k, not a re-ranked attention set.
                top_edges = base.edge_gate > 0.5
            else:
                top_edges = attention_topk_mask(
                    base.edge_attention, batch["edge_mask"], fraction=args.top_fraction
                )
            src_stage_id, any_low = _edge_match_features(
                batch,
                stage_id_by_node=stage_id_by_node,
                low_conf_lookup=low_conf_lookup,
            )
            node_top = model(perturb_nodes(batch, top_nodes), compute_loss=False)
            edge_top = model(perturb_edges(batch, top_edges), compute_loss=False)

            node_rand_prob = torch.zeros_like(base.prob)
            node_rand_logit = torch.zeros_like(base.logits)
            node_rand_abs = torch.zeros_like(base.logits)
            node_rand_signed = torch.zeros_like(base.logits)
            edge_rand_prob = torch.zeros_like(base.prob)
            edge_rand_logit = torch.zeros_like(base.logits)
            edge_rand_abs = torch.zeros_like(base.logits)
            edge_rand_signed = torch.zeros_like(base.logits)
            for repeat in range(args.random_repeats):
                rn = random_valid_mask(
                    batch["active_mask"], fraction=args.top_fraction,
                    keys=batch["stay_id"], seed=args.seed, repeat=repeat,
                )
                if readout == "additive_topk":
                    re = matched_random_edge_mask(
                        batch["edge_mask"],
                        top_edges,
                        batch["edge_type"],
                        src_stage_id=src_stage_id,
                        any_low=any_low if low_conf_lookup else None,
                        keys=batch["stay_id"],
                        seed=args.seed + 97,
                        repeat=repeat,
                    )
                else:
                    re = random_valid_mask(
                        batch["edge_mask"], fraction=args.top_fraction,
                        keys=batch["stay_id"], seed=args.seed + 97, repeat=repeat,
                    )
                on = model(perturb_nodes(batch, rn), compute_loss=False)
                oe = model(perturb_edges(batch, re), compute_loss=False)
                dn = base.logits - on.logits
                de = base.logits - oe.logits
                node_rand_prob += on.prob
                node_rand_logit += on.logits
                node_rand_abs += dn.abs()
                node_rand_signed += dn
                edge_rand_prob += oe.prob
                edge_rand_logit += oe.logits
                edge_rand_abs += de.abs()
                edge_rand_signed += de
            scale = float(args.random_repeats)
            node_rand_prob /= scale
            node_rand_logit /= scale
            node_rand_abs /= scale
            node_rand_signed /= scale
            edge_rand_prob /= scale
            edge_rand_logit /= scale
            edge_rand_abs /= scale
            edge_rand_signed /= scale

            jittered_batch = _jitter_graph_inputs(
                batch,
                stage_widths=stage_widths,
                seed=args.seed + 313,
                batch_index=batch_index,
                activation_sd=args.activation_jitter_sd,
                time_jitter_hours=args.time_jitter_hours,
            )
            jittered = model(jittered_batch, compute_loss=False)
            node_jaccard = topk_jaccard(
                base.node_attention, jittered.node_attention,
                batch["active_mask"], fraction=args.top_fraction,
            )
            if readout == "additive_topk" and base.edge_gate is not None and jittered.edge_gate is not None:
                edge_jaccard = mask_jaccard(
                    base.edge_gate > 0.5, jittered.edge_gate > 0.5, batch["edge_mask"]
                )
            else:
                edge_jaccard = topk_jaccard(
                    base.edge_attention, jittered.edge_attention,
                    batch["edge_mask"], fraction=args.top_fraction,
                )
            node_entropy = normalized_attention_entropy(
                base.node_attention, batch["active_mask"]
            )
            edge_entropy = normalized_attention_entropy(
                base.edge_attention, batch["edge_mask"]
            )

            tensors = {
                "stay_id": batch["stay_id"],
                "subject_id": batch["subject_id"],
                "y": batch["y"],
                "n_nodes": batch["active_mask"].sum(dim=1),
                "n_edges": batch["edge_mask"].sum(dim=1),
                "base_logit": base.logits,
                "base_prob": base.prob,
                "node_top_logit": node_top.logits,
                "node_top_prob": node_top.prob,
                "edge_top_logit": edge_top.logits,
                "edge_top_prob": edge_top.prob,
                "node_random_logit": node_rand_logit,
                "node_random_prob": node_rand_prob,
                "edge_random_logit": edge_rand_logit,
                "edge_random_prob": edge_rand_prob,
                "node_top_abs_delta_logit": (base.logits - node_top.logits).abs(),
                "node_random_abs_delta_logit": node_rand_abs,
                "node_top_signed_delta_logit": base.logits - node_top.logits,
                "node_random_signed_delta_logit": node_rand_signed,
                "edge_top_abs_delta_logit": (base.logits - edge_top.logits).abs(),
                "edge_random_abs_delta_logit": edge_rand_abs,
                "edge_top_signed_delta_logit": base.logits - edge_top.logits,
                "edge_random_signed_delta_logit": edge_rand_signed,
                "node_attention_entropy": node_entropy,
                "edge_attention_entropy": edge_entropy,
                "node_jitter_jaccard": node_jaccard,
                "edge_jitter_jaccard": edge_jaccard,
                "jitter_abs_delta_prob": (base.prob - jittered.prob).abs(),
                "n_selected": (
                    (base.edge_gate * batch["edge_mask"].to(base.edge_gate.dtype)).sum(dim=1)
                    if base.edge_gate is not None
                    else batch["edge_mask"].new_zeros(batch["edge_mask"].shape[0])
                ),
                "edge_jitter_jaccard_nonempty": torch.where(
                    (
                        (base.edge_gate * batch["edge_mask"].to(base.edge_gate.dtype)).sum(dim=1) > 0
                        if base.edge_gate is not None
                        else batch["edge_mask"].any(dim=1)
                    ),
                    edge_jaccard,
                    edge_jaccard.new_full(edge_jaccard.shape, float("nan")),
                ),
                "mean_selection_prob": (
                    (
                        (base.edge_selection_prob * batch["edge_mask"].to(base.edge_selection_prob.dtype)).sum(dim=1)
                        / batch["edge_mask"].to(dtype=base.logits.dtype).sum(dim=1).clamp_min(1.0)
                    )
                    if base.edge_selection_prob is not None
                    else batch["edge_mask"].new_zeros(batch["edge_mask"].shape[0], dtype=base.logits.dtype)
                ),
            }
            for name, tensor in tensors.items():
                rows[name].extend(tensor.detach().cpu().numpy().tolist())

            # Long-form attention exports retain graph topology for audit/case studies.
            bsz = batch["y"].shape[0]
            for i in range(bsz):
                stay = int(batch["stay_id"][i].item())
                subject = int(batch["subject_id"][i].item())
                y_i = float(batch["y"][i].item())
                active_ids = torch.nonzero(batch["active_mask"][i], as_tuple=False).flatten()
                if active_ids.numel():
                    order = active_ids[
                        torch.argsort(base.node_attention[i, active_ids], descending=True)
                    ]
                    ranks = {int(n.item()): rank + 1 for rank, n in enumerate(order)}
                    for nid_t in active_ids:
                        nid = int(nid_t.item())
                        nd = node_by_id[nid]
                        L_event = None
                        if "node_length" in batch:
                            L_event = float(batch["node_length"][i, nid].item() * 12.0)
                        node_rows.append(
                            {
                                "stay_id": stay,
                                "subject_id": subject,
                                "y": y_i,
                                "node_id": nid,
                                "var": nd["var"],
                                "L": L_event if L_event is not None else _opt_int(nd.get("L")),
                                "k": _opt_int(nd.get("k")),
                                "motif_family": _opt_int(nd.get("motif_family")),
                                "family_label": nd.get("family_label"),
                                "stage": nd["stage"],
                                "activation": float(batch["a_dense"][i, nid].item()),
                                "time_h": float(batch["node_time"][i, nid].item() * 48.0),
                                "attention": float(base.node_attention[i, nid].item()),
                                "contrib": (
                                    float(base.node_contrib[i, nid].item())
                                    if base.node_contrib is not None
                                    else float("nan")
                                ),
                                "rank": ranks[nid],
                                "is_top_fraction": bool(top_nodes[i, nid].item()),
                            }
                        )
                edge_ids = torch.nonzero(batch["edge_mask"][i], as_tuple=False).flatten()
                if edge_ids.numel():
                    order = edge_ids[
                        torch.argsort(base.edge_attention[i, edge_ids], descending=True)
                    ]
                    ranks = {int(e.item()): rank + 1 for rank, e in enumerate(order)}
                    for eid_t in edge_ids:
                        eid = int(eid_t.item())
                        src = int(batch["edge_src"][i, eid].item())
                        dst = int(batch["edge_dst"][i, eid].item())
                        et = int(batch["edge_type"][i, eid].item())
                        attr = batch["edge_attr"][i, eid]
                        edge_rows.append(
                            {
                                "stay_id": stay,
                                "subject_id": subject,
                                "y": y_i,
                                "edge_slot": eid,
                                "src_id": src,
                                "src_name": node_by_id[src]["activation_name"],
                                "src_family": _opt_int(node_by_id[src].get("motif_family")),
                                "dst_id": dst,
                                "dst_name": node_by_id[dst]["activation_name"],
                                "dst_family": _opt_int(node_by_id[dst].get("motif_family")),
                                "edge_type": EDGE_TYPE_NAMES[et],
                                "delta_t_h": float(attr[0].item() * 12.0),
                                "activation_product": float(attr[1].item()),
                                "same_variable": float(attr[2].item()),
                                "same_stage": float(attr[3].item()),
                                "support_normalized": float(attr[4].item()),
                                "attention": float(base.edge_attention[i, eid].item()),
                                "contrib": (
                                    float(base.edge_contrib[i, eid].item())
                                    if base.edge_contrib is not None
                                    else float("nan")
                                ),
                                "selection_probability": (
                                    float(base.edge_selection_prob[i, eid].item())
                                    if base.edge_selection_prob is not None
                                    else float("nan")
                                ),
                                "hard_mask": (
                                    float(base.edge_gate[i, eid].item())
                                    if base.edge_gate is not None
                                    else float("nan")
                                ),
                                "signed_value": (
                                    float(base.edge_value[i, eid].item())
                                    if base.edge_value is not None
                                    else float("nan")
                                ),
                                "message_attention": (
                                    float(base.message_attention[i, eid].item())
                                    if base.message_attention is not None
                                    else float("nan")
                                ),
                                "rank": ranks[eid],
                                "is_top_fraction": bool(top_edges[i, eid].item()),
                            }
                        )

    patient_df = pd.DataFrame(rows)
    node_df = pd.DataFrame(node_rows)
    edge_df = pd.DataFrame(edge_rows)
    patient_df.to_parquet(out_dir / "patient_fidelity_val.parquet", index=False)
    node_df.to_parquet(out_dir / "node_attention_val.parquet", index=False)
    edge_df.to_parquet(out_dir / "edge_attention_val.parquet", index=False)
    node_summary = _aggregate_node_attention(node_df)
    edge_summary = _aggregate_edge_attention(edge_df)
    node_summary.to_csv(out_dir / "node_attention_summary.csv", index=False)
    edge_summary.to_csv(out_dir / "edge_attention_summary.csv", index=False)
    path_rows = extract_precedes_paths(edge_rows, top_k_per_patient=5)
    if path_rows:
        path_df = pd.DataFrame(path_rows)
        path_df.to_parquet(out_dir / "precedes_paths_val.parquet", index=False)
        path_summary = (
            path_df.groupby(["src_name", "mid_name", "dst_name"], observed=True)
            .agg(
                patient_support=("stay_id", "nunique"),
                death_support=("y", "sum"),
                mean_abs_score=("abs_score", "mean"),
                mean_signed_score=("signed_score", "mean"),
            )
            .reset_index()
            .sort_values(["patient_support", "mean_abs_score"], ascending=False)
        )
        path_summary.to_csv(out_dir / "precedes_paths_summary.csv", index=False)
    else:
        path_summary = pd.DataFrame()

    y = patient_df["y"].to_numpy(dtype=np.float64)
    has_e = patient_df["n_edges"] > 0
    l0_selection = {
        "mean_n_selected": float(patient_df.loc[has_e, "n_selected"].mean())
        if bool(has_e.any())
        else float("nan"),
        "median_n_selected": float(patient_df.loc[has_e, "n_selected"].median())
        if bool(has_e.any())
        else float("nan"),
        "mean_n_edges": float(patient_df.loc[has_e, "n_edges"].mean())
        if bool(has_e.any())
        else float("nan"),
        "mean_selection_prob": float(patient_df.loc[has_e, "mean_selection_prob"].mean())
        if bool(has_e.any())
        else float("nan"),
        "frac_all_zero_mask": float(
            ((patient_df["n_selected"] <= 0) & has_e).sum() / has_e.sum()
        )
        if bool(has_e.any())
        else float("nan"),
        "frac_all_one_mask": float(
            ((patient_df["n_selected"] >= patient_df["n_edges"]) & has_e).sum() / has_e.sum()
        )
        if bool(has_e.any())
        else float("nan"),
        "n_patients_with_edges": int(has_e.sum()),
        "edge_topk_jaccard_nonempty_mean": float(
            patient_df.loc[patient_df["n_selected"] > 0, "edge_jitter_jaccard"].mean()
        )
        if bool((patient_df["n_selected"] > 0).any())
        else float("nan"),
        "n_patients_nonempty_selection": int((patient_df["n_selected"] > 0).sum()),
    }
    prediction_metrics = {
        "baseline": _metrics(y, patient_df["base_prob"].to_numpy()),
        "node_top_deleted": _metrics(y, patient_df["node_top_prob"].to_numpy()),
        "node_random_deleted_mean": _metrics(y, patient_df["node_random_prob"].to_numpy()),
        "edge_top_deleted": _metrics(y, patient_df["edge_top_prob"].to_numpy()),
        "edge_random_deleted_mean": _metrics(y, patient_df["edge_random_prob"].to_numpy()),
    }

    fidelity: dict[str, Any] = {}
    for kind in ["node", "edge"]:
        eligible = patient_df[f"n_{kind}s"] > 0
        top_abs = patient_df.loc[eligible, f"{kind}_top_abs_delta_logit"].to_numpy()
        random_abs = patient_df.loc[eligible, f"{kind}_random_abs_delta_logit"].to_numpy()
        top_signed = patient_df.loc[eligible, f"{kind}_top_signed_delta_logit"].to_numpy()
        random_signed = patient_df.loc[eligible, f"{kind}_random_signed_delta_logit"].to_numpy()
        fidelity[kind] = {
            "n_eligible": int(eligible.sum()),
            "top_mean_abs_delta_logit": float(top_abs.mean()),
            "random_mean_abs_delta_logit": float(random_abs.mean()),
            "top_mean_signed_delta_logit": float(top_signed.mean()),
            "random_mean_signed_delta_logit": float(random_signed.mean()),
            "fraction_top_abs_gt_random": float(np.mean(top_abs > random_abs)),
            "mean_abs_excess_top_minus_random": _bootstrap_mean_ci(
                top_abs - random_abs,
                n_boot=args.bootstrap_repeats,
                seed=args.seed + (1 if kind == "node" else 2),
            ),
            "auprc_random_minus_top": _bootstrap_metric_difference(
                y,
                patient_df[f"{kind}_random_prob"].to_numpy(),
                patient_df[f"{kind}_top_prob"].to_numpy(),
                metric="auprc",
                n_boot=args.bootstrap_repeats,
                seed=args.seed + (11 if kind == "node" else 12),
            ),
            "auroc_random_minus_top": _bootstrap_metric_difference(
                y,
                patient_df[f"{kind}_random_prob"].to_numpy(),
                patient_df[f"{kind}_top_prob"].to_numpy(),
                metric="auroc",
                n_boot=args.bootstrap_repeats,
                seed=args.seed + (21 if kind == "node" else 22),
            ),
        }
        for label, group_name in [(1.0, "death"), (0.0, "survival")]:
            group = eligible & (patient_df["y"] == label)
            fidelity[kind][f"top_mean_signed_delta_logit_{group_name}"] = float(
                patient_df.loc[group, f"{kind}_top_signed_delta_logit"].mean()
            )

    best_epoch = None
    history = list(ckpt.get("history") or [])
    if history:
        valid_history = [x for x in history if math.isfinite(float(x.get("val_auprc", float("nan"))))]
        if valid_history:
            best_epoch = int(max(valid_history, key=lambda x: x["val_auprc"])["epoch"])
    summary = {
        "stage": "GNN_interpretability_validation",
        "split": "val",
        "test_read": False,
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_best_epoch": best_epoch,
        "graph_pack": str(pack_dir.resolve()),
        "n_patients": int(len(patient_df)),
        "n_empty_graphs": int((patient_df["n_nodes"] == 0).sum()),
        "device": str(device),
        "top_fraction": float(args.top_fraction),
        "random_repeats": int(args.random_repeats),
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "deletion_contract": (
            "node deletion changes active_mask and incident edge_mask; edge deletion changes "
            "edge_mask only; a_dense/sparse/static remain fixed"
        ),
        "prediction_metrics": prediction_metrics,
        "fidelity": fidelity,
        "branch_mode": dict(ckpt.get("meta") or {}).get("branch_mode"),
        "readout": dict(ckpt.get("meta") or {}).get("readout", "pool"),
        "attention": {
            "node_normalized_entropy_mean": float(patient_df["node_attention_entropy"].mean()),
            "edge_normalized_entropy_mean": float(patient_df["edge_attention_entropy"].mean()),
            "note": (
                "lower entropy means more concentrated attention; "
                "additive_l0 ranks edges by selection probability, not softmax"
            ),
        },
        "l0_selection": l0_selection,
        "additive": {
            "has_signed_contrib": bool(
                "contrib" in node_df.columns and node_df["contrib"].notna().any()
            )
            if len(node_df)
            else False,
            "n_precedes_paths": int(len(path_rows)),
            "top_paths": (
                path_summary.head(10).to_dict("records") if len(path_summary) else []
            ),
        },
        "local_stability_fixed_topology": {
            "activation_jitter_sd": float(args.activation_jitter_sd),
            "time_jitter_hours": float(args.time_jitter_hours),
            "node_topk_jaccard_mean": float(patient_df["node_jitter_jaccard"].mean()),
            "node_topk_jaccard_median": float(patient_df["node_jitter_jaccard"].median()),
            "edge_topk_jaccard_mean": float(patient_df["edge_jitter_jaccard"].mean()),
            "edge_topk_jaccard_median": float(patient_df["edge_jitter_jaccard"].median()),
            "edge_topk_jaccard_nonempty_mean": float(
                patient_df.loc[patient_df["n_selected"] > 0, "edge_jitter_jaccard"].mean()
            )
            if bool((patient_df["n_selected"] > 0).any())
            else float("nan"),
            "mean_abs_probability_change": float(patient_df["jitter_abs_delta_prob"].mean()),
            "limitation": "topology was held fixed; this does not test threshold/edge-rebuild stability",
        },
        "top_nodes_by_mean_attention": node_summary.head(20).to_dict(orient="records"),
        "top_edges_by_mean_attention": edge_summary.head(20).to_dict(orient="records"),
        "pack_meta": pack_meta,
        "elapsed_seconds": float(time.time() - started),
    }
    _json_dump(out_dir / "interpretability_summary.json", summary)
    _write_report(Path(args.report), summary)
    return summary


def _fmt_ci(x: dict[str, float]) -> str:
    return f"{x['estimate']:.4f} [{x['ci_low']:.4f}, {x['ci_high']:.4f}]"


def _write_report(path: Path, s: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = s["prediction_metrics"]
    f = s["fidelity"]
    st = s["local_stability_fixed_topology"]
    lines = [
        "# GNN可解释性验证",
        "",
        "> 冻结低学习率完整GNN，仅使用validation；未读取test。注意力是关联性权重，不解释为因果效应。",
        "",
        "## 1. 协议",
        "",
        f"- 患者数：{s['n_patients']}；空图：{s['n_empty_graphs']}。",
        f"- 删除比例：每位患者有效节点/边的top {s['top_fraction']:.0%}。",
        f"- 随机对照：{s['random_repeats']}次等数量删除。",
        "- 节点删除仅改变active_mask及其关联边；252维激活、308维阶段特征和静态特征保持不变。",
        "",
        "## 2. 删除后整体预测表现",
        "",
        "| 条件 | AUPRC | AUROC | Brier |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "baseline": "完整图",
        "node_top_deleted": "删除top节点",
        "node_random_deleted_mean": "删除随机节点",
        "edge_top_deleted": "删除top边",
        "edge_random_deleted_mean": "删除随机边",
    }
    for key in labels:
        x = m[key]
        lines.append(f"| {labels[key]} | {x['auprc']:.4f} | {x['auroc']:.4f} | {x['brier']:.4f} |")
    lines.extend(
        [
            "",
            "## 3. 忠实度",
            "",
            "正的“top−random绝对logit变化”表示删除高注意力结构比随机删除影响更大。",
            "",
            "| 结构 | top平均绝对Δlogit | random平均绝对Δlogit | top−random（95% CI） | AUPRC(random−top, 95% CI) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for kind, label in [("node", "节点"), ("edge", "边")]:
        x = f[kind]
        lines.append(
            f"| {label} | {x['top_mean_abs_delta_logit']:.4f} | "
            f"{x['random_mean_abs_delta_logit']:.4f} | "
            f"{_fmt_ci(x['mean_abs_excess_top_minus_random'])} | "
            f"{_fmt_ci(x['auprc_random_minus_top'])} |"
        )
    lines.extend(
        [
            "",
            "## 4. 有符号风险变化与注意力集中度",
            "",
            "`Δlogit=完整图−删除后图`；正值表示该组结构支持较高风险，负值表示删除后风险反而升高。",
            "",
            "| 结构 | 全部患者平均Δlogit | 死亡患者 | 存活患者 | 归一化注意力熵 |",
            "|---|---:|---:|---:|---:|",
            f"| 节点 | {f['node']['top_mean_signed_delta_logit']:.4f} | "
            f"{f['node']['top_mean_signed_delta_logit_death']:.4f} | "
            f"{f['node']['top_mean_signed_delta_logit_survival']:.4f} | "
            f"{s['attention']['node_normalized_entropy_mean']:.4f} |",
            f"| 边 | {f['edge']['top_mean_signed_delta_logit']:.4f} | "
            f"{f['edge']['top_mean_signed_delta_logit_death']:.4f} | "
            f"{f['edge']['top_mean_signed_delta_logit_survival']:.4f} | "
            f"{s['attention']['edge_normalized_entropy_mean']:.4f} |",
            "",
            "边注意力熵接近1表示权重接近均匀，不能据此强行提取少数“关键死亡边”。",
            "",
            "## 5. 局部稳定性（固定拓扑）",
            "",
            f"- 节点top集合Jaccard：均值 {st['node_topk_jaccard_mean']:.4f}，中位数 {st['node_topk_jaccard_median']:.4f}。",
            f"- 边top集合Jaccard：均值 {st['edge_topk_jaccard_mean']:.4f}，中位数 {st['edge_topk_jaccard_median']:.4f}。",
            (
                f"- 原始选择非空患者的边 Jaccard：{st.get('edge_topk_jaccard_nonempty_mean', float('nan')):.4f}。"
                if st.get("edge_topk_jaccard_nonempty_mean") is not None
                else ""
            ),
            f"- 小扰动后平均绝对概率变化：{st['mean_abs_probability_change']:.6f}。",
            "- 此项只验证编码器在固定拓扑下的局部稳定性，不代表a=0.5阈值及重建边后的拓扑稳定性。",
            "",
        ]
    )
    l0 = s.get("l0_selection") or {}
    if s.get("readout") == "additive_l0":
        lines.extend(
            [
                "## 5.1 Hard-Concrete 边选择",
                "",
                f"- 有边患者平均原始边数：{l0.get('mean_n_edges', float('nan')):.2f}。",
                f"- 有边患者平均 hard mask 选中边数：{l0.get('mean_n_selected', float('nan')):.2f}（中位数 {l0.get('median_n_selected', float('nan')):.1f}）。",
                f"- 平均选择概率 π：{l0.get('mean_selection_prob', float('nan')):.4f}。",
                f"- 全 0 mask 患者比例：{l0.get('frac_all_zero_mask', float('nan')):.3f}；全 1：{l0.get('frac_all_one_mask', float('nan')):.3f}。",
                "- 删除验证按选择概率 π 排序；hard mask 同时关闭两层消息传递和边直接贡献。",
                "",
            ]
        )
    node_ci = f["node"]["auprc_random_minus_top"]
    edge_ci = f["edge"]["auprc_random_minus_top"]
    node_ok = float(node_ci["ci_low"]) > 0
    edge_ok = float(edge_ci["ci_low"]) > 0
    lines.extend(
        [
            "## 6. 结论",
            "",
            (
                "- **节点删除忠实度**：AUPRC(random−top) 置信下限大于 0。"
                if node_ok
                else "- **节点删除忠实度未过关**：AUPRC(random−top) 的 95% 区间仍跨 0 或为负。"
            ),
            (
                "- **边删除忠实度**：按排序删除 top 边的 AUPRC 影响显著大于等数量随机边。"
                if edge_ok
                else "- **边删除忠实度未过关**：top 边相对随机边的 AUPRC 差异 95% 区间仍跨 0 或为负。"
            ),
            (
                f"- **边选择**：平均选中 {l0.get('mean_n_selected', float('nan')):.1f} / "
                f"{l0.get('mean_n_edges', float('nan')):.1f} 条；"
                f"全 0 比例 {l0.get('frac_all_zero_mask', float('nan')):.3f}，"
                f"全 1 比例 {l0.get('frac_all_one_mask', float('nan')):.3f}。"
                if s.get("readout") == "additive_l0"
                else f"- **边注意力熵**：{s['attention']['edge_normalized_entropy_mean']:.3f}（接近 1 表示接近均匀）。"
            ),
            f"- **固定拓扑局部稳定**：边 top Jaccard 均值 {st['edge_topk_jaccard_mean']:.4f}。",
            "- 高注意力/高选择概率边称为“风险相关结构”，不得称为因果路径。",
            "",
            "## 7. 解释边界",
            "",
            "- 注意力或选择概率不提供风险方向；方向由删除前后有符号Δlogit判断。",
            "- 高注意力节点/边称为“风险相关结构”，不得称为因果路径。",
            "- h_graph是压缩表示，不是患者全部路径的可逆清单。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-pack", default=str(DEFAULT_PACK))
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--output", default=str(DEFAULT_OUT))
    p.add_argument("--report", default=str(DEFAULT_REPORT))
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--top-fraction", type=float, default=0.20)
    p.add_argument("--random-repeats", type=int, default=10)
    p.add_argument("--bootstrap-repeats", type=int, default=500)
    p.add_argument("--activation-jitter-sd", type=float, default=0.01)
    p.add_argument("--time-jitter-hours", type=float, default=1.0)
    p.add_argument("--max-stays", type=int)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
