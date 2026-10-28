#!/usr/bin/env python3
"""Path-level deletion fidelity for the frozen additive-G model.

Validation only. A 2-hop PRECEDES path is deleted by removing both of its
edges (nodes and other edges stay). Interaction is
``Δpath − Δedge1 − Δedge2``. Test packs are refused.
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

from scripts.utils.graph_data import GraphPackDataset, collate_graph_batch  # noqa: E402
from scripts.utils.graph_interpretability import (  # noqa: E402
    enumerate_precedes_paths_batch,
    fold_path_pattern,
    path_interaction,
    perturb_edges,
    select_random_indices,
    select_top_positive_indices,
    union_path_edge_mask,
)
from scripts.utils.graph_model import (  # noqa: E402
    GraphRiskModel,
    resolve_branch_mode,
    resolve_readout,
)


DEFAULT_PACK = ROOT / "data/processed/graphs/graph_final_7var_L4_6_12_K3"
DEFAULT_CKPT = ROOT / "data/processed/gnn/gnn_additive_G_graph_static/ckpt_best.pt"
DEFAULT_OUT = ROOT / "data/processed/gnn/gnn_path_fidelity_additive_G"
DEFAULT_REPORT = ROOT / "outputs/reports/19_GNN路径级忠实度.md"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def _fmt_ci(d: dict[str, float]) -> str:
    return f"{d['estimate']:.4f} [{d['ci_low']:.4f}, {d['ci_high']:.4f}]"


def _ci_excludes_zero(d: dict[str, float]) -> bool:
    return bool(d["ci_low"] > 0.0 or d["ci_high"] < 0.0)


def _risk_direction(delta: float) -> str:
    if delta > 0.0:
        return "risk"
    if delta < 0.0:
        return "protective"
    return "neutral"


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
    model = GraphRiskModel(
        n_nodes=n_nodes,
        branch_mode=resolve_branch_mode(meta.get("branch_mode") or "G"),
        readout=resolve_readout(meta.get("readout") or "additive"),
        n_layers=int(meta.get("n_layers") or 2),
        use_node_length=bool(meta.get("use_node_length", False)),
    ).to(device)
    state = ckpt.get("best_state") or ckpt["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def _forward(model: GraphRiskModel, batch: dict[str, torch.Tensor]):
    return model(batch, compute_loss=False)


def _edge_delta_map(
    model: GraphRiskModel,
    batch: dict[str, torch.Tensor],
    base_logits: torch.Tensor,
    slots_by_row: list[list[int]],
) -> list[dict[int, float]]:
    maps: list[dict[int, float]] = [{} for _ in slots_by_row]
    max_n = max((len(s) for s in slots_by_row), default=0)
    for t in range(max_n):
        selected = torch.zeros_like(batch["edge_mask"])
        active_rows = False
        for i, slots in enumerate(slots_by_row):
            if t < len(slots):
                selected[i, slots[t]] = True
                active_rows = True
        if not active_rows:
            continue
        out = _forward(model, perturb_edges(batch, selected))
        delta = (base_logits - out.logits).detach()
        for i, slots in enumerate(slots_by_row):
            if t < len(slots):
                maps[i][slots[t]] = float(delta[i].item())
    return maps


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
    node_dictionary = json.loads((pack_dir / "node_dictionary.json").read_text(encoding="utf-8"))
    if len(node_dictionary) != dataset.n_nodes:
        raise ValueError("node dictionary and graph pack disagree")
    node_by_id = {int(x["node_id"]): x for x in node_dictionary}
    model, ckpt = _load_model(ckpt_path, dataset.n_nodes, device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_batch,
    )

    patient_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for cpu_batch in tqdm(loader, desc="path fidelity val", unit="batch"):
            batch = {k: v.to(device) for k, v in cpu_batch.items()}
            base = _forward(model, batch)
            paths_by_row = enumerate_precedes_paths_batch(
                batch["edge_src"],
                batch["edge_dst"],
                batch["edge_type"],
                batch["edge_mask"],
            )
            unique_slots = []
            for paths in paths_by_row:
                slots = sorted({e for p in paths for e in (p[0], p[1])})
                unique_slots.append(slots)
            edge_deltas = _edge_delta_map(model, batch, base.logits, unique_slots)

            max_paths = max((len(p) for p in paths_by_row), default=0)
            path_delta = [[0.0] * len(p) for p in paths_by_row]
            for t in range(max_paths):
                selected_ids = [[t] if t < len(p) else [] for p in paths_by_row]
                mask = union_path_edge_mask(batch["edge_mask"], paths_by_row, selected_ids)
                out = _forward(model, perturb_edges(batch, mask))
                delta = (base.logits - out.logits).detach()
                for i, paths in enumerate(paths_by_row):
                    if t < len(paths):
                        path_delta[i][t] = float(delta[i].item())

            top_ids: list[list[int]] = []
            rand_abs = torch.zeros_like(base.logits)
            rand_signed = torch.zeros_like(base.logits)
            rand_prob = torch.zeros_like(base.prob)
            rand_logit = torch.zeros_like(base.logits)
            for i, deltas in enumerate(path_delta):
                top_ids.append(select_top_positive_indices(deltas, fraction=args.top_fraction))
            top_mask = union_path_edge_mask(batch["edge_mask"], paths_by_row, top_ids)
            top_out = _forward(model, perturb_edges(batch, top_mask))

            for repeat in range(args.random_repeats):
                rand_ids: list[list[int]] = []
                for i, paths in enumerate(paths_by_row):
                    k = len(top_ids[i])
                    rand_ids.append(
                        select_random_indices(
                            len(paths),
                            k,
                            key=int(batch["stay_id"][i].item()),
                            seed=args.seed,
                            repeat=repeat,
                        )
                    )
                rmask = union_path_edge_mask(batch["edge_mask"], paths_by_row, rand_ids)
                rout = _forward(model, perturb_edges(batch, rmask))
                d = base.logits - rout.logits
                rand_abs += d.abs()
                rand_signed += d
                rand_prob += rout.prob
                rand_logit += rout.logits
            scale = float(args.random_repeats)
            rand_abs = rand_abs / scale
            rand_signed = rand_signed / scale
            rand_prob = rand_prob / scale
            rand_logit = rand_logit / scale

            bsz = batch["y"].shape[0]
            for i in range(bsz):
                stay = int(batch["stay_id"][i].item())
                subject = int(batch["subject_id"][i].item())
                y_i = float(batch["y"][i].item())
                n_paths = len(paths_by_row[i])
                n_pos = int(sum(d > 0.0 for d in path_delta[i]))
                patient_rows.append(
                    {
                        "stay_id": stay,
                        "subject_id": subject,
                        "y": y_i,
                        "n_paths": n_paths,
                        "n_positive_paths": n_pos,
                        "n_top_paths": len(top_ids[i]),
                        "base_logit": float(base.logits[i].item()),
                        "base_prob": float(base.prob[i].item()),
                        "top_logit": float(top_out.logits[i].item()),
                        "top_prob": float(top_out.prob[i].item()),
                        "random_logit": float(rand_logit[i].item()),
                        "random_prob": float(rand_prob[i].item()),
                        "top_signed_delta_logit": float((base.logits[i] - top_out.logits[i]).item()),
                        "random_signed_delta_logit": float(rand_signed[i].item()),
                        "top_abs_delta_logit": float((base.logits[i] - top_out.logits[i]).abs().item()),
                        "random_abs_delta_logit": float(rand_abs[i].item()),
                    }
                )
                ranked = sorted(range(n_paths), key=lambda j: path_delta[i][j], reverse=True)
                rank_of = {j: r + 1 for r, j in enumerate(ranked)}
                for j, (e1, e2, n1, n2, n3) in enumerate(paths_by_row[i]):
                    d1 = node_by_id[n1]
                    d2 = node_by_id[n2]
                    d3 = node_by_id[n3]
                    d_e1 = edge_deltas[i].get(e1, float("nan"))
                    d_e2 = edge_deltas[i].get(e2, float("nan"))
                    d_path = path_delta[i][j]
                    node_c = base.node_contrib[i] if base.node_contrib is not None else None
                    edge_c = base.edge_contrib[i] if base.edge_contrib is not None else None
                    path_rows.append(
                        {
                            "stay_id": stay,
                            "subject_id": subject,
                            "y": y_i,
                            "node_1": d1["activation_name"],
                            "node_2": d2["activation_name"],
                            "node_3": d3["activation_name"],
                            "node_id_1": n1,
                            "node_id_2": n2,
                            "node_id_3": n3,
                            "time_1": float(batch["node_time"][i, n1].item() * 48.0),
                            "time_2": float(batch["node_time"][i, n2].item() * 48.0),
                            "time_3": float(batch["node_time"][i, n3].item() * 48.0),
                            "relation_1": "PRECEDES",
                            "relation_2": "PRECEDES",
                            "edge_slot_1": e1,
                            "edge_slot_2": e2,
                            "edge_contrib_1": float(edge_c[e1].item()) if edge_c is not None else float("nan"),
                            "edge_contrib_2": float(edge_c[e2].item()) if edge_c is not None else float("nan"),
                            "node_contrib_1": float(node_c[n1].item()) if node_c is not None else float("nan"),
                            "node_contrib_2": float(node_c[n2].item()) if node_c is not None else float("nan"),
                            "node_contrib_3": float(node_c[n3].item()) if node_c is not None else float("nan"),
                            "delta_edge_1_logit": d_e1,
                            "delta_edge_2_logit": d_e2,
                            "delta_path_logit": d_path,
                            "interaction_logit": path_interaction(d_path, d_e1, d_e2),
                            "risk_direction": _risk_direction(d_path),
                            "path_rank": rank_of[j],
                            "is_top_positive": j in top_ids[i],
                            "pattern": fold_path_pattern(
                                d1["var"], d1["stage"], d2["var"], d2["stage"], d3["var"], d3["stage"]
                            ),
                        }
                    )

    patient_df = pd.DataFrame(patient_rows)
    path_df = pd.DataFrame(path_rows)
    patient_df.to_parquet(out_dir / "patient_path_fidelity_val.parquet", index=False)
    path_df.to_parquet(out_dir / "patient_path_contributions.parquet", index=False)

    if len(path_df):
        pattern_df = (
            path_df.groupby("pattern", observed=True)
            .agg(
                patient_support=("stay_id", "nunique"),
                death_support=("y", "sum"),
                n_paths=("stay_id", "size"),
                mean_delta_path=("delta_path_logit", "mean"),
                frac_delta_positive=("delta_path_logit", lambda s: float((s > 0).mean())),
                mean_interaction=("interaction_logit", "mean"),
                mean_abs_delta_path=("delta_path_logit", lambda s: float(s.abs().mean())),
            )
            .reset_index()
        )
        n_death_patients = int(patient_df.loc[patient_df["y"] == 1.0, "stay_id"].nunique())
        n_surv_patients = int(patient_df.loc[patient_df["y"] == 0.0, "stay_id"].nunique())
        death_rate = (
            path_df[path_df["y"] == 1.0]
            .groupby("pattern", observed=True)["stay_id"]
            .nunique()
            / max(n_death_patients, 1)
        ).rename("death_prevalence")
        surv_rate = (
            path_df[path_df["y"] == 0.0]
            .groupby("pattern", observed=True)["stay_id"]
            .nunique()
            / max(n_surv_patients, 1)
        ).rename("survival_prevalence")
        pattern_df = pattern_df.merge(death_rate, on="pattern", how="left")
        pattern_df = pattern_df.merge(surv_rate, on="pattern", how="left")
        pattern_df = pattern_df.sort_values(
            ["patient_support", "mean_delta_path"], ascending=False
        )
        pattern_df.to_csv(out_dir / "path_pattern_summary.csv", index=False)
    else:
        pattern_df = pd.DataFrame()

    y = patient_df["y"].to_numpy(dtype=np.float64)
    eligible = patient_df["n_top_paths"] > 0
    top_abs = patient_df.loc[eligible, "top_abs_delta_logit"].to_numpy()
    rand_abs = patient_df.loc[eligible, "random_abs_delta_logit"].to_numpy()
    top_signed = patient_df.loc[eligible, "top_signed_delta_logit"].to_numpy()
    rand_signed = patient_df.loc[eligible, "random_signed_delta_logit"].to_numpy()
    fidelity = {
        "n_eligible": int(eligible.sum()),
        "n_with_any_path": int((patient_df["n_paths"] > 0).sum()),
        "n_with_positive_path": int((patient_df["n_positive_paths"] > 0).sum()),
        "top_mean_abs_delta_logit": float(top_abs.mean()) if len(top_abs) else float("nan"),
        "random_mean_abs_delta_logit": float(rand_abs.mean()) if len(rand_abs) else float("nan"),
        "top_mean_signed_delta_logit": float(top_signed.mean()) if len(top_signed) else float("nan"),
        "random_mean_signed_delta_logit": float(rand_signed.mean()) if len(rand_signed) else float("nan"),
        "fraction_top_abs_gt_random": float(np.mean(top_abs > rand_abs)) if len(top_abs) else float("nan"),
        "mean_abs_excess_top_minus_random": _bootstrap_mean_ci(
            top_abs - rand_abs, n_boot=args.bootstrap_repeats, seed=args.seed + 3
        ),
        "mean_signed_excess_top_minus_random": _bootstrap_mean_ci(
            top_signed - rand_signed, n_boot=args.bootstrap_repeats, seed=args.seed + 4
        ),
        "auprc_random_minus_top": _bootstrap_metric_difference(
            y,
            patient_df["random_prob"].to_numpy(),
            patient_df["top_prob"].to_numpy(),
            metric="auprc",
            n_boot=args.bootstrap_repeats,
            seed=args.seed + 13,
        ),
        "auroc_random_minus_top": _bootstrap_metric_difference(
            y,
            patient_df["random_prob"].to_numpy(),
            patient_df["top_prob"].to_numpy(),
            metric="auroc",
            n_boot=args.bootstrap_repeats,
            seed=args.seed + 23,
        ),
        "top_mean_signed_delta_logit_death": float(
            patient_df.loc[eligible & (patient_df["y"] == 1.0), "top_signed_delta_logit"].mean()
        ),
        "top_mean_signed_delta_logit_survival": float(
            patient_df.loc[eligible & (patient_df["y"] == 0.0), "top_signed_delta_logit"].mean()
        ),
    }
    prediction_metrics = {
        "baseline": _metrics(y, patient_df["base_prob"].to_numpy()),
        "path_top_deleted": _metrics(y, patient_df["top_prob"].to_numpy()),
        "path_random_deleted_mean": _metrics(y, patient_df["random_prob"].to_numpy()),
    }
    global_ok = (
        _ci_excludes_zero(fidelity["mean_abs_excess_top_minus_random"])
        and fidelity["mean_abs_excess_top_minus_random"]["estimate"] > 0.0
        and _ci_excludes_zero(fidelity["auprc_random_minus_top"])
        and fidelity["auprc_random_minus_top"]["estimate"] > 0.0
    )
    claim = (
        "GNN识别的患者级演化路径具有全局删除忠实度。"
        if global_ok
        else "GNN生成了具有患者级反事实效应的候选演化路径，但群体层面忠实度尚未确立。"
    )
    summary = {
        "stage": "GNN_path_fidelity_validation",
        "split": "val",
        "test_read": False,
        "checkpoint": str(ckpt_path.resolve()),
        "graph_pack": str(pack_dir.resolve()),
        "branch_mode": dict(ckpt.get("meta") or {}).get("branch_mode"),
        "readout": dict(ckpt.get("meta") or {}).get("readout"),
        "n_patients": int(len(patient_df)),
        "n_paths": int(len(path_df)),
        "device": str(device),
        "top_fraction": float(args.top_fraction),
        "random_repeats": int(args.random_repeats),
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "deletion_contract": (
            "a 2-hop PRECEDES path is deleted by removing both edges; "
            "nodes and remaining edges stay; a_dense/sparse/static stay fixed; "
            "interaction = delta_path - delta_edge1 - delta_edge2"
        ),
        "prediction_metrics": prediction_metrics,
        "fidelity": fidelity,
        "global_path_faithfulness": global_ok,
        "claim": claim,
        "top_patterns": pattern_df.head(15).to_dict("records") if len(pattern_df) else [],
        "pack_meta": pack_meta,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "path_fidelity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(Path(args.report), summary, fidelity, prediction_metrics)
    return summary


def _write_report(
    path: Path,
    summary: dict[str, Any],
    f: dict[str, Any],
    m: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GNN路径级删除忠实度",
        "",
        "> 冻结可分解 G；仅 validation；不读 test。路径删除 = 去掉两条 PRECEDES 边，节点保留。",
        "> `Δlogit = logit完整图 − logit删除后`。`interaction = Δpath − Δedge1 − Δedge2`。",
        "",
        "## 1. 协议",
        "",
        f"- 患者数：{summary['n_patients']}；两跳 PRECEDES 路径数：{summary['n_paths']}。",
        f"- 每位患者在全部两跳路径中取 ceil({summary['top_fraction']} n) 条 **正向 Δpath 最高** 的路径。",
        f"- 随机对照：同数量随机两跳路径，重复 {summary['random_repeats']} 次。",
        f"- 可评估患者（至少一条正向 top 路径）：{f['n_eligible']}。",
        "",
        "## 2. 删除后整体预测",
        "",
        "| 条件 | AUPRC | AUROC | Brier |",
        "|---|---:|---:|---:|",
        f"| 完整图 | {m['baseline']['auprc']:.4f} | {m['baseline']['auroc']:.4f} | {m['baseline']['brier']:.4f} |",
        f"| 删除 top 风险路径 | {m['path_top_deleted']['auprc']:.4f} | {m['path_top_deleted']['auroc']:.4f} | {m['path_top_deleted']['brier']:.4f} |",
        f"| 删除随机路径 | {m['path_random_deleted_mean']['auprc']:.4f} | {m['path_random_deleted_mean']['auroc']:.4f} | {m['path_random_deleted_mean']['brier']:.4f} |",
        "",
        "## 3. 路径忠实度",
        "",
        "| 指标 | 估计（95% CI） |",
        "|---|---|",
        f"| top 平均有符号 Δlogit | {f['top_mean_signed_delta_logit']:.4f} |",
        f"| random 平均有符号 Δlogit | {f['random_mean_signed_delta_logit']:.4f} |",
        f"| top−random 有符号（配对） | {_fmt_ci(f['mean_signed_excess_top_minus_random'])} |",
        f"| top−random 绝对（配对） | {_fmt_ci(f['mean_abs_excess_top_minus_random'])} |",
        f"| 患者中 top|Δ| > random|Δ| | {f['fraction_top_abs_gt_random']:.4f} |",
        f"| AUPRC(random−top) | {_fmt_ci(f['auprc_random_minus_top'])} |",
        f"| AUROC(random−top) | {_fmt_ci(f['auroc_random_minus_top'])} |",
        "",
        f"死亡患者 top 有符号 Δlogit：{f['top_mean_signed_delta_logit_death']:.4f}；"
        f"存活：{f['top_mean_signed_delta_logit_survival']:.4f}。",
        "",
        "## 4. 判定",
        "",
        f"**{summary['claim']}**",
        "",
        "判定规则：患者级绝对 top−random CI 不含 0 且为正，并且 AUPRC(random−top) CI 不含 0 且为正。",
        "注意力或路径共现本身不是因果效应。",
        "",
        "## 5. 高频折叠路径（描述性）",
        "",
        "| 模式 | 患者支持 | 死亡支持 | 平均 Δpath | Δpath>0 比例 | 平均交互 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("top_patterns") or []:
        lines.append(
            f"| {row['pattern']} | {int(row['patient_support'])} | "
            f"{int(row['death_support'])} | {row['mean_delta_path']:.4f} | "
            f"{row['frac_delta_positive']:.3f} | {row['mean_interaction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "群体折叠只用于阅读。精确节点、时间和边贡献见 `patient_path_contributions.parquet`。",
            "",
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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-fraction", type=float, default=0.20)
    p.add_argument("--random-repeats", type=int, default=10)
    p.add_argument("--bootstrap-repeats", type=int, default=500)
    p.add_argument("--max-stays", type=int)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
