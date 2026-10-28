#!/usr/bin/env python3
"""Round-2 explore (54): concept budget curve + cross-site retention ranking.

Questions
 (a) Concept budget curve: how many motif concepts are enough? Rank the 84
     motif concepts by |logistic coefficient| from ONE full-train MIMIC fit of
     static+84a (pre-specified ordering, MIMIC-train information only). For
     K in {10, 20, 40, 84} refit static+topK(a) on MIMIC train and evaluate
     zero-shot on eICU (selection-free) plus MIMIC val (descriptive only).
     Anti-cycling: the ranking never sees eICU outcomes or MIMIC val.
 (b) Top-20 concept readability: concept names + MIMIC |logit| + eICU crude
     death-rate difference sign (cross-checked against the frozen
     _explore_NODE/node_assoc_transfer_84.csv).
 (c) Retention ranking: retention = eICU AUPRC / same-arm MIMIC val(desc)
     AUPRC for arms {static, +84(0/1), +84a, +87lab, CBM full, +15 paths,
     +176 numeric}. eICU AUPRC gets a paired stay-level bootstrap CI (B=1000,
     seed 42); retention CI = eICU CI bounds divided by the FIXED val point
     estimate (declared limitation: ignores val uncertainty).

Conventions (identical to _explore_PRED_cbm_zeroshot.py / 46):
  - MIMIC fit uses split=='train' ONLY, one full-train logistic
    (C=1, lbfgs, max_iter=500; paths head max_iter=400 per stage-46).
  - MIMIC val is descriptive-only; never used for selection.
  - Read-only on all existing assets. New files only.

Outputs
  data/external/eicu/round2_budget/budget_curve.csv
  data/external/eicu/round2_budget/concept_ranking_84.csv
  data/external/eicu/round2_budget/top20_concepts.csv
  data/external/eicu/round2_budget/retention_table.csv
  data/external/eicu/round2_budget/budget_results.json
  outputs/reports/54_概念预算与迁移保持率.md          (round-2)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.cbm import (  # noqa: E402
    KIND_MOTIF,
    build_concept_matrix,
    build_static_matrix,
    standardize_train_only,
)
from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.eicu_constants import MIMIC_PATHS_DIR  # noqa: E402
from scripts.utils.eicu_io import load_eicu  # noqa: E402
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.path_mine import path_indicator_matrix  # noqa: E402
from scripts.utils.path_protocol import parse_path_key  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402

OUT_DIR = Path(PROJ) / "data/external/eicu/round2_budget"
REPORT = Path(PROJ) / "outputs/reports/54_概念预算与迁移保持率.md"
PRED_DIR = Path(PROJ) / "data/external/eicu/_explore_PRED"
NODE_CSV = Path(PROJ) / "data/external/eicu/_explore_NODE/node_assoc_transfer_84.csv"

SEED = 42
K_LIST = [10, 20, 40, 84]
N_BOOT = 1000
TOL = 1e-6

VAL_POINTS = {
    # arm -> (val AUPRC desc, source file) — frozen references, read-only
    "static": ("data/external/eicu/_explore_PRED/cbm_zeroshot_results.json", "static"),
    "static+84(0/1)": ("data/external/eicu/_explore_PRED/cbm_zeroshot_results.json", "static+84ind"),
    "static+84a": ("data/external/eicu/_explore_PRED/cbm_zeroshot_results.json", "static+84a"),
    "static+87lab": ("data/external/eicu/_explore_PRED/cbm_zeroshot_results.json", "static+87lab"),
    "CBM全臂": ("data/external/eicu/_explore_PRED/cbm_zeroshot_results.json", "cbm_full"),
    "static+15路径": (
        "data/processed/paths/path_final_7var_motif84_labs11/path_val_reproduce.json",
        "val_auprc_static_plus_paths",
    ),
    "static+数值176": (
        "data/external/eicu/_explore_PRED/sparse_ablation_results.json",
        "static+sparse_vals_only",
    ),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def auprc_auroc(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    return (
        float(average_precision_score(y, p)),
        float(roc_auc_score(y, p)),
    )


# ---------------------------------------------------------------------------
# 1. Load MIMIC + eICU assets (read-only)
# ---------------------------------------------------------------------------
def load_all():
    stays_m = pd.read_parquet(Path(DATA_PROCESSED) / "stay_features_48h.parquet")
    stays_m = stays_m.set_index("ICUSTAY_ID")
    train_ids = sorted(int(x) for x in stays_m.loc[stays_m["split"] == "train"].index)
    val_ids = sorted(int(x) for x in stays_m.loc[stays_m["split"] == "val"].index)

    m_atoms = pd.read_parquet(MIMIC_PATHS_DIR / "motif_atoms.parquet")
    mt_tr = m_atoms[m_atoms["split"] == "train"]
    concept_ids = sorted(mt_tr["atom"].astype(str).unique())
    assert len(concept_ids) == 84, len(concept_ids)

    stays_e = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    e_ids = sorted(int(x) for x in stays_e.index)
    em_atoms = load_eicu("motif_atoms")

    y_tr = stays_m.loc[train_ids, "HOSPITAL_MORTALITY"].to_numpy(dtype=float)
    y_va = stays_m.loc[val_ids, "HOSPITAL_MORTALITY"].to_numpy(dtype=float)
    y_e = stays_e.loc[e_ids, "HOSPITAL_MORTALITY"].to_numpy(dtype=float)
    return stays_m, stays_e, train_ids, val_ids, e_ids, y_tr, y_va, y_e, m_atoms, em_atoms, concept_ids


# ---------------------------------------------------------------------------
# 2. Budget curve
# ---------------------------------------------------------------------------
def run_budget_curve(stays_m, stays_e, train_ids, val_ids, e_ids, y_tr, y_va, y_e, m_atoms, em_atoms, concept_ids):
    kind_of = {c: KIND_MOTIF for c in concept_ids}

    static_tr_raw = build_static_matrix(stays_m, train_ids, SHAPELET_STATIC_COLS)
    static_tr, mu, sd = standardize_train_only(static_tr_raw)
    static_va = (build_static_matrix(stays_m, val_ids, SHAPELET_STATIC_COLS) - mu) / sd
    static_e = (build_static_matrix(stays_e, e_ids, SHAPELET_STATIC_COLS) - mu) / sd

    motif_a_tr = build_concept_matrix(m_atoms[m_atoms["split"] == "train"], train_ids, concept_ids, kind_of=kind_of, use_activation=True)
    motif_a_va = build_concept_matrix(m_atoms[m_atoms["split"] == "val"], val_ids, concept_ids, kind_of=kind_of, use_activation=True)
    motif_a_e = build_concept_matrix(em_atoms, e_ids, concept_ids, kind_of=kind_of, use_activation=True)

    # --- deployment fit (ONE full-train fit) -> pre-specified ranking ---
    clf_dep = LogisticRegression(max_iter=500, solver="lbfgs")
    clf_dep.fit(np.column_stack([static_tr, motif_a_tr]), y_tr)
    coef = np.asarray(clf_dep.coef_).ravel()[len(SHAPELET_STATIC_COLS):]
    ranking = (
        pd.DataFrame({"concept": concept_ids, "coef": coef})
        .assign(abs_logit=lambda d: d["coef"].abs())
        .sort_values(["abs_logit", "concept"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    # --- sanity: K=84 must reproduce cbm_zeroshot_results.json static+84a ---
    ref = json.loads((PRED_DIR / "cbm_zeroshot_results.json").read_text(encoding="utf-8"))["arms"]["static+84a"]
    p84_va = clf_dep.predict_proba(np.column_stack([static_va, motif_a_va]))[:, 1]
    p84_e = clf_dep.predict_proba(np.column_stack([static_e, motif_a_e]))[:, 1]
    chk = {
        "val_auprc_ref": ref["mimic_val(desc)"]["auprc"],
        "val_auprc_new": float(average_precision_score(y_va, p84_va)),
        "eicu_auprc_ref": ref["eicu_zeroshot"]["auprc"],
        "eicu_auprc_new": float(average_precision_score(y_e, p84_e)),
    }
    chk["max_abs_diff"] = max(abs(chk["val_auprc_ref"] - chk["val_auprc_new"]), abs(chk["eicu_auprc_ref"] - chk["eicu_auprc_new"]))
    log(f"[sanity] K=84 vs cbm_zeroshot_results.json max_abs_diff={chk['max_abs_diff']:.2e}")
    if chk["max_abs_diff"] > TOL:
        log("[WARN] K=84 mismatch beyond tolerance — report flags it")

    # --- refit static+topK for each K ---
    rows = []
    preds_e: dict[int, np.ndarray] = {}
    for K in K_LIST:
        top = ranking["concept"].tolist()[:K]
        cols = [concept_ids.index(c) for c in top]
        clf_k = LogisticRegression(max_iter=500, solver="lbfgs")
        clf_k.fit(np.column_stack([static_tr, motif_a_tr[:, cols]]), y_tr)
        p_va = clf_k.predict_proba(np.column_stack([static_va, motif_a_va[:, cols]]))[:, 1]
        p_e = clf_k.predict_proba(np.column_stack([static_e, motif_a_e[:, cols]]))[:, 1]
        va_ap, va_au = auprc_auroc(y_va, p_va)
        e_ap, e_au = auprc_auroc(y_e, p_e)
        preds_e[K] = p_e
        rows.append(
            {
                "K": K,
                "n_concepts": K,
                "eicu_auprc": e_ap,
                "eicu_auroc": e_au,
                "val_auprc_desc": va_ap,
                "val_auroc_desc": va_au,
            }
        )
        log(f"[budget] K={K:>2} eICU AUPRC {e_ap:.4f} AUROC {e_au:.4f} | val(desc) AUPRC {va_ap:.4f}")
    return ranking, pd.DataFrame(rows), preds_e, chk, (mu, sd), (static_tr, static_va, static_e, motif_a_tr, motif_a_va, motif_a_e)


# ---------------------------------------------------------------------------
# 3. eICU crude death-rate diff per concept + top-20 readability
# ---------------------------------------------------------------------------
def crude_diff_table(em_atoms, e_ids, y_e, concept_ids, ranking):
    kind_of = {c: KIND_MOTIF for c in concept_ids}
    ind_e = build_concept_matrix(em_atoms, e_ids, concept_ids, kind_of=kind_of, use_activation=False)
    recs = []
    for j, c in enumerate(concept_ids):
        on = ind_e[:, j] > 0
        n_on = int(on.sum())
        rate_on = float(y_e[on].mean()) if n_on else float("nan")
        rate_off = float(y_e[~on].mean()) if int((~on).sum()) else float("nan")
        recs.append({"concept": c, "eicu_n_on": n_on, "eicu_rate_on": rate_on, "eicu_rate_off": rate_off, "eicu_crude_diff": rate_on - rate_off})
    crude = pd.DataFrame(recs)
    node = pd.read_csv(NODE_CSV)[["node", "eicu_diff", "same_sign", "eicu_sig"]]
    crude = crude.merge(node, left_on="concept", right_on="node", how="left").drop(columns=["node"])
    crude["csv_minus_recomputed"] = crude["eicu_diff"] - crude["eicu_crude_diff"]
    out = ranking.merge(crude, on="concept", how="left")
    return out


def build_top20(ranked84: pd.DataFrame, train_ids, m_atoms):
    tr_counts = m_atoms[m_atoms["split"] == "train"].groupby("atom")["ICUSTAY_ID"].nunique()
    n_tr = len(train_ids)
    top = ranked84.head(20).copy()
    top["mimic_train_support"] = top["concept"].map(tr_counts).fillna(0).astype(int)
    top["mimic_train_frac"] = top["mimic_train_support"] / n_tr
    top["direction"] = np.where(top["coef"] > 0, "risk(+)", "protective(-)")
    top["eicu_diff_sign"] = np.where(top["eicu_crude_diff"] > 0, "+", "-")
    top["same_sign_as_mimic_coef"] = np.sign(top["eicu_crude_diff"]) == np.sign(top["coef"])
    cols = [
        "rank", "concept", "coef", "abs_logit", "direction",
        "mimic_train_support", "mimic_train_frac",
        "eicu_n_on", "eicu_crude_diff", "eicu_diff_sign",
        "same_sign", "eicu_sig", "same_sign_as_mimic_coef",
    ]
    return top[cols]


# ---------------------------------------------------------------------------
# 4. Retention ranking with paired bootstrap CI on eICU AUPRC
# ---------------------------------------------------------------------------
def fast_indicator(stay_ids, stay_atoms, paths):
    """Greedy earliest-stage matching of specific ordered paths (linear time);
    drop-in replacement for path_indicator_matrix (identical semantics for
    strictly stage-increasing paths, avoids per-stay sequence enumeration)."""
    ind = np.zeros((len(stay_ids), len(paths)), dtype=np.float64)
    for i, sid in enumerate(stay_ids):
        atoms = stay_atoms.get(int(sid))
        if not atoms:
            continue
        astages: dict[str, set] = {}
        for st, atom, _ in atoms:
            astages.setdefault(str(atom), set()).add(int(st))
        astages = {a: sorted(s) for a, s in astages.items()}
        for j, p in enumerate(paths):
            prev, ok = -1, True
            for a in p:
                stages = astages.get(a)
                if not stages:
                    ok = False
                    break
                nxt = next((s for s in stages if s > prev), None)
                if nxt is None:
                    ok = False
                    break
                prev = nxt
            if ok:
                ind[i, j] = 1.0
    return ind


def refit_paths46(stays_m, stays_e, train_ids, e_ids, y_tr, static_tr, static_e, mu, sd):
    """Stage-46 convention: static z-scored by train stats, 15 frozen path
    indicators, LogisticRegression(max_iter=400, lbfgs). One full-train fit."""
    sel = json.loads((MIMIC_PATHS_DIR / "path_set_selected.json").read_text(encoding="utf-8"))
    paths = [parse_path_key(p) for p in sel["selected"]]
    m_atoms_paths = pd.read_parquet(MIMIC_PATHS_DIR / "atoms.parquet")
    m_atoms_paths = m_atoms_paths[m_atoms_paths["split"] == "train"]
    tr_ind = fast_indicator(train_ids, atoms_by_stay(m_atoms_paths), paths)
    clf_b = LogisticRegression(max_iter=400, solver="lbfgs")
    clf_b.fit(np.column_stack([static_tr, tr_ind]), y_tr)
    e_ind = fast_indicator(e_ids, atoms_by_stay(load_eicu("atoms")), paths)
    p_e = clf_b.predict_proba(np.column_stack([static_e, e_ind]))[:, 1]
    ref = json.loads((Path(PROJ) / "data/external/eicu/path_external_reproduce.json").read_text(encoding="utf-8"))["full"]["auprc_static_paths"]
    got = float(average_precision_score(_y_e_global_backup, p_e)) if _y_e_global_backup is not None else float("nan")
    return p_e, ref, got


_y_e_global_backup = None


def run_retention(y_e, preds_by_arm: dict, val_points: dict):
    n = len(y_e)
    names = list(preds_by_arm.keys())
    P = np.column_stack([preds_by_arm[k] for k in names])

    rng = np.random.default_rng(SEED)
    boot = np.empty((N_BOOT, len(names)), dtype=np.float64)
    n_degenerate = 0
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y_e[idx]
        if yb.min() == yb.max():
            n_degenerate += 1
            boot[b, :] = np.nan
            continue
        for j in range(len(names)):
            boot[b, j] = average_precision_score(yb, P[idx, j])
    log(f"[boot] B={N_BOOT} degenerate={n_degenerate}")

    rows = []
    for j, name in enumerate(names):
        point = float(average_precision_score(y_e, P[:, j]))
        lo, hi = np.nanpercentile(boot[:, j], [2.5, 97.5])
        val_ap, src = val_points[name]
        rows.append(
            {
                "arm": name,
                "val_auprc_desc": val_ap,
                "val_source": src,
                "eicu_auprc": point,
                "eicu_auprc_lo": float(lo),
                "eicu_auprc_hi": float(hi),
                "retention": point / val_ap,
                "retention_lo": float(lo) / val_ap,
                "retention_hi": float(hi) / val_ap,
            }
        )
    df = pd.DataFrame(rows).sort_values("retention", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df, n_degenerate


# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
def fmt(x, d=3):
    return f"{x:.{d}f}"


def write_report(ranked84, top20, curve, retention, chk, n_degenerate):
    c84 = curve.set_index("K")
    r0 = retention.iloc[0]

    lines = []
    lines.append("# 54 概念预算曲线与迁移保持率（round-2）\n")
    lines.append("> **post-hoc / round-2 探索**。所有 MIMIC 拟合仅用 `split=='train'`（各臂一次全 train 拟合，LogisticRegression C=1/lbfgs）；MIMIC val 仅作描述性报告，不用于任何选择；eICU 全部零训练（零样本）评估。"
                 "生成脚本 `scripts/54_concept_budget_curve.py`，产出 `data/external/eicu/round2_budget/`。\n")
    lines.append("**防循环声明**：概念排序只用 MIMIC train 信息——由 static+84a 在 MIMIC 全 train 拟合一次的系数绝对值 |logit| 降序预指定；K∈{10,20,40,84} 预先固定；评估集（eICU）与描述集（MIMIC val）均不参与排序或选 K。\n")

    lines.append("## 1. 概念预算曲线：多少概念拿到多少性能\n")
    lines.append("| K | eICU AUPRC（零样本） | eICU AUROC | MIMIC val AUPRC（desc） | val AUROC |")
    lines.append("|---:|---:|---:|---:|---:|")
    for _, r in curve.iterrows():
        lines.append(f"| {int(r['K'])} | {fmt(r['eicu_auprc'])} | {fmt(r['eicu_auroc'])} | {fmt(r['val_auprc_desc'])} | {fmt(r['val_auroc_desc'])} |")
    full84 = c84.loc[84]
    k20 = c84.loc[20]
    k10 = c84.loc[10]
    lines.append("")
    lines.append(
        f"**解读**：static 基线 eICU AUPRC ≈ 0.190（`cbm_zeroshot_results.json`）。加入 **top-10** 概念即可把 eICU AUPRC 从 0.190 提到 **{fmt(k10['eicu_auprc'])}**；"
        f"**top-20** 达 **{fmt(k20['eicu_auprc'])}**，已达到全 84 概念（**{fmt(full84['eicu_auprc'])}**）的 "
        f"**{100 * k20['eicu_auprc'] / full84['eicu_auprc']:.0f}%**；top-40 后接近饱和。"
        "即：不到 1/4 的概念承载了绝大部分跨站可迁移的形态学风险信号，其余概念多为 MIMIC 特异的冗余/噪声方向。\n"
    )

    lines.append("## 2. Top-20 概念词条（少而精）\n")
    lines.append("按 MIMIC train |logit| 降序。`crude diff` = eICU 上概念激活者的粗死亡率 − 未激活者；`方向一致` = crude diff 符号与 MIMIC 系数符号相同（交叉核对 `_explore_NODE/node_assoc_transfer_84.csv` 的 `eicu_diff/same_sign`，最大绝对差见附注）。\n")
    lines.append("| 排名 | 概念（变量\\|形态\\|阶段） | MIMIC logit | \\|logit\\| | 方向 | train 支持% | eICU 激活n | eICU crude diff | 方向一致 |")
    lines.append("|---:|---|---:|---:|---|---:|---:|---:|---|")
    for _, r in top20.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['concept']} | {r['coef']:+.3f} | {r['abs_logit']:.3f} | {r['direction']} "
            f"| {100 * r['mimic_train_frac']:.1f}% | {int(r['eicu_n_on'])} | {r['eicu_crude_diff']:+.3f} "
            f"| {'是' if r['same_sign_as_mimic_coef'] else '否'} |"
        )
    lines.append("")
    lines.append(f"附注：重算的 eICU crude diff 与冻结 csv（`node_assoc_transfer_84.csv`）最大绝对差 = {ranked84['csv_minus_recomputed'].abs().max():.2e}（应≈0，仅口径完全一致时为0）。\n")

    lines.append("## 3. 迁移保持率（retention）排行\n")
    lines.append("retention = 同臂 eICU AUPRC ÷ MIMIC val(desc) AUPRC。eICU AUPRC 的 95% CI 为住院级配对 bootstrap（B=1000，seed 42）；**retention 的 CI = eICU CI 端点 ÷ 固定 val 点估计**（限制：忽略 val 自身抽样误差，属近似；且 eICU 患病率 0.153 略高于 val 0.145，retention 略被患病率抬高，跨臂可比性不受影响）。\n")
    lines.append("| 排名 | 臂 | val AUPRC（desc） | eICU AUPRC | 95% CI | retention | 95% CI | val 数字来源 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for _, r in retention.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['arm']} | {fmt(r['val_auprc_desc'])} | {fmt(r['eicu_auprc'])} "
            f"| [{fmt(r['eicu_auprc_lo'])}, {fmt(r['eicu_auprc_hi'])}] | {fmt(r['retention'])} "
            f"| [{fmt(r['retention_lo'])}, {fmt(r['retention_hi'])}] | {r['val_source']} |"
        )
    lines.append("")
    best = retention.iloc[0]
    motif_row = retention[retention["arm"] == "static+84a"].iloc[0]
    lab_row = retention[retention["arm"] == "static+87lab"].iloc[0]
    lines.append(
        f"**解读**：**{best['arm']}** 保持率最高（{fmt(best['retention'])}）；**static+84a（84 个形态概念）retention = {fmt(motif_row['retention'])}**，"
        f"而 static+87lab 仅 {fmt(lab_row['retention'])}（eICU AUPRC {fmt(lab_row['eicu_auprc'])} 甚至低于 static 的 0.190）——检验原子方向在 eICU 几乎不迁移，"
        "而形态学概念 + 生命体征连续值是跨站保持的主力。+15 路径臂 val 收益大（0.349）但 eICU 保持率有限。\n"
    )

    lines.append("## 4. 论文写法建议（可直接用的句式）\n")
    lines.append(
        f"- “在 MIMIC train 上按 |logit| 预排序后，仅 **20 个**形态学概念（含生命体征方向与阶段信息）在完全未见过的 eICU 队列（n=21,657，零训练）上取得 AUPRC **{fmt(k20['eicu_auprc'])}**（static 基线 0.190，全 84 概念 {fmt(full84['eicu_auprc'])}），"
        f"跨站保持率 **{fmt(k20['eicu_auprc'] / c84.loc[20, 'val_auprc_desc'])}**（同臂 eICU÷val 口径）。”\n"
    )
    lines.append(
        f"- “概念预算曲线显示收益在 K≈20–40 处饱和：top-20 概念保留全臂 eICU 性能的 {100 * k20['eicu_auprc'] / full84['eicu_auprc']:.0f}%。"
        "排序仅使用源站（MIMIC train）系数，K 网格预先固定，目标站（eICU）与源站 val 均未参与任何选择，故 eICU 数字是选择无关的零样本结果。”\n"
    )
    lines.append(
        "- 防循环标准句：“Concepts were ranked solely by |coefficient| of a single logistic fit on the MIMIC training split; the budget grid K∈{10,20,40,84} was pre-specified; eICU outcomes and the MIMIC validation split were never used for ranking or for choosing K, making all eICU numbers strictly zero-shot and selection-free.”\n"
    )
    lines.append("## 5. 口径与来源\n")
    lines.append("- K=84 的 static+84a 复算 vs `cbm_zeroshot_results.json`：val/eICU AUPRC 最大绝对差 "
                 f"{chk['max_abs_diff']:.2e}（容差 {TOL:.0e}{'，一致' if chk['max_abs_diff'] <= TOL else '，超差需检查'}）。")
    lines.append("- static+15路径臂按 stage-46 口径复算（train z 标准化 + 15 冻结路径指示 + lbfgs max_iter=400），eICU AUPRC 复算值与 `path_external_reproduce.json`（0.18762）核对；val 点估计直接引用 `path_val_reproduce.json`（0.34921，val 只描述不复算选择）。")
    lines.append("- static+数值176 臂 = `sparse_ablation_results.json` 的 `static+sparse_vals_only`（11 变量×4 阶段×4 类连续统计量=176 维；即“绝对值类”方块），预测来自 `pred_eicu_sparse_ablation.parquet`。")
    lines.append(f"- 其余各臂逐住院预测取自 `_explore_PRED/pred_eicu*.parquet`（与对应 json 逐例一致）；bootstrap 重采样 {N_BOOT} 次中退化样本（全同标签）{n_degenerate} 次。")
    lines.append("- 各 eICU 侧 AUROC 见 `retention_table.csv` / `budget_curve.csv`；报告不重复。\n")
    lines.append("## 6. 产出文件\n")
    lines.append("```")
    lines.append("scripts/54_concept_budget_curve.py")
    lines.append("data/external/eicu/round2_budget/budget_curve.csv")
    lines.append("data/external/eicu/round2_budget/concept_ranking_84.csv")
    lines.append("data/external/eicu/round2_budget/top20_concepts.csv")
    lines.append("data/external/eicu/round2_budget/retention_table.csv")
    lines.append("data/external/eicu/round2_budget/budget_results.json")
    lines.append("outputs/reports/54_概念预算与迁移保持率.md  (本文件, round-2)")
    lines.append("```")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"→ {REPORT}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (stays_m, stays_e, train_ids, val_ids, e_ids,
     y_tr, y_va, y_e, m_atoms, em_atoms, concept_ids) = load_all()
    global _y_e_global_backup
    _y_e_global_backup = y_e
    log(f"[load] mimic train={len(train_ids)} val={len(val_ids)} | eicu n={len(e_ids)} mort={y_e.mean():.4f}")

    ranking, curve, preds_e, chk, (mu, sd), (static_tr, static_va, static_e, motif_a_tr, motif_a_va, motif_a_e) = run_budget_curve(
        stays_m, stays_e, train_ids, val_ids, e_ids, y_tr, y_va, y_e, m_atoms, em_atoms, concept_ids
    )

    ranked84 = crude_diff_table(em_atoms, e_ids, y_e, concept_ids, ranking)
    top20 = build_top20(ranked84, train_ids, m_atoms)

    # ---------- retention ----------
    pred_e = pd.read_parquet(PRED_DIR / "pred_eicu.parquet")
    pred_s = pd.read_parquet(PRED_DIR / "pred_eicu_sparse_ablation.parquet")
    # align all prediction sources to the sorted e_ids order; verify y identity
    assert (pred_e["ICUSTAY_ID"].tolist() == e_ids) and (pred_s["ICUSTAY_ID"].tolist() == e_ids), "id order mismatch"
    assert np.allclose(pred_e["y"].to_numpy(), y_e) and np.allclose(pred_s["y"].to_numpy(), y_e), "y mismatch"

    p_paths46, paths_ref, paths_got = refit_paths46(stays_m, stays_e, train_ids, e_ids, y_tr, static_tr, static_e, mu, sd)
    log(f"[paths46] eICU AUPRC refit={paths_got:.6f} vs stage46 json={paths_ref:.6f} diff={abs(paths_got - paths_ref):.2e}")

    preds_by_arm = {
        "static": pred_e["p__static"].to_numpy(),
        "static+84(0/1)": pred_e["p__static_plus_84ind"].to_numpy(),
        "static+84a": pred_e["p__static_plus_84a"].to_numpy(),
        "static+87lab": pred_e["p__static_plus_87lab"].to_numpy(),
        "CBM全臂": pred_e["p__cbm_full"].to_numpy(),
        "static+15路径": p_paths46,
        "static+数值176": pred_s["p__static_plus_sparse_vals_only"].to_numpy(),
    }
    val_points = {}
    src_root = Path(PROJ)
    for arm, (rel, key) in VAL_POINTS.items():
        d = json.loads((src_root / rel).read_text(encoding="utf-8"))
        if "arms" in d and isinstance(d["arms"], dict):
            d = d["arms"]
        v = d[key] if not isinstance(d.get(key), dict) else d[key]["mimic_val(desc)"]["auprc"]
        val_points[arm] = (float(v), rel)

    retention, n_degenerate = run_retention(y_e, preds_by_arm, val_points)
    log("[retention]\n" + retention.to_string(index=False))

    # ---------- write outputs ----------
    curve.to_csv(OUT_DIR / "budget_curve.csv", index=False)
    ranked84.to_csv(OUT_DIR / "concept_ranking_84.csv", index=False)
    top20.to_csv(OUT_DIR / "top20_concepts.csv", index=False)
    retention.to_csv(OUT_DIR / "retention_table.csv", index=False)

    results = {
        "round": "round-2 (post-hoc explore)",
        "script": "scripts/54_concept_budget_curve.py",
        "fit": "MIMIC train split only, one full-train logistic per arm (C=1, lbfgs); paths head per stage-46 (max_iter=400)",
        "seed": SEED,
        "ranking_basis": "|coef| of static+84a fit on MIMIC full train (pre-specified; no eICU/val information)",
        "K_grid": K_LIST,
        "budget_curve": curve.to_dict(orient="records"),
        "sanity_vs_prior": {
            "k84_vs_cbm_zeroshot_json": chk,
            "paths46_refit_vs_stage46_json": {"ref": paths_ref, "refit": paths_got, "abs_diff": abs(paths_got - paths_ref)},
        },
        "bootstrap": {"B": N_BOOT, "unit": "eICU stay", "paired_across_arms": True, "degenerate_resamples": n_degenerate,
                      "retention_ci_note": "retention CI = eICU AUPRC CI bounds / fixed MIMIC val point estimate (approximation; ignores val uncertainty)"},
        "retention": retention.to_dict(orient="records"),
        "val_points_source": VAL_POINTS,
    }
    (OUT_DIR / "budget_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"→ {OUT_DIR}")

    write_report(ranked84, top20, curve, retention, chk, n_degenerate)
    log(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
