#!/usr/bin/env python3
"""84 motif-stage nodes vs death, *without* the graph model (docs reviewer concern).

Train-only ranking + 5-fold AUPRC of static vs static+node indicators.
Shows whether nodes carry outcome signal outside the weak G predictor.
Does not retune G or paths.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.path_protocol import PATH_SEED, refuse_test_path, refuse_test_split  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402

DEFAULT = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"
N_BOOT = 400


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-dir", default=str(DEFAULT))
    p.add_argument("--report", default=str(PROJ / "outputs/reports/27_节点结局关联_独立于G.md"))
    return p.parse_args()


def boot_diff(y, ind, rng, b=N_BOOT):
    n = len(y)
    stats = []
    for _ in range(int(b)):
        ix = rng.integers(0, n, n)
        yy, ii = y[ix], ind[ix].astype(bool)
        if ii.any() and (~ii).any():
            stats.append(float(yy[ii].mean() - yy[~ii].mean()))
    if not stats:
        return float("nan"), float("nan")
    a = np.array(stats)
    return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def cv_auprc(y, groups, static, extra, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, va in cv.split(static, y, groups):
        if y[tr].min() == y[tr].max() or y[va].min() == y[va].max():
            continue
        Xtr = np.column_stack([static[tr], extra[tr]]) if extra.shape[1] else static[tr]
        Xva = np.column_stack([static[va], extra[va]]) if extra.shape[1] else static[va]
        clf = LogisticRegression(max_iter=400, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        scores.append(average_precision_score(y[va], clf.predict_proba(Xva)[:, 1]))
    return float(np.mean(scores)) if scores else float("nan")


def main() -> None:
    args = parse_args()
    d = Path(args.paths_dir)
    refuse_test_path(d)
    refuse_test_path(args.report)
    atoms = pd.read_parquet(d / "motif_atoms.parquet")
    for sp in atoms["split"].astype(str).unique():
        refuse_test_split(sp)
    train = atoms[atoms["split"] == "train"]
    ev = pd.read_parquet(PROJ / "data/processed/shapelet/events/events_final_7var_motif84/stay_index.parquet")
    train_ids = sorted(int(x) for x in ev.loc[ev["split"] == "train", "ICUSTAY_ID"])
    stays = pd.read_parquet(Path(DATA_PROCESSED) / "stay_features_48h.parquet").set_index("ICUSTAY_ID")
    y = np.array([float(stays.loc[s, "HOSPITAL_MORTALITY"]) for s in train_ids])
    groups = np.array([int(stays.loc[s, "SUBJECT_ID"]) for s in train_ids])
    static = np.stack(
        [np.array([float(stays.loc[s, c]) for c in SHAPELET_STATIC_COLS], dtype=np.float64) for s in train_ids]
    )
    mu, sd = static.mean(0), np.maximum(static.std(0), 1e-6)
    static = (static - mu) / sd
    nodes = sorted(train["atom"].astype(str).unique())
    pos = {s: i for i, s in enumerate(train_ids)}
    node_index = {n: j for j, n in enumerate(nodes)}
    ind = np.zeros((len(train_ids), len(nodes)), dtype=np.float64)
    for rec in train.itertuples(index=False):
        i = pos.get(int(rec.ICUSTAY_ID))
        if i is None:
            continue
        j = node_index.get(str(rec.atom))
        if j is not None:
            ind[i, j] = 1.0
    rng = np.random.default_rng(PATH_SEED)
    rows = []
    for j, name in enumerate(nodes):
        ii = ind[:, j] > 0.5
        n1 = int(ii.sum())
        if n1 < 20 or n1 > len(train_ids) - 20:
            continue
        r1 = float(y[ii].mean())
        r0 = float(y[~ii].mean())
        lo, hi = boot_diff(y, ind[:, j], rng)
        rows.append(
            {
                "node": name,
                "n": n1,
                "death_rate_on": r1,
                "death_rate_off": r0,
                "death_rate_diff": r1 - r0,
                "diff_lo": lo,
                "diff_hi": hi,
                "ci_excludes_0": bool(np.isfinite(lo) and lo * hi > 0),
            }
        )
    tab = pd.DataFrame(rows).sort_values("death_rate_diff", key=lambda s: s.abs(), ascending=False)
    tab.to_csv(d / "node_outcome_assoc_train.csv", index=False, encoding="utf-8-sig")
    n_sig = int(tab["ci_excludes_0"].sum()) if len(tab) else 0
    _log(f"[32] nodes_tested={len(tab)} ci_excludes_0={n_sig}")
    empty = np.zeros((len(train_ids), 0))
    a_static = cv_auprc(y, groups, static, empty, PATH_SEED)
    a_nodes = cv_auprc(y, groups, static, ind, PATH_SEED)
    # high-support associated nodes only
    keep = tab.loc[tab["ci_excludes_0"], "node"].tolist()
    if keep:
        cols = [node_index[n] for n in keep]
        a_sig = cv_auprc(y, groups, static, ind[:, cols], PATH_SEED)
    else:
        a_sig = float("nan")
    payload = {
        "stage": "node_outcome_association_train",
        "uses_graph_model": False,
        "n_node_types": len(nodes),
        "n_tested_support20": int(len(tab)),
        "n_diff_ci_excludes_0": n_sig,
        "train_cv_auprc_static": a_static,
        "train_cv_auprc_static_plus_84": a_nodes,
        "train_cv_auprc_static_plus_sig_nodes": a_sig,
        "note": "AUPRC here is a bag-of-nodes logistic, not G. If this is near G (0.31) or only slightly above static, G-node fidelity cannot be read as strong outcome importance.",
        "seed": PATH_SEED,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (d / "node_outcome_assoc.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    top = tab.head(15)
    lines = [
        "# 84 节点与死亡的关联（不经过 G）",
        "",
        "> 回应「G 的 AUPRC 只有 0.313，删节点下降不能直接写成群体风险由节点承载」。",
        "",
        f"- 训练五折 AUPRC：仅静态 **{a_static:.4f}**；静态+84 节点指示 **{a_nodes:.4f}**；静态+差的 CI 不含 0 的节点 **{a_sig:.4f}**。",
        f"- 支持≥20 的节点类型 {len(tab)} 个，其中死亡率差 bootstrap 95% CI 不含 0 的 **{n_sig}** 个。",
        "- 可分解 G 的节点删除忠实度只说明这些节点对 **G 的 logit** 重要。本节是同一词典在 **结局 logistic** 上的独立关联。",
        "- 若静态+节点只略高于静态，则「节点承载群体风险」必须降为「节点忠实于弱图预测器，部分节点与死亡有独立关联」。",
        "",
        "## |死亡率差| 最大的 15 个节点（train）",
        "",
        "| 节点 | n | 出现时死亡率 | 差 [boot] | CI 跨 0 |",
        "|---|---:|---:|---|---|",
    ]
    for r in top.itertuples(index=False):
        cross = "否" if r.ci_excludes_0 else "是"
        lines.append(
            f"| {r.node} | {r.n} | {r.death_rate_on:.3f} | "
            f"{r.death_rate_diff:.3f} [{r.diff_lo:.3f},{r.diff_hi:.3f}] | {cross} |"
        )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(json.dumps({k: payload[k] for k in payload if k != "note"}, indent=2))
    _log("[32] DONE")


if __name__ == "__main__":
    main()
