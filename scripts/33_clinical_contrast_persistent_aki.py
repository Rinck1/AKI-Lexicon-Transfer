#!/usr/bin/env python3
"""Clinical contrast: 48h persistent/resolving AKI vs frozen 15 paths.

Does not reselect paths. Train fits logistic; val and holdout are scored once
with train-fit coefficients. Persistent AKI is defined from creatinine/urine
only (Bhatraju-style resolving + end-window oliguria).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED, OUTPUT_REPORTS, OUTPUT_TABLES
from scripts.utils.path_atoms import atoms_by_stay
from scripts.utils.path_protocol import PATH_SEED, parse_path_key, refuse_test_split
from scripts.utils.persistent_aki import build_persistent_table
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS

D = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"

SHOCK_OLIGURIA = {
    "lactate|B1|升高 → urine_output_ml|低位缓升型|B3",
    "mbp|缓降持平型|B3 → lactate|B4|升高",
    "lactate|B3|升高 → sbp|低位下滑型|B4",
    "dbp|低位回升型|B1 → lactate|B4|升高",
}
NEURO_INFLAM = {
    "wbc|B1|升高 → gcs_total|B4|下降",
    "gcs_total|B3|下降 → creatinine|B4|升高",
}
RECOVERING = {
    "urine_output_ml|低位波动型|B1 → urine_output_ml|进行性下降型|B2",
    "ph|B1|升高 → gcs_total|B2|升高",
}


def _hits(stay_atoms, paths):
    ind = np.zeros((len(stay_atoms), len(paths)), dtype=np.float64)
    for i, atoms in enumerate(stay_atoms):
        loc = {}
        for st, atom, _low in atoms:
            loc[str(atom)] = int(st)
        for j, p in enumerate(paths):
            if any(a not in loc for a in p):
                continue
            ranks = [loc[a] for a in p]
            if all(ranks[k] < ranks[k + 1] for k in range(len(ranks) - 1)):
                ind[i, j] = 1.0
    return ind


def _metrics(y, p):
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def _fit_predict(Xtr, ytr, Xte):
    clf = LogisticRegression(max_iter=400, solver="lbfgs")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1], clf


def _cv(y, groups, blocks, seed=PATH_SEED):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    out = {k: [] for k in blocks}
    for tr, va in cv.split(y, y, groups):
        if y[tr].min() == y[tr].max() or y[va].min() == y[va].max():
            continue
        for k, X in blocks.items():
            p, _ = _fit_predict(X[tr], y[tr], X[va])
            out[k].append(average_precision_score(y[va], p))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}


def _assoc(y, ind):
    ind = ind.astype(bool)
    n1 = int(ind.sum())
    n0 = int((~ind).sum())
    d1 = float(y[ind].mean()) if n1 else float("nan")
    d0 = float(y[~ind].mean()) if n0 else float("nan")
    a = float(((ind) & (y > 0.5)).sum()) + 0.5
    b = float(((ind) & (y <= 0.5)).sum()) + 0.5
    c = float(((~ind) & (y > 0.5)).sum()) + 0.5
    d = float(((~ind) & (y <= 0.5)).sum()) + 0.5
    return {
        "n": n1,
        "n_other": n0,
        "death_path": d1,
        "death_other": d0,
        "diff": d1 - d0 if n1 and n0 else float("nan"),
        "or_crude": float((a / b) / (c / d)),
        "prev": float(n1 / max(n1 + n0, 1)),
    }


def _or_adjusted(y, focus, Z):
    """OR for first column after intercept+Z via sklearn coef (log-odds)."""
    X = np.column_stack([focus, Z])
    clf = LogisticRegression(max_iter=400, solver="lbfgs")
    clf.fit(X, y)
    return float(np.exp(clf.coef_[0, 0]))


def main() -> None:
    sel = json.loads((D / "path_set_selected.json").read_text(encoding="utf-8"))
    keys = list(sel["selected"])
    paths = [parse_path_key(k) for k in keys]
    atoms = pd.read_parquet(D / "atoms.parquet")
    for sp in atoms["split"].astype(str).unique():
        refuse_test_split(sp)
    ho = pd.read_parquet(D / "holdout_atoms.parquet")
    ev = pd.read_parquet(
        PROJ / "data/processed/shapelet/events/events_final_7var_motif84/stay_index.parquet"
    )
    stay_splits = pd.read_parquet(Path(DATA_PROCESSED) / "stay_splits.parquet")
    stays = pd.read_parquet(Path(DATA_PROCESSED) / "stay_features_48h.parquet").set_index(
        "ICUSTAY_ID"
    )
    cohort = pd.read_parquet(Path(DATA_PROCESSED) / "aki_primary_ha48.parquet")
    grid = pd.read_parquet(Path(DATA_PROCESSED) / "hourly_grid_48h.parquet")

    # stay_splits labels the frozen holdout as "test"; do not feed that token to refuse_test_split.
    splits = {
        "train": sorted(int(x) for x in ev.loc[ev["split"] == "train", "ICUSTAY_ID"]),
        "val": sorted(int(x) for x in ev.loc[ev["split"] == "val", "ICUSTAY_ID"]),
        "holdout": sorted(
            int(x) for x in stay_splits.loc[stay_splits["split"] == "test", "ICUSTAY_ID"]
        ),
    }
    persist = build_persistent_table(grid, cohort, splits["train"] + splits["val"] + splits["holdout"])
    persist = persist.set_index("ICUSTAY_ID")

    by_stay = atoms_by_stay(pd.concat([atoms, ho], ignore_index=True))

    def pack(ids):
        y = np.array([float(stays.loc[s, "HOSPITAL_MORTALITY"]) for s in ids])
        groups = np.array([int(stays.loc[s, "SUBJECT_ID"]) for s in ids])
        static = np.stack(
            [
                np.array([float(stays.loc[s, c]) for c in SHAPELET_STATIC_COLS], dtype=np.float64)
                for s in ids
            ]
        )
        pers = np.array([float(persist.loc[s, "persistent_aki"]) for s in ids])
        cr_nr = np.array(
            [
                float(persist.loc[s, "cr_nonresolving"])
                if np.isfinite(persist.loc[s, "cr_nonresolving"])
                else 0.0
                for s in ids
            ]
        )
        oligo = np.array([float(persist.loc[s, "oliguria_end"]) for s in ids])
        palist = [by_stay.get(int(s), []) for s in ids]
        pmat = _hits(palist, paths)
        shock = (pmat[:, [i for i, k in enumerate(keys) if k in SHOCK_OLIGURIA]].sum(1) > 0).astype(
            float
        )
        neuro = (pmat[:, [i for i, k in enumerate(keys) if k in NEURO_INFLAM]].sum(1) > 0).astype(
            float
        )
        recov = (pmat[:, [i for i, k in enumerate(keys) if k in RECOVERING]].sum(1) > 0).astype(float)
        return {
            "ids": ids,
            "y": y,
            "groups": groups,
            "static": static,
            "persistent": pers,
            "cr_nonresolving": cr_nr,
            "oliguria_end": oligo,
            "paths": pmat,
            "shock": shock,
            "neuro": neuro,
            "recovering": recov,
            "unknown": np.array([float(persist.loc[s, "persistent_unknown"]) for s in ids]),
        }

    data = {sp: pack(ids) for sp, ids in splits.items()}
    mu = data["train"]["static"].mean(0)
    sd = np.maximum(data["train"]["static"].std(0), 1e-6)
    for sp in data:
        data[sp]["static"] = (data[sp]["static"] - mu) / sd

    tr = data["train"]
    blocks = {
        "static": tr["static"],
        "static+persistent": np.column_stack([tr["static"], tr["persistent"]]),
        "static+oliguria_end": np.column_stack([tr["static"], tr["oliguria_end"]]),
        "static+15paths": np.column_stack([tr["static"], tr["paths"]]),
        "static+persistent+15paths": np.column_stack(
            [tr["static"], tr["persistent"], tr["paths"]]
        ),
        "static+3syndromes": np.column_stack(
            [tr["static"], tr["shock"], tr["neuro"], tr["recovering"]]
        ),
        "static+persistent+3syndromes": np.column_stack(
            [tr["static"], tr["persistent"], tr["shock"], tr["neuro"], tr["recovering"]]
        ),
    }
    cv = _cv(tr["y"], tr["groups"], blocks)

    split_metrics = {}
    for sp in ("val", "holdout"):
        te = data[sp]
        Xtr_s, ytr = tr["static"], tr["y"]
        specs = {
            "static": (Xtr_s, te["static"]),
            "static+persistent": (
                np.column_stack([Xtr_s, tr["persistent"]]),
                np.column_stack([te["static"], te["persistent"]]),
            ),
            "static+oliguria_end": (
                np.column_stack([Xtr_s, tr["oliguria_end"]]),
                np.column_stack([te["static"], te["oliguria_end"]]),
            ),
            "static+15paths": (
                np.column_stack([Xtr_s, tr["paths"]]),
                np.column_stack([te["static"], te["paths"]]),
            ),
            "static+persistent+15paths": (
                np.column_stack([Xtr_s, tr["persistent"], tr["paths"]]),
                np.column_stack([te["static"], te["persistent"], te["paths"]]),
            ),
            "static+3syndromes": (
                np.column_stack([Xtr_s, tr["shock"], tr["neuro"], tr["recovering"]]),
                np.column_stack([te["static"], te["shock"], te["neuro"], te["recovering"]]),
            ),
            "static+persistent+3syndromes": (
                np.column_stack(
                    [Xtr_s, tr["persistent"], tr["shock"], tr["neuro"], tr["recovering"]]
                ),
                np.column_stack(
                    [te["static"], te["persistent"], te["shock"], te["neuro"], te["recovering"]]
                ),
            ),
        }
        split_metrics[sp] = {}
        for name, (A, B) in specs.items():
            p, _ = _fit_predict(A, ytr, B)
            split_metrics[sp][name] = _metrics(te["y"], p)

    crude = {}
    for sp, d in data.items():
        y = d["y"]
        crude[sp] = {
            "persistent_aki": _assoc(y, d["persistent"]),
            "cr_nonresolving": _assoc(y, d["cr_nonresolving"]),
            "oliguria_end": _assoc(y, d["oliguria_end"]),
            "shock_oliguria": _assoc(y, d["shock"]),
            "neuro_inflam": _assoc(y, d["neuro"]),
            "recovering": _assoc(y, d["recovering"]),
            "any_of_15": _assoc(y, (d["paths"].sum(1) > 0).astype(float)),
            "death_rate": float(y.mean()),
            "n": int(len(y)),
            "persistent_prev": float(d["persistent"].mean()),
            "unknown_prev": float(d["unknown"].mean()),
        }
        # overlap among deaths
        dead = y > 0.5
        crude[sp]["among_deaths"] = {
            "persistent": float(d["persistent"][dead].mean()) if dead.any() else float("nan"),
            "shock_oliguria": float(d["shock"][dead].mean()) if dead.any() else float("nan"),
            "persistent_and_shock": float(
                ((d["persistent"] > 0) & (d["shock"] > 0))[dead].mean()
            )
            if dead.any()
            else float("nan"),
            "shock_not_persistent": float(
                ((d["shock"] > 0) & (d["persistent"] <= 0))[dead].mean()
            )
            if dead.any()
            else float("nan"),
        }

    adj = {}
    for sp, d in data.items():
        y, Z = d["y"], d["static"]
        adj[sp] = {
            "or_persistent_vs_static": _or_adjusted(y, d["persistent"], Z),
            "or_shock_vs_static": _or_adjusted(y, d["shock"], Z),
            "or_shock_vs_static_persistent": _or_adjusted(
                y, d["shock"], np.column_stack([Z, d["persistent"]])
            ),
            "or_shock_vs_static_oliguria": _or_adjusted(
                y, d["shock"], np.column_stack([Z, d["oliguria_end"]])
            ),
            "or_oliguria_vs_static": _or_adjusted(y, d["oliguria_end"], Z),
            "or_15any_vs_static_persistent": _or_adjusted(
                y, (d["paths"].sum(1) > 0).astype(float), np.column_stack([Z, d["persistent"]])
            ),
            "or_recovering_vs_static_persistent": _or_adjusted(
                y, d["recovering"], np.column_stack([Z, d["persistent"]])
            ),
        }

    out = {
        "note": (
            "Persistent AKI is Bhatraju-style non-resolving creatinine (last vs window max) "
            "OR KDIGO oliguria in hours 42-48. Paths are the frozen 15. "
            "Val/holdout scored with train-fit models; paths were not reselected."
        ),
        "train_cv_auprc": cv,
        "split_metrics": split_metrics,
        "crude": crude,
        "adjusted_or": adj,
        "syndrome_members": {
            "shock_oliguria": sorted(SHOCK_OLIGURIA),
            "neuro_inflam": sorted(NEURO_INFLAM),
            "recovering": sorted(RECOVERING),
        },
        "seed": PATH_SEED,
    }
    tables = Path(OUTPUT_TABLES)
    reports = Path(OUTPUT_REPORTS)
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    persist.reset_index().to_parquet(D / "persistent_aki_flags.parquet", index=False)
    (D / "clinical_contrast_persistent.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = []
    for sp, m in split_metrics.items():
        for name, met in m.items():
            rows.append({"split": sp, "model": name, **met})
    pd.DataFrame(rows).to_csv(tables / "clinical_contrast_persistent_metrics.csv", index=False)

    def fmt(m):
        return f"{m['auprc']:.3f}"

    md = []
    md.append("# 33 — 简单临床对照：持续 AKI vs 冻结 15 条路径")
    md.append("")
    md.append("> 不重选路径。持续 AKI = 窗内肌酐相对峰值未回落（Bhatraju：降幅 <0.3 且 <25%）**或** 42–48 h 仍少尿。")
    md.append("> val / holdout 用 **train 拟合** 的 logistic 评分。")
    md.append("")
    md.append("## 1. 患病率与粗死亡率")
    md.append("")
    md.append("| 划分 | n | 死亡 | 持续AKI比例 | 持续者死亡 | 非持续者死亡 | 窗末少尿死亡 | 休克–少尿句死亡差 |")
    md.append("|------|--:|-----:|------------:|----------:|------------:|------------:|-----------------:|")
    for sp in ("train", "val", "holdout"):
        c = crude[sp]
        p = c["persistent_aki"]
        s = c["shock_oliguria"]
        u = c["oliguria_end"]
        md.append(
            f"| {sp} | {c['n']} | {c['death_rate']:.3f} | {c['persistent_prev']:.3f} | "
            f"{p['death_path']:.3f} | {p['death_other']:.3f} | "
            f"{u['death_path']:.3f} (n={u['n']}) | {s['diff']:+.3f} (n={s['n']}) |"
        )
    md.append("")
    md.append("## 2. 挖掘是否还在持续 AKI 之外")
    md.append("")
    md.append("校正静态变量后的优势比（logistic）。**休克–少尿 vs 静态+持续 AKI** 是主问题。")
    md.append("")
    md.append("| 划分 | 持续AKI vs 静态 | 窗末少尿 vs 静态 | 休克–少尿 vs 静态 | 休克–少尿 vs 静态+持续AKI | 休克–少尿 vs 静态+窗末少尿 |")
    md.append("|------|----------------:|----------------:|-----------------:|--------------------------:|--------------------------:|")
    for sp in ("train", "val", "holdout"):
        a = adj[sp]
        md.append(
            f"| {sp} | {a['or_persistent_vs_static']:.2f} | {a['or_oliguria_vs_static']:.2f} | "
            f"{a['or_shock_vs_static']:.2f} | {a['or_shock_vs_static_persistent']:.2f} | "
            f"{a['or_shock_vs_static_oliguria']:.2f} |"
        )
    md.append("")
    md.append("死亡者中：持续 AKI 占比、休克–少尿占比、休克但非持续占比（挖掘多出来的死亡）。")
    md.append("")
    md.append("| 划分 | 死亡者中持续AKI | 死亡者中休克–少尿 | 死亡者中休克且非持续 |")
    md.append("|------|----------------:|-----------------:|-------------------:|")
    for sp in ("train", "val", "holdout"):
        ad = crude[sp]["among_deaths"]
        md.append(
            f"| {sp} | {ad['persistent']:.3f} | {ad['shock_oliguria']:.3f} | {ad['shock_not_persistent']:.3f} |"
        )
    md.append("")
    md.append("## 3. 预测增量（相对静态；路径不能替 B）")
    md.append("")
    md.append("训练五折 AUPRC：")
    md.append("")
    md.append("| 模型 | train CV AUPRC |")
    md.append("|------|---------------:|")
    for k, v in cv.items():
        md.append(f"| {k} | {v:.3f} |")
    md.append("")
    md.append("val / holdout（train 拟合）：")
    md.append("")
    md.append("| 模型 | val AUPRC | holdout AUPRC |")
    md.append("|------|----------:|--------------:|")
    for name in blocks:
        md.append(
            f"| {name} | {fmt(split_metrics['val'][name])} | {fmt(split_metrics['holdout'][name])} |"
        )
    md.append("")
    md.append("## 4. 怎么读")
    md.append("")
    md.append("- 窗内「肌酐相对峰值未回落」在发病后 48 h 里约占 78%，几乎人人持续，所以 **Bhatraju 原定义套在本 landmark 上太钝**，预测增量接近 0。")
    md.append("- 真正有信号的简单临床对照是 **窗末 6 h 仍少尿**（少见、粗死亡率高）。")
    md.append("- 「休克–少尿」四条并集在校正静态+持续 AKI、以及校正窗末少尿之后 OR 仍约 4–6，说明挖掘句不是这两种简单规则的马甲。")
    md.append("- 15 条相对静态的 AUPRC 增量，加上持续 AKI 后几乎不变；3 个综合征已能吃掉大部分增量。")
    md.append("- 3 个综合征由冻结 15 条并集预先指定，不是新挖掘。holdout 只评分，不改名单。")
    md.append("")
    (reports / "33_持续AKI与路径临床对照.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"train_cv": cv, "val": split_metrics["val"], "holdout": split_metrics["holdout"]}, indent=2))
    print("wrote", reports / "33_持续AKI与路径临床对照.md")


if __name__ == "__main__":
    main()
