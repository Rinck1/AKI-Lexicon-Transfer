#!/usr/bin/env python3
"""51b — MINE 收尾：对比统计 (a)(b)(c) + eICU 内部效用（round-2 探索性）。

只读已有产物；新产物写 round2_mine/*_extras.*；不改任何现有文件。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.eicu_io import load_eicu  # noqa: E402
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402

MINE_DIR = PROJ / "data" / "external" / "eicu" / "round2_mine"


def _norm_atom(s: str) -> str:
    """'var|family|B2|B2' → 'var|family|B2'（51 号脚本把阶段拼了两次）"""
    parts = [p for p in str(s).split("|") if p != ""]
    return "|".join(parts[:3])


def main() -> None:
    out: dict = {"round": "round-2", "exploratory": True}

    top50 = pd.read_csv(MINE_DIR / "eicu_top50_readable_stable.csv")
    cmp_b = pd.read_csv(MINE_DIR / "compare_b_unordered_pairs.csv")
    cmp_a = pd.read_csv(MINE_DIR / "compare_a_mimic15_in_eicu.csv")

    # (a) 无序原子对重合计数
    out["a_unordered_pairs"] = {
        "n_mimic_pairs": int(len(cmp_b)),
        "in_stable_pool": int(cmp_b["in_eicu_stable_pool"].sum()),
        "in_top50": int(cmp_b["in_eicu_top50"].sum()),
        "any_support_ge20": int(cmp_b["in_eicu_any_ge20"].sum()),
    }

    # (b) 端点 Jaccard（含阶段原子 & 变量级两种）
    mimic_atoms = set()
    for p in cmp_a["mimic_path"]:
        for a in str(p).split(" → "):
            mimic_atoms.add(_norm_atom(a))
    eicu_atoms = set()
    for p in top50["path"]:
        for a in str(p).split(" → "):
            eicu_atoms.add(_norm_atom(a))
    inter = mimic_atoms & eicu_atoms
    jac_atom = len(inter) / len(mimic_atoms | eicu_atoms) if mimic_atoms | eicu_atoms else float("nan")
    mimic_vars = {a.split("|")[0] for a in mimic_atoms}
    eicu_vars = {a.split("|")[0] for a in eicu_atoms}
    out["b_endpoint_jaccard"] = {
        "mimic_endpoints": sorted(mimic_atoms),
        "eicu_top50_endpoints": sorted(eicu_atoms),
        "shared_atoms": sorted(inter),
        "jaccard_atom_level": round(jac_atom, 4),
        "jaccard_var_level": round(
            len(mimic_vars & eicu_vars) / len(mimic_vars | eicu_vars), 4
        ),
    }

    # (c) 变量构成对比（端点出现次数占比）
    def var_dist(atoms: set) -> dict:
        c = pd.Series([a.split("|")[0] for a in atoms]).value_counts()
        return {k: int(v) for k, v in c.items()}

    out["c_variable_composition"] = {
        "mimic15": var_dist(mimic_atoms),
        "eicu_top50": var_dist(eicu_atoms),
    }

    # eICU 内部效用：static vs static+top15 原生路径，5 折分层 CV（eICU 内部，seed 42）
    top15 = top50.head(15)
    paths = [[_norm_atom(x) for x in [r.a1, r.a2]] for r in top15.itertuples(index=False)]

    atoms = load_eicu("atoms")
    stay_atoms = atoms_by_stay(atoms)
    stays = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    ids = stays.index.to_numpy()
    y = stays.loc[ids, "HOSPITAL_MORTALITY"].to_numpy(dtype=float)
    X = stays.loc[ids, SHAPELET_STATIC_COLS].to_numpy(dtype=float)

    astages: dict[int, dict[str, set]] = {}
    for sid, lst in stay_atoms.items():
        d: dict[str, set] = {}
        for st, atom, _ in lst:
            d.setdefault(str(atom), set()).add(int(st))
        astages[int(sid)] = d

    ind = np.zeros((len(ids), 15), dtype=float)
    pos = {int(s): i for i, s in enumerate(ids)}
    for sid, d in astages.items():
        i = pos.get(sid)
        if i is None:
            continue
        for j, p in enumerate(paths):
            prev, ok = -1, True
            for a in p:
                stages = d.get(a)
                if not stages:
                    ok = False
                    break
                nxt = next((s for s in sorted(stages) if s > prev), None)
                if nxt is None:
                    ok = False
                    break
                prev = nxt
            if ok:
                ind[i, j] = 1.0

    aucs_s, aucs_sp = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        m_s = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400)).fit(X[tr], y[tr])
        aucs_s.append(average_precision_score(y[te], m_s.predict_proba(X[te])[:, 1]))
        m_sp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400)).fit(
            np.column_stack([X[tr], ind[tr]]), y[tr]
        )
        aucs_sp.append(
            average_precision_score(
                y[te], m_sp.predict_proba(np.column_stack([X[te], ind[te]]))[:, 1]
            )
        )
    out["eicu_internal_utility_cv5"] = {
        "static_mean_auprc": round(float(np.mean(aucs_s)), 4),
        "static_plus_top15_native_mean_auprc": round(float(np.mean(aucs_sp)), 4),
        "n": int(len(ids)),
        "top15_trigger_rate": [round(float(ind[:, j].mean()), 4) for j in range(15)],
    }

    dest = MINE_DIR / "mine_extras.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"→ {dest}")


if __name__ == "__main__":
    main()
