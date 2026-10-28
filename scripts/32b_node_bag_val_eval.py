#!/usr/bin/env python3
"""Same-split val AUPRC for static+84 node indicators (train-fit). Reporting only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import sys

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
from scripts.utils.constants import DATA_PROCESSED
from scripts.utils.path_protocol import PATH_SEED, refuse_test_split
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS

D = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"


def xy(stays, ids):
    y = np.array([float(stays.loc[s, "HOSPITAL_MORTALITY"]) for s in ids])
    X = np.stack(
        [np.array([float(stays.loc[s, c]) for c in SHAPELET_STATIC_COLS], dtype=np.float64) for s in ids]
    )
    return y, X


def node_mat(atoms, ids, nodes):
    pos = {int(s): i for i, s in enumerate(ids)}
    idx = {n: j for j, n in enumerate(nodes)}
    m = np.zeros((len(ids), len(nodes)))
    for rec in atoms.itertuples(index=False):
        i = pos.get(int(rec.ICUSTAY_ID))
        j = idx.get(str(rec.atom))
        if i is not None and j is not None:
            m[i, j] = 1.0
    return m


def main() -> None:
    atoms = pd.read_parquet(D / "motif_atoms.parquet")
    for sp in atoms["split"].astype(str).unique():
        refuse_test_split(sp)
    ev = pd.read_parquet(PROJ / "data/processed/shapelet/events/events_final_7var_motif84/stay_index.parquet")
    tr = sorted(int(x) for x in ev.loc[ev["split"] == "train", "ICUSTAY_ID"])
    va = sorted(int(x) for x in ev.loc[ev["split"] == "val", "ICUSTAY_ID"])
    stays = pd.read_parquet(Path(DATA_PROCESSED) / "stay_features_48h.parquet").set_index("ICUSTAY_ID")
    ytr, Xtr = xy(stays, tr)
    yva, Xva = xy(stays, va)
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    nodes = sorted(atoms.loc[atoms["split"] == "train", "atom"].astype(str).unique())
    ntr = node_mat(atoms[atoms["split"] == "train"], tr, nodes)
    nva = node_mat(atoms[atoms["split"] == "val"], va, nodes)
    clf0 = LogisticRegression(max_iter=400, solver="lbfgs").fit(Xtr, ytr)
    clf1 = LogisticRegression(max_iter=400, solver="lbfgs").fit(np.column_stack([Xtr, ntr]), ytr)
    p0, p1 = clf0.predict_proba(Xva)[:, 1], clf1.predict_proba(np.column_stack([Xva, nva]))[:, 1]
    tab = pd.read_csv(D / "node_outcome_assoc_train.csv")
    se = (tab["diff_hi"] - tab["diff_lo"]) / (2 * 1.96)
    z = 3.437  # two-sided 0.05/84
    bonf = int((tab["death_rate_diff"].abs() > z * se.replace(0, np.nan)).sum())
    out = {
        "val_auprc_static": float(average_precision_score(yva, p0)),
        "val_auprc_static_plus_84": float(average_precision_score(yva, p1)),
        "val_auroc_static": float(roc_auc_score(yva, p0)),
        "val_auroc_static_plus_84": float(roc_auc_score(yva, p1)),
        "val_brier_static": float(brier_score_loss(yva, p0)),
        "val_brier_static_plus_84": float(brier_score_loss(yva, p1)),
        "n_nodes": len(nodes),
        "n_ci95_excludes_0": int(tab["ci_excludes_0"].sum()),
        "n_bonferroni_0.05_over_84": bonf,
        "bonferroni_note": "Approximate: |diff| > z_{1-0.025/84}*se, se from bootstrap 95% CI width/3.92",
        "seed": PATH_SEED,
    }
    (D / "node_bag_val_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
