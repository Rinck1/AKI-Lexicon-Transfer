#!/usr/bin/env python3
"""Train-fold forward selection of a small readable path set (docs/20 §6 / §9.3)."""

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
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.path_mine import path_indicator_matrix  # noqa: E402
from scripts.utils.path_protocol import (  # noqa: E402
    MAX_SELECTED_PATHS,
    MIN_AUPRC_GAIN,
    PATH_SEED,
    parse_path_key,
    refuse_test_path,
    refuse_test_split,
)
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402

DEFAULT = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-dir", default=str(DEFAULT))
    return p.parse_args()


def _cv_auprc(
    y: np.ndarray,
    groups: np.ndarray,
    static: np.ndarray,
    path_ind: np.ndarray,
    *,
    seed: int,
) -> float:
    n_splits = 5
    if len(np.unique(y)) < 2:
        return float("nan")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, va in cv.split(static, y, groups):
        if y[tr].min() == y[tr].max() or y[va].min() == y[va].max():
            continue
        Xtr = np.column_stack([static[tr], path_ind[tr]]) if path_ind.shape[1] else static[tr]
        Xva = np.column_stack([static[va], path_ind[va]]) if path_ind.shape[1] else static[va]
        clf = LogisticRegression(max_iter=300, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xva)[:, 1]
        scores.append(average_precision_score(y[va], p))
    return float(np.mean(scores)) if scores else float("nan")


def main() -> None:
    args = parse_args()
    d = Path(args.paths_dir)
    refuse_test_path(d)
    stable = pd.read_parquet(d / "path_stable_train.parquet")
    atoms = pd.read_parquet(d / "atoms.parquet")
    for sp in atoms["split"].astype(str).unique():
        refuse_test_split(sp)
    train_atoms = atoms[atoms["split"] == "train"]
    stay_atoms = atoms_by_stay(train_atoms)

    ev_idx = PROJ / "data/processed/shapelet/events/events_final_7var_motif84/stay_index.parquet"
    idx = pd.read_parquet(ev_idx)
    train_ids = sorted(int(x) for x in idx.loc[idx["split"] == "train", "ICUSTAY_ID"])
    _grid, stays = load_shapelet_tables(DATA_PROCESSED)
    stay_idx = stays.set_index("ICUSTAY_ID")
    y = np.array([float(stay_idx.loc[s, "HOSPITAL_MORTALITY"]) for s in train_ids])
    groups = np.array([int(stay_idx.loc[s, "SUBJECT_ID"]) for s in train_ids])
    static = np.stack(
        [np.array([float(stay_idx.loc[s, c]) for c in SHAPELET_STATIC_COLS], dtype=np.float64) for s in train_ids]
    )
    mu, sd = static.mean(0), np.maximum(static.std(0), 1e-6)
    static = (static - mu) / sd

    # §6 readability hard constraint; univariate screen so greedy CV stays
    # in the protocol's "dozens–~200" pool (4534 stable is too large to scan).
    SCREEN_TOP = 120
    work = stable.copy()
    if work.empty:
        payload = {"selected": [], "note": "no stable paths", "test_read": False}
        (d / "path_set_selected.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _log("[23] no stable paths")
        return
    if "readable" in work.columns:
        readable = work[work["readable"].astype(bool)]
        if len(readable):
            work = readable
    work = work.copy()
    work["_abs_coef"] = work["logit_coef"].abs()
    work["_abs_diff"] = work["death_rate_diff"].abs()
    work = work.sort_values(
        ["_abs_coef", "_abs_diff", "train_support", "n_atoms"],
        ascending=[False, False, False, True],
    ).head(SCREEN_TOP).reset_index(drop=True)
    candidates = [parse_path_key(p) for p in work["path"].tolist()]
    all_ind = path_indicator_matrix(train_ids, stay_atoms, candidates)

    selected: list[int] = []
    base = _cv_auprc(y, groups, static, np.zeros((len(train_ids), 0)), seed=PATH_SEED)
    _log(f"[23] static-only CV AUPRC={base:.4f}  candidates={len(candidates)}")
    cur = base
    for _step in range(MAX_SELECTED_PATHS):
        best_j, best_s = None, cur
        for j, _p in enumerate(candidates):
            if j in selected:
                continue
            cols = selected + [j]
            sc = _cv_auprc(y, groups, static, all_ind[:, cols], seed=PATH_SEED)
            if np.isfinite(sc) and sc > best_s + MIN_AUPRC_GAIN:
                best_s, best_j = sc, j
        if best_j is None:
            break
        selected.append(best_j)
        cur = best_s
        _log(f"[23] + {path_label(candidates[best_j])}  AUPRC={cur:.4f}")

    chosen_keys = [path_label(candidates[j]) for j in selected]
    payload = {
        "stage": "path_set_forward_select",
        "test_read": False,
        "val_used": False,
        "static_cv_auprc": base,
        "selected_cv_auprc": cur,
        "n_selected": len(chosen_keys),
        "selected": chosen_keys,
        "readable": [bool(work.iloc[j]["readable"]) for j in selected],
        "seed": PATH_SEED,
        "max_selected": MAX_SELECTED_PATHS,
        "min_gain": MIN_AUPRC_GAIN,
        "screen_top": SCREEN_TOP,
        "n_screened": len(candidates),
        "screen": "readable_then_abs_logit_top",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (d / "path_set_selected.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _log(f"[23] selected {len(chosen_keys)} paths  DONE")


def path_label(atoms: tuple[str, ...]) -> str:
    from scripts.utils.path_protocol import path_key

    return path_key(atoms)


if __name__ == "__main__":
    main()
