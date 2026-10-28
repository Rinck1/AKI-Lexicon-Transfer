#!/usr/bin/env python3
"""Train-only §8.3/§8.4 audit of the frozen 15-path set. Does not read holdout labels to retune.

3: motif low-confidence rate vs train baseline (+5 pp allowed).
4: selected set vs random equal-length readable sets (train 5-fold AUPRC).
If path_holdout_reproduce.json already exists, copy its AUPRC into the report only.
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
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.path_interpret import atom_lowconf_map, motif_lowconf_rate  # noqa: E402
from scripts.utils.path_mine import path_indicator_matrix  # noqa: E402
from scripts.utils.path_protocol import (  # noqa: E402
    LOW_CONF_ENRICH_TOL,
    PATH_SEED,
    parse_path_key,
    refuse_test_path,
    refuse_test_split,
)
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402

DEFAULT = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"
N_RANDOM = 20


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-dir", default=str(DEFAULT))
    p.add_argument("--n-random", type=int, default=N_RANDOM)
    p.add_argument("--report", default=str(PROJ / "outputs/reports/23_时序路径集合审计.md"))
    return p.parse_args()


def _cv_auprc(y, groups, static, path_ind, *, seed: int) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, va in cv.split(static, y, groups):
        if y[tr].min() == y[tr].max() or y[va].min() == y[va].max():
            continue
        Xtr = np.column_stack([static[tr], path_ind[tr]]) if path_ind.shape[1] else static[tr]
        Xva = np.column_stack([static[va], path_ind[va]]) if path_ind.shape[1] else static[va]
        clf = LogisticRegression(max_iter=300, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        scores.append(average_precision_score(y[va], clf.predict_proba(Xva)[:, 1]))
    return float(np.mean(scores)) if scores else float("nan")


def main() -> None:
    args = parse_args()
    d = Path(args.paths_dir)
    refuse_test_path(d)
    refuse_test_path(args.report)
    sel = json.loads((d / "path_set_selected.json").read_text(encoding="utf-8"))
    chosen = [parse_path_key(p) for p in sel["selected"]]
    stable = pd.read_parquet(d / "path_stable_train.parquet")
    atoms = pd.read_parquet(d / "atoms.parquet")
    for sp in atoms["split"].astype(str).unique():
        refuse_test_split(sp)
    train_atoms = atoms[atoms["split"] == "train"]
    stay_map = atoms_by_stay(train_atoms)

    ev_idx = PROJ / "data/processed/shapelet/events/events_final_7var_motif84/stay_index.parquet"
    refuse_test_path(ev_idx)
    idx = pd.read_parquet(ev_idx)
    train_ids = sorted(int(x) for x in idx.loc[idx["split"] == "train", "ICUSTAY_ID"])
    stays = pd.read_parquet(Path(DATA_PROCESSED) / "stay_features_48h.parquet")
    stay_idx = stays.set_index("ICUSTAY_ID")
    y = np.array([float(stay_idx.loc[s, "HOSPITAL_MORTALITY"]) for s in train_ids])
    groups = np.array([int(stay_idx.loc[s, "SUBJECT_ID"]) for s in train_ids])
    static = np.stack(
        [np.array([float(stay_idx.loc[s, c]) for c in SHAPELET_STATIC_COLS], dtype=np.float64) for s in train_ids]
    )
    mu, sd = static.mean(0), np.maximum(static.std(0), 1e-6)
    static = (static - mu) / sd

    low_map = atom_lowconf_map(
        [(str(r.atom), bool(r.family_low_confidence), str(r.kind)) for r in train_atoms.itertuples(index=False)]
    )
    motif_train = train_atoms[train_atoms["kind"] == "motif"]
    baseline = float(motif_train["family_low_confidence"].mean()) if len(motif_train) else 0.0
    set_rate = float(np.mean([motif_lowconf_rate(p, low_map) for p in chosen])) if chosen else 0.0
    low_ok = set_rate <= baseline + LOW_CONF_ENRICH_TOL

    selected_ind = path_indicator_matrix(train_ids, stay_map, chosen)
    empty = np.zeros((len(train_ids), 0))
    a_static = _cv_auprc(y, groups, static, empty, seed=PATH_SEED)
    a_sel = _cv_auprc(y, groups, static, selected_ind, seed=PATH_SEED)

    pool = stable[(stable["readable"].astype(bool)) & (stable["n_atoms"] == 2)].copy()
    chosen_keys = set(sel["selected"])
    pool = pool[~pool["path"].isin(chosen_keys)].reset_index(drop=True)
    rng = np.random.default_rng(PATH_SEED)
    n_take = min(len(chosen), len(pool))
    random_scores = []
    for i in range(int(args.n_random)):
        pick = rng.choice(len(pool), size=n_take, replace=False)
        rand_paths = [parse_path_key(pool.iloc[int(j)]["path"]) for j in pick]
        rind = path_indicator_matrix(train_ids, stay_map, rand_paths)
        sc = _cv_auprc(y, groups, static, rind, seed=PATH_SEED)
        random_scores.append(sc)
        _log(f"[27] random {i+1}/{args.n_random} AUPRC={sc:.4f}")
    rand_mean = float(np.mean(random_scores)) if random_scores else float("nan")
    rand_hi = float(np.quantile(random_scores, 0.95)) if random_scores else float("nan")
    beat = bool(np.isfinite(a_sel) and np.isfinite(rand_mean) and a_sel > rand_mean)

    hold = {}
    hold_path = d / "path_holdout_reproduce.json"
    if hold_path.is_file():
        hold = json.loads(hold_path.read_text(encoding="utf-8"))

    payload = {
        "stage": "path_set_audit_train",
        "test_read": False,
        "val_used": False,
        "n_selected": len(chosen),
        "baseline_motif_lowconf": baseline,
        "selected_motif_lowconf": set_rate,
        "lowconf_ok": low_ok,
        "static_cv_auprc": a_static,
        "selected_cv_auprc": a_sel,
        "random_n": int(args.n_random),
        "random_mean_cv_auprc": rand_mean,
        "random_p95_cv_auprc": rand_hi,
        "selected_beats_random_mean": beat,
        "holdout_auprc_static": hold.get("holdout_auprc_static"),
        "holdout_auprc_static_plus_paths": hold.get("holdout_auprc_static_plus_paths"),
        "seed": PATH_SEED,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (d / "path_set_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# 时序路径集合审计（train，§8.3 / §8.4）",
        "",
        "> 不改名单、不读 holdout 调参。随机对照只在 train 折内。",
        "",
        f"- 低置信：基线 {baseline:.3f}，入选集合 {set_rate:.3f}，允许 +{LOW_CONF_ENRICH_TOL:.2f}。过关：{'是' if low_ok else '否'}。",
        f"- train 五折 AUPRC 静态 {a_static:.4f}；入选 15 条 {a_sel:.4f}；随机等长可读集合均值 {rand_mean:.4f}（P95 {rand_hi:.4f}）。",
        f"- 入选 > 随机均值：{'是' if beat else '否'}。",
    ]
    if hold:
        lines.append(
            f"- holdout（已跑过则只转录）：静态 {hold.get('holdout_auprc_static')}；"
            f"静态+路径 {hold.get('holdout_auprc_static_plus_paths')}。"
        )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(json.dumps(payload, ensure_ascii=False, indent=2))
    _log("[27] DONE")


if __name__ == "__main__":
    main()
