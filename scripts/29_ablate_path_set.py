#!/usr/bin/env python3
"""Pre-specified path-set ablations. Train 5-fold only; holdout scored if atoms exist, never used to pick an arm."""

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
from scripts.utils.path_ablate import (  # noqa: E402
    filter_by_sign,
    filter_has_lab,
    filter_motif_only,
    leave_one_out,
    prefix_paths,
)
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.path_mine import path_indicator_matrix  # noqa: E402
from scripts.utils.path_protocol import (  # noqa: E402
    PATH_SEED,
    parse_path_key,
    path_key,
    refuse_test_path,
    refuse_test_split,
)
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402

DEFAULT = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"
N_RANDOM = 10


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-dir", default=str(DEFAULT))
    p.add_argument("--n-random", type=int, default=N_RANDOM)
    p.add_argument("--report", default=str(PROJ / "outputs/reports/24_时序路径消融.md"))
    p.add_argument("--skip-holdout", action="store_true")
    return p.parse_args()


def _cv(y, groups, static, path_ind, *, use_static: bool, seed: int) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, va in cv.split(static, y, groups):
        if y[tr].min() == y[tr].max() or y[va].min() == y[va].max():
            continue
        parts_tr, parts_va = [], []
        if use_static:
            parts_tr.append(static[tr])
            parts_va.append(static[va])
        if path_ind.shape[1]:
            parts_tr.append(path_ind[tr])
            parts_va.append(path_ind[va])
        if not parts_tr:
            continue
        clf = LogisticRegression(max_iter=300, solver="lbfgs")
        clf.fit(np.column_stack(parts_tr), y[tr])
        scores.append(average_precision_score(y[va], clf.predict_proba(np.column_stack(parts_va))[:, 1]))
    return float(np.mean(scores)) if scores else float("nan")


def _fit_auprc(Xtr, Xho, ytr, yho) -> float:
    if ytr.min() == ytr.max() or yho.min() == yho.max():
        return float("nan")
    clf = LogisticRegression(max_iter=300, solver="lbfgs")
    clf.fit(Xtr, ytr)
    return float(average_precision_score(yho, clf.predict_proba(Xho)[:, 1]))


def _stack(static, ind, use_static: bool):
    parts = []
    if use_static:
        parts.append(static)
    if ind.shape[1]:
        parts.append(ind)
    if not parts:
        return np.zeros((static.shape[0], 1))
    return np.column_stack(parts)


def main() -> None:
    args = parse_args()
    d = Path(args.paths_dir)
    refuse_test_path(d)
    refuse_test_path(args.report)
    sel = json.loads((d / "path_set_selected.json").read_text(encoding="utf-8"))
    keys: list[str] = list(sel["selected"])
    chosen = [parse_path_key(k) for k in keys]
    stable = pd.read_parquet(d / "path_stable_train.parquet")
    sign_of = {str(r.path): int(r.logit_sign) for r in stable.itertuples(index=False)}
    atoms = pd.read_parquet(d / "atoms.parquet")
    for sp in atoms["split"].astype(str).unique():
        refuse_test_split(sp)
    stay_map = atoms_by_stay(atoms[atoms["split"] == "train"])

    ev_idx = PROJ / "data/processed/shapelet/events/events_final_7var_motif84/stay_index.parquet"
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

    long_paths: list[tuple[str, ...]] = []
    long_f = d / "path_long_interpret_train.parquet"
    if long_f.is_file():
        long_df = pd.read_parquet(long_f)
        if "independent" in long_df.columns:
            long_paths = [parse_path_key(p) for p in long_df.loc[long_df["independent"], "path"]]

    empty = np.zeros((len(train_ids), 0))
    arms: list[dict] = [
        {"id": "P0_static", "use_static": True, "paths": [], "note": "静态基线"},
        {"id": "P1_paths_only", "use_static": False, "paths": chosen, "note": "仅15条路径"},
        {"id": "P2_static_plus_15", "use_static": True, "paths": chosen, "note": "正文：静态+冻结15条"},
        {"id": "P3_prefix5", "use_static": True, "paths": prefix_paths(keys, 5), "note": "前向选择前5条"},
        {"id": "P4_prefix10", "use_static": True, "paths": prefix_paths(keys, 10), "note": "前向选择前10条"},
        {"id": "P5_risk_pos", "use_static": True, "paths": filter_by_sign(keys, sign_of, 1), "note": "仅风险升高方向"},
        {"id": "P6_risk_neg", "use_static": True, "paths": filter_by_sign(keys, sign_of, -1), "note": "仅保护方向"},
        {"id": "P7_motif_only", "use_static": True, "paths": filter_motif_only(keys), "note": "仅形态-形态"},
        {"id": "P8_has_lab", "use_static": True, "paths": filter_has_lab(keys), "note": "至少一端是检验升降"},
        {"id": "P9_indep_len3", "use_static": True, "paths": long_paths, "note": "附录3条独立3步（train）"},
    ]
    for i, k in enumerate(keys):
        arms.append(
            {
                "id": f"LOO_{i:02d}",
                "use_static": True,
                "paths": leave_one_out(keys, i),
                "note": f"去掉 {k}",
            }
        )

    pool = stable[(stable["readable"].astype(bool)) & (stable["n_atoms"] == 2)]
    pool = pool[~pool["path"].isin(keys)].reset_index(drop=True)
    rng = np.random.default_rng(PATH_SEED)
    rand_scores = []
    n_take = min(len(keys), len(pool))

    rows = []
    cache: dict[tuple[str, ...], np.ndarray] = {}

    def ind_of(paths: list[tuple[str, ...]]) -> np.ndarray:
        if not paths:
            return empty
        missing = [p for p in paths if p not in cache]
        if missing:
            mat = path_indicator_matrix(train_ids, stay_map, missing)
            for j, p in enumerate(missing):
                cache[p] = mat[:, [j]]
        return np.column_stack([cache[p] for p in paths])

    for arm in arms:
        ind = ind_of(arm["paths"])
        sc = _cv(y, groups, static, ind, use_static=bool(arm["use_static"]), seed=PATH_SEED)
        rec = {
            "arm": arm["id"],
            "n_paths": len(arm["paths"]),
            "use_static": bool(arm["use_static"]),
            "train_cv_auprc": sc,
            "note": arm["note"],
            "paths": [path_key(p) for p in arm["paths"]],
        }
        rows.append(rec)
        _log(f"[29] {arm['id']} n={rec['n_paths']} cv={sc:.4f}")

    for i in range(int(args.n_random)):
        pick = rng.choice(len(pool), size=n_take, replace=False)
        rpaths = [parse_path_key(pool.iloc[int(j)]["path"]) for j in pick]
        sc = _cv(y, groups, static, ind_of(rpaths), use_static=True, seed=PATH_SEED)
        rand_scores.append(sc)
        _log(f"[29] random {i+1}/{args.n_random} cv={sc:.4f}")

    hold_rows = []
    ho_atoms_p = d / "holdout_atoms.parquet"
    if (not args.skip_holdout) and ho_atoms_p.is_file():
        splits = pd.read_parquet(PROJ / "data/processed/splits/stay_splits.parquet")
        hold_ids = sorted(int(x) for x in splits.loc[splits["split"] == "test", "ICUSTAY_ID"])
        ho_map = atoms_by_stay(pd.read_parquet(ho_atoms_p))
        y_ho = np.array([float(stay_idx.loc[s, "HOSPITAL_MORTALITY"]) for s in hold_ids])
        X_ho_raw = np.stack(
            [np.array([float(stay_idx.loc[s, c]) for c in SHAPELET_STATIC_COLS], dtype=np.float64) for s in hold_ids]
        )
        Xtr_raw = np.stack(
            [np.array([float(stay_idx.loc[s, c]) for c in SHAPELET_STATIC_COLS], dtype=np.float64) for s in train_ids]
        )
        mu_h, sd_h = Xtr_raw.mean(0), np.maximum(Xtr_raw.std(0), 1e-6)
        Xtr_s = (Xtr_raw - mu_h) / sd_h
        Xho_s = (X_ho_raw - mu_h) / sd_h
        ho_cache: dict[tuple[str, ...], np.ndarray] = {}

        def ho_ind(paths: list[tuple[str, ...]]) -> np.ndarray:
            if not paths:
                return np.zeros((len(hold_ids), 0))
            miss = [p for p in paths if p not in ho_cache]
            if miss:
                mat = path_indicator_matrix(hold_ids, ho_map, miss)
                for j, p in enumerate(miss):
                    ho_cache[p] = mat[:, [j]]
            return np.column_stack([ho_cache[p] for p in paths])

        report_ids = {a["id"] for a in arms if not str(a["id"]).startswith("LOO_")}
        for arm in arms:
            if arm["id"] not in report_ids:
                continue
            use_st = bool(arm["use_static"])
            Xtr = _stack(static, ind_of(arm["paths"]), use_st)
            Xho = _stack(Xho_s, ho_ind(arm["paths"]), use_st)
            sc = _fit_auprc(Xtr, Xho, y, y_ho)
            hold_rows.append({"arm": arm["id"], "holdout_auprc": sc})
            _log(f"[29] holdout {arm['id']} {sc:.4f}")

    out = {
        "stage": "path_set_ablation",
        "test_used_to_select": False,
        "seed": PATH_SEED,
        "n_random": int(args.n_random),
        "random_mean_cv_auprc": float(np.mean(rand_scores)) if rand_scores else None,
        "random_p95_cv_auprc": float(np.quantile(rand_scores, 0.95)) if rand_scores else None,
        "arms": rows,
        "holdout": hold_rows,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (d / "path_ablation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([{k: r[k] for k in r if k != "paths"} for r in rows]).to_csv(
        d / "path_ablation_train.csv", index=False, encoding="utf-8-sig"
    )
    lines = [
        "# 时序路径消融（预先指定臂）",
        "",
        "> train 五折选数字；holdout 仅当 `holdout_atoms.parquet` 已存在时转录，不用于改名单。",
        "",
        f"- 随机等长可读集合 {args.n_random} 次均值 {out['random_mean_cv_auprc']}。",
        "",
        "| 臂 | 条数 | 静态 | train CV AUPRC | 说明 |",
        "|---|---:|---|---:|---|",
    ]
    for r in rows:
        if str(r["arm"]).startswith("LOO_"):
            continue
        lines.append(
            f"| {r['arm']} | {r['n_paths']} | {'是' if r['use_static'] else '否'} | "
            f"{r['train_cv_auprc']:.4f} | {r['note']} |"
        )
    lines += ["", "## 逐条去掉（相对 P2）", "", "| 去掉 | ΔCV AUPRC |", "|---|---:|"]
    p2 = next(r["train_cv_auprc"] for r in rows if r["arm"] == "P2_static_plus_15")
    for r in rows:
        if not str(r["arm"]).startswith("LOO_"):
            continue
        lines.append(f"| {r['note']} | {r['train_cv_auprc'] - p2:+.4f} |")
    if hold_rows:
        lines += ["", "## Holdout（只报告）", "", "| 臂 | holdout AUPRC |", "|---|---:|"]
        for r in hold_rows:
            lines.append(f"| {r['arm']} | {r['holdout_auprc']:.4f} |")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log("[29] DONE")


if __name__ == "__main__":
    main()
