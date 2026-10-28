#!/usr/bin/env python3
"""PRED post-hoc explore: sparse-308 validity ablations + calibration/DCA on eICU.

Questions:
 (a) Is the sparse-308 zero-shot gain driven by absolute lab/GCS values or by
     measurement-pattern features (is_measured / measured_fraction /
     hours_since)? -> split blocks.
 (b) Does gcs_total alone drive it? -> drop-gcs block.
 (c) Calibration / DCA of static+sparse308 vs static on eICU.

Fit on MIMIC train only; eICU zero-shot; MIMIC val descriptive only. post-hoc.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.cbm import build_static_matrix, standardize_train_only  # noqa: E402
from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.eicu_io import load_eicu  # noqa: E402
from scripts.utils.graph_constants import GRAPH_SPARSE_VARS, STAGE_NAMES  # noqa: E402
from scripts.utils.graph_sparse import N_SPARSE_FEATURES, fit_sparse_scaler  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts._explore_PRED_b_arm_sparse import (  # noqa: E402
    transform_sparse_2d,
    vec_sparse_matrix,
)
from scripts._explore_PRED_calibration_dca_nri import cal_stats, net_benefit  # noqa: E402

OUT_DIR = PROJ / "data/external/eicu/_explore_PRED"
STAGES = [s for s in STAGE_NAMES]


def block_masks():
    """Column masks over the 308 layout (var-major, stage-minor, 7 stats)."""
    names = []
    for v in GRAPH_SPARSE_VARS:
        for st in STAGES:
            for suf in ("last", "min", "max", "delta_from_previous_stage_last",
                        "is_measured", "measured_fraction",
                        "hours_since_last_measurement_at_stage_end"):
                names.append((v, suf))
    names = np.array(names, dtype=object)
    vals = np.array([s in ("last", "min", "max", "delta_from_previous_stage_last") for _, s in names])
    patt = ~vals
    gcs = np.array([v == "gcs_total" for v, _ in names])
    return {"vals_only": vals, "pattern_only": patt, "no_gcs": ~gcs}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_m = pd.read_parquet(Path(DATA_PROCESSED) / "hourly_grid_48h.parquet")

    from scripts.utils.shapelet_prep import load_shapelet_tables
    _grid, mimic_stays = load_shapelet_tables(DATA_PROCESSED)
    stays_m = mimic_stays.set_index("ICUSTAY_ID")
    train_ids = np.array(sorted(int(x) for x in mimic_stays.loc[mimic_stays["split"] == "train", "ICUSTAY_ID"]))
    val_ids = np.array(sorted(int(x) for x in mimic_stays.loc[mimic_stays["split"] == "val", "ICUSTAY_ID"]))

    y_tr = stays_m.loc[list(train_ids), "HOSPITAL_MORTALITY"].to_numpy(float)
    static_tr, mu, sd = standardize_train_only(build_static_matrix(stays_m, list(train_ids), SHAPELET_STATIC_COLS))
    y_va = stays_m.loc[list(val_ids), "HOSPITAL_MORTALITY"].to_numpy(float)
    static_va = (build_static_matrix(stays_m, list(val_ids), SHAPELET_STATIC_COLS) - mu) / sd

    sp_tr_raw = vec_sparse_matrix(grid_m, train_ids)
    scaler = fit_sparse_scaler(sp_tr_raw.astype(float))
    sp_tr = transform_sparse_2d(sp_tr_raw, scaler).astype(float)
    sp_va = transform_sparse_2d(vec_sparse_matrix(grid_m, val_ids), scaler).astype(float)

    grid_e = pd.read_parquet(Path(DATA_PROCESSED).parent / "external/eicu/hourly_grid_48h.parquet")
    stays_e = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    e_ids = np.array(sorted(int(x) for x in stays_e.index))
    y_e = stays_e.loc[list(e_ids), "HOSPITAL_MORTALITY"].to_numpy(float)
    static_e = (build_static_matrix(stays_e, list(e_ids), SHAPELET_STATIC_COLS) - mu) / sd
    sp_e = transform_sparse_2d(vec_sparse_matrix(grid_e, e_ids), scaler).astype(float)
    print(f"[data] {time.time()-t0:.0f}s", flush=True)

    masks = block_masks()
    arms = {"static": None, "static+sparse308": np.arange(N_SPARSE_FEATURES)}
    for k, m in masks.items():
        arms[f"static+sparse_{k}"] = np.where(m)[0]

    results = {"post_hoc": True, "arms": {}, "block_sizes": {k: len(v) for k, v in masks.items()}}
    pred = pd.DataFrame({"ICUSTAY_ID": list(e_ids), "y": y_e})
    for name, cols in arms.items():
        X = static_tr if cols is None else np.column_stack([static_tr, sp_tr[:, cols]])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs").fit(X, y_tr)
        row = {}
        ps = {}
        for set_name, y, s0, sp in [
            ("mimic_val(desc)", y_va, static_va, sp_va),
            ("eicu_zeroshot", y_e, static_e, sp_e),
        ]:
            Xs = s0 if cols is None else np.column_stack([s0, sp[:, cols]])
            p = clf.predict_proba(Xs)[:, 1]
            ps[set_name] = p
            row[set_name] = {"auprc": float(average_precision_score(y, p)),
                             "auroc": float(roc_auc_score(y, p)),
                             "brier": float(brier_score_loss(y, p))}
        results["arms"][name] = row
        pred[f"p__{name.replace('+', '_plus_')}"] = ps["eicu_zeroshot"]
        mv, ez = row["mimic_val(desc)"], row["eicu_zeroshot"]
        print(f"[{name:<26}] val {mv['auprc']:.3f} | eICU AUPRC {ez['auprc']:.3f} AUROC {ez['auroc']:.3f} Brier {ez['brier']:.3f}", flush=True)

    # calibration + DCA on eICU for static and static+sparse308
    cal = {}
    for name in ["static", "static+sparse308"]:
        col = f"p__{name.replace('+', '_plus_')}"
        p = pred[col].to_numpy(float)
        cal[name] = cal_stats(y_e, p)
    dca_rows = []
    for t in np.round(np.arange(0.05, 0.4001, 0.01), 4):
        row = {"threshold": float(t)}
        for name in ["static", "static+sparse308"]:
            row[name] = net_benefit(y_e, pred[f"p__{name.replace('+', '_plus_')}"].to_numpy(float), t)
        dca_rows.append(row)
    dca = pd.DataFrame(dca_rows)
    dca.to_csv(OUT_DIR / "dca_eicu_sparse.csv", index=False, encoding="utf-8-sig")
    results["calibration_eicu"] = cal
    results["dca_top"] = dca.iloc[::5].round(4).to_dict("records")
    pred.to_parquet(OUT_DIR / "pred_eicu_sparse_ablation.parquet", index=False)

    print("\n=== calibration (eICU) ===")
    for name, st in cal.items():
        print(f"{name:<18} slope {st['slope']:.3f} intercept {st['intercept']:+.3f} Brier {st['brier']:.4f} -> reBrier {st['brier_recalibrated_in_eicu']:.4f}")
    print("\n=== DCA (step 0.05) ===")
    print(dca.iloc[::5].round(4).to_string(index=False))
    (OUT_DIR / "sparse_ablation_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ {OUT_DIR}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
