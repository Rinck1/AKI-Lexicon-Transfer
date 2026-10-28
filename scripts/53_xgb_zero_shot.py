#!/usr/bin/env python3
"""Round-2 E4: XGBoost gradient-boosting baseline — MIMIC-train fit, eICU zero-shot.

Question (reviewer): on the SAME feature caliber as the linear numeric arm
(static 11 + 11 lab vars x 4 stages x {last,min,max,delta} = 176 numeric dims),
how much does nonlinearity + tuning add? Fit on MIMIC train ONLY
(SUBJECT-grouped 5-fold CV, within-fold early stopping on AUPRC), then deploy
zero-refit to eICU.

Fairness (docs/29): features restricted to static + lab-stage summaries within
the main model's allowed scope; no vitals morphology / motif concepts.
MIMIC val is used ONCE descriptively after the final model is frozen;
holdout is never touched. eICU is never used for any selection.

Outputs (round-2, post-hoc exploratory):
  data/external/eicu/round2_xgb/
      xgb_tuning_results.json          fold AUPRC per config, best config
      features_mimic_raw.parquet       raw 176+11 (train/val rows, float32)
      features_eicu_raw.parquet        raw 176+11 (float32)
      pred_eicu.parquet                y + predictions per arm
      pred_mimic_val_descriptive.parquet
      calibration_eicu_10bins.csv      decile bins, XGB vs linear
      importance_gain_top20.csv        gain-based top-20 (final XGB)
      importance_gain_by_stat.csv      gain share by stat/var/stage
      metrics_summary.json             all headline numbers
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
from sklearn.model_selection import StratifiedGroupKFold

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.cbm import (  # noqa: E402
    build_static_matrix,
    standardize_train_only,
)
from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.eicu_io import load_eicu  # noqa: E402
from scripts.utils.graph_constants import GRAPH_SPARSE_VARS, STAGE_NAMES  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS, STAGE_BOUNDS  # noqa: E402

OUT_DIR = PROJ / "data/external/eicu/round2_xgb"
REPORT_DIR = PROJ / "outputs/reports"
SEED = 42
ROUND = "round-2"

VARS = tuple(GRAPH_SPARSE_VARS)          # 11 lab/GCS vars (same as sparse-308 caliber)
STAGES = [(name, int(lo), int(hi)) for name, lo, hi in STAGE_BOUNDS]  # B1..B4
NUM_STATS = ("last", "min", "max", "delta")  # 176 = 11 x 4 x 4
PATT_STATS = ("is_measured", "measured_fraction", "hours_since")  # +132 (additional arm)

# numeric feature names: var-major, stage-minor, stat-minor
NUM_NAMES = [f"{v}|{st}|{s}" for v in VARS for st, _, _ in STAGES for s in NUM_STATS]
PATT_NAMES = [f"{v}|{st}|{s}" for v in VARS for st, _, _ in STAGES for s in PATT_STATS]
STATIC_NAMES = list(SHAPELET_STATIC_COLS)
assert len(NUM_NAMES) == 176 and len(VARS) == 11 and len(STAGES) == 4

TUNING_GRID = {"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1]}
N_ESTIMATORS_MAX = 3000
EARLY_STOPPING_ROUNDS = 50
XGB_FIXED = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=1.0,
    n_jobs=-1,
    random_state=SEED,
)
N_BOOTSTRAP = 1000


# ---------------------------------------------------------------------------
# Feature construction (vectorized, both sides from onset-relative hourly grids;
# adapted from scripts/utils/graph_sparse.py::raw_sparse_vector_for_stay,
# restricted to the numeric stats; missing stays NaN for XGBoost native handling)
# ---------------------------------------------------------------------------
def numeric_pattern_matrix(
    grid: pd.DataFrame,
    stay_ids: np.ndarray,
    t_hours: int = 48,
    want_pattern: bool = False,
) -> np.ndarray:
    """(n, 176[, +132]) raw features. Layout per var x stage: last,min,max,delta
    (delta = last - previous-stage last, carried forward; stage-1 delta = 0 if
    measured else NaN). Pattern stats (optional): is_measured, measured_fraction,
    hours_since (normalized by 48)."""
    n = len(stay_ids)
    n_stat = 4 + (3 if want_pattern else 0)
    out = np.full((n, len(VARS) * len(STAGES) * n_stat), np.nan, dtype=np.float64)
    sid_pos = pd.Index(stay_ids)
    keep_cols = ["ICUSTAY_ID", "HOUR"] + list(VARS) + [f"{v}_is_measured" for v in VARS]
    g = grid.loc[grid["ICUSTAY_ID"].isin(set(stay_ids.tolist())), keep_cols].copy()
    g = g.sort_values(["ICUSTAY_ID", "HOUR"], kind="stable")
    row_pos = sid_pos.get_indexer(g["ICUSTAY_ID"].to_numpy())
    hour_pos = g["HOUR"].to_numpy().astype(int)

    for v_i, var in enumerate(VARS):
        V = np.full((n, t_hours), np.nan)
        M = np.zeros((n, t_hours), dtype=bool)
        vals = g[var].to_numpy(float)
        meas = g[f"{var}_is_measured"].to_numpy(bool)
        ok = meas & np.isfinite(vals)
        V[row_pos[ok], hour_pos[ok]] = vals[ok]
        M[row_pos[ok], hour_pos[ok]] = True
        prev_last = np.nan
        lasth = np.full(n, -1, dtype=int)
        base = v_i * len(STAGES) * n_stat
        for s_i, (_name, lo, hi) in enumerate(STAGES):
            segM = M[:, lo:hi]
            segV = np.where(segM, V[:, lo:hi], np.nan)
            n_meas = segM.sum(1)
            is_meas = n_meas > 0
            rev = segM[:, ::-1]
            last_off = np.where(is_meas, segM.shape[1] - 1 - np.argmax(rev, axis=1), 0)
            last_v = np.where(is_meas, segV[np.arange(n), last_off], np.nan)
            with np.errstate(all="ignore"):
                min_v = np.nanmin(segV, axis=1)
                max_v = np.nanmax(segV, axis=1)
            min_v = np.where(is_meas, min_v, np.nan)
            max_v = np.where(is_meas, max_v, np.nan)
            if s_i == 0:
                delta = np.where(is_meas, 0.0, np.nan)
            else:
                delta = last_v - prev_last
                delta = np.where(np.isnan(delta), np.nan, delta)
            col = base + s_i * n_stat
            out[:, col + 0] = last_v
            out[:, col + 1] = min_v
            out[:, col + 2] = max_v
            out[:, col + 3] = delta
            if want_pattern:
                # hours since last measurement at stage end (history-carrying)
                lh_new = np.where(is_meas, lo + last_off, lasth)
                hrs = np.empty(n)
                neg = lh_new < 0
                if neg.any():
                    hist = M[:, :hi]
                    has_hist = hist.any(1)
                    last_hist = np.where(has_hist, t_hours - 1 - np.argmax(hist[:, ::-1], axis=1), 0)
                    hrs[neg] = np.where(has_hist, (hi - 1) - last_hist, float(t_hours))[neg]
                hrs[~neg] = (hi - 1) - lh_new[~neg]
                hrs = np.clip(hrs, 0.0, float(t_hours)) / float(t_hours)
                out[:, col + 4] = is_meas.astype(float)
                out[:, col + 5] = n_meas / float(hi - lo)
                out[:, col + 6] = hrs
                lasth = lh_new
            prev_last = np.where(np.isnan(last_v), prev_last, last_v)
    return out.astype(np.float32)


def zscore_impute_train(raw: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, ...]:
    """Train-only median fill + z-score; returns (train_X, mu, sd, median)."""
    med = np.nanmedian(ref, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def _t(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64).copy()
        nan = np.isnan(x)
        x[nan] = np.broadcast_to(med, x.shape)[nan]
        return x

    filled = _t(ref)
    mu, sd = filled.mean(0), filled.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((_t(raw) - mu) / sd).astype(np.float32), mu, sd, med


def apply_zscore(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, med: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64).copy()
    nan = np.isnan(x)
    x[nan] = np.broadcast_to(med, x.shape)[nan]
    return ((x - mu) / sd).astype(np.float32)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def boot_ci_auprc(y: np.ndarray, p1: np.ndarray, p0: np.ndarray, rng: np.random.Generator,
                  n_boot: int = N_BOOTSTRAP) -> dict:
    """Paired percentile bootstrap (eICU-side resampling) for AUPRC/AUROC diffs."""
    n = len(y)
    d_ap, d_ar, ap1 = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() in (0, n):
            continue
        a1, a0 = average_precision_score(yb, p1[idx]), average_precision_score(yb, p0[idx])
        d_ap.append(a1 - a0)
        d_ar.append(roc_auc_score(yb, p1[idx]) - roc_auc_score(yb, p0[idx]))
        ap1.append(a1)

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    return {
        "auprc_diff_ci95": ci(d_ap),
        "auroc_diff_ci95": ci(d_ar),
        "auprc_xgb_ci95": ci(ap1),
    }


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(p, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"p": p, "y": y, "bin": q})
    g = df.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean"))
    g = g.reset_index()
    g["bin_lo"] = [iv.left for iv in g["bin"]]
    g["bin_hi"] = [iv.right for iv in g["bin"]]
    return g[["bin_lo", "bin_hi", "n", "mean_pred", "obs_rate"]]


def cal_slope_intercept(y: np.ndarray, p: np.ndarray) -> dict:
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return {"slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---------------- data ----------------
    grid_m = pd.read_parquet(Path(DATA_PROCESSED) / "hourly_grid_48h.parquet")
    from scripts.utils.shapelet_prep import load_shapelet_tables

    _g, mimic_stays = load_shapelet_tables(DATA_PROCESSED)
    stays_m = mimic_stays.set_index("ICUSTAY_ID")
    train_ids = np.array(sorted(int(x) for x in mimic_stays.loc[mimic_stays["split"] == "train", "ICUSTAY_ID"]))
    val_ids = np.array(sorted(int(x) for x in mimic_stays.loc[mimic_stays["split"] == "val", "ICUSTAY_ID"]))
    y_tr = stays_m.loc[list(train_ids), "HOSPITAL_MORTALITY"].to_numpy(int)
    y_va = stays_m.loc[list(val_ids), "HOSPITAL_MORTALITY"].to_numpy(int)
    subj_tr = stays_m.loc[list(train_ids), "SUBJECT_ID"].to_numpy()

    grid_e = pd.read_parquet(PROJ / "data/external/eicu/hourly_grid_48h.parquet")
    stays_e = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    e_ids = np.array(sorted(int(x) for x in stays_e.index))
    y_e = stays_e.loc[list(e_ids), "HOSPITAL_MORTALITY"].to_numpy(int)
    assert np.isin(e_ids, grid_e["ICUSTAY_ID"].unique()).all()
    assert np.isin(train_ids, grid_m["ICUSTAY_ID"].unique()).all()
    n_subj_multi = int(pd.Series(subj_tr).duplicated().sum())
    print(f"[data] MIMIC train {len(train_ids)} (subj-dup stays {n_subj_multi}), "
          f"val {len(val_ids)} (descriptive only); eICU {len(e_ids)} | {time.time()-t0:.0f}s", flush=True)

    # ---------------- features (same builder both sides) ----------------
    num_m = numeric_pattern_matrix(grid_m, train_ids)
    num_va = numeric_pattern_matrix(grid_m, np.asarray(val_ids))
    num_e = numeric_pattern_matrix(grid_e, e_ids)
    pat_m = numeric_pattern_matrix(grid_m, train_ids, want_pattern=True)[:, 176:]
    pat_va = numeric_pattern_matrix(grid_m, np.asarray(val_ids), want_pattern=True)[:, 176:]
    pat_e = numeric_pattern_matrix(grid_e, e_ids, want_pattern=True)[:, 176:]

    stat_m = build_static_matrix(stays_m, list(train_ids), SHAPELET_STATIC_COLS)
    stat_va = build_static_matrix(stays_m, list(val_ids), SHAPELET_STATIC_COLS)
    stat_e = build_static_matrix(stays_e, list(e_ids), SHAPELET_STATIC_COLS)
    assert np.isfinite(stat_m).all() and np.isfinite(stat_e).all()

    # XGBoost: raw features, NaN kept (native missing handling); no scaling
    X_main = np.column_stack([stat_m, num_m])
    X_main_va = np.column_stack([stat_va, num_va])
    X_main_e = np.column_stack([stat_e, num_e])
    X_pat = np.column_stack([X_main, pat_m])
    X_pat_va = np.column_stack([X_main_va, pat_va])
    X_pat_e = np.column_stack([X_main_e, pat_e])
    feat_names = STATIC_NAMES + NUM_NAMES
    feat_names_pat = feat_names + PATT_NAMES

    # linear arms: train-only median-fill + z-score
    l_main, mu, sd, med = zscore_impute_train(np.column_stack([stat_m, num_m]),
                                              np.column_stack([stat_m, num_m]))
    l_main_va = apply_zscore(X_main_va, mu, sd, med)
    l_main_e = apply_zscore(X_main_e, mu, sd, med)
    l_stat, mu_s, sd_s, med_s = zscore_impute_train(stat_m, stat_m)
    l_stat_va = apply_zscore(stat_va, mu_s, sd_s, med_s)
    l_stat_e = apply_zscore(stat_e, mu_s, sd_s, med_s)

    # save raw feature tables (reproducibility)
    pd.DataFrame(np.column_stack([train_ids, y_tr, X_main]),
                 columns=["ICUSTAY_ID", "HOSPITAL_MORTALITY"] + feat_names).astype(
        {"ICUSTAY_ID": "int64", "HOSPITAL_MORTALITY": "int64"}).to_parquet(OUT_DIR / "features_mimic_raw.parquet", index=False)
    pd.DataFrame(np.column_stack([e_ids, y_e, X_main_e]),
                 columns=["ICUSTAY_ID", "HOSPITAL_MORTALITY"] + feat_names).astype(
        {"ICUSTAY_ID": "int64", "HOSPITAL_MORTALITY": "int64"}).to_parquet(OUT_DIR / "features_eicu_raw.parquet", index=False)
    print(f"[features] 176 numeric + 11 static built; NaN share "
          f"train {np.isnan(num_m).mean():.3f}, eICU {np.isnan(num_e).mean():.3f} | {time.time()-t0:.0f}s", flush=True)

    # ---------------- (2) tuning: SUBJECT-grouped 5-fold CV, early stop AUPRC ----------------
    from xgboost import XGBClassifier

    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(gkf.split(X_main, y_tr, groups=subj_tr))
    tuning = {"round": ROUND, "post_hoc": True, "n_estimators_max": N_ESTIMATORS_MAX,
              "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "fixed_params": XGB_FIXED,
              "cv": "StratifiedGroupKFold(k=5, groups=SUBJECT_ID, seed=42)",
              "selection_metric": "mean fold validation AUPRC (early-stopped)",
              "configs": []}
    best = None
    for md in TUNING_GRID["max_depth"]:
        for lr_ in TUNING_GRID["learning_rate"]:
            fold_ap, fold_bi = [], []
            for tr_i, va_i in folds:
                clf = XGBClassifier(max_depth=md, learning_rate=lr_, n_estimators=N_ESTIMATORS_MAX,
                                    early_stopping_rounds=EARLY_STOPPING_ROUNDS, **XGB_FIXED)
                clf.fit(X_main[tr_i], y_tr[tr_i], eval_set=[(X_main[va_i], y_tr[va_i])], verbose=False)
                p = clf.predict_proba(X_main[va_i])[:, 1]
                fold_ap.append(float(average_precision_score(y_tr[va_i], p)))
                fold_bi.append(int(clf.best_iteration) + 1)
            mean_ap, mean_bi = float(np.mean(fold_ap)), float(np.mean(fold_bi))
            tuning["configs"].append({"max_depth": md, "learning_rate": lr_,
                                      "fold_auprc": fold_ap, "mean_auprc": mean_ap,
                                      "fold_best_iterations": fold_bi, "mean_best_iter": mean_bi})
            print(f"[tune] depth={md} lr={lr_:<4} fold AUPRC {np.round(fold_ap,4)} mean {mean_ap:.4f} "
                  f"| best_iter {fold_bi} (mean {mean_bi:.0f}) | {time.time()-t0:.0f}s", flush=True)
            key = (mean_ap, -md)  # tie-break: shallower tree
            if best is None or key > best[0]:
                best = (key, md, lr_, mean_bi)
    _key, md_best, lr_best, ne_final = best
    ne_final = int(np.ceil(ne_final))
    tuning["best"] = {"max_depth": md_best, "learning_rate": lr_best, "n_estimators_final": ne_final}
    (OUT_DIR / "xgb_tuning_results.json").write_text(
        json.dumps(tuning, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[tune] BEST depth={md_best} lr={lr_best} n_estimators={ne_final} "
          f"(mean fold AUPRC {_key[0]:.4f}) | {time.time()-t0:.0f}s", flush=True)

    # ---------------- (3) final models (train-fit only) ----------------
    results = {"round": ROUND, "post_hoc": True,
               "tuning_best": tuning["best"],
               "fairness": ("features = 11 static + 11 lab vars x 4 stages x {last,min,max,delta}; "
                            "no vitals morphology; tuning only inside MIMIC train "
                            "(SUBJECT-grouped CV); MIMIC val descriptive once after freeze; "
                            "holdout untouched; eICU zero-refit."),
               "models": {}}

    # XGB main (static + numeric176)
    xgb = XGBClassifier(max_depth=md_best, learning_rate=lr_best, n_estimators=ne_final, **XGB_FIXED)
    xgb.fit(pd.DataFrame(X_main, columns=feat_names), y_tr)
    p_xgb_va = xgb.predict_proba(pd.DataFrame(X_main_va, columns=feat_names))[:, 1]
    p_xgb_e = xgb.predict_proba(pd.DataFrame(X_main_e, columns=feat_names))[:, 1]
    results["models"]["xgb_static_num176"] = {
        "mimic_val_descriptive": metrics(y_va, p_xgb_va),
        "eicu_zeroshot": metrics(y_e, p_xgb_e)}

    # additional labeled arm: + pattern132 (measurement-pattern stats)
    xgb_p = XGBClassifier(max_depth=md_best, learning_rate=lr_best, n_estimators=ne_final, **XGB_FIXED)
    xgb_p.fit(pd.DataFrame(X_pat, columns=feat_names_pat), y_tr)
    p_xgbp_e = xgb_p.predict_proba(pd.DataFrame(X_pat_e, columns=feat_names_pat))[:, 1]
    p_xgbp_va = xgb_p.predict_proba(pd.DataFrame(X_pat_va, columns=feat_names_pat))[:, 1]
    results["models"]["xgb_static_num176_pattern132_additional_arm"] = {
        "mimic_val_descriptive": metrics(y_va, p_xgbp_va),
        "eicu_zeroshot": metrics(y_e, p_xgbp_e)}

    # linear reproduction of the numeric arm (caliber check vs 0.415)
    lin = LogisticRegression(max_iter=1000, solver="lbfgs").fit(l_main, y_tr)
    p_lin_va = lin.predict_proba(l_main_va)[:, 1]
    p_lin_e = lin.predict_proba(l_main_e)[:, 1]
    results["models"]["logistic_static_num176_reproduction"] = {
        "mimic_val_descriptive": metrics(y_va, p_lin_va),
        "eicu_zeroshot": metrics(y_e, p_lin_e),
        "reference_round2_numeric_linear": {"auprc": 0.415, "auroc": 0.753, "brier": 0.113}}

    lin_s = LogisticRegression(max_iter=1000, solver="lbfgs").fit(l_stat, y_tr)
    p_lins_e = lin_s.predict_proba(l_stat_e)[:, 1]
    results["models"]["logistic_static_reference"] = {
        "eicu_zeroshot": metrics(y_e, p_lins_e),
        "reference_round1": {"auprc": 0.190}}

    # ---------------- bootstrap contrast on eICU ----------------
    boot = boot_ci_auprc(y_e, p_xgb_e, p_lin_e, rng)
    m_x, m_l = metrics(y_e, p_xgb_e), metrics(y_e, p_lin_e)
    results["bootstrap_eicu_xgb_minus_linear"] = {
        **boot,
        "auprc_diff_point": m_x["auprc"] - m_l["auprc"],
        "auroc_diff_point": m_x["auroc"] - m_l["auroc"],
        "brier_diff_point": m_x["brier"] - m_l["brier"]}

    # ---------------- predictions + calibration ----------------
    pd.DataFrame({"ICUSTAY_ID": e_ids, "HOSPITAL_MORTALITY": y_e,
                  "p__xgb_num176": p_xgb_e, "p__xgb_num176_pattern132": p_xgbp_e,
                  "p__logistic_num176": p_lin_e, "p__logistic_static": p_lins_e}
                 ).to_parquet(OUT_DIR / "pred_eicu.parquet", index=False)
    pd.DataFrame({"ICUSTAY_ID": val_ids, "HOSPITAL_MORTALITY": y_va,
                  "p__xgb_num176": p_xgb_va, "p__xgb_num176_pattern132": p_xgbp_va,
                  "p__logistic_num176": p_lin_va}
                 ).to_parquet(OUT_DIR / "pred_mimic_val_descriptive.parquet", index=False)

    cal_rows = []
    for arm, p in [("xgb_num176", p_xgb_e), ("xgb_num176_pattern132", p_xgbp_e),
                   ("logistic_num176", p_lin_e), ("logistic_static", p_lins_e)]:
        b = calibration_bins(y_e, p)
        b.insert(0, "arm", arm)
        cal_rows.append(b)
    pd.concat(cal_rows, ignore_index=True).to_csv(
        OUT_DIR / "calibration_eicu_10bins.csv", index=False, encoding="utf-8-sig")
    results["calibration_slope_intercept_eicu"] = {
        arm: cal_slope_intercept(y_e, p) for arm, p in
        [("xgb_num176", p_xgb_e), ("logistic_num176", p_lin_e), ("logistic_static", p_lins_e)]}

    # ---------------- (4) importance (gain) ----------------
    booster = xgb.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = pd.DataFrame({"feature": feat_names,
                        "gain": [float(gain.get(f, 0.0)) for f in feat_names]})
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    imp.sort_values("gain", ascending=False).head(20).to_csv(
        OUT_DIR / "importance_gain_top20.csv", index=False, encoding="utf-8-sig")
    is_num = imp["feature"].str.contains("|", regex=False)
    imp["var"] = imp["feature"].str.split("|", expand=True)[0]
    imp["stat"] = np.where(is_num, imp["feature"].str.split("|", expand=True)[2], "static")
    by_stat = imp.groupby("stat")["gain_share"].sum().sort_values(ascending=False)
    by_var = imp.groupby("var")["gain_share"].sum().sort_values(ascending=False)
    imp.groupby(["var", "stat"])["gain_share"].sum().reset_index().sort_values(
        "gain_share", ascending=False).to_csv(OUT_DIR / "importance_gain_by_stat.csv",
                                              index=False, encoding="utf-8-sig")
    results["importance_gain_share"] = {
        "by_stat": {k: float(v) for k, v in by_stat.items()},
        "by_var_top8": {k: float(v) for k, v in by_var.head(8).items()}}

    # ---------------- summary ----------------
    (OUT_DIR / "metrics_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== eICU zero-shot (round-2) ===")
    for k, v in results["models"].items():
        ez = v["eicu_zeroshot"]
        print(f"{k:<44} AUPRC {ez['auprc']:.3f} AUROC {ez['auroc']:.3f} Brier {ez['brier']:.3f}")
    print(f"XGB - linear: dAUPRC {results['bootstrap_eicu_xgb_minus_linear']['auprc_diff_point']:+.3f} "
          f"CI95 [{boot['auprc_diff_ci95'][0]:+.3f},{boot['auprc_diff_ci95'][1]:+.3f}] | "
          f"gain-share by stat: {dict(by_stat.round(3))}")
    print(f"→ {OUT_DIR} | [done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
