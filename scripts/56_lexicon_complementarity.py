#!/usr/bin/env python3
"""Round-3 (2026-10-11): does the 84-concept lexicon add information on top of the
strongest same-information models, zero-shot on eICU?

Question (reviewer): the lexicon arm (eICU AUPRC 0.281) is below APACHE IV (0.375)
and XGBoost (0.441). Is the lexicon merely a weaker predictor, or does it carry
information that continuous laboratory summaries do not?

Design (written into this docstring BEFORE the first run; not part of docs/31):
  - All fits on MIMIC train only (N=7,610); MIMIC val reported once, descriptively;
    holdout untouched; eICU zero-refit. Frozen assets untouched.
  - Linear family (same scaling as the existing arms):
        L0 static                         (reproduction check: eICU AUPRC 0.190)
        L1 static + 84a                   (reproduction check: 0.281)
        L2 static + num176                (reproduction check: 0.415)
        L3 static + num176 + 84a          NEW
  - XGBoost family, hyper-parameters FROZEN from scripts/53 (MIMIC-train CV:
    depth 3, lr 0.05, 147 trees):
        X0 static + num176                (reproduction check: 0.441)
        X1 static + num176 + 84a          NEW
    Sensitivity: X1 re-tuned with the identical scripts/53 grid and
    SUBJECT-grouped CV on MIMIC train (so X1 is not handicapped by X0's settings).
  - Primary contrasts: L3 - L2 and X1 - X0 on eICU, paired stay-level bootstrap
    (B=1000, seed 42) for dAUPRC and dAUROC.
  - Pre-stated reading: "complementary" if the X1 - X0 dAUPRC 95% CI excludes 0;
    "no added discrimination" otherwise (then the lexicon's role is explanation and
    monitoring, not ranking). Either result is reported.

Outputs (round-3, exploratory): data/external/eicu/round3_complementarity/
  complementarity_results.json, pred_eicu.csv
"""
from __future__ import annotations

import importlib.util
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

from scripts.utils.cbm import KIND_LAB, KIND_MOTIF, build_concept_matrix, build_static_matrix, standardize_train_only  # noqa: E402
from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.eicu_constants import MIMIC_PATHS_DIR  # noqa: E402
from scripts.utils.eicu_io import load_eicu  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402

_spec = importlib.util.spec_from_file_location("xgb53", PROJ / "scripts" / "53_xgb_zero_shot.py")
x53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x53)

OUT_DIR = PROJ / "data/external/eicu/round3_complementarity"
SEED = 42
N_BOOT = 1000
REPRO = {"L0": 0.190, "L1": 0.281, "L2": 0.415, "X0": 0.441}


def metrics(y, p):
    return {"n": int(len(y)), "prevalence": float(np.mean(y)),
            "auprc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), **x53.cal_slope_intercept(y, p)}


def paired_boot(y, p1, p0, rng, n_boot=N_BOOT):
    n = len(y)
    dap, dar = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yb = y[i]
        if yb.sum() in (0, n):
            continue
        dap.append(average_precision_score(yb, p1[i]) - average_precision_score(yb, p0[i]))
        dar.append(roc_auc_score(yb, p1[i]) - roc_auc_score(yb, p0[i]))
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return {"dauprc_point": float(average_precision_score(y, p1) - average_precision_score(y, p0)),
            "dauprc_ci95": ci(dap),
            "dauroc_point": float(roc_auc_score(y, p1) - roc_auc_score(y, p0)),
            "dauroc_ci95": ci(dar), "n_boot_used": len(dap)}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    grid_m, mimic_stays = load_shapelet_tables(DATA_PROCESSED)
    sm = mimic_stays.set_index("ICUSTAY_ID")
    tr = np.array(sorted(int(x) for x in mimic_stays.loc[mimic_stays["split"] == "train", "ICUSTAY_ID"]))
    va = np.array(sorted(int(x) for x in mimic_stays.loc[mimic_stays["split"] == "val", "ICUSTAY_ID"]))
    y_tr = sm.loc[list(tr), "HOSPITAL_MORTALITY"].to_numpy(int)
    y_va = sm.loc[list(va), "HOSPITAL_MORTALITY"].to_numpy(int)
    subj = sm.loc[list(tr), "SUBJECT_ID"].to_numpy()

    grid_e = pd.read_parquet(PROJ / "data/external/eicu/hourly_grid_48h.parquet")
    se = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    ee = np.array(sorted(int(x) for x in se.index))
    y_e = se.loc[list(ee), "HOSPITAL_MORTALITY"].to_numpy(int)

    # numeric 176 (identical builder to scripts/53)
    num_tr, num_va, num_e = (x53.numeric_pattern_matrix(grid_m, tr), x53.numeric_pattern_matrix(grid_m, va),
                             x53.numeric_pattern_matrix(grid_e, ee))
    st_tr = build_static_matrix(sm, list(tr), SHAPELET_STATIC_COLS)
    st_va = build_static_matrix(sm, list(va), SHAPELET_STATIC_COLS)
    st_e = build_static_matrix(se, list(ee), SHAPELET_STATIC_COLS)

    # cross-check against the saved scripts/53 feature tables
    saved = pd.read_parquet(PROJ / "data/external/eicu/round2_xgb/features_eicu_raw.parquet")
    saved = saved.set_index("ICUSTAY_ID").loc[list(ee)].iloc[:, 1:].to_numpy(np.float32)
    mine = np.column_stack([st_e, num_e]).astype(np.float32)
    nan_mismatch = int((np.isnan(saved) != np.isnan(mine)).sum())
    both = ~np.isnan(saved) & ~np.isnan(mine)
    feat_check = float(np.abs(saved[both] - mine[both]).max()) if nan_mismatch == 0 else float("inf")

    # 84a lexicon (identical to _explore_PRED_cbm_zeroshot)
    m_atoms = pd.read_parquet(MIMIC_PATHS_DIR / "motif_atoms.parquet")
    mt_tr = m_atoms[m_atoms["split"] == "train"]
    concepts = sorted(mt_tr["atom"].astype(str).unique())
    assert len(concepts) == 84
    kind_of = {c: KIND_MOTIF for c in concepts}
    a_tr = build_concept_matrix(mt_tr, list(tr), concepts, kind_of=kind_of, use_activation=True)
    a_va = build_concept_matrix(m_atoms[m_atoms["split"] == "val"], list(va), concepts, kind_of=kind_of, use_activation=True)
    a_e = build_concept_matrix(load_eicu("motif_atoms"), list(ee), concepts, kind_of=kind_of, use_activation=True)
    print(f"[data] train {len(tr)} val {len(va)} eICU {len(ee)}; feature check max|diff|={feat_check:.2e} "
          f"| {time.time()-t0:.0f}s", flush=True)

    # ---------------- linear family ----------------
    sts_tr, mu, sd = standardize_train_only(st_tr)
    sts_va, sts_e = (st_va - mu) / sd, (st_e - mu) / sd
    ln_tr, nmu, nsd, nmed = x53.zscore_impute_train(np.column_stack([st_tr, num_tr]), np.column_stack([st_tr, num_tr]))
    ln_va = x53.apply_zscore(np.column_stack([st_va, num_va]), nmu, nsd, nmed)
    ln_e = x53.apply_zscore(np.column_stack([st_e, num_e]), nmu, nsd, nmed)
    lin_arms = {
        "L0_static": (sts_tr, sts_va, sts_e, dict(max_iter=500)),
        "L1_static+84a": (np.column_stack([sts_tr, a_tr]), np.column_stack([sts_va, a_va]),
                          np.column_stack([sts_e, a_e]), dict(max_iter=500)),
        "L2_static+num176": (ln_tr, ln_va, ln_e, dict(max_iter=1000)),
        "L3_static+num176+84a": (np.column_stack([ln_tr, a_tr]), np.column_stack([ln_va, a_va]),
                                 np.column_stack([ln_e, a_e]), dict(max_iter=1000)),
    }
    pred_e, pred_va, res = {}, {}, {"round": "round-3", "post_hoc": True, "seed": SEED,
                                    "design": "see script docstring (written before first run)",
                                    "feature_check_vs_round2_max_abs_diff": feat_check, "models": {}}
    for k, (Xtr, Xva, Xe, kw) in lin_arms.items():
        clf = LogisticRegression(solver="lbfgs", **kw).fit(Xtr, y_tr)
        pred_va[k], pred_e[k] = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xe)[:, 1]
        print(f"[fit] {k:<24} eICU AUPRC {average_precision_score(y_e, pred_e[k]):.4f} | {time.time()-t0:.0f}s", flush=True)

    # ---------------- XGBoost family ----------------
    import xgboost
    from xgboost import XGBClassifier
    res["xgboost_version"] = xgboost.__version__
    res["xgboost_note"] = ("X0 reproduces scripts/53 only up to library-version differences; "
                           "X0 and X1 always share one version, so the X1 - X0 contrast is unaffected.")
    best = json.loads((PROJ / "data/external/eicu/round2_xgb/xgb_tuning_results.json").read_text(encoding="utf-8"))["best"]
    hp = dict(max_depth=best["max_depth"], learning_rate=best["learning_rate"], n_estimators=best["n_estimators_final"])
    Xg0 = (np.column_stack([st_tr, num_tr]), np.column_stack([st_va, num_va]), np.column_stack([st_e, num_e]))
    Xg1 = tuple(np.column_stack([a, b]) for a, b in zip(Xg0, (a_tr, a_va, a_e)))
    for k, (Xtr, Xva, Xe) in {"X0_xgb_static+num176": Xg0, "X1_xgb_static+num176+84a": Xg1}.items():
        clf = XGBClassifier(**hp, **x53.XGB_FIXED).fit(Xtr, y_tr)
        pred_va[k], pred_e[k] = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xe)[:, 1]
        print(f"[fit] {k:<24} eICU AUPRC {average_precision_score(y_e, pred_e[k]):.4f} | {time.time()-t0:.0f}s", flush=True)

    # sensitivity: re-tune X1 with the scripts/53 grid + SUBJECT-grouped CV on MIMIC train
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xg1[0], y_tr, groups=subj))
    tune = []
    for md in x53.TUNING_GRID["max_depth"]:
        for lr_ in x53.TUNING_GRID["learning_rate"]:
            aps, its = [], []
            for a_i, b_i in folds:
                c = XGBClassifier(max_depth=md, learning_rate=lr_, n_estimators=x53.N_ESTIMATORS_MAX,
                                  early_stopping_rounds=x53.EARLY_STOPPING_ROUNDS, **x53.XGB_FIXED)
                c.fit(Xg1[0][a_i], y_tr[a_i], eval_set=[(Xg1[0][b_i], y_tr[b_i])], verbose=False)
                aps.append(float(average_precision_score(y_tr[b_i], c.predict_proba(Xg1[0][b_i])[:, 1])))
                its.append(int(c.best_iteration) + 1)
            tune.append({"max_depth": md, "learning_rate": lr_, "mean_auprc": float(np.mean(aps)), "mean_best_iter": float(np.mean(its))})
            print(f"[tune X1] depth={md} lr={lr_} cv AUPRC {np.mean(aps):.4f} | {time.time()-t0:.0f}s", flush=True)
    bt = max(tune, key=lambda r: (r["mean_auprc"], -r["max_depth"]))
    c = XGBClassifier(max_depth=bt["max_depth"], learning_rate=bt["learning_rate"],
                      n_estimators=int(np.ceil(bt["mean_best_iter"])), **x53.XGB_FIXED).fit(Xg1[0], y_tr)
    k = "X1t_xgb_static+num176+84a_retuned"
    pred_va[k], pred_e[k] = c.predict_proba(Xg1[1])[:, 1], c.predict_proba(Xg1[2])[:, 1]
    res["x1_retune"] = {"grid": tune, "best": bt}

    for k in pred_e:
        res["models"][k] = {"eicu_zeroshot": metrics(y_e, pred_e[k]), "mimic_val_descriptive": metrics(y_va, pred_va[k])}
        res["models"][k]["retention_auprc"] = res["models"][k]["eicu_zeroshot"]["auprc"] / res["models"][k]["mimic_val_descriptive"]["auprc"]
    res["reproduction_checks"] = {
        key: {"expected": exp, "got": round(res["models"][name]["eicu_zeroshot"]["auprc"], 4),
              "ok": abs(res["models"][name]["eicu_zeroshot"]["auprc"] - exp) < 0.0015}
        for key, exp, name in [("L0", 0.190, "L0_static"), ("L1", 0.281, "L1_static+84a"),
                               ("L2", 0.415, "L2_static+num176"), ("X0", 0.441, "X0_xgb_static+num176")]}
    res["contrasts_eicu"] = {
        "L3_minus_L2": paired_boot(y_e, pred_e["L3_static+num176+84a"], pred_e["L2_static+num176"], rng),
        "X1_minus_X0": paired_boot(y_e, pred_e["X1_xgb_static+num176+84a"], pred_e["X0_xgb_static+num176"], rng),
        "X1t_minus_X0": paired_boot(y_e, pred_e["X1t_xgb_static+num176+84a_retuned"], pred_e["X0_xgb_static+num176"], rng),
        "L1_minus_L0": paired_boot(y_e, pred_e["L1_static+84a"], pred_e["L0_static"], rng),
    }
    res["contrasts_mimic_val_descriptive"] = {
        "L3_minus_L2": paired_boot(y_va, pred_va["L3_static+num176+84a"], pred_va["L2_static+num176"], rng),
        "X1_minus_X0": paired_boot(y_va, pred_va["X1_xgb_static+num176+84a"], pred_va["X0_xgb_static+num176"], rng),
    }
    ci = res["contrasts_eicu"]["X1_minus_X0"]["dauprc_ci95"]
    res["reading"] = "complementary" if (ci[0] > 0 or ci[1] < 0) and res["contrasts_eicu"]["X1_minus_X0"]["dauprc_point"] > 0 else (
        "lexicon degrades XGB externally" if ci[1] < 0 else "no added discrimination")
    (OUT_DIR / "complementarity_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"ICUSTAY_ID": ee, "HOSPITAL_MORTALITY": y_e, **{f"p__{k}": v for k, v in pred_e.items()}}).to_csv(
        OUT_DIR / "pred_eicu.csv", index=False)
    print(json.dumps({"repro": res["reproduction_checks"], "contrasts": res["contrasts_eicu"], "reading": res["reading"]},
                     indent=1, ensure_ascii=False))
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
