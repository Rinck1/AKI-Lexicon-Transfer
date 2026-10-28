#!/usr/bin/env python3
"""Round-3 correction to scripts/49 (2026-10-11): eICU codes "APACHE not
computed" as -1 in apachePatientResult (apachescore, predictedhospitalmortality,
predictedicumortality). scripts/49 kept those -1 rows as real predictions
(score_range [-1, 197]; first calibration decile mean prediction -0.185).

This script reproduces the original numbers (with -1 kept) as a check, then
recomputes with -1 treated as missing. The original output
(round2_apache/apache_metrics.json) is left untouched.
Output: data/external/eicu/round2_apache/apache_metrics_corrected.json,
        calibration_phm_corrected.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
from scripts.utils.eicu_io import load_eicu, resolve_eicu_dir  # noqa: E402

OUT = PROJ / "data/external/eicu/round2_apache"


def metrics(p, y):
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    r = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lg.reshape(-1, 1), y)
    return {"n": int(len(y)), "deaths": int(y.sum()), "mortality": float(y.mean()),
            "auprc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p)),
            "brier": float(np.mean((p - y) ** 2)), "recal_slope": float(r.coef_[0][0]),
            "recal_intercept": float(r.intercept_[0])}


def main():
    ap = pd.read_csv(resolve_eicu_dir() / "apachePatientResult.csv.gz")
    st = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    ap = ap.drop_duplicates(subset="patientunitstayid", keep="first")  # identical to scripts/49
    m = pd.DataFrame({"ICUSTAY_ID": st.index.to_numpy(), "y": st["HOSPITAL_MORTALITY"].to_numpy(float)}).merge(
        ap, left_on="ICUSTAY_ID", right_on="patientunitstayid", how="left")
    res = {"round": "round-3 correction of round-2 A2", "n_cohort": int(len(m))}
    for col in ["predictedhospitalmortality", "predictedicumortality"]:
        p = m[col].to_numpy(float)
        y = m["y"].to_numpy(float)
        ok = np.isfinite(p)
        neg = ok & (p < 0)
        res[col] = {"n_nonnull": int(ok.sum()), "n_minus1": int(neg.sum()),
                    "minus1_group_mortality": float(y[neg].mean()) if neg.any() else None,
                    "original_with_minus1": metrics(p[ok], y[ok]),
                    "corrected_excluding_minus1": metrics(p[ok & ~neg], y[ok & ~neg])}
    s = m["apachescore"].to_numpy(float)
    y = m["y"].to_numpy(float)
    ok = np.isfinite(s)
    good = ok & (s >= 0)
    res["apachescore"] = {"n_minus1": int((ok & (s < 0)).sum()),
                          "original": {"auroc": float(roc_auc_score(y[ok], s[ok])), "auprc": float(average_precision_score(y[ok], s[ok]))},
                          "corrected": {"n": int(good.sum()), "auroc": float(roc_auc_score(y[good], s[good])),
                                        "auprc": float(average_precision_score(y[good], s[good]))}}
    # same-subset comparators: static and +84a on the APACHE-available subset are reported by
    # scripts/59 from stored predictions; here only APACHE itself.
    p = m["predictedhospitalmortality"].to_numpy(float)
    keep = np.isfinite(p) & (p >= 0)
    res["corrected_subset_icustay_ids_n"] = int(keep.sum())
    pd.DataFrame({"ICUSTAY_ID": m.loc[keep, "ICUSTAY_ID"], "y": y[keep], "p_apache": p[keep]}).to_csv(
        OUT / "apache_pred_corrected.csv", index=False)
    b = pd.qcut(p[keep], 10, duplicates="drop")
    pd.DataFrame({"p": p[keep], "y": y[keep], "bin": b}).groupby("bin", observed=True).agg(
        pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).to_csv(OUT / "calibration_phm_corrected.csv", index=False)
    # same-subset comparison with the round-3 arms (stored predictions from scripts/56)
    pred_path = PROJ / "data/external/eicu/round3_complementarity/pred_eicu.csv"
    if pred_path.is_file():
        pr = pd.read_csv(pred_path).merge(
            pd.DataFrame({"ICUSTAY_ID": m.loc[keep, "ICUSTAY_ID"], "p__APACHE_IV_corrected": p[keep]}), on="ICUSTAY_ID")
        yy = pr["HOSPITAL_MORTALITY"].to_numpy(float)
        res["same_subset_comparison"] = {"n": int(len(pr)), "mortality": float(yy.mean()), "arms": {
            c.replace("p__", ""): {"auprc": float(average_precision_score(yy, pr[c])), "auroc": float(roc_auc_score(yy, pr[c])),
                                   "brier": float(np.mean((pr[c] - yy) ** 2))}
            for c in pr.columns if c.startswith("p__")}}
    (OUT / "apache_metrics_corrected.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
