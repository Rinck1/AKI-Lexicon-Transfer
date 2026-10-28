#!/usr/bin/env python3
"""Round-2 (49): APACHE IV native-score baseline on the eICU external cohort.

docs/31 A2 / docs/29 E0. Zero training: native apachePatientResult fields are
used as-is (no refit). Same cohort & outcome as the path reproduction
(landmark-eligible HA-AKI stays, n≈21,657, HOSPITAL_MORTALITY).

Outputs: data/external/eicu/round2_apache/apache_metrics.json
         outputs/reports/49_APACHE对照.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.eicu_io import resolve_eicu_dir  # noqa: E402
from scripts.utils.eicu_io import load_eicu  # noqa: E402

OUT_DIR = PROJ / "data" / "external" / "eicu" / "round2_apache"
REPORT = PROJ / "outputs" / "reports" / "49_APACHE对照.md"


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eicu_dir = resolve_eicu_dir()
    log(f"[apache] raw dir = {eicu_dir}")

    ap = pd.read_csv(eicu_dir / "apachePatientResult.csv.gz")
    log(f"[apache] rows={len(ap):,} cols={list(ap.columns)}")

    stays = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    y = stays["HOSPITAL_MORTALITY"].to_numpy(dtype=float)
    ids = stays.index.to_numpy()
    log(f"[cohort] n={len(ids):,} mortality={y.mean():.4f}")

    # keep first apache record per stay (a few stays have >1)
    ap = ap.drop_duplicates(subset="patientunitstayid", keep="first")
    m = pd.DataFrame({"ICUSTAY_ID": ids, "y": y}).merge(
        ap, left_on="ICUSTAY_ID", right_on="patientunitstayid", how="left"
    )
    join_rate = float(m["apachescore"].notna().mean())

    def prob_metrics(p: np.ndarray, yy: np.ndarray, tag: str) -> dict:
        out = {
            "n": int(len(yy)),
            "auprc": float(average_precision_score(yy, p)),
            "auroc": float(roc_auc_score(yy, p)),
            "brier": float(np.mean((p - yy) ** 2)),
        }
        # 10-bin calibration + logistic recalibration slope/intercept
        bins = pd.qcut(p, 10, duplicates="drop")
        cal = pd.DataFrame({"p": p, "y": yy, "bin": bins}).groupby("bin", observed=True).agg(
            pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")
        )
        cal.to_csv(OUT_DIR / f"calibration_{tag}.csv", index=False)
        from sklearn.linear_model import LogisticRegression

        lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        r = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lg.reshape(-1, 1), yy)
        out["recal_slope"] = float(r.coef_[0][0])
        out["recal_intercept"] = float(r.intercept_[0])
        return out

    res: dict = {"round": "round-2 (pre-specified docs/31 A2)", "zero_training": True}
    res["join_rate_apachescore"] = join_rate
    res["n_missing"] = int(m["apachescore"].isna().sum())
    res["missing_group_mortality"] = float(m.loc[m["apachescore"].isna(), "y"].mean())
    res["observed_group_mortality"] = float(m.loc[m["apachescore"].notna(), "y"].mean())

    phm_col = next((c for c in ap.columns if "predictedhospitalmortality" in c.lower()), None)
    if phm_col:
        p = m[phm_col].to_numpy(dtype=float)
        ok = np.isfinite(p)
        if p[ok].max() > 1.5:
            log("[apache] predictedHospitalMortality looks like percent -> /100")
            p = p / 100.0
        res["apache_predicted_hosp_mortality"] = prob_metrics(p[ok], m.loc[ok, "y"].to_numpy(float), "phm")
        log(f"[apache] hosp-mort prob: {res['apache_predicted_hosp_mortality']}")

    if "apachescore" in ap.columns:
        s = m["apachescore"].to_numpy(dtype=float)
        ok = np.isfinite(s)
        res["apache_score"] = {
            "n": int(ok.sum()),
            "auroc": float(roc_auc_score(m.loc[ok, "y"], s[ok])),
            "auprc": float(average_precision_score(m.loc[ok, "y"], s[ok])),
            "score_range": [float(np.nanmin(s)), float(np.nanmax(s))],
        }
        log(f"[apache] score: {res['apache_score']}")

    if "predictedicumortality" in ap.columns:
        p = m["predictedicumortality"].to_numpy(dtype=float)
        ok = np.isfinite(p)
        if p[ok].max() > 1.5:
            p = p / 100.0
        res["apache_predicted_icu_mortality"] = prob_metrics(p[ok], m.loc[ok, "y"].to_numpy(float), "picm")
        log(f"[apache] icu-mort prob: {res['apache_predicted_icu_mortality']}")

    (OUT_DIR / "apache_metrics.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"→ {OUT_DIR / 'apache_metrics.json'}")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
