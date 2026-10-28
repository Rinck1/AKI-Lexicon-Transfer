#!/usr/bin/env python3
"""PRED post-hoc explore: calibration + DCA + NRI on eICU (docs/28 §6, not previously done).

Inputs: predictions saved by _explore_PRED_cbm_zeroshot.py (MIMIC train-fit
models, eICU zero-shot). All eICU analysis is post-hoc exploratory.

Outputs (data/external/eicu/_explore_PRED/):
  calibration_eicu_bins.csv      10-bin (decile-of-risk) calibration tables
  calibration_summary.json       Brier / slope / intercept / recalibrated Brier
  dca_eicu.csv                   net-benefit curves, thresholds 0.05-0.40
  nri_eicu.json                  continuous NRI (event/none split) + IDI, 1000 bootstrap CI
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

OUT_DIR = PROJ / "data/external/eicu/_explore_PRED"
N_BOOT = 1000
SEED = 42
THRESHOLDS = np.round(np.arange(0.05, 0.4001, 0.01), 4)

MODELS = {
    "static": "p__static",
    "static+15paths": "p__static_plus_15paths",
    "static+84a": "p__static_plus_84a",
    "cbm_full": "p__cbm_full",
}


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def cal_stats(y, p):
    lr = LogisticRegression(max_iter=500)
    lr.fit(logit(p).reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    p_re = lr.predict_proba(logit(p).reshape(-1, 1))[:, 1]
    brier = float(np.mean((p - y) ** 2))
    brier_re = float(np.mean((p_re - y) ** 2))
    return {"slope": slope, "intercept": intercept, "brier": brier,
            "brier_recalibrated_in_eicu": brier_re}


def decile_bins(y, p, n_bins=10):
    q = pd.qcut(p, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "bin": q})
    g = df.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), observed_rate=("y", "mean"))
    g = g.reset_index().sort_values("mean_pred")
    return g


def net_benefit(y, p, t):
    n = len(y)
    tp = float(np.sum((p >= t) & (y == 1)))
    fp = float(np.sum((p >= t) & (y == 0)))
    return tp / n - fp / n * (t / (1.0 - t))


def nri_idi(y, p_new, p_old, rng):
    ev = y == 1
    no = ~ev
    d = np.sign(p_new - p_old)
    nri_ev = float(np.mean(d[ev] > 0) - np.mean(d[ev] < 0))
    nri_no = float(np.mean(d[no] > 0) - np.mean(d[no] < 0))
    idi = float(np.mean(p_new[ev] - p_old[ev]) - np.mean(p_new[no] - p_old[no]))
    boot = np.empty((N_BOOT, 3))
    n = len(y)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        db = d[idx]; yb = y[idx]
        evb = yb == 1; nob = ~evb
        boot[b, 0] = np.mean(db[evb] > 0) - np.mean(db[evb] < 0)
        boot[b, 1] = np.mean(db[nob] > 0) - np.mean(db[nob] < 0)
        boot[b, 2] = np.mean(p_new[idx][evb] - p_old[idx][evb]) - np.mean(p_new[idx][nob] - p_old[idx][nob])
    ci = {k: np.percentile(boot[:, i], [2.5, 97.5]).round(4).tolist()
          for i, k in enumerate(["nri_event", "nri_none", "idi"])}
    return {"nri_total": nri_ev + nri_no, "nri_event": nri_ev, "nri_none": nri_no,
            "idi": idi, "boot95ci": ci}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    e = pd.read_parquet(OUT_DIR / "pred_eicu.parquet")
    mv = pd.read_parquet(OUT_DIR / "pred_mimic_val.parquet")
    y_e = e["y"].to_numpy(float)
    y_v = mv["y"].to_numpy(float)
    rng = np.random.default_rng(SEED)

    # ---- calibration: eICU + MIMIC-val reference ----
    cal_rows = []
    summary = {"post_hoc": True, "n_bins": 10, "binning": "deciles of predicted risk", "models": {}}
    for name, col in MODELS.items():
        for set_name, df, y in [("eicu_zeroshot", e, y_e), ("mimic_val(desc)", mv, y_v)]:
            p = df[col].to_numpy(float)
            st = cal_stats(y, p)
            st.update({"model": name, "set": set_name})
            summary["models"].setdefault(name, {})[set_name] = st
            bins = decile_bins(y, p)
            bins.insert(0, "model", name)
            bins.insert(1, "set", set_name)
            cal_rows.append(bins)
    cal_df = pd.concat(cal_rows, ignore_index=True)
    cal_df.to_csv(OUT_DIR / "calibration_eicu_bins.csv", index=False, encoding="utf-8-sig")

    # ---- DCA (eICU) ----
    pi = float(y_e.mean())
    dca_rows = []
    for t in THRESHOLDS:
        row = {"threshold": float(t),
               "treat_all": pi - (1 - pi) * t / (1 - t), "treat_none": 0.0}
        for name, col in MODELS.items():
            row[name] = net_benefit(y_e, e[col].to_numpy(float), t)
        dca_rows.append(row)
    dca = pd.DataFrame(dca_rows)
    dca.to_csv(OUT_DIR / "dca_eicu.csv", index=False, encoding="utf-8-sig")

    # ---- NRI / IDI (eICU), vs static ----
    nri_out = {"post_hoc": True, "n_boot": N_BOOT, "reference": "static"}
    p_old = e["p__static"].to_numpy(float)
    for name in ["static+15paths", "static+84a", "cbm_full"]:
        p_new = e[MODELS[name]].to_numpy(float)
        nri_out[name] = nri_idi(y_e, p_new, p_old, rng)
    (OUT_DIR / "nri_eicu.json").write_text(json.dumps(nri_out, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "calibration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- console digest ----
    print("=== calibration (slope/intercept; ideal 1/0) ===")
    for name in MODELS:
        for s in ["eicu_zeroshot", "mimic_val(desc)"]:
            st = summary["models"][name][s]
            print(f"{name:<16}{s:<16} slope {st['slope']:.3f}  intercept {st['intercept']:+.3f}  "
                  f"Brier {st['brier']:.4f} -> reBrier {st['brier_recalibrated_in_eicu']:.4f}"
                  if s == "eicu_zeroshot" else
                  f"{name:<16}{s:<16} slope {st['slope']:.3f}  intercept {st['intercept']:+.3f}  Brier {st['brier']:.4f}")
    print("\n=== DCA net benefit (eICU) ===")
    print(dca.round(4).to_string(index=False))
    print("\n=== NRI/IDI (eICU, vs static) ===")
    for name in ["static+15paths", "static+84a", "cbm_full"]:
        r = nri_out[name]
        print(f"{name:<16} NRI {r['nri_total']:+.4f} [ev {r['nri_event']:+.4f} / none {r['nri_none']:+.4f}] "
              f"IDI {r['idi']:+.4f}  95%CI NRI {r['boot95ci']['nri_event']} / {r['boot95ci']['nri_none']}")
    print(f"\n→ {OUT_DIR}")


if __name__ == "__main__":
    main()
