#!/usr/bin/env python3
"""
06 — Features after split: train-only median impute + aggregates for baseline.

Outputs:
  data/processed/hourly_grid_48h_imputed.parquet  (values filled; masks kept)
  data/processed/stay_features_48h.parquet        (one row per stay)
  data/processed/impute_stats_train.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from scripts.utils.constants import DATA_PROCESSED, TIER_A_SHAPELET, TIER_B_SPARSE  # noqa: E402
from scripts.utils.io_utils import load_parquet, save_parquet, write_manifest  # noqa: E402

VALUE_VARS = [
    v
    for v in list(TIER_A_SHAPELET) + list(TIER_B_SPARSE)
    if v in ("creatinine", "bun", "urine_output_ml", "heart_rate", "sbp", "dbp", "mbp",
             "respiratory_rate", "spo2", "gcs_total", "gcs_motor", "gcs_eye", "gcs_verbal",
             "sodium", "potassium", "bicarbonate", "platelet", "wbc", "hemoglobin",
             "lactate", "ph", "albumin", "anion_gap", "rdw", "bilirubin_total",
             "alt", "ast", "glucose", "chloride", "calcium", "magnesium", "phosphate",
             "hematocrit", "inr", "ptt", "fio2", "alp")
]


def main() -> None:
    print("=" * 60)
    print("06: Features post-split (train-only impute)")
    print("=" * 60)

    grid = load_parquet("hourly_grid_48h", subdir="processed")
    stay_split = load_parquet("stay_splits", subdir="processed")
    primary = load_parquet("aki_primary_ha48", subdir="processed")
    try:
        comorbid = load_parquet("meas_comorbid", subdir="raw")
    except FileNotFoundError:
        comorbid = None

    grid = grid.merge(stay_split[["ICUSTAY_ID", "split"]], on="ICUSTAY_ID", how="left")
    assert grid["split"].notna().all(), "missing split labels on grid"

    train_mask = grid["split"] == "train"
    stats = {}
    grid_imp = grid.copy()

    present_vars = [v for v in VALUE_VARS if v in grid.columns]
    for v in present_vars:
        mcol = f"{v}_is_measured"
        train_vals = grid.loc[train_mask & grid[mcol].astype(bool), v] if mcol in grid.columns else grid.loc[train_mask, v]
        train_vals = train_vals.dropna()
        med = float(train_vals.median()) if len(train_vals) else 0.0
        stats[v] = {"median": med, "n_train_measured": int(len(train_vals))}
        # impute only where not measured
        if mcol in grid_imp.columns:
            miss = ~grid_imp[mcol].astype(bool)
            grid_imp.loc[miss, v] = med
        else:
            grid_imp[v] = grid_imp[v].fillna(med)

    save_parquet(grid_imp, "hourly_grid_48h_imputed", subdir="processed")
    stats_path = os.path.join(DATA_PROCESSED, "impute_stats_train.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # stay-level aggregates for baseline (using raw measured values where possible)
    feat_rows = []
    for stay_id, g in grid.groupby("ICUSTAY_ID"):
        rec = {
            "ICUSTAY_ID": int(stay_id),
            "SUBJECT_ID": int(g["SUBJECT_ID"].iloc[0]),
            "split": g["split"].iloc[0],
            "HOSPITAL_MORTALITY": int(g["HOSPITAL_MORTALITY"].iloc[0]),
        }
        for v in present_vars:
            mcol = f"{v}_is_measured"
            if mcol in g.columns:
                measured = g.loc[g[mcol].astype(bool), v]
            else:
                measured = g[v].dropna()
            rec[f"{v}_mean"] = float(measured.mean()) if len(measured) else np.nan
            rec[f"{v}_last"] = float(measured.iloc[-1]) if len(measured) else np.nan
            rec[f"{v}_min"] = float(measured.min()) if len(measured) else np.nan
            rec[f"{v}_max"] = float(measured.max()) if len(measured) else np.nan
            rec[f"{v}_std"] = float(measured.std()) if len(measured) > 1 else 0.0
            rec[f"{v}_n"] = int(len(measured))
            rec[f"{v}_miss_frac"] = float(1.0 - len(measured) / len(g))
        feat_rows.append(rec)

    feats = pd.DataFrame(feat_rows)
    # static covariates
    static_cols = ["ICUSTAY_ID", "AGE", "GENDER", "ADMISSION_TYPE", "onset_stage", "onset_hour_from_icu"]
    static_cols = [c for c in static_cols if c in primary.columns]
    feats = feats.merge(primary[static_cols], on="ICUSTAY_ID", how="left")
    feats["GENDER_M"] = (feats["GENDER"] == "M").astype(int) if "GENDER" in feats.columns else 0
    if comorbid is not None:
        ccols = [c for c in comorbid.columns if c not in ("SUBJECT_ID", "HADM_ID")]
        feats = feats.merge(comorbid[ccols], on="ICUSTAY_ID", how="left")

    # train-only median fill for stay features
    train_f = feats["split"] == "train"
    num_cols = [
        c
        for c in feats.columns
        if c not in ("ICUSTAY_ID", "SUBJECT_ID", "split", "GENDER", "ADMISSION_TYPE")
        and pd.api.types.is_numeric_dtype(feats[c])
    ]
    stay_stats = {}
    for c in num_cols:
        med = float(feats.loc[train_f, c].median()) if train_f.any() else 0.0
        if not np.isfinite(med):
            med = 0.0
        stay_stats[c] = med
        feats[c] = feats[c].fillna(med)

    save_parquet(feats, "stay_features_48h", subdir="processed")
    with open(os.path.join(DATA_PROCESSED, "stay_feature_fill_stats.json"), "w") as f:
        json.dump(stay_stats, f, indent=2)

    print(f"  stay features: {feats.shape}")
    write_manifest(
        {
            "stage": "06_features_post_split",
            "n_stays": int(feats["ICUSTAY_ID"].nunique()),
            "n_feature_cols": int(feats.shape[1]),
            "impute_stats": stats_path,
        }
    )
    print("\n06 done.")


if __name__ == "__main__":
    main()
