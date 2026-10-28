#!/usr/bin/env python3
"""
04 — Align hourly grid relative to AKI onset (NO imputation).

Primary: HOUR 0..47  |  Secondary S1: 0..71
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from scripts.utils.constants import (  # noqa: E402
    FEATURE_WINDOW_HOURS,
    FEATURE_WINDOW_HOURS_S1,
    TIER_A_SHAPELET,
    TIER_B_SPARSE,
)
from scripts.utils.io_utils import load_parquet, save_parquet, write_manifest  # noqa: E402

VALUE_VARS = [
    v
    for v in list(TIER_A_SHAPELET) + list(TIER_B_SPARSE)
    if v != "gcs_total"
]


def _load_long(stay_ids: set) -> pd.DataFrame:
    parts = []
    for name in ["meas_labs", "meas_vitals", "meas_urine"]:
        print(f"  loading {name} ...", flush=True)
        df = load_parquet(name, subdir="raw")
        df = df[df["ICUSTAY_ID"].isin(stay_ids)].copy()
        df["charttime"] = pd.to_datetime(df["charttime"])
        df = df[df["variable"].isin(VALUE_VARS)]
        parts.append(df[["ICUSTAY_ID", "charttime", "variable", "value"]])
        del df
    return pd.concat(parts, ignore_index=True)


def build_grid_fast(cohort: pd.DataFrame, long_df: pd.DataFrame, window_h: int) -> pd.DataFrame:
    """Vectorized hourly aggregation + full grid expand."""
    print(f"  building {window_h}h grid ...", flush=True)
    onset = cohort.set_index("ICUSTAY_ID")["onset_time"]
    long_df = long_df.copy()
    long_df["onset_time"] = long_df["ICUSTAY_ID"].map(onset)
    long_df = long_df.dropna(subset=["onset_time"])
    long_df["HOUR"] = np.floor(
        (long_df["charttime"] - long_df["onset_time"]).dt.total_seconds() / 3600.0
    ).astype("int32")
    long_df = long_df[(long_df["HOUR"] >= 0) & (long_df["HOUR"] < window_h)]

    agg = (
        long_df.groupby(["ICUSTAY_ID", "HOUR", "variable"], as_index=False)["value"]
        .mean()
    )
    # pivot to wide
    wide = agg.pivot_table(
        index=["ICUSTAY_ID", "HOUR"],
        columns="variable",
        values="value",
        aggfunc="mean",
    )
    wide = wide.reset_index()

    # full index of stays × hours
    stays = cohort[["ICUSTAY_ID", "SUBJECT_ID", "HOSPITAL_MORTALITY"]].copy()
    hours = pd.DataFrame({"HOUR": np.arange(window_h, dtype=np.int32)})
    stays["key"] = 1
    hours["key"] = 1
    full = stays.merge(hours, on="key").drop(columns="key")

    grid = full.merge(wide, on=["ICUSTAY_ID", "HOUR"], how="left")
    for v in VALUE_VARS:
        if v not in grid.columns:
            grid[v] = np.nan
        grid[f"{v}_is_measured"] = grid[v].notna()

    # GCS total when all components present
    comps = ["gcs_eye", "gcs_motor", "gcs_verbal"]
    if all(c in grid.columns for c in comps):
        ok = grid[[f"{c}_is_measured" for c in comps]].all(axis=1)
        grid["gcs_total"] = np.where(ok, grid[comps].sum(axis=1), np.nan)
        grid["gcs_total_is_measured"] = ok
    else:
        grid["gcs_total"] = np.nan
        grid["gcs_total_is_measured"] = False

    return grid


def main() -> None:
    print("=" * 60, flush=True)
    print("04: Align time grid relative to onset (no impute)", flush=True)
    print("=" * 60, flush=True)

    primary = load_parquet("aki_primary_ha48", subdir="processed")
    primary["onset_time"] = pd.to_datetime(primary["onset_time"])
    print(f"  primary stays={len(primary)}", flush=True)

    stay_ids = set(primary["ICUSTAY_ID"].astype(int))
    long_df = _load_long(stay_ids)
    print(f"  long measurements rows={len(long_df):,}", flush=True)

    grid48 = build_grid_fast(primary, long_df, FEATURE_WINDOW_HOURS)
    path48 = save_parquet(grid48, "hourly_grid_48h", subdir="processed")
    print(f"  48h grid: {grid48.shape} → {path48}", flush=True)

    grid72 = build_grid_fast(primary, long_df, FEATURE_WINDOW_HOURS_S1)
    path72 = save_parquet(grid72, "hourly_grid_72h", subdir="processed")
    print(f"  72h grid: {grid72.shape} → {path72}", flush=True)

    write_manifest(
        {
            "stage": "04_align_time_grid",
            "window_primary_h": FEATURE_WINDOW_HOURS,
            "window_s1_h": FEATURE_WINDOW_HOURS_S1,
            "n_rows_48h": int(len(grid48)),
            "n_stays": int(primary["ICUSTAY_ID"].nunique()),
            "imputed": False,
        }
    )
    print("\n04 done.", flush=True)


if __name__ == "__main__":
    main()
