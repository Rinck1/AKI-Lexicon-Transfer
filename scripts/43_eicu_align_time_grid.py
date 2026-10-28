#!/usr/bin/env python3
"""Onset-relative 48h grid on eICU (no imputation). Same logic as scripts/04."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import FEATURE_WINDOW_HOURS, TIER_A_SHAPELET, TIER_B_SPARSE  # noqa: E402
from scripts.utils.eicu_io import load_eicu, out_dir, save_eicu  # noqa: E402

VALUE_VARS = [
    v
    for v in list(TIER_A_SHAPELET) + list(TIER_B_SPARSE)
    if v != "gcs_total"
] + ["gcs_total"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_long(stay_ids: set[int]) -> pd.DataFrame:
    parts = []
    for name in ("meas_labs", "meas_vitals", "meas_urine"):
        _log(f"  load {name}")
        df = load_eicu(name)
        df = df[df["ICUSTAY_ID"].isin(stay_ids)].copy()
        df["charttime"] = pd.to_datetime(df["charttime"])
        df = df[df["variable"].isin(VALUE_VARS)]
        parts.append(df[["ICUSTAY_ID", "charttime", "variable", "value"]])
        del df
    return pd.concat(parts, ignore_index=True)


def build_grid(cohort: pd.DataFrame, long_df: pd.DataFrame, window_h: int) -> pd.DataFrame:
    onset = cohort.set_index("ICUSTAY_ID")["onset_time"]
    long_df = long_df.copy()
    long_df["onset_time"] = long_df["ICUSTAY_ID"].map(onset)
    long_df = long_df.dropna(subset=["onset_time"])
    long_df["HOUR"] = np.floor(
        (long_df["charttime"] - long_df["onset_time"]).dt.total_seconds() / 3600.0
    ).astype("int32")
    long_df = long_df[(long_df["HOUR"] >= 0) & (long_df["HOUR"] < window_h)]
    agg = long_df.groupby(["ICUSTAY_ID", "HOUR", "variable"], as_index=False)["value"].mean()
    wide = agg.pivot_table(
        index=["ICUSTAY_ID", "HOUR"], columns="variable", values="value", aggfunc="mean"
    ).reset_index()
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
    comps = ["gcs_eye", "gcs_motor", "gcs_verbal"]
    if all(c in grid.columns for c in comps):
        ok = grid[[f"{c}_is_measured" for c in comps]].all(axis=1)
        from_comp = ok & ~grid["gcs_total_is_measured"]
        grid.loc[from_comp, "gcs_total"] = grid.loc[from_comp, comps].sum(axis=1)
        grid.loc[from_comp, "gcs_total_is_measured"] = True
    return grid


def main() -> None:
    primary = load_eicu("aki_primary_ha48")
    primary["onset_time"] = pd.to_datetime(primary["onset_time"])
    _log(f"[43] primary stays={len(primary)}")
    stay_ids = set(int(x) for x in primary["ICUSTAY_ID"])
    long_df = _load_long(stay_ids)
    _log(f"  long rows={len(long_df):,}")
    grid = build_grid(primary, long_df, FEATURE_WINDOW_HOURS)
    path = save_eicu(grid, "hourly_grid_48h")
    measured = {
        v: float(grid[f"{v}_is_measured"].mean())
        for v in ("heart_rate", "sbp", "urine_output_ml", "creatinine", "lactate", "gcs_total")
        if f"{v}_is_measured" in grid.columns
    }
    (out_dir() / "stage43_manifest.json").write_text(
        json.dumps(
            {"stage": "43_eicu_align_time_grid", "grid_shape": list(grid.shape), "frac_measured": measured},
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(f"[43] grid {grid.shape} → {path}")
    _log(f"  frac_measured {measured}")


if __name__ == "__main__":
    main()
