#!/usr/bin/env python3
"""
07 — Variable audit on TRAIN only: four missingness metrics + tier suggestion.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from scripts.utils.constants import OUTPUT_TABLES, TIER_A_SHAPELET, TIER_B_SPARSE  # noqa: E402
from scripts.utils.io_utils import load_parquet, write_manifest  # noqa: E402
from scripts.utils.missingness import audit_many, suggest_tier  # noqa: E402
from scripts.utils.variable_tiers import protocol_tier_map  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("07: Variable audit (train only)")
    print("=" * 60)

    grid = load_parquet("hourly_grid_48h", subdir="processed")
    stay_split = load_parquet("stay_splits", subdir="processed")
    grid = grid.merge(stay_split[["ICUSTAY_ID", "split"]], on="ICUSTAY_ID")
    train = grid[grid["split"] == "train"].copy()
    print(f"  train rows={len(train)} stays={train['ICUSTAY_ID'].nunique()}")

    candidates = [v for v in list(TIER_A_SHAPELET) + list(TIER_B_SPARSE) if v in train.columns]
    audit = audit_many(train, candidates)
    hint = protocol_tier_map()
    audit["clinical_hint"] = audit["variable"].map(hint).fillna("B")
    audit["suggested_tier"] = audit.apply(
        lambda r: suggest_tier(
            r["patient_coverage"],
            r["timepoint_missing"],
            r["mean_n_meas_24h"],
            r["clinical_hint"],
        ),
        axis=1,
    )
    audit = audit.sort_values(["suggested_tier", "patient_coverage"], ascending=[True, False])

    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    path = os.path.join(OUTPUT_TABLES, "variable_audit_train.csv")
    audit.to_csv(path, index=False)
    print(audit[["variable", "patient_coverage", "timepoint_missing",
                 "mean_n_meas_24h", "clinical_hint", "suggested_tier"]].to_string(index=False))
    print(f"\n  saved {path}")

    write_manifest(
        {
            "stage": "07_variable_audit",
            "n_vars": int(len(audit)),
            "tier_counts": audit["suggested_tier"].value_counts().to_dict(),
            "path": path,
        }
    )
    print("\n07 done.")


if __name__ == "__main__":
    main()
