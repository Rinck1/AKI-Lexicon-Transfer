#!/usr/bin/env python3
"""
05 — SUBJECT_ID stratified split + leakage audit (primary HA-48 cohort).
"""
from __future__ import annotations

import os
import sys

import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from scripts.utils.constants import DATA_SPLITS, OUTPUT_TABLES, RANDOM_SEED, SPLIT_RATIOS  # noqa: E402
from scripts.utils.io_utils import load_parquet, save_parquet, write_manifest  # noqa: E402
from scripts.utils.split_utils import (  # noqa: E402
    assert_no_subject_overlap,
    stay_split_from_subject_split,
    subject_level_split,
)


def main() -> None:
    print("=" * 60)
    print("05: Subject-level split + leakage audit")
    print("=" * 60)

    primary = load_parquet("aki_primary_ha48", subdir="processed")
    print(f"  primary N={len(primary)} subjects={primary['SUBJECT_ID'].nunique()}")

    splits = subject_level_split(
        primary,
        subject_col="SUBJECT_ID",
        label_col="HOSPITAL_MORTALITY",
        ratios=SPLIT_RATIOS,
        seed=RANDOM_SEED,
    )
    pairs = assert_no_subject_overlap(splits)
    print(f"  overlap pairs (must be 0): {pairs}")

    stay_split = stay_split_from_subject_split(primary, splits)
    os.makedirs(DATA_SPLITS, exist_ok=True)
    subj_path = os.path.join(DATA_SPLITS, "subject_splits.parquet")
    stay_path = os.path.join(DATA_SPLITS, "stay_splits.parquet")
    splits.to_parquet(subj_path, index=False)
    stay_split.to_parquet(stay_path, index=False)

    # also save under processed names
    save_parquet(splits, "subject_splits", subdir="processed")
    save_parquet(stay_split, "stay_splits", subdir="processed")

    summary = (
        stay_split.merge(
            primary[["ICUSTAY_ID", "HOSPITAL_MORTALITY"]], on="ICUSTAY_ID"
        )
        .groupby("split")
        .agg(
            n_stays=("ICUSTAY_ID", "nunique"),
            n_subjects=("SUBJECT_ID", "nunique"),
            mortality=("HOSPITAL_MORTALITY", "mean"),
        )
        .reset_index()
    )
    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    sum_path = os.path.join(OUTPUT_TABLES, "split_summary.csv")
    summary.to_csv(sum_path, index=False)
    print(summary.to_string(index=False))

    write_manifest(
        {
            "stage": "05_split_subjects",
            "seed": RANDOM_SEED,
            "ratios": list(SPLIT_RATIOS),
            "overlap": pairs,
            "summary": summary.to_dict(orient="records"),
        }
    )
    print("\n05 done.")


if __name__ == "__main__":
    main()
