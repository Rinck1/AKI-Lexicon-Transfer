#!/usr/bin/env python3
"""
01 — Build adult ICU base population (coarse exclusions).

Output:
  data/raw/icu_base.parquet
  outputs/tables/patient_flow_01.csv
  data/processed/data_manifest.yaml (updated)
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
    ESRD_ICD9_CODES,
    MAX_AGE,
    MIN_AGE,
    MIN_ICU_LOS_HOURS,
    OUTPUT_TABLES,
)
from scripts.utils.io_utils import (  # noqa: E402
    append_flow_step,
    read_gzip_csv,
    resolve_mimic_dir,
    save_parquet,
    write_manifest,
)


def _normalize_icd9(code) -> str:
    if pd.isna(code):
        return ""
    s = str(code).strip().upper().replace(".", "")
    return s


def main() -> None:
    print("=" * 60)
    print("01: Build adult ICU base population")
    print("=" * 60)
    mimic = resolve_mimic_dir()
    print(f"  MIMIC_DIR = {mimic}")

    flow: list = []

    patients = read_gzip_csv(
        "PATIENTS",
        columns=["SUBJECT_ID", "GENDER", "DOB", "DOD", "EXPIRE_FLAG"],
    )
    admissions = read_gzip_csv(
        "ADMISSIONS",
        columns=[
            "SUBJECT_ID",
            "HADM_ID",
            "ADMITTIME",
            "DISCHTIME",
            "DEATHTIME",
            "ADMISSION_TYPE",
            "HOSPITAL_EXPIRE_FLAG",
            "ETHNICITY",
        ],
    )
    icustays = read_gzip_csv(
        "ICUSTAYS",
        columns=[
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "INTIME",
            "OUTTIME",
            "LOS",
            "FIRST_CAREUNIT",
            "LAST_CAREUNIT",
        ],
    )
    diagnoses = read_gzip_csv(
        "DIAGNOSES_ICD",
        columns=["SUBJECT_ID", "HADM_ID", "ICD9_CODE", "SEQ_NUM"],
    )

    append_flow_step(
        flow,
        "00_raw_icustays",
        icustays["ICUSTAY_ID"].nunique(),
        icustays["SUBJECT_ID"].nunique(),
        "all ICU stays in MIMIC-III",
    )

    for c in ["ADMITTIME", "DISCHTIME", "DEATHTIME"]:
        admissions[c] = pd.to_datetime(admissions[c])
    for c in ["INTIME", "OUTTIME"]:
        icustays[c] = pd.to_datetime(icustays[c])
    patients["DOB"] = pd.to_datetime(patients["DOB"])

    df = icustays.merge(
        admissions,
        on=["SUBJECT_ID", "HADM_ID"],
        how="inner",
        validate="many_to_one",
    )
    df = df.merge(patients, on="SUBJECT_ID", how="inner", validate="many_to_one")

    # Age at ICU intime. MIMIC shifts DOB for age>89 → huge year gaps; avoid int64 overflow.
    age_years = df["INTIME"].dt.year - df["DOB"].dt.year
    # refine with month/day when safe
    before_bday = (df["INTIME"].dt.month < df["DOB"].dt.month) | (
        (df["INTIME"].dt.month == df["DOB"].dt.month)
        & (df["INTIME"].dt.day < df["DOB"].dt.day)
    )
    age_years = age_years - before_bday.astype(int)
    age_years = age_years.where(age_years < 150, 91.5)  # de-identified elderly
    df["AGE"] = age_years.clip(lower=0, upper=MAX_AGE).astype(float)

    df["ICU_LOS_HOURS"] = (df["OUTTIME"] - df["INTIME"]).dt.total_seconds() / 3600.0
    df["HOSPITAL_MORTALITY"] = df["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int)

    append_flow_step(
        flow,
        "01_merged",
        df["ICUSTAY_ID"].nunique(),
        df["SUBJECT_ID"].nunique(),
        "ICUSTAYS ⨯ ADMISSIONS ⨯ PATIENTS",
    )

    df = df[df["AGE"] >= MIN_AGE].copy()
    append_flow_step(
        flow,
        "02_age_ge_18",
        df["ICUSTAY_ID"].nunique(),
        df["SUBJECT_ID"].nunique(),
        f"AGE >= {MIN_AGE}",
    )

    df = df[df["ICU_LOS_HOURS"] >= MIN_ICU_LOS_HOURS].copy()
    append_flow_step(
        flow,
        "03_icu_los_ge_min",
        df["ICUSTAY_ID"].nunique(),
        df["SUBJECT_ID"].nunique(),
        f"ICU_LOS_HOURS >= {MIN_ICU_LOS_HOURS}",
    )

    # ESRD / dialysis status on same HADM
    diagnoses["icd_norm"] = diagnoses["ICD9_CODE"].map(_normalize_icd9)
    esrd_hadm = set(
        diagnoses.loc[diagnoses["icd_norm"].isin(ESRD_ICD9_CODES), "HADM_ID"].unique()
    )
    df["ESRD_FLAG"] = df["HADM_ID"].isin(esrd_hadm).astype(int)
    df = df[df["ESRD_FLAG"] == 0].copy()
    append_flow_step(
        flow,
        "04_exclude_esrd_dialysis",
        df["ICUSTAY_ID"].nunique(),
        df["SUBJECT_ID"].nunique(),
        "exclude ESRD/dialysis ICD on HADM",
    )

    # Keep first ICU stay per HADM (standard)
    df = df.sort_values(["HADM_ID", "INTIME"])
    df = df.drop_duplicates(subset=["HADM_ID"], keep="first")
    append_flow_step(
        flow,
        "05_first_icu_per_hadm",
        df["ICUSTAY_ID"].nunique(),
        df["SUBJECT_ID"].nunique(),
        "first ICU stay per hospital admission",
    )

    keep_cols = [
        "SUBJECT_ID",
        "HADM_ID",
        "ICUSTAY_ID",
        "GENDER",
        "AGE",
        "ADMISSION_TYPE",
        "ETHNICITY",
        "ADMITTIME",
        "DISCHTIME",
        "DEATHTIME",
        "INTIME",
        "OUTTIME",
        "ICU_LOS_HOURS",
        "HOSPITAL_MORTALITY",
        "FIRST_CAREUNIT",
        "LAST_CAREUNIT",
        "ESRD_FLAG",
    ]
    out = df[keep_cols].reset_index(drop=True)

    path = save_parquet(out, "icu_base", subdir="raw")
    print(f"\n  Saved {path}")
    print(f"  stays={len(out)} subjects={out['SUBJECT_ID'].nunique()}")
    print(f"  mortality={out['HOSPITAL_MORTALITY'].mean()*100:.1f}%")
    print(f"  age mean={out['AGE'].mean():.1f}")

    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    flow_path = os.path.join(OUTPUT_TABLES, "patient_flow_01.csv")
    pd.DataFrame(flow).to_csv(flow_path, index=False)
    print(f"  Flow → {flow_path}")

    write_manifest(
        {
            "stage": "01_build_icu_base",
            "mimic_dir": mimic,
            "n_stays": int(len(out)),
            "n_subjects": int(out["SUBJECT_ID"].nunique()),
            "mortality": float(out["HOSPITAL_MORTALITY"].mean()),
            "age_mean": float(out["AGE"].mean()),
            "outputs": {"icu_base": path, "flow": flow_path},
        }
    )
    print("\n01 done.")


if __name__ == "__main__":
    main()
