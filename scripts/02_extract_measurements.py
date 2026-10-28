#!/usr/bin/env python3
"""
02 — Extract long-format clinical measurements for ICU base cohort.

Outputs (data/raw/):
  meas_labs.parquet, meas_vitals.parquet, meas_urine.parquet,
  meas_meds.parquet, meas_comorbid.parquet
"""
from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from scripts.utils.constants import (  # noqa: E402
    ELIXHAUSER_ICD9_MAP,
    LAB_ITEMID_TO_NAME,
    MED_CATEGORIES,
    PHYSIO_RANGES,
    URINE_ITEMIDS,
    VITAL_ITEMID_TO_NAME,
)
from scripts.utils.io_utils import (  # noqa: E402
    load_parquet,
    read_gzip_csv,
    resolve_mimic_dir,
    save_parquet,
    stream_filtered_chartevents,
    stream_gzip_csv,
    write_manifest,
)


def _clip(series: pd.Series, name: str) -> pd.Series:
    if name not in PHYSIO_RANGES:
        return series
    lo, hi = PHYSIO_RANGES[name]
    return series.where(series.between(lo, hi))


def extract_labs(cohort: pd.DataFrame) -> pd.DataFrame:
    print("\n[labs] streaming LABEVENTS ...")
    hadm_ids = set(cohort["HADM_ID"].unique())
    itemids = set(LAB_ITEMID_TO_NAME.keys())
    parts = []
    cols = ["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUEUOM"]
    for chunk in stream_gzip_csv("LABEVENTS", columns=cols, chunk_size=1_000_000):
        m = chunk["HADM_ID"].isin(hadm_ids) & chunk["ITEMID"].isin(itemids)
        c = chunk.loc[m].copy()
        if len(c) == 0:
            continue
        c["variable"] = c["ITEMID"].map(LAB_ITEMID_TO_NAME)
        c = c.dropna(subset=["VALUENUM", "variable"])
        out_rows = []
        for var, g in c.groupby("variable"):
            g = g.copy()
            g["VALUENUM"] = _clip(g["VALUENUM"], var)
            out_rows.append(g)
        if out_rows:
            parts.append(pd.concat(out_rows, ignore_index=True))
        del chunk, c
        gc.collect()
    if not parts:
        return pd.DataFrame(
            columns=["SUBJECT_ID", "HADM_ID", "CHARTTIME", "variable", "value"]
        )
    labs = pd.concat(parts, ignore_index=True)
    labs = labs.rename(columns={"VALUENUM": "value", "CHARTTIME": "charttime"})
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    labs = labs[["SUBJECT_ID", "HADM_ID", "charttime", "variable", "value"]]
    # map HADM → ICUSTAY via cohort (may be multiple; keep all stays for hadm)
    labs = labs.merge(
        cohort[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]],
        on=["SUBJECT_ID", "HADM_ID"],
        how="inner",
    )
    # keep labs during ICU ± 24h pre for baseline
    labs = labs[
        (labs["charttime"] >= labs["INTIME"] - pd.Timedelta(hours=24))
        & (labs["charttime"] <= labs["OUTTIME"] + pd.Timedelta(hours=24))
    ]
    labs = labs.drop(columns=["INTIME", "OUTTIME"])
    print(f"  labs rows={len(labs):,} vars={labs['variable'].nunique()}")
    return labs.reset_index(drop=True)


def extract_vitals(cohort: pd.DataFrame) -> pd.DataFrame:
    print("\n[vitals] streaming CHARTEVENTS (slow) ...")
    subject_ids = set(cohort["SUBJECT_ID"].astype(int).unique())
    itemids = set(VITAL_ITEMID_TO_NAME.keys())
    cols = [
        "SUBJECT_ID",
        "HADM_ID",
        "ICUSTAY_ID",
        "ITEMID",
        "CHARTTIME",
        "VALUENUM",
    ]
    stay_ids = set(cohort["ICUSTAY_ID"].astype(int).unique())
    parts = []
    n_chunks = 0
    for chunk in stream_filtered_chartevents(cols, subject_ids, itemids, chunk_size=800_000):
        n_chunks += 1
        if n_chunks % 20 == 0:
            print(f"  ... chunk {n_chunks}, kept rows so far prep")
        c = chunk[chunk["ICUSTAY_ID"].isin(stay_ids)].copy()
        if len(c) == 0:
            continue
        c["variable"] = c["ITEMID"].map(VITAL_ITEMID_TO_NAME)
        c = c.dropna(subset=["VALUENUM", "variable"])
        rows = []
        for var, g in c.groupby("variable"):
            g = g.copy()
            if var.startswith("temperature_f"):
                # convert F→C into temperature_c
                g["VALUENUM"] = (g["VALUENUM"] - 32.0) * 5.0 / 9.0
                g["variable"] = "temperature_c"
                var = "temperature_c"
            g["VALUENUM"] = _clip(g["VALUENUM"], var)
            rows.append(g)
        if rows:
            parts.append(pd.concat(rows, ignore_index=True))
        del chunk, c
        gc.collect()
    if not parts:
        return pd.DataFrame(
            columns=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "variable", "value"]
        )
    vit = pd.concat(parts, ignore_index=True)
    vit = vit.rename(columns={"VALUENUM": "value", "CHARTTIME": "charttime"})
    vit["charttime"] = pd.to_datetime(vit["charttime"])
    vit = vit[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "variable", "value"]]
    print(f"  vitals rows={len(vit):,} vars={vit['variable'].nunique()}")
    return vit.reset_index(drop=True)


def extract_urine(cohort: pd.DataFrame) -> pd.DataFrame:
    print("\n[urine] streaming OUTPUTEVENTS ...")
    stay_ids = set(cohort["ICUSTAY_ID"].astype(int).unique())
    cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUE"]
    parts = []
    for chunk in stream_gzip_csv("OUTPUTEVENTS", columns=cols, chunk_size=1_000_000):
        m = chunk["ICUSTAY_ID"].isin(stay_ids) & chunk["ITEMID"].isin(URINE_ITEMIDS)
        c = chunk.loc[m].copy()
        if len(c) == 0:
            continue
        c = c.dropna(subset=["VALUE"])
        c["VALUE"] = pd.to_numeric(c["VALUE"], errors="coerce")
        c = c.dropna(subset=["VALUE"])
        c["VALUE"] = _clip(c["VALUE"], "urine_output_ml")
        parts.append(c)
        del chunk, c
        gc.collect()
    if not parts:
        return pd.DataFrame(
            columns=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "variable", "value"]
        )
    u = pd.concat(parts, ignore_index=True)
    u = u.rename(columns={"VALUE": "value", "CHARTTIME": "charttime"})
    u["charttime"] = pd.to_datetime(u["charttime"])
    u["variable"] = "urine_output_ml"
    u = u[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "variable", "value"]]
    print(f"  urine rows={len(u):,}")
    return u.reset_index(drop=True)


def extract_meds(cohort: pd.DataFrame) -> pd.DataFrame:
    print("\n[meds] PRESCRIPTIONS (day-level) ...")
    hadm_ids = set(cohort["HADM_ID"].unique())
    cols = [
        "SUBJECT_ID",
        "HADM_ID",
        "STARTDATE",
        "ENDDATE",
        "DRUG",
        "DRUG_NAME_GENERIC",
    ]
    parts = []
    for chunk in stream_gzip_csv("PRESCRIPTIONS", columns=cols, chunk_size=500_000):
        c = chunk[chunk["HADM_ID"].isin(hadm_ids)].copy()
        if len(c) == 0:
            continue
        drug = (
            c["DRUG"].fillna("").astype(str)
            + " "
            + c["DRUG_NAME_GENERIC"].fillna("").astype(str)
        ).str.lower()
        flags = {}
        for cat, kws in MED_CATEGORIES.items():
            flags[cat] = drug.apply(lambda s, kws=kws: int(any(k in s for k in kws)))
        for cat, ser in flags.items():
            c[cat] = ser
        c = c[(c[list(MED_CATEGORIES.keys())].sum(axis=1) > 0)]
        if len(c):
            parts.append(
                c[
                    ["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE"]
                    + list(MED_CATEGORIES.keys())
                ]
            )
        del chunk, c
        gc.collect()
    if not parts:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE"])
    meds = pd.concat(parts, ignore_index=True)
    meds["STARTDATE"] = pd.to_datetime(meds["STARTDATE"], errors="coerce")
    meds["ENDDATE"] = pd.to_datetime(meds["ENDDATE"], errors="coerce")
    print(f"  meds rows={len(meds):,}")
    return meds.reset_index(drop=True)


def extract_comorbid(cohort: pd.DataFrame) -> pd.DataFrame:
    print("\n[comorbid] DIAGNOSES_ICD ...")
    diagnoses = read_gzip_csv(
        "DIAGNOSES_ICD", columns=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]
    )
    hadm_ids = set(cohort["HADM_ID"].unique())
    diagnoses = diagnoses[diagnoses["HADM_ID"].isin(hadm_ids)].copy()
    diagnoses["icd_norm"] = (
        diagnoses["ICD9_CODE"].astype(str).str.replace(".", "", regex=False).str.upper()
    )
    cats = sorted(set(ELIXHAUSER_ICD9_MAP.values()))
    diagnoses["category"] = diagnoses["icd_norm"].map(ELIXHAUSER_ICD9_MAP)
    # CKD flag
    diagnoses["CKD_FLAG"] = diagnoses["icd_norm"].str.startswith("585").astype(int)

    base = cohort[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]].drop_duplicates()
    for cat in cats:
        pos = diagnoses.loc[diagnoses["category"] == cat, "HADM_ID"].unique()
        base[cat] = base["HADM_ID"].isin(pos).astype(int)
    ckd_hadm = diagnoses.loc[diagnoses["CKD_FLAG"] == 1, "HADM_ID"].unique()
    base["CKD_FLAG"] = base["HADM_ID"].isin(ckd_hadm).astype(int)
    print(f"  comorbid stays={len(base)}")
    return base.reset_index(drop=True)


def main() -> None:
    print("=" * 60)
    print("02: Extract measurements (long tables)")
    print("=" * 60)
    mimic = resolve_mimic_dir()
    print(f"  MIMIC_DIR = {mimic}")

    cohort = load_parquet("icu_base", subdir="raw")
    for c in ["INTIME", "OUTTIME"]:
        cohort[c] = pd.to_datetime(cohort[c])
    print(f"  cohort stays={len(cohort)}")

    labs = extract_labs(cohort)
    save_parquet(labs, "meas_labs", subdir="raw")

    urine = extract_urine(cohort)
    save_parquet(urine, "meas_urine", subdir="raw")

    meds = extract_meds(cohort)
    save_parquet(meds, "meas_meds", subdir="raw")

    comorbid = extract_comorbid(cohort)
    save_parquet(comorbid, "meas_comorbid", subdir="raw")

    vitals = extract_vitals(cohort)
    save_parquet(vitals, "meas_vitals", subdir="raw")

    write_manifest(
        {
            "stage": "02_extract_measurements",
            "n_lab_rows": int(len(labs)),
            "n_vital_rows": int(len(vitals)),
            "n_urine_rows": int(len(urine)),
            "n_med_rows": int(len(meds)),
            "outputs": [
                "meas_labs",
                "meas_vitals",
                "meas_urine",
                "meas_meds",
                "meas_comorbid",
            ],
        }
    )
    print("\n02 done.")


if __name__ == "__main__":
    main()
