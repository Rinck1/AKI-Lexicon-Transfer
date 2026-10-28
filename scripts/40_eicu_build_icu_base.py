#!/usr/bin/env python3
"""eICU adult ICU base (external). Same age / LOS / ESRD spirit as scripts/01.

Writes data/external/eicu/icu_base.parquet. Does not touch MIMIC processed tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import MAX_AGE, MIN_AGE, MIN_ICU_LOS_HOURS, OUTPUT_TABLES  # noqa: E402
from scripts.utils.eicu_constants import (  # noqa: E402
    CKD_PATH_TOKENS,
    COMORBID_PATH_TOKENS,
    ESRD_PATH_TOKENS,
    offset_to_time,
    parse_eicu_age,
    parse_eicu_gender,
    path_has_any,
)
from scripts.utils.eicu_io import load_eicu, out_dir, resolve_eicu_dir, save_eicu, stream_eicu_csv  # noqa: E402
from scripts.utils.io_utils import append_flow_step  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _subject_map(uniquepids: pd.Series) -> pd.Series:
    codes, uniques = pd.factorize(uniquepids.astype(str), sort=True)
    mapping = pd.DataFrame({"uniquepid": uniques, "SUBJECT_ID": range(len(uniques))})
    save_eicu(mapping, "subject_id_map")
    return pd.Series(codes, index=uniquepids.index)


def _dialysis_stays(eicu: Path) -> set[int]:
    ids: set[int] = set()
    for chunk in stream_eicu_csv(
        "apacheApsVar",
        columns=["patientunitstayid", "dialysis"],
        eicu_dir=eicu,
    ):
        hit = chunk.loc[pd.to_numeric(chunk["dialysis"], errors="coerce") > 0, "patientunitstayid"]
        ids.update(int(x) for x in hit.dropna())
    return ids


def _history_flags(eicu: Path, stay_ids: set[int]) -> pd.DataFrame:
    rows: dict[int, dict] = {}
    cols = ["patientunitstayid", "pasthistorypath"]
    for chunk in stream_eicu_csv("pastHistory", columns=cols, eicu_dir=eicu, chunk_size=500_000):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        if chunk.empty:
            continue
        for rec in chunk.itertuples(index=False):
            sid = int(rec.patientunitstayid)
            path = str(rec.pasthistorypath or "")
            bag = rows.setdefault(
                sid,
                {
                    "ICUSTAY_ID": sid,
                    "ESRD_HIST": 0,
                    "CKD_FLAG": 0,
                    "CHF": 0,
                    "Diabetes": 0,
                    "HTN": 0,
                    "COPD": 0,
                    "LiverDisease": 0,
                    "Arrhythmia": 0,
                },
            )
            if path_has_any(path, ESRD_PATH_TOKENS):
                bag["ESRD_HIST"] = 1
            if path_has_any(path, CKD_PATH_TOKENS):
                bag["CKD_FLAG"] = 1
            for col, toks in COMORBID_PATH_TOKENS.items():
                if path_has_any(path, toks):
                    bag[col] = 1
    if not rows:
        return pd.DataFrame(
            columns=[
                "ICUSTAY_ID",
                "ESRD_HIST",
                "CKD_FLAG",
                "CHF",
                "Diabetes",
                "HTN",
                "COPD",
                "LiverDisease",
                "Arrhythmia",
            ]
        )
    return pd.DataFrame(list(rows.values()))


def main() -> None:
    eicu = resolve_eicu_dir()
    _log(f"[40] eICU_DIR={eicu}")
    flow: list = []

    patient = pd.read_csv(
        eicu / "patient.csv.gz",
        compression="gzip",
        low_memory=False,
        usecols=[
            "patientunitstayid",
            "patienthealthsystemstayid",
            "uniquepid",
            "gender",
            "age",
            "hospitalid",
            "unittype",
            "unitstaytype",
            "unitvisitnumber",
            "hospitaladmitoffset",
            "hospitaldischargeoffset",
            "hospitaldischargestatus",
            "unitdischargeoffset",
            "admissionweight",
            "unitadmitsource",
        ],
    )
    append_flow_step(
        flow,
        "00_raw_unit_stays",
        patient["patientunitstayid"].nunique(),
        patient["uniquepid"].nunique(),
        "all eICU unit stays",
    )

    patient["AGE"] = patient["age"].map(parse_eicu_age)
    patient["GENDER"] = patient["gender"].map(parse_eicu_gender)
    patient["HOSPITAL_MORTALITY"] = (
        patient["hospitaldischargestatus"].astype(str).str.strip().str.lower().eq("expired").astype(int)
    )
    status_ok = patient["hospitaldischargestatus"].astype(str).str.strip().str.lower().isin(
        {"alive", "expired"}
    )
    patient = patient.loc[status_ok].copy()
    append_flow_step(
        flow,
        "01_known_hospital_discharge_status",
        patient["patientunitstayid"].nunique(),
        patient["uniquepid"].nunique(),
        "Alive or Expired",
    )

    patient = patient[patient["AGE"] >= MIN_AGE].copy()
    patient = patient[patient["AGE"] <= MAX_AGE].copy()
    append_flow_step(
        flow,
        "02_age_ge_18",
        patient["patientunitstayid"].nunique(),
        patient["uniquepid"].nunique(),
        f"AGE >= {MIN_AGE}",
    )

    los_h = pd.to_numeric(patient["unitdischargeoffset"], errors="coerce") / 60.0
    patient["ICU_LOS_HOURS"] = los_h
    patient = patient[los_h >= MIN_ICU_LOS_HOURS].copy()
    append_flow_step(
        flow,
        "03_icu_los_ge_min",
        patient["patientunitstayid"].nunique(),
        patient["uniquepid"].nunique(),
        f"ICU_LOS_HOURS >= {MIN_ICU_LOS_HOURS}",
    )

    # First ICU visit of the hospital encounter
    uv = pd.to_numeric(patient["unitvisitnumber"], errors="coerce")
    patient = patient.loc[uv.fillna(99) == 1].copy()
    append_flow_step(
        flow,
        "04_first_unit_visit_per_hadm",
        patient["patientunitstayid"].nunique(),
        patient["uniquepid"].nunique(),
        "unitvisitnumber == 1",
    )

    stay_ids = set(int(x) for x in patient["patientunitstayid"])
    _log("[40] dialysis (apacheApsVar) + pastHistory …")
    dial = _dialysis_stays(eicu)
    hist = _history_flags(eicu, stay_ids)
    patient["ESRD_DIALYSIS"] = patient["patientunitstayid"].isin(dial).astype(int)
    if hist.empty:
        for c in ["ESRD_HIST", "CKD_FLAG", "CHF", "Diabetes", "HTN", "COPD", "LiverDisease", "Arrhythmia"]:
            patient[c] = 0
    else:
        patient = patient.merge(hist, left_on="patientunitstayid", right_on="ICUSTAY_ID", how="left")
        patient = patient.drop(columns=["ICUSTAY_ID"], errors="ignore")
        for c in ["ESRD_HIST", "CKD_FLAG", "CHF", "Diabetes", "HTN", "COPD", "LiverDisease", "Arrhythmia"]:
            patient[c] = patient[c].fillna(0).astype(int)

    patient["ESRD_FLAG"] = ((patient["ESRD_DIALYSIS"] == 1) | (patient["ESRD_HIST"] == 1)).astype(int)
    n_before = len(patient)
    patient = patient[patient["ESRD_FLAG"] == 0].copy()
    append_flow_step(
        flow,
        "05_exclude_esrd_dialysis",
        patient["patientunitstayid"].nunique(),
        patient["uniquepid"].nunique(),
        f"drop apache dialysis or HD/PD/transplant history (dropped {n_before - len(patient)})",
    )

    patient["SUBJECT_ID"] = _subject_map(patient["uniquepid"])
    patient["ICUSTAY_ID"] = patient["patientunitstayid"].astype(int)
    patient["HADM_ID"] = patient["patienthealthsystemstayid"].astype(int)
    patient["INTIME"] = offset_to_time(0)
    patient["OUTTIME"] = patient["unitdischargeoffset"].map(offset_to_time)
    patient["ADMITTIME"] = patient["hospitaladmitoffset"].map(offset_to_time)
    patient["DISCHTIME"] = patient["hospitaldischargeoffset"].map(offset_to_time)
    patient["DEATHTIME"] = patient["DISCHTIME"].where(patient["HOSPITAL_MORTALITY"] == 1, pd.NaT)
    patient["ADMISSION_TYPE"] = patient["unitadmitsource"].fillna("").astype(str)
    patient["FIRST_CAREUNIT"] = patient["unittype"].fillna("").astype(str)
    patient["weight_kg"] = pd.to_numeric(patient["admissionweight"], errors="coerce")

    keep = [
        "SUBJECT_ID",
        "HADM_ID",
        "ICUSTAY_ID",
        "uniquepid",
        "hospitalid",
        "GENDER",
        "AGE",
        "ADMISSION_TYPE",
        "ADMITTIME",
        "DISCHTIME",
        "DEATHTIME",
        "INTIME",
        "OUTTIME",
        "ICU_LOS_HOURS",
        "HOSPITAL_MORTALITY",
        "FIRST_CAREUNIT",
        "weight_kg",
        "ESRD_FLAG",
        "CKD_FLAG",
        "CHF",
        "Diabetes",
        "HTN",
        "COPD",
        "LiverDisease",
        "Arrhythmia",
    ]
    out = patient[keep].reset_index(drop=True)
    path = save_eicu(out, "icu_base")
    comorbid = out[
        ["ICUSTAY_ID", "SUBJECT_ID", "HADM_ID", "CKD_FLAG", "CHF", "Diabetes", "HTN", "COPD", "LiverDisease", "Arrhythmia"]
    ].copy()
    save_eicu(comorbid, "meas_comorbid")

    flow_path = Path(OUTPUT_TABLES)
    flow_path.mkdir(parents=True, exist_ok=True)
    flow_csv = flow_path / "eicu_patient_flow_40.csv"
    pd.DataFrame(flow).to_csv(flow_csv, index=False)
    (out_dir() / "stage40_manifest.json").write_text(
        json.dumps(
            {
                "stage": "40_eicu_build_icu_base",
                "eicu_dir": str(eicu),
                "n_stays": int(len(out)),
                "n_subjects": int(out["SUBJECT_ID"].nunique()),
                "mortality": float(out["HOSPITAL_MORTALITY"].mean()),
                "age_mean": float(out["AGE"].mean()),
                "icu_base": str(path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(f"[40] stays={len(out)} subjects={out['SUBJECT_ID'].nunique()} mort={out['HOSPITAL_MORTALITY'].mean()*100:.1f}%")
    _log(f"[40] → {path}")


if __name__ == "__main__":
    main()
