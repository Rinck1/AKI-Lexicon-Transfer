#!/usr/bin/env python3
"""eICU long-format labs / vitals / urine / GCS for the stage-40 base.

Same column contract as scripts/02 (ICUSTAY_ID, charttime, variable, value).
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import PHYSIO_RANGES  # noqa: E402
from scripts.utils.eicu_constants import (  # noqa: E402
    APERIODIC_COL_TO_VAR,
    GCS_LABEL,
    LAB_NAME_TO_VAR,
    PERIODIC_COL_TO_VAR,
    gcs_component,
    is_urine_cell,
    maybe_creatinine_to_mgdl,
    offset_to_time,
)
from scripts.utils.eicu_io import load_eicu, out_dir, resolve_eicu_dir, save_eicu, stream_eicu_csv  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _clip(series: pd.Series, name: str) -> pd.Series:
    if name not in PHYSIO_RANGES:
        return series
    lo, hi = PHYSIO_RANGES[name]
    return series.where(series.between(lo, hi))


def _attach_times(df: pd.DataFrame, offset_col: str, cohort: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        cohort[["ICUSTAY_ID", "SUBJECT_ID", "HADM_ID", "INTIME", "OUTTIME"]],
        on="ICUSTAY_ID",
        how="inner",
    )
    df["charttime"] = df[offset_col].map(offset_to_time)
    lo = df["INTIME"] - pd.Timedelta(hours=24)
    hi = df["OUTTIME"] + pd.Timedelta(hours=24)
    df = df[(df["charttime"] >= lo) & (df["charttime"] <= hi)]
    return df


def extract_labs(cohort: pd.DataFrame, eicu: Path) -> pd.DataFrame:
    _log("[41] labs …")
    stay_ids = set(int(x) for x in cohort["ICUSTAY_ID"])
    names = set(LAB_NAME_TO_VAR)
    parts: list[pd.DataFrame] = []
    cols = [
        "patientunitstayid",
        "labresultoffset",
        "labname",
        "labresult",
        "labmeasurenamesystem",
    ]
    for chunk in stream_eicu_csv("lab", columns=cols, eicu_dir=eicu, chunk_size=500_000):
        m = chunk["patientunitstayid"].isin(stay_ids) & chunk["labname"].isin(names)
        c = chunk.loc[m].copy()
        if c.empty:
            continue
        c["variable"] = c["labname"].map(LAB_NAME_TO_VAR)
        c["value"] = pd.to_numeric(c["labresult"], errors="coerce")
        c = c.dropna(subset=["value", "variable"])
        cr = c["variable"] == "creatinine"
        if cr.any():
            c.loc[cr, "value"] = [
                maybe_creatinine_to_mgdl(v, u)
                for v, u in zip(c.loc[cr, "value"], c.loc[cr, "labmeasurenamesystem"].astype(str))
            ]
        rows = []
        for var, g in c.groupby("variable"):
            g = g.copy()
            g["value"] = _clip(g["value"], str(var))
            rows.append(g)
        if rows:
            keep = pd.concat(rows, ignore_index=True)
            keep = keep.rename(columns={"patientunitstayid": "ICUSTAY_ID"})
            parts.append(keep[["ICUSTAY_ID", "labresultoffset", "variable", "value"]])
        del chunk, c
        gc.collect()
    if not parts:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "variable", "value"])
    labs = pd.concat(parts, ignore_index=True)
    labs = _attach_times(labs, "labresultoffset", cohort)
    labs = labs[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "variable", "value"]]
    _log(f"  labs rows={len(labs):,} vars={labs['variable'].nunique()}")
    return labs.reset_index(drop=True)


def _melt_vitals(chunk: pd.DataFrame, col_map: dict[str, str], offset_col: str) -> pd.DataFrame:
    present = [c for c in col_map if c in chunk.columns]
    if not present:
        return pd.DataFrame()
    id_cols = ["patientunitstayid", offset_col]
    long = chunk[id_cols + present].melt(
        id_vars=id_cols, value_vars=present, var_name="src", value_name="value"
    )
    long["variable"] = long["src"].map(col_map)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value", "variable"])
    long = long.rename(columns={"patientunitstayid": "ICUSTAY_ID", offset_col: "offset"})
    return long[["ICUSTAY_ID", "offset", "variable", "value"]]


def _vital_schema() -> pa.Schema:
    return pa.schema(
        [
            ("SUBJECT_ID", pa.int64()),
            ("HADM_ID", pa.int64()),
            ("ICUSTAY_ID", pa.int64()),
            ("charttime", pa.timestamp("ns")),
            ("variable", pa.string()),
            ("value", pa.float64()),
        ]
    )


def _write_vital_chunk(
    writer: pq.ParquetWriter,
    long: pd.DataFrame,
    cohort: pd.DataFrame,
    offset_col: str,
) -> int:
    """melt → clip per var → attach times → write one arrow batch. Returns rows written."""
    if long.empty:
        return 0
    rows = []
    for var, g in long.groupby("variable"):
        g = g.copy()
        g["value"] = _clip(g["value"], str(var))
        rows.append(g)
    merged = pd.concat(rows, ignore_index=True)
    merged = _attach_times(merged.rename(columns={"offset": offset_col}), offset_col, cohort)
    merged = merged[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "variable", "value"]]
    if merged.empty:
        return 0
    merged["SUBJECT_ID"] = merged["SUBJECT_ID"].astype("int64")
    merged["HADM_ID"] = merged["HADM_ID"].astype("int64")
    merged["ICUSTAY_ID"] = merged["ICUSTAY_ID"].astype("int64")
    merged["variable"] = merged["variable"].astype(str)
    merged["value"] = merged["value"].astype("float64")
    table = pa.Table.from_pandas(merged, schema=_vital_schema(), preserve_index=False)
    writer.write_table(table)
    return len(merged)


def extract_vitals(cohort: pd.DataFrame, eicu: Path) -> tuple[pq.ParquetWriter, int, set[str]]:
    _log("[41] vitalPeriodic / vitalAperiodic …")
    stay_ids = set(int(x) for x in cohort["ICUSTAY_ID"])
    dest = out_dir() / "meas_vitals.parquet"
    writer = pq.ParquetWriter(dest, _vital_schema())
    n_written = 0
    vars_seen: set[str] = set()
    per_cols = ["patientunitstayid", "observationoffset", *PERIODIC_COL_TO_VAR.keys()]
    for chunk in stream_eicu_csv("vitalPeriodic", columns=per_cols, eicu_dir=eicu, chunk_size=300_000):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        if chunk.empty:
            continue
        long = _melt_vitals(chunk, PERIODIC_COL_TO_VAR, "observationoffset")
        if not long.empty:
            vars_seen.update(long["variable"].unique().tolist())
            n_written += _write_vital_chunk(writer, long, cohort, "observationoffset")
        del chunk, long
        gc.collect()
    ap_cols = ["patientunitstayid", "observationoffset", *APERIODIC_COL_TO_VAR.keys()]
    for chunk in stream_eicu_csv("vitalAperiodic", columns=ap_cols, eicu_dir=eicu, chunk_size=400_000):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        if chunk.empty:
            continue
        long = _melt_vitals(chunk, APERIODIC_COL_TO_VAR, "observationoffset")
        if not long.empty:
            vars_seen.update(long["variable"].unique().tolist())
            n_written += _write_vital_chunk(writer, long, cohort, "observationoffset")
        del chunk, long
        gc.collect()
    _log(f"  vitals rows={n_written:,} vars={len(vars_seen)}")
    return writer, n_written, vars_seen


def extract_urine(cohort: pd.DataFrame, eicu: Path) -> pd.DataFrame:
    _log("[41] intakeOutput urine …")
    stay_ids = set(int(x) for x in cohort["ICUSTAY_ID"])
    parts: list[pd.DataFrame] = []
    cols = [
        "patientunitstayid",
        "intakeoutputoffset",
        "cellpath",
        "celllabel",
        "cellvaluenumeric",
    ]
    for chunk in stream_eicu_csv("intakeOutput", columns=cols, eicu_dir=eicu, chunk_size=400_000):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        if chunk.empty:
            continue
        mask = [
            is_urine_cell(lab, path)
            for lab, path in zip(chunk["celllabel"].astype(str), chunk["cellpath"].astype(str))
        ]
        c = chunk.loc[np.array(mask, dtype=bool)].copy()
        if c.empty:
            continue
        c["value"] = pd.to_numeric(c["cellvaluenumeric"], errors="coerce")
        c = c.dropna(subset=["value"])
        c["value"] = _clip(c["value"], "urine_output_ml")
        c = c.rename(columns={"patientunitstayid": "ICUSTAY_ID"})
        parts.append(c[["ICUSTAY_ID", "intakeoutputoffset", "value"]])
        del chunk, c
        gc.collect()
    if not parts:
        return pd.DataFrame(columns=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "variable", "value"])
    u = pd.concat(parts, ignore_index=True)
    u["variable"] = "urine_output_ml"
    u = _attach_times(u, "intakeoutputoffset", cohort)
    u = u[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "variable", "value"]]
    _log(f"  urine rows={len(u):,}")
    return u.reset_index(drop=True)


def extract_gcs(
    cohort: pd.DataFrame, eicu: Path, writer: pq.ParquetWriter
) -> tuple[int, set[str]]:
    _log("[41] nurseCharting GCS …")
    stay_ids = set(int(x) for x in cohort["ICUSTAY_ID"])
    n_written = 0
    vars_seen: set[str] = set()
    cols = [
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ]
    for chunk in stream_eicu_csv("nurseCharting", columns=cols, eicu_dir=eicu, chunk_size=400_000):
        lab = chunk["nursingchartcelltypevallabel"].astype(str).str.strip().str.lower()
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids) & lab.eq(GCS_LABEL)]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["variable"] = chunk["nursingchartcelltypevalname"].map(gcs_component)
        chunk["value"] = pd.to_numeric(chunk["nursingchartvalue"], errors="coerce")
        chunk = chunk.dropna(subset=["variable", "value"])
        if chunk.empty:
            continue
        chunk = chunk.rename(columns={"patientunitstayid": "ICUSTAY_ID"})
        sub = chunk[["ICUSTAY_ID", "nursingchartoffset", "variable", "value"]]
        rows = []
        for var, g in sub.groupby("variable"):
            g = g.copy()
            if var == "gcs_total":
                g["value"] = g["value"].where(g["value"].between(3, 15))
            else:
                g["value"] = _clip(g["value"], str(var))
            rows.append(g)
        merged = pd.concat(rows, ignore_index=True)
        merged = _attach_times(merged, "nursingchartoffset", cohort)
        merged = merged[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "charttime", "variable", "value"]]
        if merged.empty:
            del chunk
            gc.collect()
            continue
        vars_seen.update(merged["variable"].unique().tolist())
        merged["SUBJECT_ID"] = merged["SUBJECT_ID"].astype("int64")
        merged["HADM_ID"] = merged["HADM_ID"].astype("int64")
        merged["ICUSTAY_ID"] = merged["ICUSTAY_ID"].astype("int64")
        merged["variable"] = merged["variable"].astype(str)
        merged["value"] = merged["value"].astype("float64")
        table = pa.Table.from_pandas(merged, schema=_vital_schema(), preserve_index=False)
        writer.write_table(table)
        n_written += len(merged)
        del chunk
        gc.collect()
    _log(f"  gcs rows={n_written:,} vars={len(vars_seen)}")
    return n_written, vars_seen


def main() -> None:
    eicu = resolve_eicu_dir()
    cohort = load_eicu("icu_base")
    for c in ["INTIME", "OUTTIME"]:
        cohort[c] = pd.to_datetime(cohort[c])
    _log(f"[41] base stays={len(cohort)}")

    labs = extract_labs(cohort, eicu)
    save_eicu(labs, "meas_labs")
    urine = extract_urine(cohort, eicu)
    save_eicu(urine, "meas_urine")
    writer, n_vit, vit_vars = extract_vitals(cohort, eicu)
    n_gcs, gcs_vars = extract_gcs(cohort, eicu, writer)
    writer.close()
    n_vitals = n_vit + n_gcs
    all_vital_vars = sorted(vit_vars | gcs_vars)
    (out_dir() / "stage41_manifest.json").write_text(
        json.dumps(
            {
                "stage": "41_eicu_extract_measurements",
                "n_labs": int(len(labs)),
                "n_urine": int(len(urine)),
                "n_vitals": int(n_vitals),
                "n_vitals_physio": int(n_vit),
                "n_gcs": int(n_gcs),
                "lab_vars": sorted(labs["variable"].unique().tolist()) if len(labs) else [],
                "vital_vars": all_vital_vars,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log("[41] done")


if __name__ == "__main__":
    main()
