"""
AKI Research Project — I/O Utilities
=====================================
Read MIMIC-III CSV.GZ, stream large tables, save/load Parquet, write flow logs.
"""

from __future__ import annotations

import gc
import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Set

import pandas as pd
import yaml

from scripts.utils.constants import (
    DATA_PROCESSED,
    DATA_RAW,
    MIMIC_DIR,
    MIMIC_DIR_FALLBACK,
    PROJECT_DIR,
)


def resolve_mimic_dir() -> str:
    """Prefer local project copy, then fallback external path."""
    if os.path.isdir(MIMIC_DIR) and os.path.isfile(
        os.path.join(MIMIC_DIR, "ADMISSIONS.csv.gz")
    ):
        return MIMIC_DIR
    if os.path.isdir(MIMIC_DIR_FALLBACK) and os.path.isfile(
        os.path.join(MIMIC_DIR_FALLBACK, "ADMISSIONS.csv.gz")
    ):
        return MIMIC_DIR_FALLBACK
    raise FileNotFoundError(
        f"MIMIC tables not found in {MIMIC_DIR} or {MIMIC_DIR_FALLBACK}"
    )


def read_gzip_csv(
    table_name: str,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    dtype: Optional[Dict] = None,
    mimic_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Read a MIMIC-III table from CSV.GZ into a DataFrame."""
    base = mimic_dir or resolve_mimic_dir()
    fpath = os.path.join(base, f"{table_name}.csv.gz")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Table not found: {fpath}")
    return pd.read_csv(
        fpath,
        compression="gzip",
        usecols=columns,
        nrows=nrows,
        dtype=dtype,
        low_memory=False,
    )


def stream_gzip_csv(
    table_name: str,
    columns: Optional[List[str]] = None,
    chunk_size: int = 500_000,
    mimic_dir: Optional[str] = None,
) -> Iterator[pd.DataFrame]:
    """Stream a large MIMIC-III CSV.GZ file in chunks."""
    base = mimic_dir or resolve_mimic_dir()
    fpath = os.path.join(base, f"{table_name}.csv.gz")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Table not found: {fpath}")
    for chunk in pd.read_csv(
        fpath,
        compression="gzip",
        usecols=columns,
        chunksize=chunk_size,
        low_memory=False,
    ):
        yield chunk


def stream_filtered_chartevents(
    columns: List[str],
    subject_ids: Set[int],
    itemids: Set[int],
    chunk_size: int = 500_000,
    mimic_dir: Optional[str] = None,
) -> Iterator[pd.DataFrame]:
    """Stream CHARTEVENTS filtered by SUBJECT_ID and ITEMID."""
    base = mimic_dir or resolve_mimic_dir()
    fpath = os.path.join(base, "CHARTEVENTS.csv.gz")
    for chunk in pd.read_csv(
        fpath,
        compression="gzip",
        usecols=columns,
        chunksize=chunk_size,
        low_memory=False,
    ):
        mask = chunk["SUBJECT_ID"].isin(subject_ids) & chunk["ITEMID"].isin(itemids)
        filtered = chunk.loc[mask].copy()
        if len(filtered) > 0:
            yield filtered
        del chunk
        gc.collect()


def save_parquet(df: pd.DataFrame, name: str, subdir: str = "raw") -> str:
    """Save DataFrame as Parquet under data/<subdir>/<name>.parquet."""
    out_dir = DATA_PROCESSED if subdir == "processed" else DATA_RAW
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"{name}.parquet")
    df.to_parquet(fpath, index=False)
    return fpath


def load_parquet(name: str, subdir: str = "raw") -> pd.DataFrame:
    """Load Parquet from data/<subdir>/<name>.parquet."""
    src_dir = DATA_PROCESSED if subdir == "processed" else DATA_RAW
    fpath = os.path.join(src_dir, f"{name}.parquet")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Parquet file not found: {fpath}")
    return pd.read_parquet(fpath)


def append_flow_step(
    flow: List[dict],
    step: str,
    n_stays: int,
    n_subjects: int,
    note: str = "",
) -> List[dict]:
    """Append one CONSORT-style patient-flow row."""
    flow.append(
        {
            "step": step,
            "n_stays": int(n_stays),
            "n_subjects": int(n_subjects),
            "note": note,
        }
    )
    return flow


def save_flow_table(flow: List[dict], name: str = "patient_flow") -> str:
    from scripts.utils.constants import OUTPUT_TABLES

    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    path = os.path.join(OUTPUT_TABLES, f"{name}.csv")
    pd.DataFrame(flow).to_csv(path, index=False)
    return path


def write_manifest(payload: dict, name: str = "data_manifest.yaml") -> str:
    """Write versioned data manifest under data/processed/."""
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    payload = dict(payload)
    payload.setdefault(
        "generated_at_utc",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    path = os.path.join(DATA_PROCESSED, name)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    # also JSON for easy reading
    jpath = path.replace(".yaml", ".json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
