"""Locate eICU-CRD tables and read/write the external workspace."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd

from scripts.utils.constants import EICU_DIR, PROJECT_DIR
from scripts.utils.eicu_constants import EICU_OUT_DIR


def resolve_eicu_dir() -> Path:
    """Find patient.csv.gz. Accepts EICU_DIR, data/eicu-crd, or a project folder named *EICU*."""
    env = os.environ.get("EICU_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path(EICU_DIR))
    candidates.append(Path(PROJECT_DIR) / "data" / "eicu")
    root = Path(PROJECT_DIR)
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and "eicu" in child.name.lower():
                candidates.append(child)
    seen: set[Path] = set()
    for c in candidates:
        try:
            c = c.resolve()
        except OSError:
            continue
        if c in seen:
            continue
        seen.add(c)
        if (c / "patient.csv.gz").is_file():
            return c
    raise FileNotFoundError(
        "eICU patient.csv.gz not found. Set EICU_DIR or place tables in data/eicu-crd/."
    )


def eicu_table(name: str, eicu_dir: Optional[Path] = None) -> Path:
    base = eicu_dir or resolve_eicu_dir()
    fpath = base / f"{name}.csv.gz"
    if not fpath.is_file():
        raise FileNotFoundError(fpath)
    return fpath


def stream_eicu_csv(
    name: str,
    columns: Optional[Sequence[str]] = None,
    chunk_size: int = 400_000,
    eicu_dir: Optional[Path] = None,
) -> Iterator[pd.DataFrame]:
    fpath = eicu_table(name, eicu_dir)
    reader = pd.read_csv(
        fpath,
        compression="gzip",
        usecols=list(columns) if columns is not None else None,
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk in reader:
        yield chunk


def out_dir() -> Path:
    EICU_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return EICU_OUT_DIR


def save_eicu(df: pd.DataFrame, name: str) -> Path:
    dest = out_dir() / f"{name}.parquet"
    df.to_parquet(dest, index=False)
    return dest


def load_eicu(name: str) -> pd.DataFrame:
    src = out_dir() / f"{name}.parquet"
    if not src.is_file():
        raise FileNotFoundError(src)
    return pd.read_parquet(src)
