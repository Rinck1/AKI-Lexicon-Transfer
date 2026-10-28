"""
Four missingness metrics for variable auditing (protocol §4).
All metrics should be computed on the training set only.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.utils.constants import (
    TIER_A_MAX_TIMEPOINT_MISSING,
    TIER_A_MIN_PATIENT_COVERAGE,
)


def patient_coverage(
    df: pd.DataFrame,
    value_col: str,
    id_col: str = "ICUSTAY_ID",
    measured_col: Optional[str] = None,
) -> float:
    """Fraction of ids with ≥1 real measurement."""
    if measured_col and measured_col in df.columns:
        has = df.groupby(id_col)[measured_col].any()
    else:
        has = df.groupby(id_col)[value_col].apply(lambda s: s.notna().any())
    return float(has.mean()) if len(has) else 0.0


def timepoint_coverage(
    df: pd.DataFrame,
    value_col: str,
    measured_col: Optional[str] = None,
) -> float:
    """Fraction of rows with a real measurement."""
    if measured_col and measured_col in df.columns:
        return float(df[measured_col].astype(bool).mean())
    return float(df[value_col].notna().mean())


def mean_measurement_count(
    df: pd.DataFrame,
    value_col: str,
    id_col: str = "ICUSTAY_ID",
    hour_col: str = "HOUR",
    max_hour: Optional[int] = None,
    measured_col: Optional[str] = None,
) -> float:
    """Mean number of measured hours per id within hour < max_hour."""
    sub = df
    if max_hour is not None and hour_col in df.columns:
        sub = df[df[hour_col] < max_hour]
    if measured_col and measured_col in sub.columns:
        counts = sub.groupby(id_col)[measured_col].sum()
    else:
        counts = sub.groupby(id_col)[value_col].apply(lambda s: s.notna().sum())
    return float(counts.mean()) if len(counts) else 0.0


def mean_max_consecutive_missing(
    df: pd.DataFrame,
    value_col: str,
    id_col: str = "ICUSTAY_ID",
    hour_col: str = "HOUR",
    measured_col: Optional[str] = None,
) -> float:
    """Mean over ids of the longest consecutive missing-hour run."""

    def _max_run(g: pd.DataFrame) -> int:
        g = g.sort_values(hour_col)
        if measured_col and measured_col in g.columns:
            miss = ~g[measured_col].astype(bool).values
        else:
            miss = g[value_col].isna().values
        best = cur = 0
        for m in miss:
            if m:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    runs = df.groupby(id_col, group_keys=False).apply(_max_run, include_groups=False)
    return float(runs.mean()) if len(runs) else 0.0


def audit_variable(
    df: pd.DataFrame,
    value_col: str,
    id_col: str = "ICUSTAY_ID",
    hour_col: str = "HOUR",
    measured_col: Optional[str] = None,
) -> Dict[str, float]:
    """Compute the four protocol metrics for one variable."""
    tp_cov = timepoint_coverage(df, value_col, measured_col)
    return {
        "patient_coverage": patient_coverage(df, value_col, id_col, measured_col),
        "timepoint_coverage": tp_cov,
        "timepoint_missing": 1.0 - tp_cov,
        "mean_n_meas_24h": mean_measurement_count(
            df, value_col, id_col, hour_col, max_hour=24, measured_col=measured_col
        ),
        "mean_n_meas_48h": mean_measurement_count(
            df, value_col, id_col, hour_col, max_hour=48, measured_col=measured_col
        ),
        "mean_max_consec_missing": mean_max_consecutive_missing(
            df, value_col, id_col, hour_col, measured_col
        ),
    }


def audit_many(
    df: pd.DataFrame,
    variables: List[str],
    id_col: str = "ICUSTAY_ID",
    hour_col: str = "HOUR",
) -> pd.DataFrame:
    """Audit a list of value columns; auto-detect `{var}_is_measured`."""
    rows = []
    for v in variables:
        mcol = f"{v}_is_measured" if f"{v}_is_measured" in df.columns else None
        if v not in df.columns:
            continue
        m = audit_variable(df, v, id_col, hour_col, mcol)
        m["variable"] = v
        rows.append(m)
    return pd.DataFrame(rows)


def suggest_tier(
    patient_coverage: float,
    timepoint_missing: float,
    mean_n_meas_24h: float = 0.0,
    clinical_tier_hint: str = "A",
    min_patient_cov: float = TIER_A_MIN_PATIENT_COVERAGE,
    max_tp_missing_for_a: float = TIER_A_MAX_TIMEPOINT_MISSING,
    drop_tp_missing: float = 0.95,
) -> str:
    """
    Suggest A / B / C / drop from metrics + clinical hint.

    clinical_tier_hint: protocol-assigned A/B/C before data audit.
    """
    if timepoint_missing >= drop_tp_missing and patient_coverage < 0.05:
        return "drop"
    if clinical_tier_hint == "C":
        return "C"
    if clinical_tier_hint == "B":
        return "B" if patient_coverage >= 0.05 else "drop"
    # hint A
    if (
        patient_coverage >= min_patient_cov
        and timepoint_missing <= max_tp_missing_for_a
        and mean_n_meas_24h >= 2.0
    ):
        return "A"
    if patient_coverage >= 0.20:
        return "B"
    return "drop"
