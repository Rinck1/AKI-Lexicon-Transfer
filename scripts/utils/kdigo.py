"""
KDIGO AKI detection without future-window leakage.

Baseline at time t uses only measurements strictly before t
(within BASELINE_LOOKBACK_HOURS). Absolute 0.3 mg/dL rise uses
min creatinine in [t-48h, t) only.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.utils.constants import (
    ABS_INCREASE_WINDOW_HOURS,
    BASELINE_LOOKBACK_HOURS,
    CA_MAX_HOURS,
    KDIGO_ABS_INCREASE,
    KDIGO_CR_MAX,
    KDIGO_RATIO_1,
    KDIGO_RATIO_2,
    KDIGO_RATIO_3,
    URINE_STAGE1_HOURS,
    URINE_STAGE1_ML_KG_H,
    URINE_STAGE2_HOURS,
    URINE_STAGE3_HOURS,
    URINE_STAGE3_ML_KG_H,
)


def stage_from_ratio_and_abs(
    scr: float,
    baseline: float,
    abs_increase_from_48h_min: Optional[float] = None,
) -> int:
    """
    Map creatinine relative to baseline → KDIGO stage 0–3.

    abs_increase_from_48h_min: scr - min(scr in prior 48h); used for stage-1
    absolute criterion when baseline ratio is modest.
    """
    if not np.isfinite(scr) or not np.isfinite(baseline) or baseline <= 0:
        return 0
    ratio = scr / baseline
    stage = 0
    if ratio >= KDIGO_RATIO_3 or scr >= KDIGO_CR_MAX:
        stage = 3
    elif ratio >= KDIGO_RATIO_2:
        stage = 2
    elif ratio >= KDIGO_RATIO_1:
        stage = 1
    if (
        abs_increase_from_48h_min is not None
        and np.isfinite(abs_increase_from_48h_min)
        and abs_increase_from_48h_min >= KDIGO_ABS_INCREASE
    ):
        stage = max(stage, 1)
    return int(stage)


def baseline_creatinine_prior_only(
    times_h: np.ndarray,
    scr: np.ndarray,
    t_index: int,
    lookback_h: float = BASELINE_LOOKBACK_HOURS,
) -> float:
    """
    Baseline at times_h[t_index]: min of finite scr in [t - lookback, t).

    Returns np.nan if no prior measurement in the window.
    """
    t = times_h[t_index]
    window_start = t - lookback_h
    prior = scr[(times_h >= window_start) & (times_h < t) & np.isfinite(scr)]
    if prior.size == 0:
        return float("nan")
    return float(np.min(prior))


def abs_increase_vs_48h_prior(
    times_h: np.ndarray,
    scr: np.ndarray,
    t_index: int,
    window_h: float = ABS_INCREASE_WINDOW_HOURS,
) -> float:
    """scr[t] - min(scr in [t-48h, t)); nan if insufficient history."""
    t = times_h[t_index]
    mask = (times_h >= t - window_h) & (times_h < t) & np.isfinite(scr)
    prior = scr[mask]
    if prior.size == 0 or not np.isfinite(scr[t_index]):
        return float("nan")
    return float(scr[t_index] - np.min(prior))


def creatinine_stages_for_series(
    times_h: np.ndarray,
    scr: np.ndarray,
) -> np.ndarray:
    """Per-timepoint KDIGO stage from creatinine only (causal baseline)."""
    n = len(times_h)
    stages = np.zeros(n, dtype=int)
    for i in range(n):
        base = baseline_creatinine_prior_only(times_h, scr, i)
        abs_inc = abs_increase_vs_48h_prior(times_h, scr, i)
        stages[i] = stage_from_ratio_and_abs(float(scr[i]), base, abs_inc)
    return stages


def urine_rate_ml_kg_h(
    urine_ml_per_h: np.ndarray,
    weight_kg: float,
) -> np.ndarray:
    """Convert hourly urine volume to mL/kg/h. NaN if missing urine or weight."""
    if weight_kg is None or not np.isfinite(weight_kg) or weight_kg <= 0:
        return np.full_like(urine_ml_per_h, np.nan, dtype=float)
    out = np.full(len(urine_ml_per_h), np.nan, dtype=float)
    m = np.isfinite(urine_ml_per_h)
    out[m] = urine_ml_per_h[m] / weight_kg
    return out


def urine_stage_from_rolling_rates(
    rates_ml_kg_h: np.ndarray,
    is_measured: np.ndarray,
) -> np.ndarray:
    """
    Causal urine staging on an hourly grid.

    Only hours with is_measured==True contribute; missing hours break oliguria runs
    (missing ≠ zero urine).
    """
    n = len(rates_ml_kg_h)
    stages = np.zeros(n, dtype=int)
    run_half = 0   # consecutive measured hours with rate < 0.5
    run_third = 0  # consecutive measured hours with rate < 0.3
    for i in range(n):
        if not is_measured[i] or not np.isfinite(rates_ml_kg_h[i]):
            run_half = 0
            run_third = 0
            stages[i] = 0
            continue
        r = rates_ml_kg_h[i]
        if r < URINE_STAGE3_ML_KG_H:
            run_third += 1
        else:
            run_third = 0
        if r < URINE_STAGE1_ML_KG_H:
            run_half += 1
        else:
            run_half = 0
        st = 0
        if run_third >= URINE_STAGE3_HOURS:
            st = 3
        elif run_half >= URINE_STAGE2_HOURS:
            st = 2
        elif run_half >= URINE_STAGE1_HOURS:
            st = 1
        stages[i] = st
    return stages


def combine_stages(cr_stage: int, uo_stage: int) -> int:
    """KDIGO: maximum of creatinine and urine stages."""
    return int(max(int(cr_stage), int(uo_stage)))


def detect_onset(
    times_h: np.ndarray,
    combined_stage: np.ndarray,
) -> Optional[Tuple[float, int]]:
    """
    First time stage >= 1.

    Returns (time_h, stage) or None if never AKI.
    """
    hit = np.where(combined_stage >= 1)[0]
    if hit.size == 0:
        return None
    i = int(hit[0])
    return float(times_h[i]), int(combined_stage[i])


def classify_ca_ha(
    onset_time,
    icu_intime,
    ca_max_hours: float = CA_MAX_HOURS,
) -> str:
    """Community-acquired if onset <= intime + ca_max_hours."""
    delta_h = (pd.Timestamp(onset_time) - pd.Timestamp(icu_intime)).total_seconds() / 3600.0
    if delta_h <= ca_max_hours:
        return "CA"
    return "HA"
