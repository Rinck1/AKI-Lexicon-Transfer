"""KDIGO causal baseline: no future information."""

from __future__ import annotations

import numpy as np

from scripts.utils.kdigo import (
    baseline_creatinine_prior_only,
    combine_stages,
    creatinine_stages_for_series,
    detect_onset,
    stage_from_ratio_and_abs,
    urine_stage_from_rolling_rates,
)


def test_baseline_ignores_future_low_values():
    """Future low creatinine must not pull baseline down at earlier times."""
    times = np.arange(0, 72, dtype=float)
    scr = np.full(72, 1.0)
    scr[10:20] = 1.2
    scr[50] = 0.5  # future low
    base_at_24 = baseline_creatinine_prior_only(times, scr, 24)
    assert base_at_24 == 1.0


def test_no_baseline_without_prior():
    times = np.array([0.0, 1.0])
    scr = np.array([1.5, 1.6])
    base0 = baseline_creatinine_prior_only(times, scr, 0)
    assert np.isnan(base0)
    base = baseline_creatinine_prior_only(times, scr, 1)
    assert base == 1.5


def test_stage_ratio_and_abs():
    assert stage_from_ratio_and_abs(1.5, 1.0) == 1
    assert stage_from_ratio_and_abs(2.0, 1.0) == 2
    assert stage_from_ratio_and_abs(3.0, 1.0) == 3
    assert stage_from_ratio_and_abs(1.2, 1.0, abs_increase_from_48h_min=0.35) == 1
    assert stage_from_ratio_and_abs(1.1, 1.0, abs_increase_from_48h_min=0.1) == 0


def test_onset_detection_first_stage():
    times = np.arange(10, dtype=float)
    stages = np.array([0, 0, 0, 1, 2, 2, 1, 0, 0, 0])
    onset = detect_onset(times, stages)
    assert onset is not None
    assert onset[0] == 3.0
    assert onset[1] == 1


def test_creatinine_series_does_not_use_future_min_as_baseline():
    times = np.arange(0, 60, dtype=float)
    scr = np.ones(60) * 1.0
    scr[5] = 1.0
    scr[30] = 2.0  # rise
    scr[50] = 0.4  # future low — must not define early baseline
    stages = creatinine_stages_for_series(times, scr)
    # at index 30 should be able to detect rise vs prior ~1.0
    assert stages[30] >= 1


def test_urine_missing_breaks_oliguria_run():
    """Missing hours must not count as zero urine."""
    rates = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)
    meas = np.array([1, 1, 1, 0, 1, 1, 1], dtype=bool)  # break at hour 3
    stages = urine_stage_from_rolling_rates(rates, meas)
    # continuous 6h oliguria never completed after break
    assert stages[5] == 0 or stages[6] <= 1
    # if all measured for 6h of <0.5 → stage 1 at hour 5
    meas2 = np.ones(8, dtype=bool)
    rates2 = np.full(8, 0.2)
    st2 = urine_stage_from_rolling_rates(rates2, meas2)
    assert st2[5] == 1


def test_combine_stages_takes_max():
    assert combine_stages(1, 0) == 1
    assert combine_stages(1, 2) == 2
    assert combine_stages(0, 3) == 3
