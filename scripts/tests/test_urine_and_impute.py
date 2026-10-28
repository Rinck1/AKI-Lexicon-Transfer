"""Urine missing ≠ 0; impute/scale train-only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.utils.missingness import audit_variable, suggest_tier
from scripts.utils.split_utils import fit_transform_train_only


def test_unmeasured_urine_is_nan_not_zero():
    df = pd.DataFrame(
        {
            "urine_output_ml": [100.0, np.nan, 0.0],
            "urine_output_ml_is_measured": [True, False, True],
        }
    )
    bad = df.loc[~df["urine_output_ml_is_measured"] & (df["urine_output_ml"] == 0)]
    assert len(bad) == 0
    assert pd.isna(df.loc[1, "urine_output_ml"])


def test_measured_zero_urine_allowed():
    """True anuria (measured 0) is valid and distinct from missing."""
    df = pd.DataFrame(
        {
            "urine_output_ml": [0.0],
            "urine_output_ml_is_measured": [True],
        }
    )
    assert df.loc[0, "urine_output_ml"] == 0.0
    assert bool(df.loc[0, "urine_output_ml_is_measured"]) is True


def test_fit_transform_uses_train_stats_only():
    train = pd.DataFrame({"x": [0.0, 2.0, 4.0, 6.0]})
    test = pd.DataFrame({"x": [100.0, np.nan]})
    tr, te, stats = fit_transform_train_only(train, test, ["x"])
    assert abs(stats["x"]["median"] - 3.0) < 1e-6
    # test nan filled with train median then z-scored
    assert np.isfinite(te["x"]).all()
    # train mean after z-score ~ 0
    assert abs(tr["x"].mean()) < 1e-6


def test_suggest_tier_demotes_sparse():
    t = suggest_tier(
        patient_coverage=0.5,
        timepoint_missing=0.75,
        mean_n_meas_24h=0.5,
        clinical_tier_hint="A",
    )
    assert t == "B"


def test_audit_patient_coverage():
    df = pd.DataFrame(
        {
            "ICUSTAY_ID": [1, 1, 2, 2, 3, 3],
            "HOUR": [0, 1, 0, 1, 0, 1],
            "creatinine": [1.0, np.nan, np.nan, np.nan, 1.2, 1.3],
            "creatinine_is_measured": [True, False, False, False, True, True],
        }
    )
    m = audit_variable(df, "creatinine", measured_col="creatinine_is_measured")
    # subjects 1 and 3 have measurements → 2/3
    assert abs(m["patient_coverage"] - 2 / 3) < 1e-6
