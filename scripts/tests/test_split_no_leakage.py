"""Subject-level split: zero patient overlap across sets."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.utils.split_utils import (
    assert_no_subject_overlap,
    subject_level_split,
)


def _fake_cohort(n_subjects: int = 100, stays_per: int = 2) -> pd.DataFrame:
    rows = []
    for sid in range(n_subjects):
        mort = int(sid % 5 == 0)
        for k in range(stays_per):
            rows.append(
                {
                    "SUBJECT_ID": sid,
                    "ICUSTAY_ID": sid * 10 + k,
                    "HOSPITAL_MORTALITY": mort,
                }
            )
    return pd.DataFrame(rows)


def test_subject_split_no_overlap():
    cohort = _fake_cohort()
    splits = subject_level_split(cohort)
    pairs = assert_no_subject_overlap(splits)
    assert all(v == 0 for v in pairs.values())
    # multi-stay subjects stay in one split
    stay_split = cohort.merge(splits, on="SUBJECT_ID")
    nuniq = stay_split.groupby("SUBJECT_ID")["split"].nunique()
    assert (nuniq == 1).all()


def test_detects_leakage():
    bad = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 2],
            "split": ["train", "test", "val"],
        }
    )
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_subject_overlap(bad)


def test_split_ratios_approximately():
    cohort = _fake_cohort(200, stays_per=1)
    splits = subject_level_split(cohort)
    n = splits["SUBJECT_ID"].nunique()
    train_n = (splits["split"] == "train").sum()
    assert 0.55 < train_n / n < 0.65
