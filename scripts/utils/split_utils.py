"""
Subject-level train/val/test splits and leakage audits.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.utils.constants import RANDOM_SEED, SPLIT_RATIOS


def subject_level_split(
    cohort: pd.DataFrame,
    subject_col: str = "SUBJECT_ID",
    label_col: str = "HOSPITAL_MORTALITY",
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Stratified split by unique SUBJECT_ID.

    Returns DataFrame with columns: SUBJECT_ID, split ∈ {train, val, test}.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    subj = (
        cohort[[subject_col, label_col]]
        .drop_duplicates(subset=[subject_col])
        .copy()
    )
    subjects = subj[subject_col].values
    y = subj[label_col].astype(int).values

    train_r, val_r, test_r = ratios
    # first peel off test
    idx = np.arange(len(subjects))
    idx_tv, idx_te = train_test_split(
        idx, test_size=test_r, random_state=seed, stratify=y
    )
    # then split train/val from remaining
    y_tv = y[idx_tv]
    rel_val = val_r / (train_r + val_r)
    idx_tr, idx_va = train_test_split(
        idx_tv, test_size=rel_val, random_state=seed, stratify=y_tv
    )

    rows = []
    for i in idx_tr:
        rows.append({subject_col: subjects[i], "split": "train"})
    for i in idx_va:
        rows.append({subject_col: subjects[i], "split": "val"})
    for i in idx_te:
        rows.append({subject_col: subjects[i], "split": "test"})
    return pd.DataFrame(rows)


def assert_no_subject_overlap(
    splits: pd.DataFrame,
    subject_col: str = "SUBJECT_ID",
    split_col: str = "split",
) -> Dict[str, int]:
    """
    Raise AssertionError if any SUBJECT_ID appears in more than one split.
    Returns pairwise overlap counts (should all be 0).
    """
    groups = {
        name: set(g[subject_col].tolist())
        for name, g in splits.groupby(split_col)
    }
    pairs = {}
    bad = []
    names = list(groups.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            n = len(groups[a] & groups[b])
            pairs[f"{a}-{b}"] = n
            if n > 0:
                bad.append(f"{a}∩{b}={n}")
    if bad:
        raise AssertionError("Subject leakage across splits: " + "; ".join(bad))
    return pairs


def stay_split_from_subject_split(
    cohort: pd.DataFrame,
    subject_splits: pd.DataFrame,
    subject_col: str = "SUBJECT_ID",
    stay_col: str = "ICUSTAY_ID",
) -> pd.DataFrame:
    """Map each stay to its subject's split."""
    m = subject_splits.set_index(subject_col)["split"]
    out = cohort[[stay_col, subject_col]].drop_duplicates().copy()
    out["split"] = out[subject_col].map(m)
    missing = out["split"].isna().sum()
    if missing:
        raise ValueError(f"{missing} subjects in cohort missing from splits")
    return out


def fit_transform_train_only(
    train: pd.DataFrame,
    other: pd.DataFrame,
    columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Median impute + z-score using train statistics only.

    Returns (train_out, other_out, stats).
    """
    stats = {}
    train_out = train.copy()
    other_out = other.copy()
    for col in columns:
        if col not in train_out.columns:
            continue
        med = float(np.nanmedian(train_out[col].astype(float).values))
        x = train_out[col].astype(float).fillna(med)
        mean = float(x.mean())
        std = float(x.std(ddof=0))
        if std < 1e-8:
            std = 1.0
        stats[col] = {"median": med, "mean": mean, "std": std}
        train_out[col] = (x - mean) / std
        o = other_out[col].astype(float).fillna(med)
        other_out[col] = (o - mean) / std
    return train_out, other_out, stats
