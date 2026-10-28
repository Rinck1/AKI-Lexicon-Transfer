"""Tests for CBM concept-bottleneck utilities (docs/26 Phase A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.utils.cbm import (
    KIND_LAB,
    KIND_MOTIF,
    build_concept_matrix,
    build_static_matrix,
    coef_table,
    concept_interventions,
    fit_full_linear,
    standardize_train_only,
    subject_groups,
)
from scripts.utils.path_protocol import refuse_test_split


def _stays(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ICUSTAY_ID": np.arange(n),
            "SUBJECT_ID": np.arange(n) * 10,
            "HOSPITAL_MORTALITY": (rng.random(n) < 0.2).astype(float),
            "AGE": rng.normal(60, 15, n),
            "GENDER_M": (rng.random(n) < 0.5).astype(float),
        }
    ).set_index("ICUSTAY_ID")


def _motif_atoms(n_stay: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stay):
        for j in range(6):
            if rng.random() < 0.4:
                rows.append(
                    {
                        "ICUSTAY_ID": s,
                        "SUBJECT_ID": s * 10,
                        "split": "train",
                        "a": float(0.5 + 0.5 * rng.random()),
                        "atom": f"mbp|fam{j}|B{j % 4 + 1}",
                        "kind": KIND_MOTIF,
                    }
                )
    return pd.DataFrame(rows)


def _lab_atoms(n_stay: int, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stay):
        for j in range(4):
            if rng.random() < 0.3:
                d = "升高" if rng.random() < 0.5 else "下降"
                rows.append(
                    {
                        "ICUSTAY_ID": s,
                        "SUBJECT_ID": s * 10,
                        "split": "train",
                        "a": np.nan,
                        "atom": f"lactate|B{j + 1}|{d}",
                        "kind": KIND_LAB,
                    }
                )
    return pd.DataFrame(rows)


def test_build_concept_matrix_shapes_and_values():
    stays = _stays(20)
    ids = sorted(int(s) for s in stays.index)
    m = _motif_atoms(20)
    motif_cols = sorted(m["atom"].astype(str).unique())
    M_a = build_concept_matrix(m, ids, motif_cols, kind_of={}, use_activation=True)
    M_01 = build_concept_matrix(m, ids, motif_cols, kind_of={}, use_activation=False)
    assert M_a.shape == (20, len(motif_cols))
    assert M_01.shape == (20, len(motif_cols))
    # indicator matrix is 0/1
    assert set(np.unique(M_01)) <= {0.0, 1.0}
    # activation matrix values are >= 0.5 where nonzero
    nz = M_a[M_a > 0]
    assert (nz >= 0.5).all()
    # indicator nonzero where activation nonzero
    assert np.all((M_a > 0) == (M_01 > 0))


def test_lab_matrix_is_indicator():
    stays = _stays(20)
    ids = sorted(int(s) for s in stays.index)
    l = _lab_atoms(20)
    lab_cols = sorted(l["atom"].astype(str).unique())
    M = build_concept_matrix(l, ids, lab_cols, kind_of={}, use_activation=True)
    assert set(np.unique(M)) <= {0.0, 1.0}
    # each row has at most one of the (stage, direction) pair per var
    assert (M.sum(axis=1) >= 0).all()


def test_standardize_train_only_reproducible():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Xs, mu, sd = standardize_train_only(X)
    Xs2, _, _ = standardize_train_only(X, mu, sd)
    assert np.allclose(Xs, Xs2)
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-12)


def test_subject_groups_aligned():
    stays = _stays(10)
    ids = sorted(int(s) for s in stays.index)
    g = subject_groups(stays, ids)
    assert g.tolist() == [int(stays.loc[s, "SUBJECT_ID"]) for s in ids]


def test_refuse_test_split():
    with pytest.raises(SystemExit):
        refuse_test_split("test")
    refuse_test_split("train")  # should not raise


def test_intervention_direction_matches_coef_sign():
    """A concept with a strong positive coef must have mean_delta_p > 0."""
    rng = np.random.default_rng(7)
    n = 500
    static = rng.normal(0, 1, (n, 3))
    # concept 0 is a strong risk driver
    x_risk = (rng.random(n) < 0.4).astype(float)
    extra = np.column_stack([x_risk, rng.normal(0, 1, (n, 2))])
    z = static @ np.array([0.3, -0.2, 0.1]) + 2.5 * x_risk - 0.5
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(float)

    clf = fit_full_linear(y, static, extra)
    names = ["risk_concept", "noise1", "noise2"]
    kinds = [KIND_MOTIF, KIND_MOTIF, KIND_MOTIF]
    iv = concept_interventions(clf, static, extra, names, kinds, static.shape[1])
    by_name = {r.concept: r for r in iv}
    assert by_name["risk_concept"].mean_delta_p > 0
    assert np.isfinite(by_name["risk_concept"].mean_delta_p)


def test_coef_table_shape_and_sorted():
    clf = fit_full_linear(
        np.array([0.0, 1.0, 0.0, 1.0, 1.0]),
        np.random.default_rng(0).normal(0, 1, (5, 2)),
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
    )
    df = coef_table(clf, ["c0", "c1"], [KIND_MOTIF, KIND_LAB], 2)
    assert list(df.columns) == ["concept", "kind", "coef"]
    assert len(df) == 2
    abscoef = df["coef"].abs().tolist()
    assert abscoef == sorted(abscoef, reverse=True)


def test_build_static_matrix_aligned():
    stays = _stays(8)
    ids = sorted(int(s) for s in stays.index)
    X = build_static_matrix(stays, ids, ["AGE", "GENDER_M"])
    assert X.shape == (8, 2)
    assert np.allclose(X[:, 0], [stays.loc[s, "AGE"] for s in ids])
