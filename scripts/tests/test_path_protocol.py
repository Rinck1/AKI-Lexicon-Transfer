"""docs/20 path protocol: discretization, support, test lock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.utils.graph_constants import STAGE_NAMES
from scripts.utils.graph_sparse import sparse_feature_names
from scripts.utils.path_atoms import (
    DIR_DOWN,
    DIR_UP,
    fit_lab_thresholds,
    lab_atoms_from_raw,
    motif_atoms_from_events,
)
from scripts.utils.path_mine import enumerate_stay_paths, frequent_paths
from scripts.utils.path_protocol import (
    fold_stage,
    is_readable,
    min_path_support,
    motif_atom_name,
    refuse_test_path,
    refuse_test_split,
)


def test_fold_stage_and_readable():
    assert fold_stage("B2_6_12") == "B2"
    assert motif_atom_name("mbp", "回落反跳型", "B1_0_6") == "mbp|回落反跳型|B1"
    assert is_readable(("mbp|回落反跳型|B1", "creatinine|B3|升高"))
    assert not is_readable(("heart_rate|高峰回落型|B1", "spo2|大幅振荡型|B2"))


def test_min_support_matches_edge_rule():
    assert min_path_support(7610) == 39


def test_refuse_test():
    with pytest.raises(SystemExit):
        refuse_test_split("test")
    with pytest.raises(SystemExit):
        refuse_test_path("data/processed/paths/path_test_pack")


def test_freeze_hash_and_holdout_lock(tmp_path):
    from scripts.utils.path_protocol import refuse_if_holdout_done, sha256_canonical

    a = sha256_canonical({"selected": ["x"], "seed": 42})
    b = sha256_canonical({"seed": 42, "selected": ["x"]})
    assert a == b
    assert a != sha256_canonical({"selected": ["y"], "seed": 42})
    lock = tmp_path / "path_holdout_reproduce.json"
    refuse_if_holdout_done(lock)
    lock.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        refuse_if_holdout_done(lock)


def test_lab_thresholds_ignore_val_and_missing():
    names = sparse_feature_names()
    n = 80
    raw = np.full((n, len(names)), np.nan)
    name_i = {n_: i for i, n_ in enumerate(names)}
    # first 50 train: creatinine B2 large positive delta; B1 measured last=1
    for r in range(50):
        raw[r, name_i["creatinine|B1_0_6|is_measured"]] = 1.0
        raw[r, name_i["creatinine|B1_0_6|last"]] = 1.0
        raw[r, name_i["creatinine|B2_6_12|is_measured"]] = 1.0
        raw[r, name_i["creatinine|B2_6_12|delta_from_previous_stage_last"]] = (
            1.0 if r < 25 else 3.0
        )
        raw[r, name_i["creatinine|B2_6_12|last"]] = 3.0
    # val rows 50-79: huge deltas must not move train medians
    for r in range(50, 80):
        raw[r, name_i["creatinine|B2_6_12|is_measured"]] = 1.0
        raw[r, name_i["creatinine|B2_6_12|delta_from_previous_stage_last"]] = 99.0
    stays = np.arange(n)
    thr = fit_lab_thresholds(raw, stays, train_ids=list(range(50)), names=names)
    assert thr["by_var"]["creatinine"]["pos_delta_median"] == pytest.approx(2.0)
    meta = pd.DataFrame(
        {
            "ICUSTAY_ID": stays,
            "SUBJECT_ID": stays,
            "split": ["train"] * 50 + ["val"] * 30,
            "HOSPITAL_MORTALITY": [0] * n,
        }
    )
    # unmeasured stay 0 overwrite meas=0 → no atom
    raw[0, name_i["creatinine|B2_6_12|is_measured"]] = 0.0
    atoms = lab_atoms_from_raw(raw[:50], meta.iloc[:50], thr, names=names)
    cr = atoms[atoms["var"] == "creatinine"]
    assert (cr["direction"] == DIR_UP).any()
    assert 0 not in set(cr.loc[cr["stage"] == "B2_6_12", "ICUSTAY_ID"])


def test_enumerate_and_apriori_support():
    stay_atoms = {
        1: [(0, "mbp|回落反跳型|B1", False), (1, "urine_output_ml|进行性下降型|B2", False)],
        2: [(0, "mbp|回落反跳型|B1", False), (1, "urine_output_ml|进行性下降型|B2", False)],
        3: [(0, "heart_rate|高峰回落型|B1", False)],
    }
    paths = enumerate_stay_paths(stay_atoms[1])
    assert ("mbp|回落反跳型|B1", "urine_output_ml|进行性下降型|B2") in paths
    freq = frequent_paths(stay_atoms, [1, 2, 3], min_support=2)
    assert ("mbp|回落反跳型|B1", "urine_output_ml|进行性下降型|B2") in freq
    assert len(freq[("mbp|回落反跳型|B1", "urine_output_ml|进行性下降型|B2")]) == 2
    from scripts.utils.path_mine import extend_from_parents

    stay_atoms[1].append((2, "lactate|B3|升高", False))
    stay_atoms[2].append((2, "lactate|B3|升高", False))
    grown = extend_from_parents(
        stay_atoms,
        [1, 2, 3],
        {("mbp|回落反跳型|B1", "urine_output_ml|进行性下降型|B2")},
        min_support=2,
    )
    trip = ("mbp|回落反跳型|B1", "urine_output_ml|进行性下降型|B2", "lactate|B3|升高")
    assert trip in grown
    assert len(grown[trip]) == 2


def test_motif_atoms_require_activation():
    ev = pd.DataFrame(
        {
            "ICUSTAY_ID": [1, 1],
            "SUBJECT_ID": [1, 1],
            "split": ["train", "train"],
            "HOSPITAL_MORTALITY": [0, 0],
            "var": ["mbp", "mbp"],
            "family_label": ["回落反跳型", "回落反跳型"],
            "stage": ["B1_0_6", "B2_6_12"],
            "stage_idx": [0, 1],
            "a": [0.9, 0.2],
            "family_low_confidence": [False, False],
            "is_active": [True, True],
        }
    )
    atoms = motif_atoms_from_events(ev, a_min=0.5)
    assert len(atoms) == 1
    assert atoms.iloc[0]["stage_idx"] == 0
