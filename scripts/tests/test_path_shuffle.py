"""Time-shuffle negative control: preserve bags, destroy stage–concept pairing."""

from __future__ import annotations

import pandas as pd

from scripts.utils.path_mine import enumerate_stay_paths
from scripts.utils.path_protocol import PATH_SEED, parse_path_key
from scripts.utils.path_shuffle import (
    cooccur_indicator_matrix,
    identity_bag_signature,
    n_stage_concept_pairs_changed,
    path_identities,
    shuffle_train_atom_stages,
    stage_occupancy_signature,
    strip_stage_token,
)


def _toy_atoms() -> pd.DataFrame:
    rows = [
        # stay 1: two stages — order is the signal
        dict(
            ICUSTAY_ID=1,
            SUBJECT_ID=1,
            split="train",
            HOSPITAL_MORTALITY=1.0,
            var="lactate",
            family_label="",
            stage="B1_0_6",
            stage_idx=0,
            a=0.9,
            family_low_confidence=False,
            atom="lactate|B1|升高",
            kind="lab",
            direction="升高",
            stage_fold="B1",
        ),
        dict(
            ICUSTAY_ID=1,
            SUBJECT_ID=1,
            split="train",
            HOSPITAL_MORTALITY=1.0,
            var="urine_output_ml",
            family_label="低位缓升型",
            stage="B3_12_24",
            stage_idx=2,
            a=0.8,
            family_low_confidence=False,
            atom="urine_output_ml|低位缓升型|B3",
            kind="motif",
            direction="",
            stage_fold="B3",
        ),
        # stay 2: singleton — permutation is identity
        dict(
            ICUSTAY_ID=2,
            SUBJECT_ID=2,
            split="train",
            HOSPITAL_MORTALITY=0.0,
            var="mbp",
            family_label="缓降持平型",
            stage="B2_6_12",
            stage_idx=1,
            a=0.7,
            family_low_confidence=False,
            atom="mbp|缓降持平型|B2",
            kind="motif",
            direction="",
            stage_fold="B2",
        ),
        # val must be dropped
        dict(
            ICUSTAY_ID=9,
            SUBJECT_ID=9,
            split="val",
            HOSPITAL_MORTALITY=0.0,
            var="lactate",
            family_label="",
            stage="B1_0_6",
            stage_idx=0,
            a=0.9,
            family_low_confidence=False,
            atom="lactate|B1|升高",
            kind="lab",
            direction="升高",
            stage_fold="B1",
        ),
    ]
    return pd.DataFrame(rows)


def test_strip_stage_token_motif_and_lab():
    assert strip_stage_token("lactate|B1|升高") == "lactate|升高"
    assert strip_stage_token("urine_output_ml|低位缓升型|B3") == "urine_output_ml|低位缓升型"
    assert path_identities(("lactate|B1|升高", "urine_output_ml|低位缓升型|B3")) == (
        "lactate|升高",
        "urine_output_ml|低位缓升型",
    )


def test_shuffle_drops_val_preserves_bag_and_occupancy():
    raw = _toy_atoms()
    sh = shuffle_train_atom_stages(raw, seed=PATH_SEED)
    assert set(sh["split"].unique()) == {"train"}
    assert 9 not in set(sh["ICUSTAY_ID"])
    orig_train = raw[raw["split"] == "train"]
    assert stage_occupancy_signature(orig_train).equals(stage_occupancy_signature(sh))
    assert identity_bag_signature(orig_train).equals(identity_bag_signature(sh))
    # stay 1 has two distinct stages so seed 42 should swap them
    s1 = sh[sh["ICUSTAY_ID"] == 1].sort_values("var")
    assert n_stage_concept_pairs_changed(orig_train, sh) >= 1
    # rebuilt names agree with stage_idx
    for rec in sh.itertuples(index=False):
        assert str(rec.stage_idx) in {"0", "1", "2", "3"}
        if rec.kind == "lab":
            assert rec.atom.endswith(rec.direction) or rec.direction in rec.atom
        assert rec.stage_fold in rec.atom


def test_shuffle_destroys_stage_typed_path_keeps_cooccurrence():
    raw = _toy_atoms()
    orig = raw[raw["split"] == "train"]
    path = parse_path_key("lactate|B1|升高 → urine_output_ml|低位缓升型|B3")
    orig_paths = enumerate_stay_paths(
        [(int(r.stage_idx), str(r.atom), False) for r in orig[orig["ICUSTAY_ID"] == 1].itertuples(index=False)]
    )
    assert path in orig_paths
    destroyed = False
    kept_cooccur = False
    from scripts.utils.path_shuffle import identities_by_stay

    for seed in range(32):
        sh = shuffle_train_atom_stages(orig, seed=seed)
        sh_paths = enumerate_stay_paths(
            [(int(r.stage_idx), str(r.atom), False) for r in sh[sh["ICUSTAY_ID"] == 1].itertuples(index=False)]
        )
        idents = identities_by_stay(sh)
        ind = cooccur_indicator_matrix([1, 2], idents, [path])
        assert ind[0, 0] == 1.0
        assert ind[1, 0] == 0.0
        kept_cooccur = True
        if path not in sh_paths:
            destroyed = True
            break
    assert kept_cooccur
    assert destroyed


def test_shuffle_deterministic():
    raw = _toy_atoms()
    a = shuffle_train_atom_stages(raw, seed=7)
    b = shuffle_train_atom_stages(raw, seed=7)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
