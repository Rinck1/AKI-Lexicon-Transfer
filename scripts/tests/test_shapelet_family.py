"""Cross-length motif-family alignment (synthetic templates only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.utils.graph_constants import A_MIN_EVENT, STAGE_NAMES
from scripts.utils.graph_events import select_active_events
from scripts.utils.shapelet_constants import SHAPELET_VARS
from scripts.utils.shapelet_family import (
    N_FAMILIES,
    TARGET_LEN,
    align_all_variables,
    align_variable_families,
    apply_family_labels,
    assert_mapping_complete,
    build_motif_node_dictionary,
    build_template_records,
    combine_shape_distance,
    event_display_name,
    family_shape_descriptors,
    hungarian_assign,
    is_low_confidence,
    mapping_frame,
    pairwise_shape_cost,
    remap_activation_long,
    resample_to_length,
    series_slope,
    shape_distance_components,
    shape_normalize,
)


def _lin(start: float, end: float, n: int) -> np.ndarray:
    return np.linspace(start, end, num=n)


def test_resample_preserves_endpoints():
    x = np.array([0.0, 1.0, 4.0, 9.0])
    y = resample_to_length(x, target_len=12)
    assert y.shape == (12,)
    assert y[0] == pytest.approx(0.0)
    assert y[-1] == pytest.approx(9.0)


def test_shape_normalize_start_zero_unit_range():
    x = np.array([10.0, 12.0, 11.0, 16.0])
    y, amp = shape_normalize(x)
    assert y[0] == pytest.approx(0.0)
    assert amp == pytest.approx(6.0)
    assert float(np.max(y) - np.min(y)) == pytest.approx(1.0)


def test_shape_normalize_flat_stays_zero():
    y, amp = shape_normalize(np.array([3.0, 3.0, 3.0, 3.0]))
    assert amp == pytest.approx(0.0)
    assert np.allclose(y, 0.0)


def test_rise_closer_to_rise_than_fall():
    rise = shape_normalize(_lin(0.0, 1.0, 12))[0]
    rise2 = shape_normalize(_lin(5.0, 8.0, 12))[0]
    fall = shape_normalize(_lin(1.0, 0.0, 12))[0]
    d_rr = combine_shape_distance(shape_distance_components(rise, rise2))
    d_rf = combine_shape_distance(shape_distance_components(rise, fall))
    assert d_rr < d_rf


def test_hungarian_identity_on_identical_copies():
    a = [_lin(0, 1, 12), _lin(1, 0, 12), np.sin(np.linspace(0, 2 * np.pi, 12))]
    cost, _ = pairwise_shape_cost(a, a)
    assign = hungarian_assign(cost)
    assert assign.tolist() == [0, 1, 2]


def test_l6_anchors_absorb_matching_l4_l12():
    """Rising / falling / U-shape copies at L=4,12 must map to the matching L=6 family."""
    rise = lambda n: _lin(-1.0, 1.0, n)
    fall = lambda n: _lin(1.0, -1.0, n)
    u = lambda n: np.concatenate([_lin(0.5, -0.8, n // 2), _lin(-0.8, 0.5, n - n // 2)])
    templates = {
        ("mbp", 6, 0): rise(6),
        ("mbp", 6, 1): fall(6),
        ("mbp", 6, 2): u(6),
        ("mbp", 4, 0): fall(4),
        ("mbp", 4, 1): u(4),
        ("mbp", 4, 2): rise(4),
        ("mbp", 12, 0): u(12),
        ("mbp", 12, 1): rise(12),
        ("mbp", 12, 2): fall(12),
    }
    recs = build_template_records(templates)
    assigned = align_variable_families(recs, var="mbp")
    by_lk = {(a.L, a.k): a.motif_family for a in assigned}
    assert by_lk[(6, 0)] == 0
    assert by_lk[(6, 1)] == 1
    assert by_lk[(6, 2)] == 2
    assert by_lk[(4, 2)] == 0  # rise -> family 0
    assert by_lk[(4, 0)] == 1  # fall -> family 1
    assert by_lk[(4, 1)] == 2  # U -> family 2
    assert by_lk[(12, 1)] == 0
    assert by_lk[(12, 2)] == 1
    assert by_lk[(12, 0)] == 2


def test_align_does_not_take_labels():
    import inspect

    sig = inspect.signature(align_variable_families)
    assert "y" not in sig.parameters
    assert "mortality" not in sig.parameters
    sig2 = inspect.signature(align_all_variables)
    assert "y" not in sig2.parameters


def _toy_mapping() -> pd.DataFrame:
    templates = {}
    for v in ("mbp", "urine_output_ml"):
        templates[(v, 6, 0)] = _lin(-1, 1, 6)
        templates[(v, 6, 1)] = _lin(1, -1, 6)
        templates[(v, 6, 2)] = np.sin(np.linspace(0, 2 * np.pi, 6))
        templates[(v, 4, 0)] = _lin(-1, 1, 4)
        templates[(v, 4, 1)] = _lin(1, -1, 4)
        templates[(v, 4, 2)] = np.sin(np.linspace(0, 2 * np.pi, 4))
        templates[(v, 12, 0)] = _lin(-1, 1, 12)
        templates[(v, 12, 1)] = _lin(1, -1, 12)
        templates[(v, 12, 2)] = np.sin(np.linspace(0, 2 * np.pi, 12))
    recs = build_template_records(templates)
    assigned = align_all_variables(recs, vars_=("mbp", "urine_output_ml"))
    mapping = mapping_frame(assigned)
    labels = {
        ("mbp", 0): "恢复上升型",
        ("mbp", 1): "持续下降型",
        ("mbp", 2): "波动回落型",
        ("urine_output_ml", 0): "恢复上升型",
        ("urine_output_ml", 1): "持续下降型",
        ("urine_output_ml", 2): "低位波动型",
    }
    return apply_family_labels(mapping, labels)


def test_mapping_labels_and_anchor_cover():
    mapping = _toy_mapping()
    assert mapping["family_label"].ne("").all()
    for v in ("mbp", "urine_output_ml"):
        anc = mapping.loc[(mapping["var"] == v) & (mapping["is_anchor"])]
        assert set(anc["motif_family"]) == {0, 1, 2}
        assert not anc["family_low_confidence"].any()


def test_same_length_may_share_family():
    """Two rising L=4 templates both go to the rising L=6 family; fall unused at L=4."""
    templates = {
        ("mbp", 6, 0): _lin(-1.0, 1.0, 6),
        ("mbp", 6, 1): _lin(1.0, -1.0, 6),
        ("mbp", 6, 2): np.sin(np.linspace(0, 2 * np.pi, 6)),
        ("mbp", 4, 0): _lin(-1.0, 1.0, 4),
        ("mbp", 4, 1): _lin(-0.8, 1.2, 4),
        ("mbp", 4, 2): _lin(-0.5, 0.9, 4),
        ("mbp", 12, 0): _lin(1.0, -1.0, 12),
        ("mbp", 12, 1): _lin(0.9, -0.8, 12),
        ("mbp", 12, 2): _lin(1.1, -1.2, 12),
    }
    recs = build_template_records(templates)
    assigned = align_variable_families(recs, var="mbp")
    by_lk = {(a.L, a.k): a.motif_family for a in assigned}
    assert {by_lk[(4, 0)], by_lk[(4, 1)], by_lk[(4, 2)]} == {0}
    assert {by_lk[(12, 0)], by_lk[(12, 1)], by_lk[(12, 2)]} == {1}
    assert len({a.motif_family for a in assigned if a.L == 4}) == 1


def test_low_confidence_rules():
    assert is_low_confidence(min_cost=0.0, margin=1.0, is_anchor=True) is False
    assert is_low_confidence(min_cost=0.94, margin=0.2, is_anchor=False) is True
    assert is_low_confidence(min_cost=0.20, margin=0.02, is_anchor=False) is True
    assert is_low_confidence(min_cost=0.20, margin=0.20, is_anchor=False) is False
    assert "低置信" in event_display_name("spo2", 0, "大幅振荡型", "B1_0_6", low_confidence=True)
    assert "大幅振荡型" not in event_display_name(
        "spo2", 0, "大幅振荡型", "B1_0_6", low_confidence=True
    )


def test_motif_dictionary_is_84_when_seven_vars():
    # build a dummy complete mapping with 7 vars × 9 templates
    rows = []
    labels = {0: "A型", 1: "B型", 2: "C型"}
    for v in SHAPELET_VARS:
        for L in (4, 6, 12):
            for k in range(3):
                rows.append(
                    {
                        "var": v,
                        "L": L,
                        "k": k,
                        "motif_family": k,
                        "family_label": labels[k],
                        "is_anchor": L == 6,
                        "cost_to_anchor": 0.0,
                    }
                )
    mapping = pd.DataFrame(rows)
    assert_mapping_complete(mapping)
    nd = build_motif_node_dictionary(mapping)
    assert len(nd) == 84
    keys = set(zip(nd["var"], nd["motif_family"], nd["stage"]))
    assert len(keys) == 84
    assert nd["node_id"].tolist() == list(range(84))
    assert "L" not in nd["activation_name"].iloc[0]


def test_remap_preserves_L_k_and_winner_rule():
    mapping = _toy_mapping()
    nd = build_motif_node_dictionary(
        mapping,
        vars_=("mbp", "urine_output_ml"),
        stages=STAGE_NAMES,
    )
    assert len(nd) == 2 * 3 * 4
    rows = [
        dict(
            ICUSTAY_ID=1,
            split="train",
            var="mbp",
            stage="B1_0_6",
            a=0.91,
            d_stage=0.1,
            L=12,
            k=1,
            is_valid=True,
            node_id=99,
            t_star=3.0,
            HOSPITAL_MORTALITY=1,
        ),
        dict(
            ICUSTAY_ID=1,
            split="train",
            var="mbp",
            stage="B1_0_6",
            a=0.70,
            d_stage=0.2,
            L=4,
            k=0,
            is_valid=True,
            node_id=98,
            t_star=2.0,
            HOSPITAL_MORTALITY=1,
        ),
        dict(
            ICUSTAY_ID=1,
            split="test",
            var="mbp",
            stage="B1_0_6",
            a=0.99,
            d_stage=0.05,
            L=6,
            k=0,
            is_valid=True,
            node_id=97,
            t_star=1.0,
            HOSPITAL_MORTALITY=0,
        ),
    ]
    remapped = remap_activation_long(pd.DataFrame(rows), mapping, nd, drop_test=True)
    assert "test" not in set(remapped["split"])
    assert remapped["L"].tolist() == [12, 4]
    assert remapped["k_orig"].tolist() == [1, 0]
    assert "node_id_lk" in remapped.columns
    assert "family_cost" in remapped.columns
    assert "family_low_confidence" in remapped.columns
    assert "display_name" in remapped.columns
    active = select_active_events(remapped, a_min=A_MIN_EVENT)
    assert len(active) == 1
    win = active.iloc[0]
    assert int(win["L"]) == 12
    assert int(win["k_orig"]) == 1
    assert int(win["motif_family"]) == int(
        mapping.loc[(mapping["var"] == "mbp") & (mapping["L"] == 12) & (mapping["k"] == 1), "motif_family"].iloc[0]
    )


def test_family_descriptors_ignore_mortality_column():
    mapping = _toy_mapping()
    templates = {}
    for r in mapping.itertuples():
        templates[(r.var, int(r.L), int(r.k))] = _lin(0, 1, int(r.L))
    recs = build_template_records(templates)
    assigned = align_all_variables(recs, vars_=("mbp", "urine_output_ml"))
    desc = family_shape_descriptors(recs, assigned)
    assert "HOSPITAL_MORTALITY" not in desc.columns
    assert set(desc["motif_family"]) == {0, 1, 2}


def test_series_slope_sign():
    assert series_slope(_lin(0, 1, 12)) > 0
    assert series_slope(_lin(1, 0, 12)) < 0


def test_empty_stay_survives_in_roster_not_in_winners():
    """Stays with no a>=a_min event must remain in the stay roster."""
    mapping = _toy_mapping()
    nd = build_motif_node_dictionary(
        mapping,
        vars_=("mbp", "urine_output_ml"),
        stages=STAGE_NAMES,
    )
    rows = [
        dict(
            ICUSTAY_ID=1,
            split="train",
            var="mbp",
            stage="B1_0_6",
            a=0.91,
            d_stage=0.1,
            L=6,
            k=0,
            is_valid=True,
            node_id=1,
            t_star=1.0,
            SUBJECT_ID=10,
            HOSPITAL_MORTALITY=0,
        ),
        dict(
            ICUSTAY_ID=2,
            split="train",
            var="mbp",
            stage="B1_0_6",
            a=0.10,
            d_stage=0.8,
            L=6,
            k=0,
            is_valid=True,
            node_id=1,
            t_star=1.0,
            SUBJECT_ID=11,
            HOSPITAL_MORTALITY=0,
        ),
    ]
    long_df = pd.DataFrame(rows)
    remapped = remap_activation_long(long_df, mapping, nd, drop_test=True)
    winners = select_active_events(remapped, a_min=A_MIN_EVENT)
    roster = long_df[["ICUSTAY_ID", "split"]].drop_duplicates()
    assert set(roster["ICUSTAY_ID"]) == {1, 2}
    assert set(winners["ICUSTAY_ID"]) == {1}
