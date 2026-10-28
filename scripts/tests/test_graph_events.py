"""Stage A: event selection & node dictionary (synthetic only)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.utils.graph_constants import A_MIN_EVENT, STAGE_NAMES
from scripts.utils.graph_events import (
    active_mask_vector,
    build_node_dictionary,
    dense_activation_vector,
    make_synthetic_activation_names,
    parse_activation_name,
    select_active_events,
)


def test_parse_activation_name():
    var, L, k, stage = parse_activation_name("urine_output_ml|L6|k2|B2_6_12")
    assert var == "urine_output_ml"
    assert L == 6 and k == 2 and stage == "B2_6_12"


def test_parse_activation_name_rejects_bad():
    with pytest.raises(ValueError):
        parse_activation_name("not_a_valid_name")


def test_node_dictionary_order_and_unique():
    names = make_synthetic_activation_names(
        vars_=["heart_rate", "urine_output_ml"],
        lengths=[6],
        k=2,
        stages=STAGE_NAMES[:2],
    )
    # 2 vars × 1 L × 2 k × 2 stages = 8
    assert len(names) == 8
    nodes = build_node_dictionary(names)
    assert [n.node_id for n in nodes] == list(range(8))
    assert nodes[0].activation_name == names[0]
    assert nodes[0].L == 6
    # keys must include L
    keys = {(n.var, n.L, n.k, n.stage) for n in nodes}
    assert len(keys) == 8


def test_select_active_events_rank_and_threshold():
    rows = [
        # stay 1, var urine, stage B1: two templates — higher a wins
        dict(
            ICUSTAY_ID=1,
            var="urine_output_ml",
            stage="B1_0_6",
            a=0.9,
            d_stage=0.1,
            L=6,
            k=1,
            is_valid=True,
            node_id=10,
            t_star=1.0,
        ),
        dict(
            ICUSTAY_ID=1,
            var="urine_output_ml",
            stage="B1_0_6",
            a=0.8,
            d_stage=0.05,
            L=6,
            k=0,
            is_valid=True,
            node_id=11,
            t_star=1.0,
        ),
        # below a_min
        dict(
            ICUSTAY_ID=1,
            var="heart_rate",
            stage="B1_0_6",
            a=0.4,
            d_stage=0.2,
            L=6,
            k=0,
            is_valid=True,
            node_id=20,
            t_star=2.0,
        ),
        # invalid
        dict(
            ICUSTAY_ID=1,
            var="heart_rate",
            stage="B2_6_12",
            a=0.95,
            d_stage=0.1,
            L=6,
            k=0,
            is_valid=False,
            node_id=21,
            t_star=7.0,
        ),
    ]
    df = pd.DataFrame(rows)
    active = select_active_events(df, a_min=A_MIN_EVENT)
    assert len(active) == 1
    assert int(active.iloc[0]["node_id"]) == 10
    assert int(active.iloc[0]["event_rank"]) == 1


def test_tie_break_d_stage_then_L_k_node_id():
    # same a → smaller d_stage wins
    rows = [
        dict(
            ICUSTAY_ID=2,
            var="sbp",
            stage="B1_0_6",
            a=0.7,
            d_stage=0.5,
            L=6,
            k=0,
            is_valid=True,
            node_id=1,
            t_star=0.0,
        ),
        dict(
            ICUSTAY_ID=2,
            var="sbp",
            stage="B1_0_6",
            a=0.7,
            d_stage=0.2,
            L=6,
            k=1,
            is_valid=True,
            node_id=2,
            t_star=0.0,
        ),
    ]
    active = select_active_events(pd.DataFrame(rows))
    assert int(active.iloc[0]["node_id"]) == 2


def test_dense_and_active_masks():
    full = pd.DataFrame(
        [
            {"node_id": 0, "a": 0.1},
            {"node_id": 3, "a": 0.9},
        ]
    )
    a = dense_activation_vector(full, n_nodes=5)
    assert a.shape == (5,)
    assert a[0] == pytest.approx(0.1)
    assert a[3] == pytest.approx(0.9)
    assert a[1] == 0.0
    active = pd.DataFrame([{"node_id": 3}])
    m = active_mask_vector(active, n_nodes=5)
    assert m.tolist() == [False, False, False, True, False]
