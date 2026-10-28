"""Stage A: CO_OCCUR / PRECEDES edge rules (synthetic only)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.utils.graph_builder import (
    apply_edge_vocabulary,
    build_canonical_edges_for_patient,
    build_patient_graph,
    collect_train_edge_support,
    edge_attr_matrix,
    freeze_edge_vocabulary,
)
from scripts.utils.graph_constants import (
    EDGE_TYPE_CO_OCCUR,
    EDGE_TYPE_PRECEDES,
    MAX_CANONICAL_EDGES,
    min_edge_support,
)


def _ev(node_id, var, stage, a, t_star):
    return dict(
        ICUSTAY_ID=1,
        SUBJECT_ID=1,
        split="train",
        HOSPITAL_MORTALITY=0,
        node_id=node_id,
        var=var,
        stage=stage,
        a=a,
        t_star=t_star,
        L=6,
        k=0,
        is_valid=True,
        d_stage=0.1,
        event_rank=1,
        active_mask=True,
    )


def test_co_occur_within_2h_canonical_order():
    # |dt|=1.5 -> CO_OCCUR; src < dst
    df = pd.DataFrame(
        [
            _ev(5, "hr", "B1_0_6", 0.9, 1.0),
            _ev(2, "urine", "B1_0_6", 0.8, 2.5),
        ]
    )
    edges = build_canonical_edges_for_patient(df)
    co = [e for e in edges if e.edge_type == EDGE_TYPE_CO_OCCUR]
    assert len(co) == 1
    assert co[0].src == 2 and co[0].dst == 5
    assert co[0].abs_delta_t == pytest.approx(1.5)
    pr = [e for e in edges if e.edge_type == EDGE_TYPE_PRECEDES]
    assert pr == []


def test_precedes_between_2_and_12h():
    # dt=5 -> PRECEDES 2->7
    df = pd.DataFrame(
        [
            _ev(2, "hr", "B1_0_6", 0.9, 1.0),
            _ev(7, "urine", "B2_6_12", 0.85, 6.0),
        ]
    )
    edges = build_canonical_edges_for_patient(df)
    pr = [e for e in edges if e.edge_type == EDGE_TYPE_PRECEDES]
    assert len(pr) == 1
    assert pr[0].src == 2 and pr[0].dst == 7
    assert pr[0].abs_delta_t == pytest.approx(5.0)
    assert not any(e.edge_type == EDGE_TYPE_CO_OCCUR for e in edges)


def test_no_self_loop_and_no_all_pairs_beyond_window():
    df = pd.DataFrame(
        [
            _ev(1, "hr", "B1_0_6", 0.9, 0.0),
            _ev(2, "urine", "B4_24_48", 0.9, 30.0),  # dt=30 > 12
        ]
    )
    edges = build_canonical_edges_for_patient(df)
    assert edges == []


def test_precedes_max_two_outgoing():
    # source 0, three later events within window
    rows = [_ev(0, "hr", "B1_0_6", 1.0, 0.0)]
    for i, (nid, t) in enumerate([(10, 3.0), (11, 4.0), (12, 5.0)], start=0):
        rows.append(_ev(nid, "urine", "B2_6_12", 0.9 - 0.01 * i, t))
    edges = build_canonical_edges_for_patient(pd.DataFrame(rows))
    pr = [e for e in edges if e.edge_type == EDGE_TYPE_PRECEDES and e.src == 0]
    assert len(pr) == 2
    # nearest first: dt 3 then 4 -> nodes 10, 11
    assert [e.dst for e in pr] == [10, 11]


def test_co_occur_degree_cap_four():
    # center node 0 with 6 others at identical t_star — a_prod ranks center
    # edges first (1.0*0.9 > 0.9*0.9), so node 0 saturates at 4 neighbors
    # before peripheral-peripheral pairs consume degree budget.
    rows = [_ev(0, "hr", "B1_0_6", 1.0, 1.0)]
    for i in range(1, 7):
        rows.append(_ev(i, f"v{i}", "B1_0_6", 0.9, 1.0))
    edges = build_canonical_edges_for_patient(pd.DataFrame(rows))
    co = [e for e in edges if e.edge_type == EDGE_TYPE_CO_OCCUR]
    # each edge counts for both ends; node 0 degree should be 4
    deg0 = sum(1 for e in co if e.src == 0 or e.dst == 0)
    assert deg0 == 4
    assert all(
        (e.src == 0 and e.dst in {1, 2, 3, 4})
        or (e.dst == 0 and e.src in {1, 2, 3, 4})
        for e in co
        if e.src == 0 or e.dst == 0
    )


def test_edge_vocab_support_and_val_drop():
    # train patients: edge (1,2,PRECEDES) supported by 25 patients
    train_lists = []
    for _ in range(25):
        e = build_canonical_edges_for_patient(
            pd.DataFrame(
                [
                    _ev(1, "hr", "B1_0_6", 0.9, 0.0),
                    _ev(2, "urine", "B2_6_12", 0.8, 5.0),
                ]
            )
        )
        train_lists.append(e)
    # rare edge only 1 patient
    rare = build_canonical_edges_for_patient(
        pd.DataFrame(
            [
                _ev(3, "hr", "B1_0_6", 0.9, 0.0),
                _ev(4, "spo2", "B2_6_12", 0.8, 5.0),
            ]
        )
    )
    train_lists.append(rare)

    support = collect_train_edge_support(train_lists)
    n_train = 26
    thr = min_edge_support(n_train)
    assert thr == 20
    vocab = freeze_edge_vocabulary(support, n_train)
    assert (1, 2, EDGE_TYPE_PRECEDES) in vocab
    assert (3, 4, EDGE_TYPE_PRECEDES) not in vocab

    # val patient has both
    val_edges = build_canonical_edges_for_patient(
        pd.DataFrame(
            [
                _ev(1, "hr", "B1_0_6", 0.9, 0.0),
                _ev(2, "urine", "B2_6_12", 0.8, 5.0),
                _ev(3, "hr", "B1_0_6", 0.9, 0.0),
                _ev(4, "spo2", "B2_6_12", 0.8, 5.0),
            ]
        )
    )
    kept, dropped = apply_edge_vocabulary(val_edges, vocab)
    assert dropped >= 1
    assert all(e.key() in vocab for e in kept)
    # support attached
    assert all(e.train_support >= thr for e in kept)


def test_edge_attr_shape_and_no_outcome_fields():
    edges = build_canonical_edges_for_patient(
        pd.DataFrame(
            [
                _ev(1, "hr", "B1_0_6", 0.9, 0.0),
                _ev(2, "urine", "B2_6_12", 0.8, 5.0),
            ]
        )
    )
    edges, _ = apply_edge_vocabulary(
        edges, {(1, 2, EDGE_TYPE_PRECEDES): 30}
    )
    attr = edge_attr_matrix(edges, n_train_stays=100)
    assert attr.shape == (1, 5)
    assert attr[0, 0] == pytest.approx(5.0 / 12.0)


def test_empty_patient_graph_retained():
    g = build_patient_graph(
        stay_id=99,
        subject_id=9,
        split="val",
        y=1.0,
        active_events=pd.DataFrame(),
        edge_vocab={},
        n_train_stays=100,
        apply_vocab=True,
    )
    assert g.is_empty
    assert g.n_edges == 0
    assert g.stay_id == 99


def test_max_edges_hard_fail():
    # force many events same time → many co-occur; still under 128 for 28 nodes
    # construct artificially by lowering max_edges
    rows = [_ev(i, f"v{i}", "B1_0_6", 0.9, 1.0) for i in range(10)]
    with pytest.raises(ValueError, match="MAX_CANONICAL_EDGES"):
        build_canonical_edges_for_patient(pd.DataFrame(rows), max_edges=1)
