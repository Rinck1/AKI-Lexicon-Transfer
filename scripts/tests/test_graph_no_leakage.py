"""Stage A: leakage guards for edge vocabulary (synthetic only)."""

from __future__ import annotations

import pandas as pd

from scripts.utils.graph_builder import (
    apply_edge_vocabulary,
    build_canonical_edges_for_patient,
    build_graphs_for_split,
    collect_train_edge_support,
    freeze_edge_vocabulary,
)
from scripts.utils.graph_constants import EDGE_TYPE_PRECEDES, min_edge_support
from scripts.utils.graph_events import select_active_events


def _act_row(stay, nid, var, stage, a, t, split="train", y=0):
    return dict(
        ICUSTAY_ID=stay,
        SUBJECT_ID=stay,
        split=split,
        HOSPITAL_MORTALITY=y,
        node_id=nid,
        var=var,
        L=6,
        k=0,
        stage=stage,
        stage_idx=0,
        a=a,
        d_stage=0.1,
        t_star=t,
        is_valid=True,
    )


def test_min_edge_support_formula():
    assert min_edge_support(1000) == 20  # max(20, ceil(5)) = 20
    assert min_edge_support(10000) == 50  # ceil(0.005*10000)=50


def test_edge_vocab_from_train_only_val_cannot_introduce_types():
    """
    Train defines vocab; a val-only edge type must be dropped, not added.
    """
    train_edge_lists = []
    # 25 train patients share PRECEDES 1->2
    for stay in range(25):
        df = pd.DataFrame(
            [
                _act_row(stay, 1, "hr", "B1_0_6", 0.9, 0.0),
                _act_row(stay, 2, "urine", "B2_6_12", 0.85, 5.0),
            ]
        )
        active = select_active_events(df)
        train_edge_lists.append(build_canonical_edges_for_patient(active))

    support = collect_train_edge_support(train_edge_lists)
    vocab = freeze_edge_vocabulary(support, n_train_stays=25)
    assert (1, 2, EDGE_TYPE_PRECEDES) in vocab

    # Val has train edge + novel edge 8->9
    val_df = pd.DataFrame(
        [
            _act_row(999, 1, "hr", "B1_0_6", 0.9, 0.0, split="val"),
            _act_row(999, 2, "urine", "B2_6_12", 0.85, 5.0, split="val"),
            _act_row(999, 8, "spo2", "B1_0_6", 0.9, 0.0, split="val"),
            _act_row(999, 9, "mbp", "B2_6_12", 0.85, 5.0, split="val"),
        ]
    )
    val_active = select_active_events(val_df)
    raw = build_canonical_edges_for_patient(val_active)
    kept, dropped = apply_edge_vocabulary(raw, vocab)
    keys = {e.key() for e in kept}
    assert (1, 2, EDGE_TYPE_PRECEDES) in keys
    assert (8, 9, EDGE_TYPE_PRECEDES) not in keys
    assert dropped >= 1
    # vocab size unchanged by val
    assert (8, 9, EDGE_TYPE_PRECEDES) not in vocab


def test_build_graphs_reports_dropped_unseen_on_val():
    train_lists = []
    for stay in range(25):
        df = pd.DataFrame(
            [
                _act_row(stay, 1, "hr", "B1_0_6", 0.9, 0.0),
                _act_row(stay, 2, "urine", "B2_6_12", 0.85, 5.0),
            ]
        )
        train_lists.append(
            build_canonical_edges_for_patient(select_active_events(df))
        )
    vocab = freeze_edge_vocabulary(
        collect_train_edge_support(train_lists), n_train_stays=25
    )

    val_long = pd.DataFrame(
        [
            _act_row(50, 1, "hr", "B1_0_6", 0.9, 0.0, split="val"),
            _act_row(50, 2, "urine", "B2_6_12", 0.85, 5.0, split="val"),
            _act_row(50, 8, "spo2", "B1_0_6", 0.9, 0.0, split="val"),
            _act_row(50, 9, "mbp", "B2_6_12", 0.85, 5.0, split="val"),
        ]
    )
    val_active = select_active_events(val_long)
    graphs = build_graphs_for_split(
        val_long,
        val_active,
        edge_vocab=vocab,
        n_train_stays=25,
        apply_vocab=True,
    )
    assert len(graphs) == 1
    assert graphs[0].dropped_unseen_edge >= 1


def test_no_outcome_in_edge_attr_contract():
    """Edge attr is 5 continuous/binary features — no death frequency slots."""
    from scripts.utils.graph_constants import N_EDGE_ATTR

    assert N_EDGE_ATTR == 5
    df = pd.DataFrame(
        [
            _act_row(1, 1, "hr", "B1_0_6", 0.9, 0.0),
            _act_row(1, 2, "urine", "B2_6_12", 0.8, 5.0),
        ]
    )
    edges = build_canonical_edges_for_patient(select_active_events(df))
    # attach fake support without using y
    for e in edges:
        e.train_support = 25
    from scripts.utils.graph_builder import edge_attr_matrix

    attr = edge_attr_matrix(edges, n_train_stays=100)
    assert attr.shape[1] == 5
