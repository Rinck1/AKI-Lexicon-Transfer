"""Tests for frozen-GNN interpretability perturbations."""

from __future__ import annotations

import torch

from scripts.utils.graph_constants import EDGE_TYPE_PRECEDES
from scripts.utils.graph_data import collate_graph_batch, make_synthetic_patient
from scripts.utils.graph_interpretability import (
    attention_topk_mask,
    enumerate_precedes_paths_batch,
    enumerate_simple_k_hop_precedes,
    extract_precedes_paths,
    fold_path_pattern,
    mask_jaccard,
    matched_random_edge_mask,
    path_interaction,
    perturb_edges,
    perturb_nodes,
    random_valid_mask,
    select_random_indices,
    select_top_positive_indices,
    topk_jaccard,
    union_path_edge_mask,
)


def _batch() -> dict[str, torch.Tensor]:
    patient = make_synthetic_patient(
        n_nodes=6,
        stay_id=17,
        subject_id=23,
        y=1.0,
        active_ids=[0, 1, 2, 3],
        edges=[
            (0, 1, EDGE_TYPE_PRECEDES),
            (1, 2, EDGE_TYPE_PRECEDES),
            (2, 3, EDGE_TYPE_PRECEDES),
        ],
        a_values={0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6},
        times={0: 1.0, 1: 6.0, 2: 12.0, 3: 18.0},
    )
    return collate_graph_batch([patient])


def test_attention_topk_mask_uses_only_valid_entries():
    scores = torch.tensor([[0.2, 0.9, 0.8, 99.0]])
    valid = torch.tensor([[True, True, True, False]])
    selected = attention_topk_mask(scores, valid, fraction=0.5)
    assert selected.tolist() == [[False, True, True, False]]


def test_random_valid_mask_matches_topk_count_and_is_deterministic():
    valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    keys = torch.tensor([101, 202])
    a = random_valid_mask(valid, fraction=0.5, keys=keys, seed=42, repeat=3)
    b = random_valid_mask(valid, fraction=0.5, keys=keys, seed=42, repeat=3)
    assert torch.equal(a, b)
    assert (a & ~valid).sum() == 0
    assert a.sum(dim=1).tolist() == [2, 1]


def test_matched_random_preserves_count_and_type_mix():
    valid = torch.tensor([[True] * 8 + [False] * 4])
    etype = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]])
    selected = torch.zeros_like(valid)
    selected[0, 0] = True
    selected[0, 1] = True
    selected[0, 4] = True  # 2 CO + 1 PRE
    keys = torch.tensor([7])
    a = matched_random_edge_mask(
        valid, selected, etype, keys=keys, seed=42, repeat=0
    )
    b = matched_random_edge_mask(
        valid, selected, etype, keys=keys, seed=42, repeat=0
    )
    assert torch.equal(a, b)
    assert int(a.sum().item()) == 3
    assert int((a[0, :8] & (etype[0, :8] == 0)).sum().item()) == 2
    assert int((a[0, :8] & (etype[0, :8] == 1)).sum().item()) == 1
    empty_both = mask_jaccard(
        torch.zeros_like(valid), torch.zeros_like(valid), valid
    )
    assert torch.isnan(empty_both).all()


def test_node_perturbation_removes_incident_edges_but_keeps_dense_branch():
    batch = _batch()
    selected = torch.zeros_like(batch["active_mask"])
    selected[0, 1] = True
    changed = perturb_nodes(batch, selected)

    assert not bool(changed["active_mask"][0, 1])
    assert torch.equal(changed["a_dense"], batch["a_dense"])
    # Edges 0->1 and 1->2 disappear; 2->3 remains.
    assert changed["edge_mask"][0, :3].tolist() == [False, False, True]
    # Original batch is not mutated.
    assert batch["active_mask"][0, 1]
    assert batch["edge_mask"][0, :3].all()


def test_edge_perturbation_only_changes_edge_mask():
    batch = _batch()
    selected = torch.zeros_like(batch["edge_mask"])
    selected[0, 1] = True
    changed = perturb_edges(batch, selected)
    assert changed["edge_mask"][0, :3].tolist() == [True, False, True]
    assert torch.equal(changed["active_mask"], batch["active_mask"])
    assert torch.equal(changed["a_dense"], batch["a_dense"])


def test_topk_jaccard_handles_empty_and_nonempty_rows():
    base_scores = torch.tensor([[0.9, 0.8, 0.1], [0.0, 0.0, 0.0]])
    new_scores = torch.tensor([[0.7, 0.1, 0.8], [0.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, True], [False, False, False]])
    out = topk_jaccard(base_scores, new_scores, valid, fraction=2 / 3)
    assert out[0] == 1 / 3
    assert torch.isnan(out[1])


def test_extract_precedes_paths_keeps_connected_length2_chain():
    edges = [
        {
            "stay_id": 1,
            "y": 1.0,
            "src_id": 0,
            "dst_id": 1,
            "src_name": "a",
            "dst_name": "b",
            "edge_type": "PRECEDES",
            "contrib": 0.4,
        },
        {
            "stay_id": 1,
            "y": 1.0,
            "src_id": 1,
            "dst_id": 2,
            "src_name": "b",
            "dst_name": "c",
            "edge_type": "PRECEDES",
            "contrib": 0.3,
        },
        {
            "stay_id": 1,
            "y": 1.0,
            "src_id": 3,
            "dst_id": 4,
            "src_name": "x",
            "dst_name": "y",
            "edge_type": "CO_OCCUR",
            "contrib": 9.0,
        },
    ]
    paths = extract_precedes_paths(edges, top_k_per_patient=3)
    assert len(paths) == 1
    assert paths[0]["src_name"] == "a"
    assert paths[0]["mid_name"] == "b"
    assert paths[0]["dst_name"] == "c"
    assert abs(paths[0]["signed_score"] - 0.7) < 1e-6


def test_enumerate_precedes_paths_finds_two_hop_chain():
    batch = _batch()
    paths = enumerate_precedes_paths_batch(
        batch["edge_src"], batch["edge_dst"], batch["edge_type"], batch["edge_mask"]
    )
    triples = {(p[2], p[3], p[4]) for p in paths[0]}
    assert (0, 1, 2) in triples
    assert (1, 2, 3) in triples
    assert (0, 1, 3) not in triples
    hops3 = enumerate_simple_k_hop_precedes(
        batch["edge_src"], batch["edge_dst"], batch["edge_type"], batch["edge_mask"], hops=3
    )
    assert hops3[0] == [(0, 1, 2, 3)]


def test_union_path_edge_mask_deletes_both_edges_only():
    batch = _batch()
    paths = enumerate_precedes_paths_batch(
        batch["edge_src"], batch["edge_dst"], batch["edge_type"], batch["edge_mask"]
    )
    chosen = next(i for i, p in enumerate(paths[0]) if p[2:] == (0, 1, 2))
    mask = union_path_edge_mask(batch["edge_mask"], paths, [[chosen]])
    changed = perturb_edges(batch, mask)
    assert changed["edge_mask"][0, :3].tolist() == [False, False, True]
    assert torch.equal(changed["active_mask"], batch["active_mask"])
    assert torch.equal(changed["a_dense"], batch["a_dense"])


def test_select_top_positive_ignores_negative_and_caps_by_fraction():
    assert select_top_positive_indices([0.4, -0.9, 0.2, 0.1], fraction=0.5) == [0, 2]
    assert select_top_positive_indices([-0.2, -0.1], fraction=0.5) == []
    a = select_random_indices(5, 2, key=17, seed=42, repeat=1)
    b = select_random_indices(5, 2, key=17, seed=42, repeat=1)
    assert a == b
    assert len(set(a)) == 2


def test_path_interaction_is_joint_minus_sum_of_singles():
    assert abs(path_interaction(0.5, 0.2, 0.1) - 0.2) < 1e-12
    assert fold_path_pattern(
        "mbp", "B1_0_6", "urine_output_ml", "B2_6_12", "spo2", "B3_12_24"
    ) == "MBP(B1) → urine(B2) → SpO2(B3)"
