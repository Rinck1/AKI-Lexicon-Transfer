"""Stage B / G0: synthetic GNN forward/backward & directional tests (docs/13 §11.3)."""

from __future__ import annotations

import copy

import pytest
import torch

from scripts.utils.graph_constants import (
    EDGE_TYPE_CO_OCCUR,
    EDGE_TYPE_PRECEDES,
    EMPTY_GRAPH_EMBED_DIM,
    HIDDEN_DIM,
    MAX_CANONICAL_EDGES,
)
from scripts.utils.graph_data import (
    assert_batch_schema,
    collate_graph_batch,
    make_synthetic_batch,
    make_synthetic_patient,
)
from scripts.utils.graph_model import GraphRiskModel, require_torch_2


@pytest.fixture
def n_nodes() -> int:
    return 12


@pytest.fixture
def model(n_nodes: int) -> GraphRiskModel:
    torch.manual_seed(0)
    return GraphRiskModel(n_nodes=n_nodes)


def test_torch_version_gate():
    require_torch_2()  # should not raise on CI/dev


def test_forward_shapes_and_finite(model: GraphRiskModel, n_nodes: int):
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=4, seed=1)
    assert_batch_schema(batch, n_nodes)
    out = model(batch)
    b = batch["y"].shape[0]
    assert out.logits.shape == (b,)
    assert out.prob.shape == (b,)
    assert out.h_graph.shape == (b, EMPTY_GRAPH_EMBED_DIM)
    assert out.node_attention.shape == (b, n_nodes)
    assert out.edge_attention.shape == (b, MAX_CANONICAL_EDGES)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.loss).all()
    assert torch.isfinite(out.h_graph).all()


def test_backward_reaches_node_embedding(model: GraphRiskModel, n_nodes: int):
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=3, seed=2)
    model.zero_grad(set_to_none=True)
    out = model(batch)
    out.loss.backward()
    assert model.node_emb.weight.grad is not None
    assert model.node_emb.weight.grad.abs().sum() > 0


def test_backward_reaches_in_out_relation_weights(model: GraphRiskModel, n_nodes: int):
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=3, seed=3)
    model.zero_grad(set_to_none=True)
    out = model(batch)
    out.loss.backward()
    layer0 = model.layers[0]
    for name, p in [
        ("W_pred", layer0.W_pred.weight),
        ("W_succ", layer0.W_succ.weight),
        ("W_co", layer0.W_co.weight),
    ]:
        assert p.grad is not None, name
        assert p.grad.abs().sum() > 0, f"{name} has zero grad"


def test_backward_reaches_attention_and_head(model: GraphRiskModel, n_nodes: int):
    # Multi-neighbor CO_OCCUR so segment-softmax attention has >1 member (nonzero grad).
    p = make_synthetic_patient(
        n_nodes=n_nodes,
        stay_id=1,
        y=1.0,
        active_ids=[0, 1, 2, 3],
        edges=[
            (0, 1, EDGE_TYPE_CO_OCCUR),
            (0, 2, EDGE_TYPE_CO_OCCUR),
            (0, 3, EDGE_TYPE_CO_OCCUR),
        ],
        a_values={0: 0.95, 1: 0.9, 2: 0.85, 3: 0.8},
        times={0: 1.0, 1: 1.2, 2: 1.4, 3: 1.6},
        seed=4,
    )
    batch = collate_graph_batch([p])
    model.zero_grad(set_to_none=True)
    out = model(batch)
    out.loss.backward()
    att_w = model.layers[0].att.net[0].weight
    assert att_w.grad is not None and att_w.grad.abs().sum() > 0
    head_w = model.risk_head[0].weight
    assert head_w.grad is not None and head_w.grad.abs().sum() > 0
    # node / edge pool attention also receive grads
    assert model.node_pool.v.weight.grad is not None
    assert model.node_pool.v.weight.grad.abs().sum() > 0


def test_inactive_nodes_receive_zero_pool_attention(model: GraphRiskModel, n_nodes: int):
    model.eval()
    p = make_synthetic_patient(
        n_nodes=n_nodes,
        stay_id=1,
        y=1.0,
        active_ids=[1, 3],
        edges=[(1, 3, EDGE_TYPE_PRECEDES)],
        a_values={1: 0.9, 3: 0.8},
        times={1: 1.0, 3: 6.0},
        seed=5,
    )
    batch = collate_graph_batch([p])
    with torch.no_grad():
        out = model(batch, compute_loss=False)
    omega = out.node_attention[0]
    inactive = ~batch["active_mask"][0]
    assert (omega[inactive].abs() < 1e-6).all()
    assert omega[batch["active_mask"][0]].sum() == pytest.approx(1.0, abs=1e-5)


def test_empty_graph_forward_finite(model: GraphRiskModel, n_nodes: int):
    p = make_synthetic_patient(
        n_nodes=n_nodes,
        stay_id=0,
        y=0.0,
        active_ids=[],
        edges=[],
        seed=6,
    )
    batch = collate_graph_batch([p])
    out = model(batch)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.h_graph).all()
    # empty graph: no active nodes → zero node attention; uses empty_graph_embedding
    att_sum = float(out.node_attention[0].detach().abs().sum().item())
    assert att_sum == pytest.approx(0.0, abs=1e-6)


def test_edge_order_permutation_invariant(model: GraphRiskModel, n_nodes: int):
    model.eval()
    edges = [
        (0, 1, EDGE_TYPE_PRECEDES),
        (1, 2, EDGE_TYPE_PRECEDES),
        (3, 4, EDGE_TYPE_CO_OCCUR),
    ]
    p0 = make_synthetic_patient(
        n_nodes=n_nodes,
        stay_id=1,
        y=1.0,
        active_ids=[0, 1, 2, 3, 4],
        edges=edges,
        a_values={0: 0.9, 1: 0.85, 2: 0.8, 3: 0.7, 4: 0.75},
        times={0: 1.0, 1: 6.0, 2: 12.0, 3: 2.0, 4: 2.5},
        seed=7,
    )
    p1 = make_synthetic_patient(
        n_nodes=n_nodes,
        stay_id=1,
        y=1.0,
        active_ids=[0, 1, 2, 3, 4],
        edges=list(reversed(edges)),
        a_values={0: 0.9, 1: 0.85, 2: 0.8, 3: 0.7, 4: 0.75},
        times={0: 1.0, 1: 6.0, 2: 12.0, 3: 2.0, 4: 2.5},
        seed=7,
    )
    with torch.no_grad():
        o0 = model(collate_graph_batch([p0]), compute_loss=False)
        o1 = model(collate_graph_batch([p1]), compute_loss=False)
    assert torch.allclose(o0.logits, o1.logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(o0.h_graph, o1.h_graph, atol=1e-5, rtol=1e-5)


def test_reversing_precedes_changes_output(model: GraphRiskModel, n_nodes: int):
    model.eval()
    base = dict(
        n_nodes=n_nodes,
        stay_id=1,
        y=1.0,
        active_ids=[0, 1],
        a_values={0: 0.95, 1: 0.6},
        times={0: 1.0, 1: 8.0},
        seed=8,
    )
    p_fwd = make_synthetic_patient(**base, edges=[(0, 1, EDGE_TYPE_PRECEDES)])
    p_rev = make_synthetic_patient(**base, edges=[(1, 0, EDGE_TYPE_PRECEDES)])
    with torch.no_grad():
        o_fwd = model(collate_graph_batch([p_fwd]), compute_loss=False)
        o_rev = model(collate_graph_batch([p_rev]), compute_loss=False)
    assert not torch.allclose(o_fwd.logits, o_rev.logits, atol=1e-5)
    assert not torch.allclose(o_fwd.h_graph, o_rev.h_graph, atol=1e-5)


def test_reversing_cooccur_does_not_change_output(model: GraphRiskModel, n_nodes: int):
    """CO_OCCUR is expanded bidirectionally; storage order must not matter."""
    model.eval()
    base = dict(
        n_nodes=n_nodes,
        stay_id=1,
        y=0.0,
        active_ids=[2, 5],
        a_values={2: 0.8, 5: 0.7},
        times={2: 3.0, 5: 3.5},
        seed=9,
    )
    # force non-canonical storage by bypassing make's sort: set after pad
    p_ab = make_synthetic_patient(**base, edges=[(2, 5, EDGE_TYPE_CO_OCCUR)])
    p_ba = make_synthetic_patient(**base, edges=[(5, 2, EDGE_TYPE_CO_OCCUR)])
    # make_synthetic_patient canonicalizes CO_OCCUR to min,max — force reverse storage
    # by swapping tensors after construction
    p_ba = copy.deepcopy(p_ab)
    # find the single edge and swap src/dst
    mask = p_ba["edge_mask"]
    assert int(mask.sum()) == 1
    idx = int(mask.nonzero(as_tuple=False)[0].item())
    s, d = int(p_ba["edge_src"][idx]), int(p_ba["edge_dst"][idx])
    p_ba["edge_src"][idx] = d
    p_ba["edge_dst"][idx] = s

    with torch.no_grad():
        o0 = model(collate_graph_batch([p_ab]), compute_loss=False)
        o1 = model(collate_graph_batch([p_ba]), compute_loss=False)
    assert torch.allclose(o0.logits, o1.logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(o0.h_graph, o1.h_graph, atol=1e-5, rtol=1e-5)


def test_two_hop_chain_affects_two_layer_representation(n_nodes: int):
    """With 2 layers, removing early chain source changes h_graph."""
    torch.manual_seed(10)
    m = GraphRiskModel(n_nodes=n_nodes, n_layers=2)
    m.eval()
    common = dict(
        n_nodes=n_nodes,
        stay_id=1,
        y=1.0,
        a_values={0: 0.95, 1: 0.9, 2: 0.85},
        times={0: 1.0, 1: 6.0, 2: 12.0},
        seed=10,
    )
    full = make_synthetic_patient(
        **common,
        active_ids=[0, 1, 2],
        edges=[
            (0, 1, EDGE_TYPE_PRECEDES),
            (1, 2, EDGE_TYPE_PRECEDES),
        ],
    )
    # drop node 0 from topology (still in a_dense for fair comparison of graph path)
    no_src = make_synthetic_patient(
        **common,
        active_ids=[1, 2],
        edges=[(1, 2, EDGE_TYPE_PRECEDES)],
    )
    with torch.no_grad():
        o_full = m(collate_graph_batch([full]), compute_loss=False)
        o_cut = m(collate_graph_batch([no_src]), compute_loss=False)
    assert not torch.allclose(o_full.h_graph, o_cut.h_graph, atol=1e-5)


def test_pos_weight_loss_runs(model: GraphRiskModel, n_nodes: int):
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=4, seed=11)
    out = model(batch, pos_weight=2.5)
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_residual_gate_zero_matches_b_logits(n_nodes: int):
    torch.manual_seed(0)
    b = GraphRiskModel(n_nodes=n_nodes, branch_mode="B")
    r = GraphRiskModel(n_nodes=n_nodes, branch_mode="B", residual_graph=True)
    r.load_b_branch(b.state_dict())
    r.freeze_b_params()
    assert float(r.graph_gate.item()) == 0.0
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=3, seed=13)
    b.eval()
    r.eval()
    with torch.no_grad():
        o_b = b(batch, compute_loss=False)
        o_r = r(batch, compute_loss=False)
    assert torch.allclose(o_b.logits, o_r.logits, atol=1e-5)
    assert any(not p.requires_grad for n, p in r.named_parameters() if n.startswith("a_proj"))
    assert r.graph_gate.requires_grad
    assert r.graph_delta_head.weight.requires_grad


def test_branch_zero_keeps_fusion_dim_and_param_count(n_nodes: int):
    torch.manual_seed(0)
    full = GraphRiskModel(n_nodes=n_nodes, branch_mode="C")
    arm_a = GraphRiskModel(n_nodes=n_nodes, branch_mode="A")
    arm_b = GraphRiskModel(n_nodes=n_nodes, branch_mode="B")
    arm_ag = GraphRiskModel(n_nodes=n_nodes, branch_mode="AG")
    n_full = sum(p.numel() for p in full.parameters())
    assert n_full == sum(p.numel() for p in arm_a.parameters())
    assert n_full == sum(p.numel() for p in arm_b.parameters())
    assert n_full == sum(p.numel() for p in arm_ag.parameters())
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=2, seed=12)
    out_a = arm_a(batch, compute_loss=False)
    out_b = arm_b(batch, compute_loss=False)
    out_ag = arm_ag(batch, compute_loss=False)
    assert out_a.h_graph.abs().sum() == 0
    assert out_b.h_graph.abs().sum() == 0
    assert out_ag.h_graph.abs().sum() > 0
    assert out_a.logits.shape == out_b.logits.shape == out_ag.logits.shape


def test_node_length_channel_changes_in_dim(n_nodes: int):
    torch.manual_seed(0)
    plain = GraphRiskModel(n_nodes=n_nodes, branch_mode="G", readout="additive")
    with_len = GraphRiskModel(
        n_nodes=n_nodes, branch_mode="G", readout="additive", use_node_length=True
    )
    assert plain.node_in.in_features + 1 == with_len.node_in.in_features
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=2, seed=3)
    batch["node_length"] = torch.zeros_like(batch["a_dense"])
    batch["node_length"][:, 0] = 0.5
    with_len.eval()
    out = with_len(batch, compute_loss=False)
    assert out.logits.shape == (2,)


def test_additive_gate_no_softmax_and_completeness(n_nodes: int):
    torch.manual_seed(0)
    model = GraphRiskModel(
        n_nodes=n_nodes, branch_mode="G", readout="additive_gate"
    )
    model.eval()
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=3, seed=11)
    out = model(batch, compute_loss=False)
    g = out.edge_gate
    mask = batch["edge_mask"]
    assert g is not None and out.edge_value is not None
    assert torch.all((g >= 0) & (g <= 1) | ~mask)
    # Independent sigmoid: row sums need not be 1.
    row_sum = (g * mask.to(g.dtype)).sum(dim=-1)
    has_e = mask.any(dim=-1)
    if bool(has_e.any()):
        assert not torch.allclose(row_sum[has_e], torch.ones_like(row_sum[has_e]), atol=1e-3)
    n_e = mask.to(g.dtype).sum(dim=-1).clamp_min(1.0)
    rebuilt = out.node_contrib.sum(dim=-1) + n_e.rsqrt() * out.edge_contrib.sum(dim=-1)
    assert torch.allclose(out.graph_logit, rebuilt, atol=1e-5)
    assert torch.allclose(out.logits, out.tabular_logit + out.graph_logit, atol=1e-5)
    empty = make_synthetic_patient(
        n_nodes=n_nodes, stay_id=99, subject_id=99, y=0.0, active_ids=[], edges=[]
    )
    out_e = model(collate_graph_batch([empty]), compute_loss=False)
    assert float(out_e.graph_logit.item()) == 0.0


def test_additive_gate_aux_losses_and_stability(n_nodes: int):
    torch.manual_seed(1)
    model = GraphRiskModel(
        n_nodes=n_nodes, branch_mode="G", readout="additive_gate"
    )
    model.train()
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=4, seed=8)
    out = model(batch, compute_loss=True)
    assert out.loss_sparse is not None and out.loss_stable is not None
    assert float(out.loss_sparse.detach()) > 0.0
    assert float(out.mean_gate.detach()) > 0.0
    # Total loss includes aux terms.
    expect = out.loss_bce + model.lambda_sparse * out.loss_sparse + model.lambda_stable * out.loss_stable
    assert torch.allclose(out.loss, expect, atol=1e-6)
    out.loss.backward()
    assert model.edge_gate.weight.grad is not None
    assert torch.isfinite(model.edge_gate.weight.grad).all()


def test_additive_still_loads_without_gate_params(n_nodes: int):
    model = GraphRiskModel(n_nodes=n_nodes, branch_mode="G", readout="additive")
    assert model.edge_gate is None
    assert model.node_in.in_features == 16 + 3
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=2, seed=1)
    out = model(batch, compute_loss=False)
    assert out.graph_logit is not None


def test_additive_zero_init_graph_logit(n_nodes: int):
    model = GraphRiskModel(n_nodes=n_nodes, branch_mode="G", readout="additive")
    model.eval()
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=2, seed=7)
    out = model(batch, compute_loss=False)
    assert torch.allclose(out.graph_logit, torch.zeros_like(out.graph_logit), atol=1e-6)
    assert torch.allclose(out.node_contrib, torch.zeros_like(out.node_contrib), atol=1e-6)
    assert torch.allclose(out.edge_contrib, torch.zeros_like(out.edge_contrib), atol=1e-6)


def test_additive_completeness_and_empty_graph(n_nodes: int):
    torch.manual_seed(0)
    model = GraphRiskModel(n_nodes=n_nodes, branch_mode="AG", readout="additive")
    model.eval()
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=3, seed=21)
    out = model(batch, compute_loss=False)
    rebuilt = out.node_contrib.sum(dim=-1) + out.edge_contrib.sum(dim=-1)
    assert torch.allclose(out.graph_logit, rebuilt, atol=1e-5)
    assert torch.allclose(out.logits, out.tabular_logit + out.graph_logit, atol=1e-5)
    assert out.message_attention.shape == out.edge_attention.shape
    empty = make_synthetic_patient(
        n_nodes=n_nodes,
        stay_id=99,
        subject_id=99,
        y=0.0,
        active_ids=[],
        edges=[],
    )
    empty_batch = collate_graph_batch([empty])
    out_e = model(empty_batch, compute_loss=False)
    assert float(out_e.graph_logit.item()) == 0.0
    assert torch.isfinite(out_e.logits).all()


def test_g_branch_keeps_graph_and_zeros_act_sparse(n_nodes: int):
    torch.manual_seed(1)
    model = GraphRiskModel(n_nodes=n_nodes, branch_mode="G", readout="additive")
    model.eval()
    assert model.use_graph and (not model.use_act) and (not model.use_sparse)
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=2, seed=22)
    out = model(batch, compute_loss=False)
    assert out.node_contrib is not None
    assert torch.allclose(out.graph_logit, torch.zeros_like(out.graph_logit), atol=1e-6)
    # tabular path must not see a_dense or sparse
    a_p = model.a_proj(batch["a_dense"])
    assert a_p.abs().sum() > 0
    z_tab = torch.cat(
        [
            batch["a_dense"].new_zeros(2, out.h_graph.shape[1]),
            torch.zeros_like(a_p),
            torch.zeros_like(model.sparse_proj(batch["sparse"])),
            model.static_proj(batch["static"]),
        ],
        dim=-1,
    )
    tab = model.risk_head(z_tab).squeeze(-1)
    assert torch.allclose(out.tabular_logit, tab, atol=1e-5)


def test_additive_ms_three_layers_shapes_and_identity(n_nodes: int):
    torch.manual_seed(0)
    model = GraphRiskModel(
        n_nodes=n_nodes, branch_mode="G", readout="additive_ms", n_layers=3
    )
    model.eval()
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=3, seed=31)
    out = model(batch, compute_loss=False)
    b, n = batch["y"].shape[0], n_nodes
    assert out.h0.shape == (b, n, 64)
    assert out.h1.shape == (b, n, 64)
    assert out.h2.shape == (b, n, 64)
    assert out.h3.shape == (b, n, 64)
    assert out.node_contrib_by_layer.shape == (b, n, 4)
    assert torch.allclose(
        out.node_contrib, out.node_contrib_by_layer.sum(dim=-1), atol=1e-5
    )
    rebuilt = out.node_contrib.sum(dim=-1) + out.edge_contrib.sum(dim=-1)
    assert torch.allclose(out.graph_logit, rebuilt, atol=1e-5)
    assert torch.allclose(out.logits, out.tabular_logit + out.graph_logit, atol=1e-5)
    inactive = ~batch["active_mask"]
    assert out.node_contrib_by_layer[inactive].abs().sum() == 0


def test_additive_ms_three_layers_all_receive_grad(n_nodes: int):
    torch.manual_seed(1)
    model = GraphRiskModel(
        n_nodes=n_nodes, branch_mode="G", readout="additive_ms", n_layers=3
    )
    with torch.no_grad():
        model.node_risk.weight.normal_(0.0, 0.05)
        model.edge_risk.weight.normal_(0.0, 0.05)
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=3, seed=32)
    model.zero_grad(set_to_none=True)
    out = model(batch)
    out.loss.backward()
    for i, layer in enumerate(model.layers):
        assert layer.W_pred.weight.grad is not None, i
        assert layer.W_pred.weight.grad.abs().sum() > 0, i


def test_additive_ms_four_node_chain_source_changes_h3():
    torch.manual_seed(2)
    model = GraphRiskModel(n_nodes=8, branch_mode="G", readout="additive_ms", n_layers=3)
    model.eval()
    common = dict(
        n_nodes=8,
        stay_id=1,
        y=1.0,
        active_ids=[0, 1, 2, 3],
        edges=[
            (0, 1, EDGE_TYPE_PRECEDES),
            (1, 2, EDGE_TYPE_PRECEDES),
            (2, 3, EDGE_TYPE_PRECEDES),
        ],
        times={0: 1.0, 1: 6.0, 2: 12.0, 3: 18.0},
        seed=2,
    )
    full = make_synthetic_patient(**common, a_values={0: 0.95, 1: 0.9, 2: 0.85, 3: 0.8})
    cut = make_synthetic_patient(**common, a_values={0: 0.10, 1: 0.9, 2: 0.85, 3: 0.8})
    with torch.no_grad():
        o_full = model(collate_graph_batch([full]), compute_loss=False)
        o_cut = model(collate_graph_batch([cut]), compute_loss=False)
    assert not torch.allclose(o_full.h3[0, 3], o_cut.h3[0, 3], atol=1e-5)


def test_additive_ms_trace_optional_and_has_canonical_slot(n_nodes: int):
    model = GraphRiskModel(
        n_nodes=n_nodes, branch_mode="G", readout="additive_ms", n_layers=3
    )
    model.eval()
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=2, seed=33)
    out_off = model(batch, compute_loss=False)
    assert out_off.message_trace is None
    out_on = model(batch, compute_loss=False, return_trace=True)
    assert out_on.message_trace is not None
    assert len(out_on.message_trace) == 3
    for tr in out_on.message_trace:
        assert tr.layer in {1, 2, 3}
        assert tr.edge_slot.ndim == 1
        assert tr.src_node.shape == tr.edge_slot.shape
        assert tr.eta.shape == tr.edge_slot.shape
        if tr.edge_slot.numel():
            assert int(tr.edge_slot.min()) >= 0


def test_additive_ms_empty_graph_forwards():
    model = GraphRiskModel(n_nodes=8, branch_mode="G", readout="additive_ms", n_layers=3)
    model.eval()
    empty = make_synthetic_patient(
        n_nodes=8, stay_id=3, subject_id=3, y=0.0, active_ids=[], edges=[]
    )
    out = model(collate_graph_batch([empty]), compute_loss=False)
    assert torch.isfinite(out.logits).all()
    assert float(out.graph_logit.item()) == 0.0
    assert out.h3 is not None


def test_segment_softmax_all_invalid_is_zero_not_nan():
    from scripts.utils.graph_model import _segment_softmax

    scores = torch.tensor([1.0, 2.0, 3.0])
    seg = torch.tensor([0, 0, 1])
    mask = torch.zeros(3, dtype=torch.bool)
    eta = _segment_softmax(scores, seg, 2, mask)
    assert torch.isfinite(eta).all()
    assert torch.allclose(eta, torch.zeros_like(eta))


def test_additive_l0_eval_hard_mask_and_bounded_value(n_nodes: int):
    torch.manual_seed(0)
    model = GraphRiskModel(n_nodes=n_nodes, branch_mode="G", readout="additive_l0")
    model.eval()
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=3, seed=11)
    mask = batch["edge_mask"]
    out = model(batch, compute_loss=False)
    g = out.edge_gate
    assert g is not None and out.edge_value is not None
    assert out.edge_selection_prob is not None
    assert set(g[mask].unique().tolist()) <= {0.0, 1.0}
    # Default type prior is -3 → all hard masks off.
    assert torch.all(g[mask] == 0)
    model.edge_value.weight.data.fill_(8.0)
    model.edge_value.bias.data.fill_(8.0)
    model.edge_type_prior.weight.data.fill_(5.0)
    out_on = model(batch, compute_loss=False)
    assert torch.all(out_on.edge_gate[mask] == 1)
    assert torch.all(out_on.edge_value.abs() <= model.v_max + 1e-5)
    n_sel = out_on.edge_gate.sum(dim=-1).clamp_min(1.0)
    rebuilt = out_on.node_contrib.sum(dim=-1) + n_sel.rsqrt() * out_on.edge_contrib.sum(
        dim=-1
    )
    assert torch.allclose(out_on.graph_logit, rebuilt, atol=1e-5)
    assert torch.allclose(out_on.logits, out_on.tabular_logit + out_on.graph_logit, atol=1e-5)


def test_additive_l0_zero_mask_matches_removed_edge(n_nodes: int):
    torch.manual_seed(2)
    model = GraphRiskModel(
        n_nodes=n_nodes, branch_mode="G", readout="additive_l0", n_layers=2
    )
    model.eval()
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=3, seed=9)
    # slot 0 of the pack is an empty graph; use the PRECEDES chain patient
    b = 1
    slots = batch["edge_mask"][b].nonzero(as_tuple=False).flatten()
    assert slots.numel() >= 1
    e0 = int(slots[0].item())
    z_ones = batch["edge_mask"].to(dtype=torch.float32)
    z_off = z_ones.clone()
    z_off[b, e0] = 0.0
    batch_rm = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch_rm["edge_mask"] = batch["edge_mask"].clone()
    batch_rm["edge_mask"][b, e0] = False
    h_rm, _, _ = model._run_encoder_states(
        batch_rm["a_dense"], batch_rm, batch_rm["active_mask"], edge_z=z_ones
    )
    h_off, _, _ = model._run_encoder_states(
        batch["a_dense"], batch, batch["active_mask"], edge_z=z_off
    )
    assert torch.allclose(h_off[-1], h_rm[-1], atol=1e-5)
    h_full, _, _ = model._run_encoder_states(
        batch["a_dense"], batch, batch["active_mask"], edge_z=z_ones
    )
    assert not torch.allclose(h_full[-1][b], h_off[-1][b], atol=1e-6)


def test_additive_l0_aux_losses_and_empty_graph(n_nodes: int):
    torch.manual_seed(3)
    model = GraphRiskModel(n_nodes=n_nodes, branch_mode="G", readout="additive_l0")
    model.train()
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=4, seed=8)
    out = model(batch, compute_loss=True)
    assert out.loss_sparse is not None and out.loss_stable is not None
    assert float(out.loss_sparse.detach()) > 0.0
    expect = (
        out.loss_bce
        + model.lambda_l0 * out.loss_sparse
        + model.lambda_rank * out.loss_stable
    )
    assert torch.allclose(out.loss, expect, atol=1e-6)
    out.loss.backward()
    assert model.edge_type_prior.weight.grad is not None
    assert torch.isfinite(model.edge_type_prior.weight.grad).all()
    assert model.edge_delta.weight.grad is not None
    model.eval()
    empty = make_synthetic_patient(
        n_nodes=n_nodes, stay_id=99, subject_id=99, y=0.0, active_ids=[], edges=[]
    )
    out_e = model(collate_graph_batch([empty]), compute_loss=False)
    assert float(out_e.graph_logit.item()) == 0.0
    assert torch.isfinite(out_e.logits).all()


def test_additive_still_loads_without_l0_params(n_nodes: int):
    model = GraphRiskModel(n_nodes=n_nodes, branch_mode="G", readout="additive")
    assert model.edge_type_prior is None
    assert model.edge_delta is None
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=2, seed=1)
    out = model(batch, compute_loss=False)
    assert out.graph_logit is not None


def test_additive_topk_exact_budget_and_bounded_value(n_nodes: int):
    from scripts.utils.graph_model import edge_budget_k, hard_topk_mask

    torch.manual_seed(0)
    model = GraphRiskModel(n_nodes=n_nodes, branch_mode="G", readout="additive_topk")
    model.eval()
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=4, seed=11)
    out = model(batch, compute_loss=False)
    mask = batch["edge_mask"]
    n_e = mask.sum(dim=-1)
    k = edge_budget_k(n_e)
    n_sel = (out.edge_gate * mask.to(out.edge_gate.dtype)).sum(dim=-1)
    assert torch.equal(n_sel.to(torch.long), k)
    assert set(out.edge_gate[mask].unique().tolist()) <= {0.0, 1.0}
    model.edge_value.weight.data.fill_(8.0)
    model.edge_value.bias.data.fill_(8.0)
    out2 = model(batch, compute_loss=False)
    assert torch.all(out2.edge_value.abs() <= model.v_max + 1e-5)
    rebuilt = out2.node_contrib.sum(dim=-1) + n_sel.clamp_min(1).rsqrt() * out2.edge_contrib.sum(
        dim=-1
    )
    assert torch.allclose(out2.graph_logit, rebuilt, atol=1e-5)
    # Isolated unit: K=max(1, ceil(0.2 E))
    valid = torch.tensor([[True] * 10 + [False] * 6])
    scores = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    z = hard_topk_mask(scores, valid, fraction=0.2)
    assert int(z.sum().item()) == 2


def test_additive_topk_ste_reaches_score_and_no_l0(n_nodes: int):
    torch.manual_seed(4)
    model = GraphRiskModel(n_nodes=n_nodes, branch_mode="G", readout="additive_topk")
    model.train()
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=4, seed=8)
    out = model(batch, compute_loss=True)
    expect = out.loss_bce + model.lambda_rank * out.loss_stable
    assert torch.allclose(out.loss, expect, atol=1e-6)
    assert float(out.loss_sparse.detach()) == 0.0
    out.loss.backward()
    assert model.edge_score.weight.grad is not None
    assert torch.isfinite(model.edge_score.weight.grad).all()
    model.eval()
    empty = make_synthetic_patient(
        n_nodes=n_nodes, stay_id=99, subject_id=99, y=0.0, active_ids=[], edges=[]
    )
    out_e = model(collate_graph_batch([empty]), compute_loss=False)
    assert float(out_e.graph_logit.item()) == 0.0
    assert float(out_e.edge_gate.sum().item()) == 0.0


def test_train_scripts_refuse_test_in_path():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "gnn_path_fid",
        Path(__file__).resolve().parents[1] / "19_gnn_path_fidelity.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    try:
        mod._load_val_pack(Path("data/processed/graphs/graph_test_pack"), None)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for a test pack path")
