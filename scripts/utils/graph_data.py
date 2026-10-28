"""
Graph tensor schema, packing helpers, Dataset / collate (docs/13 §7).

Stage B: in-memory / synthetic packs only. No MIMIC I/O, no test packs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.utils.graph_builder import CanonicalEdge, PatientGraph, edge_attr_matrix
from scripts.utils.graph_constants import (
    EDGE_TYPE_CO_OCCUR,
    EDGE_TYPE_PRECEDES,
    MAX_CANONICAL_EDGES,
    N_EDGE_ATTR,
    N_SPARSE_FEATURES,
    N_STATIC_FEATURES,
    T_HOURS_GRAPH,
)


@dataclass
class PatientGraphTensors:
    """Single-patient tensors matching docs/13 §7.1 (variable E, padded later)."""

    a_dense: torch.Tensor  # (N,)
    node_time: torch.Tensor  # (N,) t_star/48
    node_time_within_stage: torch.Tensor  # (N,)
    active_mask: torch.Tensor  # (N,) bool
    edge_src: torch.Tensor  # (E,) int64
    edge_dst: torch.Tensor  # (E,) int64
    edge_type: torch.Tensor  # (E,) int64
    edge_attr: torch.Tensor  # (E, 5)
    sparse: torch.Tensor  # (308,)
    static: torch.Tensor  # (11,)
    y: torch.Tensor  # ()
    stay_id: torch.Tensor  # ()
    subject_id: torch.Tensor  # ()

    @property
    def n_nodes(self) -> int:
        return int(self.a_dense.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_src.shape[0])


def patient_graph_to_tensors(
    graph: PatientGraph,
    *,
    a_dense: np.ndarray | torch.Tensor,
    node_time: np.ndarray | torch.Tensor | None = None,
    node_time_within_stage: np.ndarray | torch.Tensor | None = None,
    sparse: np.ndarray | torch.Tensor | None = None,
    static: np.ndarray | torch.Tensor | None = None,
    n_train_stays: int = 1,
) -> PatientGraphTensors:
    """Convert PatientGraph + dense vectors into tensors (unpadded edges)."""
    a = torch.as_tensor(a_dense, dtype=torch.float32)
    n = int(a.numel())
    active = torch.zeros(n, dtype=torch.bool)
    for nid in graph.active_node_ids:
        if 0 <= int(nid) < n:
            active[int(nid)] = True

    if node_time is None:
        nt = torch.zeros(n, dtype=torch.float32)
    else:
        nt = torch.as_tensor(node_time, dtype=torch.float32)

    if node_time_within_stage is None:
        ntw = torch.zeros(n, dtype=torch.float32)
    else:
        ntw = torch.as_tensor(node_time_within_stage, dtype=torch.float32)

    edges = list(graph.edges)
    e = len(edges)
    if e > MAX_CANONICAL_EDGES:
        raise ValueError(f"E={e} exceeds MAX_CANONICAL_EDGES={MAX_CANONICAL_EDGES}")

    if e == 0:
        src = torch.zeros(0, dtype=torch.long)
        dst = torch.zeros(0, dtype=torch.long)
        et = torch.zeros(0, dtype=torch.long)
        ea = torch.zeros(0, N_EDGE_ATTR, dtype=torch.float32)
    else:
        src = torch.tensor([ed.src for ed in edges], dtype=torch.long)
        dst = torch.tensor([ed.dst for ed in edges], dtype=torch.long)
        et = torch.tensor([ed.edge_type for ed in edges], dtype=torch.long)
        ea = torch.as_tensor(edge_attr_matrix(edges, n_train_stays), dtype=torch.float32)

    sp = (
        torch.zeros(N_SPARSE_FEATURES, dtype=torch.float32)
        if sparse is None
        else torch.as_tensor(sparse, dtype=torch.float32)
    )
    st = (
        torch.zeros(N_STATIC_FEATURES, dtype=torch.float32)
        if static is None
        else torch.as_tensor(static, dtype=torch.float32)
    )
    if sp.numel() != N_SPARSE_FEATURES:
        raise ValueError(f"sparse dim {sp.numel()} != {N_SPARSE_FEATURES}")
    if st.numel() != N_STATIC_FEATURES:
        raise ValueError(f"static dim {st.numel()} != {N_STATIC_FEATURES}")

    return PatientGraphTensors(
        a_dense=a,
        node_time=nt,
        node_time_within_stage=ntw,
        active_mask=active,
        edge_src=src,
        edge_dst=dst,
        edge_type=et,
        edge_attr=ea,
        sparse=sp,
        static=st,
        y=torch.tensor(float(graph.y), dtype=torch.float32),
        stay_id=torch.tensor(int(graph.stay_id), dtype=torch.long),
        subject_id=torch.tensor(int(graph.subject_id), dtype=torch.long),
    )


def pad_edges(
    sample: PatientGraphTensors,
    max_edges: int = MAX_CANONICAL_EDGES,
) -> dict[str, torch.Tensor]:
    """Pad variable-length edges to max_edges; add edge_mask."""
    e = sample.n_edges
    if e > max_edges:
        raise ValueError(f"E={e} > max_edges={max_edges}")

    def _pad_1d(t: torch.Tensor, fill: int | float = 0) -> torch.Tensor:
        out = t.new_full((max_edges,), fill)
        if e:
            out[:e] = t
        return out

    edge_src = _pad_1d(sample.edge_src, 0)
    edge_dst = _pad_1d(sample.edge_dst, 0)
    edge_type = _pad_1d(sample.edge_type, 0)
    edge_attr = sample.edge_attr.new_zeros((max_edges, N_EDGE_ATTR))
    if e:
        edge_attr[:e] = sample.edge_attr
    edge_mask = torch.zeros(max_edges, dtype=torch.bool)
    if e:
        edge_mask[:e] = True

    return {
        "a_dense": sample.a_dense,
        "node_time": sample.node_time,
        "node_time_within_stage": sample.node_time_within_stage,
        "active_mask": sample.active_mask,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_type": edge_type,
        "edge_attr": edge_attr,
        "edge_mask": edge_mask,
        "sparse": sample.sparse,
        "static": sample.static,
        "y": sample.y,
        "stay_id": sample.stay_id,
        "subject_id": sample.subject_id,
    }


def collate_graph_batch(
    samples: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack padded patient dicts into a batch (B, ...)."""
    if not samples:
        raise ValueError("empty batch")
    keys = samples[0].keys()
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.stack([s[k] for s in samples], dim=0)
    return out


class GraphPackDataset(Dataset):
    """
    In-memory list of padded patient graph dicts.

    Stage C/D will load graph_pack_{split}.pt into this structure.
    Stage B uses synthetic lists only.
    """

    def __init__(self, patients: Sequence[dict[str, torch.Tensor]]):
        self.patients = list(patients)
        if self.patients:
            n0 = int(self.patients[0]["a_dense"].shape[0])
            for i, p in enumerate(self.patients):
                if int(p["a_dense"].shape[0]) != n0:
                    raise ValueError(f"patient {i} N mismatch")
                if "edge_mask" not in p:
                    raise ValueError(f"patient {i} missing edge_mask (pad first)")

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.patients[idx]

    @property
    def n_nodes(self) -> int:
        if not self.patients:
            return 0
        return int(self.patients[0]["a_dense"].shape[0])


def make_synthetic_patient(
    *,
    n_nodes: int,
    stay_id: int = 0,
    subject_id: int = 0,
    y: float = 0.0,
    active_ids: Sequence[int] | None = None,
    edges: Sequence[tuple[int, int, int]] | None = None,
    a_values: dict[int, float] | None = None,
    times: dict[int, float] | None = None,
    edge_attrs: Sequence[Sequence[float]] | None = None,
    sparse: np.ndarray | None = None,
    static: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Build one padded synthetic patient graph for Stage B tests / dry-run.

    edges: list of (src, dst, edge_type) with edge_type 0=CO_OCCUR, 1=PRECEDES.
    times: absolute hours after onset for t_star (stored as t/48).
    """
    rng = np.random.RandomState(seed + stay_id)
    a = np.zeros(n_nodes, dtype=np.float32)
    nt = np.zeros(n_nodes, dtype=np.float32)
    ntw = np.zeros(n_nodes, dtype=np.float32)

    act = list(active_ids) if active_ids is not None else []
    if a_values:
        for i, v in a_values.items():
            a[int(i)] = float(v)
    else:
        for i in act:
            a[int(i)] = float(rng.uniform(0.55, 0.99))

    if times:
        for i, t in times.items():
            nt[int(i)] = float(t) / float(T_HOURS_GRAPH)
            ntw[int(i)] = min(1.0, float(t) / 6.0)  # crude stage-local proxy
    else:
        for i in act:
            t = float(rng.uniform(0.0, 40.0))
            nt[int(i)] = t / float(T_HOURS_GRAPH)
            ntw[int(i)] = min(1.0, (t % 12.0) / 12.0)

    edge_list = list(edges) if edges is not None else []
    e = len(edge_list)
    if e > MAX_CANONICAL_EDGES:
        raise ValueError("too many synthetic edges")

    can_edges: list[CanonicalEdge] = []
    for j, (s, d, et) in enumerate(edge_list):
        s, d, et = int(s), int(d), int(et)
        if et == EDGE_TYPE_CO_OCCUR and s > d:
            s, d = d, s
        ap = float(a[s] * a[d]) if s < n_nodes and d < n_nodes else 0.5
        can_edges.append(
            CanonicalEdge(
                src=s,
                dst=d,
                edge_type=et,
                abs_delta_t=2.0 if et == EDGE_TYPE_CO_OCCUR else 5.0,
                a_prod=ap,
                same_variable=0.0,
                same_stage=1.0 if et == EDGE_TYPE_CO_OCCUR else 0.0,
                train_support=25,
            )
        )

    g = PatientGraph(
        stay_id=stay_id,
        subject_id=subject_id,
        split="synthetic",
        y=y,
        active_node_ids=sorted(set(int(i) for i in act)),
        edges=can_edges,
        n_events=len(act),
    )

    if edge_attrs is not None and len(edge_attrs) != e:
        raise ValueError("edge_attrs length mismatch")

    sample = patient_graph_to_tensors(
        g,
        a_dense=a,
        node_time=nt,
        node_time_within_stage=ntw,
        sparse=sparse if sparse is not None else rng.randn(N_SPARSE_FEATURES).astype(np.float32) * 0.1,
        static=static if static is not None else rng.randn(N_STATIC_FEATURES).astype(np.float32) * 0.1,
        n_train_stays=100,
    )

    if edge_attrs is not None and e:
        ea = torch.tensor(edge_attrs, dtype=torch.float32)
        sample.edge_attr = ea

    return pad_edges(sample)


def make_synthetic_batch(
    n_nodes: int = 12,
    batch_size: int = 4,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Small mixed batch: nonempty graphs + one empty graph."""
    patients: list[dict[str, torch.Tensor]] = []
    # empty
    patients.append(
        make_synthetic_patient(
            n_nodes=n_nodes,
            stay_id=0,
            y=0.0,
            active_ids=[],
            edges=[],
            seed=seed,
        )
    )
    # chain PRECEDES 0->1->2
    patients.append(
        make_synthetic_patient(
            n_nodes=n_nodes,
            stay_id=1,
            y=1.0,
            active_ids=[0, 1, 2],
            edges=[(0, 1, EDGE_TYPE_PRECEDES), (1, 2, EDGE_TYPE_PRECEDES)],
            a_values={0: 0.9, 1: 0.85, 2: 0.8},
            times={0: 1.0, 1: 6.0, 2: 12.0},
            seed=seed,
        )
    )
    # CO_OCCUR pair
    patients.append(
        make_synthetic_patient(
            n_nodes=n_nodes,
            stay_id=2,
            y=0.0,
            active_ids=[3, 4],
            edges=[(3, 4, EDGE_TYPE_CO_OCCUR)],
            a_values={3: 0.7, 4: 0.75},
            times={3: 2.0, 4: 2.5},
            seed=seed,
        )
    )
    # mixed
    for i in range(3, batch_size):
        patients.append(
            make_synthetic_patient(
                n_nodes=n_nodes,
                stay_id=i,
                y=float(i % 2),
                active_ids=[i % n_nodes, (i + 1) % n_nodes],
                edges=[(i % n_nodes, (i + 1) % n_nodes, EDGE_TYPE_PRECEDES)],
                seed=seed + i,
            )
        )
    return collate_graph_batch(patients[:batch_size])


def assert_batch_schema(batch: dict[str, torch.Tensor], n_nodes: int) -> None:
    """Raise if batch tensors violate Stage B pack contract."""
    b = batch["a_dense"].shape[0]
    emax = MAX_CANONICAL_EDGES
    shapes = {
        "a_dense": (b, n_nodes),
        "node_time": (b, n_nodes),
        "node_time_within_stage": (b, n_nodes),
        "active_mask": (b, n_nodes),
        "edge_src": (b, emax),
        "edge_dst": (b, emax),
        "edge_type": (b, emax),
        "edge_attr": (b, emax, N_EDGE_ATTR),
        "edge_mask": (b, emax),
        "sparse": (b, N_SPARSE_FEATURES),
        "static": (b, N_STATIC_FEATURES),
        "y": (b,),
        "stay_id": (b,),
        "subject_id": (b,),
    }
    for k, sh in shapes.items():
        if k not in batch:
            raise KeyError(f"missing batch key {k}")
        if tuple(batch[k].shape) != sh:
            raise ValueError(f"{k} shape {tuple(batch[k].shape)} != {sh}")


def pack_dict_from_patients(
    patients: Sequence[dict[str, torch.Tensor]],
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a list of padded patients into a graph_pack-like dict."""
    return {
        "patients": list(patients),
        "n_nodes": int(patients[0]["a_dense"].shape[0]) if patients else 0,
        "max_edges": MAX_CANONICAL_EDGES,
        "meta": meta or {"synthetic": True, "test_built": False},
    }
