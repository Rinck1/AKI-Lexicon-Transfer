"""
Patient graph construction (docs/13 §5).

Stage A: pure Python/numpy edge rules + train edge vocabulary.
No MIMIC I/O, no GNN.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from scripts.utils.graph_constants import (
    CO_OCCUR_MAX_HOURS,
    EDGE_TYPE_CO_OCCUR,
    EDGE_TYPE_PRECEDES,
    MAX_CANONICAL_EDGES,
    MAX_CO_OCCUR_NEIGHBORS,
    MAX_PRECEDES_OUT,
    N_EDGE_ATTR,
    PRECEDES_MAX_HOURS,
    PRECEDES_MIN_HOURS,
    min_edge_support,
)
from scripts.utils.graph_events import NodeType


@dataclass
class CanonicalEdge:
    src: int
    dst: int
    edge_type: int  # 0 CO_OCCUR, 1 PRECEDES
    abs_delta_t: float
    a_prod: float
    same_variable: float
    same_stage: float
    train_support: int = 0

    def key(self) -> tuple[int, int, int]:
        return (self.src, self.dst, self.edge_type)


@dataclass
class PatientGraph:
    stay_id: int
    subject_id: int
    split: str
    y: float
    active_node_ids: list[int]
    edges: list[CanonicalEdge] = field(default_factory=list)
    dropped_unseen_edge: int = 0
    n_events: int = 0

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def is_empty(self) -> bool:
        return self.n_events == 0


def _event_records(active_events: pd.DataFrame) -> list[dict]:
    """Normalize one patient's active events to dicts."""
    rows = []
    for _, r in active_events.iterrows():
        rows.append(
            {
                "node_id": int(r["node_id"]),
                "var": str(r["var"]),
                "stage": str(r["stage"]),
                "a": float(r["a"]),
                "t_star": float(r["t_star"]),
            }
        )
    return rows


def build_canonical_edges_for_patient(
    active_events: pd.DataFrame,
    *,
    co_max_h: float = CO_OCCUR_MAX_HOURS,
    precedes_min_h: float = PRECEDES_MIN_HOURS,
    precedes_max_h: float = PRECEDES_MAX_HOURS,
    max_co_neighbors: int = MAX_CO_OCCUR_NEIGHBORS,
    max_precedes_out: int = MAX_PRECEDES_OUT,
    max_edges: int = MAX_CANONICAL_EDGES,
) -> list[CanonicalEdge]:
    """
    Build CO_OCCUR + PRECEDES edges from active events of one patient.

    CO_OCCUR: |dt| <= 2h, undirected canonical (min_id, max_id), deg<=4.
    PRECEDES: 2h < dt <= 12h, directed, max 2 outgoing per source.
    """
    events = _event_records(active_events)
    if len(events) < 2:
        return []

    # --- CO_OCCUR candidates ---
    co_cands: list[tuple[float, float, int, int, float, float]] = []
    # (abs_dt, -a_prod, id_a, id_b, same_var, same_stage) with id_a < id_b
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            ei, ej = events[i], events[j]
            if ei["node_id"] == ej["node_id"]:
                continue
            dt = abs(ei["t_star"] - ej["t_star"])
            if dt > co_max_h + 1e-12:
                continue
            a_prod = ei["a"] * ej["a"]
            id_a, id_b = sorted((ei["node_id"], ej["node_id"]))
            same_var = 1.0 if ei["var"] == ej["var"] else 0.0
            same_stage = 1.0 if ei["stage"] == ej["stage"] else 0.0
            co_cands.append((dt, -a_prod, id_a, id_b, same_var, same_stage, a_prod))

    co_cands.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    degree: dict[int, int] = defaultdict(int)
    co_edges: list[CanonicalEdge] = []
    for dt, _neg_ap, id_a, id_b, same_var, same_stage, a_prod in co_cands:
        if degree[id_a] >= max_co_neighbors or degree[id_b] >= max_co_neighbors:
            continue
        co_edges.append(
            CanonicalEdge(
                src=id_a,
                dst=id_b,
                edge_type=EDGE_TYPE_CO_OCCUR,
                abs_delta_t=float(dt),
                a_prod=float(a_prod),
                same_variable=float(same_var),
                same_stage=float(same_stage),
            )
        )
        degree[id_a] += 1
        degree[id_b] += 1

    # --- PRECEDES candidates ---
    by_src: dict[int, list[tuple[float, float, int, float, float, float]]] = defaultdict(
        list
    )
    # src -> list of (dt, -a_dst, dst_id, a_prod, same_var, same_stage)
    for i in range(len(events)):
        for j in range(len(events)):
            if i == j:
                continue
            eu, ew = events[i], events[j]
            if eu["node_id"] == ew["node_id"]:
                continue
            dt = ew["t_star"] - eu["t_star"]
            if not (precedes_min_h < dt <= precedes_max_h + 1e-12):
                continue
            a_prod = eu["a"] * ew["a"]
            same_var = 1.0 if eu["var"] == ew["var"] else 0.0
            same_stage = 1.0 if eu["stage"] == ew["stage"] else 0.0
            by_src[eu["node_id"]].append(
                (
                    float(dt),
                    -float(ew["a"]),
                    int(ew["node_id"]),
                    float(a_prod),
                    float(same_var),
                    float(same_stage),
                )
            )

    pr_edges: list[CanonicalEdge] = []
    for src, cands in by_src.items():
        cands.sort(key=lambda x: (x[0], x[1], x[2]))
        for dt, _neg_a, dst, a_prod, same_var, same_stage in cands[:max_precedes_out]:
            pr_edges.append(
                CanonicalEdge(
                    src=int(src),
                    dst=int(dst),
                    edge_type=EDGE_TYPE_PRECEDES,
                    abs_delta_t=float(dt),
                    a_prod=float(a_prod),
                    same_variable=float(same_var),
                    same_stage=float(same_stage),
                )
            )

    edges = co_edges + pr_edges
    if len(edges) > max_edges:
        raise ValueError(
            f"canonical edges {len(edges)} exceed MAX_CANONICAL_EDGES={max_edges}"
        )
    # Deterministic order for storage
    edges.sort(key=lambda e: (e.edge_type, e.src, e.dst, e.abs_delta_t))
    return edges


def collect_train_edge_support(
    patient_edge_lists: Sequence[Sequence[CanonicalEdge]],
) -> dict[tuple[int, int, int], int]:
    """
    Count distinct patients supporting each (src, dst, edge_type).
    Each patient contributes at most 1 to a key.
    """
    support: dict[tuple[int, int, int], int] = defaultdict(int)
    for edges in patient_edge_lists:
        seen: set[tuple[int, int, int]] = set()
        for e in edges:
            k = e.key()
            if k in seen:
                continue
            seen.add(k)
            support[k] += 1
    return dict(support)


def freeze_edge_vocabulary(
    support: dict[tuple[int, int, int], int],
    n_train_stays: int,
) -> dict[tuple[int, int, int], int]:
    """Keep edges with train patient support >= min_edge_support."""
    thr = min_edge_support(n_train_stays)
    return {k: s for k, s in support.items() if s >= thr}


def apply_edge_vocabulary(
    edges: Sequence[CanonicalEdge],
    vocab: dict[tuple[int, int, int], int],
) -> tuple[list[CanonicalEdge], int]:
    """Filter edges to train vocab; attach train_support; return kept + n_dropped."""
    kept: list[CanonicalEdge] = []
    dropped = 0
    for e in edges:
        k = e.key()
        if k not in vocab:
            dropped += 1
            continue
        kept.append(
            CanonicalEdge(
                src=e.src,
                dst=e.dst,
                edge_type=e.edge_type,
                abs_delta_t=e.abs_delta_t,
                a_prod=e.a_prod,
                same_variable=e.same_variable,
                same_stage=e.same_stage,
                train_support=int(vocab[k]),
            )
        )
    return kept, dropped


def edge_attr_matrix(
    edges: Sequence[CanonicalEdge],
    n_train_stays: int,
) -> np.ndarray:
    """
    (E, 5) features:
      abs_delta_t/12, a_prod, same_variable, same_stage,
      log1p(train_support)/log1p(n_train)
    """
    e = len(edges)
    out = np.zeros((e, N_EDGE_ATTR), dtype=np.float32)
    denom = float(np.log1p(max(n_train_stays, 1)))
    for i, ed in enumerate(edges):
        out[i, 0] = ed.abs_delta_t / PRECEDES_MAX_HOURS
        out[i, 1] = ed.a_prod
        out[i, 2] = ed.same_variable
        out[i, 3] = ed.same_stage
        out[i, 4] = float(np.log1p(ed.train_support)) / denom if denom > 0 else 0.0
    return out


def build_patient_graph(
    *,
    stay_id: int,
    subject_id: int,
    split: str,
    y: float,
    active_events: pd.DataFrame,
    edge_vocab: dict[tuple[int, int, int], int] | None,
    n_train_stays: int,
    apply_vocab: bool = True,
) -> PatientGraph:
    """Build one patient graph; optionally filter by frozen train edge vocab."""
    events = active_events
    n_events = int(len(events))
    raw_edges = build_canonical_edges_for_patient(events)
    dropped = 0
    if apply_vocab:
        if edge_vocab is None:
            raise ValueError("edge_vocab required when apply_vocab=True")
        edges, dropped = apply_edge_vocabulary(raw_edges, edge_vocab)
    else:
        edges = list(raw_edges)

    active_ids = (
        sorted(events["node_id"].astype(int).unique().tolist()) if n_events else []
    )
    return PatientGraph(
        stay_id=int(stay_id),
        subject_id=int(subject_id),
        split=str(split),
        y=float(y),
        active_node_ids=active_ids,
        edges=edges,
        dropped_unseen_edge=int(dropped),
        n_events=n_events,
    )


def build_graphs_for_split(
    activations_long: pd.DataFrame,
    active_events: pd.DataFrame,
    *,
    edge_vocab: dict[tuple[int, int, int], int] | None,
    n_train_stays: int,
    apply_vocab: bool,
    stay_meta: pd.DataFrame | None = None,
) -> list[PatientGraph]:
    """
    Build graphs for all stays in stay_meta (or unique ICUSTAY_IDs in activations).

    stay_meta columns if provided: ICUSTAY_ID, SUBJECT_ID, split, HOSPITAL_MORTALITY
    """
    if stay_meta is None:
        stays = (
            activations_long[["ICUSTAY_ID", "SUBJECT_ID", "split", "HOSPITAL_MORTALITY"]]
            .drop_duplicates("ICUSTAY_ID")
            .copy()
        )
    else:
        stays = stay_meta.copy()

    graphs: list[PatientGraph] = []
    for _, row in stays.iterrows():
        sid = int(row["ICUSTAY_ID"])
        ev = active_events.loc[active_events["ICUSTAY_ID"] == sid]
        graphs.append(
            build_patient_graph(
                stay_id=sid,
                subject_id=int(row.get("SUBJECT_ID", -1)),
                split=str(row.get("split", "unknown")),
                y=float(row.get("HOSPITAL_MORTALITY", 0)),
                active_events=ev,
                edge_vocab=edge_vocab,
                n_train_stays=n_train_stays,
                apply_vocab=apply_vocab,
            )
        )
    return graphs


def empty_graph_rate(graphs: Sequence[PatientGraph]) -> float:
    if not graphs:
        return float("nan")
    return float(sum(1 for g in graphs if g.is_empty) / len(graphs))
