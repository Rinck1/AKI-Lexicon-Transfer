"""
Event layer for evolution graphs (docs/13 §3–§4).

Stage A: pure functions on in-memory tables / synthetic rows.
Does not read MIMIC or test splits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from scripts.utils.graph_constants import (
    ACTIVATION_NAME_SEP,
    A_MIN_EVENT,
    STAGE_NAMES,
)

_ACT_RE = re.compile(
    r"^(?P<var>[^|]+)\|L(?P<L>\d+)\|k(?P<k>\d+)\|(?P<stage>.+)$"
)


@dataclass(frozen=True)
class NodeType:
    node_id: int
    var: str
    L: int
    k: int
    stage: str
    stage_idx: int
    activation_name: str


def parse_activation_name(name: str) -> tuple[str, int, int, str]:
    """Parse 'var|Lx|ky|stage' → (var, L, k, stage)."""
    m = _ACT_RE.match(str(name).strip())
    if not m:
        raise ValueError(f"invalid activation name: {name!r}")
    return (
        m.group("var"),
        int(m.group("L")),
        int(m.group("k")),
        m.group("stage"),
    )


def stage_index(stage: str) -> int:
    try:
        return STAGE_NAMES.index(stage)
    except ValueError as e:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_NAMES}") from e


def build_node_dictionary(activation_names: Sequence[str]) -> list[NodeType]:
    """
    Freeze node types from checkpoint activation_names order.
    Key: (var, L, k, stage). IDs are 0..N-1 in the given order.
    """
    nodes: list[NodeType] = []
    seen: set[tuple[str, int, int, str]] = set()
    for i, name in enumerate(activation_names):
        var, L, k, stage = parse_activation_name(name)
        key = (var, L, k, stage)
        if key in seen:
            raise ValueError(f"duplicate node key {key} at index {i}")
        seen.add(key)
        nodes.append(
            NodeType(
                node_id=i,
                var=var,
                L=L,
                k=k,
                stage=stage,
                stage_idx=stage_index(stage),
                activation_name=str(name),
            )
        )
    return nodes


def node_dictionary_frame(nodes: Sequence[NodeType]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "node_id": n.node_id,
                "var": n.var,
                "L": n.L,
                "k": n.k,
                "stage": n.stage,
                "stage_idx": n.stage_idx,
                "activation_name": n.activation_name,
            }
            for n in nodes
        ]
    )


def make_synthetic_activation_names(
    vars_: Sequence[str],
    lengths: Sequence[int],
    k: int,
    stages: Sequence[str] = STAGE_NAMES,
) -> list[str]:
    """Deterministic names matching ShapeletRiskModel.activation_names order."""
    names: list[str] = []
    for v in vars_:
        for L in lengths:
            for ki in range(int(k)):
                for st in stages:
                    names.append(
                        f"{v}{ACTIVATION_NAME_SEP}L{int(L)}"
                        f"{ACTIVATION_NAME_SEP}k{ki}{ACTIVATION_NAME_SEP}{st}"
                    )
    return names


def select_active_events(
    activations: pd.DataFrame,
    *,
    a_min: float = A_MIN_EVENT,
) -> pd.DataFrame:
    """
    Select topology events: valid, a>=a_min, rank-1 within (ICUSTAY_ID, var, stage).

    Required columns:
      ICUSTAY_ID, var, stage, a, d_stage, L, k, is_valid
    Optional but preserved if present:
      SUBJECT_ID, split, HOSPITAL_MORTALITY, node_id, t_star, stage_idx, ...

    Tie-break (docs/13 §4.2): a desc, d_stage asc, L asc, k asc, node_id asc.
    """
    required = {
        "ICUSTAY_ID",
        "var",
        "stage",
        "a",
        "d_stage",
        "L",
        "k",
        "is_valid",
    }
    missing = required - set(activations.columns)
    if missing:
        raise ValueError(f"activations missing columns: {sorted(missing)}")

    df = activations.copy()
    if "node_id" not in df.columns:
        # stable placeholder for tie-break when dictionary not joined yet
        df["node_id"] = np.arange(len(df), dtype=np.int64)

    mask = df["is_valid"].astype(bool) & (df["a"].astype(float) >= float(a_min))
    cand = df.loc[mask].copy()
    if cand.empty:
        out = cand.copy()
        out["event_rank"] = pd.Series(dtype=np.int32)
        out["active_mask"] = pd.Series(dtype=bool)
        return out

    cand = cand.sort_values(
        by=["ICUSTAY_ID", "var", "stage", "a", "d_stage", "L", "k", "node_id"],
        ascending=[True, True, True, False, True, True, True, True],
        kind="mergesort",
    )
    cand["event_rank"] = (
        cand.groupby(["ICUSTAY_ID", "var", "stage"], sort=False).cumcount() + 1
    )
    active = cand.loc[cand["event_rank"] == 1].copy()
    active["active_mask"] = True
    return active.reset_index(drop=True)


def dense_activation_vector(
    patient_rows: pd.DataFrame,
    n_nodes: int,
    *,
    node_id_col: str = "node_id",
    a_col: str = "a",
) -> np.ndarray:
    """Build a_dense (N,) from all node rows for one patient (zeros if missing)."""
    a = np.zeros(n_nodes, dtype=np.float32)
    if patient_rows.empty:
        return a
    ids = patient_rows[node_id_col].astype(int).to_numpy()
    vals = patient_rows[a_col].astype(np.float32).to_numpy()
    for i, v in zip(ids, vals):
        if 0 <= i < n_nodes:
            a[i] = v
    return a


def active_mask_vector(
    active_events: pd.DataFrame,
    n_nodes: int,
    *,
    node_id_col: str = "node_id",
) -> np.ndarray:
    m = np.zeros(n_nodes, dtype=bool)
    if active_events.empty:
        return m
    for i in active_events[node_id_col].astype(int).to_numpy():
        if 0 <= i < n_nodes:
            m[i] = True
    return m
