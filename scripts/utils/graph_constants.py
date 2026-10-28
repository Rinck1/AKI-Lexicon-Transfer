"""
Frozen constants for patient-level Shapelet evolution graphs (docs/13).

Stage A: pure data rules only — no GNN, no real MIMIC I/O.
"""

from __future__ import annotations

from scripts.utils.shapelet_constants import (
    SHAPELET_STATIC_COLS,
    SHAPELET_VARS,
    STAGE_BOUNDS,
    T_HOURS,
)

# ---------------------------------------------------------------------------
# Variables (must match Shapelet skeleton; NOT TIER_A / TIER_B audit pools)
# ---------------------------------------------------------------------------
GRAPH_SHAPELET_VARS: tuple[str, ...] = tuple(SHAPELET_VARS)

# 11 sparse stage-background vars for z_sparse (docs/13 §3.5) — NOT TIER_B_SPARSE
GRAPH_SPARSE_VARS: tuple[str, ...] = (
    "creatinine",
    "bun",
    "potassium",
    "sodium",
    "bicarbonate",
    "lactate",
    "ph",
    "platelet",
    "wbc",
    "hemoglobin",
    "gcs_total",
)

GRAPH_STATIC_COLS: tuple[str, ...] = tuple(SHAPELET_STATIC_COLS)

STAGE_NAMES: tuple[str, ...] = tuple(name for name, _lo, _hi in STAGE_BOUNDS)
N_STAGES: int = len(STAGE_BOUNDS)
T_HOURS_GRAPH: int = int(T_HOURS)

# Per-stage sparse feature suffixes → 11 × 4 × 7 = 308
SPARSE_STAGE_FEATURE_SUFFIXES: tuple[str, ...] = (
    "last",
    "min",
    "max",
    "delta_from_previous_stage_last",
    "is_measured",
    "measured_fraction",
    "hours_since_last_measurement_at_stage_end",
)
N_SPARSE_FEATURES: int = (
    len(GRAPH_SPARSE_VARS) * N_STAGES * len(SPARSE_STAGE_FEATURE_SUFFIXES)
)
N_STATIC_FEATURES: int = len(GRAPH_STATIC_COLS)

# ---------------------------------------------------------------------------
# Event selection (topology vs dense activations)
# ---------------------------------------------------------------------------
# Soft activations are almost always >0; only high-a valid peaks enter topology.
A_MIN_EVENT: float = 0.5
# Max active events per patient: 7 vars × 4 stages
MAX_ACTIVE_EVENTS_PER_PATIENT: int = len(GRAPH_SHAPELET_VARS) * N_STAGES  # 28

# Motif-family node scheme: (var, motif_family, stage) = 7 × 3 × 4 = 84
# Distinct from the 252-node (var, L, k, stage) scheme and from the 28-node
# (var, stage) collapse. Mapping is frozen from Shapelet templates only.
N_MOTIF_FAMILIES: int = 3
N_MOTIF_NODES: int = len(GRAPH_SHAPELET_VARS) * N_MOTIF_FAMILIES * N_STAGES  # 84
NODE_SCHEME_LK: str = "var_L_k_stage"
NODE_SCHEME_MOTIF: str = "var_motif_family_stage"

# ---------------------------------------------------------------------------
# Edge policy (hours after onset, using t_star)
# ---------------------------------------------------------------------------
CO_OCCUR_MAX_HOURS: float = 2.0
PRECEDES_MIN_HOURS: float = 2.0
PRECEDES_MAX_HOURS: float = 12.0

MAX_CO_OCCUR_NEIGHBORS: int = 4  # per node
MAX_PRECEDES_OUT: int = 2  # per source event

# Canonical storage upper bound (hard fail if exceeded)
MAX_CANONICAL_EDGES: int = 128
# Soft design bound at 28 events: PRECEDES<=56, CO_OCCUR<=56
MAX_PRECEDES_SOFT: int = MAX_ACTIVE_EVENTS_PER_PATIENT * MAX_PRECEDES_OUT
MAX_CO_OCCUR_SOFT: int = (MAX_ACTIVE_EVENTS_PER_PATIENT * MAX_CO_OCCUR_NEIGHBORS) // 2

# Edge type ids (stored on edges; GNN expands CO_OCCUR bidirectionally later)
EDGE_TYPE_CO_OCCUR: int = 0
EDGE_TYPE_PRECEDES: int = 1
EDGE_TYPE_NAMES: tuple[str, ...] = ("CO_OCCUR", "PRECEDES")

# Continuous edge feature dim (docs/13 §5.6):
# abs_delta_t/12, a_u*a_w, same_variable, same_stage, log1p(support)/log1p(n_train)
N_EDGE_ATTR: int = 5

# Train edge vocabulary support (docs/13 §5.5)
EDGE_SUPPORT_FRAC: float = 0.005
EDGE_SUPPORT_ABS_MIN: int = 20


def min_edge_support(n_train_stays: int) -> int:
    """Minimum distinct train patients supporting an edge type."""
    import math

    return max(EDGE_SUPPORT_ABS_MIN, int(math.ceil(EDGE_SUPPORT_FRAC * n_train_stays)))


# ---------------------------------------------------------------------------
# GNN architecture (docs/13 §8) — Stage B freezes these dims
# ---------------------------------------------------------------------------
NODE_TYPE_DIM: int = 16
HIDDEN_DIM: int = 64
EMPTY_GRAPH_EMBED_DIM: int = 96  # h_node(64) || h_edge(32)
EDGE_ENC_HIDDEN: int = 16
EDGE_ENC_OUT: int = 16
EDGE_POOL_DIM: int = 32
N_GNN_LAYERS: int = 2
GNN_DROPOUT: float = 0.20

# Internal message relations (model expands canonical edges)
REL_FROM_PREDECESSOR: int = 0  # PRECEDES u->w : w reads u
REL_FROM_SUCCESSOR: int = 1  # PRECEDES u->w : u reads w
REL_CO_OCCUR: int = 2
N_MESSAGE_RELS: int = 3
MESSAGE_REL_NAMES: tuple[str, ...] = (
    "FROM_PREDECESSOR",
    "FROM_SUCCESSOR",
    "CO_OCCUR",
)

# Non-graph branches + fusion (docs/13 §8.8)
A_PROJ_DIM: int = 64
SPARSE_PROJ_DIM: int = 64
STATIC_PROJ_DIM: int = 32
# h_graph(96) + a(64) + sparse(64) + static(32) = 256
FUSION_DIM: int = EMPTY_GRAPH_EMBED_DIM + A_PROJ_DIM + SPARSE_PROJ_DIM + STATIC_PROJ_DIM
RISK_HIDDEN: int = 128

# Training defaults (docs/13 §9.3) — Stage B records; real train is Stage D
GNN_LR: float = 1e-3
GNN_WEIGHT_DECAY: float = 1e-4
BATCH_GRAPHS: int = 32
BATCH_GRAPHS_OOM: int = 16
GNN_MAX_EPOCHS: int = 100
GNN_PATIENCE: int = 15
GNN_MIN_DELTA: float = 1e-4
GNN_GRAD_CLIP: float = 1.0
GNN_SEED: int = 42
GNN_MONITOR: str = "val_AUPRC"

# Gated edge readout (independent sigmoid; no edge softmax)
# Frozen for the 84-node comparison — not a search.
EDGE_GATE_LAMBDA_SPARSE: float = 0.02
EDGE_GATE_LAMBDA_STABLE: float = 0.05
EDGE_GATE_JITTER_A: float = 0.01
EDGE_GATE_JITTER_HOURS: float = 1.0
EDGE_GATE_INIT_BIAS: float = -0.5

# Hard-Concrete / L0 unified edge mask (not a search)
# log_alpha ≈ -3 → P(z≠0) ≈ 0.20 under default stretch params.
EDGE_L0_LAMBDA: float = 0.05
EDGE_L0_LAMBDA_RANK: float = 0.02
EDGE_L0_LOG_ALPHA_INIT: float = -3.0
EDGE_VALUE_MAX: float = 2.0

# Fixed-budget straight-through top-k (not a search). K = max(1, ceil(0.2 E)).
EDGE_TOPK_FRACTION: float = 0.20
EDGE_TOPK_LAMBDA_RANK: float = 0.02

# ---------------------------------------------------------------------------
# Activation name / node key format (must match ShapeletRiskModel.activation_names)
# ---------------------------------------------------------------------------
# Example: urine_output_ml|L6|k0|B1_0_6
ACTIVATION_NAME_SEP: str = "|"
