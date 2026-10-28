"""
Shapelet hyperparameters — approved package for RTX 4070 Laptop (8GB).

See docs/12_Shapelet实现规范.md §9A / §9B (device-aware freeze).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Variables (three-layer design: Layer-1 skeleton only)
# ---------------------------------------------------------------------------
SHAPELET_VARS: tuple[str, ...] = (
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "respiratory_rate",
    "spo2",
    "urine_output_ml",
)

SHAPELET_STATIC_COLS: tuple[str, ...] = (
    "AGE",
    "GENDER_M",
    "onset_stage",
    "onset_hour_from_icu",
    "CHF",
    "Diabetes",
    "HTN",
    "CKD_FLAG",
    "COPD",
    "LiverDisease",
    "Arrhythmia",
)

T_HOURS = 48

STAGE_BOUNDS: tuple[tuple[str, int, int], ...] = (
    ("B1_0_6", 0, 6),
    ("B2_6_12", 6, 12),
    ("B3_12_24", 12, 24),
    ("B4_24_48", 24, 48),
)

# ---------------------------------------------------------------------------
# Structure: debug → pilot → paper final
# ---------------------------------------------------------------------------
# Smoke / single-variable debug
DEBUG_LENGTHS: tuple[int, ...] = (6,)
DEBUG_K = 2

# Phase-1 full training on 7 vars (implementation milestone)
PILOT_LENGTHS: tuple[int, ...] = (6,)
PILOT_K = 3

# Paper main analysis (scientific target — not optional forever)
FINAL_LENGTHS: tuple[int, ...] = (4, 6, 12)
FINAL_K = 3

# Default "run mode" for training CLI: debug | pilot | final
DEFAULT_RUN_MODE = "pilot"

# Backward-compatible aliases used by early drafts
DEFAULT_LENGTHS = PILOT_LENGTHS
DEFAULT_K = PILOT_K

# ---------------------------------------------------------------------------
# Soft-DTW / activation (temperatures SPLIT — do not share one τ)
# ---------------------------------------------------------------------------
GAMMA_SOFTDTW = 0.1  # Soft-DTW path softness
TAU_POOL = 0.1  # stage softmin / soft-argmin concentration
TAU_ACTIVATION = 0.5  # distance → activation sigmoid steepness

# Missing-aware cost (assumes distance normalized by L — see shapelet_cost)
LAMBDA_M = 0.5
LAMBDA_DELTA = 0.1
DELTA_MAX = 48.0
MIN_OBS_FRAC = 0.5  # window must have ≥50% measured hours else skip match

# Sakoe-Chiba style band (max |i-j| offset); reduces pathological warping
WARPING_RADIUS: dict[int, int] = {
    4: 1,
    6: 1,
    12: 2,
}

# Activation scheme B: a = sigmoid((q - d) / TAU_ACTIVATION)
# q: per (var, L, k) = median of train distances after freeze; unified quantile policy
Q_QUANTILE_POLICY = 0.50  # use same quantile for all templates (not per-node val search)

# Graph threshold (post-hoc; not used in Shapelet loss)
# a_min chosen on val when building patient graphs
A_MIN_DEFAULT = 0.5

# ---------------------------------------------------------------------------
# Loss regularizers
# ---------------------------------------------------------------------------
LAMBDA_DIV = 0.1
LAMBDA_TIME = 0.01
LAMBDA_SP = 1e-3

# ---------------------------------------------------------------------------
# Optimizer (grouped LRs — do NOT use single lr=1e-2 for all)
# ---------------------------------------------------------------------------
LR_SHAPELET = 3e-3
LR_ALPHA = 1e-3
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

MAX_EPOCHS = 100
PATIENCE = 15
MONITOR = "val_auprc"  # early stop metric
SEED = 42

# RTX 4070 Laptop 8GB: micro-batch + accumulation → effective 32
BATCH_STAYS = 16
GRAD_ACCUM = 2
# OOM fallback: BATCH_STAYS=8, GRAD_ACCUM=4

DTYPE_SOFTDTW = "float32"  # never fp16 for Soft-DTW DP
FULL_STARTS_FOR_FINAL = True  # pilot/final: all valid starts; no start subsampling

# Legacy names (map to new)
DEFAULT_GAMMA = GAMMA_SOFTDTW
DEFAULT_TAU = TAU_ACTIVATION
DEFAULT_LAMBDA_M = LAMBDA_M
DEFAULT_LAMBDA_DELTA = LAMBDA_DELTA
DEFAULT_DELTA_MAX = DELTA_MAX
DEFAULT_LR = LR_SHAPELET
DEFAULT_WEIGHT_DECAY = WEIGHT_DECAY
DEFAULT_EPOCHS = MAX_EPOCHS
DEFAULT_PATIENCE = PATIENCE
DEFAULT_BATCH_STAYS = BATCH_STAYS


def lengths_and_k(mode: str) -> tuple[tuple[int, ...], int]:
    mode = mode.lower()
    if mode == "debug":
        return DEBUG_LENGTHS, DEBUG_K
    if mode == "pilot":
        return PILOT_LENGTHS, PILOT_K
    if mode == "final":
        return FINAL_LENGTHS, FINAL_K
    raise ValueError(f"unknown run mode {mode!r}; expected debug, pilot, or final")


def node_dict_size(mode: str = "final") -> int:
    """Number of global node types: 7 × K × n_L × 4 stages."""
    lengths, k = lengths_and_k(mode)
    return len(SHAPELET_VARS) * k * len(lengths) * len(STAGE_BOUNDS)
