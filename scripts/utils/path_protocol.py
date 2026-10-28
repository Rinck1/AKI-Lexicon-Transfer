"""Frozen temporal-path mining rules (docs/20). No GNN edge scores."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.utils.graph_constants import EDGE_SUPPORT_ABS_MIN, EDGE_SUPPORT_FRAC
from scripts.utils.graph_interpretability import fold_stage_label

PATH_SEED = 42
PATH_MIN_LEN = 2
PATH_MAX_LEN = 4
N_BOOT = 100
BOOT_SUBJECT_FRAC = 0.80
BOOT_RETAIN = 0.70
N_LOGIT_BOOT_CI = 200
MIN_POS_DELTA = 20
MAX_SELECTED_PATHS = 15
MIN_AUPRC_GAIN = 0.001
PATH_GROW_PARENT_TOP = 120
VAL_MIN_SUPPORT = 20
LOW_CONF_ENRICH_TOL = 0.05

RENAL_PERFUSION_TOKENS: tuple[str, ...] = (
    "urine_output_ml",
    "creatinine",
    "bun",
    "mbp",
    "lactate",
    "gcs_total",
)


def refuse_test_path(path: Path | str) -> None:
    text = str(path).replace("\\", "/").lower()
    if "test" in Path(text).name.lower() or "/test" in text or "\\test" in str(path).lower():
        raise SystemExit(f"refusing a path that looks like test: {path}")


def refuse_test_split(split: str) -> None:
    if str(split).strip().lower() == "test":
        raise SystemExit("refusing test split (docs/20 lock)")


def sha256_canonical(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def refuse_if_holdout_done(path: Path | str) -> None:
    p = Path(path)
    if p.is_file():
        raise SystemExit(f"holdout already used once (docs/20): {p}")


def min_path_support(n_train_stays: int) -> int:
    return max(EDGE_SUPPORT_ABS_MIN, int(math.ceil(EDGE_SUPPORT_FRAC * int(n_train_stays))))


def fold_stage(stage: str) -> str:
    return fold_stage_label(stage)


def motif_atom_name(var: str, family_label: str, stage: str) -> str:
    return f"{var}|{family_label}|{fold_stage(stage)}"


def lab_atom_name(var: str, stage: str, direction: str) -> str:
    return f"{var}|{fold_stage(stage)}|{direction}"


def path_key(atoms: tuple[str, ...]) -> str:
    return " → ".join(atoms)


def parse_path_key(key: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(key).split("→") if part.strip())


def is_readable(atoms: tuple[str, ...]) -> bool:
    blob = " ".join(atoms)
    return any(tok in blob for tok in RENAL_PERFUSION_TOKENS)


def stage_rank(atom: str) -> int:
    for i, tag in enumerate(("B1", "B2", "B3", "B4")):
        if f"|{tag}|" in f"|{atom}|" or atom.endswith(f"|{tag}") or f"|{tag}|" in atom:
            return i
        if atom.endswith(tag) or f"|{tag}" in atom:
            return i
    return -1
