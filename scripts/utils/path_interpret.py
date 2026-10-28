"""Train-only nested interpretability for length-3/4 paths (docs/20 §4 length priority)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from scripts.utils.path_protocol import path_key

VAR_ZH = {
    "urine_output_ml": "尿量",
    "creatinine": "肌酐",
    "bun": "尿素氮",
    "mbp": "平均动脉压",
    "lactate": "乳酸",
    "gcs_total": "GCS",
    "sbp": "收缩压",
    "dbp": "舒张压",
    "heart_rate": "心率",
    "respiratory_rate": "呼吸频率",
    "spo2": "血氧",
    "potassium": "血钾",
    "sodium": "血钠",
    "bicarbonate": "碳酸氢根",
    "ph": "pH",
    "platelet": "血小板",
    "wbc": "白细胞",
    "hemoglobin": "血红蛋白",
}

SUPPORT_RATIO_MAX = 0.90


def contiguous_subpaths(atoms: tuple[str, ...], *, min_len: int = 2) -> list[tuple[str, ...]]:
    """Proper contiguous sub-sequences shorter than the full path."""
    n = len(atoms)
    out: list[tuple[str, ...]] = []
    for length in range(int(min_len), n):
        for i in range(0, n - length + 1):
            out.append(atoms[i : i + length])
    return out


def grow_parent(atoms: tuple[str, ...]) -> tuple[str, ...]:
    if len(atoms) < 3:
        raise ValueError("grow_parent requires length >= 3")
    return atoms[:-1]


def stay_contains_path(stage_atoms: list[tuple[int, str, bool]], path: tuple[str, ...]) -> bool:
    loc: dict[str, int] = {}
    for st, atom, _low in stage_atoms:
        loc[str(atom)] = int(st)
    if any(a not in loc for a in path):
        return False
    ranks = [loc[a] for a in path]
    return all(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1))


def indicator_for_paths(
    stay_ids: list[int],
    stay_atoms: dict[int, list[tuple[int, str, bool]]],
    paths: list[tuple[str, ...]],
) -> np.ndarray:
    ind = np.zeros((len(stay_ids), len(paths)), dtype=np.float64)
    for i, sid in enumerate(stay_ids):
        atoms = stay_atoms.get(int(sid), [])
        for j, p in enumerate(paths):
            if stay_contains_path(atoms, p):
                ind[i, j] = 1.0
    return ind


def atom_gloss(atom: str) -> str:
    parts = [p.strip() for p in str(atom).split("|")]
    if len(parts) == 3 and parts[2] in {"升高", "下降"}:
        var, stage, direction = parts
        return f"{VAR_ZH.get(var, var)}在{stage}{direction}"
    if len(parts) == 3:
        var, family, stage = parts
        return f"{VAR_ZH.get(var, var)}在{stage}呈{family}"
    return str(atom)


def path_narrative(atoms: tuple[str, ...], *, sign: int) -> str:
    bits = [atom_gloss(a) for a in atoms]
    if not bits:
        body = "空路径"
    elif len(bits) == 1:
        body = bits[0]
    elif len(bits) == 2:
        body = f"先{bits[0]}，随后{bits[1]}"
    else:
        body = f"先{bits[0]}" + "".join(f"，再{b}" for b in bits[1:])
    if sign > 0:
        tail = "与住院死亡风险升高同向"
    elif sign < 0:
        tail = "与住院死亡风险降低同向（保护/幸存者轨迹，不作机制承诺）"
    else:
        tail = "方向不定"
    return f"{body}。{tail}。"


def is_redundant(*, support_long: int, support_short: int, auprc_gain: float, min_gain: float) -> bool:
    if support_short <= 0:
        return True
    ratio = float(support_long) / float(support_short)
    return ratio >= SUPPORT_RATIO_MAX or (not np.isfinite(auprc_gain)) or float(auprc_gain) < float(min_gain)


def motif_lowconf_rate(atoms: tuple[str, ...], atom_lowconf: dict[str, float]) -> float:
    motif = [a for a in atoms if not a.endswith("|升高") and not a.endswith("|下降")]
    if not motif:
        return 0.0
    return float(np.mean([float(atom_lowconf.get(a, 0.0)) for a in motif]))


def atom_lowconf_map(rows: list[tuple[str, bool, str]]) -> dict[str, float]:
    """rows: (atom, low_conf, kind). Rate among train motif occurrences."""
    hits: dict[str, list[float]] = defaultdict(list)
    for atom, low, kind in rows:
        if str(kind) != "motif":
            continue
        hits[str(atom)].append(1.0 if low else 0.0)
    return {a: float(np.mean(v)) for a, v in hits.items()}
