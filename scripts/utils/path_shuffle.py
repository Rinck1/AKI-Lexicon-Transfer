"""Within-stay stage permutation for the path-mining time-shuffle control.

Preserves each stay's concept bag (variable, motif family or lab direction)
and the occupancy of stage slots. Rewrites atom names to the shuffled stage.
Train only; never touches val/holdout thresholds or the frozen 15-path set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.utils.graph_constants import STAGE_NAMES
from scripts.utils.path_atoms import lab_atom_name, motif_atom_name
from scripts.utils.path_protocol import fold_stage, refuse_test_split

STAGE_TOKENS: tuple[str, ...] = ("B1", "B2", "B3", "B4")


def strip_stage_token(atom: str) -> str:
    """Drop B1–B4 from an atom name. Motif → var|family; lab → var|direction."""
    parts = [p for p in str(atom).split("|") if p not in STAGE_TOKENS]
    return "|".join(parts)


def concept_identity_from_row(row: pd.Series) -> str:
    kind = str(row.get("kind", ""))
    var = str(row["var"])
    if kind == "lab":
        return f"{var}|{row['direction']}"
    family = str(row.get("family_label", ""))
    if family:
        return f"{var}|{family}"
    return strip_stage_token(str(row["atom"]))


def _rebuild_atom(
    *,
    kind: str,
    var: str,
    family_label: str,
    direction: str,
    stage_idx: int,
) -> tuple[str, str, str]:
    if not 0 <= int(stage_idx) < len(STAGE_NAMES):
        raise ValueError(f"stage_idx out of range: {stage_idx}")
    stage = STAGE_NAMES[int(stage_idx)]
    if str(kind) == "lab":
        atom = lab_atom_name(str(var), stage, str(direction))
    else:
        atom = motif_atom_name(str(var), str(family_label), stage)
    return atom, stage, fold_stage(stage)


def shuffle_train_atom_stages(
    atoms: pd.DataFrame,
    *,
    seed: int,
    split_col: str = "split",
) -> pd.DataFrame:
    """Permute stage_idx within each train stay; rebuild atom / stage columns.

    Stays with one atom are unchanged. Non-train rows are dropped so val/holdout
    cannot leak into the remine directory.
    """
    if split_col in atoms.columns:
        for sp in atoms[split_col].astype(str).unique():
            refuse_test_split(sp)
        work = atoms.loc[atoms[split_col].astype(str) == "train"].copy()
    else:
        work = atoms.copy()
    if work.empty:
        return work.reset_index(drop=True)

    need = ["ICUSTAY_ID", "stage_idx", "atom", "kind", "var"]
    miss = [c for c in need if c not in work.columns]
    if miss:
        raise KeyError(f"atoms missing {miss}")

    rng = np.random.default_rng(int(seed))
    new_idx = work["stage_idx"].to_numpy(dtype=np.int16, copy=True)
    stay_ids = work["ICUSTAY_ID"].to_numpy()
    order = np.argsort(stay_ids, kind="mergesort")
    sorted_stays = stay_ids[order]
    breaks = np.flatnonzero(np.diff(sorted_stays) != 0) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(order)]])
    for a, b in zip(starts, ends):
        loc = order[a:b]
        if len(loc) <= 1:
            continue
        new_idx[loc] = rng.permutation(new_idx[loc])
    work = work.copy()
    work["stage_idx"] = new_idx

    atoms_new: list[str] = []
    stages_new: list[str] = []
    folds_new: list[str] = []
    family = work["family_label"].astype(str) if "family_label" in work.columns else pd.Series("", index=work.index)
    direction = work["direction"].astype(str) if "direction" in work.columns else pd.Series("", index=work.index)
    kind = work["kind"].astype(str)
    var = work["var"].astype(str)
    for k, v, fam, direc, st in zip(kind, var, family, direction, work["stage_idx"].to_numpy()):
        atom, stage, fold = _rebuild_atom(
            kind=k, var=v, family_label=fam, direction=direc, stage_idx=int(st)
        )
        atoms_new.append(atom)
        stages_new.append(stage)
        folds_new.append(fold)
    work["atom"] = atoms_new
    work["stage"] = stages_new
    if "stage_fold" in work.columns:
        work["stage_fold"] = folds_new
    return work.reset_index(drop=True)


def identities_by_stay(atoms: pd.DataFrame) -> dict[int, set[str]]:
    """Stage-stripped concept identities present in each stay."""
    out: dict[int, set[str]] = {}
    for sid, atom in zip(atoms["ICUSTAY_ID"].to_numpy(), atoms["atom"].astype(str)):
        out.setdefault(int(sid), set()).add(strip_stage_token(atom))
    return out


def path_identities(atoms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(strip_stage_token(a) for a in atoms)


def cooccur_indicator_matrix(
    stay_ids: list[int],
    stay_idents: dict[int, set[str]],
    paths: list[tuple[str, ...]],
) -> np.ndarray:
    """1 if every stage-stripped identity of the path is present in the stay."""
    ind = np.zeros((len(stay_ids), len(paths)), dtype=np.float64)
    idents = [path_identities(p) for p in paths]
    for i, sid in enumerate(stay_ids):
        have = stay_idents.get(int(sid), set())
        for j, ids in enumerate(idents):
            if ids and all(x in have for x in ids):
                ind[i, j] = 1.0
    return ind


def stage_occupancy_signature(atoms: pd.DataFrame) -> pd.Series:
    """Per-stay tuple of sorted stage_idx; shuffle must preserve this."""
    return (
        atoms.groupby("ICUSTAY_ID")["stage_idx"]
        .apply(lambda s: tuple(sorted(int(x) for x in s)))
        .sort_index()
    )


def identity_bag_signature(atoms: pd.DataFrame) -> pd.Series:
    return (
        atoms.groupby("ICUSTAY_ID")["atom"]
        .apply(lambda s: tuple(sorted(strip_stage_token(str(a)) for a in s)))
        .sort_index()
    )


def n_stage_concept_pairs_changed(original: pd.DataFrame, shuffled: pd.DataFrame) -> int:
    """Stays where at least one (identity, stage_idx) pair moved."""

    def pairs(df: pd.DataFrame) -> dict[int, set[tuple[str, int]]]:
        out: dict[int, set[tuple[str, int]]] = {}
        for sid, atom, st in zip(
            df["ICUSTAY_ID"].to_numpy(),
            df["atom"].astype(str),
            df["stage_idx"].to_numpy(),
        ):
            out.setdefault(int(sid), set()).add((strip_stage_token(atom), int(st)))
        return out

    a, b = pairs(original), pairs(shuffled)
    n = 0
    for sid, pa in a.items():
        if pa != b.get(sid, set()):
            n += 1
    return n
