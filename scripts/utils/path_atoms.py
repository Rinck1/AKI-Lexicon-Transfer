"""Build motif and lab stage atoms (docs/20 §3). Thresholds are train-only."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from scripts.utils.graph_constants import (
    A_MIN_EVENT,
    GRAPH_SPARSE_VARS,
    SPARSE_STAGE_FEATURE_SUFFIXES,
    STAGE_NAMES,
)
from scripts.utils.graph_sparse import sparse_feature_names
from scripts.utils.path_protocol import (
    MIN_POS_DELTA,
    fold_stage,
    lab_atom_name,
    motif_atom_name,
)

DIR_UP = "升高"
DIR_DOWN = "下降"
_SUFFIX_LAST = "last"
_SUFFIX_DELTA = "delta_from_previous_stage_last"
_SUFFIX_MEAS = "is_measured"


def _col(var: str, stage: str, suf: str) -> str:
    return f"{var}|{stage}|{suf}"


def motif_atoms_from_events(events: pd.DataFrame, *, a_min: float = A_MIN_EVENT) -> pd.DataFrame:
    """One row per winner event with a>=a_min."""
    need = [
        "ICUSTAY_ID",
        "SUBJECT_ID",
        "split",
        "HOSPITAL_MORTALITY",
        "var",
        "family_label",
        "stage",
        "stage_idx",
        "a",
        "family_low_confidence",
    ]
    miss = [c for c in need if c not in events.columns]
    if miss:
        raise KeyError(f"events missing {miss}")
    keep = events["a"].to_numpy() >= float(a_min)
    if "is_active" in events.columns:
        keep = keep & events["is_active"].fillna(False).to_numpy()
    df = events.loc[keep, need].copy()
    df["atom"] = [
        motif_atom_name(str(v), str(f), str(s))
        for v, f, s in zip(df["var"], df["family_label"], df["stage"])
    ]
    df["kind"] = "motif"
    df["direction"] = ""
    df["stage_fold"] = [fold_stage(str(s)) for s in df["stage"]]
    return df.reset_index(drop=True)


def fit_lab_thresholds(
    raw: np.ndarray,
    stay_ids: np.ndarray,
    train_ids: Iterable[int],
    *,
    names: list[str] | None = None,
) -> dict:
    """Train-only Δ medians (B2–B4) and B1 last median. Missing stays stay silent."""
    names = names or sparse_feature_names()
    name_to_i = {n: i for i, n in enumerate(names)}
    train_set = {int(x) for x in train_ids}
    mask = np.array([int(s) in train_set for s in stay_ids], dtype=bool)
    X = raw[mask]
    out: dict = {"vars": list(GRAPH_SPARSE_VARS), "min_pos_delta": MIN_POS_DELTA, "by_var": {}}
    for var in GRAPH_SPARSE_VARS:
        pos_d: list[float] = []
        neg_d: list[float] = []
        b1_last: list[float] = []
        for stage in STAGE_NAMES:
            im = name_to_i[_col(var, stage, _SUFFIX_MEAS)]
            idl = name_to_i[_col(var, stage, _SUFFIX_DELTA)]
            il = name_to_i[_col(var, stage, _SUFFIX_LAST)]
            meas = X[:, im] > 0.5
            if stage == STAGE_NAMES[0]:
                vals = X[meas, il]
                b1_last.extend(float(v) for v in vals if np.isfinite(v))
            else:
                deltas = X[meas, idl]
                for d in deltas:
                    if not np.isfinite(d):
                        continue
                    if d > 0:
                        pos_d.append(float(d))
                    elif d < 0:
                        neg_d.append(float(d))
        pos_med = float(np.median(pos_d)) if len(pos_d) >= MIN_POS_DELTA else None
        neg_med = float(np.median(neg_d)) if len(neg_d) >= MIN_POS_DELTA else None
        last_med = float(np.median(b1_last)) if len(b1_last) >= MIN_POS_DELTA else None
        out["by_var"][var] = {
            "pos_delta_median": pos_med,
            "neg_delta_median": neg_med,
            "b1_last_median": last_med,
            "n_pos_delta": len(pos_d),
            "n_neg_delta": len(neg_d),
            "n_b1_last": len(b1_last),
        }
    return out


def _direction(delta: float, pos_med: float | None, neg_med: float | None) -> str | None:
    if not np.isfinite(delta):
        return None
    if pos_med is not None and delta > float(pos_med):
        return DIR_UP
    if neg_med is not None and delta < float(neg_med):
        return DIR_DOWN
    return None


def lab_atoms_from_raw(
    raw: np.ndarray,
    stay_meta: pd.DataFrame,
    thresholds: dict,
    *,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Apply frozen train thresholds. stay_meta aligned with raw rows."""
    names = names or sparse_feature_names()
    name_to_i = {n: i for i, n in enumerate(names)}
    if len(stay_meta) != raw.shape[0]:
        raise ValueError("stay_meta rows must match raw")
    rows: list[dict] = []
    by_var = thresholds["by_var"]
    for r, rec in enumerate(stay_meta.itertuples(index=False)):
        sid = int(rec.ICUSTAY_ID)
        subj = int(rec.SUBJECT_ID)
        split = str(rec.split)
        y = float(rec.HOSPITAL_MORTALITY)
        for var in GRAPH_SPARSE_VARS:
            spec = by_var[var]
            pos_med = spec.get("pos_delta_median")
            neg_med = spec.get("neg_delta_median")
            last_med = spec.get("b1_last_median")
            for s_i, stage in enumerate(STAGE_NAMES):
                im = name_to_i[_col(var, stage, _SUFFIX_MEAS)]
                if raw[r, im] <= 0.5:
                    continue
                if s_i == 0:
                    last = raw[r, name_to_i[_col(var, stage, _SUFFIX_LAST)]]
                    if last_med is None or not np.isfinite(last):
                        continue
                    direction = _direction(float(last) - float(last_med), pos_med, neg_med)
                else:
                    delta = raw[r, name_to_i[_col(var, stage, _SUFFIX_DELTA)]]
                    direction = _direction(float(delta), pos_med, neg_med)
                if direction is None:
                    continue
                rows.append(
                    {
                        "ICUSTAY_ID": sid,
                        "SUBJECT_ID": subj,
                        "split": split,
                        "HOSPITAL_MORTALITY": y,
                        "var": var,
                        "family_label": "",
                        "stage": stage,
                        "stage_idx": s_i,
                        "a": np.nan,
                        "family_low_confidence": False,
                        "atom": lab_atom_name(var, stage, direction),
                        "kind": "lab",
                        "direction": direction,
                        "stage_fold": fold_stage(stage),
                    }
                )
    return pd.DataFrame(rows)


def atoms_by_stay(atoms: pd.DataFrame) -> dict[int, list[tuple[int, str, bool]]]:
    """stay → list of (stage_idx, atom, low_conf)."""
    out: dict[int, list[tuple[int, str, bool]]] = {}
    for rec in atoms.itertuples(index=False):
        sid = int(rec.ICUSTAY_ID)
        out.setdefault(sid, []).append(
            (int(rec.stage_idx), str(rec.atom), bool(rec.family_low_confidence))
        )
    return out
