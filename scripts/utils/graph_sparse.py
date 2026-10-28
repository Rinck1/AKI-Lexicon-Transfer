"""
Sparse stage-background features z_sparse (docs/13 §3.5).

11 vars × 4 stages × 7 stats = 308. Train-only median fill + z-score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.utils.graph_constants import (
    GRAPH_SPARSE_VARS,
    N_SPARSE_FEATURES,
    SPARSE_STAGE_FEATURE_SUFFIXES,
    STAGE_NAMES,
    T_HOURS_GRAPH,
)
from scripts.utils.shapelet_constants import STAGE_BOUNDS


def sparse_feature_names(
    vars_: Sequence[str] = GRAPH_SPARSE_VARS,
    stages: Sequence[str] = STAGE_NAMES,
    suffixes: Sequence[str] = SPARSE_STAGE_FEATURE_SUFFIXES,
) -> list[str]:
    names: list[str] = []
    for v in vars_:
        for st in stages:
            for suf in suffixes:
                names.append(f"{v}|{st}|{suf}")
    assert len(names) == N_SPARSE_FEATURES
    return names


def _stage_slices() -> list[tuple[str, int, int]]:
    return [(name, int(lo), int(hi)) for name, lo, hi in STAGE_BOUNDS]


def _series_for_var(
    g: pd.DataFrame,
    var: str,
    t_hours: int = T_HOURS_GRAPH,
) -> tuple[np.ndarray, np.ndarray]:
    """(T,) values (nan if missing) and measured mask."""
    mcol = f"{var}_is_measured"
    by_h = g.set_index("HOUR") if "HOUR" in g.columns else g
    x = np.full(t_hours, np.nan, dtype=np.float64)
    m = np.zeros(t_hours, dtype=bool)
    for h in range(t_hours):
        if h not in by_h.index:
            continue
        row = by_h.loc[h]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        val = row[var] if var in row.index else np.nan
        measured = bool(row[mcol]) if mcol in row.index else pd.notna(val)
        if measured and pd.notna(val):
            x[h] = float(val)
            m[h] = True
    return x, m


def raw_sparse_vector_for_stay(
    g: pd.DataFrame | None,
    vars_: Sequence[str] = GRAPH_SPARSE_VARS,
    t_hours: int = T_HOURS_GRAPH,
) -> np.ndarray:
    """
    Compute raw (pre-impute, pre-scale) sparse features for one stay.
    Missing continuous stats stay as nan; flags are 0/1.
    """
    out = np.full(N_SPARSE_FEATURES, np.nan, dtype=np.float64)
    stages = _stage_slices()
    if g is None or g.empty:
        # all flags 0
        idx = 0
        for _v in vars_:
            for _st in stages:
                # last,min,max,delta nan; is_measured, measured_frac 0; hours nan
                out[idx + 4] = 0.0
                out[idx + 5] = 0.0
                idx += 7
        return out.astype(np.float32)

    idx = 0
    prev_last: dict[str, float] = {}
    for var in vars_:
        x, m = _series_for_var(g, var, t_hours=t_hours)
        last_seen_hour = -1
        last_seen_val = np.nan
        for s_i, (_name, lo, hi) in enumerate(stages):
            seg_m = m[lo:hi]
            seg_x = x[lo:hi]
            n_hours = max(hi - lo, 1)
            n_meas = int(seg_m.sum())
            is_meas = 1.0 if n_meas > 0 else 0.0
            meas_frac = float(n_meas) / float(n_hours)

            if n_meas > 0:
                vals = seg_x[seg_m]
                last_v = float(vals[-1])
                min_v = float(vals.min())
                max_v = float(vals.max())
                # update last-seen within stage (hour of last measurement)
                meas_hours = np.where(seg_m)[0] + lo
                last_seen_hour = int(meas_hours[-1])
                last_seen_val = last_v
            else:
                last_v = np.nan
                min_v = np.nan
                max_v = np.nan

            if s_i == 0:
                delta = np.nan if np.isnan(last_v) else 0.0
            else:
                pl = prev_last.get(var, np.nan)
                if np.isnan(last_v) or np.isnan(pl):
                    delta = np.nan
                else:
                    delta = last_v - pl

            # hours since last measurement at stage end (any history up to hi-1)
            if last_seen_hour < 0:
                # search full history before stage end
                hist_m = m[:hi]
                if hist_m.any():
                    last_h = int(np.where(hist_m)[0][-1])
                    hrs = float((hi - 1) - last_h)
                else:
                    hrs = float(t_hours)
            else:
                hrs = float((hi - 1) - last_seen_hour)
            hrs = min(max(hrs, 0.0), float(t_hours)) / float(t_hours)

            out[idx + 0] = last_v
            out[idx + 1] = min_v
            out[idx + 2] = max_v
            out[idx + 3] = delta
            out[idx + 4] = is_meas
            out[idx + 5] = meas_frac
            out[idx + 6] = hrs

            if not np.isnan(last_v):
                prev_last[var] = last_v
            idx += 7

    return out.astype(np.float32)


@dataclass
class SparseScaler:
    """Median fill + mean/std on continuous slots; flags left as-is."""

    median: np.ndarray  # (308,)
    mean: np.ndarray
    std: np.ndarray
    is_continuous: np.ndarray  # bool (308,)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        x = raw.astype(np.float64).copy()
        cont = self.is_continuous
        nan = np.isnan(x)
        x[nan & cont] = self.median[nan & cont]
        x[nan & ~cont] = 0.0
        sd = np.maximum(self.std, 1e-6)
        x[cont] = (x[cont] - self.mean[cont]) / sd[cont]
        return x.astype(np.float32)

    def to_json(self) -> dict:
        return {
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "is_continuous": self.is_continuous.astype(bool).tolist(),
            "feature_names": sparse_feature_names(),
        }

    @classmethod
    def from_json(cls, obj: dict) -> "SparseScaler":
        return cls(
            median=np.asarray(obj["median"], dtype=np.float64),
            mean=np.asarray(obj["mean"], dtype=np.float64),
            std=np.asarray(obj["std"], dtype=np.float64),
            is_continuous=np.asarray(obj["is_continuous"], dtype=bool),
        )


def continuous_mask() -> np.ndarray:
    """True for last/min/max/delta; False for flags and hours_since (already [0,1])."""
    m = np.zeros(N_SPARSE_FEATURES, dtype=bool)
    for i in range(0, N_SPARSE_FEATURES, 7):
        m[i : i + 4] = True  # last,min,max,delta
        # hours_since already normalized to [0,1] — treat as continuous lightly:
        m[i + 6] = True
    return m


def fit_sparse_scaler(raw_matrix: np.ndarray) -> SparseScaler:
    """raw_matrix: (N, 308) with nans."""
    cont = continuous_mask()
    med = np.nanmedian(raw_matrix, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    filled = raw_matrix.copy()
    nan = np.isnan(filled)
    # broadcast median fill for stats
    for j in range(filled.shape[1]):
        if cont[j]:
            filled[nan[:, j], j] = med[j]
        else:
            filled[nan[:, j], j] = 0.0
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    # non-continuous: identity scale
    mean = np.where(cont, mean, 0.0)
    std = np.where(cont, std, 1.0)
    return SparseScaler(
        median=med.astype(np.float64),
        mean=mean.astype(np.float64),
        std=std.astype(np.float64),
        is_continuous=cont,
    )


def build_sparse_matrix_for_stays(
    grid: pd.DataFrame,
    stay_ids: Sequence[int],
    *,
    show_progress: bool = True,
) -> np.ndarray:
    """(N, 308) raw sparse features."""
    from tqdm.auto import tqdm

    grid_by = {int(sid): g for sid, g in grid.groupby("ICUSTAY_ID")}
    rows = []
    it = stay_ids
    if show_progress:
        it = tqdm(list(stay_ids), desc="sparse features", unit="stay")
    for sid in it:
        rows.append(raw_sparse_vector_for_stay(grid_by.get(int(sid))))
    return np.stack(rows, axis=0)


def save_sparse_scaler(path: Path, scaler: SparseScaler) -> None:
    path.write_text(json.dumps(scaler.to_json(), indent=2), encoding="utf-8")


def load_sparse_scaler(path: Path) -> SparseScaler:
    return SparseScaler.from_json(json.loads(path.read_text(encoding="utf-8")))
