"""
Shapelet data prep: hourly grid → (x, m, δ, static, y) tensors.

Protocol (docs/12):
  - Use raw hourly_grid (not median-imputed values for matching).
  - μ,σ fit on train measured values only; missing stays 0 via x_safe.
  - δ from mask (compute_delta); never_measured → t+1 capped at δ_max.
  - Static covariates from stay_features_48h; train-only z-score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from scripts.utils.constants import DATA_PROCESSED
from scripts.utils.shapelet_constants import (
    SEED,
    SHAPELET_STATIC_COLS,
    SHAPELET_VARS,
    STAGE_BOUNDS,
    T_HOURS,
)
from scripts.utils.shapelet_cost import compute_delta


@dataclass
class Standardizer:
    """Per-variable mean/std on train measured hours only."""

    mean: dict[str, float]
    std: dict[str, float]

    def transform_value(self, var: str, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        mu = self.mean[var]
        sd = max(self.std[var], 1e-6)
        out = np.zeros_like(x, dtype=np.float32)
        measured = m > 0.5
        out[measured] = ((x[measured] - mu) / sd).astype(np.float32)
        return out


@dataclass
class StaticScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        sd = np.maximum(self.std, 1e-6)
        return ((x - self.mean) / sd).astype(np.float32)


@dataclass
class ShapeletBundle:
    """In-memory tensors for one split (or full cohort)."""

    stay_ids: np.ndarray  # (N,) int64
    subject_ids: np.ndarray  # (N,)
    split: np.ndarray  # (N,) object/str
    y: np.ndarray  # (N,) float32
    # (N, T, V) — z-scored where measured, 0 where missing
    x: np.ndarray
    m: np.ndarray  # (N, T, V) float32 0/1
    delta: np.ndarray  # (N, T, V)
    static: np.ndarray  # (N, C) z-scored
    vars: tuple[str, ...]
    static_cols: tuple[str, ...]


def stage_start_masks(n_starts: int, device: torch.device | None = None) -> torch.Tensor:
    """Boolean masks (n_stages, n_starts) for window *starts* in B1–B4.

    Stage membership is by start index t ∈ [0, n_starts), mapped to hour t
    with the same bounds as STAGE_BOUNDS (end exclusive). Starts with
    t >= n_starts are already excluded by construction.
    """
    starts = torch.arange(n_starts, device=device)
    masks = []
    for _name, lo, hi in STAGE_BOUNDS:
        masks.append((starts >= lo) & (starts < hi))
    return torch.stack(masks, dim=0)


def fit_value_standardizer(
    grid: pd.DataFrame,
    train_ids: Iterable[int],
    vars_: Sequence[str] = SHAPELET_VARS,
) -> Standardizer:
    train_set = set(int(i) for i in train_ids)
    g = grid[grid["ICUSTAY_ID"].isin(train_set)]
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for v in vars_:
        mcol = f"{v}_is_measured"
        if v not in g.columns:
            raise KeyError(f"grid missing column {v}")
        if mcol in g.columns:
            vals = g.loc[g[mcol].astype(bool), v]
        else:
            vals = g[v].dropna()
        vals = pd.to_numeric(vals, errors="coerce").dropna()
        if len(vals) == 0:
            mean[v], std[v] = 0.0, 1.0
        else:
            mean[v] = float(vals.mean())
            std[v] = float(vals.std(ddof=0)) if len(vals) > 1 else 1.0
            if std[v] < 1e-6:
                std[v] = 1.0
    return Standardizer(mean=mean, std=std)


def fit_static_scaler(
    stays: pd.DataFrame,
    train_ids: Iterable[int],
    cols: Sequence[str] = SHAPELET_STATIC_COLS,
) -> StaticScaler:
    train_set = set(int(i) for i in train_ids)
    sub = stays[stays["ICUSTAY_ID"].isin(train_set)]
    missing = [c for c in cols if c not in sub.columns]
    if missing:
        raise KeyError(f"stay table missing static cols: {missing}")
    mat = sub[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    # median impute for static fit only (train)
    med = np.nanmedian(mat, axis=0)
    inds = np.where(np.isnan(mat))
    mat[inds] = np.take(med, inds[1])
    return StaticScaler(mean=mat.mean(axis=0), std=mat.std(axis=0))


def _pivot_stay_series(
    g: pd.DataFrame,
    var: str,
    t_hours: int = T_HOURS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (T,) value and mask for one stay, HOUR 0..T-1."""
    mcol = f"{var}_is_measured"
    by_h = g.set_index("HOUR")
    x = np.full(t_hours, np.nan, dtype=np.float64)
    m = np.zeros(t_hours, dtype=np.float32)
    for h in range(t_hours):
        if h not in by_h.index:
            continue
        row = by_h.loc[h]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        val = row[var] if var in row.index else np.nan
        measured = False
        if mcol in row.index:
            measured = bool(row[mcol])
        else:
            measured = pd.notna(val)
        if measured and pd.notna(val):
            x[h] = float(val)
            m[h] = 1.0
    return x.astype(np.float32), m


def build_bundle(
    grid: pd.DataFrame,
    stays: pd.DataFrame,
    value_scaler: Standardizer,
    static_scaler: StaticScaler,
    vars_: Sequence[str] = SHAPELET_VARS,
    static_cols: Sequence[str] = SHAPELET_STATIC_COLS,
    stay_ids: Sequence[int] | None = None,
    t_hours: int = T_HOURS,
    desc: str = "build stays",
    show_progress: bool = True,
) -> ShapeletBundle:
    """Build tensors for selected stays (default: all in stays table)."""
    if stay_ids is None:
        ids = stays["ICUSTAY_ID"].astype(int).tolist()
    else:
        ids = [int(i) for i in stay_ids]

    stay_idx = stays.set_index("ICUSTAY_ID")
    n = len(ids)
    v = len(vars_)
    c = len(static_cols)

    x_all = np.zeros((n, t_hours, v), dtype=np.float32)
    m_all = np.zeros((n, t_hours, v), dtype=np.float32)
    d_all = np.zeros((n, t_hours, v), dtype=np.float32)
    static_all = np.zeros((n, c), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    subject_ids = np.zeros(n, dtype=np.int64)
    splits = np.empty(n, dtype=object)

    grid_by = {
        int(sid): g
        for sid, g in tqdm(
            grid.groupby("ICUSTAY_ID"),
            total=grid["ICUSTAY_ID"].nunique(),
            desc=f"{desc}: index grid",
            disable=not show_progress,
            leave=False,
        )
    }

    stay_iter = enumerate(ids)
    if show_progress:
        stay_iter = enumerate(
            tqdm(ids, desc=desc, total=n, unit="stay", leave=True)
        )

    for i, sid in stay_iter:
        row = stay_idx.loc[sid]
        y[i] = float(row["HOSPITAL_MORTALITY"])
        subject_ids[i] = int(row["SUBJECT_ID"]) if "SUBJECT_ID" in row.index else -1
        splits[i] = str(row["split"]) if "split" in row.index else "unknown"

        raw_static = []
        for col in static_cols:
            val = row[col] if col in row.index else np.nan
            raw_static.append(float(val) if pd.notna(val) else float("nan"))
        svec = np.asarray(raw_static, dtype=np.float64)
        # fill nan with train mean (already in scaler)
        nan_m = np.isnan(svec)
        svec[nan_m] = static_scaler.mean[nan_m]
        static_all[i] = static_scaler.transform(svec.reshape(1, -1))[0]

        g = grid_by.get(int(sid))
        if g is None:
            continue
        for j, var in enumerate(vars_):
            raw, m = _pivot_stay_series(g, var, t_hours=t_hours)
            x_z = value_scaler.transform_value(var, raw, m)
            x_all[i, :, j] = x_z
            m_all[i, :, j] = m
            # delta on mask (torch for consistency with training path)
            d_t = compute_delta(torch.from_numpy(m).unsqueeze(0)).squeeze(0).numpy()
            d_all[i, :, j] = d_t.astype(np.float32)

    return ShapeletBundle(
        stay_ids=np.asarray(ids, dtype=np.int64),
        subject_ids=subject_ids,
        split=splits,
        y=y,
        x=x_all,
        m=m_all,
        delta=d_all,
        static=static_all,
        vars=tuple(vars_),
        static_cols=tuple(static_cols),
    )


def load_shapelet_tables(
    processed_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load hourly grid + stay features (with split & static)."""
    root = Path(processed_dir or DATA_PROCESSED)
    grid_path = root / "hourly_grid_48h.parquet"
    stay_path = root / "stay_features_48h.parquet"
    if not grid_path.is_file():
        raise FileNotFoundError(grid_path)
    if not stay_path.is_file():
        raise FileNotFoundError(stay_path)
    grid = pd.read_parquet(grid_path)
    stays = pd.read_parquet(stay_path)
    need = ["ICUSTAY_ID", "split", "HOSPITAL_MORTALITY"] + list(SHAPELET_STATIC_COLS)
    miss = [c for c in need if c not in stays.columns]
    if miss:
        raise KeyError(f"stay_features missing: {miss}")
    return grid, stays


def prepare_splits(
    vars_: Sequence[str] = SHAPELET_VARS,
    static_cols: Sequence[str] = SHAPELET_STATIC_COLS,
    processed_dir: str | Path | None = None,
    max_stays_per_split: int | None = None,
    seed: int = SEED,
    show_progress: bool = True,
    build_test: bool = False,
) -> dict[str, ShapeletBundle]:
    """Fit scalers on train; build train/val(/test) bundles.

    Parameters
    ----------
    max_stays_per_split :
        If set, subsample stays per split (debug smoke only). Scalers still
        use **full train** unless this is used with care — here scalers use
        all train IDs, tensors may be subsampled.
    build_test :
        If False (default), skip building the test tensor pack until explicitly
        requested (avoids long idle before training).
    """
    if show_progress:
        print("[prep] reading parquet…", flush=True)
    grid, stays = load_shapelet_tables(processed_dir)
    if show_progress:
        print(
            f"[prep] grid rows={len(grid)} stays={len(stays)}  fitting train scalers…",
            flush=True,
        )
    train_ids = stays.loc[stays["split"] == "train", "ICUSTAY_ID"].astype(int).tolist()
    value_scaler = fit_value_standardizer(grid, train_ids, vars_=vars_)
    static_scaler = fit_static_scaler(stays, train_ids, cols=static_cols)

    out: dict[str, ShapeletBundle] = {}
    rng = np.random.default_rng(seed)
    split_names = ("train", "val", "test") if build_test else ("train", "val")
    for split_name in split_names:
        ids = stays.loc[stays["split"] == split_name, "ICUSTAY_ID"].astype(int).tolist()
        if max_stays_per_split is not None and len(ids) > max_stays_per_split:
            ids = rng.choice(ids, size=max_stays_per_split, replace=False).tolist()
        out[split_name] = build_bundle(
            grid,
            stays,
            value_scaler,
            static_scaler,
            vars_=vars_,
            static_cols=static_cols,
            stay_ids=ids,
            desc=f"prep {split_name}",
            show_progress=show_progress,
        )
    # attach scalers for export
    out["_value_scaler"] = value_scaler  # type: ignore[assignment]
    out["_static_scaler"] = static_scaler  # type: ignore[assignment]
    if show_progress:
        print("[prep] done", flush=True)
    return out


class ShapeletStayDataset(Dataset):
    def __init__(self, bundle: ShapeletBundle):
        self.bundle = bundle

    def __len__(self) -> int:
        return int(self.bundle.y.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        b = self.bundle
        return {
            "x": torch.from_numpy(b.x[idx]),  # (T, V)
            "m": torch.from_numpy(b.m[idx]),
            "delta": torch.from_numpy(b.delta[idx]),
            "static": torch.from_numpy(b.static[idx]),
            "y": torch.tensor(b.y[idx], dtype=torch.float32),
            "stay_id": torch.tensor(b.stay_ids[idx], dtype=torch.long),
        }


def make_synthetic_batch(
    batch_size: int = 16,
    t_hours: int = T_HOURS,
    n_vars: int = 7,
    n_static: int = len(SHAPELET_STATIC_COLS),
    device: str | torch.device = "cpu",
    seed: int = SEED,
) -> dict[str, torch.Tensor]:
    """Synthetic tensors for G1 dry-run (no patient data)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(batch_size, t_hours, n_vars, generator=g)
    m = (torch.rand(batch_size, t_hours, n_vars, generator=g) < 0.72).float()
    # recompute delta per var channel
    deltas = []
    for j in range(n_vars):
        deltas.append(compute_delta(m[:, :, j]))
    delta = torch.stack(deltas, dim=-1)
    x = x * m  # zero missing (already x_safe style)
    static = torch.randn(batch_size, n_static, generator=g)
    y = (torch.rand(batch_size, generator=g) < 0.15).float()
    batch = {
        "x": x.to(device),
        "m": m.to(device),
        "delta": delta.to(device),
        "static": static.to(device),
        "y": y.to(device),
        "stay_id": torch.arange(batch_size, device=device),
    }
    return batch


def train_pos_weight(y_train: np.ndarray) -> float:
    """n_neg / n_pos for BCEWithLogits."""
    y = y_train.astype(np.float64)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos <= 0:
        return 1.0
    return n_neg / n_pos
