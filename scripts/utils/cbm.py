"""CBM concept-bottleneck utilities (docs/26 Phase A).

Pure functions that build the human-readable concept matrix (84 motif concepts
with continuous activation `a`, plus 11-lab direction indicators), fit a linear
"concept -> death" bottleneck, run 5-fold CV on the train split only, and run
concept-level interventions. Never reads val/holdout/test; never rewrites
frozen assets (84 mapping, 11-lab thresholds, 15 paths).

All evaluation is train-split 5-fold, matching scripts/32 node-outcome metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

KIND_MOTIF = "motif"
KIND_LAB = "lab"


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------
def build_static_matrix(
    stays: pd.DataFrame,
    stay_ids: Sequence[int],
    static_cols: Sequence[str],
) -> np.ndarray:
    """(n, |static|) float64 matrix aligned to stay_ids order."""
    rows = []
    for s in stay_ids:
        rows.append([float(stays.loc[int(s), c]) for c in static_cols])
    return np.stack(rows).astype(np.float64)


def standardize_train_only(
    X: np.ndarray,
    mu: np.ndarray | None = None,
    sd: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X_scaled, mu, sd). If mu/sd are None they are fit on X (train-only)."""
    if mu is None:
        mu = X.mean(axis=0)
    if sd is None:
        sd = np.maximum(X.std(axis=0), 1e-6)
    return (X - mu) / sd, mu, sd


def build_concept_matrix(
    atoms: pd.DataFrame,
    stay_ids: Sequence[int],
    concept_ids: Sequence[str],
    *,
    kind_of: dict[str, str],
    use_activation: bool = True,
) -> np.ndarray:
    """(n, len(concept_ids)) concept matrix.

    motif concepts -> continuous `a` when use_activation else 1 (indicator);
    lab concepts   -> 1 indicator (direction is part of the concept identity).
    Rows with no row in `atoms` stay 0.
    """
    pos = {int(s): i for i, s in enumerate(stay_ids)}
    cidx = {c: j for j, c in enumerate(concept_ids)}
    M = np.zeros((len(stay_ids), len(concept_ids)), dtype=np.float64)
    for rec in atoms.itertuples(index=False):
        i = pos.get(int(rec.ICUSTAY_ID))
        if i is None:
            continue
        j = cidx.get(str(rec.atom))
        if j is None:
            continue
        if kind_of.get(str(rec.atom), rec.kind) == KIND_MOTIF and use_activation:
            a = float(rec.a) if rec.a is not None and np.isfinite(rec.a) else 1.0
            M[i, j] = a
        else:
            M[i, j] = 1.0
    return M


def subject_groups(stays: pd.DataFrame, stay_ids: Sequence[int]) -> np.ndarray:
    return np.array([int(stays.loc[int(s), "SUBJECT_ID"]) for s in stay_ids])


# ---------------------------------------------------------------------------
# 5-fold CV (train split only)
# ---------------------------------------------------------------------------
def _cv_scores(
    y: np.ndarray,
    groups: np.ndarray,
    static: np.ndarray,
    extra: np.ndarray,
    seed: int,
    n_splits: int = 5,
) -> dict[str, float]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    auprc, auroc, brier = [], [], []
    for tr, va in cv.split(static, y, groups):
        if y[tr].min() == y[tr].max() or y[va].min() == y[va].max():
            continue
        Xtr = np.column_stack([static[tr], extra[tr]]) if extra.shape[1] else static[tr]
        Xva = np.column_stack([static[va], extra[va]]) if extra.shape[1] else static[va]
        clf = LogisticRegression(max_iter=500, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xva)[:, 1]
        auprc.append(average_precision_score(y[va], p))
        auroc.append(roc_auc_score(y[va], p))
        brier.append(brier_score_loss(y[va], p))
    return {
        "auprc": float(np.mean(auprc)) if auprc else float("nan"),
        "auroc": float(np.mean(auroc)) if auroc else float("nan"),
        "brier": float(np.mean(brier)) if brier else float("nan"),
        "n_folds_used": int(len(auprc)),
    }


def fit_full_linear(
    y: np.ndarray,
    static: np.ndarray,
    extra: np.ndarray,
) -> LogisticRegression:
    X = np.column_stack([static, extra]) if extra.shape[1] else static
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Concept intervention (linear model => closed-form Δp)
# ---------------------------------------------------------------------------
@dataclass
class InterventionRow:
    concept: str
    kind: str
    n_with: int
    mean_delta_p: float  # p_orig - p_zeroed; >0 => risk concept
    frac_with: float
    mean_abs_coef: float


def concept_interventions(
    clf: LogisticRegression,
    static: np.ndarray,
    extra: np.ndarray,
    concept_names: Sequence[str],
    kind_of: Sequence[str],
    n_static: int,
) -> list[InterventionRow]:
    """For each concept, zero it out among rows where it is active; report mean Δp.

    clf was fit on column_stack([static, extra]); concept columns live in the
    `extra` block (columns [n_static : n_static + len(concept_names)]).
    """
    X = np.column_stack([static, extra]) if extra.shape[1] else static
    coef = np.asarray(clf.coef_).ravel()
    inter = float(clf.intercept_[0])
    z = X @ coef + inter
    p_orig = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def _sigmoid(v: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(v, -50, 50)))

    rows: list[InterventionRow] = []
    for j, name in enumerate(concept_names):
        col = n_static + j
        active = extra[:, j] > 0.5
        n_with = int(active.sum())
        if n_with < 1:
            rows.append(
                InterventionRow(name, kind_of[j], 0, float("nan"), 0.0, abs(coef[col]))
            )
            continue
        xv = extra[active, j]
        delta = coef[col] * xv
        p_zero = _sigmoid(z[active] - delta)
        mean_dp = float((p_orig[active] - p_zero).mean())
        rows.append(
            InterventionRow(
                concept=name,
                kind=kind_of[j],
                n_with=n_with,
                mean_delta_p=mean_dp,
                frac_with=n_with / extra.shape[0],
                mean_abs_coef=abs(coef[col]),
            )
        )
    return rows


def coef_table(
    clf: LogisticRegression,
    concept_names: Sequence[str],
    kind_of: Sequence[str],
    n_static: int,
) -> pd.DataFrame:
    coef = np.asarray(clf.coef_).ravel()
    recs = []
    for j, name in enumerate(concept_names):
        recs.append(
            {
                "concept": name,
                "kind": kind_of[j],
                "coef": float(coef[n_static + j]),
            }
        )
    df = pd.DataFrame(recs)
    return df.sort_values("coef", key=lambda s: s.abs(), ascending=False)
