"""
Cross-length Shapelet motif-family alignment (84-node construction).

Maps frozen templates (var, L, k) -> motif_family in {0,1,2} using only
template geometry. L=6 templates are anchors; L=4 and L=12 each choose
the nearest anchor independently (not a forced 1-1 Hungarian cover).
Clinical labels are applied after inspecting the aligned curves — never
from val/test mortality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from scripts.utils.graph_constants import STAGE_NAMES
from scripts.utils.shapelet_constants import (
    FINAL_K,
    FINAL_LENGTHS,
    GAMMA_SOFTDTW,
    SHAPELET_VARS,
)
from scripts.utils.soft_dtw import soft_dtw_normalized

TARGET_LEN = 12
ANCHOR_L = 6
N_FAMILIES = 3
SHAPE_AMP_EPS = 1e-8
DISTANCE_WEIGHTS: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20)
# Independent-nearest audit flags (anchors are never low-confidence).
LOW_CONFIDENCE_COST: float = 0.50
LOW_CONFIDENCE_MARGIN: float = 0.05


@dataclass(frozen=True)
class TemplateRecord:
    var: str
    L: int
    k: int
    raw: np.ndarray
    resampled: np.ndarray
    shape: np.ndarray
    amplitude: float
    start: float
    end: float
    raw_slope: float


@dataclass
class FamilyAssignment:
    var: str
    L: int
    k: int
    motif_family: int
    is_anchor: bool
    cost_to_anchor: float
    components: dict[str, float] = field(default_factory=dict)
    family_cost: float = 0.0
    second_best_cost: float = float("nan")
    family_margin: float = float("nan")
    family_low_confidence: bool = False
    costs_to_families: dict[int, float] = field(default_factory=dict)


def is_low_confidence(
    *,
    min_cost: float,
    margin: float,
    is_anchor: bool,
    cost_thr: float = LOW_CONFIDENCE_COST,
    margin_thr: float = LOW_CONFIDENCE_MARGIN,
) -> bool:
    """Anchors define the family. Non-anchors: far from all, or two families nearly tied."""
    if is_anchor:
        return False
    if not np.isfinite(min_cost):
        return True
    if float(min_cost) > float(cost_thr):
        return True
    if np.isfinite(margin) and float(margin) < float(margin_thr):
        return True
    return False


def resample_to_length(x: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """Linear resample a 1-D template onto ``target_len`` points."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"template must have length >= 2, got {arr.size}")
    t = int(target_len)
    if t < 2:
        raise ValueError(f"target_len must be >= 2, got {t}")
    if arr.size == t:
        return arr.copy()
    src = np.linspace(0.0, 1.0, num=arr.size)
    dst = np.linspace(0.0, 1.0, num=t)
    return np.interp(dst, src, arr)


def shape_normalize(x: np.ndarray, eps: float = SHAPE_AMP_EPS) -> tuple[np.ndarray, float]:
    """Subtract start, then divide by peak-to-peak amplitude.

    Returns (normalized, amplitude). Flat templates stay at 0 after start-shift.
    """
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    shifted = arr - arr[0]
    amp = float(np.max(shifted) - np.min(shifted))
    if amp < float(eps):
        return shifted, amp
    return shifted / amp, amp


def series_slope(x: np.ndarray) -> float:
    """OLS slope of y against unit-interval x."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    t = np.linspace(0.0, 1.0, num=arr.size)
    t = t - t.mean()
    y = arr - arr.mean()
    denom = float(np.dot(t, t))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(t, y) / denom)


def first_diff_direction(x: np.ndarray) -> np.ndarray:
    d = np.diff(np.asarray(x, dtype=np.float64).reshape(-1))
    return np.sign(d)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation without np.corrcoef (MKL abort on some Windows builds)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("corr length mismatch")
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if na < 1e-12 and nb < 1e-12 else 0.0
    c = float(np.dot(a, b) / (na * nb))
    if not np.isfinite(c):
        return 0.0
    return float(np.clip(c, -1.0, 1.0))


def _soft_dtw_shape_distance(a: np.ndarray, b: np.ndarray, gamma: float = GAMMA_SOFTDTW) -> float:
    a_t = torch.as_tensor(a, dtype=torch.float32)
    b_t = torch.as_tensor(b, dtype=torch.float32)
    # squared Euclidean local cost, matching Shapelet training convention
    cost = (a_t[:, None] - b_t[None, :]) ** 2
    d = soft_dtw_normalized(cost, gamma=float(gamma))
    return float(d.detach().cpu())


def shape_distance_components(
    a: np.ndarray,
    b: np.ndarray,
    *,
    gamma: float = GAMMA_SOFTDTW,
) -> dict[str, float]:
    """Four complementary shape distances, each roughly in a comparable range."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("shape series must share length")

    d_sdtw = _soft_dtw_shape_distance(a, b, gamma=gamma)
    # Soft-DTW can be slightly negative; clip for a distance interpretation
    d_sdtw = float(max(d_sdtw, 0.0))

    corr = _safe_corr(a, b)
    d_corr = 0.5 * (1.0 - corr)

    da, db = first_diff_direction(a), first_diff_direction(b)
    # 0 if every step agrees, 1 if every step disagrees (zeros count as mismatch vs nonzero)
    agree = da == db
    d_diff = float(1.0 - np.mean(agree)) if agree.size else 0.0

    d_slope = abs(series_slope(a) - series_slope(b))
    # shape-normalized series live in ~[-1, 1]; slope over [0,1] is typically O(1)
    d_slope = float(min(d_slope / 2.0, 1.0))

    return {
        "soft_dtw": d_sdtw,
        "corr": d_corr,
        "diff_dir": d_diff,
        "slope": d_slope,
        "corr_raw": corr,
        "slope_a": series_slope(a),
        "slope_b": series_slope(b),
    }


def combine_shape_distance(
    components: Mapping[str, float],
    weights: Sequence[float] = DISTANCE_WEIGHTS,
) -> float:
    w_s, w_c, w_d, w_sl = (float(x) for x in weights)
    wsum = w_s + w_c + w_d + w_sl
    if wsum <= 0:
        raise ValueError("distance weights must sum to > 0")
    return float(
        (
            w_s * float(components["soft_dtw"])
            + w_c * float(components["corr"])
            + w_d * float(components["diff_dir"])
            + w_sl * float(components["slope"])
        )
        / wsum
    )


def pairwise_shape_cost(
    series_a: Sequence[np.ndarray],
    series_b: Sequence[np.ndarray],
    *,
    gamma: float = GAMMA_SOFTDTW,
    weights: Sequence[float] = DISTANCE_WEIGHTS,
) -> tuple[np.ndarray, list[list[dict[str, float]]]]:
    """Cost matrix (n_a, n_b) and per-cell component dicts."""
    n_a, n_b = len(series_a), len(series_b)
    cost = np.zeros((n_a, n_b), dtype=np.float64)
    comps: list[list[dict[str, float]]] = [[{} for _ in range(n_b)] for _ in range(n_a)]
    for i, sa in enumerate(series_a):
        for j, sb in enumerate(series_b):
            c = shape_distance_components(sa, sb, gamma=gamma)
            cost[i, j] = combine_shape_distance(c, weights)
            comps[i][j] = c
    return cost, comps


def hungarian_assign(cost: np.ndarray) -> np.ndarray:
    """Return array ``assign`` of shape (n_row,) with assign[i] = matched column."""
    mat = np.asarray(cost, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Hungarian cost must be square, got {mat.shape}")
    rows, cols = linear_sum_assignment(mat)
    out = np.empty(mat.shape[0], dtype=np.int64)
    out[rows] = cols
    return out


def build_template_records(
    templates: Mapping[tuple[str, int, int], np.ndarray],
    *,
    target_len: int = TARGET_LEN,
) -> list[TemplateRecord]:
    rows: list[TemplateRecord] = []
    for (var, L, k), raw in templates.items():
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size != int(L):
            raise ValueError(f"template {(var, L, k)} length {arr.size} != L={L}")
        resampled = resample_to_length(arr, target_len=target_len)
        shape, amp = shape_normalize(resampled)
        rows.append(
            TemplateRecord(
                var=str(var),
                L=int(L),
                k=int(k),
                raw=arr.copy(),
                resampled=resampled,
                shape=shape,
                amplitude=float(amp),
                start=float(arr[0]),
                end=float(arr[-1]),
                raw_slope=float(arr[-1] - arr[0]),
            )
        )
    return rows


def align_variable_families(
    records: Sequence[TemplateRecord],
    *,
    var: str,
    n_families: int = N_FAMILIES,
    anchor_L: int = ANCHOR_L,
    gamma: float = GAMMA_SOFTDTW,
    weights: Sequence[float] = DISTANCE_WEIGHTS,
) -> list[FamilyAssignment]:
    """Assign each (L, k) independently to the nearest L=6 anchor.

    Same-length templates may share a family. A family may have no member
    at a given length. Does not force a 1-1 cover.
    """
    recs = [r for r in records if r.var == var]
    if not recs:
        raise ValueError(f"no templates for var={var!r}")
    anchors = sorted([r for r in recs if r.L == int(anchor_L)], key=lambda r: r.k)
    if len(anchors) != int(n_families):
        raise ValueError(
            f"{var}: expected {n_families} L={anchor_L} anchors, got {len(anchors)}"
        )
    # Family id := anchor k (0,1,2). Do not sort by slope / mortality.
    if {r.k for r in anchors} != set(range(n_families)):
        raise ValueError(f"{var}: anchor k values must be 0..{n_families-1}")

    out: list[FamilyAssignment] = []
    # Distances among anchors (audit only; assignment stays identity).
    if len(anchors) >= 2:
        anc_cost, _ = pairwise_shape_cost(
            [r.shape for r in anchors],
            [r.shape for r in anchors],
            gamma=gamma,
            weights=weights,
        )
    else:
        anc_cost = np.zeros((len(anchors), len(anchors)), dtype=np.float64)

    for i, r in enumerate(anchors):
        row = np.asarray(anc_cost[i], dtype=np.float64).copy()
        row[i] = np.inf
        second = float(np.min(row)) if np.isfinite(row).any() else float("nan")
        margin = float(second) if np.isfinite(second) else float("nan")
        costs = {int(anchors[j].k): float(anc_cost[i, j]) for j in range(len(anchors))}
        costs[int(r.k)] = 0.0
        out.append(
            FamilyAssignment(
                var=var,
                L=r.L,
                k=r.k,
                motif_family=int(r.k),
                is_anchor=True,
                cost_to_anchor=0.0,
                components={
                    "soft_dtw": 0.0,
                    "corr": 0.0,
                    "diff_dir": 0.0,
                    "slope": 0.0,
                    "corr_raw": 1.0,
                    "slope_a": series_slope(r.shape),
                    "slope_b": series_slope(r.shape),
                },
                family_cost=0.0,
                second_best_cost=second,
                family_margin=margin,
                family_low_confidence=False,
                costs_to_families=costs,
            )
        )

    other_Ls = sorted({r.L for r in recs if r.L != int(anchor_L)})
    anchor_shapes = [r.shape for r in anchors]
    for L in other_Ls:
        group = sorted([r for r in recs if r.L == L], key=lambda r: r.k)
        if not group:
            continue
        cost, comps = pairwise_shape_cost(
            [r.shape for r in group],
            anchor_shapes,
            gamma=gamma,
            weights=weights,
        )
        for i, r in enumerate(group):
            row = np.asarray(cost[i], dtype=np.float64)
            order = np.argsort(row)
            j = int(order[0])
            best = float(row[j])
            second = float(row[order[1]]) if order.size > 1 else float("nan")
            margin = (second - best) if np.isfinite(second) else float("nan")
            fam = int(anchors[j].k)
            out.append(
                FamilyAssignment(
                    var=var,
                    L=r.L,
                    k=r.k,
                    motif_family=fam,
                    is_anchor=False,
                    cost_to_anchor=best,
                    components=dict(comps[i][j]),
                    family_cost=best,
                    second_best_cost=second,
                    family_margin=margin,
                    family_low_confidence=is_low_confidence(
                        min_cost=best, margin=margin, is_anchor=False
                    ),
                    costs_to_families={
                        int(anchors[jj].k): float(row[jj]) for jj in range(len(anchors))
                    },
                )
            )
    out.sort(key=lambda a: (a.L, a.k))
    return out


def align_all_variables(
    records: Sequence[TemplateRecord],
    *,
    vars_: Sequence[str] | None = None,
    n_families: int = N_FAMILIES,
    anchor_L: int = ANCHOR_L,
    gamma: float = GAMMA_SOFTDTW,
    weights: Sequence[float] = DISTANCE_WEIGHTS,
) -> list[FamilyAssignment]:
    vars_use = list(vars_ or sorted({r.var for r in records}))
    assigned: list[FamilyAssignment] = []
    for v in vars_use:
        assigned.extend(
            align_variable_families(
                records,
                var=v,
                n_families=n_families,
                anchor_L=anchor_L,
                gamma=gamma,
                weights=weights,
            )
        )
    return assigned


def family_shape_descriptors(
    records: Sequence[TemplateRecord],
    assignments: Sequence[FamilyAssignment],
) -> pd.DataFrame:
    """Per-(var, family) descriptors from the L=6 anchor + member mean shape."""
    rec_ix = {(r.var, r.L, r.k): r for r in records}
    rows = []
    for var in sorted({a.var for a in assignments}):
        for fam in sorted({a.motif_family for a in assignments if a.var == var}):
            members = [a for a in assignments if a.var == var and a.motif_family == fam]
            shapes = [rec_ix[(m.var, m.L, m.k)].shape for m in members]
            raws = [rec_ix[(m.var, m.L, m.k)].raw for m in members]
            mean_shape = np.mean(np.stack(shapes, axis=0), axis=0)
            anchor = next(m for m in members if m.is_anchor)
            a_rec = rec_ix[(anchor.var, anchor.L, anchor.k)]
            dlt = np.diff(mean_shape)
            n_turns = int(np.sum((dlt[1:] * dlt[:-1]) < 0)) if dlt.size > 1 else 0
            rows.append(
                {
                    "var": var,
                    "motif_family": int(fam),
                    "anchor_k": int(anchor.k),
                    "n_members": len(members),
                    "mean_shape_start": float(mean_shape[0]),
                    "mean_shape_end": float(mean_shape[-1]),
                    "mean_shape_net": float(mean_shape[-1] - mean_shape[0]),
                    "mean_shape_slope": series_slope(mean_shape),
                    "mean_shape_turns": n_turns,
                    "mean_shape_amp": float(np.max(mean_shape) - np.min(mean_shape)),
                    "anchor_raw_start": float(a_rec.start),
                    "anchor_raw_end": float(a_rec.end),
                    "anchor_raw_slope": float(a_rec.raw_slope),
                    "anchor_amplitude": float(a_rec.amplitude),
                    "member_mean_raw_start": float(np.mean([r[0] for r in raws])),
                    "mean_shape": mean_shape.tolist(),
                    "anchor_shape": a_rec.shape.tolist(),
                    "member_L": [int(m.L) for m in members],
                    "member_k": [int(m.k) for m in members],
                }
            )
    return pd.DataFrame(rows)


def mapping_frame(
    assignments: Sequence[FamilyAssignment],
    labels: Mapping[tuple[str, int], str] | None = None,
) -> pd.DataFrame:
    rows = []
    for a in assignments:
        lab = ""
        if labels is not None:
            lab = str(labels.get((a.var, a.motif_family), ""))
        rows.append(
            {
                "var": a.var,
                "L": int(a.L),
                "k": int(a.k),
                "motif_family": int(a.motif_family),
                "is_anchor": bool(a.is_anchor),
                "cost_to_anchor": float(a.cost_to_anchor),
                "family_cost": float(a.family_cost),
                "second_best_cost": float(a.second_best_cost),
                "family_margin": float(a.family_margin),
                "family_low_confidence": bool(a.family_low_confidence),
                "soft_dtw": float(a.components.get("soft_dtw", 0.0)),
                "corr_dist": float(a.components.get("corr", 0.0)),
                "diff_dir": float(a.components.get("diff_dir", 0.0)),
                "slope_dist": float(a.components.get("slope", 0.0)),
                "corr_raw": float(a.components.get("corr_raw", np.nan)),
                "cost_f0": float(a.costs_to_families.get(0, np.nan)),
                "cost_f1": float(a.costs_to_families.get(1, np.nan)),
                "cost_f2": float(a.costs_to_families.get(2, np.nan)),
                "family_label": lab,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["var", "L", "k"]).reset_index(drop=True)


def apply_family_labels(
    mapping: pd.DataFrame,
    labels: Mapping[tuple[str, int], str],
) -> pd.DataFrame:
    out = mapping.copy()
    out["family_label"] = [
        str(labels[(str(r.var), int(r.motif_family))])
        for r in out.itertuples()
    ]
    missing = out["family_label"].eq("").sum()
    if missing:
        raise ValueError(f"{int(missing)} mapping rows missing family_label")
    return out


def motif_activation_name(var: str, family_label: str, stage: str) -> str:
    return f"{var}|{family_label}|{stage}"


def event_display_name(
    var: str,
    motif_family: int,
    family_label: str,
    stage: str,
    *,
    low_confidence: bool,
) -> str:
    """Patient-facing name. Low-confidence cross-scale maps keep family id, not the clinical label."""
    if low_confidence:
        return f"{var}|f{int(motif_family)}（低置信跨尺度映射）|{stage}"
    return motif_activation_name(var, family_label, stage)


def build_motif_node_dictionary(
    mapping: pd.DataFrame,
    *,
    vars_: Sequence[str] = SHAPELET_VARS,
    stages: Sequence[str] = STAGE_NAMES,
    n_families: int = N_FAMILIES,
) -> pd.DataFrame:
    """84-row dictionary keyed by (var, motif_family, stage)."""
    if "family_label" not in mapping.columns:
        raise ValueError("mapping must include family_label")
    label_of: dict[tuple[str, int], str] = {}
    for r in mapping.itertuples():
        key = (str(r.var), int(r.motif_family))
        lab = str(r.family_label)
        if key in label_of and label_of[key] != lab:
            raise ValueError(f"conflicting labels for {key}: {label_of[key]!r} vs {lab!r}")
        label_of[key] = lab

    rows = []
    nid = 0
    for v in vars_:
        for fam in range(int(n_families)):
            if (v, fam) not in label_of:
                raise ValueError(f"missing family label for {(v, fam)}")
            lab = label_of[(v, fam)]
            for si, st in enumerate(stages):
                rows.append(
                    {
                        "node_id": nid,
                        "var": v,
                        "motif_family": fam,
                        "family_label": lab,
                        "stage": st,
                        "stage_idx": int(si),
                        "activation_name": motif_activation_name(v, lab, st),
                        "L": pd.NA,
                        "k": pd.NA,
                    }
                )
                nid += 1
    df = pd.DataFrame(rows)
    if len(df) != len(vars_) * int(n_families) * len(stages):
        raise RuntimeError("unexpected motif dictionary size")
    return df


def lookup_family(
    mapping: pd.DataFrame,
) -> dict[tuple[str, int, int], tuple[int, str]]:
    """(var, L, k) -> (motif_family, family_label)."""
    out: dict[tuple[str, int, int], tuple[int, str]] = {}
    for r in mapping.itertuples():
        out[(str(r.var), int(r.L), int(r.k))] = (
            int(r.motif_family),
            str(r.family_label),
        )
    return out


def lookup_family_audit(mapping: pd.DataFrame) -> dict[tuple[str, int, int], dict]:
    """(var, L, k) -> family + confidence audit fields."""
    out: dict[tuple[str, int, int], dict] = {}
    for r in mapping.itertuples():
        out[(str(r.var), int(r.L), int(r.k))] = {
            "motif_family": int(r.motif_family),
            "family_label": str(r.family_label),
            "family_cost": float(getattr(r, "family_cost", getattr(r, "cost_to_anchor", 0.0))),
            "second_best_cost": float(getattr(r, "second_best_cost", float("nan"))),
            "family_margin": float(getattr(r, "family_margin", float("nan"))),
            "family_low_confidence": bool(getattr(r, "family_low_confidence", False)),
            "is_anchor": bool(getattr(r, "is_anchor", False)),
        }
    return out


def remap_activation_long(
    long_df: pd.DataFrame,
    mapping: pd.DataFrame,
    motif_nodes: pd.DataFrame,
    *,
    drop_test: bool = True,
) -> pd.DataFrame:
    """Attach motif_family / new node_id. Original L/k kept as audit fields.

    Does not drop non-winner rows; caller should run select_active_events
    and optionally collapse to winner-only before building graphs.
    """
    required = {"var", "L", "k", "stage"}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"activations missing {sorted(missing)}")

    lut = lookup_family_audit(mapping)
    node_lut = {
        (str(r.var), int(r.motif_family), str(r.stage)): int(r.node_id)
        for r in motif_nodes.itertuples()
    }
    name_lut = {
        (str(r.var), int(r.motif_family), str(r.stage)): str(r.activation_name)
        for r in motif_nodes.itertuples()
    }

    df = long_df.copy()
    if drop_test and "split" in df.columns:
        df = df.loc[df["split"].astype(str) != "test"].copy()

    fams = []
    labs = []
    new_ids = []
    new_names = []
    disp_names = []
    costs = []
    seconds = []
    margins = []
    low_flags = []
    for r in df.itertuples():
        key = (str(r.var), int(r.L), int(r.k))
        if key not in lut:
            raise KeyError(f"no family mapping for {key}")
        rec = lut[key]
        fam = int(rec["motif_family"])
        lab = str(rec["family_label"])
        low = bool(rec["family_low_confidence"])
        nkey = (str(r.var), int(fam), str(r.stage))
        if nkey not in node_lut:
            raise KeyError(f"no motif node for {nkey}")
        fams.append(fam)
        labs.append(lab)
        new_ids.append(node_lut[nkey])
        new_names.append(name_lut[nkey])
        disp_names.append(
            event_display_name(str(r.var), fam, lab, str(r.stage), low_confidence=low)
        )
        costs.append(float(rec["family_cost"]))
        seconds.append(float(rec["second_best_cost"]))
        margins.append(float(rec["family_margin"]))
        low_flags.append(low)

    if "node_id" in df.columns:
        df = df.rename(columns={"node_id": "node_id_lk"})
    if "k" in df.columns:
        df["k_orig"] = df["k"].astype(np.int16)
    df["motif_family"] = np.asarray(fams, dtype=np.int16)
    df["family_label"] = labs
    df["node_id"] = np.asarray(new_ids, dtype=np.int32)
    df["activation_name"] = new_names
    df["display_name"] = disp_names
    df["family_cost"] = np.asarray(costs, dtype=np.float32)
    df["second_best_cost"] = np.asarray(seconds, dtype=np.float32)
    df["family_margin"] = np.asarray(margins, dtype=np.float32)
    df["family_low_confidence"] = np.asarray(low_flags, dtype=bool)
    return df.reset_index(drop=True)


def winner_only_dense_rows(
    remapped: pd.DataFrame,
    *,
    a_min: float,
) -> pd.DataFrame:
    """One strongest event per (stay, var, stage), already mapped to 84 nodes."""
    from scripts.utils.graph_events import select_active_events

    active = select_active_events(remapped, a_min=a_min)
    return active


def extract_templates_from_checkpoint(
    ckpt_path,
    *,
    vars_: Sequence[str] | None = None,
    lengths: Sequence[int] = FINAL_LENGTHS,
    k: int = FINAL_K,
) -> dict[tuple[str, int, int], np.ndarray]:
    """Load frozen Shapelet templates. No labels, no val/test data."""
    from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS
    from scripts.utils.shapelet_model import ShapeletRiskModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = dict(ckpt.get("meta") or {})
    vars_use = tuple(vars_ or meta.get("vars") or SHAPELET_VARS)
    lengths_use = tuple(meta.get("lengths") or lengths)
    k_use = int(meta.get("k") or k)
    model = ShapeletRiskModel(
        vars_=vars_use,
        lengths=lengths_use,
        k=k_use,
        n_static=len(SHAPELET_STATIC_COLS),
        pos_weight=1.0,
        head_mode=str(ckpt.get("head_mode") or meta.get("head") or "combined"),
        q_init=0.0,
    )
    state = ckpt.get("best_state") or ckpt["model_state"]
    model.load_state_dict(state)
    model.eval()
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for i, key in enumerate(model.template_keys()):
        out[key] = model.shapelets[i].detach().cpu().numpy().astype(np.float64)
    return out


def assert_mapping_complete(
    mapping: pd.DataFrame,
    *,
    vars_: Sequence[str] = SHAPELET_VARS,
    lengths: Sequence[int] = FINAL_LENGTHS,
    k: int = FINAL_K,
    n_families: int = N_FAMILIES,
) -> None:
    n_expected = len(vars_) * len(lengths) * int(k)
    if len(mapping) != n_expected:
        raise ValueError(f"mapping has {len(mapping)} rows, expected {n_expected}")
    for v in vars_:
        sub = mapping.loc[mapping["var"] == v]
        if sub.duplicated(subset=["L", "k"]).any():
            raise ValueError(f"{v}: duplicate (L, k) in mapping")
        anchors = sub.loc[sub["L"] == ANCHOR_L] if "L" in sub.columns else sub
        if "is_anchor" in sub.columns:
            anchors = sub.loc[sub["is_anchor"].astype(bool)]
        if len(anchors) != int(n_families):
            raise ValueError(f"{v}: expected {n_families} anchors, got {len(anchors)}")
        fams = set(int(x) for x in anchors["motif_family"])
        if fams != set(range(n_families)):
            raise ValueError(f"{v}: anchor families {fams} != {set(range(n_families))}")
        # Independent nearest: same-length templates may share a family.
    if "family_label" in mapping.columns and mapping["family_label"].astype(str).str.len().min() == 0:
        raise ValueError("empty family_label present")
