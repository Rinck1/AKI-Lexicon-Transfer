#!/usr/bin/env python3
"""PRED post-hoc explore: B-arm zero-shot feasibility + B-linear proxy.

(1) Vectorized port of scripts/utils/graph_sparse.py::raw_sparse_vector_for_stay
    (11 vars x 4 stages x 7 stats = 308), validated exactly against the
    reference implementation on sampled MIMIC stays.
(2) Sparse features recomputed for BOTH sides from hourly grids; SparseScaler
    fit on MIMIC train only; logistic arms:
      static+sparse308        and  B-proxy: static+84a(+87lab)+sparse308
    evaluated zero-shot on eICU (+ MIMIC val descriptive).
(3) B-arm (MLP ckpt gnn_ablate_B_act_sparse_static) deployment checklist:
    enumerates required inputs and locates the missing pieces on eICU.

post-hoc exploratory; fit uses MIMIC train split only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.cbm import (  # noqa: E402
    KIND_LAB,
    KIND_MOTIF,
    build_concept_matrix,
    build_static_matrix,
    standardize_train_only,
)
from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.eicu_constants import MIMIC_PATHS_DIR  # noqa: E402
from scripts.utils.eicu_io import load_eicu  # noqa: E402
from scripts.utils.graph_constants import GRAPH_SPARSE_VARS, STAGE_NAMES  # noqa: E402
from scripts.utils.graph_sparse import (  # noqa: E402
    N_SPARSE_FEATURES,
    continuous_mask,
    fit_sparse_scaler,
    raw_sparse_vector_for_stay,
)
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS, STAGE_BOUNDS  # noqa: E402

OUT_DIR = PROJ / "data/external/eicu/_explore_PRED"
SEED = 42
SUFFIXES = ("last", "min", "max", "delta_from_previous_stage_last",
            "is_measured", "measured_fraction", "hours_since_last_measurement_at_stage_end")


def vec_sparse_matrix(grid: pd.DataFrame, stay_ids: np.ndarray, t_hours: int = 48) -> np.ndarray:
    """Vectorized equivalent of graph_sparse.raw_sparse_vector_for_stay per stay."""
    n = len(stay_ids)
    sid_pos = pd.Index(stay_ids)
    keep = grid["ICUSTAY_ID"].isin(sid_pos)
    g = grid.loc[keep, ["ICUSTAY_ID", "HOUR"] + list(GRAPH_SPARSE_VARS)
                      + [f"{v}_is_measured" for v in GRAPH_SPARSE_VARS]].copy()
    g = g.sort_values(["ICUSTAY_ID", "HOUR"], kind="stable")
    row_pos = sid_pos.get_indexer(g["ICUSTAY_ID"].to_numpy())
    hour_pos = g["HOUR"].to_numpy().astype(int)
    out = np.empty((n, N_SPARSE_FEATURES), dtype=np.float64)
    stage_bounds = [(int(lo), int(hi)) for _, lo, hi in STAGE_BOUNDS]
    for v_i, var in enumerate(GRAPH_SPARSE_VARS):
        V = np.full((n, t_hours), np.nan)
        M = np.zeros((n, t_hours), dtype=bool)
        vals = g[var].to_numpy(float)
        meas = g[f"{var}_is_measured"].to_numpy(bool)
        ok = meas & np.isfinite(vals)
        V[row_pos[ok], hour_pos[ok]] = vals[ok]
        M[row_pos[ok], hour_pos[ok]] = True
        lasth = np.full(n, -1, dtype=int)
        prev_last = np.full(n, np.nan)
        base = v_i * len(STAGE_NAMES) * 7
        for s_i, (lo, hi) in enumerate(stage_bounds):
            segM = M[:, lo:hi]
            segV = np.where(segM, V[:, lo:hi], np.nan)
            n_meas = segM.sum(1)
            is_meas = n_meas > 0
            meas_frac = n_meas / float(hi - lo)
            rev = segM[:, ::-1]
            last_off = np.where(is_meas, segM.shape[1] - 1 - np.argmax(rev, axis=1), 0)
            last_v = np.where(is_meas, segV[np.arange(n), last_off], np.nan)
            with np.errstate(all="ignore"):
                min_v = np.nanmin(segV, axis=1)
                max_v = np.nanmax(segV, axis=1)
            min_v = np.where(is_meas, min_v, np.nan)
            max_v = np.where(is_meas, max_v, np.nan)
            if s_i == 0:
                delta = np.where(np.isnan(last_v), np.nan, 0.0)
            else:
                delta = last_v - prev_last
                bad = np.isnan(last_v) | np.isnan(prev_last)
                delta = np.where(bad, np.nan, delta)
            # hours since last measurement at stage end (uses current-stage update first)
            lh_new = np.where(is_meas, lo + last_off, lasth)
            hrs = np.empty(n)
            neg = lh_new < 0
            if neg.any():
                hist = M[:, :hi]
                has_hist = hist.any(1)
                last_hist = np.where(has_hist, t_hours - 1 - np.argmax(hist[:, ::-1], axis=1), 0)
                hrs_neg = np.where(has_hist, (hi - 1) - last_hist, float(t_hours))
                hrs[neg] = hrs_neg[neg]
            hrs[~neg] = (hi - 1) - lh_new[~neg]
            hrs = np.clip(hrs, 0.0, float(t_hours)) / float(t_hours)
            col = base + s_i * 7
            out[:, col + 0] = last_v
            out[:, col + 1] = min_v
            out[:, col + 2] = max_v
            out[:, col + 3] = delta
            out[:, col + 4] = is_meas
            out[:, col + 5] = meas_frac
            out[:, col + 6] = hrs
            lasth = lh_new
            prev_last = np.where(np.isnan(last_v), prev_last, last_v)
    return out.astype(np.float32)


def validate_port(grid: pd.DataFrame, stay_ids: list[int], rng) -> float:
    grid_by = {int(s): gg for s, gg in grid.groupby("ICUSTAY_ID")}
    worst = 0.0
    for sid in stay_ids:
        ref = raw_sparse_vector_for_stay(grid_by.get(int(sid)))
        mine = vec_sparse_matrix(grid, np.array([sid]))[0]
        d = np.nanmax(np.abs(np.nan_to_num(ref, nan=0.0) - np.nan_to_num(mine.astype(float), nan=0.0)))
        worst = max(worst, float(d))
        assert np.array_equal(np.isnan(ref), np.isnan(mine.astype(float))), f"nan pattern differs {sid}"
    return worst


def transform_sparse_2d(raw: np.ndarray, scaler) -> np.ndarray:
    """2D batch equivalent of graph_sparse.SparseScaler.transform (which is 1D-only)."""
    x = raw.astype(np.float64).copy()
    cont = scaler.is_continuous
    nan = np.isnan(x)
    x[nan & cont] = np.broadcast_to(scaler.median, x.shape)[nan & cont]
    x[nan & ~cont] = 0.0
    sd = np.maximum(scaler.std, 1e-6)
    x[:, cont] = (x[:, cont] - scaler.mean[cont]) / sd[cont]
    return x.astype(np.float32)


def validate_transform(scaler, raw: np.ndarray) -> float:
    worst = 0.0
    for i in range(min(20, len(raw))):
        ref = scaler.transform(raw[i].astype(float))
        mine = transform_sparse_2d(raw[i : i + 1], scaler)[0]
        worst = max(worst, float(np.nanmax(np.abs(ref - mine))))
    return worst


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---------------- (1) validate vectorized port ----------------
    grid_m = pd.read_parquet(Path(DATA_PROCESSED) / "hourly_grid_48h.parquet")
    sample_ids = [int(s) for s in rng.choice(grid_m["ICUSTAY_ID"].unique(), size=12, replace=False)]
    worst = validate_port(grid_m, sample_ids, rng)
    print(f"[validate] vectorized sparse port vs reference: max abs diff = {worst:.2e} over 12 stays", flush=True)
    assert worst < 1e-6, "port mismatch"

    # ---------------- MIMIC side ----------------
    from scripts.utils.shapelet_prep import load_shapelet_tables

    _grid, mimic_stays = load_shapelet_tables(DATA_PROCESSED)
    stays_m = mimic_stays.set_index("ICUSTAY_ID")
    train_ids = np.array(sorted(int(x) for x in mimic_stays.loc[mimic_stays["split"] == "train", "ICUSTAY_ID"]))
    val_ids = np.array(sorted(int(x) for x in mimic_stays.loc[mimic_stays["split"] == "val", "ICUSTAY_ID"]))

    m_atoms = pd.read_parquet(MIMIC_PATHS_DIR / "motif_atoms.parquet")
    l_atoms = pd.read_parquet(MIMIC_PATHS_DIR / "lab_atoms.parquet")
    mt_tr, lt_tr = m_atoms[m_atoms["split"] == "train"], l_atoms[l_atoms["split"] == "train"]
    motif_concepts = sorted(mt_tr["atom"].astype(str).unique())
    lab_concepts = sorted(lt_tr["atom"].astype(str).unique())
    kind_of = {c: KIND_MOTIF for c in motif_concepts}
    kind_of.update({c: KIND_LAB for c in lab_concepts})

    y_tr = stays_m.loc[list(train_ids), "HOSPITAL_MORTALITY"].to_numpy(float)
    static_tr_raw = build_static_matrix(stays_m, list(train_ids), SHAPELET_STATIC_COLS)
    static_tr, mu, sd = standardize_train_only(static_tr_raw)

    motif_a_tr = build_concept_matrix(mt_tr, list(train_ids), motif_concepts, kind_of=kind_of, use_activation=True)
    lab_tr = build_concept_matrix(lt_tr, list(train_ids), lab_concepts, kind_of=kind_of, use_activation=True)

    print(f"[sparse] computing MIMIC train {len(train_ids)} stays ...", flush=True)
    sp_tr_raw = vec_sparse_matrix(grid_m, train_ids)
    scaler = fit_sparse_scaler(sp_tr_raw.astype(float))
    sp_tr = transform_sparse_2d(sp_tr_raw, scaler)

    cont = continuous_mask()
    nan_frac = float(np.isnan(sp_tr_raw[:, cont]).mean())
    print(f"[sparse] transform-vs-reference max diff = {validate_transform(scaler, sp_tr_raw):.2e}")
    print(f"[sparse] train done {time.time()-t0:.0f}s; continuous-cell nan rate (raw) = {nan_frac:.3f}", flush=True)

    val_grid_ids = val_ids[np.isin(val_ids, grid_m["ICUSTAY_ID"].unique())]
    y_va = stays_m.loc[list(val_ids), "HOSPITAL_MORTALITY"].to_numpy(float)
    static_va = (build_static_matrix(stays_m, list(val_ids), SHAPELET_STATIC_COLS) - mu) / sd
    motif_a_va = build_concept_matrix(m_atoms[m_atoms["split"] == "val"], list(val_ids), motif_concepts, kind_of=kind_of, use_activation=True)
    lab_va = build_concept_matrix(l_atoms[l_atoms["split"] == "val"], list(val_ids), lab_concepts, kind_of=kind_of, use_activation=True)
    sp_va = transform_sparse_2d(vec_sparse_matrix(grid_m, val_grid_ids), scaler)
    # stays missing from grid: all-zero flags row (should not happen; grid is 48h x all stays)
    assert len(val_grid_ids) == len(val_ids)

    # ---------------- eICU side ----------------
    grid_e = pd.read_parquet(Path(DATA_PROCESSED).parent / "external/eicu/hourly_grid_48h.parquet")
    stays_e = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    e_ids = np.array(sorted(int(x) for x in stays_e.index))
    y_e = stays_e.loc[list(e_ids), "HOSPITAL_MORTALITY"].to_numpy(float)
    static_e = (build_static_matrix(stays_e, list(e_ids), SHAPELET_STATIC_COLS) - mu) / sd
    em, el = load_eicu("motif_atoms"), load_eicu("lab_atoms")
    motif_a_e = build_concept_matrix(em, list(e_ids), motif_concepts, kind_of=kind_of, use_activation=True)
    lab_e = build_concept_matrix(el, list(e_ids), lab_concepts, kind_of=kind_of, use_activation=True)
    print(f"[sparse] computing eICU {len(e_ids)} stays ...", flush=True)
    e_grid_ids = e_ids[np.isin(e_ids, grid_e["ICUSTAY_ID"].unique())]
    assert len(e_grid_ids) == len(e_ids)
    sp_e = transform_sparse_2d(vec_sparse_matrix(grid_e, e_ids), scaler)
    print(f"[sparse] eICU done {time.time()-t0:.0f}s", flush=True)

    # ---------------- arms ----------------
    arms = {
        "static": static_tr,
        "static+sparse308": np.column_stack([static_tr, sp_tr]),
        "Bproxy_static_84a_sparse": np.column_stack([static_tr, motif_a_tr, sp_tr]),
        "Bproxy_full_84a_87lab_sparse": np.column_stack([static_tr, motif_a_tr, lab_tr, sp_tr]),
    }
    X_va_arm = {
        "static": static_va,
        "static+sparse308": np.column_stack([static_va, sp_va]),
        "Bproxy_static_84a_sparse": np.column_stack([static_va, motif_a_va, sp_va]),
        "Bproxy_full_84a_87lab_sparse": np.column_stack([static_va, motif_a_va, lab_va, sp_va]),
    }
    X_e_arm = {
        "static": static_e,
        "static+sparse308": np.column_stack([static_e, sp_e]),
        "Bproxy_static_84a_sparse": np.column_stack([static_e, motif_a_e, sp_e]),
        "Bproxy_full_84a_87lab_sparse": np.column_stack([static_e, motif_a_e, lab_e, sp_e]),
    }

    results = {"post_hoc": True, "port_max_abs_diff": worst,
               "train_raw_continuous_nan_rate": nan_frac, "arms": {}}
    pred = pd.DataFrame({"ICUSTAY_ID": list(e_ids), "y": y_e})
    pred_val = pd.DataFrame({"ICUSTAY_ID": list(val_ids), "y": y_va})
    for name, X in arms.items():
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X, y_tr)
        row = {}
        for set_name, y, Xd in [("mimic_val(desc)", y_va, X_va_arm), ("eicu_zeroshot", y_e, X_e_arm)]:
            p = clf.predict_proba(Xd[name])[:, 1]
            row[set_name] = {"n": int(len(y)), "prevalence": float(y.mean()),
                             "auprc": float(average_precision_score(y, p)),
                             "auroc": float(roc_auc_score(y, p)),
                             "brier": float(brier_score_loss(y, p))}
            if set_name == "eicu_zeroshot":
                pred[f"p__{name}"] = p
            else:
                pred_val[f"p__{name}"] = p
        results["arms"][name] = row
        mv, ez = row["mimic_val(desc)"], row["eicu_zeroshot"]
        print(f"[{name:<30}] val AUPRC {mv['auprc']:.3f} (desc) | eICU AUPRC {ez['auprc']:.3f} "
              f"AUROC {ez['auroc']:.3f} Brier {ez['brier']:.3f}", flush=True)
    pred.to_parquet(OUT_DIR / "pred_eicu_sparse_arms.parquet", index=False)
    pred_val.to_parquet(OUT_DIR / "pred_mimic_val_sparse_arms.parquet", index=False)

    # ---------------- (3) B-arm deployment checklist ----------------
    ckpt = PROJ / "data/processed/gnn/gnn_ablate_B_act_sparse_static/ckpt_best.pt"
    checklist = {
        "post_hoc": True,
        "model": str(ckpt),
        "model_exists": ckpt.is_file(),
        "branch_flags": {"use_act": True, "use_sparse": True, "use_static": True, "use_graph": False},
        "required_inputs": {
            "act_bag_252": "per-stay 252-dim activation bag over node types var|L|k|stage "
                           "(activation_name), needs activations_long.parquet-equivalent on eICU",
            "sparse_308": "recomputable for eICU from hourly_grid_48h.parquet (validated port, this script)",
            "static_11": "available in eicu stay_features_48h.parquet",
        },
        "eicu_assets": {
            "activations_long_exists": any(
                (PROJ / "data/external/eicu").glob("*activation*")),
            "motif_atoms_winner_only": "data/external/eicu/motif_atoms.parquet (84 family-level winner events, no (L,k) identity)",
            "sparse_recomputable": True,
            "hourly_grid_rows": int(len(grid_e)),
        },
        "blocker": "eICU has NO activations_long: winner motif atoms collapse (var,L,k)->family (84 concepts); "
                   "the 252-node activation bag requires re-running frozen-template shapelet matching "
                   "(scripts/14-15 equivalent) over eICU hourly grids with shapelet_frozen_qp50.pt templates "
                   "(torch soft-DTW, ~21,657 stays x 7 vars x 48h).",
        "risk": "feature-drift: winner/event_rank policy, a_min=0.5 filter, empty-graph stay handling "
                "(MIMIC train_empty=635), and grid definition differences between MIMIC and eICU pipelines.",
        "feasible_in_30_60min": False,
    }
    results["B_arm_deployment_checklist"] = checklist
    (OUT_DIR / "b_arm_feasibility.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {OUT_DIR}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
