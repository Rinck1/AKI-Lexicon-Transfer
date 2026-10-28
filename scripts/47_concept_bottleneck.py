#!/usr/bin/env python3
"""CBM concept-bottleneck Phase A (docs/26).

Fits a linear "concept -> death" bottleneck over the human-readable concept
layer (84 motif concepts with continuous activation `a`, plus 11-lab direction
indicators). Evaluation is train-split 5-fold only — never reads val / holdout
/ test, never rewrites frozen assets.

Modes:
  --dry-run   synthetic smoke (default)
  --train     real train-split 5-fold CV + intervention + coefficient table

Usage:
  python -u scripts/47_concept_bottleneck.py --dry-run
  python -u scripts/47_concept_bottleneck.py --train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.cbm import (  # noqa: E402
    KIND_LAB,
    KIND_MOTIF,
    build_concept_matrix,
    build_static_matrix,
    coef_table,
    concept_interventions,
    fit_full_linear,
    standardize_train_only,
    subject_groups,
    _cv_scores,
)
from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.path_protocol import (  # noqa: E402
    PATH_SEED,
    refuse_test_path,
    refuse_test_split,
)
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402

EVENTS_DIR = PROJ / "data/processed/shapelet/events/events_final_7var_motif84"
PATHS_DIR = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"
OUT_DIR = PROJ / "data/processed/cbm/cbm_phase_a"


def _log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Synthetic dry-run
# ---------------------------------------------------------------------------
def _synthetic_stays(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stays = pd.DataFrame(
        {
            "ICUSTAY_ID": np.arange(n),
            "SUBJECT_ID": np.arange(n),
            "HOSPITAL_MORTALITY": (rng.random(n) < 0.30).astype(float),
            "AGE": rng.normal(60, 15, n),
            "GENDER_M": (rng.random(n) < 0.55).astype(float),
        }
    )
    for c in SHAPELET_STATIC_COLS:
        if c not in stays.columns:
            stays[c] = rng.normal(0, 1, n)
    return stays.set_index("ICUSTAY_ID")


def _synthetic_atoms(n_stay: int, n_motif: int, n_lab: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    m_rows = []
    for s in range(n_stay):
        for j in range(n_motif):
            if rng.random() < 0.3:
                m_rows.append(
                    {
                        "ICUSTAY_ID": s,
                        "SUBJECT_ID": s,
                        "split": "train",
                        "HOSPITAL_MORTALITY": 0.0,
                        "var": "mbp",
                        "family_label": "f",
                        "stage": f"B{j % 4 + 1}",
                        "stage_idx": j % 4,
                        "a": float(0.5 + 0.5 * rng.random()),
                        "atom": f"mbp|fam{j}|B{j % 4 + 1}",
                        "kind": KIND_MOTIF,
                        "direction": "",
                    }
                )
    l_rows = []
    for s in range(n_stay):
        for j in range(n_lab):
            if rng.random() < 0.25:
                direction = "升高" if rng.random() < 0.5 else "下降"
                l_rows.append(
                    {
                        "ICUSTAY_ID": s,
                        "SUBJECT_ID": s,
                        "split": "train",
                        "HOSPITAL_MORTALITY": 0.0,
                        "var": "lactate",
                        "family_label": "",
                        "stage": f"B{j % 4 + 1}",
                        "stage_idx": j % 4,
                        "a": np.nan,
                        "atom": f"lactate|B{j % 4 + 1}|{direction}",
                        "kind": KIND_LAB,
                        "direction": direction,
                    }
                )
    return pd.DataFrame(m_rows), pd.DataFrame(l_rows)


def run_dry_run(seed: int = PATH_SEED) -> dict:
    stays = _synthetic_stays(100, seed)
    stay_ids = sorted(int(s) for s in stays.index)
    m_atoms, l_atoms = _synthetic_atoms(len(stay_ids), 12, 6, seed)

    motif_concepts = sorted(m_atoms["atom"].astype(str).unique())
    lab_concepts = sorted(l_atoms["atom"].astype(str).unique())
    concept_ids = motif_concepts + lab_concepts
    kinds = ([KIND_MOTIF] * len(motif_concepts)) + ([KIND_LAB] * len(lab_concepts))

    y = np.array([float(stays.loc[s, "HOSPITAL_MORTALITY"]) for s in stay_ids])
    groups = subject_groups(stays, stay_ids)
    static_raw = build_static_matrix(stays, stay_ids, SHAPELET_STATIC_COLS)
    static, _, _ = standardize_train_only(static_raw)

    motif_a = build_concept_matrix(m_atoms, stay_ids, motif_concepts, kind_of={}, use_activation=True)
    motif_01 = build_concept_matrix(m_atoms, stay_ids, motif_concepts, kind_of={}, use_activation=False)
    lab_01 = build_concept_matrix(l_atoms, stay_ids, lab_concepts, kind_of={}, use_activation=True)

    empty = np.zeros((len(stay_ids), 0))
    report = {
        "mode": "dry-run",
        "n_stay": len(stay_ids),
        "n_motif_concepts": len(motif_concepts),
        "n_lab_concepts": len(lab_concepts),
        "cv_static": _cv_scores(y, groups, static, empty, seed),
        "cv_static_plus_84_indicator": _cv_scores(y, groups, static, motif_01, seed),
        "cv_static_plus_84_activation": _cv_scores(y, groups, static, motif_a, seed),
        "cv_static_plus_84a_plus_lab": _cv_scores(
            y, groups, static, np.column_stack([motif_a, lab_01]), seed
        ),
    }
    full = fit_full_linear(y, static, np.column_stack([motif_a, lab_01]))
    iv = concept_interventions(
        full, static, np.column_stack([motif_a, lab_01]), concept_ids, kinds, static.shape[1]
    )
    report["n_intervention_concepts"] = len([r for r in iv if r.n_with > 0])
    report["intervention_top_risk"] = [
        {"concept": r.concept, "n": r.n_with, "mean_dp": round(r.mean_delta_p, 5)}
        for r in sorted(iv, key=lambda r: r.mean_delta_p if np.isfinite(r.mean_delta_p) else -1e9, reverse=True)[:3]
    ]
    _log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if report["cv_static_plus_84a_plus_lab"]["n_folds_used"] < 5:
        raise RuntimeError("dry-run: fewer than 5 folds used")
    _log("[47] dry-run OK")
    return report


# ---------------------------------------------------------------------------
# Real train-split run
# ---------------------------------------------------------------------------
def run_train(args: argparse.Namespace) -> dict:
    refuse_test_path(args.events_dir)
    refuse_test_path(args.paths_dir)
    refuse_test_path(args.out_dir)

    idx = pd.read_parquet(Path(args.events_dir) / "stay_index.parquet")
    for sp in idx["split"].astype(str).unique():
        refuse_test_split(sp)
    train_ids = sorted(int(x) for x in idx.loc[idx["split"] == "train", "ICUSTAY_ID"])

    m_atoms = pd.read_parquet(Path(args.paths_dir) / "motif_atoms.parquet")
    l_atoms = pd.read_parquet(Path(args.paths_dir) / "lab_atoms.parquet")
    for df in (m_atoms, l_atoms):
        for sp in df["split"].astype(str).unique():
            refuse_test_split(sp)
    m_atoms = m_atoms[m_atoms["split"] == "train"]
    l_atoms = l_atoms[l_atoms["split"] == "train"]

    stays = pd.read_parquet(Path(DATA_PROCESSED) / "stay_features_48h.parquet").set_index("ICUSTAY_ID")

    motif_concepts = sorted(m_atoms["atom"].astype(str).unique())
    lab_concepts = sorted(l_atoms["atom"].astype(str).unique())
    concept_ids = motif_concepts + lab_concepts
    kinds = ([KIND_MOTIF] * len(motif_concepts)) + ([KIND_LAB] * len(lab_concepts))
    _log(
        f"[47] train stays={len(train_ids)}  motif_concepts={len(motif_concepts)}  "
        f"lab_concepts={len(lab_concepts)}"
    )

    y = np.array([float(stays.loc[s, "HOSPITAL_MORTALITY"]) for s in train_ids])
    groups = subject_groups(stays, train_ids)
    static_raw = build_static_matrix(stays, train_ids, SHAPELET_STATIC_COLS)
    static, mu, sd = standardize_train_only(static_raw)

    motif_a = build_concept_matrix(m_atoms, train_ids, motif_concepts, kind_of={}, use_activation=True)
    motif_01 = build_concept_matrix(m_atoms, train_ids, motif_concepts, kind_of={}, use_activation=False)
    lab_01 = build_concept_matrix(l_atoms, train_ids, lab_concepts, kind_of={}, use_activation=True)

    empty = np.zeros((len(train_ids), 0))
    cv_static = _cv_scores(y, groups, static, empty, args.seed)
    cv_84ind = _cv_scores(y, groups, static, motif_01, args.seed)
    cv_84a = _cv_scores(y, groups, static, motif_a, args.seed)
    cv_84a_lab = _cv_scores(y, groups, static, np.column_stack([motif_a, lab_01]), args.seed)

    full = fit_full_linear(y, static, np.column_stack([motif_a, lab_01]))
    iv = concept_interventions(
        full, static, np.column_stack([motif_a, lab_01]), concept_ids, kinds, static.shape[1]
    )
    coef_df = coef_table(full, concept_ids, kinds, static.shape[1])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(out / "cbm_coef_table.csv", index=False, encoding="utf-8-sig")

    iv_rows = [
        {
            "concept": r.concept,
            "kind": r.kind,
            "n_with": r.n_with,
            "mean_delta_p": r.mean_delta_p,
            "frac_with": round(r.frac_with, 5),
            "mean_abs_coef": round(r.mean_abs_coef, 6),
        }
        for r in iv
        if r.n_with > 0
    ]
    iv_sorted = sorted(
        iv_rows, key=lambda r: r["mean_delta_p"] if np.isfinite(r["mean_delta_p"]) else -1e9, reverse=True
    )
    report = {
        "stage": "cbm_phase_a_train",
        "seed": args.seed,
        "test_read": False,
        "val_read": False,
        "holdout_read": False,
        "n_train_stays": len(train_ids),
        "n_motif_concepts": len(motif_concepts),
        "n_lab_concepts": len(lab_concepts),
        "n_concepts": len(concept_ids),
        "cv_static": cv_static,
        "cv_static_plus_84_indicator": cv_84ind,
        "cv_static_plus_84_activation": cv_84a,
        "cv_static_plus_84a_plus_lab": cv_84a_lab,
        "intervention_top_risk": iv_sorted[:10],
        "intervention_top_protective": iv_sorted[-10:][::-1],
        "n_intervention_concepts": len(iv_rows),
        "coef_table": str((out / "cbm_coef_table.csv").resolve()),
        "events_dir": str(Path(args.events_dir).resolve()),
        "paths_dir": str(Path(args.paths_dir).resolve()),
        "events_stay_index_sha256": sha256_file(Path(args.events_dir) / "stay_index.parquet"),
        "motif_atoms_sha256": sha256_file(Path(args.paths_dir) / "motif_atoms.parquet"),
        "lab_atoms_sha256": sha256_file(Path(args.paths_dir) / "lab_atoms.parquet"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "cbm_train_cv.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out / "cbm_intervention.json").write_text(
        json.dumps(iv_rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    manifest = {
        "protocol": "docs/26_CBM概念瓶颈实验协议.md",
        "model": "linear logistic: logit = b0 + static + 84 motif a + 87 lab direction",
        "evaluation": "train-split 5-fold (StratifiedGroupKFold by SUBJECT_ID)",
        "val_used": False,
        "holdout_used": False,
        "intervention": "concept-level zeroing among active rows, mean Δp",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "cbm_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _log(json.dumps({k: report[k] for k in report if k not in ("intervention_top_risk", "intervention_top_protective")}, indent=2, ensure_ascii=False, default=str))
    _log("[47] train DONE")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--seed", type=int, default=PATH_SEED)
    p.add_argument("--events-dir", type=str, default=str(EVENTS_DIR))
    p.add_argument("--paths-dir", type=str, default=str(PATHS_DIR))
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train and not args.dry_run:
        args.dry_run = True
    if args.dry_run:
        run_dry_run(args.seed)
        return
    run_train(args)


if __name__ == "__main__":
    main()
