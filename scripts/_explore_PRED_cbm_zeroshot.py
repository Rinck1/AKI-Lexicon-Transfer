#!/usr/bin/env python3
"""PRED post-hoc explore: CBM linear arms zero-shot transfer MIMIC -> eICU.

Fits logistic arms on MIMIC train split ONLY (one full-train fit, same
convention as scripts/46 for path coefficients), then predicts eICU
zero-shot. No selection/tuning on MIMIC val/holdout; the MIMIC val row is
descriptive-only. Read-only on existing assets; outputs go to
data/external/eicu/_explore_PRED/.

Arms:
  static            11 static cols (SHAPELET_STATIC_COLS)
  static+15paths    frozen 15 path indicators (path_set_selected.json)
  static+84a        + 84 motif concepts, continuous activation a  (CBM motif)
  static+84ind      + 84 motif concepts, 0/1 indicator
  static+87lab      + 87 lab direction atoms                       (CBM lab)
  cbm_full          static + 84a + 87lab                            (CBM linear arm)

post-hoc exploratory; frozen name lists are not modified.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

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
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.path_protocol import parse_path_key  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402

OUT_DIR = Path(PROJ) / "data/external/eicu/_explore_PRED"
SEED = 42


def fast_indicator(stay_ids, stay_atoms, paths):
    """Verbatim port from scripts/_tmp_urine_threshold.py (15 frozen paths)."""
    ind = np.zeros((len(stay_ids), len(paths)), dtype=np.float64)
    for i, sid in enumerate(stay_ids):
        atoms = stay_atoms.get(int(sid))
        if not atoms:
            continue
        astages: dict[str, set] = {}
        for st, atom, _ in atoms:
            astages.setdefault(atom, set()).add(int(st))
        astages = {a: sorted(s) for a, s in astages.items()}
        for j, p in enumerate(paths):
            prev = -1
            ok = True
            for a in p:
                stages = astages.get(a)
                if not stages:
                    ok = False
                    break
                nxt = next((s for s in stages if s > prev), None)
                if nxt is None:
                    ok = False
                    break
                prev = nxt
            if ok:
                ind[i, j] = 1.0
    return ind


def metrics(y, p):
    return {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- MIMIC side (fit = train only) ----------------
    _grid, mimic_stays = load_shapelet_tables(DATA_PROCESSED)
    stays_m = mimic_stays.set_index("ICUSTAY_ID")
    train_ids = sorted(int(x) for x in mimic_stays.loc[mimic_stays["split"] == "train", "ICUSTAY_ID"])
    val_ids = sorted(int(x) for x in mimic_stays.loc[mimic_stays["split"] == "val", "ICUSTAY_ID"])

    m_atoms = pd.read_parquet(MIMIC_PATHS_DIR / "motif_atoms.parquet")
    l_atoms = pd.read_parquet(MIMIC_PATHS_DIR / "lab_atoms.parquet")
    mt_tr = m_atoms[m_atoms["split"] == "train"]
    lt_tr = l_atoms[l_atoms["split"] == "train"]

    motif_concepts = sorted(mt_tr["atom"].astype(str).unique())
    lab_concepts = sorted(lt_tr["atom"].astype(str).unique())
    assert len(motif_concepts) == 84 and len(lab_concepts) == 87, (
        len(motif_concepts),
        len(lab_concepts),
    )
    kind_of = {c: KIND_MOTIF for c in motif_concepts}
    kind_of.update({c: KIND_LAB for c in lab_concepts})

    y_tr = stays_m.loc[train_ids, "HOSPITAL_MORTALITY"].to_numpy(dtype=float)
    static_raw_tr = build_static_matrix(stays_m, train_ids, SHAPELET_STATIC_COLS)
    static_tr, mu, sd = standardize_train_only(static_raw_tr)

    motif_a_tr = build_concept_matrix(mt_tr, train_ids, motif_concepts, kind_of=kind_of, use_activation=True)
    motif_01_tr = build_concept_matrix(mt_tr, train_ids, motif_concepts, kind_of=kind_of, use_activation=False)
    lab_tr = build_concept_matrix(lt_tr, train_ids, lab_concepts, kind_of=kind_of, use_activation=True)

    sel = json.loads((MIMIC_PATHS_DIR / "path_set_selected.json").read_text(encoding="utf-8"))
    paths = [parse_path_key(p) for p in sel["selected"]]
    ma_tr_atoms = atoms_by_stay(m_atoms[m_atoms["split"] == "train"])
    paths_tr = fast_indicator(train_ids, ma_tr_atoms, paths)

    def fit_arm(name, X):
        clf = LogisticRegression(max_iter=500, solver="lbfgs")
        clf.fit(X, y_tr)
        print(f"[fit] {name:<22} n={X.shape[0]} d={X.shape[1]} done {time.time()-t0:.0f}s", flush=True)
        return clf

    arms = {
        "static": static_tr,
        "static+15paths": np.column_stack([static_tr, paths_tr]),
        "static+84a": np.column_stack([static_tr, motif_a_tr]),
        "static+84ind": np.column_stack([static_tr, motif_01_tr]),
        "static+87lab": np.column_stack([static_tr, lab_tr]),
        "cbm_full": np.column_stack([static_tr, motif_a_tr, lab_tr]),
    }
    clfs = {k: fit_arm(k, X) for k, X in arms.items()}

    # ---------------- evaluation sets ----------------
    # MIMIC val (descriptive only)
    y_va = stays_m.loc[val_ids, "HOSPITAL_MORTALITY"].to_numpy(dtype=float)
    static_va = (build_static_matrix(stays_m, val_ids, SHAPELET_STATIC_COLS) - mu) / sd
    motif_a_va = build_concept_matrix(m_atoms[m_atoms["split"] == "val"], val_ids, motif_concepts, kind_of=kind_of, use_activation=True)
    motif_01_va = build_concept_matrix(m_atoms[m_atoms["split"] == "val"], val_ids, motif_concepts, kind_of=kind_of, use_activation=False)
    lab_va = build_concept_matrix(l_atoms[l_atoms["split"] == "val"], val_ids, lab_concepts, kind_of=kind_of, use_activation=True)
    paths_va = fast_indicator(val_ids, atoms_by_stay(m_atoms[m_atoms["split"] == "val"]), paths)
    X_va = {
        "static": static_va,
        "static+15paths": np.column_stack([static_va, paths_va]),
        "static+84a": np.column_stack([static_va, motif_a_va]),
        "static+84ind": np.column_stack([static_va, motif_01_va]),
        "static+87lab": np.column_stack([static_va, lab_va]),
        "cbm_full": np.column_stack([static_va, motif_a_va, lab_va]),
    }

    # eICU zero-shot
    stays_e = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    e_ids = sorted(int(x) for x in stays_e.index)
    y_e = stays_e.loc[e_ids, "HOSPITAL_MORTALITY"].to_numpy(dtype=float)
    static_e = (build_static_matrix(stays_e, e_ids, SHAPELET_STATIC_COLS) - mu) / sd
    em = load_eicu("motif_atoms")
    el = load_eicu("lab_atoms")
    unseen_m = sorted(set(em["atom"].astype(str).unique()) - set(motif_concepts))
    unseen_l = sorted(set(el["atom"].astype(str).unique()) - set(lab_concepts))
    motif_a_e = build_concept_matrix(em, e_ids, motif_concepts, kind_of=kind_of, use_activation=True)
    motif_01_e = build_concept_matrix(em, e_ids, motif_concepts, kind_of=kind_of, use_activation=False)
    lab_e = build_concept_matrix(el, e_ids, lab_concepts, kind_of=kind_of, use_activation=True)
    paths_e = fast_indicator(e_ids, atoms_by_stay(load_eicu("atoms")), paths)
    print(
        f"[eicu] n={len(e_ids)} mort={y_e.mean():.4f} unseen motif={len(unseen_m)} unseen lab={len(unseen_l)} "
        f"motif_cov={float((motif_01_e.sum(1) > 0).mean()):.3f} lab_cov={float((lab_e.sum(1) > 0).mean()):.3f} "
        f"path_any={float((paths_e.sum(1) > 0).mean()):.3f}",
        flush=True,
    )
    X_e = {
        "static": static_e,
        "static+15paths": np.column_stack([static_e, paths_e]),
        "static+84a": np.column_stack([static_e, motif_a_e]),
        "static+84ind": np.column_stack([static_e, motif_01_e]),
        "static+87lab": np.column_stack([static_e, lab_e]),
        "cbm_full": np.column_stack([static_e, motif_a_e, lab_e]),
    }

    # ---------------- evaluate + save ----------------
    pred_cols = {}
    results = {"post_hoc": True, "fit": "mimic train split only, one full-train logistic (C=1, lbfgs)", "seed": SEED, "arms": {}}
    frames = {}
    for name in arms:
        row = {}
        for set_name, y, Xd in [("mimic_val(desc)", y_va, X_va), ("eicu_zeroshot", y_e, X_e)]:
            p = clfs[name].predict_proba(Xd[name])[:, 1]
            row[set_name] = metrics(y, p)
            frames[(set_name, name)] = p
        results["arms"][name] = row
        mv, ez = row["mimic_val(desc)"], row["eicu_zeroshot"]
        print(
            f"[{name:<16}] val AUPRC {mv['auprc']:.3f} (desc) | eICU AUPRC {ez['auprc']:.3f} "
            f"AUROC {ez['auroc']:.3f} Brier {ez['brier']:.3f}",
            flush=True,
        )

    for set_name, ids, y, key in [("eicu", e_ids, y_e, "eicu_zeroshot"), ("mimic_val", val_ids, y_va, "mimic_val(desc)")]:
        df = pd.DataFrame({"ICUSTAY_ID": ids, "y": y})
        for name in arms:
            df[f"p__{name.replace('+', '_plus_')}"] = frames[(key, name)]
        df.to_parquet(OUT_DIR / f"pred_{set_name}.parquet", index=False)

    results["coverage"] = {
        "eicu_unseen_motif_concepts": unseen_m,
        "eicu_unseen_lab_concepts": unseen_l,
        "n_motif_concepts": len(motif_concepts),
        "n_lab_concepts": len(lab_concepts),
        "n_paths": len(paths),
    }
    (OUT_DIR / "cbm_zeroshot_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"→ {OUT_DIR}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
