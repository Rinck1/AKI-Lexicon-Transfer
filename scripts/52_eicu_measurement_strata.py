#!/usr/bin/env python3
"""Round-2 (52): pre-specified measurement-density stratification (docs/31 A3).

Hypothesis (fixed before results): the anti-transfer of the 87 discretized lab
atoms and the lost paths are partly driven by sparser measurement practice in
eICU; both should attenuate in adequately-measured strata.

Strata (pre-specified):
  S1 = GCS adequately measured (>=2 hours with gcs_total_is_measured in the
       48h grid; sensitivity >=1)
  S2 = urine complete (n_urine_hours_measured >= 12, docs/24)
  S3 = S1 & S2

Within each stratum (eICU subsets, zero refit — coefficients are one full
MIMIC-train fit per arm):
  (a) zero-shot AUPRC of static / static+84a / static+87lab
  (b) protocol-criterion same-sign count for the 15 frozen paths
  (c) crude death-rate-diff sign agreement with MIMIC train for gcs_total and
      lactate lab atoms

Outputs: data/external/eicu/round2_strata/strata_results.json
         outputs/reports/52_测量分层归因.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

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


def fast_indicator(stay_ids, stay_atoms, paths):
    """Greedy earliest-stage matching of specific ordered paths (linear time).
    Replacement for path_indicator_matrix, which enumerates candidate
    sequences per stay and blows up on 21k eICU stays."""
    ind = np.zeros((len(stay_ids), len(paths)), dtype=np.float64)
    for i, sid in enumerate(stay_ids):
        atoms = stay_atoms.get(int(sid))
        if not atoms:
            continue
        astages: dict[str, set] = {}
        for st, atom, _ in atoms:
            astages.setdefault(str(atom), set()).add(int(st))
        astages = {a: sorted(s) for a, s in astages.items()}
        for j, p in enumerate(paths):
            prev, ok = -1, True
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
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402

OUT_DIR = PROJ / "data/external/eicu" / "round2_strata"
REPORT = PROJ / "outputs" / "reports" / "52_测量分层归因.md"


def log(m: str) -> None:
    print(m, flush=True)


def metrics(y, p):
    return {
        "n": int(len(y)),
        "mortality": float(np.mean(y)),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- strata ----------------
    grid = pd.read_parquet(load_eicu.__self__.dir / "hourly_grid_48h.parquet") if False else pd.read_parquet(
        PROJ / "data/external/eicu/hourly_grid_48h.parquet", columns=["ICUSTAY_ID", "gcs_total_is_measured"]
    )
    gcs_hours = grid.groupby("ICUSTAY_ID")["gcs_total_is_measured"].sum()
    primary = pd.read_parquet(PROJ / "data/external/eicu/aki_primary_ha48.parquet")
    urine_h = primary.set_index("ICUSTAY_ID")["n_urine_hours_measured"]

    stays_e = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    e_ids = sorted(int(x) for x in stays_e.index)
    y_e_all = stays_e.loc[e_ids, "HOSPITAL_MORTALITY"].to_numpy(dtype=float)
    idx_e = pd.Index(e_ids)
    gcs_n = gcs_hours.reindex(idx_e).fillna(0)
    uri_n = urine_h.reindex(idx_e).fillna(0)

    strata = {
        "full": np.ones(len(e_ids), dtype=bool),
        "S1_gcs_ge2": (gcs_n >= 2).to_numpy(),
        "S1_gcs_ge1_sens": (gcs_n >= 1).to_numpy(),
        "S2_urine_ge12": (uri_n >= 12).to_numpy(),
        "S3_gcs2_and_urine12": ((gcs_n >= 2) & (uri_n >= 12)).to_numpy(),
    }
    for k, m_ in strata.items():
        log(f"[strata] {k}: n={int(m_.sum()):,} mort={y_e_all[m_].mean():.4f}")

    # ---------------- MIMIC train fits (one per arm, full train) ----------------
    stays_m = pd.read_parquet(Path(DATA_PROCESSED) / "stay_features_48h.parquet").set_index("ICUSTAY_ID")
    train_ids = sorted(int(x) for x in stays_m.loc[stays_m["split"] == "train"].index)
    y_tr = stays_m.loc[train_ids, "HOSPITAL_MORTALITY"].to_numpy(dtype=float)

    m_motif = pd.read_parquet(MIMIC_PATHS_DIR / "motif_atoms.parquet")
    m_lab = pd.read_parquet(MIMIC_PATHS_DIR / "lab_atoms.parquet")
    motif_ids = sorted(m_motif.loc[m_motif["split"] == "train", "atom"].astype(str).unique())
    lab_ids = sorted(m_lab.loc[m_lab["split"] == "train", "atom"].astype(str).unique())
    log(f"[mimic] concepts: motif={len(motif_ids)} lab={len(lab_ids)}")

    st_tr_raw = build_static_matrix(stays_m, train_ids, SHAPELET_STATIC_COLS)
    st_tr, mu, sd = standardize_train_only(st_tr_raw)
    st_e = (build_static_matrix(stays_e, e_ids, SHAPELET_STATIC_COLS) - mu) / sd

    A_tr = build_concept_matrix(m_motif[m_motif["split"] == "train"], train_ids, motif_ids,
                                kind_of={c: KIND_MOTIF for c in motif_ids}, use_activation=True)
    L_tr = build_concept_matrix(m_lab[m_lab["split"] == "train"], train_ids, lab_ids,
                                kind_of={c: KIND_LAB for c in lab_ids}, use_activation=False)
    e_motif = load_eicu("motif_atoms")
    e_lab = load_eicu("lab_atoms")
    A_e = build_concept_matrix(e_motif, e_ids, motif_ids,
                               kind_of={c: KIND_MOTIF for c in motif_ids}, use_activation=True)
    L_e = build_concept_matrix(e_lab, e_ids, lab_ids,
                               kind_of={c: KIND_LAB for c in lab_ids}, use_activation=False)

    arms: dict[str, tuple] = {}
    for name, (Xtr_extra, Xe_extra) in {
        "static": (None, None),
        "static+84a": (A_tr, A_e),
        "static+87lab": (L_tr, L_e),
    }.items():
        Xtr = st_tr if Xtr_extra is None else np.column_stack([st_tr, Xtr_extra])
        Xe = st_e if Xe_extra is None else np.column_stack([st_e, Xe_extra])
        clf = LogisticRegression(max_iter=500, solver="lbfgs").fit(Xtr, y_tr)
        arms[name] = (clf.predict_proba(Xe)[:, 1], clf)
        log(f"[fit] {name} done")

    # ---------------- (b) 15 paths: protocol same-sign per stratum ----------------
    err = pd.read_csv(PROJ / "data/external/eicu/round2_erratum/path_sign_erratum_table.csv")
    sel = json.loads((MIMIC_PATHS_DIR / "path_set_selected.json").read_text(encoding="utf-8"))
    paths = [parse_path_key(p) for p in sel["selected"]]
    e_atoms = load_eicu("atoms")
    ind_e = fast_indicator(e_ids, atoms_by_stay(e_atoms), paths)
    train_dir = err.set_index("no") if "no" in err.columns else err
    dir_list = []
    for i in range(15):
        row = err[err["no"] == i + 1].iloc[0]
        dir_list.append(1.0 if str(row["mimic_train_direction"]).startswith("risk") else -1.0)

    def path_same_sign(mask: np.ndarray) -> dict:
        yy = y_e_all[mask]
        out = {"n_same": 0, "n_support_ok": 0, "detail": []}
        for j in range(15):
            on = ind_e[mask, j] > 0
            n_on = int(on.sum())
            if n_on < 20:
                out["detail"].append({"j": j + 1, "n": n_on, "sign": None})
                continue
            d = float(yy[on].mean() - yy[~on].mean())
            out["n_support_ok"] += 1
            same = (np.sign(d) == dir_list[j])
            out["n_same"] += int(same)
            out["detail"].append({"j": j + 1, "n": n_on, "diff": round(d, 4), "same": bool(same)})
        return out

    # ---------------- (c) gcs/lactate atom sign agreement per stratum ----------------
    def atom_sign_table(mask: np.ndarray, vars_: list[str]) -> dict:
        yy = y_e_all[mask]
        res = {}
        for var in vars_:
            agree, tot = 0, 0
            for j, c in enumerate(lab_ids):
                if not c.startswith(var + "|"):
                    continue
                # MIMIC train crude sign
                m_col = L_tr[:, j] > 0
                if m_col.sum() < 20:
                    continue
                m_sign = np.sign(y_tr[m_col].mean() - y_tr[~m_col].mean())
                on = L_e[mask, j] > 0
                if on.sum() < 20 or (~on).sum() < 20:
                    continue
                e_sign = np.sign(yy[on].mean() - yy[~on].mean())
                tot += 1
                agree += int(e_sign == m_sign)
            res[var] = {"n_atoms_eval": tot, "n_agree": agree}
        return res

    # ---------------- run all strata ----------------
    results: dict = {"round": "round-2 (pre-specified docs/31 A3)", "zero_refit": True}
    results["strata_n"] = {k: {"n": int(m.sum()), "mortality": float(y_e_all[m].mean())} for k, m in strata.items()}
    results["arms_by_stratum"] = {}
    results["paths_by_stratum"] = {}
    results["gcs_lactate_by_stratum"] = {}
    for k, mask in strata.items():
        results["arms_by_stratum"][k] = {
            name: metrics(y_e_all[mask], p[mask]) for name, (p, _) in arms.items()
        }
        results["paths_by_stratum"][k] = path_same_sign(mask)
        results["gcs_lactate_by_stratum"][k] = atom_sign_table(mask, ["gcs_total", "lactate"])
        log(f"[done] {k}")

    (OUT_DIR / "strata_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ["strata_n", "arms_by_stratum"]}, indent=2, ensure_ascii=False))
    for k in strata:
        ps = results["paths_by_stratum"][k]
        log(f"[paths] {k}: same={ps['n_same']}/15 support_ok={ps['n_support_ok']} | gcs={results['gcs_lactate_by_stratum'][k]['gcs_total']} lactate={results['gcs_lactate_by_stratum'][k]['lactate']}")
    log(f"→ {OUT_DIR / 'strata_results.json'}")


if __name__ == "__main__":
    main()
