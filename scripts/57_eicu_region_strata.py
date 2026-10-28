#!/usr/bin/env python3
"""Round-3 (2026-10-11): does the lexicon transfer uniformly across eICU regions,
hospital types and individual hospitals?

Design (written before the first run; descriptive, no pass/fail gate):
  - Strata from eICU `hospital` table: region (Midwest/Northeast/South/West/unknown),
    teaching status, bed-size category. Full eICU row = reproduction check.
  - Per stratum, the same three lexicon layers as the main analysis:
      L1 distribution: JSD + Spearman rho of 84-node prevalence vs MIMIC holdout
                       (scripts/48 definitions; full-cohort check 0.132 / 0.89)
      L2 association : crude death-rate difference per node (node on vs off),
                       sign agreement with MIMIC train (84 nodes and the 34
                       Bonferroni nodes); "replicated" = same sign and 95% CI
                       excluding 0 (Wald CI for a difference of proportions;
                       full-cohort check vs bootstrap-based 62/84, 30/34)
      L3 zero-shot   : static vs static+84a logistic fitted once on MIMIC train
                       (as in _explore_PRED_cbm_zeroshot); AUPRC/AUROC, paired
                       bootstrap CI (B=500) for the increment; retention =
                       stratum AUPRC / MIMIC val AUPRC of the same arm
  - Hospital level (hospitals with >=100 stays and >=15 deaths): per-hospital
    AUROC increment (+84a minus static), bootstrap SE (B=200), DerSimonian-Laird
    random-effects pooling with tau^2 and I^2. AUROC is used because AUPRC is not
    comparable across hospitals with different mortality.
Outputs: data/external/eicu/round3_region/region_results.json, strata_table.csv,
         hospital_table.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.cbm import KIND_MOTIF, build_concept_matrix, build_static_matrix, standardize_train_only  # noqa: E402
from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.eicu_constants import MIMIC_PATHS_DIR  # noqa: E402
from scripts.utils.eicu_io import eicu_table, load_eicu  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402

OUT_DIR = PROJ / "data/external/eicu/round3_region"
SEED = 42
MIN_SUPPORT = 20


def jsd(p, q):
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return float(jensenshannon(p, q, base=2.0))


def node_assoc(y, ind):
    """Crude death-rate diff (on - off) and Wald 95% CI per column."""
    n1 = ind.sum(0)
    n0 = len(y) - n1
    d1 = (ind * y[:, None]).sum(0)
    r1 = np.divide(d1, n1, out=np.full(len(n1), np.nan), where=n1 > 0)
    r0 = np.divide(y.sum() - d1, n0, out=np.full(len(n0), np.nan), where=n0 > 0)
    diff = r1 - r0
    se = np.sqrt(np.divide(r1 * (1 - r1), n1, out=np.full(len(n1), np.nan), where=n1 > 0)
                 + np.divide(r0 * (1 - r0), n0, out=np.full(len(n0), np.nan), where=n0 > 0))
    return diff, diff - 1.96 * se, diff + 1.96 * se, n1


def boot_delta(y, p1, p0, rng, b, metric):
    n = len(y)
    out = []
    for _ in range(b):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, n):
            continue
        out.append(metric(y[i], p1[i]) - metric(y[i], p0[i]))
    return np.array(out)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- MIMIC: fit static and static+84a on train once ----
    _g, ms = load_shapelet_tables(DATA_PROCESSED)
    sm = ms.set_index("ICUSTAY_ID")
    tr = sorted(int(x) for x in ms.loc[ms["split"] == "train", "ICUSTAY_ID"])
    va = sorted(int(x) for x in ms.loc[ms["split"] == "val", "ICUSTAY_ID"])
    y_tr = sm.loc[tr, "HOSPITAL_MORTALITY"].to_numpy(float)
    y_va = sm.loc[va, "HOSPITAL_MORTALITY"].to_numpy(float)
    m_atoms = pd.read_parquet(MIMIC_PATHS_DIR / "motif_atoms.parquet")
    concepts = sorted(m_atoms.loc[m_atoms["split"] == "train", "atom"].astype(str).unique())
    kind_of = {c: KIND_MOTIF for c in concepts}
    st_tr, mu, sd = standardize_train_only(build_static_matrix(sm, tr, SHAPELET_STATIC_COLS))
    a_tr = build_concept_matrix(m_atoms[m_atoms["split"] == "train"], tr, concepts, kind_of=kind_of, use_activation=True)
    st_va = (build_static_matrix(sm, va, SHAPELET_STATIC_COLS) - mu) / sd
    a_va = build_concept_matrix(m_atoms[m_atoms["split"] == "val"], va, concepts, kind_of=kind_of, use_activation=True)
    c0 = LogisticRegression(max_iter=500, solver="lbfgs").fit(st_tr, y_tr)
    c1 = LogisticRegression(max_iter=500, solver="lbfgs").fit(np.column_stack([st_tr, a_tr]), y_tr)
    val_ap = {"static": float(average_precision_score(y_va, c0.predict_proba(st_va)[:, 1])),
              "static+84a": float(average_precision_score(y_va, c1.predict_proba(np.column_stack([st_va, a_va]))[:, 1]))}

    # MIMIC references: holdout prevalence (L1) and train crude diffs (L2)
    hold = pd.read_parquet(MIMIC_PATHS_DIR / "holdout_atoms.parquet")
    n_hold = hold["ICUSTAY_ID"].nunique()
    hm = hold[hold["kind"] == "motif"]
    prev_hold = np.array([hm.loc[hm["atom"] == c, "ICUSTAY_ID"].nunique() / n_hold for c in concepts])
    ref = pd.read_csv(PROJ / "data/external/eicu/_explore_NODE/node_assoc_transfer_84.csv", encoding="utf-8-sig").set_index("node")
    m_diff = ref.loc[concepts, "mimic_diff"].to_numpy(float)
    m_bonf = ref.loc[concepts, "mimic_bonf"].to_numpy(bool)
    assert m_bonf.sum() == 34

    # ---- eICU ----
    se_ = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    ee = sorted(int(x) for x in se_.index)
    y_e = se_.loc[ee, "HOSPITAL_MORTALITY"].to_numpy(float)
    st_e = (build_static_matrix(se_, ee, SHAPELET_STATIC_COLS) - mu) / sd
    em = load_eicu("motif_atoms")
    a_e = build_concept_matrix(em, ee, concepts, kind_of=kind_of, use_activation=True)
    ind_e = build_concept_matrix(em, ee, concepts, kind_of=kind_of, use_activation=False)
    p0 = c0.predict_proba(st_e)[:, 1]
    p1 = c1.predict_proba(np.column_stack([st_e, a_e]))[:, 1]
    # prevalence denominators follow scripts/48: stays with >=1 atom of any kind
    has_atom = set(load_eicu("atoms")["ICUSTAY_ID"].astype(int).unique())
    atom_mask = np.array([s in has_atom for s in ee])

    base = load_eicu("icu_base")[["ICUSTAY_ID", "hospitalid"]].drop_duplicates("ICUSTAY_ID").set_index("ICUSTAY_ID")
    hosp = pd.read_csv(eicu_table("hospital")).set_index("hospitalid")
    hid = base.loc[ee, "hospitalid"].to_numpy()
    meta = hosp.reindex(hid)
    strata = {"full": np.ones(len(ee), bool)}
    for col, lab in [("region", "region"), ("teachingstatus", "teaching"), ("numbedscategory", "beds")]:
        v = meta[col].astype(object).where(meta[col].notna(), "unknown").astype(str).to_numpy()
        for u in sorted(set(v)):
            strata[f"{lab}={u}"] = v == u
    print(f"[data] eICU {len(ee)} stays, {len(set(hid))} hospitals; strata {len(strata)} | {time.time()-t0:.0f}s", flush=True)

    rows = []
    for name, m in strata.items():
        y, I = y_e[m], ind_e[m]
        mm = m & atom_mask
        prev = ind_e[mm].sum(0) / max(mm.sum(), 1)
        rho_l1 = float(spearmanr(prev_hold, prev).correlation)
        diff, lo, hi, n1 = node_assoc(y, I)
        ok = np.isfinite(diff)
        same = (np.sign(diff) == np.sign(m_diff)) & ok
        sig = (lo * hi > 0) & ok
        row = {"stratum": name, "n": int(m.sum()), "deaths": int(y.sum()), "mortality": float(y.mean()),
               "n_hospitals": int(len(set(hid[m]))),
               "L1_jsd_vs_holdout": jsd(prev_hold, prev), "L1_spearman": rho_l1,
               "L2_same_sign_84": int(same.sum()), "L2_bonf34_same_sign": int(same[m_bonf].sum()),
               "L2_bonf34_replicated": int((same & sig)[m_bonf].sum()),
               "L2_spearman_diff": float(spearmanr(m_diff[ok], diff[ok]).correlation)}
        if y.sum() >= 10:
            ap0, ap1 = average_precision_score(y, p0[m]), average_precision_score(y, p1[m])
            d = boot_delta(y, p1[m], p0[m], rng, 500, average_precision_score)
            row.update({"L3_auprc_static": float(ap0), "L3_auprc_84a": float(ap1),
                        "L3_dauprc": float(ap1 - ap0), "L3_dauprc_lo": float(np.percentile(d, 2.5)),
                        "L3_dauprc_hi": float(np.percentile(d, 97.5)),
                        "L3_auroc_static": float(roc_auc_score(y, p0[m])), "L3_auroc_84a": float(roc_auc_score(y, p1[m])),
                        "retention_84a": float(ap1 / val_ap["static+84a"])})
        rows.append(row)
        print(f"[stratum] {name:<22} n={row['n']:>6} mort={row['mortality']:.3f} JSD={row['L1_jsd_vs_holdout']:.3f} "
              f"rho={rho_l1:.2f} L2 {row['L2_same_sign_84']}/84 bonf {row['L2_bonf34_replicated']}/34 "
              f"AUPRC {row.get('L3_auprc_static', float('nan')):.3f}->{row.get('L3_auprc_84a', float('nan')):.3f} "
              f"| {time.time()-t0:.0f}s", flush=True)
    st_df = pd.DataFrame(rows)
    st_df.to_csv(OUT_DIR / "strata_table.csv", index=False, encoding="utf-8-sig")

    # ---- hospital level ----
    hrows = []
    for h in sorted(set(hid)):
        m = hid == h
        y = y_e[m]
        if m.sum() < 100 or y.sum() < 15 or y.sum() == m.sum():
            continue
        r0, r1 = roc_auc_score(y, p0[m]), roc_auc_score(y, p1[m])
        d = boot_delta(y, p1[m], p0[m], rng, 200, roc_auc_score)
        hrows.append({"hospitalid": int(h), "n": int(m.sum()), "deaths": int(y.sum()),
                      "region": str(hosp.loc[h, "region"]) if h in hosp.index else "unknown",
                      "auroc_static": float(r0), "auroc_84a": float(r1), "dauroc": float(r1 - r0),
                      "se": float(d.std(ddof=1))})
    hd = pd.DataFrame(hrows)
    hd.to_csv(OUT_DIR / "hospital_table.csv", index=False, encoding="utf-8-sig")
    w = 1 / hd["se"] ** 2
    fe = float((w * hd["dauroc"]).sum() / w.sum())
    q = float((w * (hd["dauroc"] - fe) ** 2).sum())
    k = len(hd)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (hd["se"] ** 2 + tau2)
    re = float((wr * hd["dauroc"]).sum() / wr.sum())
    re_se = float(np.sqrt(1 / wr.sum()))
    hosp_summary = {"k_hospitals": k, "stays_covered": int(hd["n"].sum()),
                    "dauroc_median": float(hd["dauroc"].median()),
                    "dauroc_iqr": [float(hd["dauroc"].quantile(.25)), float(hd["dauroc"].quantile(.75))],
                    "share_positive": float((hd["dauroc"] > 0).mean()),
                    "re_pooled_dauroc": re, "re_ci95": [re - 1.96 * re_se, re + 1.96 * re_se],
                    "tau2": float(tau2), "Q": q, "I2": float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0,
                    "prediction_interval95": [re - 1.96 * np.sqrt(tau2 + re_se ** 2), re + 1.96 * np.sqrt(tau2 + re_se ** 2)]}
    res = {"round": "round-3", "post_hoc": True, "seed": SEED, "mimic_val_auprc": val_ap,
           "reproduction_full": {k2: st_df.iloc[0][k2] for k2 in
                                 ["L1_jsd_vs_holdout", "L1_spearman", "L2_same_sign_84", "L2_bonf34_replicated",
                                  "L3_auprc_static", "L3_auprc_84a"]},
           "expected_full": {"L1_jsd_vs_holdout": 0.1318, "L1_spearman": 0.8917, "L2_same_sign_84": 62,
                             "L2_bonf34_replicated": 30, "L3_auprc_static": 0.190, "L3_auprc_84a": 0.281},
           "strata": rows, "hospital_level": hosp_summary}
    (OUT_DIR / "region_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"repro": res["reproduction_full"], "hospital": hosp_summary}, indent=1, default=float))
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
