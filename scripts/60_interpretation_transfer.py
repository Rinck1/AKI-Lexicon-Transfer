#!/usr/bin/env python3
"""Round-3 (2026-10-23): do the *explanations* of common model families transfer across
databases as well as the 84-concept lexicon does?

Motivation (advisor request): compare with explanation methods used in related AKI /
ICU-mortality papers. Those papers report one explanation (SHAP bar plot, regression
coefficients) from one database. Nobody asks whether that explanation is *the same
explanation* in a second database. We apply the same test the lexicon passed in L2:
fit the explanation separately in each site and compare.

Design (written BEFORE the first run; post-hoc / exploratory, not part of docs/31):
  Feature sets (both MIMIC-train-frozen scalers; no eICU information is used):
      S1 = static 11 + numeric 176   (the feature family of typical published models)
      S2 = static 11 + lexicon 84a   (the lexicon family)
  Explanation methods (importance, direction):
      LR   : |standardised coefficient|, sign(coefficient)            (L2-regularised LR)
      XGB  : mean |TreeSHAP|, sign(Spearman(x, SHAP))                  (frozen scripts/53 hyper-params)
      UNI  : |standardised mean difference dead-vs-alive|, its sign   (univariate; the L2 analogue)
  Protocol (sample size is equalised, otherwise cross-site vs within-site is unfair):
      repeat R = 30: split MIMIC train into two SUBJECT-disjoint halves A, B (n ~ 3,805);
      draw a random eICU sample E of the same n. Fit the explanation on A, B and E.
      cross-site  = metric(A, E)       within-site baseline = metric(A, B)
      A positive gap (baseline - cross-site) means the explanation is less reproducible
      across databases than within one.
  Metrics: Spearman rho of importances over features; top-10 / top-20 Jaccard (top-k by A);
           sign agreement among A's top-10 features; rho over variable-level groups.
  Also: full-sample point estimate (all MIMIC train vs all eICU).
Outputs: data/external/eicu/round3_interpretation/interpretation_transfer.json,
         top_features_fullsample.csv
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.cbm import KIND_MOTIF, build_concept_matrix, build_static_matrix  # noqa: E402
from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.eicu_constants import MIMIC_PATHS_DIR  # noqa: E402
from scripts.utils.eicu_io import load_eicu  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402

_spec = importlib.util.spec_from_file_location("xgb53", PROJ / "scripts" / "53_xgb_zero_shot.py")
x53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x53)

OUT_DIR = PROJ / "data/external/eicu/round3_interpretation"
SEED = 42
R = 30
TOPK = (10, 20)


# ---------------------------------------------------------------- explanation methods
def expl_uni(X, y):
    d = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        a, b = X[y == 1, j], X[y == 0, j]
        sd = np.sqrt((a.var() + b.var()) / 2)
        d[j] = 0.0 if sd < 1e-9 else (a.mean() - b.mean()) / sd
    return np.abs(d), np.sign(d)


def expl_lr(X, y):
    clf = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X, y)
    c = clf.coef_[0]
    return np.abs(c), np.sign(c)


def make_expl_xgb(hp):
    import xgboost
    from xgboost import XGBClassifier

    def f(X, y):
        clf = XGBClassifier(**hp, **{**x53.XGB_FIXED, "n_jobs": 4}).fit(X, y)
        sv = clf.get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)[:, :-1]
        imp = np.abs(sv).mean(0)
        dirn = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            ok = ~np.isnan(X[:, j])
            if ok.sum() < 30 or np.std(X[ok, j]) < 1e-9 or np.std(sv[ok, j]) < 1e-12:
                continue
            dirn[j] = np.sign(spearmanr(X[ok, j], sv[ok, j])[0])
        return imp, dirn
    return f


# ---------------------------------------------------------------------- metrics
def group_sum(imp, groups):
    g = pd.Series(imp).groupby(np.asarray(groups)).sum()
    return g


def compare(a, b, groups):
    """a, b = (importance, direction) from two fits."""
    ia, da = a
    ib, db = b
    rho = float(spearmanr(ia, ib)[0])
    out = {"rho": rho}
    order_a, order_b = np.argsort(-ia), np.argsort(-ib)
    for k in TOPK:
        sa, sb = set(order_a[:k].tolist()), set(order_b[:k].tolist())
        out[f"jacc_top{k}"] = len(sa & sb) / len(sa | sb)
    top = order_a[:10]
    out["sign_agree_top10"] = float(np.mean(da[top] == db[top]))
    ga, gb = group_sum(ia, groups), group_sum(ib, groups)
    out["rho_group"] = float(spearmanr(ga.values, gb.reindex(ga.index).values)[0])
    return out


def summarise(rows):
    keys = rows[0].keys()
    return {k: {"mean": float(np.nanmean([r[k] for r in rows])), "sd": float(np.nanstd([r[k] for r in rows]))} for k in keys}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    grid_m, mimic_stays = load_shapelet_tables(DATA_PROCESSED)
    sm = mimic_stays.set_index("ICUSTAY_ID")
    tr = np.array(sorted(int(x) for x in mimic_stays.loc[mimic_stays["split"] == "train", "ICUSTAY_ID"]))
    y_tr = sm.loc[list(tr), "HOSPITAL_MORTALITY"].to_numpy(int)
    subj = sm.loc[list(tr), "SUBJECT_ID"].to_numpy()
    grid_e = pd.read_parquet(PROJ / "data/external/eicu/hourly_grid_48h.parquet")
    se = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    ee = np.array(sorted(int(x) for x in se.index))
    y_e = se.loc[list(ee), "HOSPITAL_MORTALITY"].to_numpy(int)

    num_tr, num_e = x53.numeric_pattern_matrix(grid_m, tr), x53.numeric_pattern_matrix(grid_e, ee)
    st_tr = build_static_matrix(sm, list(tr), SHAPELET_STATIC_COLS)
    st_e = build_static_matrix(se, list(ee), SHAPELET_STATIC_COLS)
    m_atoms = pd.read_parquet(MIMIC_PATHS_DIR / "motif_atoms.parquet")
    mt_tr = m_atoms[m_atoms["split"] == "train"]
    concepts = sorted(mt_tr["atom"].astype(str).unique())
    assert len(concepts) == 84
    kind_of = {c: KIND_MOTIF for c in concepts}
    a_tr = build_concept_matrix(mt_tr, list(tr), concepts, kind_of=kind_of, use_activation=True)
    a_e = build_concept_matrix(load_eicu("motif_atoms"), list(ee), concepts, kind_of=kind_of, use_activation=True)
    print(f"[data] MIMIC train {len(tr)} eICU {len(ee)} | {time.time()-t0:.0f}s", flush=True)

    # ---- feature sets, frozen MIMIC-train scaling (identical for both sites) ----
    raw_tr, raw_e = np.column_stack([st_tr, num_tr]), np.column_stack([st_e, num_e])
    # XGB uses raw (NaN native) ; LR/UNI use median-filled z-scores from MIMIC train
    z_tr, mu, sd, med = x53.zscore_impute_train(raw_tr, raw_tr)
    z_e = x53.apply_zscore(raw_e, mu, sd, med)
    smu, ssd = st_tr.mean(0), np.where(st_tr.std(0) < 1e-6, 1.0, st_tr.std(0))
    s2_tr = np.column_stack([(st_tr - smu) / ssd, a_tr])
    s2_e = np.column_stack([(st_e - smu) / ssd, a_e])
    s2raw_tr, s2raw_e = np.column_stack([st_tr, a_tr]), np.column_stack([st_e, a_e])

    var_s1 = list(SHAPELET_STATIC_COLS) + [n.split("|")[0] for n in x53.NUM_NAMES]
    var_s2 = list(SHAPELET_STATIC_COLS) + [c.split("|")[0] for c in concepts]
    names_s1 = list(SHAPELET_STATIC_COLS) + list(x53.NUM_NAMES)
    names_s2 = list(SHAPELET_STATIC_COLS) + list(concepts)
    sets = {
        "S1_static+num176": dict(groups=var_s1, names=names_s1, std=(z_tr, z_e), raw=(raw_tr, raw_e)),
        "S2_static+lexicon84": dict(groups=var_s2, names=names_s2, std=(s2_tr, s2_e), raw=(s2raw_tr, s2raw_e)),
    }

    best = json.loads((PROJ / "data/external/eicu/round2_xgb/xgb_tuning_results.json").read_text(encoding="utf-8"))["best"]
    hp = dict(max_depth=best["max_depth"], learning_rate=best["learning_rate"], n_estimators=best["n_estimators_final"])
    methods = {"UNI": ("std", expl_uni), "LR": ("std", expl_lr), "XGB": ("raw", make_expl_xgb(hp))}

    # ---- full-sample point estimates ----
    full, topcsv = {}, []
    for sname, S in sets.items():
        for mname, (kind, fn) in methods.items():
            Xm, Xe = S[kind]
            em, ee_ = fn(Xm, y_tr), fn(Xe, y_e)
            full[f"{sname}|{mname}"] = compare(em, ee_, S["groups"])
            for site, e in (("MIMIC", em), ("eICU", ee_)):
                for j in np.argsort(-e[0])[:10]:
                    topcsv.append({"set": sname, "method": mname, "site": site, "feature": S["names"][j],
                                   "importance": float(e[0][j]), "direction": int(e[1][j])})
            print(f"[full] {sname:<22} {mname:<4} rho={full[f'{sname}|{mname}']['rho']:.2f} "
                  f"sign@10={full[f'{sname}|{mname}']['sign_agree_top10']:.2f} | {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(topcsv).to_csv(OUT_DIR / "top_features_fullsample.csv", index=False)

    # ---- repeated equal-n protocol ----
    uniq = np.unique(subj)
    cross = {k: [] for k in full}
    within = {k: [] for k in full}
    for r in range(R):
        perm = rng.permutation(uniq)
        half = set(perm[: len(perm) // 2].tolist())
        ia = np.array([s in half for s in subj])
        ib = ~ia
        n = int(min(ia.sum(), ib.sum()))
        a_idx = rng.permutation(np.where(ia)[0])[:n]
        b_idx = rng.permutation(np.where(ib)[0])[:n]
        e_idx = rng.permutation(len(ee))[:n]
        for sname, S in sets.items():
            for mname, (kind, fn) in methods.items():
                Xm, Xe = S[kind]
                ea, eb, ex = fn(Xm[a_idx], y_tr[a_idx]), fn(Xm[b_idx], y_tr[b_idx]), fn(Xe[e_idx], y_e[e_idx])
                cross[f"{sname}|{mname}"].append(compare(ea, ex, S["groups"]))
                within[f"{sname}|{mname}"].append(compare(ea, eb, S["groups"]))
        print(f"[rep {r+1}/{R}] {time.time()-t0:.0f}s", flush=True)

    res = {"round": "round-3", "post_hoc": True, "seed": SEED, "R": R, "n_per_fit": n,
           "design": "see script docstring (written before first run)", "full_sample": full, "repeated": {}}
    for k in full:
        c, w = summarise(cross[k]), summarise(within[k])
        res["repeated"][k] = {"cross_site": c, "within_site": w,
                              "gap_rho": w["rho"]["mean"] - c["rho"]["mean"],
                              "gap_sign_top10": w["sign_agree_top10"]["mean"] - c["sign_agree_top10"]["mean"]}
        print(f"[res] {k:<28} rho cross {c['rho']['mean']:.2f} vs within {w['rho']['mean']:.2f} | "
              f"sign@10 {c['sign_agree_top10']['mean']:.2f} vs {w['sign_agree_top10']['mean']:.2f} | "
              f"top10 jacc {c['jacc_top10']['mean']:.2f} vs {w['jacc_top10']['mean']:.2f}", flush=True)
    (OUT_DIR / "interpretation_transfer.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
