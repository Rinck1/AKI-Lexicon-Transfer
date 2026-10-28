#!/usr/bin/env python3
"""Round-3 (2026-10-11): sensitivity of the main findings to the HA-AKI definition
and to the KDIGO "SCr >= 4.0 mg/dL" stage-3 shortcut.

Why: the locked cohort labels AKI as hospital-acquired when KDIGO onset occurs
after ICU admission (CA_MAX_HOURS = 0, relative to INTIME). The usual literature
definition is onset >= 48 h after HOSPITAL admission. In addition,
kdigo.stage_from_ratio_and_abs assigns stage 3 whenever SCr >= 4.0, whereas KDIGO
requires SCr >= 4.0 together with an acute rise.

Design (written before the first run; descriptive, no gate; frozen assets untouched):
  S0 = locked cohort (reproduction row).
  S1 = strict HA: onset >= 48 h after hospital admission (ADMITTIME).
  S2 = S0 minus stays whose creatinine onset was reached ONLY through the
       SCr >= 4.0 shortcut (ratio to 7-day prior-minimum < 1.5 and rise vs 48-h
       prior-minimum < 0.3 mg/dL at onset).
  S3 = S1 and S2.
  For each subset, on eICU (zero-shot):
    lexicon L1 (JSD/rho vs MIMIC holdout restricted to the same definition),
    L2 (84-node sign agreement and 34-Bonferroni replication vs the fixed MIMIC
        train reference), L3 (static vs static+84a, models fitted on the FULL MIMIC
        train = deployment as locked; plus a refit on the MIMIC-train subset meeting
        the same definition), and the 15 frozen paths under the protocol
        same-sign definition (sign of eICU crude death-rate difference vs sign of
        the MIMIC-train coefficient; full-cohort check = 11/15).
Outputs: data/external/eicu/round3_ha_sensitivity/ha_sensitivity_results.json,
         subset_table.csv
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.cbm import KIND_MOTIF, build_concept_matrix, build_static_matrix, standardize_train_only  # noqa: E402
from scripts.utils.constants import (  # noqa: E402
    ABS_INCREASE_WINDOW_HOURS, BASELINE_LOOKBACK_HOURS, DATA_PROCESSED, DATA_RAW, KDIGO_ABS_INCREASE, KDIGO_CR_MAX, KDIGO_RATIO_1)
from scripts.utils.eicu_constants import MIMIC_PATHS_DIR  # noqa: E402
from scripts.utils.eicu_io import load_eicu  # noqa: E402
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.path_protocol import parse_path_key  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402

_spec = importlib.util.spec_from_file_location("pz", PROJ / "scripts" / "_explore_PRED_cbm_zeroshot.py")
pz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pz)

OUT_DIR = PROJ / "data/external/eicu/round3_ha_sensitivity"
EICU_DIR = PROJ / "data/external/eicu"
SEED = 42
STRICT_HA_HOURS = 48.0


def hours_from_admit(primary: pd.DataFrame, base: pd.DataFrame) -> pd.Series:
    b = base.drop_duplicates("ICUSTAY_ID").set_index("ICUSTAY_ID")
    adm = pd.to_datetime(b.loc[primary["ICUSTAY_ID"], "ADMITTIME"].to_numpy())
    h = (pd.to_datetime(primary["onset_time"]).to_numpy() - adm) / pd.Timedelta(hours=1)
    return pd.Series(np.asarray(h, float), index=primary["ICUSTAY_ID"].to_numpy())


def scr4_only_flags(primary: pd.DataFrame, labs: pd.DataFrame) -> pd.Series:
    """True if creatinine onset was reached only via SCr >= 4.0 (no ratio/abs criterion)."""
    cr = labs[labs["variable"] == "creatinine"][["ICUSTAY_ID", "charttime", "value"]].dropna()
    cr = cr[cr["ICUSTAY_ID"].isin(set(primary["ICUSTAY_ID"]))].sort_values(["ICUSTAY_ID", "charttime"])
    groups = {k: g for k, g in cr.groupby("ICUSTAY_ID")}
    out = {}
    for r in primary.itertuples(index=False):
        flag = False
        if str(r.onset_by).startswith("creatinine") and r.ICUSTAY_ID in groups:
            g = groups[r.ICUSTAY_ID]
            t = pd.to_datetime(g["charttime"]).to_numpy()
            v = g["value"].to_numpy(float)
            t0 = np.datetime64(pd.Timestamp(r.onset_time))
            at = np.where(t == t0)[0]
            if len(at):
                scr = float(np.max(v[at]))
                prior7 = v[(t >= t0 - np.timedelta64(int(BASELINE_LOOKBACK_HOURS * 60), "m")) & (t < t0)]
                prior48 = v[(t >= t0 - np.timedelta64(int(ABS_INCREASE_WINDOW_HOURS * 60), "m")) & (t < t0)]
                ratio = scr / prior7.min() if prior7.size and prior7.min() > 0 else np.nan
                absinc = scr - prior48.min() if prior48.size else np.nan
                ratio_ok = np.isfinite(ratio) and ratio >= KDIGO_RATIO_1
                abs_ok = np.isfinite(absinc) and absinc >= KDIGO_ABS_INCREASE
                flag = bool(scr >= KDIGO_CR_MAX and not ratio_ok and not abs_ok)
        out[int(r.ICUSTAY_ID)] = flag
    return pd.Series(out)


def jsd(p, q):
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2.0))


def node_assoc(y, ind):
    n1 = ind.sum(0)
    n0 = len(y) - n1
    d1 = (ind * y[:, None]).sum(0)
    with np.errstate(all="ignore"):
        r1, r0 = d1 / n1, (y.sum() - d1) / n0
        se = np.sqrt(r1 * (1 - r1) / n1 + r0 * (1 - r0) / n0)
    d = r1 - r0
    return d, d - 1.96 * se, d + 1.96 * se


def boot_delta(y, p1, p0, rng, b=500):
    n = len(y)
    out = []
    for _ in range(b):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, n):
            continue
        out.append(average_precision_score(y[i], p1[i]) - average_precision_score(y[i], p0[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---------- cohorts + definition flags ----------
    pm = pd.read_parquet(Path(DATA_PROCESSED) / "aki_primary_ha48.parquet")
    pe = pd.read_parquet(EICU_DIR / "aki_primary_ha48.parquet")
    h_m = hours_from_admit(pm, pd.read_parquet(Path(DATA_RAW) / "icu_base.parquet"))
    h_e = hours_from_admit(pe, pd.read_parquet(EICU_DIR / "icu_base.parquet"))
    s4_m = scr4_only_flags(pm, pd.read_parquet(Path(DATA_RAW) / "meas_labs.parquet"))
    print(f"[flags] MIMIC scr4-only {int(s4_m.sum())} | {time.time()-t0:.0f}s", flush=True)
    s4_e = scr4_only_flags(pe, pd.read_parquet(EICU_DIR / "meas_labs.parquet"))
    print(f"[flags] eICU scr4-only {int(s4_e.sum())} | {time.time()-t0:.0f}s", flush=True)
    timing = {}
    for side, h, s4, p in [("MIMIC", h_m, s4_m, pm), ("eICU", h_e, s4_e, pe)]:
        timing[side] = {"n": int(len(h)), "onset_from_admit_h_quartiles": [float(x) for x in h.quantile([.25, .5, .75])],
                        "share_onset_ge48h_after_admit": float((h >= STRICT_HA_HOURS).mean()),
                        "share_onset_lt24h_after_admit": float((h < 24).mean()),
                        "onset_from_icu_h_median": float(p["onset_hour_from_icu"].median()),
                        "n_scr4_only": int(s4.sum()), "share_scr4_only": float(s4.mean()),
                        "mortality_strict": float(p.set_index("ICUSTAY_ID").loc[h.index[h >= STRICT_HA_HOURS], "HOSPITAL_MORTALITY"].mean()),
                        "mortality_nonstrict": float(p.set_index("ICUSTAY_ID").loc[h.index[h < STRICT_HA_HOURS], "HOSPITAL_MORTALITY"].mean())}

    def subset_ids(h, s4, ids, which):
        ids = np.asarray(ids)
        strict = (h.reindex(ids) >= STRICT_HA_HOURS).to_numpy()
        clean = ~s4.reindex(ids).fillna(False).to_numpy(bool)
        return {"S0_locked": np.ones(len(ids), bool), "S1_strict_HA48": strict,
                "S2_no_scr4_shortcut": clean, "S3_strict_and_no_scr4": strict & clean}[which]

    # ---------- MIMIC side ----------
    _g, ms = load_shapelet_tables(DATA_PROCESSED)
    sm = ms.set_index("ICUSTAY_ID")
    tr = np.array(sorted(int(x) for x in ms.loc[ms["split"] == "train", "ICUSTAY_ID"]))
    y_tr = sm.loc[list(tr), "HOSPITAL_MORTALITY"].to_numpy(float)
    m_atoms = pd.read_parquet(MIMIC_PATHS_DIR / "motif_atoms.parquet")
    concepts = sorted(m_atoms.loc[m_atoms["split"] == "train", "atom"].astype(str).unique())
    kind_of = {c: KIND_MOTIF for c in concepts}
    st_tr_raw = build_static_matrix(sm, list(tr), SHAPELET_STATIC_COLS)
    a_tr = build_concept_matrix(m_atoms[m_atoms["split"] == "train"], list(tr), concepts, kind_of=kind_of, use_activation=True)
    hold = pd.read_parquet(MIMIC_PATHS_DIR / "holdout_atoms.parquet")
    ref = pd.read_csv(EICU_DIR / "_explore_NODE/node_assoc_transfer_84.csv", encoding="utf-8-sig").set_index("node")
    m_diff = ref.loc[concepts, "mimic_diff"].to_numpy(float)
    m_bonf = ref.loc[concepts, "mimic_bonf"].to_numpy(bool)

    # path coefficients (MIMIC train direction) from the round-2 erratum file
    err = json.loads((EICU_DIR / "round2_erratum/corrected_same_sign.json").read_text(encoding="utf-8"))
    sel = json.loads((MIMIC_PATHS_DIR / "path_set_selected.json").read_text(encoding="utf-8"))
    paths = [parse_path_key(p) for p in sel["selected"]]
    coef = {r["path"].replace(" ", ""): r["mimic_train_coef"] for r in err["paths"]}
    train_sign = np.array([np.sign(coef[" → ".join(p).replace(" ", "")]) for p in paths])

    # ---------- eICU side ----------
    se_ = load_eicu("stay_features_48h").set_index("ICUSTAY_ID")
    ee = np.array(sorted(int(x) for x in se_.index))
    y_e = se_.loc[list(ee), "HOSPITAL_MORTALITY"].to_numpy(float)
    em = load_eicu("motif_atoms")
    a_e = build_concept_matrix(em, list(ee), concepts, kind_of=kind_of, use_activation=True)
    ind_e = build_concept_matrix(em, list(ee), concepts, kind_of=kind_of, use_activation=False)
    has_atom = set(load_eicu("atoms")["ICUSTAY_ID"].astype(int).unique())
    atom_mask = np.array([s in has_atom for s in ee])
    path_e = pz.fast_indicator(list(ee), atoms_by_stay(load_eicu("atoms")), paths)
    st_e_raw = build_static_matrix(se_, list(ee), SHAPELET_STATIC_COLS)

    # locked deployment models (full MIMIC train)
    st_tr, mu, sd = standardize_train_only(st_tr_raw)
    c0 = LogisticRegression(max_iter=500, solver="lbfgs").fit(st_tr, y_tr)
    c1 = LogisticRegression(max_iter=500, solver="lbfgs").fit(np.column_stack([st_tr, a_tr]), y_tr)
    st_e = (st_e_raw - mu) / sd
    p0_e, p1_e = c0.predict_proba(st_e)[:, 1], c1.predict_proba(np.column_stack([st_e, a_e]))[:, 1]

    rows = []
    for which in ["S0_locked", "S1_strict_HA48", "S2_no_scr4_shortcut", "S3_strict_and_no_scr4"]:
        me = subset_ids(h_e, s4_e, ee, which)
        mt = subset_ids(h_m, s4_m, tr, which)
        hold_ids = np.array(sorted(hold["ICUSTAY_ID"].astype(int).unique()))
        mh = subset_ids(h_m, s4_m, hold_ids, which)
        keep_h = set(hold_ids[mh])
        hm = hold[(hold["kind"] == "motif") & hold["ICUSTAY_ID"].isin(keep_h)]
        prev_h = np.array([hm.loc[hm["atom"] == c, "ICUSTAY_ID"].nunique() / len(keep_h) for c in concepts])
        mm = me & atom_mask
        prev_e = ind_e[mm].sum(0) / mm.sum()
        y = y_e[me]
        d, lo, hi = node_assoc(y, ind_e[me])
        same = np.sign(d) == np.sign(m_diff)
        sig = lo * hi > 0
        # paths: protocol same-sign
        pd_, _, _ = node_assoc(y, path_e[me])
        psame = np.sign(pd_) == train_sign
        k_same = int(psame.sum())
        # refit on the MIMIC-train subset meeting the same definition
        s_mu, s_sd = st_tr_raw[mt].mean(0), np.maximum(st_tr_raw[mt].std(0), 1e-6)
        r0 = LogisticRegression(max_iter=500, solver="lbfgs").fit((st_tr_raw[mt] - s_mu) / s_sd, y_tr[mt])
        r1 = LogisticRegression(max_iter=500, solver="lbfgs").fit(np.column_stack([(st_tr_raw[mt] - s_mu) / s_sd, a_tr[mt]]), y_tr[mt])
        se_s = (st_e_raw[me] - s_mu) / s_sd
        q0, q1 = r0.predict_proba(se_s)[:, 1], r1.predict_proba(np.column_stack([se_s, a_e[me]]))[:, 1]
        row = {"subset": which, "eicu_n": int(me.sum()), "eicu_mortality": float(y.mean()),
               "mimic_train_n": int(mt.sum()), "mimic_holdout_n": int(len(keep_h)),
               "L1_jsd": jsd(prev_h, prev_e), "L1_spearman": float(spearmanr(prev_h, prev_e).correlation),
               "L2_same_sign_84": int(same.sum()), "L2_bonf34_replicated": int((same & sig)[m_bonf].sum()),
               "L3_auprc_static": float(average_precision_score(y, p0_e[me])),
               "L3_auprc_84a": float(average_precision_score(y, p1_e[me])),
               "L3_dauprc_ci95": boot_delta(y, p1_e[me], p0_e[me], rng),
               "L3_auroc_static": float(roc_auc_score(y, p0_e[me])), "L3_auroc_84a": float(roc_auc_score(y, p1_e[me])),
               "L3_refit_auprc_static": float(average_precision_score(y, q0)),
               "L3_refit_auprc_84a": float(average_precision_score(y, q1)),
               "paths_protocol_same_sign": k_same,
               "paths_sign_test_p": float(binomtest(k_same, 15, 0.5).pvalue),
               "paths_support_ge20": int((path_e[me].sum(0) >= 20).sum())}
        rows.append(row)
        print(f"[{which}] eICU n={row['eicu_n']} mort={row['eicu_mortality']:.3f} JSD={row['L1_jsd']:.3f} rho={row['L1_spearman']:.2f} "
              f"L2 {row['L2_same_sign_84']}/84 bonf {row['L2_bonf34_replicated']}/34 AUPRC {row['L3_auprc_static']:.3f}->{row['L3_auprc_84a']:.3f} "
              f"(refit {row['L3_refit_auprc_static']:.3f}->{row['L3_refit_auprc_84a']:.3f}) paths {k_same}/15 | {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "subset_table.csv", index=False, encoding="utf-8-sig")
    res = {"round": "round-3", "post_hoc": True, "seed": SEED, "strict_ha_hours": STRICT_HA_HOURS,
           "timing": timing, "subsets": rows,
           "reproduction_S0_expected": {"L1_jsd": 0.1318, "L2_same_sign_84": 62, "L2_bonf34_replicated": 30,
                                        "L3_auprc_static": 0.190, "L3_auprc_84a": 0.281, "paths_protocol_same_sign": 11}}
    (OUT_DIR / "ha_sensitivity_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(timing, indent=1))
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
