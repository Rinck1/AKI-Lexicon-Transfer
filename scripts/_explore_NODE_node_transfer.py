#!/usr/bin/env python3
"""NODE 探索（post-hoc）：84 motif 节点 / 87 检验原子的 结局关联 与 零样本节点袋预测
是否从 MIMIC 迁移到 eICU。

- 只读现有文件；产物仅写 data/external/eicu/_explore_NODE/
- MIMIC 侧只用 split=='train'
- 全部 eICU 分析为 post-hoc 探索性
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.eicu_io import load_eicu  # noqa: E402
from scripts.utils.path_protocol import PATH_SEED  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402

PATHS_DIR = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"
OUT_DIR = PROJ / "data/external/eicu/_explore_NODE"
N_BOOT = 400
SEED = PATH_SEED
MIN_SUPPORT = 20


def boot_ci_diff(y: np.ndarray, ind: np.ndarray, rng: np.random.Generator, b: int = N_BOOT):
    """Same spec as scripts/32: full-sample resample, diff of means, 2.5/97.5 pct."""
    n = len(y)
    stats = np.empty(b, dtype=np.float64)
    for k in range(b):
        ix = rng.integers(0, n, n)
        yy, ii = y[ix], ind[ix].astype(bool)
        if ii.any() and (~ii).any():
            stats[k] = yy[ii].mean() - yy[~ii].mean()
        else:
            stats[k] = np.nan
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def indicator_matrix(stay_ids: list[int], atoms: pd.DataFrame, names: list[str]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(stay_ids)}
    j_of = {n: j for j, n in enumerate(names)}
    ind = np.zeros((len(stay_ids), len(names)), dtype=np.float64)
    for rec in atoms.itertuples(index=False):
        i = idx.get(int(rec.ICUSTAY_ID))
        if i is None:
            continue
        j = j_of.get(str(rec.atom))
        if j is not None:
            ind[i, j] = 1.0
    return ind


def crude_assoc_table(y, ind, names, rng, cohort_label) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(names):
        ii = ind[:, j] > 0.5
        n1 = int(ii.sum())
        n0 = int((~ii).sum())
        if n1 < 5 or n0 < 5:
            rows.append({"atom": name, "n_on": n1, "n_off": n0, "rate_on": np.nan,
                         "rate_off": np.nan, "diff": np.nan, "lo": np.nan, "hi": np.nan,
                         "ci_excludes_0": False, "support_ok": False})
            continue
        r1, r0 = float(y[ii].mean()), float(y[~ii].mean())
        lo, hi = boot_ci_diff(y, ii.astype(np.float64), rng)
        rows.append({
            "atom": name, "n_on": n1, "n_off": n0, "rate_on": r1, "rate_off": r0,
            "diff": r1 - r0, "lo": lo, "hi": hi,
            "ci_excludes_0": bool(np.isfinite(lo) and lo * hi > 0),
            "support_ok": bool(MIN_SUPPORT <= n1 <= len(y) - MIN_SUPPORT),
            "cohort": cohort_label,
        })
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---------------- MIMIC train ----------------
    _, mimic_stays = load_shapelet_tables(DATA_PROCESSED)
    tr = mimic_stays[mimic_stays["split"] == "train"]
    train_ids = sorted(int(x) for x in tr["ICUSTAY_ID"])
    y_tr = tr.set_index("ICUSTAY_ID").loc[train_ids, "HOSPITAL_MORTALITY"].to_numpy(float)
    S_tr = tr.set_index("ICUSTAY_ID").loc[train_ids, list(SHAPELET_STATIC_COLS)].to_numpy(float)
    mu, sd = S_tr.mean(0), np.maximum(S_tr.std(0), 1e-6)
    print(f"[mimic] train n={len(train_ids)} mort={y_tr.mean():.4f}")

    m_motif = pd.read_parquet(PATHS_DIR / "motif_atoms.parquet")
    m_lab = pd.read_parquet(PATHS_DIR / "lab_atoms.parquet")
    node_names = sorted(m_motif.loc[m_motif["split"] == "train", "atom"].astype(str).unique())
    lab_names = sorted(m_lab.loc[m_lab["split"] == "train", "atom"].astype(str).unique())
    print(f"[atoms] motif={len(node_names)} lab={len(lab_names)}")

    ind_node_tr = indicator_matrix(train_ids, m_motif[m_motif["split"] == "train"], node_names)
    ind_lab_tr = indicator_matrix(train_ids, m_lab[m_lab["split"] == "train"], lab_names)

    # MIMIC motif assoc: reuse stored file (computed by script 32 on train)
    mimic_node = pd.read_csv(PATHS_DIR / "node_outcome_assoc_train.csv").set_index("node")
    assert len(mimic_node) == 84
    # Bonferroni subset re-derivation (matches node_bag_val_eval.json: 34)
    se_b = (mimic_node["diff_hi"] - mimic_node["diff_lo"]) / 3.92
    z_b = norm.ppf(1 - 0.025 / 84)
    mimic_node["bonferroni"] = mimic_node["death_rate_diff"].abs() > z_b * se_b
    assert int(mimic_node["bonferroni"].sum()) == 34, "Bonferroni subset mismatch"
    assert int(mimic_node["ci_excludes_0"].sum()) == 52

    # MIMIC lab assoc (train-only, same crude spec)
    mimic_lab = crude_assoc_table(y_tr, ind_lab_tr, lab_names, rng, "MIMIC train").set_index("atom")
    print(f"[mimic] lab atoms ci_excl0: {int(mimic_lab['ci_excludes_0'].sum())}/{len(mimic_lab)}")

    # ---------------- eICU ----------------
    stays_e = load_eicu("stay_features_48h")
    e_ids = sorted(int(x) for x in stays_e["ICUSTAY_ID"])
    idx_e = stays_e.set_index("ICUSTAY_ID")
    y_e = idx_e.loc[e_ids, "HOSPITAL_MORTALITY"].to_numpy(float)
    S_e = idx_e.loc[e_ids, list(SHAPELET_STATIC_COLS)].to_numpy(float)
    print(f"[eicu] n={len(e_ids)} mort={y_e.mean():.4f}")

    a_e = pd.read_parquet(PROJ / "data/external/eicu/atoms.parquet")
    ind_node_e = indicator_matrix(e_ids, a_e[a_e["kind"] == "motif"], node_names)
    ind_lab_e = indicator_matrix(e_ids, a_e[a_e["kind"] == "lab"], lab_names)

    eicu_node = crude_assoc_table(y_e, ind_node_e, node_names, rng, "eICU").set_index("atom")
    eicu_lab = crude_assoc_table(y_e, ind_lab_e, lab_names, rng, "eICU").set_index("atom")

    # ---------------- L2: node association transfer ----------------
    cmp = pd.DataFrame({
        "node": node_names,
        "mimic_n": [int(mimic_node.loc[n, "n"]) for n in node_names],
        "mimic_diff": [float(mimic_node.loc[n, "death_rate_diff"]) for n in node_names],
        "mimic_sig": [bool(mimic_node.loc[n, "ci_excludes_0"]) for n in node_names],
        "mimic_bonf": [bool(mimic_node.loc[n, "bonferroni"]) for n in node_names],
        "eicu_n": [int(eicu_node.loc[n, "n_on"]) for n in node_names],
        "eicu_prev": [float(eicu_node.loc[n, "n_on"]) / len(e_ids) for n in node_names],
        "eicu_rate_on": [float(eicu_node.loc[n, "rate_on"]) for n in node_names],
        "eicu_rate_off": [float(eicu_node.loc[n, "rate_off"]) for n in node_names],
        "eicu_diff": [float(eicu_node.loc[n, "diff"]) for n in node_names],
        "eicu_lo": [float(eicu_node.loc[n, "lo"]) for n in node_names],
        "eicu_hi": [float(eicu_node.loc[n, "hi"]) for n in node_names],
    })
    cmp["eicu_sig"] = cmp["eicu_lo"] * cmp["eicu_hi"] > 0
    cmp["eicu_support"] = cmp["eicu_n"].between(MIN_SUPPORT, len(e_ids) - MIN_SUPPORT)
    cmp["same_sign"] = np.sign(cmp["mimic_diff"]) == np.sign(cmp["eicu_diff"])
    cmp["same_sign_and_sig"] = cmp["same_sign"] & cmp["eicu_sig"]

    r_node = spearmanr(cmp["mimic_diff"], cmp["eicu_diff"], nan_policy="omit")
    n_all = len(cmp)
    n_same = int(cmp["same_sign"].sum())
    s52 = cmp[cmp["mimic_sig"]]
    s34 = cmp[cmp["mimic_bonf"]]
    res_node = {
        "n_nodes": n_all,
        "sign_agreement": n_same,
        "sign_agreement_rate": n_same / n_all,
        "spearman_diff_rho": float(r_node.statistic),
        "spearman_diff_p": float(r_node.pvalue),
        "mimic_sig52_same_sign": int(s52["same_sign"].sum()),
        "mimic_sig52_same_and_eicu_sig": int(s52["same_sign_and_sig"].sum()),
        "mimic_sig52_same_and_eicu_sig_support20": int(
            (s52["same_sign_and_sig"] & s52["eicu_support"]).sum()),
        "mimic_bonf34_same_sign": int(s34["same_sign"].sum()),
        "mimic_bonf34_same_and_eicu_sig": int(s34["same_sign_and_sig"].sum()),
        "mimic_bonf34_same_and_eicu_sig_support20": int(
            (s34["same_sign_and_sig"] & s34["eicu_support"]).sum()),
        "eicu_sig_total": int(cmp["eicu_sig"].sum()),
        "eicu_support_total": int(cmp["eicu_support"].sum()),
    }
    print("\n[L2 node assoc transfer]", json.dumps(res_node, ensure_ascii=False, indent=1))

    # ---------------- lab atom direction transfer ----------------
    lcmp = pd.DataFrame({
        "atom": lab_names,
        "mimic_n": [int(mimic_lab.loc[n, "n_on"]) for n in lab_names],
        "mimic_diff": [float(mimic_lab.loc[n, "diff"]) for n in lab_names],
        "mimic_sig": [bool(mimic_lab.loc[n, "ci_excludes_0"]) for n in lab_names],
        "eicu_n": [int(eicu_lab.loc[n, "n_on"]) for n in lab_names],
        "eicu_diff": [float(eicu_lab.loc[n, "diff"]) for n in lab_names],
        "eicu_lo": [float(eicu_lab.loc[n, "lo"]) for n in lab_names],
        "eicu_hi": [float(eicu_lab.loc[n, "hi"]) for n in lab_names],
    })
    lcmp["eicu_sig"] = lcmp["eicu_lo"] * lcmp["eicu_hi"] > 0
    lcmp["same_sign"] = np.sign(lcmp["mimic_diff"]) == np.sign(lcmp["eicu_diff"])
    lcmp["same_sign_and_sig"] = lcmp["same_sign"] & lcmp["eicu_sig"]
    r_lab = spearmanr(lcmp["mimic_diff"], lcmp["eicu_diff"], nan_policy="omit")
    res_lab = {
        "n_lab_atoms": len(lcmp),
        "sign_agreement": int(lcmp["same_sign"].sum()),
        "sign_agreement_rate": float(lcmp["same_sign"].mean()),
        "spearman_diff_rho": float(r_lab.statistic),
        "mimic_sig_count": int(lcmp["mimic_sig"].sum()),
        "mimic_sig_same_and_eicu_sig": int(lcmp.loc[lcmp["mimic_sig"], "same_sign_and_sig"].sum()),
        "eicu_sig_total": int(lcmp["eicu_sig"].sum()),
    }
    print("[lab atoms]", json.dumps(res_lab, ensure_ascii=False, indent=1))

    # ---------------- L3: zero-shot node-bag prediction ----------------
    def fit_lr(X):
        return LogisticRegression(max_iter=400, solver="lbfgs").fit(X, y_tr)

    Z_tr = (S_tr - mu) / sd
    Z_e = (S_e - mu) / sd
    feats = {
        "static": Z_tr,
        "static+84nodes": np.column_stack([Z_tr, ind_node_tr]),
        "static+87labs": np.column_stack([Z_tr, ind_lab_tr]),
        "static+84+87": np.column_stack([Z_tr, ind_node_tr, ind_lab_tr]),
    }
    evals = {}
    for key, Xtr in feats.items():
        clf = fit_lr(Xtr)
        Xe = {
            "static": Z_e,
            "static+84nodes": np.column_stack([Z_e, ind_node_e]),
            "static+87labs": np.column_stack([Z_e, ind_lab_e]),
            "static+84+87": np.column_stack([Z_e, ind_node_e, ind_lab_e]),
        }[key]
        p = clf.predict_proba(Xe)[:, 1]
        evals[key] = {
            "auprc": float(average_precision_score(y_e, p)),
            "auroc": float(roc_auc_score(y_e, p)),
            "brier": float(brier_score_loss(y_e, p)),
        }
        print(f"[L3 zero-shot {key:>16}] " + json.dumps(evals[key]))
    res_pred = {"cohort": "eICU full (n=21657, zero-shot, MIMIC-train-frozen)", **evals}

    # ---------------- L2b: adjusted (static + single node) signs on eICU ----------------
    adj_rows = []
    for j, name in enumerate(node_names):
        clf = LogisticRegression(max_iter=200, solver="lbfgs").fit(
            np.column_stack([Z_e, ind_node_e[:, j]]), y_e)
        beta_e = float(clf.coef_[0, -1])
        clf_t = LogisticRegression(max_iter=200, solver="lbfgs").fit(
            np.column_stack([Z_tr, ind_node_tr[:, j]]), y_tr)
        beta_t = float(clf_t.coef_[0, -1])
        adj_rows.append({"node": name, "beta_mimic_train_adj": beta_t,
                         "beta_eicu_adj": beta_e,
                         "adj_same_sign": np.sign(beta_t) == np.sign(beta_e),
                         "adj_eicu_same_as_crude_mimic": np.sign(beta_e) == np.sign(
                             float(mimic_node.loc[name, "death_rate_diff"]))})
    adj = pd.DataFrame(adj_rows)
    res_adj = {
        "adjusted_sign_agreement_mimic_vs_eicu": int(adj["adj_same_sign"].sum()),
        "adjusted_sign_agreement_rate": float(adj["adj_same_sign"].mean()),
        "eicu_adjusted_same_as_mimic_crude": int(adj["adj_eicu_same_as_crude_mimic"].sum()),
        "mimic_adj_vs_mimic_crude_same": int(
            (np.sign(adj["beta_mimic_train_adj"]) == np.sign(cmp["mimic_diff"])).sum()),
    }
    print("[L2b adjusted]", json.dumps(res_adj, indent=1))

    # ---------------- representative examples ----------------
    examples = [
        "urine_output_ml|低位缓升型|B4",
        "urine_output_ml|低位缓升型|B3",
        "urine_output_ml|进行性下降型|B2",
        "heart_rate|持续上升型|B1",
        "heart_rate|持续上升型|B4",
        "sbp|低位下滑型|B4",
        "spo2|高位回落型|B1",
        "mbp|缓降持平型|B4",
    ]
    print("\n[examples] node | MIMIC diff | eICU diff (CI) | same?")
    for n_ in examples:
        r = cmp.set_index("node").loc[n_]
        print(f"  {n_:<38} {r['mimic_diff']:+.3f} | {r['eicu_diff']:+.3f} "
              f"({r['eicu_lo']:+.3f},{r['eicu_hi']:+.3f}) n={r['eicu_n']} same={bool(r['same_sign'])}")

    # ---------------- write outputs ----------------
    cmp.sort_values("mimic_diff", key=lambda s: s.abs(), ascending=False).to_csv(
        OUT_DIR / "node_assoc_transfer_84.csv", index=False, encoding="utf-8-sig")
    lcmp.sort_values("mimic_diff", key=lambda s: s.abs(), ascending=False).to_csv(
        OUT_DIR / "lab_assoc_transfer_87.csv", index=False, encoding="utf-8-sig")
    adj.to_csv(OUT_DIR / "node_assoc_adjusted_signs.csv", index=False, encoding="utf-8-sig")
    eicu_node.reset_index().to_csv(OUT_DIR / "eicu_node_outcome_assoc_raw.csv",
                                   index=False, encoding="utf-8-sig")
    eicu_lab.reset_index().to_csv(OUT_DIR / "eicu_lab_outcome_assoc_raw.csv",
                                  index=False, encoding="utf-8-sig")
    summary = {
        "post_hoc_note": "All eICU analyses are post-hoc exploratory; MIMIC side uses train only.",
        "cohorts": {"mimic_train": {"n": len(train_ids), "mortality": float(y_tr.mean())},
                    "eicu": {"n": len(e_ids), "mortality": float(y_e.mean())}},
        "L1_alphabet_distribution_reference": {
            "jsd_train_vs_eicu": 0.1330, "jsd_within_mimic": 0.0155,
            "spearman_train_vs_eicu": 0.888, "spearman_within_mimic": 0.9986,
            "source": "outputs/tables/eicu_motif_consistency.csv + motif_consistency.json"},
        "L2_node_outcome_assoc": res_node,
        "L2_lab_atom_assoc": res_lab,
        "L2b_adjusted_signs": res_adj,
        "L3_zero_shot_prediction": res_pred,
        "L4_paths_reference": {
            "same_sign_paths": "9/15 (<10 gate)",
            "auprc_static": 0.1895, "auprc_static_plus_paths": 0.1876,
            "auroc_static": 0.5724,
            "source": "data/external/eicu/path_external_reproduce.json"},
    }
    (OUT_DIR / "layer_transfer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[done] {time.time()-t0:.1f}s → {OUT_DIR}")


if __name__ == "__main__":
    main()
