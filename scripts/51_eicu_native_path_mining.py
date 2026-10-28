#!/usr/bin/env python3
"""51: eICU 原生 2-step 阶段序路径挖掘 + 与 MIMIC 冻结 15 条的结构对比（round-2 探索性）。

Protocol mirrors docs/20 (scripts/22) applied NATIVELY to the eICU full cohort:
  - strictly stage-increasing 2-step atom paths (one atom per (stage, atom))
  - support >= max(20, ceil(0.005 * N));  N = 21657  ->  109
  - bootstrap 100 x 80% subjects (subject == stay in eICU), >=70% support recurrence
  - static-adjusted logistic (IRLS, l2 = 1e-3), Wald 95% CI excluding 0
  - >=70% bootstrap same-sign (crude death-rate diff sign)
  - readability (round-2 task tokens): urine_output_ml / creatinine / bun / mbp /
    sbp / dbp / lactate / gcs_total / ph on at least one endpoint

ROUND-2 EXPLORATORY: eICU has no train/val; full cohort + stability only.
Results are never used to rewrite the frozen MIMIC path list.
Read-only everywhere else. Data outputs -> data/external/eicu/round2_mine/.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.eicu_constants import EICU_OUT_DIR, MIMIC_PATHS_DIR  # noqa: E402
from scripts.utils.graph_constants import EDGE_SUPPORT_ABS_MIN, EDGE_SUPPORT_FRAC  # noqa: E402
from scripts.utils.path_mine import crude_assoc, logistic_path_coef  # noqa: E402
from scripts.utils.path_protocol import (  # noqa: E402
    BOOT_RETAIN,
    BOOT_SUBJECT_FRAC,
    N_BOOT,
    path_key,
    parse_path_key,
)
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402

OUT = EICU_OUT_DIR / "round2_mine"
SEED = 42
STAGES = ["B1", "B2", "B3", "B4"]
# round-2 readability tokens (exact variable namespace, not substring)
READ_VARS = frozenset(
    {
        "urine_output_ml",
        "creatinine",
        "bun",
        "mbp",
        "sbp",
        "dbp",
        "lactate",
        "gcs_total",
        "ph",
    }
)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def atom_var(atom: str) -> str:
    return str(atom).split("|", 1)[0]


def atom_stage_idx(atom: str) -> int:
    for p in str(atom).split("|"):
        if p in STAGES:
            return STAGES.index(p)
    raise ValueError(f"no stage token in atom: {atom}")


def is_readable_task(a1: str, a2: str) -> bool:
    return atom_var(a1) in READ_VARS or atom_var(a2) in READ_VARS


def strip_stage(atom: str) -> str:
    return "|".join(p for p in str(atom).split("|") if p not in STAGES)


def stripped_pair(a1: str, a2: str) -> tuple[str, str]:
    return tuple(sorted((strip_stage(a1), strip_stage(a2))))  # type: ignore[return-value]


def pair_key(a1: str, a2: str, aid: dict[str, int], n_nodes: int) -> int:
    n1 = aid[a1] * 4 + atom_stage_idx(a1)
    n2 = aid[a2] * 4 + atom_stage_idx(a2)
    return n1 * n_nodes + n2


def key_to_atoms(k: int, vocab: list[str], n_nodes: int) -> tuple[str, str]:
    a1 = vocab[k // n_nodes // 4] + "|" + STAGES[(k // n_nodes) % 4]
    a2 = vocab[(k % n_nodes) // 4] + "|" + STAGES[(k % n_nodes) % 4]
    return a1, a2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    atoms = pd.read_parquet(EICU_OUT_DIR / "atoms.parquet")
    stays = pd.read_parquet(EICU_OUT_DIR / "stay_features_48h.parquet")

    # ------------------------------------------------------------------ universe
    stay_ids = sorted(int(x) for x in stays["ICUSTAY_ID"])
    n = len(stay_ids)
    min_support = max(EDGE_SUPPORT_ABS_MIN, int(math.ceil(EDGE_SUPPORT_FRAC * n)))
    row_of = {s: i for i, s in enumerate(stay_ids)}
    sf = stays.set_index("ICUSTAY_ID").loc[stay_ids]
    y = sf["HOSPITAL_MORTALITY"].to_numpy(float)

    # sanity: outcome consistent between atoms and stay_features
    a_y = atoms.groupby("ICUSTAY_ID")["HOSPITAL_MORTALITY"].first()
    mism = sum(1 for s, v in a_y.items() if float(v) != y[row_of[int(s)]])
    log(f"[check] outcome mismatches atoms vs stay_features: {mism}")

    # static matrix, z-scored as in scripts/22
    X = sf[list(SHAPELET_STATIC_COLS)].to_numpy(float)
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-6)
    static_z = (X - mu) / sd

    # ------------------------------------------------------- stay -> node arrays
    vocab = sorted(atoms["atom"].unique())
    aid = {a: i for i, a in enumerate(vocab)}
    n_nodes = len(vocab) * 4
    g = atoms[["ICUSTAY_ID", "atom", "stage_idx"]].copy()
    g["node"] = [aid[a] * 4 + int(s) for a, s in zip(g["atom"], g["stage_idx"])]
    by_stay: dict[int, list[np.ndarray]] = {}
    node_stages: dict[int, dict[int, tuple[int, int]]] = {}
    for sid, sub in g.groupby("ICUSTAY_ID"):
        arrs = []
        nmin: dict[int, int] = {}
        nmax: dict[int, int] = {}
        for st, nd in zip(sub["stage_idx"].to_numpy(int), sub["node"].to_numpy(int)):
            nmin[nd] = min(nmin.get(nd, 4), st)
            nmax[nd] = max(nmax.get(nd, -1), st)
        for st in range(4):
            u = np.unique(sub.loc[sub["stage_idx"] == st, "node"].to_numpy(np.int64))
            if len(u):
                arrs.append(u)
        if arrs:
            by_stay[int(sid)] = arrs
            node_stages[int(sid)] = {nd: (nmin[nd], nmax[nd]) for nd in nmin}
    log(f"[enum] stays={n} (with atoms {len(by_stay)}) vocab={len(vocab)} min_support={min_support}")

    # ------------------------------- pass 1: all per-stay 2-step pair keys + rows
    def build_keys_rows() -> tuple[np.ndarray, np.ndarray]:
        ks: list[np.ndarray] = []
        rs: list[np.ndarray] = []
        for sid in stay_ids:
            arrs = by_stay.get(sid)
            if not arrs:
                continue
            r = row_of[sid]
            for i in range(len(arrs)):
                for j in range(i + 1, len(arrs)):
                    ks.append((arrs[i][:, None] * n_nodes + arrs[j][None, :]).ravel())
                    rs.append(np.full(arrs[i].size * arrs[j].size, r, dtype=np.int32))
        K_ = np.concatenate(ks)
        R_ = np.concatenate(rs)
        order_ = np.argsort(K_, kind="stable")
        return K_[order_], R_[order_]

    K, R = build_keys_rows()
    uk, cnt = np.unique(K, return_counts=True)
    n_ge20 = int((cnt >= 20).sum())
    cand_keys = np.sort(uk[cnt >= min_support])
    S = len(cand_keys)
    log(f"[enum] distinct 2-step paths={len(uk)}  >=20: {n_ge20}  candidates(>= {min_support})={S}")

    m20 = cnt >= 20
    o20 = np.argsort(-cnt[m20])
    recs20 = [
        key_to_atoms(int(k), vocab, n_nodes) + (int(c),)
        for k, c in zip(uk[m20][o20], cnt[m20][o20])
    ]
    pd.DataFrame(recs20, columns=["a1", "a2", "support"]).to_parquet(
        OUT / "eicu_2step_counts_ge20.parquet", index=False
    )

    # ------------------------------------------------- indicator matrix for cands
    ind = np.zeros((n, S), dtype=bool)
    loc = np.clip(np.searchsorted(cand_keys, K), 0, S - 1)
    hit = cand_keys[loc] == K
    ind[R[hit], loc[hit]] = True
    del loc, hit
    sup = ind.sum(0)
    assert (sup == cnt[np.searchsorted(uk, cand_keys)]).all()
    log("[ind] candidate indicator built")
    del K, R  # freed; rebuilt later for targeted path lookups

    # ------------------------------------------ bootstrap 100x80% support recurrence
    rng = np.random.default_rng(SEED)
    k_boot = max(1, int(round(BOOT_SUBJECT_FRAC * n)))
    boots = [rng.choice(n, size=k_boot, replace=False) for _ in range(N_BOOT)]
    rec = np.zeros(S)
    for b in boots:
        rec += ind[b].sum(0) >= min_support
    rec /= N_BOOT
    surv = np.where(rec >= BOOT_RETAIN)[0]
    log(f"[boot] support-recurrence >= {BOOT_RETAIN}: {len(surv)}/{S}")

    # ------------------------------------ crude + static-adjusted logit (Wald CI)
    rows: list[dict] = []
    for jj, j in enumerate(surv):
        indj = ind[:, j].astype(float)
        crude = crude_assoc(y, indj)
        coef, se, _ = logistic_path_coef(y, indj, static_z)
        if np.isfinite(coef) and np.isfinite(se) and se > 0:
            lo, hi = coef - 1.96 * se, coef + 1.96 * se
            ci_ok = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
        else:
            lo = hi = float("nan")
            ci_ok = False
        sign = 0 if not np.isfinite(coef) or coef == 0 else (1 if coef > 0 else -1)
        a1, a2 = key_to_atoms(int(cand_keys[j]), vocab, n_nodes)
        rows.append(
            {
                "path": path_key((a1, a2)),
                "a1": a1,
                "a2": a2,
                "support": int(sup[j]),
                "death_support": int(y[ind[:, j]].sum()),
                **crude,
                "logit_coef": coef,
                "logit_se": se,
                "logit_ci_low": lo,
                "logit_ci_high": hi,
                "logit_sign": sign,
                "boot_support_rate": float(rec[j]),
                "ci_excl_0": bool(ci_ok),
            }
        )
        if (jj + 1) % 1000 == 0:
            log(f"[logit] {jj + 1}/{len(surv)} scored")
    df = pd.DataFrame(rows)
    sign_arr = df["logit_sign"].to_numpy()
    df["_col"] = surv  # candidate column index per survivor row

    # -------------------------------------------- vectorized bootstrap same-sign
    sign_hits = np.zeros(len(df))
    for b in boots:
        yb = y[b]
        indb = ind[b]
        n1 = indb.sum(0).astype(float)
        s1 = yb @ indb
        n0 = float(len(yb)) - n1
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = s1 / n1 - (yb.sum() - s1) / n0
        ds = diff[df["_col"].to_numpy()]
        valid = n1[df["_col"].to_numpy()] >= min_support
        sign_hits += valid & (np.sign(ds) == sign_arr)
    df["boot_sign_rate"] = sign_hits / N_BOOT
    df["stable"] = (
        (df["support"] >= min_support)
        & (df["boot_support_rate"] >= BOOT_RETAIN)
        & (df["boot_sign_rate"] >= BOOT_RETAIN)
        & df["ci_excl_0"]
        & (df["logit_sign"] != 0)
    )
    df["readable_task"] = [is_readable_task(a1, a2) for a1, a2 in zip(df["a1"], df["a2"])]
    df["_abs"] = df["logit_coef"].abs()
    df = df.sort_values(["stable", "_abs"], ascending=[False, False]).reset_index(drop=True)
    df.drop(columns=["_col", "_abs"]).to_csv(OUT / "eicu_survivors_full.csv", index=False)
    n_stable = int(df["stable"].sum())
    n_sr = int((df["stable"] & df["readable_task"]).sum())
    log(f"[done] survivors={len(df)} stable={n_stable} stable&readable(round2 def)={n_sr}")

    top50 = df[df["stable"] & df["readable_task"]].head(50).copy()
    top50.drop(columns=["_col", "_abs"]).to_csv(OUT / "eicu_top50_readable_stable.csv", index=False)

    # ==================================================================
    # Structural comparison vs the frozen MIMIC-15
    # ==================================================================
    sel_json = json.loads((MIMIC_PATHS_DIR / "path_set_selected.json").read_text(encoding="utf-8"))
    mimic15 = list(sel_json["selected"])

    # targeted path indicator via per-stay node stage ranges:
    # 2-step feasible iff min_stage(a1) < max_stage(a2) in the same stay
    def path_indicator(a1: str, a2: str) -> np.ndarray:
        n1 = aid[a1] * 4 + atom_stage_idx(a1)
        n2 = aid[a2] * 4 + atom_stage_idx(a2)
        v = np.zeros(n, dtype=float)
        for sid, d in node_stages.items():
            e1 = d.get(n1)
            e2 = d.get(n2)
            if e1 is not None and e2 is not None and e1[0] < e2[1]:
                v[row_of[sid]] = 1.0
        return v

    def support_of(a1: str, a2: str) -> int:
        kk = pair_key(a1, a2, aid, n_nodes)
        m = np.searchsorted(uk, kk)
        return int(cnt[m]) if (m < len(uk) and uk[m] == kk) else 0

    def evaluate_path(a1: str, a2: str) -> dict:
        indj = path_indicator(a1, a2)
        crude = crude_assoc(y, indj)
        assert crude["n_path"] == support_of(a1, a2), f"support mismatch {a1} -> {a2}"
        coef, se, _ = logistic_path_coef(y, indj, static_z)
        if np.isfinite(coef) and np.isfinite(se) and se > 0:
            lo, hi = coef - 1.96 * se, coef + 1.96 * se
            ci_ok = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
        else:
            lo = hi = float("nan")
            ci_ok = False
        drd = crude["death_rate_diff"]
        sgn = 0 if not np.isfinite(drd) or drd == 0 else (1 if drd > 0 else -1)
        return {
            "eicu_support": crude["n_path"],
            "eicu_death_rate_path": crude["death_rate_path"],
            "eicu_death_rate_other": crude["death_rate_other"],
            "eicu_crude_sign": sgn,
            "eicu_logit_coef": coef,
            "eicu_ci_low": lo,
            "eicu_ci_high": hi,
            "eicu_ci_excl_0": bool(ci_ok),
        }

    # MIMIC frozen signs from the MIMIC train stable table
    mim_stable = pd.read_csv(MIMIC_PATHS_DIR / "path_stable_train.csv")
    mim_sign_of = mim_stable.set_index("path")["logit_sign"].to_dict()

    # (a) exact 2-step overlap
    stable_paths = set(df.loc[df["stable"], "path"])
    top50_paths = set(top50["path"])
    rows_a = []
    for p in mimic15:
        a1, a2 = parse_path_key(p)
        ev = evaluate_path(a1, a2)
        rows_a.append(
            {
                "mimic_path": p,
                "mimic_logit_sign": int(mim_sign_of.get(p, 0)),
                "eicu_support": ev["eicu_support"],
                "eicu_support_ge_min": int(ev["eicu_support"] >= min_support),
                "eicu_crude_sign": ev["eicu_crude_sign"],
                "same_sign": int(int(mim_sign_of.get(p, 0)) == ev["eicu_crude_sign"] != 0),
                "in_eicu_stable_pool": p in stable_paths,
                "in_eicu_top50": p in top50_paths,
                **{k: v for k, v in ev.items() if k not in ("eicu_support", "eicu_crude_sign")},
            }
        )
    da = pd.DataFrame(rows_a)
    da.to_csv(OUT / "compare_a_mimic15_in_eicu.csv", index=False)
    a_stable = int(da["in_eicu_stable_pool"].sum())
    a_top50 = int(da["in_eicu_top50"].sum())
    a_ge_min = int(da["eicu_support_ge_min"].sum())
    a_ci = int(da["eicu_ci_excl_0"].sum())
    a_same = int(da["same_sign"].sum())
    log(
        f"[cmp-a] MIMIC15 in eICU: stable={a_stable}/15 top50={a_top50}/15 "
        f"sup>={min_support}={a_ge_min}/15 same_sign={a_same}/15 ci_excl0={a_ci}/15"
    )

    # (b) unordered stage-stripped atom pairs
    top50_pairs = {stripped_pair(a1, a2) for a1, a2 in zip(top50["a1"], top50["a2"])}
    stable_pairs = {stripped_pair(r.a1, r.a2) for r in df[df["stable"]].itertuples()}
    cnt20 = pd.read_parquet(OUT / "eicu_2step_counts_ge20.parquet")
    ge20_pairs = {stripped_pair(r.a1, r.a2) for r in cnt20.itertuples()}
    rows_b = []
    for p in mimic15:
        a1, a2 = parse_path_key(p)
        sp = stripped_pair(a1, a2)
        rows_b.append(
            {
                "mimic_path": p,
                "stripped_unordered_pair": " + ".join(sp),
                "in_eicu_top50": sp in top50_pairs,
                "in_eicu_stable_pool": sp in stable_pairs,
                "in_eicu_any_ge20": sp in ge20_pairs,
            }
        )
    db = pd.DataFrame(rows_b)
    db.to_csv(OUT / "compare_b_unordered_pairs.csv", index=False)
    b_top50 = int(db["in_eicu_top50"].sum())
    b_stable = int(db["in_eicu_stable_pool"].sum())
    b_ge20 = int(db["in_eicu_any_ge20"].sum())
    log(
        f"[cmp-b] stripped unordered pairs of MIMIC15 in eICU: top50={b_top50}/15 "
        f"stable={b_stable}/15 any>=20={b_ge20}/15"
    )

    # (c) endpoint atom Jaccard
    mim_eps: set[str] = set()
    mim_eps_st: set[str] = set()
    for p in mimic15:
        a1, a2 = parse_path_key(p)
        mim_eps.update((a1, a2))
        mim_eps_st.update(strip_stage(a) for a in (a1, a2))
    eic_eps = set(top50["a1"]) | set(top50["a2"])
    eic_eps_st = {strip_stage(a) for a in eic_eps}

    def jac(s1: set, s2: set) -> float:
        u = s1 | s2
        return len(s1 & s2) / len(u) if u else float("nan")

    jac_exact = jac(eic_eps, mim_eps)
    jac_stripped = jac(eic_eps_st, mim_eps_st)

    # pool level: eICU stable pool vs MIMIC stable pool (MIMIC train, 2-step only)
    mim_stable_paths = set(mim_stable.loc[mim_stable["stable"], "path"])
    mim_stable_2step = {p for p in mim_stable_paths if len(parse_path_key(p)) == 2}
    mim_stable_pairs: set[tuple[str, str]] = set()
    mim_stable_eps: set[str] = set()
    for p in mim_stable_2step:
        a1, a2 = parse_path_key(p)
        mim_stable_pairs.add(stripped_pair(a1, a2))
        mim_stable_eps.update(strip_stage(a) for a in (a1, a2))
    eic_stable_eps = {strip_stage(a) for a in df.loc[df["stable"], "a1"]}
    eic_stable_eps |= {strip_stage(a) for a in df.loc[df["stable"], "a2"]}
    pool_exact = len(stable_paths & mim_stable_paths)
    pool_pair_frac = len(mim_stable_pairs & stable_pairs) / max(1, len(mim_stable_pairs))
    dc = {
        "top50_vs_mimic15_endpoint_jaccard_exact": jac_exact,
        "top50_vs_mimic15_endpoint_jaccard_stripped": jac_stripped,
        "n_eicu_top50_endpoint_atoms": len(eic_eps),
        "n_mimic15_endpoint_atoms": len(mim_eps),
        "n_endpoint_overlap_exact": len(eic_eps & mim_eps),
        "stablepool_exact_path_overlap_count": pool_exact,
        "stablepool_pair_frac_of_mimic_found_in_eicu": pool_pair_frac,
        "stablepool_pair_jaccard": jac(mim_stable_pairs, stable_pairs),
        "stablepool_endpoint_jaccard_stripped": jac(eic_stable_eps, mim_stable_eps),
        "n_mimic_stable_paths": len(mim_stable_paths),
        "n_mimic_stable_2step": len(mim_stable_2step),
        "n_eicu_stable_paths": n_stable,
    }
    (OUT / "compare_c_endpoint_jaccard.json").write_text(
        json.dumps(dc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(
        f"[cmp-c] endpoint Jaccard top50-vs-15: exact={jac_exact:.3f} stripped={jac_stripped:.3f}; "
        f"pool exact overlap={pool_exact}; stable-pool pair frac={pool_pair_frac:.3f}"
    )

    # (d) endpoint variable composition
    def var_dist(endpoints: list[str]) -> dict[str, float]:
        return pd.Series([atom_var(a) for a in endpoints]).value_counts(normalize=True).to_dict()

    d_eic = var_dist(list(top50["a1"]) + list(top50["a2"]))
    d_mim = var_dist([a for p in mimic15 for a in parse_path_key(p)])
    allv = sorted(set(d_eic) | set(d_mim))
    dd = pd.DataFrame(
        {
            "var": allv,
            "eicu_top50_frac": [d_eic.get(v, 0.0) for v in allv],
            "mimic15_frac": [d_mim.get(v, 0.0) for v in allv],
        }
    ).sort_values("eicu_top50_frac", ascending=False)
    dd.to_csv(OUT / "compare_d_endpoint_var_dist.csv", index=False)
    tvd = 0.5 * float(np.abs(dd["eicu_top50_frac"] - dd["mimic15_frac"]).sum())
    v1 = np.sqrt(np.array([d_eic.get(v, 0.0) for v in allv]))
    v2 = np.sqrt(np.array([d_mim.get(v, 0.0) for v in allv]))
    cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    renal_eic = float(sum(f for v, f in d_eic.items() if v in READ_VARS))
    renal_mim = float(sum(f for v, f in d_mim.items() if v in READ_VARS))
    log(
        f"[cmp-d] endpoint var dist: TVD={tvd:.3f} cosine={cos:.3f} "
        f"renal/perf/metabolic share eICU={renal_eic:.3f} MIMIC={renal_mim:.3f}"
    )

    # ==================================================================
    # Internal utility: 5-fold CV AUPRC, static vs static+paths (eICU only)
    # ==================================================================
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    def cv_auprc(ind_mat: np.ndarray) -> list[float]:
        aucs = []
        for tr, te in kf.split(np.arange(n)):
            mu_f, sd_f = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-6)
            Xtr = np.column_stack([(X[tr] - mu_f) / sd_f, ind_mat[tr]])
            Xte = np.column_stack([(X[te] - mu_f) / sd_f, ind_mat[te]])
            clf = LogisticRegression(max_iter=500).fit(Xtr, y[tr])
            aucs.append(float(average_precision_score(y[te], clf.predict_proba(Xte)[:, 1])))
        return aucs

    top15 = df[df["stable"] & df["readable_task"]].head(15)
    native_paths = list(top15["path"])
    ind_native = (
        np.column_stack([path_indicator(*parse_path_key(p)) for p in native_paths])
        if native_paths
        else np.zeros((n, 0))
    )
    ind_mimic = np.column_stack([path_indicator(*parse_path_key(p)) for p in mimic15])
    res_u: dict = {
        "static_cv_auprc": cv_auprc(np.zeros((n, 0))),
        "static_plus_native_top15_cv_auprc": cv_auprc(ind_native),
        "static_plus_mimic15_zeroshot_cv_auprc": cv_auprc(ind_mimic),
        "native_top15_paths": native_paths,
        "n_native_paths_used": len(native_paths),
    }
    for k in [
        "static_cv_auprc",
        "static_plus_native_top15_cv_auprc",
        "static_plus_mimic15_zeroshot_cv_auprc",
    ]:
        v = res_u[k]
        res_u[k + "_mean"] = float(np.mean(v))
        res_u[k + "_sd"] = float(np.std(v))
    (OUT / "utility_cv.json").write_text(
        json.dumps(res_u, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(
        f"[util] 5-fold AUPRC static={res_u['static_cv_auprc_mean']:.4f} "
        f"+native15={res_u['static_plus_native_top15_cv_auprc_mean']:.4f} "
        f"+mimic15(zero-shot)={res_u['static_plus_mimic15_zeroshot_cv_auprc_mean']:.4f}"
    )

    # ------------------------------------------------------------------ manifest
    manifest = {
        "round": "round2_exploratory",
        "cohort": "eICU full (no train/val); stability via bootstrap only",
        "n_stays": n,
        "n_with_atoms": len(by_stay),
        "mortality_rate": float(y.mean()),
        "min_support": min_support,
        "n_distinct_2step": int(len(uk)),
        "n_ge20": n_ge20,
        "n_candidates": int(S),
        "n_boot_survivors": int(len(surv)),
        "n_stable": n_stable,
        "n_stable_readable_round2def": n_sr,
        "n_boot": N_BOOT,
        "boot_subject_frac": BOOT_SUBJECT_FRAC,
        "boot_retain": BOOT_RETAIN,
        "compare": {
            "a_exact_in_stable": a_stable,
            "a_exact_in_top50": a_top50,
            "a_exact_support_ge_min": a_ge_min,
            "a_exact_same_sign": a_same,
            "a_exact_ci_excl0": a_ci,
            "b_unordered_in_top50": b_top50,
            "b_unordered_in_stable": b_stable,
            "b_unordered_any_ge20": b_ge20,
            **dc,
            "endpoint_var_tvd": tvd,
            "endpoint_var_cosine": cos,
            "renal_share_eicu_top50": renal_eic,
            "renal_share_mimic15": renal_mim,
        },
        "utility": {
            "static_cv_auprc_mean": res_u["static_cv_auprc_mean"],
            "native15_cv_auprc_mean": res_u["static_plus_native_top15_cv_auprc_mean"],
            "mimic15_zeroshot_cv_auprc_mean": res_u["static_plus_mimic15_zeroshot_cv_auprc_mean"],
        },
        "read_vars_round2": sorted(READ_VARS),
        "seed": SEED,
        "runtime_s": round(time.time() - T0, 1),
    }
    (OUT / "mine_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
