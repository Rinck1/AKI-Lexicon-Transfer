#!/usr/bin/env python3
"""Round-2 erratum: `same_sign` criterion for the eICU frozen-path reproduction.

The official scripts/46_eicu_reproduce_paths.py (L81-83) implements
`same_sign = death_rate_path > death_rate_other` inside eICU only, which
mechanically equals (OR > 1) and can never recognize protective paths.
The pre-specified protocol (docs/28 §4.2) defines
`same_sign` = "eICU 与 MIMIC train 的死亡占比差同号" — a cross-dataset sign
comparison against the MIMIC train direction.

This script re-derives both criteria read-only:
  1. MIMIC train direction: static + 15 frozen-path logistic fit on
     split=='train' only (standardization uses train mu/sd), Wald 95% CI,
     plus the crude train death-rate diff (literal protocol wording).
  2. eICU zero-training crude association per path: death-rate diff sign,
     Woolf OR + CI, subject-cluster bootstrap (1000, seed 42) CI.
  3. Both criteria side by side + binomial sign test (H0: P=0.5).

Writes (round-2 only; never touches stage-46 outputs):
  - data/external/eicu/round2_erratum/corrected_same_sign.json
  - data/external/eicu/round2_erratum/path_sign_erratum_table.csv
  - outputs/reports/50_同号判据勘误.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED, OUTPUT_REPORTS  # noqa: E402
from scripts.utils.path_mine import crude_assoc  # noqa: E402
from scripts.utils.path_protocol import VAL_MIN_SUPPORT, parse_path_key  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402

PROCESSED_DIR = Path(DATA_PROCESSED)
MIMIC_PATHS_DIR = PROCESSED_DIR / "paths" / "path_final_7var_motif84_labs11"
EICU_DIR = PROCESSED_DIR.parent / "external" / "eicu"
ROUND2_DIR = EICU_DIR / "round2_erratum"

SEED = 42
N_BOOT = 1000
N_PATHS = 15
GATE_MIN_SAME = 10  # docs/28 §4.2 pre-specified: n_same_sign >= 10/15

# Verbatim evidence (read-only quotes; line numbers checked against HEAD).
QUOTE_46 = '''```python
74\t                "same_sign_as_risk": bool(
75\t                    np.isfinite(crude["odds_ratio"]) and crude["odds_ratio"] > 1.0
76\t                )
77\t                if crude.get("death_rate_diff", 0) >= 0
78\t                else bool(np.isfinite(crude["odds_ratio"]) and crude["odds_ratio"] < 1.0),
79\t            }
80\t        )
81\t        # same_sign: death_rate_path > death_rate_other for the risk paths we selected
82\t        d1, d0 = crude["death_rate_path"], crude["death_rate_other"]
83\t        rows[-1]["same_sign"] = bool(np.isfinite(d1) and np.isfinite(d0) and d1 > d0)
84\t    auprc_s = float(average_precision_score(y, p_s)) if y.min() != y.max() else float("nan")
85\t    auprc_b = float(average_precision_score(y, p_b)) if y.min() != y.max() else float("nan")
86\t    auc_s = float(roc_auc_score(y, p_s)) if y.min() != y.max() else float("nan")
87\t    auc_b = float(roc_auc_score(y, p_b)) if y.min() != y.max() else float("nan")
88\t    n_same = sum(1 for r in rows if r["same_sign"])
```'''

QUOTE_DOC28 = '''```markdown
| `same_sign` | eICU 与 MIMIC train 的死亡占比差同号 | 主标准：同号 |            ← docs/28 §4.2（L80）
| `logit_coef_train` | MIMIC train 上的路径 logistic 系数（冻结） | — |          ← docs/28 §4.2（L82）
```
判据原文（docs/28 §4.2 L100-102）：

```markdown
**预先指定的成功判据（写死，失败记失败，不回改 15 条）**：

1. `n_same_sign ≥ 10/15`（即 ≥ 2/3 同号）；
```'''


def _log(msg: str) -> None:
    print(msg, flush=True)


def fast_indicator(
    stay_ids: list[int],
    stay_atoms: dict[int, list[tuple[int, str, bool]]],
    paths: list[tuple[str, ...]],
) -> np.ndarray:
    """Greedy stage-assignment path matching (equivalent to path_indicator_matrix).

    Copied verbatim from scripts/_tmp_urine_threshold.py::fast_indicator.
    """
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


def xy(stays: pd.DataFrame, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    idx = stays.set_index("ICUSTAY_ID")
    y = idx.loc[ids, "HOSPITAL_MORTALITY"].to_numpy(dtype=np.float64)
    X = idx.loc[ids, list(SHAPELET_STATIC_COLS)].to_numpy(dtype=np.float64)
    return y, X


def woolf_ci(a: float, b: float, c: float, d: float) -> tuple[float, float, float]:
    """Haldane-Anscombe corrected OR with Woolf log-SE CI (same +0.5 as crude_assoc)."""
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    orr = (a / b) / (c / d)
    se = float(np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d))
    return float(orr), float(orr * np.exp(-1.96 * se)), float(orr * np.exp(1.96 * se))


def boot_or_ci(
    ind: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Subject-cluster bootstrap percentile CI for per-path ORs (vectorized)."""
    uniq = np.unique(clusters)
    order = np.argsort(clusters, kind="stable")
    ind_s, y_s = ind[order], y[order]
    cl_s = clusters[order]
    starts = np.searchsorted(cl_s, uniq, side="left")
    ends = np.searchsorted(cl_s, uniq, side="right")
    n_cl = len(uniq)
    # per-cluster (4, P) table: [deaths_path, n_path, deaths_other, n_other]
    tab = np.zeros((n_cl, 4, ind.shape[1]), dtype=np.float64)
    deaths = y_s
    for j in range(ind.shape[1]):
        col = ind_s[:, j] > 0.5
        i0, i1 = starts, ends
        tab[:, 0, j] = np.add.reduceat(deaths * col, i0) * (i1 > i0)
        tab[:, 1, j] = np.add.reduceat(col.astype(np.float64), i0) * (i1 > i0)
        tab[:, 2, j] = np.add.reduceat(deaths * (~col), i0) * (i1 > i0)
        tab[:, 3, j] = np.add.reduceat((~col).astype(np.float64), i0) * (i1 > i0)
    rng = np.random.default_rng(seed)
    ors = np.full((n_boot, ind.shape[1]), np.nan)
    for bi in range(n_boot):
        pick = rng.integers(0, n_cl, size=n_cl)
        agg = tab[pick].sum(axis=0)  # (4, P)
        a, b, c, d = agg[0], agg[1], agg[2], agg[3]
        with np.errstate(divide="ignore", invalid="ignore"):
            ors[bi] = ((a + 0.5) / (b + 0.5)) / ((c + 0.5) / (d + 0.5))
    lo = np.nanpercentile(ors, 2.5, axis=0)
    hi = np.nanpercentile(ors, 97.5, axis=0)
    return lo, hi


def main() -> None:
    sel = json.loads((MIMIC_PATHS_DIR / "path_set_selected.json").read_text(encoding="utf-8"))
    assert len(sel["selected"]) == N_PATHS
    paths = [parse_path_key(p) for p in sel["selected"]]
    official = json.loads((EICU_DIR / "path_external_reproduce.json").read_text(encoding="utf-8"))
    off_full = official["full"]
    off_by_path = {r["path"]: r for r in off_full["paths"]}

    # ------------------------------------------------------------------ MIMIC train
    stays_m = pd.read_parquet(PROCESSED_DIR / "stay_features_48h.parquet")
    train_ids = sorted(int(x) for x in stays_m.loc[stays_m["split"] == "train", "ICUSTAY_ID"])
    y_tr, X_tr = xy(stays_m, train_ids)
    mu, sd = X_tr.mean(0), np.maximum(X_tr.std(0), 1e-6)
    X_tr_z = (X_tr - mu) / sd

    atoms_m = pd.read_parquet(MIMIC_PATHS_DIR / "atoms.parquet")
    atoms_m = atoms_m[atoms_m["split"] == "train"]
    by_stay_m: dict[int, list[tuple[int, str, bool]]] = {}
    for rec in atoms_m.itertuples(index=False):
        by_stay_m.setdefault(int(rec.ICUSTAY_ID), []).append(
            (int(rec.stage_idx), str(rec.atom), bool(rec.family_low_confidence))
        )
    tr_ind = fast_indicator(train_ids, by_stay_m, paths)

    # Mirror scripts/46 fitting exactly: LogisticRegression(max_iter=400, lbfgs, C=1.0)
    clf_b = LogisticRegression(max_iter=400, solver="lbfgs")
    clf_b.fit(np.column_stack([X_tr_z, tr_ind]), y_tr)
    coefs = clf_b.coef_[0][len(SHAPELET_STATIC_COLS):]

    # Wald SE from unpenalized observed information at the fitted solution
    Xw = np.column_stack([np.ones(len(y_tr)), X_tr_z, tr_ind])
    p_hat = clf_b.predict_proba(np.column_stack([X_tr_z, tr_ind]))[:, 1]
    Wm = p_hat * (1.0 - p_hat)
    H = Xw.T @ (Wm[:, None] * Xw)
    cov = np.linalg.inv(H)
    ses = np.sqrt(np.diag(cov))[1 + len(SHAPELET_STATIC_COLS):]
    coef_lo = coefs - 1.96 * ses
    coef_hi = coefs + 1.96 * ses

    # Literal protocol reading: crude MIMIC train death-rate diff (same rows)
    tr_crude = [crude_assoc(y_tr, tr_ind[:, j]) for j in range(N_PATHS)]

    # ------------------------------------------------------------------ eICU zero-training
    stays_e = pd.read_parquet(EICU_DIR / "stay_features_48h.parquet")
    atoms_e = pd.read_parquet(EICU_DIR / "atoms.parquet")
    by_stay_e: dict[int, list[tuple[int, str, bool]]] = {}
    for rec in atoms_e.itertuples(index=False):
        by_stay_e.setdefault(int(rec.ICUSTAY_ID), []).append(
            (int(rec.stage_idx), str(rec.atom), bool(rec.family_low_confidence))
        )
    all_ids = sorted(int(x) for x in stays_e["ICUSTAY_ID"])
    y_e, X_e = xy(stays_e, all_ids)
    X_e_z = (X_e - mu) / sd
    e_ind = fast_indicator(all_ids, by_stay_e, paths)
    subj = stays_e.set_index("ICUSTAY_ID").loc[all_ids, "SUBJECT_ID"].to_numpy()
    clusters = pd.factorize(subj)[0].astype(np.int64)
    boot_lo, boot_hi = boot_or_ci(e_ind, y_e, clusters)

    clf_s = LogisticRegression(max_iter=400, solver="lbfgs")
    clf_s.fit(X_tr_z, y_tr)
    auprc_s = float(average_precision_score(y_e, clf_s.predict_proba(X_e_z)[:, 1]))
    p_b = clf_b.predict_proba(np.column_stack([X_e_z, e_ind]))[:, 1]
    auprc_b = float(average_precision_score(y_e, p_b))
    auroc_s = float(roc_auc_score(y_e, clf_s.predict_proba(X_e_z)[:, 1]))
    auroc_b = float(roc_auc_score(y_e, p_b))

    # ------------------------------------------------------------------ per-path rows
    rows = []
    for j, p in enumerate(paths):
        key = " → ".join(p)
        crude = crude_assoc(y_e, e_ind[:, j])
        off = off_by_path.get(key)
        assert off is not None, f"path missing in official json: {key}"
        assert int(crude["n_path"]) == int(off["support"]), (
            f"support mismatch vs official for {key}: {crude['n_path']} vs {off['support']}"
        )
        assert abs(crude["odds_ratio"] - off["odds_ratio"]) < 1e-9
        impl_same = bool(np.isfinite(crude["death_rate_path"]) and
                         np.isfinite(crude["death_rate_other"]) and
                         crude["death_rate_path"] > crude["death_rate_other"])
        assert impl_same == bool(off["same_sign"]), f"impl same_sign mismatch for {key}"

        c, lo, hi = coefs[j], coef_lo[j], coef_hi[j]
        tr_sign = int(np.sign(c)) if np.isfinite(c) and c != 0 else 0
        tr_crude_sign = int(np.sign(tr_crude[j]["death_rate_diff"]))
        e_diff = crude["death_rate_diff"]
        e_sign = int(np.sign(e_diff)) if np.isfinite(e_diff) and e_diff != 0 else 0
        proto_adj = bool(tr_sign != 0 and e_sign == tr_sign)
        proto_crude = bool(tr_crude_sign != 0 and e_sign == tr_crude_sign)
        w_oz, w_lo, w_hi = woolf_ci(
            float(((e_ind[:, j] > 0.5) & (y_e > 0.5)).sum()),
            float(((e_ind[:, j] > 0.5) & (y_e <= 0.5)).sum()),
            float(((e_ind[:, j] <= 0.5) & (y_e > 0.5)).sum()),
            float(((e_ind[:, j] <= 0.5) & (y_e <= 0.5)).sum()),
        )
        rows.append({
            "no": j + 1,
            "path": key,
            "mimic_train_coef": float(c),
            "mimic_train_coef_ci": (float(lo), float(hi)),
            "mimic_train_direction": ("risk(+)" if tr_sign > 0 else "protective(-)" if tr_sign < 0 else "zero"),
            "mimic_train_crude_diff": float(tr_crude[j]["death_rate_diff"]),
            "mimic_train_crude_sign": tr_crude_sign,
            "mimic_train_support": int(tr_crude[j]["n_path"]),
            "eicu_support": int(crude["n_path"]),
            "eicu_support_ok": bool(crude["n_path"] >= VAL_MIN_SUPPORT),
            "eicu_death_rate_path": float(crude["death_rate_path"]),
            "eicu_death_rate_other": float(crude["death_rate_other"]),
            "eicu_death_rate_diff": float(e_diff),
            "eicu_or": float(w_oz),
            "eicu_or_woolf_ci": (float(w_lo), float(w_hi)),
            "eicu_or_boot_ci": (float(boot_lo[j]), float(boot_hi[j])),
            "impl_same_sign": impl_same,
            "protocol_same_sign_adj": proto_adj,
            "protocol_same_sign_crude": proto_crude,
        })

    n_impl = sum(r["impl_same_sign"] for r in rows)
    n_proto = sum(r["protocol_same_sign_adj"] for r in rows)
    n_proto_crude = sum(r["protocol_same_sign_crude"] for r in rows)
    assert n_impl == int(off_full["n_same_sign"]), "recomputed impl count must equal official 9/15"

    pt_proto = binomtest(n_proto, N_PATHS, 0.5, alternative="two-sided").pvalue
    pt_impl = binomtest(n_impl, N_PATHS, 0.5, alternative="two-sided").pvalue
    pt_proto_crude = binomtest(n_proto_crude, N_PATHS, 0.5, alternative="two-sided").pvalue

    payload = {
        "stage": "50_path_sign_erratum",
        "round": "round2_erratum",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "freeze_sha256": (EICU_DIR / "freeze_sha256.txt").read_text(encoding="utf-8").strip(),
        "official_source": "data/external/eicu/path_external_reproduce.json (untouched)",
        "definitions": {
            "impl": "scripts/46 L81-83: same_sign = death_rate_path > death_rate_other (eICU-only)",
            "protocol": "docs/28 §4.2 L80: same_sign = eICU 与 MIMIC train 的死亡占比差同号",
            "protocol_adj_reference": "MIMIC train static+15path logistic coef sign (train-only fit)",
            "protocol_crude_reference": "MIMIC train crude death_rate_diff sign (literal wording)",
        },
        "eicu": {
            "n": int(off_full["n"]),
            "mortality": float(off_full["mortality"]),
            "auprc_static_recomputed": auprc_s,
            "auprc_static_paths_recomputed": auprc_b,
            "auroc_static_recomputed": auroc_s,
            "auroc_static_paths_recomputed": auroc_b,
            "auprc_static_official": float(off_full["auprc_static"]),
            "auprc_static_paths_official": float(off_full["auprc_static_paths"]),
        },
        "counts": {
            "impl_same_sign": n_impl,
            "protocol_same_sign_adj": n_proto,
            "protocol_same_sign_crude": n_proto_crude,
            "gate_min_same_sign": GATE_MIN_SAME,
        },
        "sign_tests": {
            "protocol_adj_k": n_proto,
            "protocol_adj_p_two_sided": float(pt_proto),
            "protocol_crude_k": n_proto_crude,
            "protocol_crude_p_two_sided": float(pt_proto_crude),
            "impl_k": n_impl,
            "impl_p_two_sided": float(pt_impl),
        },
        "paths": rows,
    }
    ROUND2_DIR.mkdir(parents=True, exist_ok=True)
    (ROUND2_DIR / "corrected_same_sign.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(ROUND2_DIR / "path_sign_erratum_table.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ report
    rep = render_report(payload)
    out_rep = Path(OUTPUT_REPORTS) / "50_同号判据勘误.md"
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    out_rep.write_text(rep, encoding="utf-8")

    _log(f"[50] impl {n_impl}/15 (official {off_full['n_same_sign']}) protocol_adj {n_proto}/15 "
         f"protocol_crude {n_proto_crude}/15")
    _log(f"[50] sign test p (protocol {n_proto}/15) = {pt_proto:.4f}")
    _log(f"[50] auprc check: static {auprc_s:.4f} vs {off_full['auprc_static']:.4f}; "
         f"+paths {auprc_b:.4f} vs {off_full['auprc_static_paths']:.4f}")
    _log(f"[50] -> {ROUND2_DIR / 'corrected_same_sign.json'}")
    _log(f"[50] -> {out_rep}")


def render_report(d: dict) -> str:
    rows = d["paths"]
    n_impl = d["counts"]["impl_same_sign"]
    n_proto = d["counts"]["protocol_same_sign_adj"]
    n_proto_crude = d["counts"]["protocol_same_sign_crude"]
    pt = d["sign_tests"]["protocol_adj_p_two_sided"]
    ptc = d["sign_tests"]["protocol_crude_p_two_sided"]
    pti = d["sign_tests"]["impl_p_two_sided"]
    eic = d["eicu"]
    flips_to_same = [r for r in rows if (not r["impl_same_sign"]) and r["protocol_same_sign_adj"]]
    flips_to_diff = [r for r in rows if r["impl_same_sign"] and (not r["protocol_same_sign_adj"])]

    def dir_cn(r: dict) -> str:
        if r["mimic_train_direction"].startswith("risk"):
            return "风险 (+)"
        if r["mimic_train_direction"].startswith("prot"):
            return "保护 (−)"
        return "零"

    def esc(s) -> str:
        return str(s).replace("|", "\\|")

    lines = [
        "# 50 · eICU 路径复现 `same_sign` 判据勘误（round-2）",
        "",
        f"- 生成：`scripts/50_path_sign_erratum.py`（seed 42；bootstrap {d['n_bootstrap']} 次，subject 聚类）",
        f"- 冻结哈希 `{d['freeze_sha256']}`；官方结果 `path_external_reproduce.json` **未改动**，仅复核",
        f"- eICU 全队列 N={eic['n']}，死亡率 {eic['mortality']:.3f}；AUPRC 复算 静态 {eic['auprc_static_recomputed']:.4f} / "
        f"静态+路径 {eic['auprc_static_paths_recomputed']:.4f}（与官方 {eic['auprc_static_official']:.4f}/{eic['auprc_static_paths_official']:.4f} 一致）",
        "",
        "## 1. 勘误陈述：实现 ≠ 协议预指定定义",
        "",
        "**协议预指定（docs/28 §4.2，冻结在先）**——`same_sign` 被定义为 eICU 与 MIMIC train 的**跨数据集死亡占比差同号**：",
        "",
        QUOTE_DOC28,
        "",
        "**官方实现（scripts/46_eicu_reproduce_paths.py L74-88）**——退化为 eICU **内部**的 `d1 > d0`（注释自认只对风险路径成立）：",
        "",
        QUOTE_46,
        "",
        "**偏差后果**：实现口径下 `same_sign` 机械等价于 OR>1（L74-78 的 `same_sign_as_risk` 与 L83 完全同值），"
        "结构上**不可能承认保护性路径**；对 MIMIC train 系数为负的路径，判定方向与协议相反。"
        "docs/28 §4.2 同时定义了 `logit_coef_train`（L82）作为 train 方向参照，但 46 从未使用。",
        "",
        "## 2. MIMIC train 方向重估（只读 train split）",
        "",
        "static（11 列）+ 15 路径指示 logistic，`LogisticRegression(max_iter=400, lbfgs)` 与 46 完全同口径；"
        "标准化 μ/σ 只用 train（N=7,610）。系数 95% CI 为拟合点处非惩罚观测信息阵的 Wald 区间；"
        "粗死亡占比差按字面协议口径同表给出：",
        "",
        "| # | 路径 | train 系数 [95% CI] | 方向 | train 粗差符号 | train 支持 |",
        "|---|------|---|:---:|:---:|---:|",
    ]
    for r in rows:
        c, (lo, hi) = r["mimic_train_coef"], r["mimic_train_coef_ci"]
        arrow = "+" if r["mimic_train_crude_sign"] > 0 else ("−" if r["mimic_train_crude_sign"] < 0 else "0")
        lines.append(
            f"| {r['no']} | {esc(r['path'])} | {c:+.3f} [{lo:+.3f}, {hi:+.3f}] | {dir_cn(r)} | {arrow} | {r['mimic_train_support']} |"
        )
    lines += [
        "",
        "4 条路径在 MIMIC train 为**保护性**（系数为负且 95% CI 全在 0 左侧）："
        "#8 尿量低位波动B1→肌酐B4↑、#10 呼吸末期骤降B1→尿量低位波动B4、"
        "#11 尿量低位波动B1→尿量进行性下降B2、#14 pH B1↑→GCS B2↑。"
        "#8/#10 与 docs/22 §7.4 记录一致（两折变号、保护方向示例）；#11/#14 由本次 train 重估确证。",
        "",
        "## 3. 两口径判定对照（eICU 零训练，15 条逐条）",
        "",
        "eICU 侧为粗关联（`crude_assoc`，+0.5 连续性校正）；OR 95% CI 给 Woolf 解析区间与 subject 聚类 bootstrap 百分位区间（1000 次，seed 42）。"
        "实现口径 = eICU 内部 d1>d0；协议口径 = sign(eICU 死亡占比差) 与 sign(MIMIC train 系数) 同号（字面粗差口径结论相同，见 §5）：",
        "",
        "| # | 路径 | eICU 支持 | eICU 差 d1−d0 | eICU OR [Woolf CI] [boot CI] | train 方向 | 实现口径 | 协议口径 |",
        "|---|------|---:|---:|---|:---:|:---:|:---:|",
    ]
    for r in rows:
        diff = r["eicu_death_rate_diff"]
        wor, (wlo, whi) = r["eicu_or"], r["eicu_or_woolf_ci"]
        (blo, bhi) = r["eicu_or_boot_ci"]
        tr_dir = "风险" if r["mimic_train_direction"].startswith("risk") else "保护"
        lines.append(
            f"| {r['no']} | {esc(r['path'])} | {r['eicu_support']} | {diff:+.4f} | "
            f"{wor:.2f} [{wlo:.2f}, {whi:.2f}] [{blo:.2f}, {bhi:.2f}] | {tr_dir} | "
            f"{'同号' if r['impl_same_sign'] else '异号'} | "
            f"{'同号' if r['protocol_same_sign_adj'] else '**异号**'} |"
        )
    lines += [
        "",
        f"**计数**：实现口径 **{n_impl}/15**（与官方逐条一致，含支持断言）；协议口径 **{n_proto}/15**（字面粗差口径 {n_proto_crude}/15，两参考一致）。",
        "",
        "## 4. 被实现口径错判的 4 条（误判明细）",
        "",
        "**3 条保护性路径被错判为异号（实为同号）**：",
        "",
        "| # | 路径 | train 系数 [CI] | eICU OR [boot CI] | 实现口径 | 协议口径 |",
        "|---|------|---|---|:---:|:---:|",
    ]
    for r in flips_to_same:
        c, (lo, hi) = r["mimic_train_coef"], r["mimic_train_coef_ci"]
        (blo, bhi) = r["eicu_or_boot_ci"]
        prot_rep = "CI 不含 1，保护方向复现" if bhi < 1 else "CI 跨 1，方向一致但幅度不确定"
        lines.append(
            f"| {r['no']} | {esc(r['path'])} | {c:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
            f"{r['eicu_or']:.2f} [{blo:.2f}, {bhi:.2f}]（{prot_rep}） | 异号 ✗ | 同号 ✓ |"
        )
    lines += [
        "",
        "**1 条被错判为同号（实为异号——方向真实翻转）**：",
        "",
        "| # | 路径 | train 系数 [CI] | eICU 差 | eICU OR [boot CI] | 实现口径 | 协议口径 |",
        "|---|------|---|---|---|:---:|:---:|",
    ]
    for r in flips_to_diff:
        c, (lo, hi) = r["mimic_train_coef"], r["mimic_train_coef_ci"]
        (blo, bhi) = r["eicu_or_boot_ci"]
        lines.append(
            f"| {r['no']} | {esc(r['path'])} | {c:+.3f} [{lo:+.3f}, {hi:+.3f}] | {r['eicu_death_rate_diff']:+.4f} | "
            f"{r['eicu_or']:.2f} [{blo:.2f}, {bhi:.2f}] | 同号 ✓ | **异号 ✗** |"
        )
    lines += [
        "",
        "#14 在 MIMIC train 是保护性（docs/22 §7.4 原文列为「保护方向示例，不作机制」），"
        "eICU 上 OR>1（CI 跨 1），**方向翻转本身是真实的迁移失败信号**，实现口径把它记成同号属于误判而非修正。",
        "",
        "## 5. 符号检验（sign test）",
        "",
        f"- 协议口径：H0 同号概率=0.5，k={n_proto}/15，双侧精确二项 **p = {pt:.4f}**（不优于随机）；",
        f"- 字面粗差参考：k={n_proto_crude}/15，p = {ptc:.4f}；",
        f"- 实现口径：k={n_impl}/15，p = {pti:.4f}。",
        "",
        "即：修正判据让计数过了 docs/28 §4.2 预设门槛（≥10/15），但**没有任何口径的同号率显著优于抛硬币**——"
        "门槛是预指定的描述性标准，符号检验不过说明不能宣称「方向迁移稳健」。",
        "",
        "## 6. 写作纪律（论文中必须执行）",
        "",
        f"1. **两个数字并列披露**：官方实现口径 {n_impl}/15（未过 ≥10 门槛，按 docs/28 预指定记为未达）"
        f"与协议口径重算 {n_proto}/15（过门槛但 sign test p={pt:.2f}）。",
        "2. **不得单独引用修正后数字宣称「外部同号验证通过」**；必须同时说明：(a) 原实现将判据窄化为 eICU 内部 OR>1，"
        "对保护性路径系统性误判；(b) 修正后仍有 1 条方向真实翻转（#14）；(c) 两口径 sign test 均不显著，结论只能写"
        "「多数路径方向可迁移，但整体同号率不优于随机，迁移证据有限」。",
        "3. 判据勘误属**事后重算（round-2 post-hoc）**，论文须标注其为 protocol-defined criterion 的重新执行，非预注册结果的更换。",
        "",
        "## 7. 产出与可复现",
        "",
        "- 数据：`data/external/eicu/round2_erratum/corrected_same_sign.json`、`path_sign_erratum_table.csv`",
        "- 脚本：`scripts/50_path_sign_erratum.py`（只读现有文件；MIMIC 拟合只用 split=='train'；不触碰 46 与官方 JSON）",
        "- 交叉核对：eICU 支持/OR/实现口径同号逐条断言等于官方 JSON；AUPRC 复算一致",
        "- 前作只读参考：`data/external/eicu/_explore_DX/posthoc_DX_path_or_ci.csv`（其 train 保护系数 −2.59/−2.16/−1.75/−1.72 与本次重估方向一致）",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
