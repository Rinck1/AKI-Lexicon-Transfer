#!/usr/bin/env python3
"""Reproduce the frozen 15 paths on eICU. Coefficients fit on MIMIC train only.

Reports the full landmark HA-AKI set and the pre-specified urine-complete subset.

KNOWN ISSUE (kept as-is for provenance; see outputs/reports/50_同号判据勘误.md):
the ``same_sign`` field below is computed as ``death_rate_path > death_rate_other``
inside eICU only. docs/28 §4.2 defines it as "same sign as the MIMIC-train
difference", so the 4 paths that are protective in MIMIC train (#8/10/11/14) are
judged in the wrong direction. Official round-1 output = 9/15 (implementation
criterion); protocol criterion recomputed by scripts/50 = 11/15 (sign test p=0.12).
Do not reuse this field for new analyses; use scripts/50_path_sign_erratum.py.
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

from scripts.utils.constants import DATA_PROCESSED, OUTPUT_REPORTS, OUTPUT_TABLES  # noqa: E402
from scripts.utils.eicu_constants import (  # noqa: E402
    MIMIC_PATHS_DIR,
    URINE_COMPLETE_MIN_HOURS,
)
from scripts.utils.eicu_io import load_eicu, out_dir  # noqa: E402
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.path_mine import crude_assoc, path_indicator_matrix  # noqa: E402
from scripts.utils.path_protocol import VAL_MIN_SUPPORT, parse_path_key  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _xy(stays: pd.DataFrame, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    idx = stays.set_index("ICUSTAY_ID")
    y = np.array([float(idx.loc[s, "HOSPITAL_MORTALITY"]) for s in ids])
    X = np.stack(
        [np.array([float(idx.loc[s, c]) for c in SHAPELET_STATIC_COLS], dtype=np.float64) for s in ids]
    )
    return y, X


def _eval_block(
    *,
    label: str,
    ids: list[int],
    y: np.ndarray,
    Xs: np.ndarray,
    clf_static: LogisticRegression,
    clf_both: LogisticRegression,
    paths: list[tuple[str, ...]],
    stay_atoms: dict,
    y_by: dict[int, float],
) -> dict:
    ind = path_indicator_matrix(ids, stay_atoms, paths)
    p_s = clf_static.predict_proba(Xs)[:, 1]
    p_b = clf_both.predict_proba(np.column_stack([Xs, ind]))[:, 1]
    rows = []
    for j, p in enumerate(paths):
        col = ind[:, j]
        crude = crude_assoc(y, col)
        rows.append(
            {
                "path": " → ".join(p),
                "support": int(col.sum()),
                "support_ok": bool(col.sum() >= VAL_MIN_SUPPORT),
                "death_rate_path": crude["death_rate_path"],
                "death_rate_other": crude["death_rate_other"],
                "odds_ratio": crude["odds_ratio"],
                "same_sign_as_risk": bool(
                    np.isfinite(crude["odds_ratio"]) and crude["odds_ratio"] > 1.0
                )
                if crude.get("death_rate_diff", 0) >= 0
                else bool(np.isfinite(crude["odds_ratio"]) and crude["odds_ratio"] < 1.0),
            }
        )
        # same_sign: death_rate_path > death_rate_other for the risk paths we selected
        d1, d0 = crude["death_rate_path"], crude["death_rate_other"]
        rows[-1]["same_sign"] = bool(np.isfinite(d1) and np.isfinite(d0) and d1 > d0)
    auprc_s = float(average_precision_score(y, p_s)) if y.min() != y.max() else float("nan")
    auprc_b = float(average_precision_score(y, p_b)) if y.min() != y.max() else float("nan")
    auc_s = float(roc_auc_score(y, p_s)) if y.min() != y.max() else float("nan")
    auc_b = float(roc_auc_score(y, p_b)) if y.min() != y.max() else float("nan")
    n_same = sum(1 for r in rows if r["same_sign"])
    n_sup = sum(1 for r in rows if r["support_ok"])
    return {
        "label": label,
        "n": int(len(ids)),
        "mortality": float(y.mean()),
        "auprc_static": auprc_s,
        "auprc_static_paths": auprc_b,
        "auroc_static": auc_s,
        "auroc_static_paths": auc_b,
        "n_same_sign": int(n_same),
        "n_support_ge_20": int(n_sup),
        "paths": rows,
    }


def main() -> None:
    freeze = json.loads((MIMIC_PATHS_DIR / "path_freeze_manifest.json").read_text(encoding="utf-8"))
    got = (out_dir() / "freeze_sha256.txt").read_text(encoding="utf-8").strip()
    if got != freeze["freeze_sha256"]:
        raise SystemExit(f"freeze hash mismatch eicu={got} mimic={freeze['freeze_sha256']}")
    sel = json.loads((MIMIC_PATHS_DIR / "path_set_selected.json").read_text(encoding="utf-8"))
    paths = [parse_path_key(p) for p in sel["selected"]]

    atoms = load_eicu("atoms")
    stay_atoms = atoms_by_stay(atoms)
    primary = load_eicu("aki_primary_ha48")
    stays_e = load_eicu("stay_features_48h")
    all_ids = sorted(int(x) for x in stays_e["ICUSTAY_ID"])
    y_e, X_e = _xy(stays_e, all_ids)

    _grid, mimic_stays = load_shapelet_tables(DATA_PROCESSED)
    train_ids = sorted(
        int(x) for x in mimic_stays.loc[mimic_stays["split"] == "train", "ICUSTAY_ID"]
    )
    y_tr, X_tr = _xy(mimic_stays, train_ids)
    mu, sd = X_tr.mean(0), np.maximum(X_tr.std(0), 1e-6)
    X_tr_z = (X_tr - mu) / sd
    X_e_z = (X_e - mu) / sd

    # Path indicators on MIMIC train for the joint head (same 15 frozen keys)
    mimic_atoms = pd.read_parquet(MIMIC_PATHS_DIR / "atoms.parquet")
    # train+val atoms file — filter train
    mimic_atoms = mimic_atoms[mimic_atoms["split"] == "train"]
    tr_stay_atoms = atoms_by_stay(mimic_atoms)
    tr_ind = path_indicator_matrix(train_ids, tr_stay_atoms, paths)

    clf_s = LogisticRegression(max_iter=400, solver="lbfgs")
    clf_s.fit(X_tr_z, y_tr)
    clf_b = LogisticRegression(max_iter=400, solver="lbfgs")
    clf_b.fit(np.column_stack([X_tr_z, tr_ind]), y_tr)

    y_by = {int(r.ICUSTAY_ID): float(r.HOSPITAL_MORTALITY) for r in stays_e.itertuples(index=False)}
    full = _eval_block(
        label="all_landmark_ha_aki",
        ids=all_ids,
        y=y_e,
        Xs=X_e_z,
        clf_static=clf_s,
        clf_both=clf_b,
        paths=paths,
        stay_atoms=stay_atoms,
        y_by=y_by,
    )
    uc = set(
        int(x)
        for x in primary.loc[primary.get("urine_complete", 0) == 1, "ICUSTAY_ID"]
    )
    uc_ids = [i for i in all_ids if i in uc]
    if uc_ids:
        idx = {s: k for k, s in enumerate(all_ids)}
        sel_i = [idx[s] for s in uc_ids]
        urine_block = _eval_block(
            label="urine_complete",
            ids=uc_ids,
            y=y_e[sel_i],
            Xs=X_e_z[sel_i],
            clf_static=clf_s,
            clf_both=clf_b,
            paths=paths,
            stay_atoms=stay_atoms,
            y_by=y_by,
        )
    else:
        urine_block = {"label": "urine_complete", "n": 0, "note": "no stays met urine-complete rule"}

    payload = {
        "stage": "46_eicu_reproduce_paths",
        "freeze_sha256": freeze["freeze_sha256"],
        "refit": False,
        "urine_complete_min_hours": URINE_COMPLETE_MIN_HOURS,
        "n_paths": len(paths),
        "full": full,
        "urine_complete": urine_block,
    }
    dest = out_dir() / "path_external_reproduce.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for r in full["paths"]:
        rows.append({"subset": "all", **r})
    if urine_block.get("paths"):
        for r in urine_block["paths"]:
            rows.append({"subset": "urine_complete", **r})
    tab = Path(OUTPUT_TABLES)
    tab.mkdir(parents=True, exist_ok=True)
    csv_path = tab / "eicu_path_reproduce.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    rep_dir = Path(OUTPUT_REPORTS)
    rep_dir.mkdir(parents=True, exist_ok=True)
    report = rep_dir / "34_eICU路径复现.md"
    lines = [
        "# eICU 冻结路径复现",
        "",
        f"- freeze `{freeze['freeze_sha256']}`",
        f"- 系数只在 MIMIC-III train 拟合；eICU 零训练",
        f"- 全队列 N={full['n']} 死亡={full['mortality']:.3f} 静态 AUPRC={full['auprc_static']:.3f} → 静态+路径 {full['auprc_static_paths']:.3f}",
        f"- 同号 {full['n_same_sign']}/15，支持≥20 {full['n_support_ge_20']}/15",
        f"- 尿量完整子集（≥{URINE_COMPLETE_MIN_HOURS}h）N={urine_block.get('n', 0)}",
        "",
        "| 路径 | 支持 | 同号 | OR |",
        "|---|---:|:---:|---:|",
    ]
    for r in full["paths"]:
        lines.append(
            f"| {r['path']} | {r['support']} | {'是' if r['same_sign'] else '否'} | {r['odds_ratio']:.2f} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"[46] full N={full['n']} AUPRC {full['auprc_static']:.3f}→{full['auprc_static_paths']:.3f}")
    _log(f"     same_sign={full['n_same_sign']}/15 support={full['n_support_ge_20']}/15")
    _log(f"     → {dest}")
    _log(f"     → {report}")


if __name__ == "__main__":
    main()
