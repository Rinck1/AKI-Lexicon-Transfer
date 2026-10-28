#!/usr/bin/env python3
"""One-shot val reproduction of frozen train paths (docs/20 §5 / §9.4). Refuses test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.path_mine import (  # noqa: E402
    crude_assoc,
    logistic_path_coef,
    path_indicator_matrix,
)
from scripts.utils.path_protocol import (  # noqa: E402
    VAL_MIN_SUPPORT,
    parse_path_key,
    path_key,
    refuse_test_path,
    refuse_test_split,
)
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402

DEFAULT = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-dir", default=str(DEFAULT))
    p.add_argument("--report", default=str(PROJ / "outputs/reports/20_时序路径val复现.md"))
    return p.parse_args()


def _split_xy(stays, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    stay_idx = stays.set_index("ICUSTAY_ID")
    y = np.array([float(stay_idx.loc[s, "HOSPITAL_MORTALITY"]) for s in ids])
    X = np.stack(
        [np.array([float(stay_idx.loc[s, c]) for c in SHAPELET_STATIC_COLS], dtype=np.float64) for s in ids]
    )
    return y, X


def main() -> None:
    args = parse_args()
    d = Path(args.paths_dir)
    refuse_test_path(d)
    refuse_test_path(args.report)
    sel = json.loads((d / "path_set_selected.json").read_text(encoding="utf-8"))
    paths = [parse_path_key(p) for p in sel.get("selected") or []]
    atoms = pd.read_parquet(d / "atoms.parquet")
    for sp in atoms["split"].astype(str).unique():
        refuse_test_split(sp)
    ev_idx = PROJ / "data/processed/shapelet/events/events_final_7var_motif84/stay_index.parquet"
    idx = pd.read_parquet(ev_idx)
    train_ids = sorted(int(x) for x in idx.loc[idx["split"] == "train", "ICUSTAY_ID"])
    val_ids = sorted(int(x) for x in idx.loc[idx["split"] == "val", "ICUSTAY_ID"])
    _grid, stays = load_shapelet_tables(DATA_PROCESSED)
    y_tr, X_tr = _split_xy(stays, train_ids)
    y_va, X_va = _split_xy(stays, val_ids)
    mu, sd = X_tr.mean(0), np.maximum(X_tr.std(0), 1e-6)
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd

    tr_atoms = atoms_by_stay(atoms[atoms["split"] == "train"])
    va_atoms = atoms_by_stay(atoms[atoms["split"] == "val"])
    tr_ind = path_indicator_matrix(train_ids, tr_atoms, paths) if paths else np.zeros((len(train_ids), 0))
    va_ind = path_indicator_matrix(val_ids, va_atoms, paths) if paths else np.zeros((len(val_ids), 0))

    rows = []
    for j, p in enumerate(paths):
        crude = crude_assoc(y_va, va_ind[:, j])
        coef, se, _ = logistic_path_coef(y_va, va_ind[:, j], X_va)
        train_coef, _, _ = logistic_path_coef(y_tr, tr_ind[:, j], X_tr)
        same = (
            np.isfinite(coef)
            and np.isfinite(train_coef)
            and np.sign(coef) == np.sign(train_coef)
            and np.sign(train_coef) != 0
        )
        rows.append(
            {
                "path": path_key(p),
                "val_support": int(va_ind[:, j].sum()),
                "val_death_rate_path": crude["death_rate_path"],
                "val_death_rate_other": crude["death_rate_other"],
                "val_death_rate_diff": crude["death_rate_diff"],
                "val_odds_ratio": crude["odds_ratio"],
                "val_logit_coef": coef,
                "train_logit_coef": train_coef,
                "same_sign": bool(same),
                "val_support_ok": int(va_ind[:, j].sum()) >= VAL_MIN_SUPPORT,
            }
        )

    # Train-fit logistic → val AUPRC
    def _auprc(Xtr, Xva, ytr, yva):
        if yva.min() == yva.max() or ytr.min() == ytr.max():
            return float("nan")
        clf = LogisticRegression(max_iter=300, solver="lbfgs")
        clf.fit(Xtr, ytr)
        return float(average_precision_score(yva, clf.predict_proba(Xva)[:, 1]))

    a_static = _auprc(X_tr, X_va, y_tr, y_va)
    if paths:
        a_paths = _auprc(
            np.column_stack([X_tr, tr_ind]),
            np.column_stack([X_va, va_ind]),
            y_tr,
            y_va,
        )
    else:
        a_paths = a_static

    summary = {
        "stage": "path_val_reproduce_once",
        "test_read": False,
        "val_used": "once",
        "n_paths": len(paths),
        "n_same_sign": int(sum(r["same_sign"] for r in rows)),
        "n_val_support_ok": int(sum(r["val_support_ok"] for r in rows)),
        "val_auprc_static": a_static,
        "val_auprc_static_plus_paths": a_paths,
        "paths": rows,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (d / "path_val_reproduce.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# 时序路径 val 一次复现",
        "",
        "> 规则与名单只在 train 上冻结。本文件是唯一一次 val 复现；未读 test。",
        "",
        f"- 选中路径：{len(paths)}。同号 {summary['n_same_sign']}；val 支持≥{VAL_MIN_SUPPORT}：{summary['n_val_support_ok']}。",
        f"- val AUPRC 静态：{a_static:.4f}；静态+路径：{a_paths:.4f}。",
        "",
        "| 路径 | val支持 | 死亡率差 | val系数 | train系数 | 同号 | 支持过关 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['path']} | {r['val_support']} | {r['val_death_rate_diff']:.4f} | "
            f"{r['val_logit_coef']:.3f} | {r['train_logit_coef']:.3f} | "
            f"{'是' if r['same_sign'] else '否'} | {'是' if r['val_support_ok'] else '否'} |"
        )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(json.dumps({k: summary[k] for k in summary if k != "paths"}, ensure_ascii=False, indent=2))
    _log("[24] DONE (val used once)")


if __name__ == "__main__":
    main()
