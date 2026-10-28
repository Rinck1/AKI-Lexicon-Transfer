#!/usr/bin/env python3
"""One-shot holdout (official test split) after freeze hash. docs/20 §5.

Does not retune. Refuses a second run. Coefficients fit on train only.
Applies frozen Shapelet q, family map, and lab thresholds to holdout stays.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

import importlib.util  # noqa: E402

_spec15 = importlib.util.spec_from_file_location(
    "export15", PROJ / "scripts" / "15_export_shapelet_events.py"
)
_mod15 = importlib.util.module_from_spec(_spec15)
assert _spec15.loader is not None
_spec15.loader.exec_module(_mod15)
export_split_activations = _mod15.export_split_activations
load_frozen_model = _mod15.load_frozen_model
from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.graph_constants import A_MIN_EVENT  # noqa: E402
from scripts.utils.graph_events import select_active_events  # noqa: E402
from scripts.utils.graph_sparse import build_sparse_matrix_for_stays, sparse_feature_names  # noqa: E402
from scripts.utils.path_atoms import atoms_by_stay, lab_atoms_from_raw, motif_atoms_from_events  # noqa: E402
from scripts.utils.path_mine import crude_assoc, logistic_path_coef, path_indicator_matrix  # noqa: E402
from scripts.utils.path_protocol import (  # noqa: E402
    MAX_SELECTED_PATHS,
    MIN_AUPRC_GAIN,
    PATH_SEED,
    VAL_MIN_SUPPORT,
    parse_path_key,
    path_key,
    refuse_if_holdout_done,
    refuse_test_path,
    sha256_canonical,
)
from scripts.utils.shapelet_constants import BATCH_STAYS, SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_family import remap_activation_long  # noqa: E402
from scripts.utils.shapelet_prep import (  # noqa: E402
    build_bundle,
    fit_static_scaler,
    fit_value_standardizer,
    load_shapelet_tables,
)

DEFAULT = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"
CKPT = PROJ / "data/processed/shapelet/final/final_7var_L4_6_12_K3/ckpt_best.pt"
FROZEN_Q = PROJ / "data/processed/shapelet/events/events_final_7var_L4_6_12_K3/shapelet_frozen_qp50.pt"
MOTIF_DIR = PROJ / "data/processed/shapelet/events/events_final_7var_motif84"


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-dir", default=str(DEFAULT))
    p.add_argument("--report", default=str(PROJ / "outputs/reports/22_时序路径holdout复现.md"))
    return p.parse_args()


def _split_xy(stay_idx: pd.DataFrame, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([float(stay_idx.loc[s, "HOSPITAL_MORTALITY"]) for s in ids])
    X = np.stack(
        [np.array([float(stay_idx.loc[s, c]) for c in SHAPELET_STATIC_COLS], dtype=np.float64) for s in ids]
    )
    return y, X


def _write_freeze(d: Path, sel: dict, thr: dict) -> dict:
    payload = {
        "stage": "path_set_frozen_for_holdout",
        "selected": list(sel["selected"]),
        "seed": PATH_SEED,
        "min_gain": MIN_AUPRC_GAIN,
        "max_selected": MAX_SELECTED_PATHS,
        "screen": sel.get("screen"),
        "screen_top": sel.get("screen_top"),
        "lab_thresholds": thr,
        "val_used": "once",
        "holdout_used": False,
    }
    digest = sha256_canonical({k: payload[k] for k in ("selected", "seed", "min_gain", "screen", "lab_thresholds")})
    payload["freeze_sha256"] = digest
    payload["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dest = d / "path_freeze_manifest.json"
    if dest.is_file():
        old = json.loads(dest.read_text(encoding="utf-8"))
        if old.get("freeze_sha256") != digest:
            raise SystemExit("freeze hash mismatch vs existing path_freeze_manifest.json")
        return old
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _export_holdout_motif(grid, stays, hold_ids: list[int]) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"[26] export holdout events device={device} n={len(hold_ids)}")
    model, info = load_frozen_model(CKPT, device)
    blob = torch.load(FROZEN_Q, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state"])
    model.eval()
    model.freeze_templates(True)
    train_ids = stays.loc[stays["split"] == "train", "ICUSTAY_ID"].astype(int).tolist()
    value_scaler = fit_value_standardizer(grid, train_ids, vars_=info["vars"])
    static_scaler = fit_static_scaler(stays, train_ids, cols=SHAPELET_STATIC_COLS)
    bundle = build_bundle(
        grid,
        stays,
        value_scaler,
        static_scaler,
        vars_=info["vars"],
        static_cols=SHAPELET_STATIC_COLS,
        stay_ids=hold_ids,
        desc="prep holdout",
        show_progress=True,
    )
    raw = export_split_activations(
        model,
        bundle,
        device,
        BATCH_STAYS,
        info["activation_names"],
        split_label="holdout",
    )
    mapping = pd.read_parquet(MOTIF_DIR / "template_family_mapping.parquet")
    nodes = pd.read_parquet(MOTIF_DIR / "node_dictionary.parquet")
    remapped = remap_activation_long(raw, mapping, nodes, drop_test=False)
    winners = select_active_events(remapped, a_min=A_MIN_EVENT)
    winners = winners.copy()
    winners["is_active"] = True
    motif = motif_atoms_from_events(winners, a_min=A_MIN_EVENT)
    return motif


def main() -> None:
    args = parse_args()
    d = Path(args.paths_dir)
    refuse_test_path(d)
    refuse_test_path(args.report)
    out_json = d / "path_holdout_reproduce.json"
    refuse_if_holdout_done(out_json)

    sel = json.loads((d / "path_set_selected.json").read_text(encoding="utf-8"))
    thr = json.loads((d / "lab_thresholds.json").read_text(encoding="utf-8"))
    freeze = _write_freeze(d, sel, thr)
    _log(f"[26] freeze_sha256={freeze['freeze_sha256']}")
    paths = [parse_path_key(p) for p in sel["selected"]]

    splits = pd.read_parquet(PROJ / "data/processed/splits/stay_splits.parquet")
    hold_ids = sorted(int(x) for x in splits.loc[splits["split"] == "test", "ICUSTAY_ID"])
    train_ids = sorted(int(x) for x in splits.loc[splits["split"] == "train", "ICUSTAY_ID"])
    _log(f"[26] train={len(train_ids)} holdout={len(hold_ids)}")

    _log("[26] loading grid (train scalers + holdout labs)…")
    grid, stays = load_shapelet_tables(DATA_PROCESSED)
    stay_idx = stays.set_index("ICUSTAY_ID")

    motif = _export_holdout_motif(grid, stays, hold_ids)
    roster = stays.loc[stays["ICUSTAY_ID"].isin(hold_ids), ["ICUSTAY_ID", "SUBJECT_ID", "split", "HOSPITAL_MORTALITY"]].copy()
    roster = roster.drop_duplicates("ICUSTAY_ID")
    roster["split"] = "holdout"
    roster = roster.set_index("ICUSTAY_ID").loc[hold_ids].reset_index()
    raw = build_sparse_matrix_for_stays(grid, hold_ids, show_progress=True)
    lab = lab_atoms_from_raw(raw, roster, thr, names=sparse_feature_names())
    hold_atoms = pd.concat([motif, lab], ignore_index=True)
    hold_atoms.to_parquet(d / "holdout_atoms.parquet", index=False)
    _log(f"[26] holdout motif={len(motif)} lab={len(lab)}")

    train_atoms = pd.read_parquet(d / "atoms.parquet")
    train_atoms = train_atoms[train_atoms["split"] == "train"]
    y_tr, X_tr = _split_xy(stay_idx, train_ids)
    y_ho, X_ho = _split_xy(stay_idx, hold_ids)
    mu, sd = X_tr.mean(0), np.maximum(X_tr.std(0), 1e-6)
    X_tr = (X_tr - mu) / sd
    X_ho = (X_ho - mu) / sd
    tr_ind = path_indicator_matrix(train_ids, atoms_by_stay(train_atoms), paths)
    ho_ind = path_indicator_matrix(hold_ids, atoms_by_stay(hold_atoms), paths)

    rows = []
    for j, p in enumerate(paths):
        crude = crude_assoc(y_ho, ho_ind[:, j])
        coef, se, _ = logistic_path_coef(y_ho, ho_ind[:, j], X_ho)
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
                "holdout_support": int(ho_ind[:, j].sum()),
                "holdout_death_rate_path": crude["death_rate_path"],
                "holdout_death_rate_other": crude["death_rate_other"],
                "holdout_death_rate_diff": crude["death_rate_diff"],
                "holdout_odds_ratio": crude["odds_ratio"],
                "holdout_logit_coef": coef,
                "train_logit_coef": train_coef,
                "same_sign": bool(same),
                "holdout_support_ok": int(ho_ind[:, j].sum()) >= VAL_MIN_SUPPORT,
            }
        )

    def _auprc(Xtr, Xva, ytr, yva):
        if yva.min() == yva.max() or ytr.min() == ytr.max():
            return float("nan")
        clf = LogisticRegression(max_iter=300, solver="lbfgs")
        clf.fit(Xtr, ytr)
        return float(average_precision_score(yva, clf.predict_proba(Xva)[:, 1]))

    a_static = _auprc(X_tr, X_ho, y_tr, y_ho)
    a_paths = _auprc(np.column_stack([X_tr, tr_ind]), np.column_stack([X_ho, ho_ind]), y_tr, y_ho)
    n_same = int(sum(r["same_sign"] for r in rows))
    n_ok = int(sum(r["holdout_support_ok"] for r in rows))
    summary = {
        "stage": "path_holdout_reproduce_once",
        "holdout_used": "once",
        "freeze_sha256": freeze["freeze_sha256"],
        "n_holdout": len(hold_ids),
        "n_paths": len(paths),
        "n_same_sign": n_same,
        "n_holdout_support_ok": n_ok,
        "n_pass_both": int(sum(r["same_sign"] and r["holdout_support_ok"] for r in rows)),
        "holdout_auprc_static": a_static,
        "holdout_auprc_static_plus_paths": a_paths,
        "paths": rows,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# 时序路径 holdout 一次复现",
        "",
        "> 15 条名单与规则已哈希冻结。本文件是官方 test 划分的唯一一次读取；系数仍用 train 拟合。不得据此改阈值或重选。",
        "",
        f"- freeze `{freeze['freeze_sha256'][:16]}…`",
        f"- 选中 15 条。同号 {n_same}；支持≥{VAL_MIN_SUPPORT}：{n_ok}；两项都过：{summary['n_pass_both']}。",
        f"- holdout AUPRC 静态：{a_static:.4f}；静态+路径：{a_paths:.4f}。",
        "",
        "| 路径 | 支持 | 死亡率差 | holdout系数 | train系数 | 同号 | 支持过关 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['path']} | {r['holdout_support']} | {r['holdout_death_rate_diff']:.4f} | "
            f"{r['holdout_logit_coef']:.3f} | {r['train_logit_coef']:.3f} | "
            f"{'是' if r['same_sign'] else '否'} | {'是' if r['holdout_support_ok'] else '否'} |"
        )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(json.dumps({k: summary[k] for k in summary if k != "paths"}, ensure_ascii=False, indent=2))
    _log("[26] DONE (holdout used once)")


if __name__ == "__main__":
    main()
