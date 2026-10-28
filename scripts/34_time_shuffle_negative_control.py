#!/usr/bin/env python3
"""Train-only time-shuffle negative control for stage-ordered paths (docs/20).

Within each train stay, permute stage slots across existing atoms and rewrite
atom names. Concept bags and stage occupancy stay; order does not.

Does not retune the frozen 15-path set. Remine + forward-select live in a
separate directory. Val / holdout / test are not read for selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.path_mine import path_indicator_matrix  # noqa: E402
from scripts.utils.path_protocol import (  # noqa: E402
    PATH_SEED,
    parse_path_key,
    refuse_test_path,
    refuse_test_split,
    sha256_canonical,
)
from scripts.utils.path_shuffle import (  # noqa: E402
    cooccur_indicator_matrix,
    identities_by_stay,
    identity_bag_signature,
    n_stage_concept_pairs_changed,
    path_identities,
    shuffle_train_atom_stages,
    stage_occupancy_signature,
)
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402

DEFAULT_SRC = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"
DEFAULT_OUT = PROJ / "data/processed/paths/path_time_shuffle_train_seed42"
REPORT = PROJ / "outputs/reports/34_时间打乱负对照.md"
TABLE = PROJ / "outputs/tables/time_shuffle_negative_control.json"


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=PATH_SEED)
    p.add_argument("--n-shuffle-eval", type=int, default=5)
    p.add_argument("--skip-remine", action="store_true")
    return p.parse_args()


def _cv_auprc(
    y: np.ndarray,
    groups: np.ndarray,
    static: np.ndarray,
    path_ind: np.ndarray,
    *,
    seed: int,
) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, va in cv.split(static, y, groups):
        if y[tr].min() == y[tr].max() or y[va].min() == y[va].max():
            continue
        Xtr = np.column_stack([static[tr], path_ind[tr]]) if path_ind.shape[1] else static[tr]
        Xva = np.column_stack([static[va], path_ind[va]]) if path_ind.shape[1] else static[va]
        clf = LogisticRegression(max_iter=300, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        scores.append(average_precision_score(y[va], clf.predict_proba(Xva)[:, 1]))
    return float(np.mean(scores)) if scores else float("nan")


def _load_train_design(train_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _grid, stays = load_shapelet_tables(DATA_PROCESSED)
    stay_idx = stays.set_index("ICUSTAY_ID")
    y = np.array([float(stay_idx.loc[s, "HOSPITAL_MORTALITY"]) for s in train_ids])
    groups = np.array([int(stay_idx.loc[s, "SUBJECT_ID"]) for s in train_ids])
    static = np.stack(
        [np.array([float(stay_idx.loc[s, c]) for c in SHAPELET_STATIC_COLS], dtype=np.float64) for s in train_ids]
    )
    mu, sd = static.mean(0), np.maximum(static.std(0), 1e-6)
    return y, groups, (static - mu) / sd


def _run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(PROJ / "scripts" / script), *extra]
    _log("[34] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJ))


def _mean_support(ind: np.ndarray) -> list[float]:
    if ind.size == 0:
        return []
    return [float(ind[:, j].mean()) for j in range(ind.shape[1])]


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)
    refuse_test_path(src)
    refuse_test_path(out)
    out.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((src / "path_set_selected.json").read_text(encoding="utf-8"))
    frozen_keys: list[str] = list(frozen["selected"])
    frozen_paths = [parse_path_key(k) for k in frozen_keys]
    orig_atoms_all = pd.read_parquet(src / "atoms.parquet")
    for sp in orig_atoms_all["split"].astype(str).unique():
        refuse_test_split(sp)
    orig_train = orig_atoms_all[orig_atoms_all["split"].astype(str) == "train"].copy()

    ev_idx = PROJ / "data/processed/shapelet/events/events_final_7var_motif84/stay_index.parquet"
    refuse_test_path(ev_idx)
    idx = pd.read_parquet(ev_idx)
    train_ids = sorted(int(x) for x in idx.loc[idx["split"] == "train", "ICUSTAY_ID"])
    y, groups, static = _load_train_design(train_ids)
    _log(f"[34] train stays={len(train_ids)} train_atoms={len(orig_train)} frozen={len(frozen_paths)}")

    orig_stay = atoms_by_stay(orig_train)
    orig_ind = path_indicator_matrix(train_ids, orig_stay, frozen_paths)
    orig_idents = identities_by_stay(orig_train)
    co_ind = cooccur_indicator_matrix(train_ids, orig_idents, frozen_paths)

    static_cv = _cv_auprc(y, groups, static, np.zeros((len(train_ids), 0)), seed=PATH_SEED)
    frozen_orig_cv = _cv_auprc(y, groups, static, orig_ind, seed=PATH_SEED)
    cooccur_cv = _cv_auprc(y, groups, static, co_ind, seed=PATH_SEED)
    _log(f"[34] static CV AUPRC={static_cv:.4f}")
    _log(f"[34] frozen-15 on original atoms CV AUPRC={frozen_orig_cv:.4f}")
    _log(f"[34] stage-stripped co-occurrence of same 15 CV AUPRC={cooccur_cv:.4f}")

    sh_main = shuffle_train_atom_stages(orig_train, seed=int(args.seed))
    assert stage_occupancy_signature(orig_train).equals(stage_occupancy_signature(sh_main))
    assert identity_bag_signature(orig_train).equals(identity_bag_signature(sh_main))
    n_changed = n_stage_concept_pairs_changed(orig_train, sh_main)
    sh_main.to_parquet(out / "atoms.parquet", index=False)
    _log(f"[34] wrote shuffled train atoms seed={args.seed} stays_changed={n_changed}/{sh_main['ICUSTAY_ID'].nunique()}")

    shuffle_rows = []
    for k in range(int(args.n_shuffle_eval)):
        seed_k = int(args.seed) + k
        sh = sh_main if k == 0 else shuffle_train_atom_stages(orig_train, seed=seed_k)
        sh_stay = atoms_by_stay(sh)
        sh_ind = path_indicator_matrix(train_ids, sh_stay, frozen_paths)
        cv = _cv_auprc(y, groups, static, sh_ind, seed=PATH_SEED)
        rec = {
            "seed": seed_k,
            "frozen15_cv_auprc": cv,
            "mean_path_prevalence": float(np.mean(_mean_support(sh_ind))) if sh_ind.size else 0.0,
            "n_paths_with_support": int((sh_ind.sum(0) >= 39).sum()) if sh_ind.size else 0,
            "stays_with_any_frozen_path": float((sh_ind.sum(1) > 0).mean()) if sh_ind.size else 0.0,
        }
        shuffle_rows.append(rec)
        _log(
            f"[34] shuffle seed={seed_k} frozen-15 CV={cv:.4f} "
            f"mean_prev={rec['mean_path_prevalence']:.4f} "
            f"n_ge39={rec['n_paths_with_support']}"
        )

    sh_cvs = np.array([r["frozen15_cv_auprc"] for r in shuffle_rows], dtype=np.float64)
    orig_prev = _mean_support(orig_ind)
    sh0_stay = atoms_by_stay(sh_main)
    sh0_ind = path_indicator_matrix(train_ids, sh0_stay, frozen_paths)
    sh0_prev = _mean_support(sh0_ind)

    remine: dict = {"ran": False}
    if not args.skip_remine:
        _run("22_mine_temporal_paths.py", ["--atoms-dir", str(out), "--out", str(out)])
        _run(
            "22_mine_temporal_paths.py",
            ["--atoms-dir", str(out), "--out", str(out), "--extend"],
        )
        _run("23_select_path_set.py", ["--paths-dir", str(out)])
        remine_sel = json.loads((out / "path_set_selected.json").read_text(encoding="utf-8"))
        remine_man = json.loads((out / "mine_manifest.json").read_text(encoding="utf-8"))
        orig_man = json.loads((src / "mine_manifest.json").read_text(encoding="utf-8"))
        new_keys = list(remine_sel.get("selected", []))
        overlap = set(frozen_keys) & set(new_keys)
        orig_idents_paths = {path_identities(parse_path_key(k)) for k in frozen_keys}
        new_idents_paths = {path_identities(parse_path_key(k)) for k in new_keys}
        orig_bags = {frozenset(path_identities(parse_path_key(k))) for k in frozen_keys}
        new_bags = {frozenset(path_identities(parse_path_key(k))) for k in new_keys}
        remine = {
            "ran": True,
            "n_frequent": int(remine_man.get("n_frequent", 0)),
            "n_stable": int(remine_man.get("n_stable", 0)),
            "n_stable_readable": int(remine_man.get("n_stable_readable", 0)),
            "n_stable_by_len": remine_man.get("n_stable_by_len", {}),
            "orig_n_frequent": int(orig_man.get("n_frequent", 0)),
            "orig_n_stable": int(orig_man.get("n_stable", 0)),
            "orig_n_stable_readable": int(orig_man.get("n_stable_readable", 0)),
            "static_cv_auprc": remine_sel.get("static_cv_auprc"),
            "selected_cv_auprc": remine_sel.get("selected_cv_auprc"),
            "n_selected": int(remine_sel.get("n_selected", 0)),
            "selected": new_keys,
            "exact_overlap_with_frozen15": sorted(overlap),
            "n_exact_overlap": len(overlap),
            "n_identity_pair_overlap": len(orig_idents_paths & new_idents_paths),
            "n_identity_bag_overlap": len(orig_bags & new_bags),
            "identity_bag_overlap": [sorted(x) for x in orig_bags & new_bags],
        }
        _log(
            f"[34] remine selected={remine['n_selected']} "
            f"CV={remine['selected_cv_auprc']} "
            f"exact_overlap={remine['n_exact_overlap']} "
            f"identity_overlap={remine['n_identity_pair_overlap']}"
        )

    payload = {
        "stage": "time_shuffle_negative_control",
        "test_read": False,
        "val_used": False,
        "holdout_used": False,
        "frozen_paths_retuned": False,
        "seed": int(args.seed),
        "n_train": len(train_ids),
        "n_train_atoms": int(len(orig_train)),
        "n_stays_stage_concept_changed": int(n_changed),
        "static_cv_auprc": static_cv,
        "frozen15_original_cv_auprc": frozen_orig_cv,
        "frozen15_original_selected_cv_auprc_from_23": frozen.get("selected_cv_auprc"),
        "cooccur_stage_stripped_cv_auprc": cooccur_cv,
        "frozen15_shuffled_cv_auprc_mean": float(np.nanmean(sh_cvs)),
        "frozen15_shuffled_cv_auprc_std": float(np.nanstd(sh_cvs)),
        "frozen15_shuffled_cv_by_seed": shuffle_rows,
        "frozen15_original_prevalence": orig_prev,
        "frozen15_shuffled_seed_main_prevalence": sh0_prev,
        "delta_frozen15_cv_vs_original": float(np.nanmean(sh_cvs) - frozen_orig_cv),
        "delta_frozen15_cv_vs_static": float(np.nanmean(sh_cvs) - static_cv),
        "delta_cooccur_vs_frozen_original": float(cooccur_cv - frozen_orig_cv),
        "frozen_keys": frozen_keys,
        "remine": remine,
        "protocol": {
            "shuffle": "within-stay permutation of stage_idx; atom names rewritten",
            "preserves": ["stage occupancy per stay", "stage-stripped concept bag"],
            "destroys": ["which concept sits in which stage"],
            "min_support": 39,
            "forward_select": "same as 23; separate directory",
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "payload_sha256": "",
    }
    payload["payload_sha256"] = sha256_canonical({k: v for k, v in payload.items() if k != "payload_sha256"})
    TABLE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "time_shuffle_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    REPORT.write_text(_render_report(payload), encoding="utf-8")
    _log(f"[34] wrote {TABLE}")
    _log(f"[34] wrote {REPORT}")
    _log("[34] DONE")


def _render_report(p: dict) -> str:
    rem = p.get("remine") or {}
    remine_block = "_未跑（--skip-remine）。_\n"
    if rem.get("ran"):
        sel = rem.get("selected") or []
        lines = "\n".join(f"- `{s}`" for s in sel)
        remine_block = (
            f"| 量 | 原始 train | 打乱后重挖 |\n"
            f"|---|---:|---:|\n"
            f"| frequent | {rem.get('orig_n_frequent')} | {rem.get('n_frequent')} |\n"
            f"| stable | {rem.get('orig_n_stable')} | {rem.get('n_stable')} |\n"
            f"| stable readable | {rem.get('orig_n_stable_readable')} | {rem.get('n_stable_readable')} |\n"
            f"| 前向选择 CV AUPRC | {p['frozen15_original_selected_cv_auprc_from_23']:.4f} | "
            f"{float(rem.get('selected_cv_auprc') or float('nan')):.4f} |\n"
            f"| 与冻结 15 条完全重合 | — | {rem.get('n_exact_overlap')} |\n"
            f"| 去阶段后有序对重合 | — | {rem.get('n_identity_pair_overlap')} |\n"
            f"| 去阶段后无序袋子重合 | — | {rem.get('n_identity_bag_overlap')} |\n\n"
            f"打乱后新选出的路径：\n\n{lines}\n"
        )
    sh_lines = "\n".join(
        f"| {r['seed']} | {r['frozen15_cv_auprc']:.4f} | {r['mean_path_prevalence']:.4f} | "
        f"{r['n_paths_with_support']} |"
        for r in p["frozen15_shuffled_cv_by_seed"]
    )
    return f"""# 34 — 时间打乱负对照（仅 train）

> 每个 train stay：**保留出现过的概念、保留各阶段事件个数**，随机重排「哪个概念落在哪个阶段」，并重写原子名。
> **不**重选冻结的 15 条；重挖写在独立目录。val / holdout / test 未用于选择。

## 1. 打乱是否按设计

| 项 | 值 |
|---|---:|
| train stays | {p['n_train']} |
| train 原子 | {p['n_train_atoms']} |
| 主种子 | {p['seed']} |
| 阶段占用签名被破坏的 stay 数 | 0（协议要求保持） |
| 概念袋子被破坏的 stay 数 | 0（协议要求保持） |
| 至少一个 (概念, 阶段) 对改变的 stay | {p['n_stays_stage_concept_changed']} |

## 2. 冻结 15 条：顺序 vs 共现

同一套静态协变量、同一五折分组。

| 模型 | train 五折 AUPRC |
|---|---:|
| 仅静态 | {p['static_cv_auprc']:.4f} |
| 静态 + 冻结 15 条（原始阶段序） | {p['frozen15_original_cv_auprc']:.4f} |
| 静态 + 同一 15 对的无序共现（去掉阶段） | {p['cooccur_stage_stripped_cv_auprc']:.4f} |
| 静态 + 冻结 15 条（打乱后的原子，{len(p['frozen15_shuffled_cv_by_seed'])} 个种子均值） | {p['frozen15_shuffled_cv_auprc_mean']:.4f} ± {p['frozen15_shuffled_cv_auprc_std']:.4f} |

- 打乱后冻结 15 条相对原始：Δ = {p['delta_frozen15_cv_vs_original']:+.4f}
- 打乱后冻结 15 条相对仅静态：Δ = {p['delta_frozen15_cv_vs_static']:+.4f}
- 无序共现相对原始有序 15 条：Δ = {p['delta_cooccur_vs_frozen_original']:+.4f}

### 各打乱种子（冻结 15 条原样去匹配）

| seed | CV AUPRC | 路径平均患病率 | 支持≥39 的条数 |
|---|---:|---:|---:|
{sh_lines}

怎么读：

- 若打乱后冻结 15 条掉到接近静态，而无序共现仍接近原始 15 条，则 **15 条里的箭头（阶段位置）在干活**。
- 若无序共现已经吃掉几乎全部增益，则挖掘到的主要是 **谁和谁一起出现**，不是先后。
- 打乱后仍能用**同一协议**挖出接近原始增益的新 15 条，说明顺序标签可被别的配对替代，不能把具体箭头当稳定轨迹。

## 3. 打乱后按原协议重挖（不覆盖冻结集）

{remine_block}

## 4. 冻结声明

- 原 15 条路径、lab 阈值、家族映射：**未改**。
- 本对照只写 `data/processed/paths/path_time_shuffle_train_seed42/` 与本报告。
- 未读 holdout/test；val 未用于打乱或选择。
"""


if __name__ == "__main__":
    main()
