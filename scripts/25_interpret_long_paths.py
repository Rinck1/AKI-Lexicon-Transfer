#!/usr/bin/env python3
"""Train-only nested interpretability for length-3/4 paths. Does not read val/test."""

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
from sklearn.model_selection import StratifiedGroupKFold

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.path_interpret import (  # noqa: E402
    SUPPORT_RATIO_MAX,
    atom_lowconf_map,
    contiguous_subpaths,
    grow_parent,
    indicator_for_paths,
    is_redundant,
    motif_lowconf_rate,
    path_narrative,
)
from scripts.utils.path_protocol import (  # noqa: E402
    LOW_CONF_ENRICH_TOL,
    MIN_AUPRC_GAIN,
    PATH_SEED,
    parse_path_key,
    path_key,
    refuse_test_path,
    refuse_test_split,
)
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402

DEFAULT = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-dir", default=str(DEFAULT))
    p.add_argument("--report", default=str(PROJ / "outputs/reports/21_时序路径3-4步可解释性.md"))
    return p.parse_args()


def _cv_scores(
    y: np.ndarray,
    groups: np.ndarray,
    static: np.ndarray,
    columns: list[np.ndarray],
    *,
    seed: int,
) -> list[float]:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    acc = [[] for _ in columns]
    for tr, va in cv.split(static, y, groups):
        if y[tr].min() == y[tr].max() or y[va].min() == y[va].max():
            continue
        for k, col in enumerate(columns):
            Xtr = np.column_stack([static[tr], col[tr]]) if col.size else static[tr]
            Xva = np.column_stack([static[va], col[va]]) if col.size else static[va]
            clf = LogisticRegression(max_iter=300, solver="lbfgs")
            clf.fit(Xtr, y[tr])
            acc[k].append(average_precision_score(y[va], clf.predict_proba(Xva)[:, 1]))
    return [float(np.mean(v)) if v else float("nan") for v in acc]


def main() -> None:
    args = parse_args()
    d = Path(args.paths_dir)
    refuse_test_path(d)
    refuse_test_path(args.report)
    stable = pd.read_parquet(d / "path_stable_train.parquet")
    cand = pd.read_parquet(d / "path_candidates_train.parquet")
    atoms = pd.read_parquet(d / "atoms.parquet")
    for sp in atoms["split"].astype(str).unique():
        refuse_test_split(sp)
    train_atoms = atoms[atoms["split"] == "train"]
    stay_map = atoms_by_stay(train_atoms)

    ev_idx = PROJ / "data/processed/shapelet/events/events_final_7var_motif84/stay_index.parquet"
    refuse_test_path(ev_idx)
    idx = pd.read_parquet(ev_idx)
    train_ids = sorted(int(x) for x in idx.loc[idx["split"] == "train", "ICUSTAY_ID"])
    stays = pd.read_parquet(Path(DATA_PROCESSED) / "stay_features_48h.parquet")
    stay_idx = stays.set_index("ICUSTAY_ID")
    y = np.array([float(stay_idx.loc[s, "HOSPITAL_MORTALITY"]) for s in train_ids])
    groups = np.array([int(stay_idx.loc[s, "SUBJECT_ID"]) for s in train_ids])
    static = np.stack(
        [np.array([float(stay_idx.loc[s, c]) for c in SHAPELET_STATIC_COLS], dtype=np.float64) for s in train_ids]
    )
    mu, sd = static.mean(0), np.maximum(static.std(0), 1e-6)
    static = (static - mu) / sd

    long_df = stable[stable["n_atoms"] >= 3].copy()
    support_of = {str(r.path): int(r.train_support) for r in cand.itertuples(index=False)}
    support_of.update({str(r.path): int(r.train_support) for r in stable.itertuples(index=False)})
    sign_of = {str(r.path): int(r.logit_sign) for r in stable.itertuples(index=False)}

    needed: list[tuple[str, ...]] = []
    specs = []
    for rec in long_df.itertuples(index=False):
        atoms_t = parse_path_key(rec.path)
        parent = grow_parent(atoms_t)
        subs = contiguous_subpaths(atoms_t)
        specs.append((atoms_t, parent, subs, rec))
        needed.extend([atoms_t, parent, *subs])
    uniq: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for p in needed:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    _log(f"[25] long={len(specs)} unique_indicators={len(uniq)}")
    ind = indicator_for_paths(train_ids, stay_map, uniq)
    empty = np.zeros((len(train_ids), 0))
    base = _cv_scores(y, groups, static, [empty], seed=PATH_SEED)[0]
    _log(f"[25] static CV AUPRC={base:.4f}")
    score_of: dict[tuple[str, ...], float] = {}
    for i, p in enumerate(uniq):
        score_of[p] = _cv_scores(y, groups, static, [ind[:, [i]]], seed=PATH_SEED)[0]
        if (i + 1) % 25 == 0 or i + 1 == len(uniq):
            _log(f"[25] unique CV {i+1}/{len(uniq)}")

    low_map = atom_lowconf_map(
        [
            (str(r.atom), bool(r.family_low_confidence), str(r.kind))
            for r in train_atoms.itertuples(index=False)
        ]
    )
    motif_train = train_atoms[train_atoms["kind"] == "motif"]
    baseline_low = float(motif_train["family_low_confidence"].mean()) if len(motif_train) else 0.0

    rows = []
    for atoms_t, parent, subs, rec in specs:
        short_keys: list[tuple[str, ...]] = []
        seen_s: set[tuple[str, ...]] = set()
        for s in [parent, *subs]:
            if s in seen_s:
                continue
            seen_s.add(s)
            short_keys.append(s)
        short_pairs = [(path_key(s), score_of[s]) for s in short_keys]
        best_name, best_short = max(short_pairs, key=lambda t: (t[1] if np.isfinite(t[1]) else -1.0))
        long_s = score_of[atoms_t]
        gain = long_s - best_short
        parent_key = path_key(parent)
        parent_supp = support_of.get(parent_key, 0)
        ratio = float(rec.train_support) / float(parent_supp) if parent_supp else float("nan")
        redundant = is_redundant(
            support_long=int(rec.train_support),
            support_short=int(parent_supp) if parent_supp else 0,
            auprc_gain=gain,
            min_gain=MIN_AUPRC_GAIN,
        )
        low_r = motif_lowconf_rate(atoms_t, low_map)
        independent = (not redundant) and bool(rec.readable)
        rows.append(
            {
                "path": rec.path,
                "n_atoms": int(rec.n_atoms),
                "train_support": int(rec.train_support),
                "parent": parent_key,
                "parent_support": int(parent_supp),
                "support_ratio": ratio,
                "best_short": best_name,
                "logit_coef": float(rec.logit_coef),
                "logit_sign": int(rec.logit_sign),
                "parent_sign": int(sign_of.get(parent_key, 0)),
                "same_sign_as_parent": int(rec.logit_sign) == int(sign_of.get(parent_key, 0))
                and int(rec.logit_sign) != 0,
                "cv_auprc_long": long_s,
                "cv_auprc_best_short": best_short,
                "cv_auprc_gain": gain,
                "redundant": redundant,
                "independent": independent,
                "readable": bool(rec.readable),
                "motif_lowconf_rate": low_r,
                "narrative": path_narrative(atoms_t, sign=int(rec.logit_sign)),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(["independent", "cv_auprc_gain", "n_atoms"], ascending=[False, False, True])
    out.to_parquet(d / "path_long_interpret_train.parquet", index=False)
    out.to_csv(d / "path_long_interpret_train.csv", index=False, encoding="utf-8-sig")
    indep = out[out["independent"]].copy()
    motif_atoms = []
    for p in indep["path"]:
        for a in parse_path_key(p):
            if not a.endswith("|升高") and not a.endswith("|下降"):
                motif_atoms.append(a)
    set_low = float(np.mean([low_map.get(a, 0.0) for a in motif_atoms])) if motif_atoms else 0.0
    low_ok = set_low <= baseline_low + LOW_CONF_ENRICH_TOL
    payload = {
        "stage": "path_long_interpret_train",
        "test_read": False,
        "val_used": False,
        "n_long_stable": int(len(out)),
        "n_independent": int(len(indep)),
        "n_independent_3": int((indep["n_atoms"] == 3).sum()) if len(indep) else 0,
        "n_independent_4": int((indep["n_atoms"] == 4).sum()) if len(indep) else 0,
        "static_cv_auprc": base,
        "min_gain": MIN_AUPRC_GAIN,
        "support_ratio_max": SUPPORT_RATIO_MAX,
        "baseline_motif_lowconf": baseline_low,
        "independent_motif_lowconf": set_low,
        "lowconf_ok": low_ok,
        "seed": PATH_SEED,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (d / "path_long_interpret.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# 时序路径 3–4 步 train 可解释性",
        "",
        "> 只在 train 上做嵌套检验。val 已在 2 步名单上用过一次，本文件 **不读 val/test**，不得把下列路径写成 val 复现过的「关键死亡路径」。",
        "",
        "## 判定规则",
        "",
        "- 对象：已稳定的 3/4 步路径。",
        "- 对照：其生长前缀与所有连续更短子路径中，train 五折 AUPRC 最高者。",
        f"- 独立可解释：相对最佳短路径 AUPRC 增益 ≥ {MIN_AUPRC_GAIN}，且支持 / 前缀支持 < {SUPPORT_RATIO_MAX}（否则视为同义加长）。",
        "- 短路径优先：未过关的 3/4 步只当 2 步的装饰，不进正文。",
        "",
        f"- 稳定 3/4 步：{payload['n_long_stable']}。独立可解释：{payload['n_independent']}（3 步 {payload['n_independent_3']}，4 步 {payload['n_independent_4']}）。",
        f"- 静态 CV AUPRC：{base:.4f}。",
        f"- 低置信：全体形态原子 {baseline_low:.3f}；独立集合 {set_low:.3f}；未富集：{'是' if low_ok else '否'}。",
        "",
    ]
    if indep.empty:
        lines += [
            "## 结论",
            "",
            "没有 3/4 步在 train 折内比自己的短前缀多出可分辨信息。正式解释单位仍是已冻结的 15 条 2 步路径。",
            "",
        ]
    else:
        lines += [
            "## 独立可解释路径（按增益）",
            "",
            "| 步数 | 支持 | 相对前缀 | 增益 | 符号 | 叙述 |",
            "|---:|---:|---:|---:|---|---|",
        ]
        for r in indep.itertuples(index=False):
            lines.append(
                f"| {r.n_atoms} | {r.train_support} | {r.support_ratio:.2f} | "
                f"{r.cv_auprc_gain:.4f} | {'↑' if r.logit_sign > 0 else '↓'} | {r.narrative} |"
            )
        lines += [
            "",
            "完整表：`path_long_interpret_train.csv`。这些路径可以写进方法/附录作 **train 稳定的加长轨迹**；正式群体复现留给 test 或外队列。",
            "",
        ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(json.dumps(payload, ensure_ascii=False, indent=2))
    _log("[25] DONE (train only, val not used)")


if __name__ == "__main__":
    main()
