#!/usr/bin/env python3
"""Frozen 15-path set: death-diff CIs, calibration, NRI/IDI, decision curve.

Does not reselect paths. Train-fit coefficients; val/holdout are reporting only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.path_mine import path_indicator_matrix  # noqa: E402
from scripts.utils.path_protocol import (  # noqa: E402
    PATH_SEED,
    parse_path_key,
    path_key,
    refuse_test_path,
)
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402

DEFAULT = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"
N_BOOT = 1000


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths-dir", default=str(DEFAULT))
    p.add_argument("--report", default=str(PROJ / "outputs/reports/26_路径区间校准与分层效用.md"))
    return p.parse_args()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return float(center - half), float(center + half)


def boot_diff(y: np.ndarray, ind: np.ndarray, rng: np.random.Generator, b: int = N_BOOT) -> tuple[float, float, float]:
    n = len(y)
    stats = []
    for _ in range(b):
        ix = rng.integers(0, n, n)
        yy, ii = y[ix], ind[ix].astype(bool)
        if ii.sum() == 0 or (~ii).sum() == 0:
            continue
        stats.append(float(yy[ii].mean() - yy[~ii].mean()))
    if not stats:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(stats)
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def fit_proba(Xtr, ytr, X) -> np.ndarray:
    clf = LogisticRegression(max_iter=400, solver="lbfgs")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(X)[:, 1]


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    rows = []
    for i in range(n_bins):
        m = (p > q[i]) & (p <= q[i + 1])
        if m.sum() == 0:
            continue
        rows.append(
            {
                "bin": i + 1,
                "n": int(m.sum()),
                "mean_pred": float(p[m].mean()),
                "mean_obs": float(y[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def nri_idi(y: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> dict:
    ev, ne = y > 0.5, y <= 0.5
    up_e = float((p1[ev] > p0[ev]).mean()) if ev.any() else float("nan")
    down_e = float((p1[ev] < p0[ev]).mean()) if ev.any() else float("nan")
    up_n = float((p1[ne] > p0[ne]).mean()) if ne.any() else float("nan")
    down_n = float((p1[ne] < p0[ne]).mean()) if ne.any() else float("nan")
    nri = (up_e - down_e) + (down_n - up_n)
    idi = float(p1[ev].mean() - p0[ev].mean()) - float(p1[ne].mean() - p0[ne].mean())
    return {
        "nri_continuous": nri,
        "nri_event": up_e - down_e,
        "nri_nonevent": down_n - up_n,
        "idi": idi,
        "frac_event_up": up_e,
        "frac_nonevent_down": down_n,
    }


def dca(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    n = len(y)
    prev = float(y.mean())
    out = []
    for t in thresholds:
        pred = p >= t
        tp = float(((pred) & (y > 0.5)).sum())
        fp = float(((pred) & (y <= 0.5)).sum())
        odds = t / max(1.0 - t, 1e-9)
        nb = tp / n - fp / n * odds
        nb_all = prev - (1.0 - prev) * odds
        out.append({"threshold": float(t), "nb_model": nb, "nb_all": nb_all, "nb_none": 0.0})
    return out


def xy_static(stay_idx, ids):
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
    paths = [parse_path_key(p) for p in sel["selected"]]
    splits = pd.read_parquet(PROJ / "data/processed/splits/stay_splits.parquet")
    train_ids = sorted(int(x) for x in splits.loc[splits["split"] == "train", "ICUSTAY_ID"])
    val_ids = sorted(int(x) for x in splits.loc[splits["split"] == "val", "ICUSTAY_ID"])
    ho_ids = sorted(int(x) for x in splits.loc[splits["split"] == "test", "ICUSTAY_ID"])
    stays = pd.read_parquet(Path(DATA_PROCESSED) / "stay_features_48h.parquet")
    stay_idx = stays.set_index("ICUSTAY_ID")
    atoms = pd.read_parquet(d / "atoms.parquet")
    ho_atoms = pd.read_parquet(d / "holdout_atoms.parquet")
    ytr, Xtr = xy_static(stay_idx, train_ids)
    yva, Xva = xy_static(stay_idx, val_ids)
    yho, Xho = xy_static(stay_idx, ho_ids)
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
    Xtr, Xva, Xho = (Xtr - mu) / sd, (Xva - mu) / sd, (Xho - mu) / sd
    _log("[31] indicators…")
    itr = path_indicator_matrix(train_ids, atoms_by_stay(atoms[atoms["split"] == "train"]), paths)
    iva = path_indicator_matrix(val_ids, atoms_by_stay(atoms[atoms["split"] == "val"]), paths)
    iho = path_indicator_matrix(ho_ids, atoms_by_stay(ho_atoms), paths)
    rng = np.random.default_rng(PATH_SEED)

    ci_rows = []
    for split, y, ind in (("val", yva, iva), ("holdout", yho, iho)):
        for j, p in enumerate(paths):
            ii = ind[:, j] > 0.5
            n1, k1 = int(ii.sum()), int((y[ii] > 0.5).sum())
            n0, k0 = int((~ii).sum()), int((y[~ii] > 0.5).sum())
            lo1, hi1 = wilson(k1, n1)
            lo0, hi0 = wilson(k0, n0)
            diff = (k1 / n1 - k0 / n0) if n1 and n0 else float("nan")
            _, dlo, dhi = boot_diff(y, ind[:, j], rng)
            ci_rows.append(
                {
                    "split": split,
                    "path": path_key(p),
                    "n_path": n1,
                    "deaths_path": k1,
                    "rate_path": (k1 / n1) if n1 else float("nan"),
                    "rate_path_lo": lo1,
                    "rate_path_hi": hi1,
                    "n_other": n0,
                    "rate_other": (k0 / n0) if n0 else float("nan"),
                    "rate_other_lo": lo0,
                    "rate_other_hi": hi0,
                    "rate_diff": diff,
                    "rate_diff_lo": dlo,
                    "rate_diff_hi": dhi,
                }
            )
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(d / "path_rate_intervals.csv", index=False, encoding="utf-8-sig")

    p0_va = fit_proba(Xtr, ytr, Xva)
    p1_va = fit_proba(np.column_stack([Xtr, itr]), ytr, np.column_stack([Xva, iva]))
    p0_ho = fit_proba(Xtr, ytr, Xho)
    p1_ho = fit_proba(np.column_stack([Xtr, itr]), ytr, np.column_stack([Xho, iho]))
    p0_tr = fit_proba(Xtr, ytr, Xtr)
    p1_tr = fit_proba(np.column_stack([Xtr, itr]), ytr, np.column_stack([Xtr, itr]))

    cal_parts = []
    for name, y, p0, p1 in (
        ("train_in_sample", ytr, p0_tr, p1_tr),
        ("val", yva, p0_va, p1_va),
        ("holdout", yho, p0_ho, p1_ho),
    ):
        a = calibration_table(y, p0)
        a["model"] = "static"
        a["split"] = name
        b = calibration_table(y, p1)
        b["model"] = "static_plus_paths"
        b["split"] = name
        cal_parts += [a, b]
    cal = pd.concat(cal_parts, ignore_index=True)
    cal.to_csv(d / "path_calibration_bins.csv", index=False, encoding="utf-8-sig")

    def pack(name, y, p0, p1):
        return {
            "split": name,
            "auprc_static": float(average_precision_score(y, p0)),
            "auprc_paths": float(average_precision_score(y, p1)),
            "auroc_static": float(roc_auc_score(y, p0)),
            "auroc_paths": float(roc_auc_score(y, p1)),
            "brier_static": float(brier_score_loss(y, p0)),
            "brier_paths": float(brier_score_loss(y, p1)),
            **nri_idi(y, p0, p1),
            "mean_p_static": float(p0.mean()),
            "mean_p_paths": float(p1.mean()),
        }

    metrics = [
        pack("train_in_sample", ytr, p0_tr, p1_tr),
        pack("val", yva, p0_va, p1_va),
        pack("holdout", yho, p0_ho, p1_ho),
    ]
    th = np.round(np.linspace(0.05, 0.40, 8), 3)
    dca_rows = []
    for name, y, p0, p1 in (("val", yva, p0_va, p1_va), ("holdout", yho, p0_ho, p1_ho)):
        for model, p in (("static", p0), ("static_plus_paths", p1)):
            for rec in dca(y, p, th):
                rec.update({"split": name, "model": model})
                dca_rows.append(rec)
    dca_df = pd.DataFrame(dca_rows)
    dca_df.to_csv(d / "path_decision_curve.csv", index=False, encoding="utf-8-sig")

    summary = {
        "stage": "path_uncertainty_utility",
        "reselected": False,
        "seed": PATH_SEED,
        "n_boot": N_BOOT,
        "metrics": metrics,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (d / "path_uncertainty_utility.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def fmt_m(m):
        return (
            f"- {m['split']}: AUPRC {m['auprc_static']:.3f}→{m['auprc_paths']:.3f}；"
            f"AUROC {m['auroc_static']:.3f}→{m['auroc_paths']:.3f}；"
            f"Brier {m['brier_static']:.3f}→{m['brier_paths']:.3f}；"
            f"连续 NRI {m['nri_continuous']:.3f}（事件 {m['nri_event']:.3f}，非事件 {m['nri_nonevent']:.3f}）；"
            f"IDI {m['idi']:.3f}。"
        )

    lines = [
        "# 路径集合：死亡率差区间、校准与分层效用",
        "",
        "> 15 条已冻结。本文件只补不确定性与效用，不重选。训练上的校准是样本内，偏乐观。",
        "",
        "## 模型比较",
        "",
    ]
    lines += [fmt_m(m) for m in metrics]
    lines += [
        "",
        "## 决策曲线",
        "",
        "阈值 0.05–0.40 的净收益见 `path_decision_curve.csv`。若某阈值上 static+paths 的 nb 不超过 static，该阈值不能声称改善分层。",
        "",
        "## 路径死亡率差（bootstrap 95% CI）",
        "",
        "| 划分 | 路径 | n | 路径死亡率 [Wilson] | 差 [boot] | 差的 CI 跨 0 |",
        "|---|---|---:|---|---|---|",
    ]
    for r in ci.itertuples(index=False):
        cross = "是" if (not np.isfinite(r.rate_diff_lo)) or r.rate_diff_lo * r.rate_diff_hi <= 0 else "否"
        lines.append(
            f"| {r.split} | {r.path} | {r.n_path} | "
            f"{r.rate_path:.3f} [{r.rate_path_lo:.3f},{r.rate_path_hi:.3f}] | "
            f"{r.rate_diff:.3f} [{r.rate_diff_lo:.3f},{r.rate_diff_hi:.3f}] | {cross} |"
        )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(json.dumps(metrics, indent=2))
    _log("[31] DONE")


if __name__ == "__main__":
    main()
