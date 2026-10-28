#!/usr/bin/env python3
"""
Experiment C — diagnostics on trained Shapelet checkpoints (no test tuning).

C1: val Brier + logistic calibration slope/intercept
C2: activation health (mean_a, by mortality)
C3: template S/α export + pairwise cosine
C4: early-stop / history summary from meta

Usage:
  python -u scripts/12_shapelet_diagnostics.py
  python -u scripts/12_shapelet_diagnostics.py --tags static_qzero,shapelet_qzero,combined_qzero
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.shapelet_constants import (  # noqa: E402
    BATCH_STAYS,
    SEED,
    SHAPELET_STATIC_COLS,
    SHAPELET_VARS,
)
from scripts.utils.shapelet_model import ShapeletRiskModel  # noqa: E402
from scripts.utils.shapelet_prep import (  # noqa: E402
    ShapeletStayDataset,
    prepare_splits,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


@torch.no_grad()
def collect_predictions(
    model: ShapeletRiskModel,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> dict[str, np.ndarray]:
    model.eval()
    ys, ps, acts = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch["x"].to(device)
        m = batch["m"].to(device)
        d = batch["delta"].to(device)
        s = batch["static"].to(device)
        y = batch["y"].to(device)
        out = model(x, m, d, s, y=None)
        ys.append(y.cpu().numpy())
        ps.append(out.prob.cpu().numpy())
        acts.append(out.activations.cpu().numpy())
    y = np.concatenate(ys).astype(np.float64)
    p = np.concatenate(ps).astype(np.float64)
    a = np.concatenate(acts).astype(np.float64)
    return {"y": y, "p": p, "a": a}


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> dict:
    """Logistic recalibration: logit(p) -> y; report slope & intercept."""
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    if y.min() == y.max():
        return {"slope": float("nan"), "intercept": float("nan")}
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    # sklearn >=1.2 uses penalty=None; older uses penalty='none'
    try:
        clf.fit(logit, y.astype(int))
    except Exception:
        clf = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
        clf.fit(logit, y.astype(int))
    return {
        "slope": float(clf.coef_.ravel()[0]),
        "intercept": float(clf.intercept_.ravel()[0]),
    }


def activation_stats(y: np.ndarray, a: np.ndarray, names: list[str]) -> dict:
    mean_a = a.mean(axis=0)
    dead = y >= 0.5
    alive = ~dead
    mean_dead = a[dead].mean(axis=0) if dead.any() else np.full(a.shape[1], np.nan)
    mean_alive = a[alive].mean(axis=0) if alive.any() else np.full(a.shape[1], np.nan)
    return {
        "mean_a_overall": float(a.mean()),
        "mean_a_dead": float(a[dead].mean()) if dead.any() else float("nan"),
        "mean_a_alive": float(a[alive].mean()) if alive.any() else float("nan"),
        "frac_a_lt_0_1": float((a < 0.1).mean()),
        "frac_a_gt_0_9": float((a > 0.9).mean()),
        "per_feature": [
            {
                "name": names[i] if i < len(names) else f"a{i}",
                "mean": float(mean_a[i]),
                "mean_dead": float(mean_dead[i]),
                "mean_alive": float(mean_alive[i]),
                "delta_dead_minus_alive": float(mean_dead[i] - mean_alive[i]),
            }
            for i in range(a.shape[1])
        ],
    }


def template_report(model: ShapeletRiskModel) -> dict:
    keys = model.template_keys()
    templates = []
    vecs = []
    for i, key in enumerate(keys):
        s = model.shapelets[i].detach().cpu().numpy()
        al = model.alpha_logits[i].detach().cpu()
        alpha = (al.shape[0] * torch.softmax(al, dim=-1)).numpy()
        vecs.append(s / (np.linalg.norm(s) + 1e-8))
        templates.append(
            {
                "key": list(key),
                "S": s.tolist(),
                "alpha": alpha.tolist(),
                "alpha_logits": al.numpy().tolist(),
                "q": float(model.q[i].detach().cpu()),
            }
        )
    cos = {}
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            cos[f"{keys[i]}__vs__{keys[j]}"] = float(np.dot(vecs[i], vecs[j]))
    return {"templates": templates, "pairwise_cosine": cos}


def history_summary(meta: dict) -> dict:
    hist = meta.get("history") or []
    if not hist:
        return {"n_epochs": 0}
    best = max(hist, key=lambda r: r.get("val_auprc", -1.0))
    auprcs = [r.get("val_auprc", float("nan")) for r in hist]
    # platform: last 5 epochs max-min
    tail = auprcs[-5:] if len(auprcs) >= 5 else auprcs
    return {
        "n_epochs": len(hist),
        "best_epoch": best.get("epoch"),
        "best_val_auprc": best.get("val_auprc"),
        "best_val_auroc": best.get("val_auroc"),
        "last_val_auprc": hist[-1].get("val_auprc"),
        "last_val_auroc": hist[-1].get("val_auroc"),
        "meta_best_metric": meta.get("best_metric"),
        "last5_auprc_range": float(np.nanmax(tail) - np.nanmin(tail)),
        "phases": sorted({r.get("phase", "phase1") for r in hist}),
    }


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[ShapeletRiskModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt.get("meta") or {}
    head_mode = ckpt.get("head_mode") or meta.get("head") or "combined"
    vars_ = tuple(meta.get("vars") or ["urine_output_ml"])
    lengths = tuple(meta.get("lengths") or [6])
    k = int(meta.get("k") or 2)
    n_static = int(meta.get("n_static") or len(SHAPELET_STATIC_COLS))
    model = ShapeletRiskModel(
        vars_=vars_,
        lengths=lengths,
        k=k,
        n_static=n_static,
        head_mode=head_mode,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, {**meta, "head_mode": head_mode, "activation_names": ckpt.get("activation_names")}


def diagnose_one(
    tag: str,
    ckpt_path: Path,
    packs: dict,
    device: torch.device,
    out_dir: Path,
) -> dict:
    _log(f"\n===== C diagnostics: {tag} =====")
    model, meta = load_model_from_ckpt(ckpt_path, device)
    names = meta.get("activation_names") or model.activation_names()

    train_loader = DataLoader(
        ShapeletStayDataset(packs["train"]),
        batch_size=BATCH_STAYS,
        shuffle=False,
    )
    val_loader = DataLoader(
        ShapeletStayDataset(packs["val"]),
        batch_size=BATCH_STAYS,
        shuffle=False,
    )

    # C4 history
    c4 = history_summary(meta)

    # C1 + C2 on val (and train mean_a)
    val = collect_predictions(model, val_loader, device, desc=f"val {tag}")
    train = collect_predictions(model, train_loader, device, desc=f"train {tag}")

    y_v, p_v, a_v = val["y"], val["p"], val["a"]
    c1 = {
        "n_val": int(len(y_v)),
        "prevalence": float(y_v.mean()),
        "auroc": float(roc_auc_score(y_v, p_v)) if y_v.min() != y_v.max() else float("nan"),
        "auprc": float(average_precision_score(y_v, p_v)) if y_v.min() != y_v.max() else float("nan"),
        "brier": float(brier_score_loss(y_v, p_v)),
        "calibration": calibration_slope_intercept(y_v, p_v),
        "p_mean": float(p_v.mean()),
        "p_std": float(p_v.std()),
    }

    c2 = {
        "val": activation_stats(y_v, a_v, names),
        "train": {
            "mean_a_overall": float(train["a"].mean()),
            "mean_a_dead": float(train["a"][train["y"] >= 0.5].mean()),
            "mean_a_alive": float(train["a"][train["y"] < 0.5].mean()),
            "frac_a_lt_0_1": float((train["a"] < 0.1).mean()),
            "frac_a_gt_0_9": float((train["a"] > 0.9).mean()),
        },
    }

    # C3 templates (skip meaningful S for static-only if unused)
    c3 = template_report(model)
    # save templates npz
    if model.head_mode != "static":
        npz_path = out_dir / f"templates_{tag}.npz"
        payload = {}
        for i, t in enumerate(c3["templates"]):
            payload[f"S_{i}"] = np.asarray(t["S"], dtype=np.float32)
            payload[f"alpha_{i}"] = np.asarray(t["alpha"], dtype=np.float32)
            payload[f"key_{i}"] = np.asarray(t["key"])
        payload["q"] = np.asarray([t["q"] for t in c3["templates"]], dtype=np.float32)
        np.savez(npz_path, **payload)
        c3["templates_npz"] = str(npz_path)

    report = {
        "tag": tag,
        "ckpt": str(ckpt_path),
        "head_mode": model.head_mode,
        "C1_probability_quality": c1,
        "C2_activation_health": c2,
        "C3_templates": {
            "pairwise_cosine": c3["pairwise_cosine"],
            "q": [t["q"] for t in c3["templates"]],
            "templates_npz": c3.get("templates_npz"),
            # keep short S for json readability
            "S_preview": [t["S"] for t in c3["templates"]],
            "alpha_preview": [t["alpha"] for t in c3["templates"]],
        },
        "C4_history": c4,
    }
    out_json = out_dir / f"diagnostics_{tag}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"  saved {out_json}")
    _log(
        f"  C1 AUPRC={c1['auprc']:.4f} AUROC={c1['auroc']:.4f} Brier={c1['brier']:.4f} "
        f"cal_slope={c1['calibration']['slope']:.3f} cal_int={c1['calibration']['intercept']:.3f}"
    )
    _log(
        f"  C2 val mean_a={c2['val']['mean_a_overall']:.4f} "
        f"dead={c2['val']['mean_a_dead']:.4f} alive={c2['val']['mean_a_alive']:.4f} "
        f"sat_hi={c2['val']['frac_a_gt_0_9']:.3f} sat_lo={c2['val']['frac_a_lt_0_1']:.3f}"
    )
    _log(f"  C3 cosine={c3['pairwise_cosine']}")
    _log(
        f"  C4 best_ep={c4.get('best_epoch')} best_auprc={c4.get('best_val_auprc')} "
        f"n_epochs={c4.get('n_epochs')} last5_range={c4.get('last5_auprc_range')}"
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Shapelet experiment C diagnostics")
    p.add_argument(
        "--tags",
        default="static_qzero,shapelet_qzero,combined_qzero",
        help="comma-separated tags matching checkpoint_debug_{tag}.pt",
    )
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--max-stays", type=int, default=None)
    args = p.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "cpu" else args.device)
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = PROJ / "data" / "processed" / "shapelet"
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    # data once (urine only for debug tags)
    _log("[C] loading splits…")
    packs = prepare_splits(
        vars_=("urine_output_ml",),
        static_cols=SHAPELET_STATIC_COLS,
        max_stays_per_split=args.max_stays,
        seed=SEED,
        show_progress=True,
        build_test=False,
    )

    all_reports = []
    for tag in tags:
        ckpt = out_dir / f"checkpoint_debug_{tag}.pt"
        if not ckpt.is_file():
            _log(f"[skip] missing {ckpt}")
            continue
        all_reports.append(diagnose_one(tag, ckpt, packs, device, out_dir))

    summary = {
        "experiment": "C",
        "device": str(device),
        "tags": tags,
        "table": [
            {
                "tag": r["tag"],
                "head": r["head_mode"],
                "auprc": r["C1_probability_quality"]["auprc"],
                "auroc": r["C1_probability_quality"]["auroc"],
                "brier": r["C1_probability_quality"]["brier"],
                "cal_slope": r["C1_probability_quality"]["calibration"]["slope"],
                "cal_intercept": r["C1_probability_quality"]["calibration"]["intercept"],
                "mean_a_val": r["C2_activation_health"]["val"]["mean_a_overall"],
                "mean_a_dead": r["C2_activation_health"]["val"]["mean_a_dead"],
                "mean_a_alive": r["C2_activation_health"]["val"]["mean_a_alive"],
                "cosine": r["C3_templates"]["pairwise_cosine"],
                "best_epoch": r["C4_history"].get("best_epoch"),
                "n_epochs": r["C4_history"].get("n_epochs"),
            }
            for r in all_reports
        ],
    }
    sum_path = out_dir / "diagnostics_C_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"\n[C] summary -> {sum_path}")
    _log("\n| tag | AUPRC | AUROC | Brier | cal_slope | mean_a | a_dead-a_alive |")
    _log("|-----|-------|-------|-------|-----------|--------|----------------|")
    for row in summary["table"]:
        delta = row["mean_a_dead"] - row["mean_a_alive"]
        _log(
            f"| {row['tag']} | {row['auprc']:.4f} | {row['auroc']:.4f} | "
            f"{row['brier']:.4f} | {row['cal_slope']:.3f} | {row['mean_a_val']:.4f} | "
            f"{delta:+.4f} |"
        )


if __name__ == "__main__":
    main()
