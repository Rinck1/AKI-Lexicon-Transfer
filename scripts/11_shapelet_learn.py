#!/usr/bin/env python3
"""
11 — Shapelet learning CLI (G1 skeleton).

Implements training loop structure per docs/12 §9 / §9B.
Does **not** run full patient training unless --train is passed.

Usage:
  # Safe default: synthetic one-step fwd+bwd (no patient I/O)
  python scripts/11_shapelet_learn.py --dry-run

  # Opt-in real training (reads hourly_grid; still respects --mode)
  python scripts/11_shapelet_learn.py --train --mode debug
  python scripts/11_shapelet_learn.py --train --mode pilot

  # Tiny real subsample smoke (not paper run)
  python scripts/11_shapelet_learn.py --train --mode debug --max-stays 32 --max-epochs 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.shapelet_constants import (  # noqa: E402
    BATCH_STAYS,
    DEFAULT_RUN_MODE,
    GRAD_ACCUM,
    GRAD_CLIP,
    LR_ALPHA,
    LR_HEAD,
    LR_SHAPELET,
    MAX_EPOCHS,
    MONITOR,
    PATIENCE,
    SEED,
    SHAPELET_STATIC_COLS,
    SHAPELET_VARS,
    WEIGHT_DECAY,
    lengths_and_k,
)
from scripts.utils.shapelet_model import ShapeletRiskModel  # noqa: E402
from scripts.utils.shapelet_prep import (  # noqa: E402
    ShapeletStayDataset,
    make_synthetic_batch,
    prepare_splits,
    train_pos_weight,
)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_vars(mode: str, vars_arg: str | None) -> tuple[str, ...]:
    if vars_arg:
        chosen = tuple(v.strip() for v in vars_arg.split(",") if v.strip())
        unknown = [v for v in chosen if v not in SHAPELET_VARS]
        if unknown:
            raise ValueError(f"unknown vars {unknown}; allowed {SHAPELET_VARS}")
        return chosen
    if mode == "debug":
        return ("urine_output_ml",)
    return tuple(SHAPELET_VARS)


def _log(msg: str) -> None:
    """Always flush so batch logs show immediately in terminals / redirected files."""
    print(msg, flush=True)


def run_dry_run(device: torch.device, mode: str, vars_: tuple[str, ...]) -> dict:
    """G1 gate: production model path on synthetic batch only."""
    lengths, k = lengths_and_k(mode)
    n_vars = len(vars_)
    model = ShapeletRiskModel(
        vars_=vars_,
        lengths=lengths,
        k=k,
        n_static=len(SHAPELET_STATIC_COLS),
        pos_weight=5.887,
    ).to(device)
    model.train()
    batch = make_synthetic_batch(
        batch_size=BATCH_STAYS,
        n_vars=n_vars,
        n_static=len(SHAPELET_STATIC_COLS),
        device=device,
        seed=SEED,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(
        batch["x"],
        batch["m"],
        batch["delta"],
        batch["static"],
        batch["y"],
    )
    out.loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_mb = (
        round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        if device.type == "cuda"
        else None
    )
    s_grad = bool(torch.isfinite(model.shapelets[0].grad).all())
    a_grad = bool(torch.isfinite(model.alpha_logits[0].grad).all())
    result = {
        "mode": mode,
        "vars": list(vars_),
        "lengths": list(lengths),
        "k": k,
        "n_templates": model.n_templates,
        "n_act": model.n_act,
        "device": str(device),
        "fwd_bwd_s": round(dt, 3),
        "peak_MB": peak_mb,
        "loss": float(out.loss.detach().cpu()),
        "loss_bce": float(out.loss_bce.detach().cpu()),
        "shape_grad_finite": s_grad,
        "alpha_grad_finite": a_grad,
        "prob_mean": float(out.prob.detach().mean().cpu()),
    }
    _log("[dry-run] shapelet G1 skeleton OK")
    for key, val in result.items():
        _log(f"  {key}: {val}")
    return result


@torch.no_grad()
def evaluate_auprc_auroc(
    model: ShapeletRiskModel,
    loader: DataLoader,
    device: torch.device,
    desc: str = "val",
    show_progress: bool = True,
):
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    ys, ps = [], []
    it = loader
    if show_progress:
        it = tqdm(loader, desc=desc, leave=False, unit="batch")
    for batch in it:
        x = batch["x"].to(device)
        m = batch["m"].to(device)
        d = batch["delta"].to(device)
        s = batch["static"].to(device)
        y = batch["y"].to(device)
        out = model(x, m, d, s, y=None)
        ys.append(y.cpu().numpy())
        ps.append(out.prob.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    if y.min() == y.max():
        return {"auroc": float("nan"), "auprc": float("nan"), "n": int(len(y))}
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "n": int(len(y)),
    }


@torch.no_grad()
def collect_train_stage_distances(
    model: ShapeletRiskModel,
    loader: DataLoader,
    device: torch.device,
    max_per_template: int = 80_000,
) -> torch.Tensor:
    """Gather stage softmin distances d_b per template on train (for q).

    Empty-stage barriers (~1e4) are dropped. Returns (n_templates, n_samples)
    with equalized length via truncation to the shortest template count
    (capped by max_per_template).
    """
    if model.head_mode == "static":
        raise RuntimeError("distance collection not used for static head")
    model.eval()
    buckets: list[list[torch.Tensor]] = [[] for _ in range(model.n_templates)]
    var_index = {v: i for i, v in enumerate(model.vars)}
    for batch in tqdm(loader, desc="collect train d for q", leave=True, unit="batch"):
        x = batch["x"].to(device)
        m = batch["m"].to(device)
        dlt = batch["delta"].to(device)
        for ti, (v, _L, _k) in enumerate(model.template_keys()):
            j = var_index[v]
            d_stage, _t_star, _dist = model.match_one_template(
                x[:, :, j], m[:, :, j], dlt[:, :, j], ti
            )
            flat = d_stage.reshape(-1)
            # drop empty-stage barrier values from stage_softmin_pool
            keep = flat < 1.0e3
            if keep.any():
                buckets[ti].append(flat[keep].detach().cpu())
    rows = []
    for ti, parts in enumerate(buckets):
        if not parts:
            rows.append(torch.zeros(1))
            continue
        cat = torch.cat(parts, dim=0)
        if cat.numel() > max_per_template:
            # deterministic subsample for reproducibility
            idx = torch.linspace(0, cat.numel() - 1, steps=max_per_template).long()
            cat = cat[idx]
        rows.append(cat)
    n_min = min(r.numel() for r in rows)
    d_flat = torch.stack([r[:n_min] for r in rows], dim=0)
    return d_flat


def run_epoch_loop(
    model: ShapeletRiskModel,
    opt: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    max_epochs: int,
    patience: int,
    min_delta: float,
    accum: int,
    log_every: int,
    phase_name: str,
    history: list,
    epoch_offset: int = 0,
) -> tuple[float, dict | None, list]:
    """Train up to max_epochs with early stop; append to history; return best."""
    best_metric = -1.0
    best_state = None
    bad_epochs = 0
    n_batches = len(train_loader)

    for local_ep in range(1, max_epochs + 1):
        epoch = epoch_offset + local_ep
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        n_steps = 0
        t_epoch = time.perf_counter()
        pbar = tqdm(
            train_loader,
            desc=f"{phase_name} ep{epoch:03d}",
            total=n_batches,
            unit="batch",
            leave=True,
        )
        for step, batch in enumerate(pbar, start=1):
            t0 = time.perf_counter()
            x = batch["x"].to(device)
            m = batch["m"].to(device)
            d = batch["delta"].to(device)
            s = batch["static"].to(device)
            y = batch["y"].to(device)
            out = model(x, m, d, s, y)
            loss = out.loss / accum
            loss.backward()
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                opt.zero_grad(set_to_none=True)
            batch_loss = float(out.loss.detach().cpu())
            running += batch_loss
            n_steps += 1
            avg = running / n_steps
            pbar.set_postfix(
                loss=f"{batch_loss:.4f}",
                avg=f"{avg:.4f}",
                bce=f"{float(out.loss_bce.detach().cpu()):.3f}",
                refresh=False,
            )
            if log_every > 0 and (step % log_every == 0 or step == n_batches):
                dt = time.perf_counter() - t0
                _log(
                    f"  [{phase_name}] epoch {epoch:03d}  batch {step:04d}/{n_batches:04d}  "
                    f"loss={batch_loss:.4f}  avg={avg:.4f}  batch_s={dt:.2f}"
                )
        if n_steps % accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)

        val_m = evaluate_auprc_auroc(
            model,
            val_loader,
            device,
            desc=f"val {phase_name} ep{epoch:03d}",
            show_progress=True,
        )
        train_loss = running / max(n_steps, 1)
        rec = {
            "epoch": epoch,
            "phase": phase_name,
            "train_loss": train_loss,
            "val_auroc": val_m["auroc"],
            "val_auprc": val_m["auprc"],
            "q": [float(v) for v in model.q.detach().cpu().tolist()],
        }
        history.append(rec)
        metric = val_m["auprc"] if MONITOR == "val_auprc" else val_m["auroc"]
        epoch_s = time.perf_counter() - t_epoch
        _log(
            f"  [{phase_name}] epoch {epoch:03d}  DONE  loss={train_loss:.4f}  "
            f"val_AUPRC={val_m['auprc']:.4f}  val_AUROC={val_m['auroc']:.4f}  "
            f"epoch_s={epoch_s:.1f}"
        )
        if metric == metric and metric > best_metric + min_delta:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                _log(
                    f"  [{phase_name}] early stop at epoch {epoch} "
                    f"(patience={patience}, min_delta={min_delta})"
                )
                break

    return best_metric, best_state, history


def run_train(args: argparse.Namespace, device: torch.device, vars_: tuple[str, ...]) -> None:
    lengths, k = lengths_and_k(args.mode)
    head_mode = str(args.head).lower()
    _log(
        f"[train] mode={args.mode} head={head_mode} L={lengths} K={k} "
        f"vars={vars_} device={device}"
    )
    _log("[train] loading splits (train-only scalers)…")
    packs = prepare_splits(
        vars_=vars_,
        static_cols=SHAPELET_STATIC_COLS,
        max_stays_per_split=args.max_stays,
        seed=SEED,
        show_progress=True,
        build_test=bool(args.eval_test),
    )
    train_b = packs["train"]
    val_b = packs["val"]
    pos_w = train_pos_weight(train_b.y)
    _log(f"  train N={len(train_b.y)} val N={len(val_b.y)} pos_weight={pos_w:.3f}")
    if head_mode == "static":
        _log("  ablation=static: Soft-DTW skipped; only θ·C + bias")
    elif head_mode == "shapelet":
        _log("  ablation=shapelet: activations only; static θ disabled")
    else:
        _log("  ablation=combined: Shapelet activations + 11 static covariates")

    q_policy = str(args.q_policy).lower()
    q_quantile = float(args.q_quantile)
    if head_mode == "static" and q_policy != "zero":
        _log("  [warn] static head ignores q-policy (no Soft-DTW); forcing zero")
        q_policy = "zero"

    model = ShapeletRiskModel(
        vars_=vars_,
        lengths=lengths,
        k=k,
        n_static=len(SHAPELET_STATIC_COLS),
        pos_weight=pos_w,
        head_mode=head_mode,
        q_init=0.0,
    ).to(device)

    train_loader = DataLoader(
        ShapeletStayDataset(train_b),
        batch_size=args.batch_stays,
        shuffle=True,
        drop_last=False,
    )
    # no shuffle for distance collection (reproducible order)
    train_loader_q = DataLoader(
        ShapeletStayDataset(train_b),
        batch_size=args.batch_stays,
        shuffle=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        ShapeletStayDataset(val_b),
        batch_size=args.batch_stays,
        shuffle=False,
    )
    log_every = int(args.log_every)
    min_delta = float(args.min_delta)
    accum = args.grad_accum
    _log(
        f"[train] batches/epoch={len(train_loader)}  batch_stays={args.batch_stays}  "
        f"q_policy={q_policy}  q_quantile={q_quantile}  log_every={log_every}"
    )

    history: list = []
    # ----- Phase 1: joint (or static) train with q fixed at init (0) -----
    opt = torch.optim.AdamW(
        model.grouped_parameter_groups(LR_SHAPELET, LR_ALPHA, LR_HEAD, WEIGHT_DECAY)
    )
    best_metric, best_state, history = run_epoch_loop(
        model,
        opt,
        train_loader,
        val_loader,
        device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=min_delta,
        accum=accum,
        log_every=log_every,
        phase_name="phase1",
        history=history,
        epoch_offset=0,
    )
    if best_state is not None:
        model.load_state_dict(best_state)

    q_values_phase1 = [float(v) for v in model.q.detach().cpu().tolist()]
    q_values_final = list(q_values_phase1)
    phase2_best = None
    phase2_state = None

    # ----- Phase 2 (B1): freeze S,α; set q from train d; retrain head -----
    if q_policy == "train_median" and head_mode != "static":
        _log(
            f"[B1] calibrate q: freeze templates, train-distance quantile="
            f"{q_quantile} (P{int(q_quantile * 100)})"
        )
        model.freeze_templates(True)
        d_flat = collect_train_stage_distances(model, train_loader_q, device)
        model.set_q_from_distances(d_flat, quantile=q_quantile)
        q_values_final = [float(v) for v in model.q.detach().cpu().tolist()]
        _log(f"[B1] q before={q_values_phase1}")
        _log(f"[B1] q after ={q_values_final}")
        _log(
            f"[B1] d samples/template={d_flat.shape[1]}  "
            f"d mean={[round(float(d_flat[i].mean()), 4) for i in range(d_flat.shape[0])]}"
        )

        # val with new q, old head (no retrain yet)
        pre_head = evaluate_auprc_auroc(
            model, val_loader, device, desc="val after q, before head retrain"
        )
        _log(
            f"[B1] val after q / before head retrain: "
            f"AUPRC={pre_head['auprc']:.4f} AUROC={pre_head['auroc']:.4f}"
        )

        opt_head = torch.optim.AdamW(
            model.head_parameter_groups(LR_HEAD, WEIGHT_DECAY)
        )
        phase2_epochs = int(args.q_head_epochs)
        phase2_pat = int(args.q_head_patience)
        n_phase1_epochs = max((h["epoch"] for h in history), default=0)
        phase2_best, phase2_state, history = run_epoch_loop(
            model,
            opt_head,
            train_loader,
            val_loader,
            device,
            max_epochs=phase2_epochs,
            patience=phase2_pat,
            min_delta=min_delta,
            accum=accum,
            log_every=log_every,
            phase_name="phase2_head",
            history=history,
            epoch_offset=n_phase1_epochs,
        )
        # Saved arm uses calibrated q + head retrain (protocol path), even if
        # phase2 val is not higher than phase1 (still report both metrics).
        if phase2_state is not None:
            model.load_state_dict(phase2_state)
            best_state = phase2_state
        phase1_best_metric = best_metric
        best_metric = (
            float(phase2_best) if phase2_best is not None else float(best_metric)
        )
        _log(
            f"[B1] phase1_best_AUPRC={phase1_best_metric:.4f}  "
            f"phase2_best_AUPRC={best_metric:.4f}"
        )
        model.freeze_templates(False)
    elif q_policy == "zero":
        _log("[B2-arm] q_policy=zero: q remains 0 (no train-distance calibration)")
        phase1_best_metric = best_metric
    else:
        phase1_best_metric = best_metric

    out_dir = PROJ / "data" / "processed" / "shapelet"
    out_dir.mkdir(parents=True, exist_ok=True)
    q_tag = "qzero" if q_policy == "zero" else f"qp{int(q_quantile * 100)}"
    tag = f"{args.mode}_{head_mode}_{q_tag}"
    ckpt_path = out_dir / f"checkpoint_{tag}.pt"
    meta = {
        "mode": args.mode,
        "head": head_mode,
        "q_policy": q_policy,
        "q_quantile": q_quantile if q_policy == "train_median" else None,
        "q_phase1": q_values_phase1,
        "q_final": q_values_final,
        "vars": list(vars_),
        "lengths": list(lengths),
        "k": k,
        "n_static": len(SHAPELET_STATIC_COLS),
        "static_cols": list(SHAPELET_STATIC_COLS),
        "monitor": MONITOR,
        "best_metric": best_metric,
        "phase1_best_metric": phase1_best_metric,
        "phase2_best_metric": phase2_best,
        "history": history,
        "seed": SEED,
        "note": (
            "B1/B2: phase1 trains with q=0; train_median freezes S,α, sets q from "
            "train stage softmin distances, then retrains head only. "
            "test not evaluated unless --eval-test."
        ),
    }
    payload = {
        "model_state": best_state if best_state is not None else model.state_dict(),
        "meta": meta,
        "keys": model.template_keys(),
        "activation_names": model.activation_names(),
        "head_mode": head_mode,
        "q_policy": q_policy,
    }
    torch.save(payload, ckpt_path)
    meta_path = out_dir / f"meta_{tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"[train] saved {ckpt_path}")
    _log(f"[train] saved {meta_path}")

    # compact B2 comparison helper file when both arms exist later
    summary_path = out_dir / f"q_ablation_summary_{args.mode}_{head_mode}.json"
    arm = {
        "tag": tag,
        "q_policy": q_policy,
        "q_final": q_values_final,
        "best_metric": best_metric,
        "n_history": len(history),
        "ckpt": str(ckpt_path),
    }
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {"arms": []}
    else:
        summary = {"mode": args.mode, "head": head_mode, "arms": []}
    arms = [a for a in summary.get("arms", []) if a.get("q_policy") != q_policy]
    arms.append(arm)
    summary["arms"] = arms
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"[train] updated {summary_path}")

    if args.eval_test:
        _log("[train] WARNING: evaluating test once (do not use for tuning)")
        if "test" not in packs:
            packs = prepare_splits(
                vars_=vars_,
                static_cols=SHAPELET_STATIC_COLS,
                max_stays_per_split=args.max_stays,
                seed=SEED,
                show_progress=True,
                build_test=True,
            )
        test_b = packs["test"]
        test_loader = DataLoader(
            ShapeletStayDataset(test_b),
            batch_size=args.batch_stays,
            shuffle=False,
        )
        if best_state is not None:
            model.load_state_dict(best_state)
        test_m = evaluate_auprc_auroc(
            model, test_loader, device, desc="test", show_progress=True
        )
        _log(f"  test AUPRC={test_m['auprc']:.4f} AUROC={test_m['auroc']:.4f}")
        meta["test"] = test_m
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shapelet learning (G1 skeleton)")
    p.add_argument(
        "--mode",
        choices=("debug", "pilot", "final"),
        default=DEFAULT_RUN_MODE,
        help="structure mode (default pilot); dry-run often uses debug",
    )
    p.add_argument(
        "--vars",
        default=None,
        help="comma-separated subset of SHAPELET_VARS; debug default=urine_output_ml",
    )
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="synthetic batch only; no patient data (G1 gate)",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="opt-in real training on hourly_grid",
    )
    p.add_argument("--max-stays", type=int, default=None, help="subsample per split")
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--batch-stays", type=int, default=BATCH_STAYS)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument(
        "--eval-test",
        action="store_true",
        help="after train, evaluate test once (forbidden for tuning)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="also print a text line every N batches; 0 = tqdm progress bar only (default)",
    )
    p.add_argument(
        "--head",
        choices=("combined", "static", "shapelet"),
        default="combined",
        help=(
            "risk-head ablation: combined=Shapelet a + static C (default); "
            "static=only 11 covariates; shapelet=only activations (no static)"
        ),
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="min val-metric improvement to reset early-stop patience (default 1e-4)",
    )
    p.add_argument(
        "--q-policy",
        choices=("zero", "train_median"),
        default="zero",
        help=(
            "B1/B2: zero=q stays 0; train_median=after phase1 freeze S,α, set q from "
            "train stage-distance quantile, retrain head (β/θ)"
        ),
    )
    p.add_argument(
        "--q-quantile",
        type=float,
        default=0.5,
        help="train distance quantile for q when --q-policy train_median (default 0.5)",
    )
    p.add_argument(
        "--q-head-epochs",
        type=int,
        default=20,
        help="phase2 head-only epochs after q calibration (default 20)",
    )
    p.add_argument(
        "--q-head-patience",
        type=int,
        default=8,
        help="phase2 early-stop patience (default 8)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = pick_device(args.device)
    # dry-run is default when neither --train nor --dry-run (safe)
    if not args.train:
        args.dry_run = True
    if args.train and args.dry_run:
        # explicit --train wins if both passed without meaning; prefer train
        if "--dry-run" in sys.argv and "--train" in sys.argv:
            raise SystemExit("pass only one of --dry-run / --train")
        args.dry_run = False

    mode = args.mode
    if args.dry_run and args.mode == DEFAULT_RUN_MODE and "--mode" not in sys.argv:
        mode = "debug"  # lighter default for skeleton smoke
    vars_ = resolve_vars(mode, args.vars)

    _log("=" * 60)
    _log("11: Shapelet learn (G1 skeleton)")
    _log("=" * 60)

    if args.dry_run:
        run_dry_run(device, mode, vars_)
        _log("\n[gate] G1 dry-run done. No patient data read. No templates exported.")
        return

    run_train(args, device, vars_)


if __name__ == "__main__":
    main()
