#!/usr/bin/env python3
"""
Pilot Shapelet training — 7 vars × L=6 × K=3 (docs/12 §9B).

Hard gates only (implementation smoke, not paper claim):
  G1  training stable   — no NaN/Inf; loss finite; no explosion
  G2  templates healthy — not collapsed; activations not all dead/saturated
  G3  val >> random     — AUPRC clearly above prevalence; AUROC > 0.55

Frequent checkpoints + resume (laptop-friendly interruptible runs).

Usage (cmd, single line):
  python -u scripts/13_shapelet_pilot.py --train
  python -u scripts/13_shapelet_pilot.py --train --max-epochs 5
  python -u scripts/13_shapelet_pilot.py --train --resume auto
  python -u scripts/13_shapelet_pilot.py --gates-only --resume auto

Defaults:
  head=combined, q_policy=zero, save every 1 epoch, batch=16, accum=2
"""

from __future__ import annotations

import argparse
import json
import math
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
    GRAD_ACCUM,
    GRAD_CLIP,
    LR_ALPHA,
    LR_HEAD,
    LR_SHAPELET,
    MONITOR,
    PATIENCE,
    PILOT_K,
    PILOT_LENGTHS,
    SEED,
    SHAPELET_STATIC_COLS,
    SHAPELET_VARS,
    WEIGHT_DECAY,
)
from scripts.utils.shapelet_model import ShapeletRiskModel  # noqa: E402
from scripts.utils.shapelet_prep import (  # noqa: E402
    ShapeletStayDataset,
    prepare_splits,
    train_pos_weight,
)

# Load evaluate helper from 11 (module name starts with digit)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "shapelet_learn_11", PROJ / "scripts" / "11_shapelet_learn.py"
)
_mod11 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod11)
evaluate_auprc_auroc = _mod11.evaluate_auprc_auroc


OUT_DIR = PROJ / "data" / "processed" / "shapelet" / "pilot"
DEFAULT_RUN_NAME = "pilot_7var_L6_K3"


def _log(msg: str) -> None:
    print(msg, flush=True)


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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dir(run_name: str) -> Path:
    d = OUT_DIR / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_history_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path,
    *,
    model: ShapeletRiskModel,
    opt: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    best_state: dict | None,
    history: list,
    meta_extra: dict,
    is_best: bool = False,
) -> None:
    payload = {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state": opt.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_state": best_state,
        "history": history,
        "keys": model.template_keys(),
        "activation_names": model.activation_names(),
        "head_mode": model.head_mode,
        "meta": meta_extra,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.save(payload, path)
    _log(f"  [ckpt] saved {path} (epoch={epoch}, best={best_metric:.4f})")
    if is_best:
        best_path = path.parent / "ckpt_best.pt"
        torch.save(payload, best_path)
        _log(f"  [ckpt] saved {best_path}")


def load_resume(path: Path, device: torch.device) -> dict:
    _log(f"[resume] loading {path}")
    return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def collect_val_activations(
    model: ShapeletRiskModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, acts = [], []
    for batch in tqdm(loader, desc="gate activations", leave=False):
        x = batch["x"].to(device)
        m = batch["m"].to(device)
        d = batch["delta"].to(device)
        s = batch["static"].to(device)
        y = batch["y"].to(device)
        out = model(x, m, d, s, y=None)
        ys.append(y.cpu().numpy())
        acts.append(out.activations.cpu().numpy())
    return np.concatenate(ys), np.concatenate(acts)


def evaluate_hard_gates(
    model: ShapeletRiskModel,
    history: list,
    val_loader: DataLoader,
    device: torch.device,
    *,
    prevalence: float,
    auprc_mult: float = 1.3,
    auroc_min: float = 0.55,
    cos_max: float = 0.95,
    mean_a_lo: float = 0.01,
    mean_a_hi: float = 0.99,
    sat_max: float = 0.98,
) -> dict:
    """Three pilot hard gates only."""
    gates: dict = {"passed_all": False, "checks": {}}

    # --- G1 training stability ---
    losses = [h.get("train_loss") for h in history]
    auprcs = [h.get("val_auprc") for h in history]
    finite_loss = all(
        isinstance(x, (int, float)) and math.isfinite(float(x)) for x in losses
    )
    finite_val = all(
        isinstance(x, (int, float)) and math.isfinite(float(x)) for x in auprcs
    )
    no_explode = True
    if losses:
        # explosion: any loss > 50 or jumps 20x from epoch1 median-ish
        no_explode = max(float(x) for x in losses) < 50.0
    improving_or_flat = True
    if len(losses) >= 3:
        # not strictly increasing forever at the end
        improving_or_flat = float(losses[-1]) < float(losses[0]) * 1.5
    g1_ok = bool(history) and finite_loss and finite_val and no_explode and improving_or_flat
    gates["checks"]["G1_training_stable"] = {
        "pass": g1_ok,
        "n_epochs": len(history),
        "finite_loss": finite_loss,
        "finite_val": finite_val,
        "no_explode": no_explode,
        "loss_first": float(losses[0]) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
        "best_val_auprc": float(max(auprcs)) if auprcs else None,
    }

    # --- G2 templates / activations ---
    y, a = collect_val_activations(model, val_loader, device)
    mean_a = float(a.mean())
    frac_lo = float((a < 0.05).mean())
    frac_hi = float((a > 0.95).mean())
    # pairwise cosine within each (var, L) group
    keys = model.template_keys()
    cos_list = []
    by_group: dict[tuple[str, int], list[int]] = {}
    for i, (v, L, _k) in enumerate(keys):
        by_group.setdefault((v, int(L)), []).append(i)
    max_abs_cos = 0.0
    for idxs in by_group.values():
        if len(idxs) < 2:
            continue
        vecs = []
        for i in idxs:
            s = model.shapelets[i].detach().float().cpu()
            s = s / (s.norm() + 1e-8)
            vecs.append(s)
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                c = float((vecs[i] * vecs[j]).sum().item())
                cos_list.append(c)
                max_abs_cos = max(max_abs_cos, abs(c))
    not_collapsed = max_abs_cos < cos_max
    not_all_dead = mean_a > mean_a_lo and frac_lo < sat_max
    not_all_sat = mean_a < mean_a_hi and frac_hi < sat_max
    g2_ok = not_collapsed and not_all_dead and not_all_sat
    gates["checks"]["G2_templates_activations"] = {
        "pass": g2_ok,
        "max_abs_cosine_same_var_L": max_abs_cos,
        "mean_a_val": mean_a,
        "frac_a_lt_0_05": frac_lo,
        "frac_a_gt_0_95": frac_hi,
        "n_cos_pairs": len(cos_list),
        "not_collapsed": not_collapsed,
        "not_all_dead": not_all_dead,
        "not_all_saturated": not_all_sat,
    }

    # --- G3 val >> random ---
    last = history[-1] if history else {}
    best_auprc = float(max(auprcs)) if auprcs else float("nan")
    aurocs = [
        float(h["val_auroc"])
        for h in history
        if isinstance(h.get("val_auroc"), (int, float))
        and math.isfinite(float(h["val_auroc"]))
    ]
    best_auroc = float(max(aurocs)) if aurocs else float("nan")
    # re-eval current model (usually best loaded)
    metrics = evaluate_auprc_auroc(
        model, val_loader, device, desc="gate val metrics", show_progress=True
    )
    thr_auprc = prevalence * auprc_mult
    g3_ok = (
        math.isfinite(metrics["auprc"])
        and metrics["auprc"] >= thr_auprc
        and math.isfinite(metrics["auroc"])
        and metrics["auroc"] >= auroc_min
    )
    gates["checks"]["G3_val_above_random"] = {
        "pass": g3_ok,
        "val_auprc": metrics["auprc"],
        "val_auroc": metrics["auroc"],
        "prevalence": prevalence,
        "auprc_threshold": thr_auprc,
        "auroc_threshold": auroc_min,
        "history_best_auprc": best_auprc,
        "history_best_auroc": best_auroc,
        "last_epoch_auprc": last.get("val_auprc"),
    }

    gates["passed_all"] = all(c["pass"] for c in gates["checks"].values())
    gates["summary"] = {
        "G1": gates["checks"]["G1_training_stable"]["pass"],
        "G2": gates["checks"]["G2_templates_activations"]["pass"],
        "G3": gates["checks"]["G3_val_above_random"]["pass"],
        "passed_all": gates["passed_all"],
    }
    return gates


def train_pilot(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    set_seed(SEED)
    rdir = run_dir(args.run_name)
    _log("=" * 60)
    _log("13: Shapelet PILOT (7 vars × L=6 × K=3)")
    _log("=" * 60)
    _log(f"  device={device}  run_dir={rdir}")
    _log(f"  save_every={args.save_every}  max_epochs={args.max_epochs}  resume={args.resume}")

    lengths = PILOT_LENGTHS
    k = PILOT_K
    vars_ = tuple(SHAPELET_VARS)
    head_mode = args.head

    _log("[pilot] loading splits (7 vars, train-only scalers)…")
    packs = prepare_splits(
        vars_=vars_,
        static_cols=SHAPELET_STATIC_COLS,
        max_stays_per_split=args.max_stays,
        seed=SEED,
        show_progress=True,
        build_test=False,
    )
    train_b = packs["train"]
    val_b = packs["val"]
    pos_w = train_pos_weight(train_b.y)
    prevalence = float(val_b.y.mean())
    _log(
        f"  train N={len(train_b.y)} val N={len(val_b.y)} "
        f"pos_weight={pos_w:.3f} val_prevalence={prevalence:.4f}"
    )
    _log(f"  n_templates={len(vars_) * len(lengths) * k}  n_act={len(vars_) * len(lengths) * k * 4}")

    model = ShapeletRiskModel(
        vars_=vars_,
        lengths=lengths,
        k=k,
        n_static=len(SHAPELET_STATIC_COLS),
        pos_weight=pos_w,
        head_mode=head_mode,
        q_init=0.0,
    ).to(device)

    opt = torch.optim.AdamW(
        model.grouped_parameter_groups(LR_SHAPELET, LR_ALPHA, LR_HEAD, WEIGHT_DECAY)
    )

    history: list = []
    start_epoch = 0
    best_metric = -1.0
    best_state = None
    bad_epochs = 0

    resume_path: Path | None = None
    if args.resume:
        if args.resume == "auto":
            cand = rdir / "ckpt_last.pt"
            resume_path = cand if cand.is_file() else None
            if resume_path is None:
                _log("[resume] auto: no ckpt_last.pt — fresh start")
        else:
            resume_path = Path(args.resume)
            if not resume_path.is_file():
                raise FileNotFoundError(resume_path)

    if resume_path is not None:
        ckpt = load_resume(resume_path, device)
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state") is not None:
            try:
                opt.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                _log(f"[resume] optimizer state skip ({e})")
        history = list(ckpt.get("history") or [])
        start_epoch = int(ckpt.get("epoch") or 0)
        best_metric = float(ckpt.get("best_metric") or -1.0)
        best_state = ckpt.get("best_state")
        # recompute bad_epochs from history tail
        if history and best_metric > 0:
            bad_epochs = 0
            for h in reversed(history):
                m = h.get("val_auprc")
                if m is None or not math.isfinite(float(m)):
                    break
                if float(m) > best_metric - 1e-12:
                    break
                bad_epochs += 1
        _log(
            f"[resume] epoch={start_epoch} best_metric={best_metric:.4f} "
            f"history_len={len(history)} bad_epochs≈{bad_epochs}"
        )

    if args.gates_only:
        if best_state is not None:
            model.load_state_dict(best_state)
        val_loader = DataLoader(
            ShapeletStayDataset(val_b), batch_size=args.batch_stays, shuffle=False
        )
        gates = evaluate_hard_gates(
            model,
            history,
            val_loader,
            device,
            prevalence=prevalence,
            auprc_mult=args.auprc_mult,
            auroc_min=args.auroc_min,
        )
        gate_path = rdir / "pilot_gates.json"
        gate_path.write_text(json.dumps(gates, indent=2), encoding="utf-8")
        _log(json.dumps(gates["summary"], indent=2))
        _log(f"[gates] saved {gate_path}")
        return

    train_loader = DataLoader(
        ShapeletStayDataset(train_b),
        batch_size=args.batch_stays,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        ShapeletStayDataset(val_b),
        batch_size=args.batch_stays,
        shuffle=False,
    )
    n_batches = len(train_loader)
    accum = args.grad_accum
    min_delta = float(args.min_delta)
    history_path = rdir / "history.jsonl"
    meta_base = {
        "run_name": args.run_name,
        "mode": "pilot",
        "vars": list(vars_),
        "lengths": list(lengths),
        "k": k,
        "head": head_mode,
        "seed": SEED,
        "batch_stays": args.batch_stays,
        "grad_accum": accum,
        "monitor": MONITOR,
        "note": "pilot hard gates only; interruptible; not paper final",
    }

    max_epochs = int(args.max_epochs)
    patience = int(args.patience)
    _log(
        f"[pilot] train from epoch {start_epoch + 1} .. {max_epochs}  "
        f"batches/epoch={n_batches}"
    )

    for epoch in range(start_epoch + 1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        n_steps = 0
        t0 = time.perf_counter()
        pbar = tqdm(
            train_loader,
            desc=f"pilot ep{epoch:03d}",
            total=n_batches,
            unit="batch",
            leave=True,
        )
        nan_batch = False
        for step, batch in enumerate(pbar, start=1):
            x = batch["x"].to(device)
            m = batch["m"].to(device)
            d = batch["delta"].to(device)
            s = batch["static"].to(device)
            y = batch["y"].to(device)
            out = model(x, m, d, s, y)
            loss = out.loss / accum
            if not torch.isfinite(loss):
                _log(f"  [warn] non-finite loss at epoch {epoch} step {step}")
                nan_batch = True
                break
            loss.backward()
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                opt.zero_grad(set_to_none=True)
            batch_loss = float(out.loss.detach().cpu())
            running += batch_loss
            n_steps += 1
            pbar.set_postfix(
                loss=f"{batch_loss:.4f}",
                avg=f"{running / n_steps:.4f}",
                refresh=False,
            )
        if n_steps > 0 and n_steps % accum != 0 and not nan_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if nan_batch or n_steps == 0:
            train_loss = float("nan")
            val_m = {"auprc": float("nan"), "auroc": float("nan")}
        else:
            train_loss = running / n_steps
            val_m = evaluate_auprc_auroc(
                model,
                val_loader,
                device,
                desc=f"val ep{epoch:03d}",
                show_progress=True,
            )

        rec = {
            "epoch": epoch,
            "phase": "pilot",
            "train_loss": train_loss,
            "val_auprc": val_m["auprc"],
            "val_auroc": val_m["auroc"],
            "epoch_s": time.perf_counter() - t0,
        }
        history.append(rec)
        append_history_jsonl(history_path, rec)
        _log(
            f"  [pilot] epoch {epoch:03d}  DONE  loss={train_loss:.4f}  "
            f"val_AUPRC={val_m['auprc']:.4f}  val_AUROC={val_m['auroc']:.4f}  "
            f"epoch_s={rec['epoch_s']:.1f}"
        )

        metric = val_m["auprc"] if MONITOR == "val_auprc" else val_m["auroc"]
        is_best = False
        if (
            isinstance(metric, (int, float))
            and math.isfinite(float(metric))
            and float(metric) > best_metric + min_delta
        ):
            best_metric = float(metric)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
            is_best = True
        else:
            bad_epochs += 1

        # frequent checkpoints
        meta_now = {
            **meta_base,
            "epoch": epoch,
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
            "pos_weight": pos_w,
            "val_prevalence": prevalence,
        }
        save_checkpoint(
            rdir / "ckpt_last.pt",
            model=model,
            opt=opt,
            epoch=epoch,
            best_metric=best_metric,
            best_state=best_state,
            history=history,
            meta_extra=meta_now,
            is_best=is_best,
        )
        if args.save_every > 0 and (epoch % args.save_every == 0 or is_best):
            save_checkpoint(
                rdir / f"ckpt_ep{epoch:03d}.pt",
                model=model,
                opt=opt,
                epoch=epoch,
                best_metric=best_metric,
                best_state=best_state,
                history=history,
                meta_extra=meta_now,
                is_best=False,
            )
        # also dump light meta each epoch
        (rdir / "meta_latest.json").write_text(
            json.dumps({**meta_now, "history_tail": history[-5:]}, indent=2),
            encoding="utf-8",
        )

        if nan_batch:
            _log("[pilot] abort: non-finite loss — resume later after fix if needed")
            break
        if bad_epochs >= patience:
            _log(
                f"  [pilot] early stop at epoch {epoch} "
                f"(patience={patience}, min_delta={min_delta})"
            )
            break

    # load best for gates
    if best_state is not None:
        model.load_state_dict(best_state)
    gates = evaluate_hard_gates(
        model,
        history,
        val_loader,
        device,
        prevalence=prevalence,
        auprc_mult=args.auprc_mult,
        auroc_min=args.auroc_min,
    )
    gate_path = rdir / "pilot_gates.json"
    gate_path.write_text(json.dumps(gates, indent=2), encoding="utf-8")
    final_meta = {
        **meta_base,
        "best_metric": best_metric,
        "n_history": len(history),
        "last_epoch": history[-1]["epoch"] if history else start_epoch,
        "gates": gates["summary"],
        "gate_detail_path": str(gate_path),
    }
    (rdir / "meta_final.json").write_text(
        json.dumps(final_meta, indent=2), encoding="utf-8"
    )
    save_checkpoint(
        rdir / "ckpt_final.pt",
        model=model,
        opt=opt,
        epoch=history[-1]["epoch"] if history else start_epoch,
        best_metric=best_metric,
        best_state=best_state,
        history=history,
        meta_extra=final_meta,
        is_best=False,
    )

    _log("\n======== PILOT HARD GATES ========")
    for name, chk in gates["checks"].items():
        status = "PASS" if chk["pass"] else "FAIL"
        _log(f"  {name}: {status}")
    _log(f"  ALL: {'PASS' if gates['passed_all'] else 'FAIL'}")
    _log(f"  details -> {gate_path}")
    _log("  resume: python -u scripts/13_shapelet_pilot.py --train --resume auto")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shapelet pilot 7×L6×K3 with hard gates")
    p.add_argument("--train", action="store_true", help="run pilot training")
    p.add_argument(
        "--gates-only",
        action="store_true",
        help="only evaluate hard gates on --resume checkpoint",
    )
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    p.add_argument("--head", default="combined", choices=("combined", "shapelet"))
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--batch-stays", type=int, default=BATCH_STAYS)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="also save ckpt_epXXX.pt every N epochs (default 1=every epoch)",
    )
    p.add_argument(
        "--resume",
        default=None,
        help='path to ckpt, or "auto" for run_dir/ckpt_last.pt',
    )
    p.add_argument("--max-stays", type=int, default=None, help="debug subsample only")
    p.add_argument(
        "--auprc-mult",
        type=float,
        default=1.3,
        help="G3: require val AUPRC >= prevalence * mult (default 1.3)",
    )
    p.add_argument(
        "--auroc-min",
        type=float,
        default=0.55,
        help="G3: require val AUROC >= this (default 0.55)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train and not args.gates_only:
        _log("pass --train and/or --gates-only")
        _log("example: python -u scripts/13_shapelet_pilot.py --train --max-epochs 3")
        _log("resume:  python -u scripts/13_shapelet_pilot.py --train --resume auto")
        sys.exit(0)
    # remove bad import attempt if present
    train_pilot(args)


if __name__ == "__main__":
    main()
