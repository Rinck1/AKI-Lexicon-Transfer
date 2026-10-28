#!/usr/bin/env python3
"""
GNN graph-level death risk training (docs/13 §9–§10.3).

Modes:
  --dry-run   synthetic forward/backward (Stage B / G0)
  --train     real training on graph packs (Stage D)

Usage:
  python -u scripts/17_train_gnn_risk.py --dry-run
  python -u scripts/17_train_gnn_risk.py --train ^
    --graph-pack data/processed/graphs/graph_final_7var_L4_6_12_K3 ^
    --max-stays 128 --max-epochs 2 --device auto
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

from scripts.utils.graph_constants import (  # noqa: E402
    BATCH_GRAPHS,
    EMPTY_GRAPH_EMBED_DIM,
    FUSION_DIM,
    GNN_DROPOUT,
    GNN_GRAD_CLIP,
    GNN_LR,
    GNN_MAX_EPOCHS,
    GNN_MIN_DELTA,
    GNN_MONITOR,
    GNN_PATIENCE,
    GNN_SEED,
    GNN_WEIGHT_DECAY,
    HIDDEN_DIM,
    N_GNN_LAYERS,
    N_SPARSE_FEATURES,
    N_STATIC_FEATURES,
    NODE_TYPE_DIM,
)
from scripts.utils.graph_data import (  # noqa: E402
    GraphPackDataset,
    assert_batch_schema,
    collate_graph_batch,
    make_synthetic_batch,
)
from scripts.utils.graph_model import (  # noqa: E402
    BRANCH_MODES,
    GraphRiskModel,
    require_torch_2,
    resolve_branch_mode,
    resolve_readout,
)

DEFAULT_GRAPH_PACK = (
    PROJ / "data" / "processed" / "graphs" / "graph_final_7var_L4_6_12_K3"
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def set_seed(seed: int = GNN_SEED) -> None:
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


def run_dir_for_pack(graph_pack: Path, run_name: str | None) -> Path:
    name = run_name or f"gnn_{graph_pack.name}"
    d = PROJ / "data" / "processed" / "gnn" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_graph_pack(
    pack_dir: Path,
    split: str,
    *,
    max_stays: int | None,
    seed: int,
    forbid_test: bool = True,
) -> tuple[GraphPackDataset, dict]:
    if forbid_test and split == "test":
        raise SystemExit("refusing to load test pack (docs/13 default lock)")
    path = pack_dir / f"graph_pack_{split}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    patients = list(obj["patients"])
    meta = dict(obj.get("meta") or {})
    if max_stays is not None and len(patients) > max_stays:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(patients), size=int(max_stays), replace=False)
        idx = sorted(int(i) for i in idx)
        patients = [patients[i] for i in idx]
        meta["max_stays_subsample"] = int(max_stays)
        meta["subsample_seed"] = int(seed)
    ds = GraphPackDataset(patients)
    return ds, meta


def subject_leakage_audit(train_ds: GraphPackDataset, val_ds: GraphPackDataset) -> dict:
    tr = set(int(p["subject_id"].item()) for p in train_ds.patients)
    va = set(int(p["subject_id"].item()) for p in val_ds.patients)
    overlap = tr & va
    return {
        "train_subjects": len(tr),
        "val_subjects": len(va),
        "overlap": len(overlap),
        "pass": len(overlap) == 0,
    }


def train_pos_weight_from_ds(ds: GraphPackDataset) -> float:
    ys = np.array([float(p["y"].item()) for p in ds.patients], dtype=np.float64)
    n_pos = float(ys.sum())
    n_neg = float(len(ys) - n_pos)
    if n_pos <= 0:
        return 1.0
    return n_neg / n_pos


def empty_graph_count(ds: GraphPackDataset) -> int:
    return sum(1 for p in ds.patients if not bool(p["active_mask"].any().item()))


@torch.no_grad()
def evaluate_loader(
    model: GraphRiskModel,
    loader: DataLoader,
    device: torch.device,
    *,
    pos_weight: float | None = None,
    desc: str = "val",
) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    ys, ps, losses = [], [], []
    n_empty = 0
    n_sel_sum = 0.0
    n_e_sum = 0.0
    pi_sum = 0.0
    n_pat_with_e = 0
    n_all0 = 0
    n_all1 = 0
    for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, pos_weight=pos_weight, compute_loss=True)
        ys.append(batch["y"].detach().cpu().numpy())
        ps.append(out.prob.detach().cpu().numpy())
        losses.append(float(out.loss.detach().cpu()))
        n_empty += int((~batch["active_mask"].any(dim=-1)).sum().item())
        mask = batch["edge_mask"]
        n_e = mask.to(dtype=torch.float32).sum(dim=-1)
        has_e = n_e > 0
        n_pat_with_e += int(has_e.sum().item())
        n_e_sum += float(n_e[has_e].sum().detach().cpu()) if bool(has_e.any()) else 0.0
        if out.edge_gate is not None:
            n_sel = (out.edge_gate * mask.to(out.edge_gate.dtype)).sum(dim=-1)
            if bool(has_e.any()):
                n_sel_sum += float(n_sel[has_e].sum().detach().cpu())
                n_all0 += int(((n_sel <= 0) & has_e).sum().item())
                n_all1 += int(((n_sel >= n_e - 1e-6) & has_e).sum().item())
        if out.edge_selection_prob is not None:
            pi = out.edge_selection_prob * mask.to(out.edge_selection_prob.dtype)
            pi_sum += float(pi.sum().detach().cpu())
    y = np.concatenate(ys) if ys else np.zeros(0)
    p = np.concatenate(ps) if ps else np.zeros(0)
    if len(y) == 0 or y.min() == y.max():
        auprc, auroc = float("nan"), float("nan")
    else:
        auprc = float(average_precision_score(y, p))
        auroc = float(roc_auc_score(y, p))
    brier = float(np.mean((p - y) ** 2)) if len(y) else float("nan")
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "auprc": auprc,
        "auroc": auroc,
        "brier": brier,
        "n": int(len(y)),
        "n_empty_in_eval": n_empty,
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "mean_n_selected": (n_sel_sum / n_pat_with_e) if n_pat_with_e else float("nan"),
        "mean_pi": (pi_sum / n_e_sum) if n_e_sum > 0 else float("nan"),
        "frac_all_zero_mask": (n_all0 / n_pat_with_e) if n_pat_with_e else float("nan"),
        "frac_all_one_mask": (n_all1 / n_pat_with_e) if n_pat_with_e else float("nan"),
    }


def save_checkpoint(
    path: Path,
    *,
    model: GraphRiskModel,
    opt: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    best_state: dict | None,
    history: list,
    meta: dict,
    is_best: bool = False,
) -> None:
    payload = {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state": opt.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_state": best_state,
        "history": history,
        "meta": meta,
        "n_nodes": model.n_nodes,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.save(payload, path)
    _log(f"  [ckpt] {path.name} epoch={epoch} best={best_metric:.4f}")
    if is_best:
        best_path = path.parent / "ckpt_best.pt"
        torch.save(payload, best_path)
        _log(f"  [ckpt] {best_path.name}")


def append_history(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_eval_ckpt(args: argparse.Namespace, device: torch.device) -> dict:
    """Val-only metrics for a frozen checkpoint. Never loads test."""
    require_torch_2()
    pack_dir = Path(args.graph_pack) if args.graph_pack else DEFAULT_GRAPH_PACK
    ckpt_path = Path(args.eval_ckpt)
    if args.unlock_test:
        raise SystemExit("eval-ckpt must not use --unlock-test")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_nodes = int(ckpt.get("n_nodes") or 252)
    ckpt_meta = dict(ckpt.get("meta") or {})
    branch = resolve_branch_mode(ckpt_meta.get("branch_mode") or args.branch)
    readout = resolve_readout(ckpt_meta.get("readout") or args.readout)
    n_layers = int(ckpt_meta.get("n_layers") or args.n_layers)
    val_ds, _ = load_graph_pack(
        pack_dir, "val", max_stays=args.max_stays, seed=int(args.seed) + 1
    )
    train_ds, _ = load_graph_pack(
        pack_dir, "train", max_stays=args.max_stays, seed=args.seed
    )
    audit = subject_leakage_audit(train_ds, val_ds)
    if not audit["pass"]:
        raise RuntimeError(f"SUBJECT leakage: {audit}")
    pos_w = train_pos_weight_from_ds(train_ds)
    model = GraphRiskModel(
        n_nodes=n_nodes,
        branch_mode=branch,
        readout=readout,
        n_layers=n_layers,
        use_node_length=bool(ckpt_meta.get("use_node_length", False)),
    ).to(device)
    state = ckpt.get("best_state") or ckpt["model_state"]
    model.load_state_dict(state)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_batch,
    )
    metrics = evaluate_loader(
        model, val_loader, device, pos_weight=pos_w, desc="eval val"
    )
    report = {
        "eval_ckpt": str(ckpt_path.resolve()),
        "branch": branch,
        "pos_weight": pos_w,
        "subject_audit": audit,
        "val": metrics,
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_best_metric": ckpt.get("best_metric"),
        "test_built": False,
    }
    _log("[17] eval-ckpt")
    _log(json.dumps(report, indent=2, default=str))
    return report


def run_dry_run(device: torch.device, n_nodes: int = 16, batch_size: int = 4) -> dict:
    require_torch_2()
    set_seed(GNN_SEED)
    cfg = {
        "stage": "B_skeleton",
        "mode": "dry-run",
        "n_nodes": n_nodes,
        "device": str(device),
        "test_built": False,
        "paper_eligible": False,
    }
    _log(json.dumps(cfg, indent=2))
    batch = make_synthetic_batch(n_nodes=n_nodes, batch_size=batch_size, seed=GNN_SEED)
    assert_batch_schema(batch, n_nodes)
    batch = {k: v.to(device) for k, v in batch.items()}
    model = GraphRiskModel(n_nodes=n_nodes).to(device)
    out = model(batch, pos_weight=2.0)
    out.loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GNN_GRAD_CLIP))
    layer0 = model.layers[0]
    rel = {
        "W_pred": float(layer0.W_pred.weight.grad.abs().sum()),
        "W_succ": float(layer0.W_succ.weight.grad.abs().sum()),
        "W_co": float(layer0.W_co.weight.grad.abs().sum()),
    }
    report = {
        **cfg,
        "loss": float(out.loss.item()),
        "logits_finite": bool(torch.isfinite(out.logits).all().item()),
        "grad_norm": grad_norm,
        "rel_grad_l1": rel,
        "rel_grads_all_nonzero": all(v > 0 for v in rel.values()),
    }
    _log(json.dumps(report, indent=2))
    if not report["logits_finite"] or not report["rel_grads_all_nonzero"]:
        raise RuntimeError(f"dry-run failed: {report}")
    _log("[17] dry-run OK")
    return report


def run_train(args: argparse.Namespace, device: torch.device) -> dict:
    require_torch_2()
    set_seed(args.seed)

    pack_dir = Path(args.graph_pack) if args.graph_pack else DEFAULT_GRAPH_PACK
    if not pack_dir.is_dir():
        raise FileNotFoundError(pack_dir)
    # hard refuse test pack presence usage
    if (pack_dir / "graph_pack_test.pt").is_file() and args.unlock_test:
        _log("[warn] test pack exists but unlock-test not used for training loop")
    if args.unlock_test:
        raise SystemExit(
            "Stage D smoke/full train must not use --unlock-test; "
            "test remains locked."
        )

    rdir = run_dir_for_pack(pack_dir, args.run_name)
    history_path = rdir / "history.jsonl"

    _log("=" * 60)
    _log("17: GNN risk train (Stage D)")
    _log("=" * 60)
    _log(f"  pack={pack_dir}")
    _log(f"  out={rdir}")
    branch_mode = resolve_branch_mode(args.branch)
    readout = resolve_readout(args.readout)
    _log(
        f"  branch={branch_mode}  flags={BRANCH_MODES[branch_mode]}  "
        f"readout={readout}  max_stays={args.max_stays}  "
        f"max_epochs={args.max_epochs}  batch={args.batch_size}  seed={args.seed}"
    )

    train_ds, train_meta = load_graph_pack(
        pack_dir, "train", max_stays=args.max_stays, seed=args.seed
    )
    val_ds, val_meta = load_graph_pack(
        pack_dir, "val", max_stays=args.max_stays, seed=args.seed + 1
    )
    n_nodes = train_ds.n_nodes
    if val_ds.n_nodes != n_nodes:
        raise ValueError(f"n_nodes train={n_nodes} val={val_ds.n_nodes}")

    audit = subject_leakage_audit(train_ds, val_ds)
    _log(f"  SUBJECT audit: {audit}")
    if not audit["pass"]:
        raise RuntimeError(f"SUBJECT_ID leakage train/val: {audit}")

    pos_w = train_pos_weight_from_ds(train_ds)
    n_empty_tr = empty_graph_count(train_ds)
    n_empty_va = empty_graph_count(val_ds)
    y_tr = np.array([float(p["y"].item()) for p in train_ds.patients])
    _log(
        f"  train N={len(train_ds)} empty={n_empty_tr}  "
        f"val N={len(val_ds)} empty={n_empty_va}  "
        f"pos_weight={pos_w:.4f}  train_prev={y_tr.mean():.4f}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graph_batch,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_batch,
    )

    residual_from_b = Path(args.residual_from_b) if args.residual_from_b else None
    if residual_from_b is not None and not residual_from_b.is_file():
        raise FileNotFoundError(residual_from_b)

    n_layers = int(args.n_layers)
    use_node_length = bool(args.use_node_length)
    if use_node_length:
        sample0 = train_ds.patients[0]
        if "node_length" not in sample0:
            raise SystemExit("--use-node-length set but pack has no node_length tensor")
    model = GraphRiskModel(
        n_nodes=n_nodes,
        n_sparse=N_SPARSE_FEATURES,
        n_static=N_STATIC_FEATURES,
        n_layers=n_layers,
        dropout=GNN_DROPOUT,
        branch_mode="B" if residual_from_b is not None else branch_mode,
        residual_graph=residual_from_b is not None,
        readout="pool" if residual_from_b is not None else readout,
        use_node_length=use_node_length,
    ).to(device)

    if residual_from_b is not None:
        b_ckpt = torch.load(residual_from_b, map_location="cpu", weights_only=False)
        b_state = b_ckpt.get("best_state") or b_ckpt["model_state"]
        loaded = model.load_b_branch(b_state)
        model.freeze_b_params()
        with torch.no_grad():
            model.graph_gate.zero_()
        _log(
            f"  residual: loaded {len(loaded)} B tensors from {residual_from_b}  "
            f"gate={float(model.graph_gate.detach())}  B frozen"
        )

    n_params = sum(p.numel() for p in model.parameters())
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(
        f"  model: n_nodes={n_nodes} layers={n_layers} "
        f"h={HIDDEN_DIM} h_graph={EMPTY_GRAPH_EMBED_DIM} fusion={FUSION_DIM} "
        f"params={n_params} trainable={n_train_p} residual={residual_from_b is not None}"
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=GNN_WEIGHT_DECAY,
    )

    history: list = []
    start_epoch = 0
    best_metric = -1.0
    best_state = None
    bad_epochs = 0

    if args.resume:
        if args.resume == "auto":
            cand = rdir / "ckpt_last.pt"
            resume_path = cand if cand.is_file() else None
        else:
            resume_path = Path(args.resume)
        if resume_path is not None and resume_path.is_file():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            if ckpt.get("optimizer_state") is not None:
                try:
                    opt.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    _log(f"[resume] optimizer skip ({e})")
            history = list(ckpt.get("history") or [])
            start_epoch = int(ckpt.get("epoch") or 0)
            best_metric = float(ckpt.get("best_metric") or -1.0)
            best_state = ckpt.get("best_state")
            _log(
                f"[resume] epoch={start_epoch} best={best_metric:.4f} "
                f"history_len={len(history)}"
            )

    max_epochs = int(args.max_epochs or GNN_MAX_EPOCHS)
    patience = int(args.patience)
    min_delta = float(args.min_delta)
    meta_base = {
        "stage": "D_gnn_train",
        "paper_eligible": False,  # Stage D gate runs are engineering only
        "test_built": False,
        "graph_pack": str(pack_dir.resolve()),
        "n_nodes": n_nodes,
        "n_layers": n_layers,
        "multiscale_layers": list(range(n_layers + 1)) if readout == "additive_ms" else None,
        "node_contribution_decomposable": readout
        in {"additive", "additive_ms", "additive_gate", "additive_l0", "additive_topk"},
        "test_read": False,
        "hidden_dim": HIDDEN_DIM,
        "node_type_dim": NODE_TYPE_DIM,
        "dropout": GNN_DROPOUT,
        "fusion_dim": FUSION_DIM,
        "h_graph_dim": EMPTY_GRAPH_EMBED_DIM,
        "lr": args.lr,
        "weight_decay": GNN_WEIGHT_DECAY,
        "batch_size": args.batch_size,
        "max_stays": args.max_stays,
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "monitor": GNN_MONITOR,
        "seed": args.seed,
        "pos_weight": pos_w,
        "subject_audit": audit,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "train_empty": n_empty_tr,
        "val_empty": n_empty_va,
        "n_params": n_params,
        "branch_mode": "B" if residual_from_b is not None else branch_mode,
        "branch_flags": BRANCH_MODES["B" if residual_from_b is not None else branch_mode],
        "readout": "pool" if residual_from_b is not None else readout,
        "use_node_length": use_node_length,
        "lambda_sparse": getattr(model, "lambda_sparse", None),
        "lambda_stable": getattr(model, "lambda_stable", None),
        "lambda_l0": getattr(model, "lambda_l0", None),
        "lambda_rank": getattr(model, "lambda_rank", None),
        "v_max": getattr(model, "v_max", None),
        "residual_from_b": str(residual_from_b.resolve()) if residual_from_b else None,
        "n_trainable": n_train_p,
        "pack_train_meta": train_meta,
        "pack_val_meta": val_meta,
    }
    (rdir / "meta_config.json").write_text(
        json.dumps(meta_base, indent=2, default=str), encoding="utf-8"
    )

    # peak memory helper
    def mem_mb() -> float | None:
        if device.type != "cuda":
            return None
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if residual_from_b is not None and start_epoch == 0:
        init_m = evaluate_loader(
            model, val_loader, device, pos_weight=pos_w, desc="val residual init"
        )
        _log(
            f"  [residual init] gate=0  val_AUPRC={init_m['auprc']:.4f}  "
            f"val_AUROC={init_m['auroc']:.4f}  val_Brier={init_m['brier']:.4f}  "
            f"(should match frozen B)"
        )

    for epoch in range(start_epoch + 1, max_epochs + 1):
        model.train()
        running = 0.0
        run_gate = 0.0
        run_pi = 0.0
        run_sp = 0.0
        run_st = 0.0
        n_steps = 0
        t0 = time.perf_counter()
        pbar = tqdm(train_loader, desc=f"gnn ep{epoch:03d}", unit="batch")
        nan_batch = False
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            # schema check once cheaply on first batch of first epoch
            if epoch == start_epoch + 1 and n_steps == 0:
                assert_batch_schema(batch, n_nodes)
            out = model(batch, pos_weight=pos_w)
            loss = out.loss
            if not torch.isfinite(loss):
                _log(f"  [warn] non-finite loss epoch={epoch} step={n_steps}")
                nan_batch = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GNN_GRAD_CLIP)
            opt.step()
            bl = float(loss.detach().cpu())
            running += bl
            if out.mean_gate is not None:
                run_gate += float(out.mean_gate.detach().cpu())
            if out.edge_selection_prob is not None:
                em = batch["edge_mask"].to(out.edge_selection_prob.dtype)
                n_ve = em.sum().clamp_min(1.0)
                run_pi += float((out.edge_selection_prob * em).sum().detach().cpu() / n_ve)
            if out.loss_sparse is not None:
                run_sp += float(out.loss_sparse.detach().cpu())
            if out.loss_stable is not None:
                run_st += float(out.loss_stable.detach().cpu())
            n_steps += 1
            pbar.set_postfix(loss=f"{bl:.4f}", avg=f"{running / n_steps:.4f}")

        if nan_batch or n_steps == 0:
            train_loss = float("nan")
            val_m = {
                "loss": float("nan"),
                "auprc": float("nan"),
                "auroc": float("nan"),
                "brier": float("nan"),
                "n": 0,
                "n_empty_in_eval": 0,
                "prevalence": float("nan"),
            }
        else:
            train_loss = running / n_steps
            val_m = evaluate_loader(
                model,
                val_loader,
                device,
                pos_weight=pos_w,
                desc=f"val ep{epoch:03d}",
            )

        rec = {
            "epoch": epoch,
            "phase": "gnn",
            "train_loss": train_loss,
            "val_loss": val_m["loss"],
            "val_auprc": val_m["auprc"],
            "val_auroc": val_m["auroc"],
            "val_brier": val_m["brier"],
            "val_n": val_m["n"],
            "val_empty": val_m["n_empty_in_eval"],
            "epoch_s": time.perf_counter() - t0,
            "peak_mem_mb": mem_mb(),
            "mean_gate": (run_gate / n_steps) if n_steps else None,
            "mean_pi": (run_pi / n_steps) if n_steps else None,
            "val_mean_n_selected": val_m.get("mean_n_selected"),
            "val_mean_pi": val_m.get("mean_pi"),
            "val_frac_all_zero_mask": val_m.get("frac_all_zero_mask"),
            "val_frac_all_one_mask": val_m.get("frac_all_one_mask"),
            "loss_sparse": (run_sp / n_steps) if n_steps else None,
            "loss_stable": (run_st / n_steps) if n_steps else None,
            "gate": (
                float(model.graph_gate.detach().cpu())
                if getattr(model, "graph_gate", None) is not None
                else None
            ),
        }
        history.append(rec)
        append_history(history_path, rec)
        _log(
            f"  [gnn] epoch {epoch:03d} DONE  loss={train_loss:.4f}  "
            f"val_loss={val_m['loss']:.4f}  "
            f"val_AUPRC={val_m['auprc']:.4f}  val_AUROC={val_m['auroc']:.4f}  "
            f"val_Brier={val_m['brier']:.4f}  "
            f"gate={rec['gate']}  mean_g={rec['mean_gate']}  "
            f"mean_pi={rec['mean_pi']}  n_sel={rec['val_mean_n_selected']}  "
            f"empty_val={val_m['n_empty_in_eval']}  "
            f"epoch_s={rec['epoch_s']:.1f}  mem_mb={rec['peak_mem_mb']}"
        )

        metric = val_m["auprc"] if GNN_MONITOR == "val_AUPRC" else val_m["auroc"]
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

        meta_now = {
            **meta_base,
            "epoch": epoch,
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
            "history_tail": history[-5:],
        }
        save_checkpoint(
            rdir / "ckpt_last.pt",
            model=model,
            opt=opt,
            epoch=epoch,
            best_metric=best_metric,
            best_state=best_state,
            history=history,
            meta=meta_now,
            is_best=is_best,
        )
        (rdir / "meta_latest.json").write_text(
            json.dumps(meta_now, indent=2, default=str), encoding="utf-8"
        )

        if nan_batch:
            _log("[17] abort: non-finite loss")
            break
        if bad_epochs >= patience:
            _log(
                f"  [17] early stop epoch={epoch} "
                f"patience={patience} min_delta={min_delta}"
            )
            break

    end_meta = {
        **meta_base,
        "best_metric": best_metric,
        "n_history": len(history),
        "last_epoch": history[-1]["epoch"] if history else start_epoch,
        "peak_mem_mb": mem_mb(),
        "status": "completed",
    }
    (rdir / "meta_end.json").write_text(
        json.dumps(end_meta, indent=2, default=str), encoding="utf-8"
    )
    save_checkpoint(
        rdir / "ckpt_end.pt",
        model=model,
        opt=opt,
        epoch=history[-1]["epoch"] if history else start_epoch,
        best_metric=best_metric,
        best_state=best_state,
        history=history,
        meta=end_meta,
        is_best=False,
    )

    report = {
        "out_dir": str(rdir),
        "best_metric": best_metric,
        "history": history,
        "subject_audit": audit,
        "pos_weight": pos_w,
        "n_params": n_params,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "empty_train": n_empty_tr,
        "empty_val": n_empty_va,
        "peak_mem_mb": mem_mb(),
        "paper_eligible": False,
        "test_built": False,
    }
    (rdir / "train_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    _log("[17] train DONE")
    _log(json.dumps({k: report[k] for k in report if k != "history"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNN risk training Stage B/D")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--n-nodes", type=int, default=16, help="dry-run synthetic nodes")
    p.add_argument("--batch-size", type=int, default=BATCH_GRAPHS)
    p.add_argument("--seed", type=int, default=GNN_SEED)
    p.add_argument(
        "--graph-pack",
        type=str,
        default=str(DEFAULT_GRAPH_PACK),
        help="directory with graph_pack_train.pt / graph_pack_val.pt",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--branch",
        type=str,
        default="C",
        help="head ablation: C/full | A/act_static | B/act_sparse_static | AG/A+G | G/graph_static",
    )
    p.add_argument(
        "--readout",
        type=str,
        default="pool",
        help="graph readout: pool | additive | additive_ms | additive_gate | additive_l0 | additive_topk",
    )
    p.add_argument(
        "--n-layers",
        type=int,
        default=N_GNN_LAYERS,
        help="message-passing depth; 3 layers support at most 4-node / 3-hop paths",
    )
    p.add_argument(
        "--use-node-length",
        action="store_true",
        help="append L/12 to node features (84-node packs). Default off so 252 ckpts stay loadable.",
    )
    p.add_argument(
        "--eval-ckpt",
        type=str,
        default=None,
        help="eval-only: load this ckpt on val pack (no train, no test)",
    )
    p.add_argument(
        "--residual-from-b",
        type=str,
        default=None,
        help="freeze B (act+sparse+static+head) from this ckpt; train gate*delta_graph",
    )
    p.add_argument("--max-stays", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=GNN_PATIENCE)
    p.add_argument("--min-delta", type=float, default=GNN_MIN_DELTA)
    p.add_argument("--lr", type=float, default=GNN_LR)
    p.add_argument("--resume", type=str, default=None, help='path or "auto"')
    p.add_argument(
        "--unlock-test",
        action="store_true",
        help="forbidden in Stage D; kept only to hard-refuse",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_ckpt:
        args.dry_run = False
    elif not args.train and not args.dry_run:
        args.dry_run = True
    if args.train and args.dry_run:
        if "--dry-run" in sys.argv and "--train" in sys.argv:
            raise SystemExit("pass only one of --train / --dry-run")
        args.dry_run = False

    set_seed(args.seed)
    device = pick_device(args.device)
    _log(f"[17] device={device} torch={torch.__version__}")

    if args.eval_ckpt:
        run_eval_ckpt(args, device)
        return
    if args.dry_run:
        run_dry_run(device, n_nodes=args.n_nodes, batch_size=min(args.batch_size, 4))
        return

    run_train(args, device)


if __name__ == "__main__":
    main()
