#!/usr/bin/env python3
"""
Export frozen Shapelet events for evolution graphs (docs/13 §3, §10.1).

1. Load ckpt_best.pt (templates frozen; no further Shapelet training)
2. Calibrate q = train P50 of valid stage distances (train only)
3. Save shapelet_frozen_qp50.pt
4. Export train/val activations_long + node dictionary
5. Never builds/exports test unless --unlock-test (default off)

Usage:
  python -u scripts/15_export_shapelet_events.py ^
    --checkpoint data/processed/shapelet/final/final_7var_L4_6_12_K3/ckpt_best.pt ^
    --q-policy train_p50 --splits train,val --device auto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.graph_constants import A_MIN_EVENT, STAGE_NAMES  # noqa: E402
from scripts.utils.graph_events import (  # noqa: E402
    build_node_dictionary,
    node_dictionary_frame,
    select_active_events,
)
from scripts.utils.shapelet_constants import (  # noqa: E402
    BATCH_STAYS,
    SEED,
    SHAPELET_STATIC_COLS,
    SHAPELET_VARS,
    STAGE_BOUNDS,
    T_HOURS,
)
from scripts.utils.shapelet_model import ShapeletRiskModel  # noqa: E402
from scripts.utils.shapelet_prep import (  # noqa: E402
    ShapeletStayDataset,
    prepare_splits,
)


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_model(ckpt_path: Path, device: torch.device) -> tuple[ShapeletRiskModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = dict(ckpt.get("meta") or {})
    vars_ = tuple(meta.get("vars") or SHAPELET_VARS)
    lengths = tuple(meta.get("lengths") or (4, 6, 12))
    k = int(meta.get("k") or 3)
    head = str(ckpt.get("head_mode") or meta.get("head") or "combined")
    pos_w = float(meta.get("pos_weight") or 1.0)
    model = ShapeletRiskModel(
        vars_=vars_,
        lengths=lengths,
        k=k,
        n_static=len(SHAPELET_STATIC_COLS),
        pos_weight=pos_w,
        head_mode=head,
        q_init=0.0,
    ).to(device)
    state = ckpt.get("best_state") or ckpt["model_state"]
    model.load_state_dict(state)
    model.eval()
    model.freeze_templates(True)
    # Prefer activation_names / keys from ckpt for dictionary order
    if ckpt.get("activation_names"):
        model._export_activation_names = list(ckpt["activation_names"])  # type: ignore[attr-defined]
    if ckpt.get("keys"):
        # rebuild keys order check
        keys = [tuple(x) if not isinstance(x, tuple) else x for x in ckpt["keys"]]
        if list(model.template_keys()) != list(keys):
            _log(
                f"[warn] template keys mismatch ckpt vs model: "
                f"{len(keys)} vs {model.n_templates} — using model order"
            )
    return model, {
        "ckpt": ckpt,
        "meta": meta,
        "vars": vars_,
        "lengths": lengths,
        "k": k,
        "head": head,
        "activation_names": list(
            getattr(model, "_export_activation_names", model.activation_names())
        ),
    }


@torch.no_grad()
def collect_train_stage_distances(
    model: ShapeletRiskModel,
    loader: DataLoader,
    device: torch.device,
    max_per_template: int = 80_000,
) -> torch.Tensor:
    """Same protocol as 11_shapelet_learn: valid stage d < 1e3 only."""
    model.eval()
    buckets: list[list[torch.Tensor]] = [[] for _ in range(model.n_templates)]
    var_index = {v: i for i, v in enumerate(model.vars)}
    for batch in tqdm(loader, desc="collect train d for q", unit="batch"):
        x = batch["x"].to(device)
        m = batch["m"].to(device)
        dlt = batch["delta"].to(device)
        for ti, (v, _L, _k) in enumerate(model.template_keys()):
            j = var_index[v]
            d_stage, _t, _dist = model.match_one_template(
                x[:, :, j], m[:, :, j], dlt[:, :, j], ti
            )
            flat = d_stage.reshape(-1)
            keep = flat < 1.0e3
            if keep.any():
                buckets[ti].append(flat[keep].detach().cpu())
    rows = []
    for parts in buckets:
        if not parts:
            rows.append(torch.zeros(1))
            continue
        cat = torch.cat(parts, dim=0)
        if cat.numel() > max_per_template:
            idx = torch.linspace(0, cat.numel() - 1, steps=max_per_template).long()
            cat = cat[idx]
        rows.append(cat)
    n_min = min(r.numel() for r in rows)
    return torch.stack([r[:n_min] for r in rows], dim=0)


def _stage_lo_hi() -> list[tuple[str, int, int]]:
    return [(n, int(lo), int(hi)) for n, lo, hi in STAGE_BOUNDS]


@torch.no_grad()
def export_split_activations(
    model: ShapeletRiskModel,
    bundle,
    device: torch.device,
    batch_size: int,
    activation_names: list[str],
    split_label: str,
) -> pd.DataFrame:
    """Long table: one row per (stay, node_type)."""
    loader = DataLoader(
        ShapeletStayDataset(bundle),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    nodes = build_node_dictionary(activation_names)
    n_nodes = len(nodes)
    stage_meta = _stage_lo_hi()
    rows: list[dict] = []

    # map stay_id -> subject / y / split from bundle
    id_to_i = {int(s): i for i, s in enumerate(bundle.stay_ids.tolist())}

    model.eval()
    for batch in tqdm(loader, desc=f"export {split_label}", unit="batch"):
        x = batch["x"].to(device)
        m = batch["m"].to(device)
        dlt = batch["delta"].to(device)
        stay_ids = batch["stay_id"].cpu().numpy().astype(np.int64)
        a, d_stages, t_star = model.forward_activations(x, m, dlt)
        # a: (B, n_act), d_stages: (B, n_tmpl, n_stages), t_star same
        bsz = a.shape[0]
        a_np = a.cpu().numpy()
        d_np = d_stages.cpu().numpy()
        t_np = t_star.cpu().numpy()

        for bi in range(bsz):
            sid = int(stay_ids[bi])
            bi_bundle = id_to_i.get(sid, -1)
            subject = (
                int(bundle.subject_ids[bi_bundle]) if bi_bundle >= 0 else -1
            )
            y = float(bundle.y[bi_bundle]) if bi_bundle >= 0 else 0.0
            # flatten template major, stage minor (matches activation_names)
            for ni, node in enumerate(nodes):
                ti = ni // model.n_stages
                si = ni % model.n_stages
                d_s = float(d_np[bi, ti, si])
                t_s = float(t_np[bi, ti, si])
                is_valid = bool(d_s < 1.0e3)
                st_name, lo, hi = stage_meta[si]
                width = max(hi - lo, 1)
                t_within = float(np.clip((t_s - lo) / width, 0.0, 1.0))
                rows.append(
                    {
                        "ICUSTAY_ID": sid,
                        "SUBJECT_ID": subject,
                        "split": split_label,
                        "HOSPITAL_MORTALITY": np.int8(y),
                        "node_id": np.int32(node.node_id),
                        "var": node.var,
                        "L": np.int16(node.L),
                        "k": np.int16(node.k),
                        "stage": st_name,
                        "stage_idx": np.int8(si),
                        "a": np.float32(a_np[bi, ni]),
                        "d_stage": np.float32(d_s),
                        "t_star": np.float32(t_s),
                        "t_star_norm": np.float32(t_s / float(T_HOURS)),
                        "t_within_stage": np.float32(t_within),
                        "is_valid": is_valid,
                    }
                )

    df = pd.DataFrame(rows)
    # active mask via rank-1 policy
    active = select_active_events(df, a_min=A_MIN_EVENT)
    active_keys = set(
        zip(
            active["ICUSTAY_ID"].astype(int),
            active["node_id"].astype(int),
        )
    )
    df["is_active"] = [
        (int(r.ICUSTAY_ID), int(r.node_id)) in active_keys for r in df.itertuples()
    ]
    return df


def leakage_audit(df: pd.DataFrame) -> dict:
    train_subj = set(df.loc[df["split"] == "train", "SUBJECT_ID"].astype(int))
    val_subj = set(df.loc[df["split"] == "val", "SUBJECT_ID"].astype(int))
    test_subj = set(df.loc[df["split"] == "test", "SUBJECT_ID"].astype(int))
    return {
        "train_subjects": len(train_subj),
        "val_subjects": len(val_subj),
        "test_subjects": len(test_subj),
        "train_val_overlap": len(train_subj & val_subj),
        "train_test_overlap": len(train_subj & test_subj),
        "val_test_overlap": len(val_subj & test_subj),
        "test_exported": "test" in set(df["split"].unique()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Shapelet events + calibrate q")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            PROJ
            / "data/processed/shapelet/final/final_7var_L4_6_12_K3/ckpt_best.pt"
        ),
    )
    p.add_argument("--q-policy", choices=("train_p50", "zero"), default="train_p50")
    p.add_argument("--q-quantile", type=float, default=0.5)
    p.add_argument("--splits", type=str, default="train,val")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--batch-stays", type=int, default=BATCH_STAYS)
    p.add_argument("--max-stays", type=int, default=None, help="debug subsample")
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="events output folder name (default from checkpoint parent)",
    )
    p.add_argument(
        "--unlock-test",
        action="store_true",
        help="explicitly allow test split export (default: never)",
    )
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "test" in splits and not args.unlock_test:
        raise SystemExit(
            "refusing to export test without --unlock-test "
            "(docs/13: test locked until explicit unlock)"
        )
    if args.unlock_test and "test" not in splits:
        splits.append("test")

    _log("=" * 60)
    _log("15: export Shapelet events + q calibration")
    _log("=" * 60)
    _log(f"  checkpoint={ckpt_path}")
    _log(f"  device={device}  q_policy={args.q_policy}  splits={splits}")

    model, info = load_frozen_model(ckpt_path, device)
    act_names = info["activation_names"]
    n_nodes = len(act_names)
    _log(
        f"  templates={model.n_templates}  node_types={n_nodes}  "
        f"head={info['head']}  lengths={info['lengths']} k={info['k']}"
    )
    if n_nodes != model.n_act:
        _log(f"[warn] n_act model={model.n_act} names={n_nodes}")

    run_name = args.run_name or f"events_{ckpt_path.parent.name}"
    out_dir = PROJ / "data" / "processed" / "shapelet" / "events" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("[15] prepare train/val tensors (train-only scalers)…")
    packs = prepare_splits(
        vars_=info["vars"],
        static_cols=SHAPELET_STATIC_COLS,
        max_stays_per_split=args.max_stays,
        seed=args.seed,
        show_progress=True,
        build_test=bool(args.unlock_test and "test" in splits),
    )
    value_scaler = packs["_value_scaler"]
    static_scaler = packs["_static_scaler"]
    train_b = packs["train"]

    # --- q calibration ---
    q_values = [float(v) for v in model.q.detach().cpu().tolist()]
    if args.q_policy == "train_p50":
        _log(
            f"[15] calibrate q = train P{int(args.q_quantile * 100)} "
            "(templates frozen; no Shapelet update)"
        )
        train_loader = DataLoader(
            ShapeletStayDataset(train_b),
            batch_size=args.batch_stays,
            shuffle=False,
        )
        d_flat = collect_train_stage_distances(model, train_loader, device)
        model.set_q_from_distances(d_flat, quantile=float(args.q_quantile))
        q_values = [float(v) for v in model.q.detach().cpu().tolist()]
        _log(
            f"  d samples/template={d_flat.shape[1]}  "
            f"q_mean={float(np.mean(q_values)):.4f}  "
            f"q_min={min(q_values):.4f}  q_max={max(q_values):.4f}"
        )
    else:
        _log("[15] q_policy=zero — keeping q=0 (not recommended for graph events)")

    # --- freeze package ---
    frozen_path = out_dir / "shapelet_frozen_qp50.pt"
    frozen_payload = {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "keys": model.template_keys(),
        "activation_names": act_names,
        "head_mode": info["head"],
        "q_policy": args.q_policy,
        "q_quantile": float(args.q_quantile) if args.q_policy == "train_p50" else None,
        "q": q_values,
        "value_scaler": {
            "mean": value_scaler.mean,
            "std": value_scaler.std,
        },
        "static_scaler": {
            "mean": static_scaler.mean.tolist(),
            "std": static_scaler.std.tolist(),
            "cols": list(SHAPELET_STATIC_COLS),
        },
        "stage_bounds": list(STAGE_BOUNDS),
        "meta": {
            **info["meta"],
            "source_checkpoint": str(ckpt_path.resolve()),
            "source_checkpoint_sha256": sha256_file(ckpt_path),
            "q_policy": args.q_policy,
            "q_export": "train_p50" if args.q_policy == "train_p50" else "zero",
            "templates_frozen": True,
            "paper_eligible_structure": True,
            "test_built": False,
        },
        "seed": args.seed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.save(frozen_payload, frozen_path)
    _log(f"[15] saved {frozen_path}")

    # --- node dictionary ---
    nodes = build_node_dictionary(act_names)
    nd_df = node_dictionary_frame(nodes)
    nd_df.to_parquet(out_dir / "node_dictionary.parquet", index=False)
    (out_dir / "node_dictionary.json").write_text(
        json.dumps(nd_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    _log(f"[15] node_dictionary N={len(nd_df)}")

    # --- export activations ---
    parts = []
    for sp in splits:
        if sp not in packs or sp.startswith("_"):
            if sp == "test" and "test" not in packs:
                raise SystemExit("test pack not built")
            continue
        df = export_split_activations(
            model,
            packs[sp],
            device,
            args.batch_stays,
            act_names,
            split_label=sp,
        )
        parts.append(df)
        n_stay = df["ICUSTAY_ID"].nunique()
        n_act = int(df["is_active"].sum())
        _log(
            f"  {sp}: stays={n_stay} rows={len(df)} active_events={n_act} "
            f"active/stay={n_act / max(n_stay, 1):.2f}"
        )

    long_df = pd.concat(parts, ignore_index=True)
    long_path = out_dir / "activations_long.parquet"
    long_df.to_parquet(long_path, index=False)
    _log(f"[15] wrote {long_path} rows={len(long_df)}")

    audit = leakage_audit(long_df)
    if audit["train_val_overlap"] > 0:
        raise RuntimeError(f"SUBJECT_ID train/val overlap: {audit}")
    (out_dir / "leakage_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    _log(f"[15] leakage audit: {audit}")

    event_policy = {
        "a_min": A_MIN_EVENT,
        "rank": "top1_per_var_stage",
        "tie_break": ["a_desc", "d_stage_asc", "L_asc", "k_asc", "node_id_asc"],
        "q_policy": args.q_policy,
        "q_quantile": args.q_quantile,
        "max_active_events": len(info["vars"]) * len(STAGE_NAMES),
    }
    (out_dir / "event_policy.json").write_text(
        json.dumps(event_policy, indent=2), encoding="utf-8"
    )

    manifest = {
        "run_name": run_name,
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "frozen_qp50": str(frozen_path.resolve()),
        "frozen_qp50_sha256": sha256_file(frozen_path),
        "n_node_types": len(nd_df),
        "splits": splits,
        "test_built": False,
        "q_policy": args.q_policy,
        "event_policy": event_policy,
        "leakage_audit": audit,
        "n_rows": int(len(long_df)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _log("[15] DONE")
    _log(f"  events dir: {out_dir}")
    _log("  next: python -u scripts/16_build_evolution_graphs.py --events " + str(out_dir))


if __name__ == "__main__":
    main()
