#!/usr/bin/env python3
"""
Freeze 84-node motif families from the frozen Shapelet checkpoint.

1. Load ckpt_best.pt templates only (no val/test labels).
2. Resample each template to 12 points; keep raw amplitude + shape-normalized form.
3. L=6 k={0,1,2} are family anchors; L=4 and L=12 are Hungarian-matched.
4. Inspect aligned curves, then write clinical family labels.
5. Remap existing 252-node train/val events onto (var, motif_family, stage).
6. Never reads test rows; never uses HOSPITAL_MORTALITY for alignment or naming.

Usage:
  python -u scripts/20_map_template_families.py --inspect
  python -u scripts/20_map_template_families.py --freeze --remap-events
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Torch + conda MKL both ship OpenMP; allow the process to start on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.graph_constants import A_MIN_EVENT, STAGE_NAMES  # noqa: E402
from scripts.utils.graph_events import select_active_events  # noqa: E402
from scripts.utils.shapelet_constants import (  # noqa: E402
    FINAL_K,
    FINAL_LENGTHS,
    GAMMA_SOFTDTW,
    SEED,
    SHAPELET_VARS,
)
from scripts.utils.shapelet_family import (  # noqa: E402
    ANCHOR_L,
    DISTANCE_WEIGHTS,
    N_FAMILIES,
    TARGET_LEN,
    align_all_variables,
    apply_family_labels,
    assert_mapping_complete,
    build_motif_node_dictionary,
    build_template_records,
    extract_templates_from_checkpoint,
    family_shape_descriptors,
    mapping_frame,
    remap_activation_long,
)

# Clinical labels are filled AFTER inspecting the aligned frozen templates.
# They are a frozen lookup on (var, motif_family). They are NOT derived from
# val/test mortality, edge support, or GNN metrics.
FROZEN_FAMILY_LABELS: dict[tuple[str, int], str] = {
    # Named from frozen L6 anchors + Hungarian members (2026-08-17 inspect).
    # Not rise/fall/flat by fiat; not derived from mortality.
    ("heart_rate", 0): "高峰回落型",
    ("heart_rate", 1): "低位回升型",
    ("heart_rate", 2): "持续上升型",
    ("sbp", 0): "低位下滑型",
    ("sbp", 1): "大幅波动型",
    ("sbp", 2): "恢复上升型",
    ("dbp", 0): "后期下降型",
    ("dbp", 1): "持续下降型",
    ("dbp", 2): "低位回升型",
    ("mbp", 0): "回落反跳型",
    ("mbp", 1): "缓降持平型",
    ("mbp", 2): "恢复上升型",
    ("respiratory_rate", 0): "低谷回升型",
    ("respiratory_rate", 1): "波动上升型",
    ("respiratory_rate", 2): "末期骤降型",
    ("spo2", 0): "大幅振荡型",
    ("spo2", 1): "低位波动型",
    ("spo2", 2): "高位回落型",
    ("urine_output_ml", 0): "进行性下降型",
    ("urine_output_ml", 1): "低位波动型",
    ("urine_output_ml", 2): "低位缓升型",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_ckpt() -> Path:
    return PROJ / "data/processed/shapelet/final/final_7var_L4_6_12_K3/ckpt_best.pt"


def default_events() -> Path:
    return PROJ / "data/processed/shapelet/events/events_final_7var_L4_6_12_K3"


def default_out() -> Path:
    return PROJ / "data/processed/shapelet/families/final_7var_L4_6_12_K3"


def plot_variable_families(
    records,
    assignments,
    labels: dict[tuple[str, int], str],
    out_dir: Path,
) -> None:
    rec_ix = {(r.var, r.L, r.k): r for r in records}
    vars_ = []
    for a in assignments:
        if a.var not in vars_:
            vars_.append(a.var)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    t = np.linspace(0.0, 1.0, num=TARGET_LEN)
    colors = {4: "#1f77b4", 6: "#d62728", 12: "#2ca02c"}
    styles = {4: "--", 6: "-", 12: ":"}
    for font_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        try:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue
    for var in vars_:
        fig, axes = plt.subplots(1, N_FAMILIES, figsize=(12.5, 3.4), sharey=True)
        for fam, ax in enumerate(axes):
            members = [a for a in assignments if a.var == var and a.motif_family == fam]
            for a in sorted(members, key=lambda x: x.L):
                rec = rec_ix[(a.var, a.L, a.k)]
                ax.plot(
                    t,
                    rec.shape,
                    color=colors[a.L],
                    linestyle=styles[a.L],
                    linewidth=2.0 if a.is_anchor else 1.3,
                    label=f"L{a.L} k{a.k}" + (" (anchor)" if a.is_anchor else ""),
                )
            lab = labels.get((var, fam), f"family{fam}")
            ax.set_title(f"{var}\n{lab} (f{fam})", fontsize=9)
            ax.axhline(0.0, color="0.7", linewidth=0.6)
            ax.set_xlabel("resampled t")
            if fam == 0:
                ax.set_ylabel("shape-normalized")
            ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"family_shapes_{var}.png", dpi=140)
        plt.close(fig)


def print_inspect_tables(mapping: pd.DataFrame, desc: pd.DataFrame) -> None:
    _log("")
    _log("=== L6 anchors (family identity = original k) ===")
    anc = mapping.loc[mapping["is_anchor"]].copy()
    _log(
        anc[["var", "k", "motif_family", "corr_raw"]]
        .sort_values(["var", "k"])
        .to_string(index=False)
    )
    _log("")
    _log("=== Independent nearest-anchor (L4 / L12; 1-1 NOT forced) ===")
    non = mapping.loc[~mapping["is_anchor"]].copy()
    cols = [
        "var",
        "L",
        "k",
        "motif_family",
        "family_cost",
        "second_best_cost",
        "family_margin",
        "family_low_confidence",
        "cost_f0",
        "cost_f1",
        "cost_f2",
    ]
    cols = [c for c in cols if c in non.columns]
    _log(non[cols].sort_values(["var", "L", "k"]).to_string(index=False))
    if "family_low_confidence" in mapping.columns:
        n_low = int(mapping["family_low_confidence"].sum())
        _log(f"  low_confidence templates: {n_low} / {len(mapping)}")
    _log("")
    _log("=== Family descriptors (for clinical naming; no mortality) ===")
    show = desc.drop(columns=["mean_shape", "anchor_shape"], errors="ignore")
    _log(show.sort_values(["var", "motif_family"]).to_string(index=False))


def write_mapping_artifacts(
    out_dir: Path,
    *,
    mapping: pd.DataFrame,
    desc: pd.DataFrame,
    motif_nodes: pd.DataFrame | None,
    ckpt_path: Path,
    labels: dict[tuple[str, int], str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(out_dir / "template_family_mapping.parquet", index=False)
    mapping.to_csv(out_dir / "template_family_mapping.csv", index=False)
    desc_store = desc.copy()
    desc_store.to_parquet(out_dir / "family_shape_descriptors.parquet", index=False)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "anchor_L": ANCHOR_L,
        "n_families": N_FAMILIES,
        "target_len": TARGET_LEN,
        "gamma": GAMMA_SOFTDTW,
        "distance_weights": {
            "soft_dtw": DISTANCE_WEIGHTS[0],
            "corr": DISTANCE_WEIGHTS[1],
            "diff_dir": DISTANCE_WEIGHTS[2],
            "slope": DISTANCE_WEIGHTS[3],
        },
        "shape_norm": "(template - template[0]) / peak_to_peak",
        "assignment": "independent_nearest_anchor",
        "hungarian": False,
        "low_confidence_rule": "min_cost > 0.50 OR (second_best - best) < 0.05; anchors exempt",
        "labels_source": "frozen_lookup_after_curve_inspection",
        "used_val_mortality": False,
        "used_test": False,
        "family_labels": {f"{v}|f{f}": lab for (v, f), lab in sorted(labels.items())},
        "n_mapping_rows": int(len(mapping)),
        "n_motif_nodes": int(len(motif_nodes)) if motif_nodes is not None else None,
    }
    (out_dir / "family_alignment.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if motif_nodes is not None:
        motif_nodes.to_parquet(out_dir / "node_dictionary_motif84.parquet", index=False)
        (out_dir / "node_dictionary_motif84.json").write_text(
            json.dumps(motif_nodes.to_dict(orient="records"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def remap_events_dir(
    *,
    src_events: Path,
    out_events: Path,
    mapping: pd.DataFrame,
    motif_nodes: pd.DataFrame,
    family_dir: Path,
    ckpt_path: Path,
) -> None:
    long_path = src_events / "activations_long.parquet"
    if not long_path.is_file():
        raise FileNotFoundError(long_path)
    long_df = pd.read_parquet(long_path)
    if "split" in long_df.columns and "test" in set(long_df["split"].astype(str)):
        _log("[20] dropping test rows from source activations")
        long_df = long_df.loc[long_df["split"].astype(str) != "test"].copy()

    remapped = remap_activation_long(long_df, mapping, motif_nodes, drop_test=True)
    winners = select_active_events(remapped, a_min=A_MIN_EVENT)

    stay_cols = [c for c in ("ICUSTAY_ID", "SUBJECT_ID", "split", "HOSPITAL_MORTALITY") if c in long_df.columns]
    stay_index = (
        long_df[stay_cols]
        .drop_duplicates(subset=["ICUSTAY_ID"])
        .sort_values("ICUSTAY_ID")
        .reset_index(drop=True)
    )

    out_events.mkdir(parents=True, exist_ok=True)
    remapped.to_parquet(out_events / "activations_long_all_mapped.parquet", index=False)
    # 16 consumes activations_long; winner-only avoids 84-d node_id collisions.
    # Empty stays (no a>=a_min event) are listed in stay_index.parquet so 16
    # still builds empty graphs and does not shrink the train/val roster.
    winners.to_parquet(out_events / "activations_long.parquet", index=False)
    stay_index.to_parquet(out_events / "stay_index.parquet", index=False)
    motif_nodes.to_parquet(out_events / "node_dictionary.parquet", index=False)
    (out_events / "node_dictionary.json").write_text(
        json.dumps(motif_nodes.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    mapping.to_parquet(out_events / "template_family_mapping.parquet", index=False)

    frozen_src = src_events / "shapelet_frozen_qp50.pt"
    if frozen_src.is_file():
        shutil.copy2(frozen_src, out_events / "shapelet_frozen_qp50.pt")

    for name in ("leakage_audit.json",):
        src = src_events / name
        if src.is_file():
            shutil.copy2(src, out_events / name)

    event_policy = {
        "node_scheme": "var_motif_family_stage",
        "n_node_types": int(len(motif_nodes)),
        "n_families_per_var": N_FAMILIES,
        "a_min": A_MIN_EVENT,
        "rank": "top1_per_var_stage",
        "tie_break": ["a_desc", "d_stage_asc", "L_asc", "k_asc", "node_id_lk_asc"],
        "L_role": "event_feature_and_audit",
        "k_role": "audit_only",
        "assignment": "independent_nearest_anchor",
        "family_cost_role": "audit_only",
        "family_low_confidence_role": "audit_only",
        "low_confidence_rule": "min_cost > 0.50 OR margin < 0.05; anchors exempt",
        "mapping_source": str((family_dir / "template_family_mapping.parquet").resolve()),
        "used_val_mortality": False,
        "test_exported": False,
    }
    (out_events / "event_policy.json").write_text(
        json.dumps(event_policy, indent=2),
        encoding="utf-8",
    )

    n_stay = int(stay_index["ICUSTAY_ID"].nunique()) if len(stay_index) else 0
    n_active_stay = int(winners["ICUSTAY_ID"].nunique()) if len(winners) else 0
    manifest = {
        "run_name": out_events.name,
        "node_scheme": "var_motif_family_stage",
        "n_node_types": int(len(motif_nodes)),
        "source_events": str(src_events.resolve()),
        "source_checkpoint": str(ckpt_path.resolve()),
        "source_checkpoint_sha256": sha256_file(ckpt_path),
        "mapping_sha256": sha256_file(family_dir / "template_family_mapping.parquet"),
        "n_rows_all_mapped": int(len(remapped)),
        "n_winner_rows": int(len(winners)),
        "n_stays": n_stay,
        "n_stays_with_events": n_active_stay,
        "n_empty_stays": int(n_stay - n_active_stay),
        "active_per_stay": float(len(winners) / max(n_stay, 1)),
        "splits": sorted(set(stay_index["split"].astype(str))) if len(stay_index) else [],
        "test_built": False,
        "used_val_mortality": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
    }
    if "split" in stay_index.columns:
        for sp in ("train", "val"):
            sub_stays = stay_index.loc[stay_index["split"] == sp]
            sub_win = winners.loc[winners["split"] == sp] if len(winners) else winners
            manifest[f"{sp}_stays"] = int(sub_stays["ICUSTAY_ID"].nunique()) if len(sub_stays) else 0
            manifest[f"{sp}_winners"] = int(len(sub_win))
            manifest[f"{sp}_stays_with_events"] = (
                int(sub_win["ICUSTAY_ID"].nunique()) if len(sub_win) else 0
            )
    (out_events / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    _log(f"[20] remapped events -> {out_events}")
    _log(
        f"  all_mapped={len(remapped)}  winners={len(winners)}  "
        f"stays={n_stay}  nodes={len(motif_nodes)}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze 84-node motif-family mapping")
    p.add_argument("--checkpoint", type=str, default=str(default_ckpt()))
    p.add_argument("--events", type=str, default=str(default_events()))
    p.add_argument("--out-dir", type=str, default=str(default_out()))
    p.add_argument(
        "--out-events",
        type=str,
        default=str(PROJ / "data/processed/shapelet/events/events_final_7var_motif84"),
    )
    p.add_argument("--inspect", action="store_true", help="align + print descriptors, no freeze")
    p.add_argument("--freeze", action="store_true", help="write frozen mapping + labels")
    p.add_argument("--remap-events", action="store_true", help="write 84-node events dir")
    p.add_argument(
        "--allow-numeric-labels",
        action="store_true",
        help="if frozen labels missing, use family0/1/2 placeholders",
    )
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def resolve_labels(desc: pd.DataFrame, allow_numeric: bool) -> dict[tuple[str, int], str]:
    labels = dict(FROZEN_FAMILY_LABELS)
    if labels:
        needed = {(str(r.var), int(r.motif_family)) for r in desc.itertuples()}
        missing = needed - set(labels)
        extra = set(labels) - needed
        if missing:
            raise SystemExit(f"FROZEN_FAMILY_LABELS missing {sorted(missing)}")
        if extra:
            _log(f"[warn] unused label keys: {sorted(extra)}")
        return labels
    if not allow_numeric:
        raise SystemExit(
            "FROZEN_FAMILY_LABELS is empty. Run --inspect, assign clinical names "
            "from the aligned curves (not mortality), then --freeze."
        )
    return {(str(r.var), int(r.motif_family)): f"family{int(r.motif_family)}" for r in desc.itertuples()}


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (args.inspect or args.freeze or args.remap_events):
        args.inspect = True

    _log("=" * 60)
    _log("20: motif-family mapping from frozen Shapelet templates")
    _log("=" * 60)
    _log(f"  checkpoint={ckpt_path}")
    _log(f"  out={out_dir}")

    templates = extract_templates_from_checkpoint(ckpt_path)
    _log(f"  templates={len(templates)}  vars={list(SHAPELET_VARS)}")
    records = build_template_records(templates, target_len=TARGET_LEN)
    assigned = align_all_variables(
        records,
        vars_=SHAPELET_VARS,
        n_families=N_FAMILIES,
        anchor_L=ANCHOR_L,
    )
    mapping = mapping_frame(assigned)
    desc = family_shape_descriptors(records, assigned)
    print_inspect_tables(mapping, desc)

    fig_dir = out_dir / "figures"
    plot_variable_families(records, assigned, {}, fig_dir)
    _log(f"[20] wrote unlabeled family plots -> {fig_dir}")

    # Always dump unlabeled inspect tables so naming can be audited later.
    mapping.to_csv(out_dir / "inspect_assignment.csv", index=False)
    desc.drop(columns=["mean_shape", "anchor_shape"], errors="ignore").to_csv(
        out_dir / "inspect_descriptors.csv",
        index=False,
    )

    if args.inspect and not args.freeze and not args.remap_events:
        _log("[20] inspect-only DONE (no labels frozen)")
        return

    labels = resolve_labels(desc, allow_numeric=args.allow_numeric_labels)
    mapping = apply_family_labels(mapping, labels)
    assert_mapping_complete(
        mapping,
        vars_=SHAPELET_VARS,
        lengths=FINAL_LENGTHS,
        k=FINAL_K,
        n_families=N_FAMILIES,
    )
    motif_nodes = build_motif_node_dictionary(mapping)
    plot_variable_families(records, assigned, labels, fig_dir)

    if args.freeze:
        write_mapping_artifacts(
            out_dir,
            mapping=mapping,
            desc=desc,
            motif_nodes=motif_nodes,
            ckpt_path=ckpt_path,
            labels=labels,
        )
        _log(f"[20] froze mapping -> {out_dir / 'template_family_mapping.parquet'}")
        _log(f"  n_templates={len(mapping)}  n_nodes={len(motif_nodes)}")

    if args.remap_events:
        if not args.freeze:
            # still write mapping so remap has a hashable artifact
            write_mapping_artifacts(
                out_dir,
                mapping=mapping,
                desc=desc,
                motif_nodes=motif_nodes,
                ckpt_path=ckpt_path,
                labels=labels,
            )
        remap_events_dir(
            src_events=Path(args.events),
            out_events=Path(args.out_events),
            mapping=mapping,
            motif_nodes=motif_nodes,
            family_dir=out_dir,
            ckpt_path=ckpt_path,
        )

    _log("[20] DONE")


if __name__ == "__main__":
    main()
