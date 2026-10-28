#!/usr/bin/env python3
"""Summarize already-trained GNN/table ablations. Does not retrain or read a new test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.path_protocol import refuse_test_path  # noqa: E402

GNN_ROOT = PROJ / "data/processed/gnn"
REPORT = PROJ / "outputs/reports/25_GNN已有消融汇总.md"

# Frozen paper table: directories that already exist from prior locked runs.
ARMS = (
    ("gnn_ablate_A_act_static", "A 激活袋+静态"),
    ("gnn_ablate_B_act_sparse_static", "B 激活+稀疏检验+静态"),
    ("gnn_ablate_AG_act_static_graph", "旧池化 AG"),
    ("gnn_full_final_lr2e4", "旧池化 C"),
    ("gnn_additive_G_graph_static", "可分解 G 图+静态"),
    ("gnn_additive_AG_act_static", "可分解 AG"),
    ("gnn_additive_G_motif84", "84节点 可分解 G"),
    ("gnn_additive_gate_G_motif84", "连续门控"),
    ("gnn_additive_l0_G_motif84", "Hard-Concrete L0"),
    ("gnn_additive_topk_G_motif84", "STE top-k"),
    ("gnn_additive_G3_multiscale_seed42", "三层多尺度 G"),
    ("gnn_residual_gate_from_B", "残差门（冻结 B）"),
)


def _best_row(hist: list[dict]) -> dict:
    good = [h for h in hist if h.get("val_auprc") is not None]
    if not good:
        return {}
    return max(good, key=lambda h: float(h["val_auprc"]))


def main() -> None:
    refuse_test_path(GNN_ROOT)
    refuse_test_path(REPORT)
    lines = [
        "# GNN / 表格臂已有消融汇总",
        "",
        "> 只读已有 `train_report.json` 的 best val AUPRC。不重训，不开新 test。",
        "",
        "| 臂 | 目录 | best epoch | val AUPRC | val AUROC | val Brier |",
        "|---|---|---:|---:|---:|---:|",
    ]
    rows = []
    for dirname, label in ARMS:
        report = GNN_ROOT / dirname / "train_report.json"
        if not report.is_file():
            lines.append(f"| {label} | `{dirname}` | — | 缺失 | — | — |")
            continue
        payload = json.loads(report.read_text(encoding="utf-8"))
        best = _best_row(list(payload.get("history") or []))
        rec = {
            "label": label,
            "dir": dirname,
            "epoch": best.get("epoch"),
            "val_auprc": best.get("val_auprc"),
            "val_auroc": best.get("val_auroc"),
            "val_brier": best.get("val_brier"),
        }
        rows.append(rec)
        lines.append(
            f"| {label} | `{dirname}` | {rec['epoch']} | "
            f"{float(rec['val_auprc']):.4f} | {float(rec['val_auroc']):.4f} | "
            f"{float(rec['val_brier']):.4f} |"
        )
    out = GNN_ROOT / "gnn_ablation_summary.json"
    out.write_text(json.dumps({"arms": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[30] wrote {REPORT}", flush=True)


if __name__ == "__main__":
    main()
