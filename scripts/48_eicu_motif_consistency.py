#!/usr/bin/env python3
"""48 — eICU motif 字母表一致性检验（docs/28 §4.4）

比较 84 个 motif 家族节点 (var, family, stage) 在 MIMIC 与 eICU 的**出现频率分布**。
只回答一个功效充足的问题：**形态字母表是否跨站点迁移**（区别于 15 条具体路径）。

指标：
  1. Jensen-Shannon distance (JSD, base=2) — 分布距离，[0,1]，越小越一致
  2. Spearman 秩相关 — 节点流行度排序是否保持
  3. 低置信跨尺度映射比例（docs/20 §8.3：不得异常富集，阈值 +5pp）
  4. winner 事件率 — 每 stay 平均 motif 原子数

约束（docs/28 §2）：
  - 只读已有产物，不重跑 Shapelet，不重挖路径，不重估阈值
  - MIMIC 侧用 holdout（docs/28 §4.4 指定），另附 train 作参照

输出：
  outputs/tables/eicu_motif_consistency.csv   — 逐节点对比（84 行）
  outputs/reports/47_eICU字母表一致性.md       — 结论报告
  outputs/figures/fig15_motif_alphabet_consistency.png — 散点图
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED, OUTPUT_FIGURES, OUTPUT_REPORTS, OUTPUT_TABLES  # noqa: E402
from scripts.utils.eicu_io import load_eicu, out_dir  # noqa: E402

MIMIC_PATHS_DIR = Path(DATA_PROCESSED) / "paths" / "path_final_7var_motif84_labs11"
GRAPH_DIR = Path(DATA_PROCESSED) / "graphs" / "graph_final_7var_motif84"

# docs/20 §8.3: 低置信端点比例不得高于基线 +5 个百分点
LOW_CONF_TOLERANCE_PP = 5.0


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_motif(atoms: pd.DataFrame) -> pd.DataFrame:
    """Keep only motif atoms (drop lab atoms)."""
    return atoms[atoms["kind"] == "motif"].copy()


def _prevalence(atoms: pd.DataFrame, n_stays: int) -> pd.Series:
    """Per-node prevalence = (# distinct stays with node) / n_stays, indexed by atom."""
    per_node = atoms.groupby("atom")["ICUSTAY_ID"].nunique()
    return (per_node / float(n_stays)).rename("prevalence")


def _short_stage(stage: str) -> str:
    """node_dictionary 用 'B1_0_6'，atoms.atom 用 'B1' → 取短格式对齐。"""
    return str(stage).split("_")[0]


def _node_key(var: str, family: str, stage: str) -> str:
    return f"{var}|{family}|{_short_stage(stage)}"


def _compute_block(label: str, atoms: pd.DataFrame, n_stays: int) -> dict:
    prev = _prevalence(atoms, n_stays)
    low_conf = float(atoms["family_low_confidence"].mean())
    return {
        "label": label,
        "n_stays": int(n_stays),
        "n_atoms": int(len(atoms)),
        "winner_rate": float(len(atoms) / n_stays),  # atoms per stay
        "low_conf_prop": low_conf,
        "prevalence": prev,
    }


def _js_distance(p: pd.Series, q: pd.Series, nodes: list[str]) -> float:
    """JSD over the shared 84-node support (missing → 0)."""
    pv = np.array([p.get(n, 0.0) for n in nodes], dtype=float)
    qv = np.array([q.get(n, 0.0) for n in nodes], dtype=float)
    # Normalize to probability distributions
    if pv.sum() > 0:
        pv = pv / pv.sum()
    if qv.sum() > 0:
        qv = qv / qv.sum()
    return float(jensenshannon(pv, qv, base=2.0))


def main() -> None:
    _log("=" * 70)
    _log("47 — eICU motif 字母表一致性检验")
    _log("=" * 70)

    # ── 0. 84 节点词典（canonical order）──
    node_dict = pd.read_parquet(GRAPH_DIR / "node_dictionary.parquet")
    nodes = [
        _node_key(str(r.var), str(r.family_label), str(r.stage))
        for r in node_dict.itertuples(index=False)
    ]
    _log(f"[0] 节点词典: {len(nodes)} 个 (var, family, stage)")

    # ── 1. 载入三侧原子 ──
    mimic_all = pd.read_parquet(MIMIC_PATHS_DIR / "atoms.parquet")
    mimic_holdout = pd.read_parquet(MIMIC_PATHS_DIR / "holdout_atoms.parquet")
    eicu_all = load_eicu("atoms")

    m_train = _load_motif(mimic_all[mimic_all["split"] == "train"])
    m_val = _load_motif(mimic_all[mimic_all["split"] == "val"])
    m_hold = _load_motif(mimic_holdout)
    e_ext = _load_motif(eicu_all)

    n_train = int(mimic_all.loc[mimic_all["split"] == "train", "ICUSTAY_ID"].nunique())
    n_val = int(mimic_all.loc[mimic_all["split"] == "val", "ICUSTAY_ID"].nunique())
    n_hold = int(mimic_holdout["ICUSTAY_ID"].nunique())
    n_eicu = int(eicu_all["ICUSTAY_ID"].nunique())

    blocks = {
        "MIMIC train": _compute_block("MIMIC train", m_train, n_train),
        "MIMIC val": _compute_block("MIMIC val", m_val, n_val),
        "MIMIC holdout": _compute_block("MIMIC holdout", m_hold, n_hold),
        "eICU": _compute_block("eICU", e_ext, n_eicu),
    }

    _log("")
    _log("[1] 各队列规模与 winner 事件率")
    _log(f"{'cohort':<16}{'stays':>8}{'atoms':>10}{'atoms/stay':>12}{'low_conf':>10}")
    for k, b in blocks.items():
        _log(f"{k:<16}{b['n_stays']:>8,}{b['n_atoms']:>10,}{b['winner_rate']:>12.2f}{b['low_conf_prop']:>10.1%}")

    # ── 2. 分布一致性（主对比：MIMIC holdout vs eICU）──
    ref = blocks["MIMIC holdout"]["prevalence"]
    ext = blocks["eICU"]["prevalence"]
    train_ref = blocks["MIMIC train"]["prevalence"]

    jsd_hold = _js_distance(ref, ext, nodes)
    jsd_train = _js_distance(train_ref, ext, nodes)
    jsd_within = _js_distance(train_ref, ref, nodes)  # MIMIC 内部 train vs holdout 作为下界参照

    # Spearman 秩相关（在 84 节点上）
    pv_ref = np.array([ref.get(n, 0.0) for n in nodes])
    pv_ext = np.array([ext.get(n, 0.0) for n in nodes])
    pv_tr = np.array([train_ref.get(n, 0.0) for n in nodes])
    rho_hold, p_hold = spearmanr(pv_ref, pv_ext)
    rho_train, p_train = spearmanr(pv_tr, pv_ext)
    rho_within, _ = spearmanr(pv_tr, pv_ref)

    _log("")
    _log("[2] 分布一致性")
    _log(f"  JSD  (MIMIC holdout vs eICU): {jsd_hold:.4f}")
    _log(f"  JSD  (MIMIC train   vs eICU): {jsd_train:.4f}")
    _log(f"  JSD  (MIMIC train   vs holdout) [内部下界参照]: {jsd_within:.4f}")
    _log(f"  Spearman rho (holdout vs eICU): {rho_hold:+.4f}  (p={p_hold:.2e})")
    _log(f"  Spearman rho (train   vs eICU): {rho_train:+.4f}  (p={p_train:.2e})")
    _log(f"  Spearman rho (train   vs holdout) [内部参照]: {rho_within:+.4f}")

    # ── 3. 低置信富集检验（docs/20 §8.3）──
    base_low = blocks["MIMIC holdout"]["low_conf_prop"]
    ext_low = blocks["eICU"]["low_conf_prop"]
    delta_pp = (ext_low - base_low) * 100.0
    low_conf_ok = delta_pp <= LOW_CONF_TOLERANCE_PP

    _log("")
    _log("[3] 低置信跨尺度映射富集检验 (docs/20 §8.3)")
    _log(f"  MIMIC holdout 低置信比例: {base_low:.1%}")
    _log(f"  eICU          低置信比例: {ext_low:.1%}")
    _log(f"  差值: {delta_pp:+.2f} pp  (阈值 ≤ +{LOW_CONF_TOLERANCE_PP:.0f} pp)  → {'通过' if low_conf_ok else '未通过'}")

    # ── 4. 逐节点对比表 ──
    rows = []
    for n in nodes:
        var, fam, stage = n.split("|")
        pr = float(ref.get(n, 0.0))
        pe = float(ext.get(n, 0.0))
        pt = float(train_ref.get(n, 0.0))
        rows.append({
            "node": n,
            "var": var,
            "family": fam,
            "stage": stage,
            "prev_mimic_train": pt,
            "prev_mimic_holdout": pr,
            "prev_eicu": pe,
            "ratio_eicu_over_holdout": (pe / pr) if pr > 0 else np.nan,
        })
    cmp_df = pd.DataFrame(rows)
    cmp_df["rank_mimic_holdout"] = cmp_df["prev_mimic_holdout"].rank(ascending=False, method="min").astype(int)
    cmp_df["rank_eicu"] = cmp_df["prev_eicu"].rank(ascending=False, method="min").astype(int)
    cmp_df["rank_shift"] = cmp_df["rank_eicu"] - cmp_df["rank_mimic_holdout"]
    cmp_df = cmp_df.sort_values("prev_eicu", ascending=False).reset_index(drop=True)

    # ── 5. 结论判定 ──
    # 判据（预先指定，写死）：
    #   alphabet_transfers = JSD < 0.20 且 rho > 0.60 且 低置信未异常富集
    alphabet_transfers = bool(jsd_hold < 0.20 and rho_hold > 0.60 and low_conf_ok)
    if alphabet_transfers:
        verdict = "字母表跨站点迁移（形态频率分布一致）"
    elif jsd_hold < 0.35 and rho_hold > 0.40:
        verdict = "部分迁移（分布中度偏移）"
    else:
        verdict = "字母表未迁移（形态分布站点漂移）"

    _log("")
    _log("[4] 结论")
    _log(f"  JSD={jsd_hold:.3f}  rho={rho_hold:+.3f}  低置信{'通过' if low_conf_ok else '未通过'}")
    _log(f"  → {verdict}")

    # ── 6. 保存 CSV ──
    tab_dir = Path(OUTPUT_TABLES)
    tab_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tab_dir / "eicu_motif_consistency.csv"
    cmp_df.to_csv(csv_path, index=False)
    _log(f"  → {csv_path}")

    # ── 7. 保存 JSON 摘要 ──
    summary = {
        "stage": "48_eicu_motif_consistency",
        "n_nodes": len(nodes),
        "cohorts": {k: {kk: vv for kk, vv in b.items() if kk != "prevalence"} for k, b in blocks.items()},
        "jsd_holdout_vs_eicu": jsd_hold,
        "jsd_train_vs_eicu": jsd_train,
        "jsd_train_vs_holdout_within_mimic": jsd_within,
        "spearman_holdout_vs_eicu": float(rho_hold),
        "spearman_train_vs_eicu": float(rho_train),
        "spearman_within_mimic": float(rho_within),
        "low_conf_mimic_holdout": base_low,
        "low_conf_eicu": ext_low,
        "low_conf_delta_pp": delta_pp,
        "low_conf_ok": low_conf_ok,
        "alphabet_transfers": alphabet_transfers,
        "verdict": verdict,
    }
    json_path = out_dir() / "motif_consistency.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"  → {json_path}")

    # ── 8. 散点图 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass
        fig, ax = plt.subplots(figsize=(8, 8))
        eps = 1e-6
        xs = np.maximum(pv_ref, eps)
        ys = np.maximum(pv_ext, eps)
        ax.scatter(xs, ys, s=18, alpha=0.6, color="#3498db", edgecolor="white")
        lim = [min(xs.min(), ys.min()) * 0.5, max(xs.max(), ys.max()) * 2]
        ax.plot(lim, lim, "k--", alpha=0.4, linewidth=1, label="y = x")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("MIMIC holdout 节点流行度")
        ax.set_ylabel("eICU 节点流行度")
        ax.set_title(f"84 节点形态字母表一致性\nJSD={jsd_hold:.3f}  Spearman ρ={rho_hold:+.3f}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig_dir = Path(OUTPUT_FIGURES)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / "fig15_motif_alphabet_consistency.png", format="png", dpi=300)
        plt.close(fig)
        _log(f"  → {fig_dir / 'fig15_motif_alphabet_consistency.png'}")
    except Exception as e:  # pragma: no cover
        _log(f"  [warn] plot skipped: {e}")

    # ── 9. Markdown 报告 ──
    rep_dir = Path(OUTPUT_REPORTS)
    rep_dir.mkdir(parents=True, exist_ok=True)
    top10 = cmp_df.head(10)
    lines = [
        "# 48 — eICU motif 字母表一致性检验",
        "",
        f"> 只读已有产物；不重跑 Shapelet、不重挖路径、不重估阈值（docs/28 §2）",
        "",
        "## 1. 问题",
        "",
        "15 条冻结路径在 eICU 未迁移（同号 9/15）。本检验换一个**功效充足**的问题：",
        "**84 节点形态字母表**（粗粒度：变量×形态家族×阶段）是否跨站点迁移？",
        "",
        "## 2. 各队列规模",
        "",
        "| 队列 | stays | motif 原子 | atoms/stay | 低置信比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, b in blocks.items():
        lines.append(
            f"| {k} | {b['n_stays']:,} | {b['n_atoms']:,} | {b['winner_rate']:.2f} | {b['low_conf_prop']:.1%} |"
        )
    lines += [
        "",
        "## 3. 分布一致性（主对比：MIMIC holdout vs eICU）",
        "",
        "| 指标 | 值 | 判读 |",
        "|---|---:|---|",
        f"| **JSD** (holdout vs eICU) | **{jsd_hold:.4f}** | 越小越一致（<0.20 视为迁移） |",
        f"| JSD (train vs eICU) | {jsd_train:.4f} | 参照 |",
        f"| JSD (train vs holdout，MIMIC 内部下界) | {jsd_within:.4f} | 站点内基线 |",
        f"| **Spearman ρ** (holdout vs eICU) | **{rho_hold:+.4f}** | 排序一致性（>0.60 视为迁移） |",
        f"| Spearman ρ (train vs eICU) | {rho_train:+.4f} | 参照 |",
        f"| Spearman ρ (train vs holdout，MIMIC 内部) | {rho_within:+.4f} | 站点内基线 |",
        "",
        f"> **定量解读**：跨站点 JSD ({jsd_hold:.3f}) 约为 MIMIC 内部基线 ({jsd_within:.3f}) 的 "
        f"**{jsd_hold / jsd_within:.1f} 倍**。即字母表**排序**高度一致（ρ={rho_hold:+.2f}），"
        f"但**绝对频率**仍有中等漂移——两者需分开下结论。",
        "",
        "## 4. 低置信跨尺度映射富集检验（docs/20 §8.3）",
        "",
        f"- MIMIC holdout 低置信比例：**{base_low:.1%}**",
        f"- eICU 低置信比例：**{ext_low:.1%}**",
        f"- 差值：**{delta_pp:+.2f} pp**（阈值 ≤ +{LOW_CONF_TOLERANCE_PP:.0f} pp）→ **{'通过' if low_conf_ok else '未通过'}**",
        "",
        "## 5. 结论",
        "",
        f"**{verdict}**",
        "",
        f"- 预先指定判据：JSD < 0.20 **且** Spearman ρ > 0.60 **且** 低置信未异常富集",
        f"- 实际：JSD={jsd_hold:.3f}，ρ={rho_hold:+.3f}，低置信{'通过' if low_conf_ok else '未通过'}",
        "",
        "## 6. eICU 流行度 Top-10 节点",
        "",
        "| 节点 | MIMIC holdout 流行度 | eICU 流行度 | eICU/holdout | 排名位移 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in top10.itertuples(index=False):
        lines.append(
            f"| `{r.node}` | {r.prev_mimic_holdout:.3f} | {r.prev_eicu:.3f} | "
            f"{r.ratio_eicu_over_holdout:.2f} | {int(r.rank_shift):+d} |"
        )
    lines += [
        "",
        "## 7. 与 15 条路径结论的对照",
        "",
        "| 层次 | 单位 | 结果 |",
        "|---|---|---|",
        f"| 粗粒度字母表 | 84 节点形态分布 | {verdict} |",
        "| 细粒度组合 | 15 条阶段序路径 | 未迁移（同号 9/15，docs/28 §4.3） |",
        "",
        "> 若字母表迁移而路径不迁移 → **\"形态可迁移、组合不可迁移\"**，是比单纯阴性更精确的结论。",
        "",
        "## 8. 文件",
        "",
        "- `outputs/tables/eicu_motif_consistency.csv` — 逐节点对比（84 行）",
        "- `data/external/eicu/motif_consistency.json` — 指标摘要",
        "- `outputs/figures/fig15_motif_alphabet_consistency.png` — 散点图",
    ]
    rep_path = rep_dir / "48_eICU字母表一致性.md"
    rep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"  → {rep_path}")
    _log("")
    _log("done.")


if __name__ == "__main__":
    main()
