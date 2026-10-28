# AKI-Trajectory

**Transferability of interpretable temporal concepts after ICU-onset acute kidney injury (MIMIC-III → eICU-CRD, zero-shot external validation)**
**ICU 起病急性肾损伤后可解释时序概念的迁移性（MIMIC-III → eICU-CRD 零样本外部验证）**

Code and aggregate results for an interpretable model of in-hospital death after ICU-onset AKI. The model's explanation units are an 84-state lexicon of named vital-sign / urine-output trend states (7 signals × 3 shape families × 4 post-onset stages) learned with missing-aware Soft-DTW shapelets, and 15 prespecified stage-ordered two-step paths. The units are frozen on MIMIC-III and tested without re-estimation in eICU-CRD. A unit "transfers" if it keeps its frequency, its direction of association with death, and its predictive value.

> **No patient-level data are included.** MIMIC-III and eICU-CRD are credentialed PhysioNet datasets; obtain them yourself under the PhysioNet Credentialed Health Data License. Everything under `results/` is aggregate output (counts, coefficients, intervals, hospital-level summaries).

## 研究设计（锁定）

- 开发库 MIMIC-III v1.4；外部库 eICU-CRD v2.0（N = 21,657，191 家医院），零重估、零重校准
- 人群：成人、每名患者第一次 ICU 起病 AKI；t0 = KDIGO（肌酐 ∪ 尿量）发病；特征窗 [t0, t0+48h)，阶段 B1 0–6h / B2 6–12h / B3 12–24h / B4 24–48h
- landmark 设计：t0+48h 前死亡或存活出院者排除；结局 = t0+48h 之后的住院死亡
- 按 `SUBJECT_ID` 划分 train/val/holdout = 7,610 / 2,537 / 2,537，seed 42；所有阈值、插补、选择只用 train
- 15 条路径冻结，SHA-256 `966ca9be40f9c7abdca6f95416434bcd9abda4d82e50523f4a3e70d5471458b7`；val 与 holdout 各使用一次

## 主要结果

| 单元 | 结果 |
|---|---|
| 84 状态词典 | 分布与排序保持（秩相关 ρ = 0.89）；Bonferroni 显著的 34 个状态中 30 个在 eICU 复现；零样本 AUPRC 由静态协变量的 0.190 升至加入词典后的 0.281 |
| 图模型 | 删除节点降低 val AUPRC，删除学得的先后边无可检出影响 |
| 15 条冻结路径 | 方向保持 11/15（预设阈值 10；双侧符号检验 P = 0.12，无定论）；在 eICU 原生挖掘中 0/15 实例完全重合，10/15 的无序原子对共有 |
| 离散化检验方向 | 不迁移：外部 AUPRC 0.170 < 静态 0.190；同批检验的连续特征（176 个数值摘要）则迁移良好（0.415） |

词典单独使用的外部 AUPRC 低于 APACHE IV（修正后见 `results/eicu/round2_apache/apache_metrics_corrected.json`）。关联为粗死亡率差，不是因果效应；临床医生对状态命名的评估尚未完成。

## 仓库结构

```
scripts/     01–09 数据与基线 · 11–15 Shapelet · 16–19 演化图/GNN · 20 84 节点
             21–34 时序路径与对照实验 · 40–48 eICU 管道 · 47 CBM
             49–58、60 eICU 外部验证各轮分析 · utils/ · tests/
             _explore_*  被结果引用的 post-hoc 探索脚本（节点迁移、CBM 零样本、校准/DCA、稀疏消融）
docs/        协议与实现规范（10、12、13、19、20、24、26–29、31）
results/     聚合结果快照：reports/（实验报告）· tables/ · eicu/（eICU 各轮 JSON/CSV）
```

`results/` 是快照；脚本运行时把产物写入 `data/` 与 `outputs/`（已被 `.gitignore` 忽略）。

## 运行

```bash
pip install -r requirements.txt            # torch 按 CUDA 版本另装
python scripts/run_pipeline.py             # MIMIC 01–08
python scripts/run_eicu_external.py        # eICU 40–46
python -m pytest scripts/tests -q
```

数据路径见 `scripts/utils/constants.py`（环境变量 `EICU_DIR` 等）。**不得重跑** `24`（val）与 `26`（holdout），也不得依据 eICU 结果修改冻结资产。

## 已知问题（保留原始产物，不删除）

1. `46_eicu_reproduce_paths.py` 的 `same_sign` 字段按 eICU 内部方向判定，而协议要求与 MIMIC-train 差值同号，导致早期输出为 9/15。按协议重算为 11/15，见 `scripts/50_path_sign_erratum.py` 与 `results/reports/50_同号判据勘误.md`。新分析请用 50，不要复用 46 的该字段。
2. `49_eicu_apache_baselines.py` 把 eICU 中表示"未计算"的 −1 当作真实预测。已由 `49b_apache_missing_code.py` 修正（见 `apache_metrics_corrected.json`，其中同时保留原值作核对）。旧报告 49 与旧 `apache_metrics.json` 未随仓库发布。
3. `60_interpretation_transfer.py`（跨院解释一致性）与全部 `_explore_*` 脚本是 post-hoc 探索，结论为探索性；`50` 与 `_explore_PRED_cbm_zeroshot.py` 的注释提到的 `_tmp_urine_threshold.py` 是临时脚本，未发布。
4. GNN 训练需要 PyTorch（建议 GPU）；仓库只含脚本与结果快照，不含权重和中间数据。
5. 论文图的生成脚本尚未定稿，未发布。

## 数据使用

MIMIC-III 与 eICU-CRD 受 PhysioNet Credentialed Health Data License 约束。本仓库不含任何患者级数据，也不含由其导出的患者级表（逐 stay 预测、原子表、特征表）。
