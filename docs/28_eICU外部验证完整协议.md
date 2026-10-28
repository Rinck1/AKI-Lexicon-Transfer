# 28 — eICU-CRD 外部验证完整协议（含跨站点路径稳健性表）

> **状态**: v2.0（2026-08-27，在 docs/24 v1.0 基础上补全）
> **效力**: eICU 只用于 **零训练复现** MIMIC-III 冻结的 15 条路径与节点词典。禁止在 eICU 上重估阈值、重挖路径、重训 Shapelet / GNN / CBM。
> **一句话目标**: 证明"可解释性本身跨站点迁移"，而不只是 AUC 迁移。

---

## 0. 与 docs/24 的关系

docs/24 定义了 40→46 的工程管道与预指定尿量策略，本文件在其上补三件审稿人必问的事：

1. **跨站点路径稳健性表**（第 4 节）——15 条冻结路径在 MIMIC 与 eICU 的逐条支持度 / 方向 / 效应对比。这是本文区别于"普通预测模型"的杀手锏。
2. **基线对照矩阵与公平性控制**（第 5 节，详见 docs/29）——补齐临床评分与 ML 基线的同口径对比。
3. **校准 + 决策曲线 + 净获益**（第 6 节）——把"预测准"翻译成"临床有用"。

---

## 1. 数据与对齐（继承 docs/24，一字不改）

- 来源：eICU-CRD v2.0（`data/eicu-crd/`，`patient.csv.gz` = 200,859 stay）。
- 原始表定位：`scripts/utils/eicu_io.py::resolve_eicu_dir`；产出 `data/external/eicu/`。
- 队列规则、KDIGO onset、48h landmark、住院死亡结局、84 节点字母表、`lab_thresholds.json`、`path_freeze_manifest.json` 全部沿用 MIMIC-III 定义。

| 项 | 规则 |
|----|------|
| 年龄 / ICU 停留 | ≥18 岁，ICU LOS ≥ 24h |
| ESRD | apacheApsVar.dialysis 或既往 HD/PD/移植，排除 |
| Stay | 每 hospital encounter 第一次 ICU（`unitvisitnumber==1`），再取每 subject 第一次 HA |
| t0 | KDIGO 肌酐∪尿量 |
| 窗 / 结局 | onset 后 48h；landmark 后住院死亡（`hospitaldischargestatus=Expired`） |
| 划分 | eICU 全量即"外部测试集"，**无 eICU train/val 之分**，所有系数与阈值来自 MIMIC train |

---

## 2. 冻结清单（本文件不得改）

| 项 | 值 |
|----|-----|
| 路径冻结哈希 | `966ca9be40f9c7abdca6f95416434bcd9abda4d82e50523f4a3e70d5471458b7` |
| 路径条数 | 15（`path_freeze_manifest.json::selected`） |
| 离散阈值 | `lab_thresholds.json`（train-only 中位数） |
| Shapelet 模型 | `final_7var_L4_6_12_K3/ckpt_best.pt` + `shapelet_frozen_qp50.pt` |
| 节点词典 | `graph_final_7var_motif84`（84 节点） |
| 尿量完整子集阈值 | `n_urine_hours_measured ≥ 12` |
| 静态标准化 | MIMIC train 的 μ/σ |
| 路径 logistic 系数 | 只在 MIMIC train 拟合（`46` 内重新 `LogisticRegression`） |

---

## 3. 三份预先指定的评估

| 块 | 人群 | 用途 |
|----|------|------|
| **full** | 全部 landmark-eligible HA-AKI stay | 主分析，与 MIMIC holdout 同口径 |
| **urine_complete** | full 中 `n_urine_hours_measured ≥ 12` | 尿量测量充分子集，排除"尿量缺失导致 onset 判定噪声"的疑虑 |
| **creatinine_only**（可选） | 肌酐-only KDIGO 队列 | 若做，`--creatinine-only` 另跑 `42`，**不覆盖主队列文件** |

**产出**（`data/external/eicu/`）：`path_external_reproduce.json` + `eicu_path_reproduce.csv`。

---

## 4. 跨站点路径稳健性表（本协议核心，投稿主表之一）

### 4.1 设计动机

外部验证常见错误：只报一个"外部 AUROC/AUPRC"。审稿人会问：**这条路径在 eICU 里到底还成不成立？** 本节把 15 条路径的迁移性拆成三个可复核的维度。

### 4.2 逐条路径对比表（`eicu_path_reproduce.csv` 已含大部分列）

对每条冻结路径 `p_k`，在 MIMIC holdout 与 eICU full 上分别报告：

| 列 | 定义 | 通过标准 |
|----|------|---------|
| `support` | 完整走出该序列的不同 stay 数 | ≥ 20（沿用 docs/20 §5 `VAL_MIN_SUPPORT`） |
| `support_ok` | support ≥ 20 | bool |
| `death_rate_path` / `death_rate_other` | 路径内 / 非路径死亡占比 | — |
| `death_rate_diff` | 死亡占比差（路径 − 非路径） | — |
| `odds_ratio` | 粗优势比 | — |
| `same_sign` | eICU 与 MIMIC train 的死亡占比差同号 | 主标准：同号 |
| `same_sign_holdout` | 与 MIMIC holdout 同号 | 次标准 |
| `logit_coef_train` | MIMIC train 上的路径 logistic 系数（冻结） | — |

**表 2（正文）示意**：`eICU 15 条路径逐条迁移`

| # | 路径 | MIMIC holdout n | eICU n | eICU 差 [95% CI] | 与 train 同号 | 与 holdout 同号 | eICU 支持≥20 |
|---|------|:---:|:---:|---:|:---:|:---:|:---:|
| … | `lactate|B1|升高 → urine_output_ml|低位缓升型|B3` | 125 | … | … | … | … | … |

### 4.3 集合级迁移指标（`46` 已算）

| 指标 | 定义 | 含义 |
|------|------|------|
| `auprc_static` | 仅静态 logistic 的 eICU AUPRC | 基线 |
| `auprc_static_paths` | 静态 + 15 路径指示 | 主结果 |
| `auroc_static` / `auroc_static_paths` | 同上的 AUROC | 辅 |
| `n_same_sign` | 15 条中与 MIMIC train 同号的条数 | 迁移性 |
| `n_support_ge_20` | 15 条中 eICU 支持 ≥ 20 的条数 | 迁移性 |

**预先指定的成功判据（写死，失败记失败，不回改 15 条）**：

1. `n_same_sign ≥ 10/15`（即 ≥ 2/3 同号）；
2. `n_support_ge_20 ≥ 10/15`；
3. `auprc_static_paths ≥ auprc_static`（点估计同向即可，CI 跨 0 则降为描述性）；
4. 若 3 不成立，只报描述性，不得称"路径在 eICU 有效"。

### 4.4 motif 家族分布一致性（新增，区别于 46 现有产出）

仅比较 AUC 不足以证明"形态字母表迁移了"。补一个**零假设对照**：

- 计算 eICU 上 84 个 motif 家族节点（变量×家族×阶段）的出现频率分布；
- 与 MIMIC holdout 上的分布做对比（可用 Jensen–Shannon 距离或逐节点支持度秩相关）；
- 报告：eICU 上 winner 事件率、低置信跨尺度映射占比（应对照 docs/20 §8.3，不得异常富集）。

**产出脚本**：新增 `47_eicu_motif_consistency.py`（或并入 46 报告）。此步**只读** `motif_atoms.parquet`，不重跑 Shapelet。

---

## 5. 基线对照矩阵与公平性控制

见 **docs/29**。要点：eICU 侧补 SAPS II / APACHE IV（`apachePatientResult.csv.gz`）与静态 logistic 的同口径对比；所有基线与主模型共用同一 eICU 队列、同一结局、同一 AUPRC 口径，避免"挑更好的队列"。

---

## 6. 校准、决策曲线与净获益（一区必问）

1. **校准曲线 + Brier**：eICU 上报告静态 vs 静态+路径的 Brier 与校准曲线（分 10 箱）。
2. **DCA**：阈值 0.05–0.40 上，静态+路径相对"全当高危 / 全不当高危 / 仅静态"的净获益。
3. **NRI/IDI**：报告连续 NRI 与 IDI，且**必须拆分事件侧 / 非事件侧**（MIMIC 侧已知事件侧 NRI 为负，eICU 上同样要如实报）。

---

## 7. 运行

```bash
python -u scripts/run_eicu_external.py                    # 40→46 全跑
python -u scripts/run_eicu_external.py --from-stage 41 --to-stage 46  # 40 已完成时
```

阶段：40 底盘 → 41 长表 → 42 KDIGO → 43 网格 → 44 静态 → 45 冻结原子 → 46 15 条复现。

---

## 8. 失败处理（预先指定）

- 若 eICU 上 `n_same_sign < 10/15`：记**外部阴性**，正文如实写"路径未跨站点迁移"，不得改阈值/支持/名单后重跑。
- 若 eICU 队列死亡患病率与 MIMIC（0.145）差异 > 5 个百分点：在结果中显式标注患病率差异对 AUPRC 的影响（AUPRC 对患病率敏感）。
- 若 motif 家族分布 JSD 显著：说明字母表本身随站点漂移，此时路径迁移的解读需降级为"在 eICU 可测子集上"。

---

## 9. 成功标准汇总

与 docs/20 §5 同口径：逐条同号、支持 ≥ 20、静态+路径 AUPRC 相对静态不降、motif 家族分布不漂移、低置信不富集。四者同时满足，才允许在正文写"可解释性跨站点迁移"。
