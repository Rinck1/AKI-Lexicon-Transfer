# 53 — XGBoost 零样本外部基线（docs/31 A4 / docs/29 E4）

> round-2 预指定分析。脚本 `scripts/53_xgb_zero_shot.py`；数字出处 `data/external/eicu/round2_xgb/metrics_summary.json`。
> **公平性声明**：特征 = 11 静态 + 11 检验变量 × 4 阶段 × {last, min, max, delta}（176+11 维），与主模型可用信息同源，不含生命体征形态、不含 84 概念；超参只在 MIMIC train 内 SUBJECT 分组 5 折 CV 选择；MIMIC val 仅在模型定死后报告一次（描述性）；holdout 未触碰；eICU 零 refit。

## 1. 调参（MIMIC train 分组 CV）

最优：`max_depth=3, learning_rate=0.05, n_estimators=147`（折内 AUPRC 早停）。

## 2. eICU 零样本结果

| 模型 | MIMIC val(desc) AUPRC | eICU AUPRC | eICU AUROC | eICU Brier |
|---|---:|---:|---:|---:|
| 静态 logistic | 0.222 | 0.190 | 0.572 | 0.130 |
| +84 概念 a（本文解释臂） | ~0.33 | 0.281 | 0.692 | 0.122 |
| 线性 logistic + 数值176 | 0.535 | 0.415 | 0.753 | 0.112 |
| **XGBoost + 数值176** | **0.571** | **0.441** | **0.784** | **0.107** |
| XGBoost + 数值176 + 测量模式132（附加臂） | 0.573 | 0.439 | 0.783 | 0.107 |

- 线性复现臂 0.415 与 round-1 数值臂完全一致（口径校验通过）。
- XGBoost − 线性（同特征）：AUPRC **+0.026，bootstrap 95% CI [0.015, 0.036]**；AUROC +0.030 [0.023, 0.037]；Brier −0.005。
- 加 132 维测量模式特征几乎不涨（0.441→0.439）——测量模式在线性臂的 0.257 增量在树模型下已被数值特征吸收（缺失模式即分裂路径）。

## 3. 校准（eICU，logistic 重校准）

| 模型 | slope | intercept |
|---|---:|---:|
| XGBoost | 0.907 | 0.102 |
| 线性+数值176 | 0.707 | −0.120 |
| 静态 logistic | 0.496 | −0.706 |

XGBoost 跨站点校准接近理想（slope≈0.9），静态臂的 slope 0.50 再次确认其外部失准是区分度问题而非可校准问题。

## 4. 特征重要性（gain）

- 按统计量：last 31.4%、min 27.4%、max 24.2%、delta 12.2%、静态 4.8%——**数值水平本身 > 变化量**。
- 按变量 top：gcs_total 19.6%、bun 12.3%、platelet 8.6%、lactate 8.6%、bicarbonate 7.5%。GCS 主导与报告 52 的测量密度归因呼应：eICU 上最值钱的单变量也是测量最稀疏的变量之一。

## 5. 解读与论文写法

1. **外部预测上限刷新为 0.441**（XGBoost），非线性增益 +0.026（CI 不含 0 但幅度有限）。论文对照表以 XGBoost 为"同信息黑箱上限"行，概念臂 0.281 与之的差距即解释的标价。
2. XGBoost 校准 slope 0.91 说明"静态臂外部差"不是域不可移，而是特征与模型容量问题——支持"分层迁移"叙事的 L4 层。
3. 不得声称 XGBoost 是本文方法；它回答的是审稿人"简单模型够不够"——答案：够了，且解释层定位不与它竞争（docs/29 §7 口径）。
