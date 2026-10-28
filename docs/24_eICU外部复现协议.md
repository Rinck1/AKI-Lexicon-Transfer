# 24 — eICU-CRD 外部复现协议

> **状态**: v1.0（2026-08-25）  
> **效力**: eICU 只用于冻结 MIMIC-III 路径的零样本复现。禁止在 eICU 上重估阈值、重挖路径、重训 Shapelet / GNN。

## 数据

- 来源：eICU-CRD v2.0（已下载，`patient.csv.gz` = 200,859 stay）
- 原始表定位：`scripts/utils/eicu_io.py::resolve_eicu_dir`
- 产出：`data/external/eicu/`（**不**写入 `data/processed/` 的 MIMIC 表）

## 与 MIMIC-III 对齐的定义（一字不改）

| 项 | 规则 |
|----|------|
| 年龄 / ICU 停留 | ≥18 岁，ICU LOS ≥ 24h |
| ESRD | apacheApsVar.dialysis 或既往 HD/PD/移植 |
| Stay | 每个 hospital encounter 第一次 ICU（`unitvisitnumber==1`），再在 HA-AKI 上取每个 subject 第一次 HA |
| t0 | KDIGO 肌酐∪尿量 |
| 窗 / 结局 | onset 后 48h；landmark 后住院死亡（`hospitaldischargestatus=Expired`） |
| 字母表 | 冻结家族图 + `lab_thresholds.json` + freeze `path_freeze_manifest.json` |

## 预指定尿量策略（先写后跑）

1. **主分析**：KDIGO = 肌酐∪尿量，与 III 相同。  
2. **尿量完整子集**：上述队列中 `n_urine_hours_measured ≥ 12`（`URINE_COMPLETE_MIN_HOURS`，不对 eICU 结局搜索）。  
3. 肌酐-only 全队列 KDIGO 若要做，另开 `--creatinine-only` 重跑 42，不得覆盖主队列文件。

尿量 item：`celllabel==Urine` 或导尿管/肾造瘘等白名单；胸管 / EVD 排除。缺失仍不当 0。

## 运行

```bash
python -u scripts/run_eicu_external.py
python -u scripts/run_eicu_external.py --from-stage 40 --to-stage 42
```

阶段：40 底盘 → 41 长表 → 42 KDIGO → 43 网格 → 44 静态 → 45 冻结原子 → 46 15 条复现。

Shapelet 的 μ/σ 与静态标准化来自 **MIMIC train**；路径 logistic 系数同样只在 MIMIC train 拟合。

## 成功标准

与 `docs/20` §5 相同口径：逐条同号、支持 ≥ 20、静态+路径 AUPRC 相对静态。失败记失败，不回改 15 条。
