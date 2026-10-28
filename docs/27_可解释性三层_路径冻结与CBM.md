# 27 — 可解释性三层：15 条路径冻结 + 增强验证 + CBM

> **状态**: 已落盘结论（2026-08-27）  
> **效力**: 15 条路径 **一个字不改**。可解释性相对 Word 终稿的增强不在换路径，而在证据厚度和概念干预维。  
> **冻结**: `path_freeze_manifest.json` → `freeze_sha256 = 966ca9be40f9c7abdca6f95416434bcd9abda4d82e50523f4a3e70d5471458b7`  
> **相关**: `docs/20` 路径协议，`docs/21` 索引，`docs/22` 总述，`docs/23` 初稿，`docs/26` CBM 协议。

## 结论先行

路径没改、不必改。改的是证据链厚度，以及多出来的概念干预维。CBM 不替代 15 条时序路径。

三层分工：

| 层 | 回答什么 | 模型 | 口径 |
|----|----------|------|------|
| 轨迹 | 先发生什么、后发生什么 | 冻结 15 条 2-step + 静态 logistic | val 0.222→0.349；holdout 0.244→0.309；train 五折 0.249→0.386 |
| 证据 | 这 15 条是不是运气 / 泄漏 | 消融、LOO、随机、时间打乱、持续 AKI 对照、NRI/IDI | 全部 train 或预先指定；不重选路径 |
| 概念 | 单个可读概念置零后 \(\Delta p\) | CBM：静态 + 84 个形态 \(a\) + 87 个检验方向 | **仅 train 五折** 0.439；未上 val/holdout |

预测参照仍是表格臂 B（val AUPRC 0.582）。路径和 CBM 都不与 B 争预警。

---

## 1. 15 条路径：冻结，未改

`path_set_selected.json`（2026-08-17）与 `path_freeze_manifest.json` 的 `selected` **逐字一致**。

冻结清单字段：

- `stage`: `path_set_frozen_for_holdout`
- `val_used`: `"once"`
- `holdout_used`: `false`（**冻结当时**；复现结果见 `path_holdout_reproduce.json` 的 `holdout_used: "once"`，未重选）
- `freeze_sha256`: `966ca9be40f9c7abdca6f95416434bcd9abda4d82e50523f4a3e70d5471458b7`

消融 / 效用 / CBM 分别声明 `test_used_to_select: false`、`reselected: false`、`val_read/holdout_read: false`。不是每个 json 都同时带这两个键。

| # | 路径 |
|---|------|
| 1 | `lactate\|B1\|升高 → urine_output_ml\|低位缓升型\|B3` |
| 2 | `mbp\|缓降持平型\|B3 → lactate\|B4\|升高` |
| 3 | `wbc\|B1\|升高 → gcs_total\|B4\|下降` |
| 4 | `lactate\|B3\|升高 → sbp\|低位下滑型\|B4` |
| 5 | `gcs_total\|B3\|下降 → creatinine\|B4\|升高` |
| 6 | `creatinine\|B3\|升高 → ph\|B4\|下降` |
| 7 | `dbp\|低位回升型\|B1 → lactate\|B4\|升高` |
| 8 | `urine_output_ml\|低位波动型\|B1 → creatinine\|B4\|升高` |
| 9 | `potassium\|B3\|升高 → lactate\|B4\|下降` |
| 10 | `respiratory_rate\|末期骤降型\|B1 → urine_output_ml\|低位波动型\|B4` |
| 11 | `urine_output_ml\|低位波动型\|B1 → urine_output_ml\|进行性下降型\|B2` |
| 12 | `dbp\|持续下降型\|B1 → gcs_total\|B4\|下降` |
| 13 | `hemoglobin\|B3\|升高 → creatinine\|B4\|升高` |
| 14 | `ph\|B1\|升高 → gcs_total\|B2\|升高` |
| 15 | `ph\|B3\|升高 → creatinine\|B4\|升高` |

禁止：用 CBM 的 \(\Delta p\) 排序重拼 2-step；用 val/holdout/CBM 五折回头换这 15 条。

---

## 2. 增强验证：给路径上保险（不改名单）

产物：`path_ablation.json`、`path_uncertainty_utility.json`、`clinical_contrast_persistent.json`、时间打乱报告 `outputs/reports/34_时间打乱负对照.md`。

### 2.1 消融（train 五折）

`path_ablation.json`：P0–P9 共 10 臂 + 15 个 leave-one-out（不是「10 个对照」；P2 是主臂）。

| 臂 | 含义 | CV AUPRC |
|----|------|--------:|
| P0_static | 纯静态 | 0.249 |
| P1_paths_only | 仅 15 条 | 0.282 |
| **P2_static_plus_15** | **正文主臂** | **0.386** |
| 随机 15 条可读 2-step（10 组） | 均值 / p95 | 0.311 / 0.329 |
| LOO_00 … LOO_14 | 每次去掉一条 | 0.373–0.385 |

- 主臂 **明显高于** 随机（0.386 vs 0.311 / p95 0.329）。文中写「明显高于」，未做置换检验 p 值则不要写「显著」。
- LOO 全部低于 P2，没有「删了反而涨」的独苗。掉得最多的是去掉路径 1（乳酸 B1↑ → 尿量低位缓升 B3）：0.386→0.373。

### 2.2 时间打乱（顺序 vs 共现）

同一 15 对、train 五折：

| 模型 | AUPRC |
|------|------:|
| 静态 + 冻结 15 条（原阶段序） | 0.386 |
| 静态 + 同一 15 对无序共现 | 0.370 |
| 静态 + 冻结 15 条（打乱阶段，5 种子均值） | 0.305 ± 0.008 |

有序相对打乱 \(\Delta=-0.081\)。顺序不只是共现；15 条身份仍是时序路径。

### 2.3 NRI / IDI（1000 bootstrap）

`path_uncertainty_utility.json`。连续 NRI 为正，**事件侧为负**；正贡献几乎全来自下调非事件。

| 指标 | train（样本内） | holdout |
|------|----------------:|--------:|
| NRI_continuous | 0.512 | 0.343 |
| NRI_event | −0.287 | −0.458 |
| NRI_nonevent | 0.799 | 0.801 |
| IDI | 0.083 | 0.051 |
| frac_nonevent_down | 0.899 | 0.900 |

路径相对静态主要是识别较低风险轨迹，不是系统抬高死亡者。

### 2.4 临床对照

持续 AKI / 窗末少尿见 `clinical_contrast_persistent.json` 与 `outputs/reports/33_持续AKI与路径临床对照.md`。粗「持续 AKI」几乎无死亡差；窗末少尿和休克–少尿路径并集仍有独立关联。不据此改 15 条。

---

## 3. 新增解释臂：CBM Phase A

协议 `docs/26`。实现 `scripts/47_concept_bottleneck.py`。  
结果 `data/processed/cbm/cbm_phase_a/cbm_train_cv.json`（2026-08-27）。

不挖 2-step。线性瓶颈：

```text
logit = static + 84 motif 激活 a + 87 lab 方向
```

### 3.1 train 五折（SUBJECT 分层，seed 42）

| 模型 | 概念数 | AUPRC | AUROC |
|------|------:|------:|------:|
| 纯静态 | 0 | 0.249 | 0.662 |
| +84 motif 0/1 | 84 | 0.351 | 0.759 |
| +84 motif 连续 a | 84 | 0.360 | 0.765 |
| **+84a + 87 检验方向** | **171** | **0.439** | **0.811** |

- 静态 0.249 与路径消融 P0 **同一数字**，口径对齐。
- 连续 a 相对 0/1 仅 +0.009：解释单位在很大程度上仍是「出现与否」。
- 0.439 对路径臂 0.386：**171 个单概念 vs 15 条 2-step**，不作「更好的路径」。增益主要来自检验升降进线性头。
- **未读 val / holdout**。0.439 不可与路径 val 0.349 或 B 的 val 0.582 直接比。

### 3.2 置零干预（阳性子集平均 \(\Delta p\)）

操作：把该概念在 **已经为阳性的患者** 上置 0，重算 \(p\)。不是把方向「翻转」成反概念（GCS↓ 不会变成 GCS↑）。不是全队列平均，不是因果。

高危（置零后平均 \(p\) 下降，表中记为风险贡献 \(\Delta p>0\)）：

| 概念 | 类型 | 覆盖 | n | 阳性子集平均 Δp |
|------|------|-----:|--:|----------------:|
| `lactate\|B4\|升高` | lab | 2.5% | 187 | +0.133 |
| `gcs_total\|B1\|下降` | lab | 39.1% | 2979 | +0.120 |
| `lactate\|B3\|升高` | lab | 1.6% | 124 | +0.083 |
| `gcs_total\|B4\|下降` | lab | 6.7% | 511 | +0.079 |
| `urine_output_ml\|低位缓升型\|B4` | motif | 44.9% | 3414 | +0.074 |

保护最强：`gcs_total\|B2\|升高`（−0.110）、B3（−0.092）、B4（−0.076）。

与冻结路径端点重合：乳酸 B3/B4↑、GCS↓、尿量低位、路径 14 的 GCS↑。这是 **端点确认**，不是重选路径。

CBM 与路径可以打架，这正好说明路径不该拆成单概念：`urine_output_ml\|低位波动型\|B1` 在 CBM 里偏保护（Δp −0.050），却是路径 8、11 的起点（随后肌酐↑ / 尿量进行性下降）。单独看起点像还没恶化；2-step 才是接着变差。

---

## 4. 对「可解释性有没有帮助」的回答

有，而且是三层，互不覆盖：

1. **15 条保持纯时序路径身份** —— 回答演化序列。CBM 没有箭头，不替代。
2. **增强验证** —— 消融 / LOO / 随机 / 打乱 / NRI，把名单从「挖出来的」做成「过完阴性对照的」。审稿关切的泄漏与运气在这里回答。
3. **CBM 补概念干预** —— 路径给「尿量→肌酐」这类链；CBM 给阳性患者上置零后的 Δp。新维度是净增益；AUPRC 0.439 不作路径升级解读。

Word 终稿已有路径 val/holdout。CBM 目前只证明方法在 train 五折可行。进正文前须按 `docs/26` 申请 val/holdout 各一次；否则 CBM 放附录 / 方法学段落。

---

## 5. 论文落点（禁止写成的句子）

**可以写**

- 路径集合冻结，SHA `966ca9be…`；holdout 只复现一次。
- 静态+15 条 train 五折明显高于随机 15 条；LOO 无反转独苗。
- 连续 NRI 为正，事件侧为负，主要下调非事件。
- CBM 是同一套字母表上的线性概念瓶颈，支持置零干预；高危概念与路径端点一致。

**不可以写**

- CBM 性能更好，所以 15 条过时 / 应按 Δp 重挖。
- lactate↑B4 使「平均风险 +0.133」（漏写阳性子集）。
- 干预 = 翻转概念 / 因果 / 反事实。
- CBM 已在 val 或 holdout 上确认。
- 预测与解释已经合成一个模型且该模型就是 CBM（预警参照仍是 B）。

落盘：

- 路径：`data/processed/paths/path_final_7var_motif84_labs11/`
- CBM：`data/processed/cbm/cbm_phase_a/`
- 脚本：`29` 消融，`31` NRI，`33` 临床对照，`34` 打乱，`47` CBM
