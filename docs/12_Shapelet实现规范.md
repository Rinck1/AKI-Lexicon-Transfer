# 12 — 时间感知 Shapelet 实现规范（主分析）

> **状态**: v1.0（2026-03-26）  
> **效力**: 实现 `scripts/11_shapelet_learn.py` 与 `scripts/utils/shapelet_*.py` 时以本文为准。  
> **上位文档**: `docs/11_TASER-AKI_...技术方案.md` §3.3–§5；与本文冲突时以**更严、可实现**的条款为准。  
> **范围**: 仅 **7 个密集变量** 的 Shapelet 学习与激活导出；演化图 / GNN / RL 不在本阶段实现。  
> **公式写法**: 标准 LaTeX 数学——行内 `$...$`、独立成行 `$$...$$`；仅使用 GitHub / Typora / Obsidian / VS Code 等主流渲染器**均支持**的命令（无自定义宏、无 math 代码围栏、无 `aligned` 等环境）。

---

## 1. 目标与非目标

### 1.1 目标

1. 在 HA-AKI 主分析队列上，对 7 个连续变量学习全局 Shapelet 模板 $S$；  
2. 为每个 train/val/test 患者导出阶段激活 $a$、时间 $t^{*}$；  
3. 可选：仅用 Shapelet 激活 + 静态协变量做死亡预测，与 08/09 基线对照；  
4. 产物可复现、可进后续患者级演化图。

### 1.2 非目标（本阶段不做）

- 不对肌酐/乳酸/GCS 等做小时 Shapelet（第二层阶段摘要，另模块）；  
- 不建图、不训 GNN、不跑 RL；  
- 不以抬高 AUROC 为唯一成功标准。

---

## 2. 数据与划分

| 项目 | 规定 |
|------|------|
| 队列 | `data/processed/aki_primary_ha48.parquet` |
| 小时网格 | `data/processed/hourly_grid_48h.parquet` |
| 划分 | `stay_splits` / `stay_features` 中的 `split`（SUBJECT_ID 级，已冻结） |
| 时间 | onset 后 $\text{HOUR} \in \{0,\ldots,47\}$，$T=48$ |
| 标签 | `HOSPITAL_MORTALITY`（landmark 后住院死亡） |

**铁律**: 标准化、$\delta$ 截断统计、模板学习、阈值 $q$ **仅 fit 于 train**；val 选型；test 只评估一次。

---

## 3. 变量范围（三层中的第一层）

### 3.1 Shapelet 变量（仅此 7 个）

```text
SHAPELET_VARS = [
  heart_rate, sbp, dbp, mbp,
  respiratory_rate, spo2, urine_output_ml
]
```

网格中列名：`{var}` 与 `{var}_is_measured`。

### 3.2 静态协变量（可选进风险头，不进 Soft-DTW）

```text
AGE, GENDER_M, onset_stage, onset_hour_from_icu,
CHF, Diabetes, HTN, CKD_FLAG, COPD, LiverDisease, Arrhythmia
```

来源：`stay_features_48h` 或 `aki_primary_ha48`（性别需 0/1）。

---

## 4. 超参数（默认与搜索）

| 符号 | 含义 | 默认 | 候选（仅 val） |
|------|------|------|----------------|
| $L$ | 模板长度（小时） | 多尺度并行 | $\{4,6,12\}$ |
| $K$ | 每变量、每长度下的模板数 | **3** | $\{2,3,4\}$ |
| $\gamma$ | Soft-DTW 温度 | 0.1 | $\{0.05,0.1,0.2\}$ |
| $\tau_s$ | 激活温度 | 0.5 | $\{0.3,0.5,1.0\}$ |
| $q$ | 激活阈值（方案 B） | val 网格 | 见 §7 |
| $\lambda_m$ | 未测基础惩罚 | 0.5 | $\{0.1,0.5,1.0\}$ |
| $\lambda_\delta$ | $\delta$ 惩罚 | 0.1 | $\{0.0,0.1,0.3\}$ |
| $\delta_{\max}$ | $\delta$ 截断 | 48 | 固定 |
| 阶段 | $B_1$–$B_4$ | 固定 | 见 §6 |
| 优化 | AdamW | lr=1e-2 | — |
| 早停 | val loss 或 val AUPRC | patience=15 | — |

**主分析默认 $K=3$**；敏感性报告 $K=2,4$。

节点字典规模（后续建图）：

$$|V_G| = 7 \times K \times n_{\text{lengths}} \times 4$$

若每变量共用多尺度：实现上按 $(v, L, k, b)$ 或展平为 $(v, k_{\text{global}}, b)$；本阶段导出激活键为 `(var, length, k, stage)`。

---

## 5. 输入通道与 $\delta$

对变量 $v$、小时 $t$：

$$\tilde{x}_t = \text{z-score}(\text{value}), \qquad m_t = \text{is\_measured} \in \{0,1\}$$

（z-score 的 $\mu,\sigma$ 仅由 train 拟合；未测保持 NaN）

**$\delta$ 定义（强制）:**

| 情况 | $\delta_t$ |
|------|----------------|
| $m_t=1$ | $0$ |
| $m_t=0$ 且 $t$ 前存在测量 | $t - t_{\text{last}}$ |
| **从未测过** | $t+1$，并 `never_measured_yet=1` |
| 截断 | $\delta_t \leftarrow \min(\delta_t, 48)$ |

---

## 6. 窗口、阶段、合法起点

### 6.1 合法起点

$$0 \leq t \leq T - L = 48 - L$$

### 6.2 阶段按**窗口起点**划分

$$B_1: t \in [0, 6), \quad B_2: t \in [6, 12), \quad B_3: t \in [12, 24), \quad B_4: t \in [24, 48) \cap \{t \leq 48-L\}$$

例：$L=12,\ t=5$ 覆盖 5–16h → **仍属 $B_1$**。

阶段 $b$ 的起点集合 $T_b = \{t : \text{stage}(t)=b,\ 0 \leq t \leq 48-L\}$。  
若 $T_b$ 为空，则该阶段激活 $a=0$。

---

## 7. 局部代价与 Soft-DTW（实现铁律）

### 7.1 NaN 安全

$$x^{\text{safe}}_t = \tilde{x}_t \ \text{if}\ m_t = 1, \quad \text{else}\ 0.0$$

禁止 $m \cdot (\text{NaN} - S)^2$。

### 7.2 局部代价

$$c_{rj} = m_r \cdot \alpha_j \cdot (x^{\text{safe}}_r - S_j)^2 + (1 - m_r) \cdot \left[ \lambda_m + \lambda_\delta \log(1 + \delta_r) \right]$$

- $\alpha_j = \text{softmax}(\text{alpha\_logits})_j$，`alpha_logits` 可学习。  
- $S_j$ 可学习。

### 7.3 Soft-DTW

在代价矩阵 $C \in \mathbb{R}^{L \times L}$（窗口长 $L$ 对模板长 $L$）上计算 Soft-DTW 值 $D(t)$（PyTorch 可微实现）。

**主分析激活默认方案 B**（因缺失代价非标准，普通 Soft-DTW 可负）：

$$a = \sigma\left( \frac{q - d}{\tau_s} \right)$$

备选方案 A：Soft-DTW divergence（非负）后再 $\exp(-d/\tau)$（敏感性）。

$q$：对 train 上 $d$ 的分位数初始化，或在 val 上网格选后冻结。

### 7.4 阶段归一化 softmin

$$d_b = -\tau \log\left( \frac{1}{|T_b|} \sum_{t \in T_b} \exp(-D(t)/\tau) \right)$$

$$t^{*}_b = \text{soft-argmin}_{t \in T_b} D(t)$$

$$a_b = \sigma\left( \frac{q - d_b}{\tau_s} \right)$$

---

## 8. 风险头与损失

### 8.1 预测（Shapelet-only 头）


$$\hat{p}_i = \sigma\left( b_0 + \sum_{v,k,L,b} \beta_{v,k,L,b} \cdot a_{i,v,k,L,b} + \theta^\top C_i \right)$$

$C_i$：§3.2 静态协变量（train 标准化）。

### 8.2 损失

$$\mathcal{L} = \text{BCE}(\hat{p}, y) + \lambda_{\text{div}} \mathcal{L}_{\text{div}} + \lambda_{\text{time}} \sum_{b>1} (\beta_{\cdot,b} - \beta_{\cdot,b-1})^2 + \lambda_{\text{sp}} \|\beta\|_1$$

- $\mathcal{L}_{\text{div}}$：同变量同长度下模板两两余弦相似度惩罚（鼓励多样性）。  
- 默认 $\lambda_{\text{div}}=0.1, \lambda_{\text{time}}=0.01, \lambda_{\text{sp}}=10^{-3}$（val 可调）。

`pos_weight` 可用 train 上 $n_{\text{neg}} / n_{\text{pos}}$。

---

## 9. 训练协议（概要）

1. 读 primary 队列 + hourly_grid + split
2. 仅 train 拟合每变量 $\mu,\sigma$；transform 全员数值通道
3. 构建 Dataset：每人 7 变量 × $(T,\ \text{value},\ \text{mask},\ \text{delta})$
4. 初始化 $S,\ \alpha_{\text{raw}},\ \beta,\ \theta$
5. 每个 batch：对每变量每 $(L,k)$ 算所有合法 $t$ 的 $D(t)$ → 四阶段 $a,\ t^{*}$；拼特征 → BCE + 正则
6. 每个 epoch 在 val 上算 AUPRC/AUROC；早停
7. 冻结 $S,\alpha$；导出全员激活表；test 评估一次

**设备**: CUDA 优先；可对变量循环以控显存。

**复杂度注意**:  
对每个（人, 变量, $L$, $k$, 起点 $t$）做 $L \times L$ Soft-DTW，成本高。实现策略：

- 按变量顺序训练或共享 batch；  
- 可选 `max_starts_per_stage` 子采样（仅 train 加速，主分析最终应用全起点）；  
- 默认先 **单变量 debug**，再 7 变量。

---

## 9A. 超参数详解 + 训练全过程（面向实现与答辩）

本节用**人话**说明：每个旋钮管什么、在计算图的哪一步出现、**谁被梯度更新、谁只在验证集选定后冻结**。

### 9A.1 先分清三类“参数”

| 类型 | 例子 | 训练时会不会被梯度改？ | 怎么定 |
|------|------|------------------------|--------|
| **A. 可学习参数** | 模板 $S$、内部权重 `alpha_logits`、阶段风险 $\beta$、静态头 $\theta$ | **会**（每个 batch 反传更新） | 随机/简单初始化 → AdamW 优化 |
| **B. 结构超参** | $L,K$、阶段边界、7 个变量名单 | **不会** | 事先规定；$K,L$ 可在 val 网格选一次后冻结 |
| **C. 匹配/激活超参** | $\gamma,\tau,q,\lambda_m,\lambda_\delta,\delta_{\max}$ | **默认不反传** | 给默认值；必要时在 **val** 上网格搜索后冻结 |

你问的 $\gamma,\tau,q,\lambda_m,\lambda_\delta,\delta_{\max},a_{\min}$ 都属于 **B/C**，不是“网络权重”，但它们**决定激活 $a$ 怎么从曲线算出来**，从而间接影响 $S,\beta$ 的学习。

---

### 9A.2 一个 batch 里算什么？（训练一步拆开）

以「1 个病人、1 个变量（尿量）、1 个模板 $k$、长度 $L=6$」为例。真实训练会对 7 变量 × $K$ 个模板重复。

**【准备·只做一次，fit 在 train】**
- 尿量 $\mu,\sigma \leftarrow$ 仅 train
- 每人 48 小时：$\tilde{x},\ m,\ \delta$

**【前向·每个 batch】**

1. 对每个合法起点 $t = 0 \ldots 48-L$：取窗口 $W(t)$；用 $x^{\text{safe}},\ m,\ \delta$ 与当前 $S,\ \alpha$ 算代价矩阵 $C\ (L \times L)$；$\text{Soft-DTW}_\gamma(C) \rightarrow$ 距离 $D(t)$
2. 按起点把 $D(t)$ 分进四个阶段 $B_1 \ldots B_4$：归一化 softmin → 每阶段一个 $d_b$；soft-argmin → 每阶段一个 $t^{*}_b$
3. 激活（方案 B）：$a_b = \text{sigmoid}((q - d_b)/\tau)$
4. 对所有（变量, $k$, 阶段）的 $a$ 与静态 $C$：$\text{logit} = b_0 + \sum \beta \cdot a + \theta \cdot C$，$p = \text{sigmoid}(\text{logit})$
5. 损失：$\mathcal{L} = \text{BCE}(p, y_{\text{true}})$ + 多样性惩罚 + $\beta$ 平滑 + $\beta$ 稀疏

**【反向】**
- $\partial \mathcal{L}/\partial S,\ \partial \mathcal{L}/\partial \alpha,\ \partial \mathcal{L}/\partial \beta,\ \partial \mathcal{L}/\partial \theta \leftarrow$ 自动求导（$\gamma,\tau,q,\lambda$ 默认不求梯度，当常数）

**【更新】**
- AdamW 走一步：$S,\alpha,\beta,\theta$ 变一点

**【一个 epoch 结束】**
- 在 val 上重新算 $a$ 与 $p$（$S$ 用当前值，$q/\gamma$ 仍冻结）
- 记录 val AUPRC；若连续 patience 轮不升 → 早停

**直觉：**  
- $S,\alpha$ 决定“模板长什么样、哪点重要” → 改变 $D(t)$ 和 $a$  
- $\beta$ 决定“这个激活有多危险”  
- $\gamma,\lambda,q,\tau$ 决定“怎么量像不像、多像才算激活高” → 像尺子刻度，一般不每步学

---

### 9A.3 每个超参：人话 + 在公式里干什么 + 怎么定

#### （1）$\gamma$（gamma）— Soft-DTW 温度

| | |
|--|--|
| **人话** | 对齐时“软”还是“硬”。$\gamma$ 小 → 更接近经典 DTW（只认最优路径）；$\gamma$ 大 → 很多路径一起平均，更糊。 |
| **出现位置** | Soft-DTW 递推里的 softmin：路径代价 $R$ 的合并。 |
| **训练** | **不训练**（常数）。 |
| **默认** | 0.1 |
| **调大** | 梯度更稳，但形状区分变钝 |
| **调小** | 更贴硬对齐，可能难训、梯度尖 |
| **定法** | 默认 0.1；若 loss 长期 NaN/爆炸，试 0.2；若一切稳定可试 0.05 作敏感性 |

#### （2）激活方案 — 距离 $d$ 怎么变成 $a \in (0,1)$

| | |
|--|--|
| **人话** | 匹配距离 $d$ 越小越像；要变成 0～1 的“激活强度 $a$”。 |
| **主分析默认 B** | $a=\sigma((q-d)/\tau)$。$d$ 小 → $q-d$ 大 → $a$ 接近 1。 |
| **为何不用裸 $e^{-d/\tau}$** | 普通 Soft-DTW 的 $d$ **可能为负**，直接 exp 会假设 $d \geq 0$ 且行为怪。 |
| **备选 A** | Soft-DTW divergence（保证更合理非负）后再 $e^{-d/\tau}$。 |
| **训练** | 方案本身不训练；$S$ 通过 $d \rightarrow a \rightarrow$ 损失被间接训练。 |

#### （3）$\tau$（tau，激活温度）

| | |
|--|--|
| **人话** | 把“距离差”拉成激活时有多陡。$\tau$ 小：差一点就像/不像差很多（硬）；$\tau$ 大：过渡平滑。 |
| **出现位置** | $a=\sigma((q-d)/\tau)$；阶段 softmin 也可用同一 $\tau$。 |
| **训练** | **不训练**（常数）。 |
| **默认** | 0.5 |
| **调大** | 激活更平滑，梯度稳，区分变弱 |
| **调小** | 激活接近 0/1，可解释“有/无”，但训练可能抖 |
| **定法** | 默认 0.5；val 上可试 $\{0.3,0.5,1.0\}$ |

说明：规范里“阶段 softmin 温度”与“激活温度”**主分析共用同一个 $\tau$**，减少旋钮；敏感性可拆成两个。

#### （4）$q$ — 匹配门槛

| | |
|--|--|
| **人话** | “多好的匹配算够像”。$d=q$ 时 $a=\sigma(0)=0.5$；$d \ll q$ 时 $a$ 接近 1。 |
| **出现位置** | $a=\sigma((q-d)/\tau)$ |
| **训练** | **不反传**。 |
| **定法（强制协议）** | ① 用**当前或初始** train 上算出的 $d$ 分布，取 **中位数** 作 $q_0$；② 在 **val** 上比较 $q \in \{P_{25}, P_{50}, P_{75}\}$（相对 train 的 $d$）谁 AUPRC 好；③ **冻结 $q$**，test/外部不许再改。 |
| **错误做法** | 看 test AUC 调 $q$；每个 epoch 用全数据重估 $q$ 却不固定。 |

#### （5）$\lambda_m$ — 未测量基础惩罚

| | |
|--|--|
| **人话** | 这个小时**没测**时，在局部代价里加一笔固定罚分（与形状无关）。 |
| **出现位置** | $c_{rj}$ 中 $(1-m)[\lambda_m + \lambda_\delta \log(1+\delta)]$ |
| **训练** | **不训练**。 |
| **默认** | 0.5 |
| **调大** | 更讨厌“空洞窗口”，模型倾向在有测量的时段匹配 |
| **调小** | 更忽略缺失，可能用稀疏乱匹配 |
| **风险** | 过大时模型可能主要学“测得勤不勤”而不是波形 → **必须消融**（$\lambda_m=0$ vs $>0$） |

#### （6）$\lambda_\delta$ — 久未测 / $\delta$ 惩罚

| | |
|--|--|
| **人话** | 没测也就算了；若**很久没测**（$\delta$ 大）或**从未测过**，再加点罚。 |
| **出现位置** | $(1-m)\cdot\lambda_\delta\log(1+\delta)$ |
| **训练** | **不训练**。 |
| **默认** | 0.1 |
| **调大** | 更惩罚陈旧/空白信息 |
| **调小 / 0** | 只区分测/没测，不区分“多久没测” |
| **与 δ 定义绑定** | 从未测过：$\delta_t = t+1$；有测：0；中间缺：距上次；再 $\min(\delta, 48)$ |

#### （7）$\delta_{\max}$ — $\delta$ 上限

| | |
|--|--|
| **人话** | 距上次测量最多记到多少小时，防止极大值主导 log。 |
| **默认** | **48**（与观察窗一致） |
| **训练** | 固定规则，不训练。 |

#### （8）阶段 softmin 温度

| | |
|--|--|
| **人话** | 一个阶段里有很多起点的 $D(t)$，聚成一个 $d_b$ 时有多“只认最小 / 平均考虑”。 |
| **公式** | 归一化 softmin（除以 $|T_b|$），温度默认 **与 $\tau$ 相同 = 0.5** |
| **为何归一化** | 24–48h 起点多，不除个数会虚假更容易出小距离 |
| **训练** | 不训练。 |

#### （9）$a_{\min}$ — 进图门槛

| | |
|--|--|
| **人话** | 激活 $a$ 多高，才在**患者演化图**上当作“发生了这件事”并允许连边。 |
| **出现位置** | **建图阶段**，不在 Shapelet 训练损失里（本阶段 11 只导出连续 $a$）。 |
| **训练** | 不训练。 |
| **定法** | 在 **val** 上试阈值（如 train 激活的 P50/P75，或固定 0.4/0.5/0.6），看路径稳定性与覆盖；**冻结后**用于 test/外部。 |
| **本阶段** | 可先 **不设 $a_{\min}$**，全部导出软激活；建图脚本再读激活表并过滤。 |

---

### 9A.4 和“可学习参数”的对照（避免混）

| 名字 | 是超参吗 | 训练更新吗 |
|------|----------|------------|
| 模板 $S$ | 否，是权重 | **是** |
| $\alpha$（softmax 后） | 否，是权重 | **是**（学 logits） |
| $\beta$ 阶段风险 | 否，是权重 | **是** |
| $\gamma,\tau,q,\lambda_m,\lambda_\delta$ | **是超参** | **否**（val 选型） |
| $a_{\min}$ | **是超参** | **否**（建图时 val 选型） |
| $K,L$ | 结构超参 | **否**（val 选型） |

---

### 9A.5 推荐默认值 → 见 §9B（设备审核后冻结包）

旧稿「统一 $\tau=0.5$、统一 lr=1e-2、batch=32」**已废止**。以 §9B 为准。

---

### 9A.6 训练时日志应打印什么（便于你看懂是否在学）

每个 epoch 至少：

```text
train_loss, val_loss
val_AUPRC, val_AUROC
mean_a (train/val)          # 激活是否全 0 或全 1
mean_d per stage            # 距离是否合理
grad_norm (S)               # 是否在更新模板
```

异常对照：

| 现象 | 可能原因 |
|------|----------|
| loss 一直 NaN | $x^{\text{safe}}$ 未做；$\gamma$ 过小；lr 过大 |
| $a$ 全接近 0 或 1 | $\tau$ 过小或 $q$ 极端 |
| 模板几乎一样 | $\lambda_{\text{div}}$ 太小 |
| val 好、形状不可解释 | $\lambda_m$ 过大，学了监测频率 |

---

### 9A.7 一句话串起训练

- 曲线 + 模板 $S$ →（$\lambda_m,\lambda_\delta,\delta,\gamma$）→ 距离 $D$
- （归一化 softmin, $\tau_{\text{pool}}$）→ $d_b$（每阶段一个）
- （$q, \tau_{\text{activation}}$, sigmoid）→ 激活 $a$
- （$\beta$, 静态）→ 死亡概率 $p$
- 与真死没死比 → 主要改 $S,\alpha,\beta$
- $\gamma/\tau/q/\lambda$ 当尺子，默认不动，只在 val 上选定
- $a_{\min}$ 留给建图，本阶段可不管

---

## 9B. 设备审核冻结包（RTX 4070 Laptop 8GB）

> **设备事实**: RTX 4070 Laptop，**8GB** 显存（非桌面 12GB）；可用显存常约 6GB。  
> **关键结论**: 默认配置下瓶颈首先是 **Soft-DTW Python 循环速度**，不是显存；必须 **批量向量化 Soft-DTW** 后再开训。  
> **科学结论**: 显存只影响 batch/实现，**不**把论文主分析永久砍成单尺度；`L={6}` 仅作 debug/pilot 实现里程碑。

### 9B.1 Soft-DTW 实现要求（硬门槛）

| 项目 | 要求 |
|------|------|
| 禁止 | 对每个（人×变量×$k$×起点）用 Python 逐矩阵循环 Soft-DTW |
| 必须 | `soft_dtw_batch(cost)` GPU 向量化（输入 $N \times L \times L$；仅 $L \times L$ 步串行，$N$ 维并行） |
| 实测参考 | 向量化后 $N \approx 2.9\times 10^4$、$L=6$ 前向+反向约 **0.2s 级**；循环版同规模不可用 |
| dtype | Soft-DTW **float32 only**（$\gamma=0.1$ + 大边界，禁用 FP16） |

### 9B.2 距离尺度统一（与 $\lambda$ 配套）

多尺度 $L \in \{4,6,12\}$ 时，若 $\alpha = \text{softmax}$ 且缺失项随长度累加，**长模板距离天然更大**。

**强制：**

$$\alpha = L \cdot \text{softmax}(\text{alpha\_logits})$$

（平均权重 $\approx 1$）

$$D_{\text{used}} = \text{soft\_dtw}(C) / L$$

（按长度归一化后再进 softmin / 激活）

在此前提下可保留 $\lambda_m=0.5, \lambda_\delta=0.1$。  
**若未做 $D/L$ 归一化**，临时降为 $\lambda_m=0.1, \lambda_\delta=0.05$，并尽快补归一化。

### 9B.3 温度拆分（禁止三温共用 0.5）

| 常量 | 值 | 含义 |
|------|----|------|
| `gamma_softdtw` | 0.1 | Soft-DTW 路径软硬 |
| `tau_pool` | 0.1 | 阶段 softmin / soft-argmin 集中度 |
| `tau_activation` | 0.5 | 距离 → 激活 $a$ 的陡度 |

### 9B.4 结构阶段命名

| 模式 | $L$ | $K$ | 用途 |
|------|-----|-----|------|
| **debug** | $\{6\}$ | 2 | 单变量冒烟 |
| **pilot** | $\{6\}$ | 3 | 一期 7 变量完整训练 |
| **final** | $\{4,6,12\}$ | 3 | **论文主分析** |

- 节点键必须写 **$(v, L, k, b)$**（不可省 $L$）。  
- final 节点类型数：$7 \times 3 \times 3 \times 4 = 252$。  
- $L=\{6\}$ 只能叫 **调试/一期实现**，不能替代 final 多尺度。

### 9B.5 q 的定义（防过拟合）

**禁止**对 252 个节点分别在 val 上搜不同 $q$。

每个（变量 $v$, 长度 $L$, 模板 $k$）共用一个 $q_{v,L,k}$；其值 = 冻结模板后，训练集上该模板距离 $d$ 的分位数（默认中位数 $P_{50}$）。

若比较 $P_{25}/P_{50}/P_{75}$：对**所有模板统一**一种分位策略，一次选定。

### 9B.6 必须补充的匹配约束

| 约束 | 值 | 含义 |
|------|----|------|
| `min_observed_fraction` | 0.5 | 窗口内真实测量小时占比 $< 0.5$ → 不参与匹配（该起点无效） |
| `warping_radius` | `{4: 1, 6: 1, 12: 2}` | Soft-DTW 代价上 $\|i-j\| > \text{radius}$ 置大代价，限制过度时间扭曲 |

### 9B.7 4070 Laptop 训练工程参数

```text
batch_stays = 16
gradient_accumulation = 2   # effective batch = 32
# OOM 时: batch_stays=8, accum=4

lr_shapelet = 3e-3   # S
lr_alpha    = 1e-3   # α logits
lr_head     = 1e-3   # β, θ
weight_decay = 1e-4
grad_clip    = 1.0
max_epochs   = 100
patience     = 15
monitor      = val_AUPRC
seed         = 42
FULL_STARTS_FOR_FINAL = True   # 正式结果禁止随机抽起点
```

**不接受**: 统一 `lr=1e-2`；未向量化 Soft-DTW 直接开 7 变量全量训练。

### 9B.8 损失与类别权重

$$\lambda_{\text{div}}=0.1, \quad \lambda_{\text{time}}=0.01, \quad \lambda_{\text{sp}}=10^{-3}$$

`pos_weight` $\approx 5.887$（train: 1105 死 / 6505 活，约 14.5%）

日志必须**分别记录** BCE / div / time / sp；理想情况下各正则长期不超过 BCE 的约 10%–20%。  
加权 BCE 可能伤校准 → 最终在 val 做概率校准并报告 **Brier**。

### 9B.9 最终批准配置包（代码常量）

见 `scripts/utils/shapelet_constants.py` 中：

```text
DEBUG_* / PILOT_* / FINAL_*
GAMMA_SOFTDTW, TAU_POOL, TAU_ACTIVATION
LAMBDA_M, LAMBDA_DELTA, DELTA_MAX, MIN_OBS_FRAC, WARPING_RADIUS
LR_SHAPELET, LR_ALPHA, LR_HEAD
BATCH_STAYS, GRAD_ACCUM, MAX_EPOCHS, PATIENCE, MONITOR
```

### 9B.10 开训前检查清单（分阶段）

| 阶段 | 内容 | 状态 |
|------|------|------|
| **G0 基础算子** | 向量化 Soft-DTW + 缺失代价 + 滑窗 + min_obs + D/L；合成张量 fwd/bwd；单元测试 | **已通过**（见 §12.0） |
| **G1 训练骨架** | `shapelet_prep` / `shapelet_model` / `11_shapelet_learn` 可 CLI 启动；默认 `--dry-run` 合成 | **已就绪**（未默认开训） |
| **G2 debug 冒烟** | 单变量（建议 `urine_output_ml`）+ $L=6$ + $K=2$，真实 train 网格，2–5 epoch | 未开始 |
| **G3 pilot** | 7 变量 + $L=6$ + $K=3$，完整早停；**不调 test** | 未开始 |
| **G4 final** | $L \in \{4,6,12\}$；论文主分析；test **只评估一次** | 未开始 |

**硬门槛（任一未满足禁止进 G2）：**

1. Soft-DTW 已向量化，合成 $N \approx 3\times 10^4$、$L=6$ 可在秒级完成 fwd+bwd；  
2. $\alpha = L \cdot \text{softmax}$，$D = \text{soft\_dtw}/L$；  
3. 三温度已拆分（$\gamma$ / $\tau_{\text{pool}}$ / $\tau_{\text{act}}$）；  
4. `min_obs_frac` 与 `warping_radius` 走真实代码路径；  
5. 全项目测试绿；**未**用患者数据拟合模板、**未**写激活表、**未**碰 test 选型。

---

## 10. 代码布局

```text
scripts/
  utils/
    shapelet_constants.py   # 变量名、阶段、默认超参（§9B 冻结）
    soft_dtw.py             # 向量化 Soft-DTW + D/L
    shapelet_cost.py        # δ、x_safe、滑窗代价、shapelet_distances
    shapelet_prep.py        # 网格→张量、train-only 标准化、Dataset
    shapelet_model.py       # ShapeletBank + 阶段 softmin + 风险头 + 损失
  11_shapelet_learn.py      # CLI：默认 --dry-run；--train 才读患者数据
  tests/
    test_soft_dtw_batch.py
    test_shapelet_model_skeleton.py
docs/
  12_Shapelet实现规范.md    # 本文
```

---

## 11. 输出产物

| 路径 | 内容 |
|------|------|
| `data/processed/shapelet/templates_{var}.npz` | 每变量模板 $S$、$\alpha$ |
| `data/processed/shapelet/activations.parquet` | 长表：ICUSTAY_ID, split, var, L, k, stage, a, t_star |
| `data/processed/shapelet/meta.json` | 超参、$\mu\sigma$、$q$、seed |
| `outputs/tables/shapelet_val_metrics.csv` | val/test AUROC/AUPRC/Brier |
| `outputs/figures/shapelet_templates_*.png` | 模板曲线（可选） |

`activations` 是后续**患者级演化图**的唯一事件输入。

---

## 12. 验收标准

### 12.0 基础计算模块验收（G0，训练代码开写前）

> **目的**: 证明冻结配置在 4070 上**算得动、可反传、契约正确**；**不**等于模型已训练。  
> **数据边界**: 仅合成张量与单元测试；禁止读 `hourly_grid` 做拟合、禁止生成 templates/activations、禁止 test 评估。

| # | 检查项 | 通过标准 | 实测/状态 |
|---|--------|----------|-----------|
| G0-1 | 向量化 Soft-DTW | `soft_dtw_batch(N,L,L)`；无 per-matrix Python 循环 | 已实现 `scripts/utils/soft_dtw.py` |
| G0-2 | 独立 oracle 对照 | 批结果与标量 DP 一致（atol 1e-5）；**禁止**实现与自身比 | `test_batch_matches_single` |
| G0-3 | 反传有限 | `cost.grad` 全 finite | `test_backward_finite` + 4070 合成 |
| G0-4 | D/L 归一化 | `soft_dtw_normalized = soft_dtw / L` | `test_normalized_distance_divides_by_length` |
| G0-5 | float32 硬门 | 非 float32 / $\gamma \leq 0$ 抛错 | `test_soft_dtw_rejects_non_float32_and_bad_gamma` |
| G0-6 | α 尺度 | $\alpha = L \cdot \text{softmax}$ → mean $\approx 1$ | `test_alpha_has_mean_one` |
| G0-7 | min_obs_frac | 观测占比 <0.5 → valid=False；距离填 invalid | `test_batch_windows_apply_minimum_observed_fraction` |
| G0-8 | 生产滑窗路径 | `cost_for_all_starts_batch` → `shapelet_distances` | 已实现；单序列 API 委托 batch |
| G0-9 | 合成 Soft-DTW 压测 | $N \approx 28896$, $L=6$, fwd+bwd 秒级；grad finite | 用户 4070 已跑（~0.2s 级） |
| G0-10 | 生产核心压测 | $B=16$, banks=21 ($7\times3$), $T=48$, $L=6$, fwd+bwd；shape/$\alpha$ grad finite | 用户 4070 已跑（生产函数，非玩具） |
| G0-11 | 全项目测试 | `pytest scripts/tests` 全绿 | **25 passed** |
| G0-12 | 污染边界 | 无 templates / activations / 患者拟合 / test 选型 | **保持中** |

**G0 判定**: G0-1…G0-12 全满足 → 允许进入 **G1 训练骨架**（仍可不读全量患者数据）。  

### 12.0b G1 训练骨架验收

| # | 检查项 | 状态 |
|---|--------|------|
| G1-1 | `shapelet_prep.py`：train-only $\mu\sigma$、Dataset、合成 batch | 已写 |
| G1-2 | `shapelet_model.py`：滑窗匹配 → 阶段 softmin → $a$ → BCE+正则 | 已写 |
| G1-3 | `11_shapelet_learn.py`：默认 dry-run；`--train` 才读网格 | 已写 |
| G1-4 | `python scripts/11_shapelet_learn.py --dry-run` 损失有限、梯度 finite | 已通过 |
| G1-5 | 单元测试含模型骨架 | `test_shapelet_model_skeleton.py` |
| G1-6 | **未**默认全量 train；**未**自动 eval test | 保持 |

**当前节点**: **G1 骨架就绪**；下一门 **G2 debug 冒烟** 需显式  
`python scripts/11_shapelet_learn.py --train --mode debug`（仍建议 `--max-stays` / 少 epoch 先试）。

### 12.1 训练阶段验收（G2–G4，开训后）

1. train 上 loss 下降；无持续 NaN。  
2. val 上 AUPRC/AUROC 可计算；test 只报一次。  
3. 激活表无 NaN；$a \in (0,1)$。  
4. 同一变量不同 $k$ 的模板不完全重合（div 起作用）。  
5. 单元测试（训练模块补齐后）：  
   - $0 \times \text{NaN}$ 路径不出现；  
   - 合法起点上界 $48-L$；  
   - 阶段归属按起点；  
   - 归一化 softmin 对重复扩展起点集合不偏向。

### 12.2 `shapelet_cost` 公开 API 契约（G1 调用方必读）

生产路径（训练一步的核心）:

```text
delta = compute_delta(m)                          # (B,T)
costs, valid = cost_for_all_starts_batch(x,m,delta,S,alpha_logits)
d = shapelet_distances(costs, valid)              # (B, T-L+1)，已 /L
```

| 函数 | 输入形状 | 输出 | 备注 |
|------|----------|------|------|
| `compute_delta` | `m (…,T)` | `δ (…,T)` | 截断 `DELTA_MAX` |
| `alpha_from_logits` | `(L,)` 或 `(…,L)` | 同形 | **$L \cdot \text{softmax}$** |
| `cost_for_all_starts_batch` | `x,m,δ (B,T)`；`S,α_logits (L,)` | `costs (B,S,L,L)`，`valid (B,S)` | **主入口**；`S=T-L+1` |
| `cost_for_all_starts` | `(T,)` 单序列 | `(S,L,L)`, `(S,)` | 委托 batch |
| `cost_batch_windows` | 已切窗 `(N,L)` | `(N,L,L)` | 内部用；`α_logits` |
| `pairwise_cost_matrix` | 单窗 `(L,)` | `(L,L)` | 低层；入参是 **已缩放 α**，不是 logits |
| `shapelet_distances` | `costs (…,L,L)`, `valid` | `(…,)` | Soft-DTW$/L$；invalid→`1e4` |

**易错点（已在实现中对齐）:**

- `cost_batch_windows` / 滑窗入口参数名是 **`alpha_logits`**；`pairwise_cost_matrix` 参数名是 **`alpha`**（须先 `alpha_from_logits`）。  
- `warping_radius=None`：**一律**按 `WARPING_RADIUS[L]` 上带（单窗与 batch 一致）。  
- 阶段 softmin / $\text{sigmoid}((q-d)/\tau_{\text{act}})$ / 风险头：**不在**本模块，属 `shapelet_model`。

---


## 13. 与后续模块接口

- `activations.parquet`
- 过滤 $a \geq a_{\min}$（val 定）
- 每患者按 $t^{*}$ 连边
- 患者级图 + 共享节点字典 $(var, L, k, \text{stage})$
- GNN → $z_{\text{graph}}$
- $z_{\text{graph}} \parallel z_{\text{sparse}} \parallel z_{\text{static}}$ → 预测（$z_{\text{sparse}}$：11 变量阶段摘要）

本阶段只保证 **activations + templates** 正确可复现。

---

## 14. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-03-26 | v1.0：三层变量、NaN 安全、方案 B 激活、归一化 softmin、起点定阶段、K 为 val 超参 |
| 2026-03-26 | v1.1：新增 §9A 超参详解与训练全过程（γ/τ/q/λ/δ/a_min 人话 + 默认冻结表） |
| 2026-03-26 | v1.2：§9B RTX 4070 Laptop 审核冻结包；向量化 Soft-DTW 硬门槛；三温度拆分；D/L 与 L·softmax；debug/pilot/final |
| 2026-08-05 | v1.3：§9B.10/§12.0 G0 基础算子验收清单（25 tests + 4070 合成压测）；§12.2 shapelet_cost API 契约；G0 通过、未开训 |
| 2026-08-05 | v1.4：G1 骨架 `shapelet_prep` / `shapelet_model` / `11_shapelet_learn`；默认 dry-run；§12.0b |
| 2026-08-05 | v1.5：公式改为明文 + text 代码块；去掉 LaTeX 定界符与 math fence |
| 2026-08-05 | v1.6：公式改回标准 LaTeX（行内 `$...$` / 独立行 `$$...$$`），仅用主流渲染器（GitHub/Typora/Obsidian/VS Code）兼容的命令 |
