# TASER-AKI 患者级 Shapelet 演化图与 GNN 实现规范

> **版本**：v1.0（2026-08-07）  
> **用途**：可直接交给实现 Agent 编写 `15`–`18` 阶段脚本。  
> **效力**：涉及患者图、图边、GNN 和图级死亡预测时，以本文为准；本文覆盖 `docs/11` §6 与 `docs/12` §13 中未冻结或互相冲突的部分。  
> **不改变**：HA-AKI 队列、onset、48 h landmark、SUBJECT_ID 划分、Shapelet 数值/缺失处理和 test 锁定协议。  
> **当前状态**：`13_shapelet_pilot.py` 正在训练；GNN 尚未实现。本文只定义接口与验收，不授权开启 test 或外部验证。

---

## 0. 一页冻结包

| 项 | 冻结选择 |
|---|---|
| 图学习对象 | **每位患者一张图**，共享训练集冻结的节点类型字典 |
| 节点键 | `(var, L, k, stage)`，禁止省略 `L` |
| Pilot 节点类型 | `7 × 1 × 3 × 4 = 84` |
| Final 节点类型 | `7 × 3 × 3 × 4 = 252`，`L={4,6,12}` |
| Shapelet 状态 | 建图前冻结 `S/alpha/q/scaler`；主分析 `q=train P50` |
| 主事件规则 | 每个患者每个 `(var, stage)` 最多保留 1 个最强 `(L,k)`，且 `valid=True, a>=0.5` |
| 同时发生 | `CO_OCCUR`：`abs(delta_t) <= 2 h`，无因果方向 |
| 先后转移 | `PRECEDES`：`2 h < delta_t <= 12 h` |
| 稀疏控制 | 每个事件最多 2 个后续转移、4 个共现邻居；训练支持度过滤 |
| 预测图边特征 | 关系类型、时间间隔、激活乘积、同变量/同阶段、训练支持度 |
| 结局富集权重 | **不输入第一版 GNN**；仅用于训练后路径解释，避免目标编码循环 |
| GNN | 2 层、方向/关系感知、残差 + LayerNorm + dropout |
| 图池化 | 节点注意力池化 + 边注意力池化 |
| 最终输入 | `[h_graph || activation_projection || sparse_projection || static_projection]` |
| 标签 | landmark `t0+48 h` 后住院死亡 `HOSPITAL_MORTALITY` |
| 训练监控 | `val_AUPRC`；test 默认完全不构建、不加载 |
| 实现依赖 | 首版使用 **纯 PyTorch**，不新增 PyTorch Geometric 依赖 |

---

## 1. 目标与非目标

### 1.1 目标

1. 把冻结 Shapelet 输出的 `(a, t_star)` 转换为稀疏、可审计的患者级有向事件图；
2. 明确区分“共同发生”与“先后转移”；
3. 用共享参数的方向/关系感知 GNN 学习患者图表示 `h_graph`；
4. 联合 Shapelet 激活、稀疏阶段摘要和静态协变量预测死亡；
5. 保留患者级节点、边和模型贡献，供后续路径解释及 RL 使用；
6. 全流程可断点、可复现、可审计且不污染 test。

### 1.2 非目标

- 不把图边解释为因果关系；
- 不把 GNN 当作保证提高 AUPRC/AUROC 的工具；
- 不在 GNN 内重新学习或修改 Shapelet；
- 不用全体患者构一张图并把它当作唯一训练样本；
- 不给每条路径人工贴死亡标签；
- 不在第一版 GNN 中输入死亡富集边权 `R`；
- 不在本阶段实现 RL；
- 不根据 test 表现修改节点、阈值、边、模型或路径。

---

## 2. 实际执行链与文件布局

### 2.1 阶段编号

`14` 预留给正式多尺度 Shapelet，不得被 GNN 占用：

```text
13_shapelet_pilot.py              # 当前 L6 pilot
14_shapelet_final.py              # 预留：L={4,6,12}, K=3
15_export_shapelet_events.py      # 冻结/校准 q，导出事件长表
16_build_evolution_graphs.py      # train 建词表；生成 train/val 患者图
17_train_gnn_risk.py              # 图模型训练、验证、断点续训
18_graph_diagnostics.py           # 图健康、模型诊断、解释性产物
```

### 2.2 工具模块

```text
scripts/utils/
  graph_constants.py              # 所有冻结图/GNN常量
  graph_events.py                 # q校准、激活导出、事件选择
  graph_builder.py                # 节点字典、边生成、词表、图打包
  graph_data.py                   # Dataset / collate / pack schema
  graph_model.py                  # DirectionalGraphLayer / pooling / risk model
  graph_metrics.py                # AUPRC/AUROC/Brier/校准辅助

scripts/tests/
  test_graph_events.py
  test_graph_builder.py
  test_graph_no_leakage.py
  test_graph_model.py
  test_graph_resume.py
```

### 2.3 运行顺序

```text
frozen Shapelet checkpoint
  -> 15_export_shapelet_events
  -> activations_long.parquet + node_dictionary
  -> 16_build_evolution_graphs
  -> graph_pack_train.pt + graph_pack_val.pt
  -> 17_train_gnn_risk
  -> ckpt_best.pt + val_predictions.parquet
  -> 18_graph_diagnostics
  -> patient top nodes/edges/path families
```

实现 Agent 可以先用合成事件和 pilot checkpoint 做工程冒烟；**论文主结果必须等待 final 多尺度 checkpoint**。

---

## 3. 上游输入契约

### 3.1 Shapelet checkpoint 必须包含

```text
model_state
keys                     # [(var, L, k), ...]
activation_names         # var|Lx|kx|stage
head_mode
meta.vars
meta.lengths
meta.k
meta.seed
```

另外必须冻结并保存：

```text
value_scaler             # 7个密集变量，仅train拟合
static_scaler            # 11个静态协变量，仅train拟合
q_policy                 # train_p50
q                         # 每个(var,L,k)一个值
stage_bounds             # B1-B4
shapelet_config_hash
train_subject_hash
```

当前 pilot checkpoint 的 `q=0` 只用于 G2 工程门控，**不得直接作为正式患者图事件阈值的基础**。

### 3.2 q 校准

`15_export_shapelet_events.py` 必须执行：

1. 加载 `ckpt_best.pt`；
2. 冻结 `shapelets` 与 `alpha_logits`；
3. 仅在 train 患者上收集每个模板、每阶段的有效 `d_stage`；
4. 对每个 `(var,L,k)` 汇总全部有效 train 距离并取 `P50`；
5. 调用/等价实现 `set_q_from_distances`；
6. 保存独立的 `shapelet_frozen_qp50.pt`；
7. 后续 train/val/test/external 一律复用，不重新计算 q。

禁止按节点、阶段或结局分别搜索 q。

### 3.3 Shapelet 事件长表

输出：

```text
data/processed/shapelet/events/<run_name>/activations_long.parquet
```

一行对应一个患者的一个全局节点类型，至少包含：

| 列 | 类型 | 含义 |
|---|---|---|
| `ICUSTAY_ID` | int64 | stay 主键 |
| `SUBJECT_ID` | int64 | 泄漏审计 |
| `split` | string | train/val；默认不导出 test |
| `HOSPITAL_MORTALITY` | int8 | 图级标签，只能进入 `y`，不得进入特征 |
| `node_id` | int32 | 冻结字典 ID |
| `var` | string | 7 个密集变量之一 |
| `L` | int16 | 4/6/12 或 pilot 6 |
| `k` | int16 | 0-based 模板编号 |
| `stage` | string | B1_0_6 等 |
| `stage_idx` | int8 | 0..3 |
| `a` | float32 | q 校准后的软激活 |
| `d_stage` | float32 | 阶段距离，便于审计 |
| `t_star` | float32 | onset 后近似模式起始小时 |
| `is_valid` | bool | 该阶段至少存在有效匹配窗口 |

`t_star` 是 soft-argmin 起始位置，不是原始测量时间。

### 3.4 节点字典

输出：

```text
node_dictionary.parquet
node_dictionary.json
```

唯一键：

```text
(var, L, k, stage)
```

字典顺序必须来自冻结 checkpoint 的 `activation_names`，不得按不同 split 重排。代码不得硬编码 84 或 252，应从 checkpoint 推导 `N_NODE_TYPES`。

### 3.5 稀疏阶段摘要 `z_sparse`

固定 11 个变量：

```text
creatinine, bun, potassium, sodium, bicarbonate,
lactate, ph, platelet, wbc, hemoglobin, gcs_total
```

实现时在 `graph_constants.py` 新建 `GRAPH_SPARSE_VARS` 保存这 11 项。**不要直接导入当前 `constants.py` 的 `TIER_B_SPARSE`**：后者是变量审计层的 16 项稀疏池，与本图模型的 11 项阶段背景不是同一概念。同理，7 个图骨架变量必须使用 `shapelet_constants.py::SHAPELET_VARS`，不能使用更宽的 `TIER_A_SHAPELET`。

每个变量在 B1–B4 每阶段生成 7 个特征：

```text
last
min
max
delta_from_previous_stage_last
is_measured
measured_fraction
hours_since_last_measurement_at_stage_end
```

维度：

```text
11 × 4 × 7 = 308
```

规则：

- 只用 `hourly_grid_48h.parquet` 的真实测量与 `*_is_measured`；
- 不把缺失当 0；
- `last/min/max/delta` 先计算再用 train 中位数填补；
- 连续值标准化参数只在 train 拟合；
- `hours_since_last` 查看截至阶段末的全部既往历史，截断到 48 h，再除以 48；
- 当前阶段无测量时 `is_measured=0, measured_fraction=0`；
- 所有列顺序写入 `sparse_feature_names.json`。

### 3.6 静态协变量 `z_static`

沿用 `SHAPELET_STATIC_COLS` 的 11 列及其 train-only scaler：

```text
AGE, GENDER_M, onset_stage, onset_hour_from_icu,
CHF, Diabetes, HTN, CKD_FLAG, COPD, LiverDisease, Arrhythmia
```

---

## 4. 患者事件选择：连续激活与图拓扑分离

### 4.1 原则

所有 sigmoid 激活几乎都大于 0，不能把 `a>0` 当作“事件发生”。

主分析事件规则：

```text
is_valid == True
and a >= 0.5
and 在同一患者、同一(var, stage)内的所有(L,k)中排名第1
```

每位患者最多：

```text
7 var × 4 stage = 28 个活跃事件
```

### 4.2 排序与确定性 tie-break

同一 `(ICUSTAY_ID,var,stage)` 内按以下顺序选 1 个：

1. `a` 降序；
2. `d_stage` 升序；
3. `L` 升序；
4. `k` 升序；
5. `node_id` 升序。

输出 `event_rank=1`。其他节点仍保留在 `a_dense` 中供跳连分支使用，但 `active_mask=0`，不进入患者图拓扑。

### 4.3 空图

若患者没有任何事件：

- 保留该患者；
- `active_mask` 全 0；
- `h_graph` 使用可训练 `empty_graph_embedding`；
- 仍可通过 `a_dense/z_sparse/z_static` 预测；
- 报告各 split 的空图比例；不得删除空图患者。

---

## 5. 患者图与边定义

### 5.1 图对象

每位患者：

$$G_i=(V_G,E_i)$$

- `V_G`：共享冻结节点字典；
- `active_mask_i`：该患者真正进入拓扑的节点；
- `E_i`：只由该患者活跃事件的时间关系产生；
- 未激活节点不参与消息、注意力或池化。

### 5.2 共同发生 `CO_OCCUR`

若两个不同事件：

$$|t_u^*-t_w^*|\leq 2\ \mathrm{h}$$

则为共同发生候选。它是对称关系，无因果方向。

为限制密度：

- 每个节点最多 4 个 `CO_OCCUR` 邻居；
- 候选对按 `abs(delta_t)` 升序、`a_u*a_w` 降序、节点 ID 升序处理；
- 只有双方当前度数都小于 4 时加入；
- 存储为一条 canonical 无向边 `(min_id,max_id,CO_OCCUR)`；模型内部双向传消息。

### 5.3 先后转移 `PRECEDES`

若：

$$2\ \mathrm{h}<t_w^*-t_u^*\leq12\ \mathrm{h}$$

则 `u -> w` 为候选先后边。

为限制密度：

- 每个源事件最多保留最近 2 个后续事件；
- 排序：`delta_t` 升序、目标激活降序、目标 `node_id` 升序；
- 不额外连接所有可达后代；长路径由多跳消息表达；
- 同节点不连自环；模型层内使用单独的 self transform。

### 5.4 Canonical 边上限

对最多 28 个事件：

```text
PRECEDES <= 28 × 2 = 56
CO_OCCUR <= 28 × 4 / 2 = 56
总 canonical edges <= 112
```

存储预留：

```text
MAX_CANONICAL_EDGES = 128
```

超出必须抛错，不得静默截断。

### 5.5 训练集全局边词表

仅使用 train 患者候选边，按：

```text
(src_node_id, dst_node_id, edge_type)
```

统计唯一患者支持数。`CO_OCCUR` 使用 canonical 节点顺序。

最低支持数：

```python
min_edge_support = max(20, ceil(0.005 * n_train_stays))
```

低于支持数的边类型从所有 split 删除。val/test 只映射到冻结 train 词表：

- 不得新增边类型；
- 未见转移记录为 `dropped_unseen_edge`；
- 必须报告各 split 的 dropped 比例。

### 5.6 第一版预测边特征

每条 canonical 边：

```text
edge_type                 # CO_OCCUR / PRECEDES
abs_delta_t / 12
a_source * a_target
same_variable
same_stage
log1p(train_support) / log1p(n_train)
```

禁止放入：

```text
patient outcome
death-group edge frequency
survival-group edge frequency
R_death_enrichment
任何val/test统计
```

---

## 6. 死亡富集边与“死亡路径”的位置

### 6.1 不作为第一版 GNN 输入

旧方案：

$$R_{uw}=\log P(u\to w\mid y=1)-\log P(u\to w\mid y=0)$$

`R` 使用结局标签计算。如果把同一训练患者参与计算的 `R` 再作为其输入，会形成目标编码循环。第一版采取最安全方案：

```text
GNN只通过图级BCE标签学习死亡关联；
R仅在训练完成后用于解释和论文表格。
```

未来若必须输入 `R`，训练患者必须使用 out-of-fold/cross-fitted `R`；不得直接使用全 train 的 in-sample `R`。

### 6.2 精确路径与路径家族

精确路径：节点 ID 完全一致，例如：

```text
mbp|L6|k1|B1 -> urine|L6|k2|B2 -> spo2|L6|k0|B3
```

路径家族：忽略 `L,k`，保留变量与阶段：

```text
mbp|B1 -> urine|B2 -> spo2|B3
```

GNN 不直接输出“唯一死亡路径”。训练后只枚举长度 1–3 条边的简单路径，并同时报告：

- 死亡组携带率；
- 存活组携带率；
- 路径携带者死亡率；
- 平滑 log-odds ratio / 置信区间；
- 模型归因方向；
- 支持患者数；
- 内部测试/外部复现情况。

只有富集、正向归因且稳定的路径可称“死亡相关演化路径”；禁止称为致死因果路径。

---

## 7. 图张量与磁盘契约

### 7.1 每位患者的逻辑张量

令：

```text
N = 节点类型数（pilot 84 / final 252）
E = 当前患者 canonical edge 数（0..112）
F_sparse = 308
F_static = 11
```

张量：

| 名称 | shape | dtype | 含义 |
|---|---:|---|---|
| `a_dense` | `(N,)` | float32 | 所有节点软激活，未选事件也保留 |
| `node_time` | `(N,)` | float32 | `t_star/48` |
| `node_time_within_stage` | `(N,)` | float32 | 阶段内归一化时间 |
| `active_mask` | `(N,)` | bool | 是否进入图拓扑 |
| `edge_src` | `(E,)` | int64 | canonical 起点 |
| `edge_dst` | `(E,)` | int64 | canonical 终点 |
| `edge_type` | `(E,)` | int64 | `0=CO_OCCUR, 1=PRECEDES` |
| `edge_attr` | `(E,5)` | float32 | §5.6 连续/二值特征 |
| `sparse` | `(308,)` | float32 | train-only处理后的阶段摘要 |
| `static` | `(11,)` | float32 | train-only标准化静态协变量 |
| `y` | `()` | float32 | 图级结局 |
| `stay_id` | `()` | int64 | 审计键 |
| `subject_id` | `()` | int64 | 泄漏审计 |

### 7.2 Pack 文件

```text
data/processed/graphs/<run_name>/
  node_dictionary.parquet
  node_dictionary.json
  event_policy.json
  sparse_feature_names.json
  sparse_scaler_train.json
  edge_vocabulary_train.parquet
  graph_pack_train.pt
  graph_pack_val.pt
  graph_manifest.json
```

默认不得生成：

```text
graph_pack_test.pt
```

Pack 可以按患者列表保存可变长边；`Dataset` 取样时填充到 `MAX_CANONICAL_EDGES=128` 并返回 `edge_mask`。不要为了方便把结局统计写入节点或边特征。

### 7.3 Manifest 必须记录

```text
input checkpoint absolute path + sha256
shapelet_config_hash
node_dictionary_hash
edge_vocabulary_hash
train_subject_hash
split counts / mortality counts
q policy and q hash
event policy
edge policy
sparse/static feature names and scaler hashes
seed
created_at
code version / git commit if available
test_built=false
```

---

## 8. 第一版 GNN 结构

### 8.1 依赖边界

使用纯 PyTorch：

```text
torch.nn.Embedding
torch.nn.Linear
torch.index_add_ / scatter_reduce_
masked softmax
```

不在第一版引入 `torch_geometric`，避免 Windows/CUDA 轮子与当前环境不一致。实现必须对小图使用向量化批处理，禁止逐节点 Python 神经网络前向循环。

启动时检查 `torch>=2.0`；若当前版本没有 `scatter_reduce_`，应实现等价的数值稳定 segment softmax，而不是静默退化为逐患者/逐节点循环。

### 8.2 节点初始表示

冻结超参：

```text
node_type_dim = 16
hidden_dim = 64
```

可训练节点类型表：

$$E_{node}\in\mathbb{R}^{N\times16}$$

患者特异连续特征：

```text
[a, t_star/48, time_within_stage]
```

初始节点表示：

$$h_{iu}^{(0)}=\operatorname{GELU}(W_{node}[E_u,a_{iu},t_{iu},t^{stage}_{iu}]+b)$$

随后：

```text
h0 = LayerNorm(h0)
h0 = h0 * active_mask
```

未激活节点必须保持 0，不得仅凭节点类型嵌入参与池化。

空图使用独立可训练参数：

```text
empty_graph_embedding: (96,)
```

### 8.3 关系与方向

模型内部使用 3 种消息关系：

```text
FROM_PREDECESSOR   # 对u->w，w读取u
FROM_SUCCESSOR     # 对u->w，u读取w
CO_OCCUR           # 双向、同一关系
```

每种关系有独立变换：

```text
W_from_predecessor
W_from_successor
W_cooccur
```

这三者是所有患者共享的可训练矩阵，不是每条边一个矩阵。

### 8.4 单层方向/关系感知消息传递

对于目标节点 `u`：

$$
\tilde h_u=
W_{self}h_u
+\sum_{w\in N_{pred}(u)}\eta_{wu}W_{pred}h_w
+\sum_{w\in N_{succ}(u)}\eta_{uw}W_{succ}h_w
+\sum_{w\in N_{co}(u)}\eta^{co}_{uw}W_{co}h_w
$$

边特征先经：

```text
edge_encoder: Linear(5, 16) -> GELU -> Linear(16, 16)
```

注意力分数：

$$s_{wu}=\operatorname{MLP}_{att}[h_w,h_u,e_{wu},r_{type}]$$

在同一目标节点、同一关系邻居内做 segment softmax 得到 `eta`。`eta` 是每次前向动态值；真正训练的是 `MLP_att` 参数。

层输出：

```text
h_new = LayerNorm(h + Dropout(GELU(message)))
h_new = h_new * active_mask
```

### 8.5 层数

```text
n_layers = 2
dropout = 0.20
```

两层可以让节点感知两跳路径，如：

```text
低血压 -> 尿量下降 -> 低氧
```

第一版不得为追指标增加深层网络。若以后改变层数，只能在 train/val 开发阶段完成。

### 8.6 节点池化

节点得分：

$$r_u=v^T\tanh(W_ph_u)$$

对 active nodes 做 masked softmax：

$$\omega_u=\operatorname{softmax}(r_u)$$

节点池化：

$$h_{node}=\sum_u\omega_uh_u\in\mathbb{R}^{64}$$

保存 `omega_u` 供患者级解释，但不得把注意力直接解释为因果贡献。

### 8.7 边池化

使用最终节点表示构造 canonical 边表示：

```text
edge_repr = EdgeMLP([h_src, h_dst, relation_embedding, edge_attr])
edge_dim = 32
```

对真实边做 masked attention pool：

```text
h_edge: (32,)
```

无边患者使用全 0 `h_edge`；空图由 `empty_graph_embedding` 处理。

图表示：

```text
h_graph = concat(h_node, h_edge)    # 96维
```

`h_graph` 是患者全部保留节点/边经过两层消息传递后的压缩表示，不是可逆的路径清单。完整路径必须保留在 `graph_pack` 中。

### 8.8 非图分支与最终融合

```text
activation_projection: Linear(N, 64) -> GELU -> Dropout
sparse_projection:     Linear(308, 64) -> GELU -> Dropout
static_projection:     Linear(11, 32) -> GELU -> Dropout
```

最终融合：

```text
z = concat(h_graph[96], a_proj[64], sparse_proj[64], static_proj[32])
  = 256 dims

risk_head:
  Linear(256,128)
  GELU
  Dropout(0.20)
  Linear(128,1)
```

输出：

```text
logits: (B,)
prob: sigmoid(logits)
h_graph: (B,96)
node_attention: (B,N)
edge_attention: (B,Emax)
```

### 8.9 `ForwardOut` 建议

```python
@dataclass
class GraphForwardOut:
    logits: torch.Tensor
    prob: torch.Tensor
    h_graph: torch.Tensor
    node_attention: torch.Tensor
    edge_attention: torch.Tensor
    loss_bce: torch.Tensor
    loss: torch.Tensor
```

---

## 9. GNN 训练协议

### 9.1 冻结与可训练状态

冻结：

```text
Shapelet S
alpha
q
value/static/sparse scalers
node dictionary
event policy
edge vocabulary
graph packs
```

训练：

```text
node type embedding
node input projection
relation message matrices
edge encoder
attention networks
node/edge pooling
activation/sparse/static projections
risk head
empty graph embedding
```

### 9.2 损失

主损失：

```python
binary_cross_entropy_with_logits(
    logits,
    y.float(),
    pos_weight=train_n_negative / train_n_positive,
)
```

第一版不添加路径监督、边监督或结局富集正则。避免在图尚未验收时引入多目标损失。

注意：加权 BCE 的原始 sigmoid 不是已校准临床概率；正式报告需在模型冻结后单独校准。

### 9.3 优化器与训练参数

```text
optimizer = AdamW
lr = 1e-3
weight_decay = 1e-4
batch_graphs = 32
OOM fallback = 16
max_epochs = 100
patience = 15
min_delta = 1e-4
monitor = val_AUPRC
grad_clip = 1.0
seed = 42
device = auto
```

禁止中途根据曲线临时改变模型结构或阈值。

### 9.4 一次前向与反向

```text
batch patient graphs
  -> node initial states
  -> directed relational layer 1
  -> directed relational layer 2
  -> node pool + edge pool
  -> h_graph
  -> concatenate a/sparse/static projections
  -> logits
  -> weighted BCE
  -> backward()
  -> clip_grad_norm_(1.0)
  -> optimizer.step()
```

每位患者只有一个图级标签。节点和边没有人工死亡标签，参数通过图级误差的链式反向传播间接学习。

### 9.5 验证与 early stopping

每个 epoch 在 val 报告：

```text
AUPRC
AUROC
Brier（仅作原始分数诊断）
loss
empty/nonempty graph subgroup metrics
```

`ckpt_best.pt` 只按 `val_AUPRC` 保存。test 不得用于 early stopping、阈值、模型或路径选择。

### 9.6 Checkpoint 契约

```text
model_state
optimizer_state
epoch
best_metric
best_state
bad_epochs
history
graph_config
node_dictionary_hash
edge_vocabulary_hash
shapelet_checkpoint_hash
sparse/static scaler hashes
train_subject_hash
seed
saved_at
```

必须支持：

```powershell
python -u scripts/17_train_gnn_risk.py --train --resume auto
```

---

## 10. 脚本级详细契约

### 10.1 `15_export_shapelet_events.py`

CLI：

```powershell
python -u scripts/15_export_shapelet_events.py `
  --checkpoint data/processed/shapelet/final/<run>/ckpt_best.pt `
  --q-policy train_p50 `
  --splits train,val `
  --device auto
```

执行：

```text
load checkpoint
-> validate keys/config/split
-> prepare train/val using train-only scaler
-> collect train valid distances
-> set q=P50
-> save frozen qp50 checkpoint/scalers
-> infer a,d_stage,t_star
-> derive is_valid
-> write node dictionary and long activations
-> leakage audit
```

默认禁止 `test`。若以后解锁，必须显式 `--unlock-test` 且检测存在完整冻结模型 manifest。

### 10.2 `16_build_evolution_graphs.py`

CLI：

```powershell
python -u scripts/16_build_evolution_graphs.py `
  --events data/processed/shapelet/events/<run>/activations_long.parquet `
  --run-name graph_<run>
```

执行：

```text
validate node dictionary
-> build train-only sparse/scalers
-> apply event top1+a_min policy
-> generate candidate patient edges
-> build train edge vocabulary/support
-> freeze vocabulary
-> remap/filter train and val
-> build graph packs
-> write manifest and diagnostics
```

该脚本必须是确定性的；同一输入运行两次，node/edge hashes 完全一致。

### 10.3 `17_train_gnn_risk.py`

CLI：

```powershell
# 默认仅打印配置/合成dry-run，不读患者图训练
python -u scripts/17_train_gnn_risk.py --dry-run

# 冒烟
python -u scripts/17_train_gnn_risk.py --train --max-stays 128 --max-epochs 2

# 全量
python -u scripts/17_train_gnn_risk.py --train --device auto

# 断点续训
python -u scripts/17_train_gnn_risk.py --train --resume auto
```

默认读取 train/val pack，不得查找或加载 test pack。

### 10.4 `18_graph_diagnostics.py`

输出：

```text
graph_health_summary.json
val_predictions.parquet
node_attention_val.parquet
edge_attention_val.parquet
patient_top_subgraphs_val.parquet
graph_embedding_val.parquet
outputs/tables/gnn_val_metrics.csv
outputs/figures/graph_degree_support.png
outputs/figures/graph_event_count.png
```

第一版诊断不把 attention 直接命名为风险贡献；字段使用 `attention_weight`。若要输出贡献，必须另行使用受控的梯度/遮蔽归因方法。

---

## 11. 单元测试与无泄漏测试

### 11.1 事件测试

```text
test_node_key_includes_length
test_q_fit_uses_train_only
test_event_requires_valid_and_threshold
test_top1_per_variable_stage
test_event_tie_break_is_deterministic
test_empty_graph_patient_is_retained
```

### 11.2 边测试

```text
test_cooccur_when_abs_dt_le_2
test_cooccur_is_symmetric_relation
test_precedes_when_2_lt_dt_le_12
test_no_edge_beyond_12h
test_each_source_has_at_most_two_future_edges
test_cooccur_degree_at_most_four
test_canonical_edge_count_at_most_112
test_val_cannot_add_edge_vocabulary
test_edge_support_uses_unique_train_patients
```

### 11.3 模型测试

```text
test_forward_shapes_and_finite
test_backward_reaches_node_embedding
test_backward_reaches_in_out_relation_weights
test_backward_reaches_attention_and_head
test_inactive_nodes_receive_zero_pool_attention
test_empty_graph_forward_finite
test_edge_order_permutation_invariant
test_reversing_precedes_changes_output
test_reversing_cooccur_does_not_change_output
test_two_hop_chain_affects_two_layer_representation
```

### 11.4 泄漏测试

```text
test_subjects_disjoint_across_splits
test_label_not_present_in_node_or_edge_features
test_scalers_fit_train_only
test_edge_vocabulary_fit_train_only
test_default_pipeline_never_reads_test_pack
test_resume_rejects_hash_mismatch
```

### 11.5 Checkpoint 测试

```text
test_resume_matches_uninterrupted_next_step
test_ckpt_contains_all_hashes
test_best_selected_by_val_auprc
```

---

## 12. 分阶段验收门槛

### G0：合成图算子

- 所有 §11 模型核心单测通过；
- 前向/反向 finite；
- 三类关系参数均收到非零梯度；
- 交换 `PRECEDES` 方向会改变输出；交换 `CO_OCCUR` 存储顺序不改变输出。

### G1：事件导出

- q 仅由 train P50 产生；
- node dictionary 与 checkpoint 完全对齐；
- 每患者事件数 `0..28`；
- `is_valid=False` 节点绝不进入拓扑；
- train/val 无 SUBJECT_ID 重叠；
- 默认未读 test。

### G2：图构建

- 每患者 canonical edges `0..112`；
- 所有边满足时间与度数规则；
- val 未改变词表；
- 图包无 NaN/Inf；
- 报告空图率、孤立事件率、边丢弃率、关系类型比例；
- 同一输入重复构建 hash 一致。

### G3：训练冒烟

- 128 个 train/val 患者、2 epoch 可跑完；
- loss finite 且参数发生更新；
- checkpoint/resume 可接下一 epoch；
- 不生成 test 预测。

### G4：全量开发训练

- loss 稳定；
- `val_AUPRC >= prevalence × 1.3`；
- `val_AUROC >= 0.55`；
- 非空图与空图分层指标已报告；
- 最佳 checkpoint 和逐患者 val 预测完整。

这些是工程/信号门槛，不要求 GNN 必须大幅超过 HistGBM。

### G5：进入论文主分析前

仅保留一个必要的图结构对照：

```text
同样的 a_dense + z_sparse + z_static MLP
vs
GNN fusion
```

该对照回答“提升来自更多参数，还是图关系本身”。不在当前实现阶段增加大量模型搜索。

---

## 13. Agent 实现顺序与停点

### PR/阶段 A：数据结构与测试

```text
graph_constants.py
graph_events.py
graph_builder.py
test_graph_events.py
test_graph_builder.py
test_graph_no_leakage.py
```

只用合成数据；不得触碰 test。

### PR/阶段 B：GNN 纯 PyTorch 骨架

```text
graph_data.py
graph_model.py
test_graph_model.py
17_train_gnn_risk.py --dry-run
```

只做合成 forward/backward 与方向性测试。

### PR/阶段 C：上游接口

```text
15_export_shapelet_events.py
16_build_evolution_graphs.py
```

可使用 pilot checkpoint 做工程冒烟，但产物必须标记：

```text
smoke_only=true
shapelet_mode=pilot
q_original=zero
q_export=train_p50
paper_eligible=false
```

### PR/阶段 D：真实小样本训练

```text
17_train_gnn_risk.py --train --max-stays 128 --max-epochs 2
18_graph_diagnostics.py
```

必须由用户显式批准后运行。

### PR/阶段 E：正式多尺度

等待 `14_shapelet_final.py` 和 final checkpoint，通过 G0–G3 后再启动全量 GNN。不得用 pilot 图冒充论文主分析。

---

## 14. 禁止清单

- 禁止把所有 `a>0` 节点都连边；
- 禁止忽略共同发生关系；
- 禁止 all-pairs 连接 12 h 内所有事件；
- 禁止节点键省略 `L`；
- 禁止患者图之间直接连边；
- 禁止用全队列或 test 计算节点、q、边支持或 scaler；
- 禁止把 `HOSPITAL_MORTALITY`、死亡频率或 `R` 填进第一版 GNN 特征；
- 禁止把注意力权重直接称为因果贡献；
- 禁止只训练一张总体图；
- 禁止删除空图患者；
- 禁止因图不好看人工删边；
- 禁止默认加载 test；
- 禁止把 pilot L6 结果写成 final 多尺度结果；
- 禁止在 GNN 训练中解冻 Shapelet；
- 禁止未经 hash 检查从不匹配的 checkpoint 续训。

---

## 15. 最终解释口径

本模块允许的结论：

> 冻结 Shapelet 事件之间的共同发生和有向先后结构，可被患者级 GNN 压缩为与 landmark 后住院死亡相关的图表示；部分边和短路径在死亡患者中富集，并在冻结验证流程中复现。

不允许的结论：

> 某条图路径导致患者死亡；GNN 注意力证明了病理因果链；所有死亡患者都遵循同一固定路径。
