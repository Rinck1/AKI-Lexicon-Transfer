# GNN / 表格臂已有消融汇总

> 只读已有 `train_report.json` 的 best val AUPRC。不重训，不开新 test。

| 臂 | 目录 | best epoch | val AUPRC | val AUROC | val Brier |
|---|---|---:|---:|---:|---:|
| A 激活袋+静态 | `gnn_ablate_A_act_static` | 10 | 0.4101 | 0.7871 | 0.1827 |
| B 激活+稀疏检验+静态 | `gnn_ablate_B_act_sparse_static` | 4 | 0.5822 | 0.8560 | 0.1068 |
| 旧池化 AG | `gnn_ablate_AG_act_static_graph` | 3 | 0.3676 | 0.7736 | 0.1708 |
| 旧池化 C | `gnn_full_final_lr2e4` | 5 | 0.5776 | 0.8525 | 0.1002 |
| 可分解 G 图+静态 | `gnn_additive_G_graph_static` | 4 | 0.3131 | 0.7281 | 0.2052 |
| 可分解 AG | `gnn_additive_AG_act_static` | 5 | 0.3560 | 0.7609 | 0.2191 |
| 84节点 可分解 G | `gnn_additive_G_motif84` | 5 | 0.3090 | 0.7515 | 0.1956 |
| 连续门控 | `gnn_additive_gate_G_motif84` | 7 | 0.3220 | 0.7550 | 0.2031 |
| Hard-Concrete L0 | `gnn_additive_l0_G_motif84` | 34 | 0.3464 | 0.7689 | 0.1782 |
| STE top-k | `gnn_additive_topk_G_motif84` | 4 | 0.3151 | 0.7547 | 0.2420 |
| 三层多尺度 G | `gnn_additive_G3_multiscale_seed42` | 4 | 0.3022 | 0.7275 | 0.2273 |
| 残差门（冻结 B） | `gnn_residual_gate_from_B` | 5 | 0.5829 | 0.8550 | 0.1391 |
