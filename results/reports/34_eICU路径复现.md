# eICU 冻结路径复现

- freeze `966ca9be40f9c7abdca6f95416434bcd9abda4d82e50523f4a3e70d5471458b7`
- 系数只在 MIMIC-III train 拟合；eICU 零训练
- 全队列 N=21657 死亡=0.153 静态 AUPRC=0.190 → 静态+路径 0.188
- 同号 9/15，支持≥20 15/15
- 尿量完整子集（≥12h）N=12539

| 路径 | 支持 | 同号 | OR |
|---|---:|:---:|---:|
| lactate|B1|升高 → urine_output_ml|低位缓升型|B3 | 410 | 是 | 1.34 |
| mbp|缓降持平型|B3 → lactate|B4|升高 | 78 | 是 | 1.70 |
| wbc|B1|升高 → gcs_total|B4|下降 | 333 | 否 | 0.99 |
| lactate|B3|升高 → sbp|低位下滑型|B4 | 107 | 是 | 1.23 |
| gcs_total|B3|下降 → creatinine|B4|升高 | 164 | 否 | 0.97 |
| creatinine|B3|升高 → ph|B4|下降 | 220 | 是 | 1.14 |
| dbp|低位回升型|B1 → lactate|B4|升高 | 85 | 否 | 0.94 |
| urine_output_ml|低位波动型|B1 → creatinine|B4|升高 | 184 | 否 | 0.54 |
| potassium|B3|升高 → lactate|B4|下降 | 149 | 是 | 1.14 |
| respiratory_rate|末期骤降型|B1 → urine_output_ml|低位波动型|B4 | 88 | 否 | 0.91 |
| urine_output_ml|低位波动型|B1 → urine_output_ml|进行性下降型|B2 | 323 | 否 | 0.70 |
| dbp|持续下降型|B1 → gcs_total|B4|下降 | 120 | 是 | 1.27 |
| hemoglobin|B3|升高 → creatinine|B4|升高 | 192 | 是 | 1.17 |
| ph|B1|升高 → gcs_total|B2|升高 | 69 | 是 | 1.33 |
| ph|B3|升高 → creatinine|B4|升高 | 225 | 是 | 1.41 |
