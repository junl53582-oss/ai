# Phase 2.1-A — Execution-Aligned Label A/B Study Report
# 实盘执行对齐标签严格受控 A/B 实验研究报告

> **研究结论**: **`MIXED_EVIDENCE`**  
> **实盘许可声明**: `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 1. 实验控制变量规范 (Controlled Variables)

- **基准模型族**: `LightGBM Classification` (二分类概率预测)
- **特征集与顺序**: 严格相同 (79 因子, Feature Hash = `82dfd3e9643ae1352829e736b9c8b89d1d648b98d16ef59153f261bf7a453460`)
- **模型超参数**: 严格逐字段相同 (Config Hash = `a000f774805e21a19523fa8ea40edfd0cd6ca95a3dc14f4524a000c66278f81c`)
- **随机种子**: `42` (严格传播至 feature_fraction_seed, bagging_seed, data_random_seed)
- **时序划分**: 严格相同 Walk-Forward 滚动折划分 (Purge Gap = 25 天 >= Label Horizon 20 天)
- **训练准入池**: 严格相同的共同准入交集 (Common Train Pool Hash = `bfff9a2d0a9b52a0d4924ea36d643923d1e42ab6bd48ed60017d020d22c42bcb`)
- **唯一自变量 (Primary Change)**: **训练目标标签定义** (Legacy `ab_label_legacy` vs Execution-Aligned `ab_label_execution`)

---

## 2. 统一公平评估目标 (Common Execution OOS Target)

两个 Arm 均在完全相同的实盘执行对齐前瞻净超额收益目标 `label_net_alpha_20d` 上进行 OOS 评价：

- **共同 OOS 样本行数**: 220,913
- **共同 OOS 交易日数**: 743
- **共同 OOS 股票池数**: 300
- **共同 OOS 评价池 SHA256**: `a464b29fd12a50891ef68777791ebcb0c7c4f9fc96b59137174387100ca5fd1c`

---

## 3. 核心对比结果 (Controlled A/B Evaluation Results)

| 评价维度 / 指标 | Arm A (Legacy Label) | Arm B (Execution-Aligned Label) | 差异 (Delta B - A) |
| :--- | :---: | :---: | :---: |
| **Mean Daily OOS RankIC** | **0.043764** | **0.044698** | **+0.000935** |
| **NW20 RankICIR (年化)** | **0.344792** | **0.355503** | **+0.010712** |
| **RankIC > 0 交易日占比** | 60.43% | 60.16% | -0.27% |
| **Q5-Q1 分组收益差** | 2.340000 | 4.000000 | +1.660000 |
| **Top 10% 真实执行净超额** | 0.002870 | 0.004715 | +0.001845 |
| **分组单调性得分** | 0.8000 | 0.7000 | -0.1000 |

---

## 4. 统计检验与折稳定性 (Statistical Significance & Fold Stability)

- **20-Day Paired Block Bootstrap (2,000 Resamples)**:
  - **Mean RankIC Delta**: **+0.000935**
  - **95% 置信区间 (95% CI)**: `[-0.008533, +0.012117]`
  - **提升概率 P(Delta > 0)**: **52.10%**
  - **统计显著提升 (CI Lower > 0)**: **`False`**
- **Fold-Level 胜率实证**:
  - **滚动折总数**: 20
  - **Execution Arm 胜出折数**: 8
  - **Fold 胜率 (Fold Win Ratio)**: **40.00%**

---

## 5. 生产模型物理隔离审计 (Production Model Isolation)

- **Production Model Path**: `C:\Users\lin\Documents\股票预测\models\latest_lightgbm.pkl`
- **Exists Before Experiment**: `False`
- **SHA256 Before Experiment**: `None`
- **Exists After Experiment**: `False`
- **SHA256 After Experiment**: `None`
- **SHA256 Unchanged**: **`True`**
- **Production Models Dir Mutated**: **`False`**
- **Legacy Experiment Model Dir**: `C:\Users\lin\Documents\股票预测\reports\phase2_1_a\phase2_1_a_5924351_20260831_032109\models\legacy`
- **Execution Experiment Model Dir**: `C:\Users\lin\Documents\股票预测\reports\phase2_1_a\phase2_1_a_5924351_20260831_032109\models\execution`

---

## 6. 科学判定与结论说明 (Scientific Finding & Next Step)

- **判定状态**: **`MIXED_EVIDENCE`**
- **结论阐述**: 
  实证表明：Execution-Aligned Labels 展现出混合证据，部分指标提升但统计置信度未达完全显著要求。
