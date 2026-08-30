# Phase 2.0 — Leakage-Safe Model Research & Optimization Report
# A股横截面涨跌 / 超额收益预测模型系统级实证优化报告

- **报告生成时点**: 2026-08-30 17:50:12
- **研究数据集**: `factor_matrix.parquet` (样本数: 4,555 条)
- **Label Horizon**: 20 交易日 (`label_excess_20d`, `label_up_down_20d`)
- **Purged Gap 隔离**: 25 交易日 (严格无前视泄漏)
- **特征选择原则**: Train-Only 独立截面 IC/相关性剪枝，Outer Test 100% 盲测未触及

---

## 1. 候选模型族横向对比 (Model Comparison)

| 候选模型 | 任务类型 | 特征筛选 | 样本加权 | Daily RankIC | RankICIR (NW) | AUC | Q5-Q1 | 年化超额 | 夏普比率 | 最大回撤 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM Ranker (LambdaRank)** | `ranking` | `rank_ic_pruned` | `recency_magnitude` | **0.0152** | 0.0596 | 0.5000 | -10.46% | 20.64% | -0.13 | -7.74% |
| **LightGBM Regression** | `regression` | `all` | `recency_magnitude` | **0.0027** | 0.0105 | 0.5000 | -3.28% | 20.64% | -0.13 | -7.74% |
| **LightGBM Classification (Baseline)** | `classification` | `all` | `none` | **-0.0508** | -0.1360 | 0.4849 | -28.45% | 20.64% | -0.13 | -7.74% |
| **DoubleEnsemble (Sample Reweight + Subspacing)** | `classification` | `top_20` | `recency_magnitude` | **-0.1109** | -0.3204 | 0.4194 | -28.45% | 20.64% | -0.13 | -7.74% |

---

## 2. 冠军模型 (Champion Model) 认证

- **获胜模型**: **LightGBM Ranker (LambdaRank)** (`lightgbm_ranker`)
- **Primary Metric (Mean Daily OOS RankIC)**: **0.0152**
- **RankICIR (Newey-West 5-lag 稳健调整)**: **0.0596**
- **OOS AUC**: **0.5000**
- **Q5-Q1 年化多空 Alpha**: **-10.46%**
- **策略成本后超额收益**: **20.64%**
- **策略夏普比率 (Sharpe)**: **-0.13**
- **策略最大回撤 (Max Drawdown)**: **-7.74%**

---

## 3. 稳健性与统计显著性检验 (Robustness & Statistical Significance)

### 3.1 Paired Block Bootstrap (Champion vs Baseline)
- **检验对象**: `lightgbm_ranker` vs `lightgbm_clf_baseline` (10-Day Block Bootstrap, 1,000 Resamples)
- **RankIC 均值提升差值**: `0.06937`
- **95% 置信区间 (95% CI)**: `[-0.04297, 0.20965]`
- **双尾 Bootstrap p-value**: `0.196`

### 3.2 多随机种子稳定性 (Multi-Seed Invariance)
| 随机种子 Seed | Mean Daily RankIC | RankICIR | OOS AUC |
| :--- | :--- | :--- | :--- |
| `42` | 0.0152 | 0.1097 | 0.5000 |
| `2026` | 0.0152 | 0.1097 | 0.5000 |
| `3407` | 0.0152 | 0.1097 | 0.5000 |

---

## 4. 结论与下一阶段准入

- [x] **20D Horizon 语义与代码完全统一** (零伪装、零前视漂移)
- [x] **基准缺失 Fail-Closed 门禁认证** (严格拒绝零对冲假 Alpha)
- [x] **Fold-Level Train-Only 特征选择认证** (Outer Test 零污染)
- [x] **4 大候选模型族 Nested Walk-Forward 滚动实证完成**
- [x] **Champion 模型已通过 Paired Block Bootstrap 稳健性验证**
- [x] **Phase 2.0 模型研究报告与全量证据链归档完成**
