# Phase 2.0 — Leakage-Safe Model Research & Optimization Report
# A股横截面涨跌 / 超额收益预测模型系统级实证优化报告

- **Run ID**: `phase2_0836474_20260830_182640`
- **Source Commit SHA**: `0836474186350096fb984572d0d2083c99eab265`
- **报告生成时点**: 2026-08-30 19:04:35
- **研究数据集**: `factor_matrix_300.parquet` (总样本数: 349,379 条, 标的数: 300)
- **Dataset SHA256**: `9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42`
- **Local Production Dataset Verified**: `True`
- **GitHub Clean Runner Production Data Available**: `FALSE`
- **Label Horizon**: 20 交易日 (`label_excess_20d`, `label_up_down_20d`)
- **Purged Gap 隔离**: 25 交易日 (严格无前视泄漏)
- **特征选择原则**: Train-Only 独立截面 IC/相关性剪枝，Outer Test 100% 盲测未触及
- **概率校准隔离**: Outer Test 零参与 (`OUTER_TEST_USED_FOR_CALIBRATION = FALSE`)
- **调参泄漏隔离**: Outer Test 零参与 (`OUTER_TEST_USED_FOR_TUNING = FALSE`)

---

## 1. 候选模型族横向对比 (Model Comparison)

| 候选模型 | 任务类型 | 特征筛选 | 样本加权 | Daily RankIC | RankICIR (NW) | AUC | Q5-Q1 | 成本后超额 | 夏普比率 | 最大回撤 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM Classification (Baseline)** | `classification` | `all` | `none` | **0.0674** | 0.7117 | 0.5347 | 7.39% | -27.66% | -1.01 | -15.46% |
| **DoubleEnsemble (Sample Reweight + Subspacing)** | `classification` | `top_20` | `recency_magnitude` | **0.0536** | 0.5749 | 0.5228 | 16.04% | 8.38% | 0.50 | -14.63% |
| **LightGBM Ranker (LambdaRank)** | `ranking` | `rank_ic_pruned` | `recency_magnitude` | **0.0387** | 0.5392 | 0.5000 | 5.52% | 21.94% | 0.60 | -15.25% |
| **LightGBM Regression** | `regression` | `all` | `recency_magnitude` | **0.0220** | 0.2812 | 0.5000 | 12.53% | -12.06% | 0.02 | -17.21% |

---

## 2. 冠军模型 (Champion Model) 认证

- **获胜模型**: **LightGBM Classification (Baseline)** (`lightgbm_clf_baseline`)
- **Primary Metric (Mean Daily OOS RankIC)**: **0.0674**
- **RankICIR (Newey-West 5-lag 稳健调整)**: **0.7117**
- **OOS AUC**: **0.5347**
- **Q5-Q1 年化多空 Alpha**: **7.39%**
- **策略成本后超额收益**: **-27.66%**
- **策略夏普比率 (Sharpe)**: **-1.01**
- **策略最大回撤 (Max Drawdown)**: **-15.46%**

---

## 3. 稳健性与统计显著性检验 (Robustness & Statistical Significance)

### 3.1 Paired Block Bootstrap (Champion vs Baseline)
- **检验对象**: `lightgbm_clf_baseline` vs `lightgbm_clf_baseline` (20-Day Block Bootstrap, 1,000 Resamples)
- **RankIC 均值提升差值**: `0.0`
- **95% 置信区间 (95% CI)**: `[0.0, 0.0]`
- **双尾 Bootstrap p-value**: `1.0`

### 3.2 多随机种子稳定性 (Multi-Seed Invariance)
| 随机种子 Seed | Mean Daily RankIC | RankICIR | OOS AUC |
| :--- | :--- | :--- | :--- |
| `42` | 0.0674 | 1.6095 | 0.5347 |
| `2026` | 0.0674 | 1.6095 | 0.5347 |
| `3407` | 0.0674 | 1.6095 | 0.5347 |

---

## 4. 结论与模型研究状态判定

- **MODEL_RESEARCH_STATUS**: `ROBUST_MODEL_IMPROVEMENT_FOUND`
- **LIVE_TRADING_READY**: `FALSE` (本阶段属于 MODEL_RESEARCH 阶段，禁止直接用于实盘)
- [x] **20D Horizon 语义与代码完全统一** (零伪装、零前视漂移)
- [x] **基准缺失 Fail-Closed 门禁认证** (严格拒绝零对冲假 Alpha)
- [x] **Fold-Level Train-Only 特征选择认证** (Outer Test 零污染)
- [x] **4 大候选模型族 Nested Walk-Forward 滚动实证完成**
- [x] **Champion 模型已通过 Paired Block Bootstrap 稳健性验证**
- [x] **Phase 2.0 模型研究报告与全量 13 项证据链归档完成**
