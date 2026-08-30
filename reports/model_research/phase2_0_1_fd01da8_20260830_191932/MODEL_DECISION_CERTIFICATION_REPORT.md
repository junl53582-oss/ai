# Phase 2.0.1 — Model Decision & Statistical Certification Report
# A股模型公平比较、冠军判定、统计认证与报告一致性实证报告

- **Run ID**: `phase2_0_1_fd01da8_20260830_191932`
- **Source Commit SHA**: `fd01da829e9802804b7c5026b32d3e26a382c377`
- **Experiment Commit SHA (Corrected)**: `fd01da829e9802804b7c5026b32d3e26a382c377`
- **报告生成时点**: 2026-08-30 19:52:39
- **研究数据集**: `factor_matrix_300.parquet` (总样本数: 349,379 条, 标的数: 300)
- **Dataset SHA256**: `9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42`
- **Local Production Dataset Verified**: `True`
- **Label Horizon**: 20 交易日 (`label_excess_20d`, `label_up_down_20d`)
- **Certification NW Lag**: `20` 交易日 (`rank_icir_nw_lag20`，匹配 20D Forward Label 真实重叠期)
- **Common OOS Evaluation Pool**: `100% ENFORCED` (所有模型在包含连续超额收益的统一池上评估 RankIC)

---

## 1. 候选模型公平横向对比 (Common Ranking Pool Comparison)

| 候选模型 | 任务类型 | 特征筛选 | 样本加权 | Common OOS Rows | Mean Daily RankIC | NW5 RankICIR | NW20 RankICIR (Cert) | AUC | Q5-Q1 | 成本后超额 | 夏普比率 (Sharpe) | 最大回撤 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM Classification (Baseline)** | `classification` | `all` | `none` | 221,019 | **0.0536** | 0.6723 | **0.4189** | 0.5347 | 3.71% | -27.66% | -1.01 | -15.46% |
| **DoubleEnsemble (Sample Reweight + Subspacing)** | `classification` | `top_20` | `recency_magnitude` | 221,019 | **0.0409** | 0.5289 | **0.3451** | 0.5228 | 9.44% | 8.38% | 0.50 | -14.63% |
| **LightGBM Ranker (LambdaRank)** | `ranking` | `rank_ic_pruned` | `recency_magnitude` | 221,019 | **0.0387** | 0.5392 | **0.3487** | 0.5000 | 5.52% | 21.94% | 0.60 | -15.25% |
| **LightGBM Regression** | `regression` | `all` | `recency_magnitude` | 221,019 | **0.0220** | 0.2812 | **0.1922** | 0.5000 | 12.53% | -12.06% | 0.02 | -17.21% |

---

## 2. 预测冠军与交易信号冠军分离判定 (Champion Decisions)

### 2.1 预测质量冠军 (Prediction Champion)
- **获胜模型**: **LightGBM Classification (Baseline)** (`lightgbm_clf_baseline`)
- **Primary Metric (Mean Daily OOS RankIC)**: **0.0536**
- **RankICIR (Newey-West 20-lag 认证)**: **0.4189**
- **Q5-Q1 年化多空 Alpha**: **3.71%**
- **判定状态**: `MODEL_RESEARCH_STATUS = BASELINE_REMAINS_CHAMPION`

### 2.2 交易信号冠军 (Trading Signal Champion)
- **获胜模型**: **LightGBM Ranker (LambdaRank)** (`lightgbm_ranker`)
- **策略成本后超额收益**: **21.94%**
- **策略夏普比率 (Sharpe)**: **0.60**
- **策略最大回撤 (Max Drawdown)**: **-15.25%**
- **判定状态**: `TRADING_SIGNAL_STATUS = PROMISING_OOS_SIGNAL`

---

## 3. 配对块 Bootstrap 显著性检验 (Paired Block Bootstrap vs Baseline)

> 采用 20 交易日块采样 (20-Day Block Bootstrap, 1,000 次重抽样, 固定随机种子 42)：

| 对比模型组合 (Candidate vs Baseline) | Mean RankIC 差值 | 95% 置信区间 (95% CI) | 提升概率 P(Diff > 0) | Bootstrap p-value | 统计显著提升 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `double_ensemble_vs_baseline` | `-0.01274` | `[-0.03590, 0.01349]` | `17.7%` | `0.3540` | `FALSE` |
| `lightgbm_ranker_vs_baseline` | `-0.01492` | `[-0.03377, 0.00636]` | `7.7%` | `0.1540` | `FALSE` |
| `lightgbm_reg_baseline_vs_baseline` | `-0.03159` | `[-0.06974, 0.00732]` | `6.8%` | `0.1360` | `FALSE` |

---

## 4. 多随机种子稳定性认证 (Multi-Seed Invariance)

| 随机种子 Seed | 预测结果 Hash | Mean Daily RankIC | NW20 RankICIR | OOS AUC | 样本外评估行数 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `42` | `62568a25bba0c91d` | 0.0536 | 0.4189 | 0.5347 | 227,466 |
| `2026` | `62568a25bba0c91d` | 0.0536 | 0.4189 | 0.5347 | 227,466 |
| `3407` | `62568a25bba0c91d` | 0.0536 | 0.4189 | 0.5347 | 227,466 |

---

## 5. 决策与下一阶段准入

- **MODEL_RESEARCH_STATUS**: `BASELINE_REMAINS_CHAMPION`
- **TRADING_SIGNAL_STATUS**: `PROMISING_OOS_SIGNAL`
- **PHASE_2_1_READY**: `TRUE` (可进入 Phase 2.1 投资组合权重与执行优化)
- **LIVE_TRADING_READY**: `FALSE` (严格禁止直接用于实盘)
- [x] **Experiment Commit SHA 正确修正并归档**
- [x] **COMMON_RANKING_POOL 统一评估池严格落实**
- [x] **Newey-West Lag 20 稳健自相关校正完成**
- [x] **Candidate vs Baseline 配对 Bootstrap 检验完成**
- [x] **Prediction Champion 与 Trading Champion 分离认证**
- [x] **Fast CI 历史状态已更新为 VERIFIED (SUCCESS)**
