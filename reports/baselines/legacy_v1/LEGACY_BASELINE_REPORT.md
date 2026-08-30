# LEGACY_BASELINE_V1 — Frozen Research Baseline Report
# A股量化预测系统 Legacy V1 历史真值冻结与基准报告

> **基准定位**: 本报告将 Phase 2.0.2 经过 Walk-Forward OOS 实证检验的模型证据冻结为不可篡改的 `LEGACY_BASELINE_V1` 科学参照系。后续 Phase 2.1+ 所有实验均以此为基准进行单变量因果对比。

---

## 1. 历史血缘与配置哈希 (Provenance & Config Hashes)

- **BASELINE_ID**: `LEGACY_BASELINE_V1`
- **STATUS**: `FROZEN`
- **MODEL_EVIDENCE_SOURCE_COMMIT**: `e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48`
- **CERTIFICATION_LOGIC_SOURCE_COMMIT**: `c8ce24a8bd275153934169a8d1e989cf78d1c5e5`
- **CERTIFIED_ARTIFACT_COMMIT**: `8bfa0e49395290266e923b2c15053f594a7e7dd2`
- **DATASET_SHA256**: `9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42`
- **FEATURE_SCHEMA_HASH**: `eb0fc8adc7538549d5399c475a38cff8f1e45a23b962fabab3d1aa67082f2eaa` (79 因子)
- **ARTIFACT_HASH_MANIFEST_SHA256**: `c4537008588e6cbbe606c0dab4bc45ed4175f67526137c760bfe49b44c9607a3`
- **LEGACY_EFFECTIVE_MODEL_CONFIG_HASH**: `7d8eaad86ec95f39a5b68a12c0a7f9691b7bba98ac576445ed044108319c972e`

---

## 2. 旧 Ranker 历史模型语义订正 (Legacy Ranker Semantic Correction)

根据源码级 AST 逆向解析，历史模型调用路径特征如下：
- **Historical Artifact Model ID**: `lightgbm_ranker`
- **Historical Reported Name**: `LightGBM Ranker (LambdaRank)`
- **Corrected Legacy ID**: `legacy_ordinal_ranker`
- **Effective Estimator Class**: `LGBMRanker`
- **Effective Objective**: **`regression`** (由于历史代码分支逻辑，加载了 `LGBM_PARAMS` 且覆盖了 lambdarank 默认值)
- **Effective Metric**: `rmse`
- **Relevance Targets**: `daily_ordinal_0_to_4` (按日截面百分位离散化为 0~4 整数等级)
- **TRUE_LAMBDARANK_CERTIFIED**: **`FALSE`** (真正的 LambdaRank 损失与排序实验将在 Phase 2.1-B 开展)

---

## 3. 冻结核心基准指标 (Frozen Baseline Metrics)

### 3.1 预测质量基准 (Prediction Baseline) — `lightgbm_clf_baseline`
- **Mean Daily OOS RankIC**: **+0.0503**
- **NW20 RankICIR**: **+0.4044**
- **AUC**: **0.5319**
- **Q5-Q1 算术超额差**: **+7.17%**
- **Common Ranking Rows**: `221,019`
- **Common OOS Dates**: `744`
- **3-Seed 稳健性**: `VERIFIED_STABLE`

### 3.2 交易候选基准 (Trading Candidate Baseline) — `legacy_ordinal_ranker`
- **Cost-adjusted Excess Return**: **+5.72%**
- **Sharpe Ratio**: **+0.36**
- **Max Drawdown**: **-14.35%**
- **Real 20-Fold Win Ratio**: **55.0%**
- **Annualized Turnover Avg**: **9.11x**
- **Total Filled Trades**: `579` 笔
- **Total Costs**: `86,006.69` 元
- **Seed Robustness**: **`NOT_CERTIFIED`** (待 Phase 2.1 实测)

---

## 4. 历史标签时序与撮合脱节说明 (Legacy Label Alignment Gap)

- **Legacy Signal Time**: `T_CLOSE`
- **Legacy Label Window**: `T Close -> T+20 Close`
- **Backtest Execution Entry**: `T+1 Open`
- **Alignment Gap**: 历史标签计算的收益始于 T 日收盘价，而执行撮合始于 T+1 日开盘价。该时序脱节将作为 **Phase 2.1-A (Execution-Aligned Labels)** 的唯一核心自变量予以对齐。
