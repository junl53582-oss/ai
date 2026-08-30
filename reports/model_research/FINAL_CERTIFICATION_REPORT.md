# Phase 2.0.2 — Final Provenance & Seed Certification Report
# A股模型研究最终血缘冻结、真实随机种子认证与交易信号稳健性实证报告

## 1. Git 溯源与血缘一致性 (Git Provenance)

- **Run ID**: `phase2_0_2_e6da4a2_20260830_215520`
- **Source Commit SHA**: `e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48`
- **Previous Experiment Commit**: `fd01da829e9802804b7c5026b32d3e26a382c377`
- **Previous Hotfix Commit**: `d32269bdbde8f883c2fe4509ee55a935d9b4d710`
- **Git Worktree Clean Before Formal Run**: `TRUE`
- **报告生成时点**: 2026-08-30 22:36:10

---

## 2. 生产数据集与前置特征门禁 (Dataset)

- **Dataset Path**: `factor_matrix_300.parquet`
- **Dataset SHA256**: `9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42`
- **Total Rows**: `349,379` 条
- **Total Symbols**: `300` (全量真实 PIT 股票池)
- **Date Range**: `2021-09-29` 至 `2026-08-24`
- **Feature Schema Hash**: `eb0fc8adc7538549d5399c475a38cff8f1e45a23b962fabab3d1aa67082f2eaa`
- **Label Horizon**: `20` 交易日 (`label_excess_20d`, `label_up_down_20d`)
- **Certification NW Lag**: `20` 交易日 (`rank_icir_nw_lag20`)

---

## 3. 候选模型公平横向对比 (Common Ranking Pool Comparison)

| 候选模型 | 任务类型 | 特征筛选 | 样本加权 | OOS预测行数 | 通用排序行数 | 日期数 | Mean Daily RankIC | NW5 RankICIR | NW20 RankICIR (Cert) | AUC | Q5-Q1 算术超额差 | 成本后超额收益 | 夏普比率 (Sharpe) | 最大回撤 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM Classification (Baseline)** | `classification` | `all` | `none` | 227,466 | 221,019 | 744 | **0.0503** | 0.6354 | **0.4044** | 0.5319 | 7.17% | -29.93% | -0.74 | -15.26% |
| **DoubleEnsemble (Sample Reweight + Subspacing)** | `classification` | `top_20` | `recency_magnitude` | 227,466 | 221,019 | 744 | **0.0304** | 0.3798 | **0.2468** | 0.5183 | 5.81% | -14.94% | -0.10 | -14.25% |
| **LightGBM Ranker (LambdaRank)** | `ranking` | `rank_ic_pruned` | `recency_magnitude` | 227,466 | 221,019 | 744 | **0.0379** | 0.5310 | **0.3472** | N/A | 9.49% | 5.72% | 0.36 | -14.35% |
| **LightGBM Regression** | `regression` | `all` | `recency_magnitude` | 227,466 | 221,019 | 744 | **0.0194** | 0.2611 | **0.1772** | N/A | 4.33% | -6.44% | 0.17 | -21.83% |

---

## 4. 预测质量冠军与交易信号候选判定 (Champion & Candidate Decisions)

### 4.1 预测质量冠军 (Prediction Champion)
- **获胜模型**: **LightGBM Classification (Baseline)** (`lightgbm_clf_baseline`)
- **Common Ranking Rows**: `221,019`
- **Common OOS Dates**: `744`
- **Mean Daily OOS RankIC**: **0.0503**
- **NW20 RankICIR**: **0.4044**
- **Q5-Q1 Annualized Arithmetic Forward Excess Spread**: **7.17%**
- **MODEL_RESEARCH_STATUS**: `BASELINE_REMAINS_CHAMPION`

### 4.2 交易信号候选 (Trading Signal Candidate)
- **候选模型**: **LightGBM Ranker (LambdaRank)** (`lightgbm_ranker`)
- **Cost-adjusted Excess Return**: **5.72%**
- **Strategy Sharpe**: **0.36**
- **Strategy Max Drawdown**: **-14.35%**
- **TRADING_SIGNAL_STATUS**: `PROMISING_OOS_SIGNAL`

---

## 5. 真实多随机种子稳健性认证 (Seed Certification)

> 严格验证 `random_state`, `feature_fraction_seed`, `bagging_seed`, `data_random_seed` 4 重参数真实注入与序列化预测 Hash：

| 随机种子 Seed | random_state | feature_fraction_seed | bagging_seed | data_random_seed | 预测结果 SHA256 Hash | Mean Daily RankIC | NW20 RankICIR | AUC | 评估行数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `42` | `42` | `42` | `42` | `42` | `0912c793c922a8ea` | 0.0503 | 0.4044 | 0.5319 | 221,019 |
| `2026` | `2026` | `2026` | `2026` | `2026` | `b52900d7c0fb782a` | 0.0455 | 0.3396 | 0.5290 | 221,019 |
| `3407` | `3407` | `3407` | `3407` | `3407` | `e20ce8cec27e2969` | 0.0459 | 0.3560 | 0.5305 | 221,019 |

- **SEED_ROBUSTNESS_STATUS**: `VERIFIED_STABLE` (已确认 4 重种子参数 100% 注入模型底层，在 3 独立随机种子下产出不同且确定性的预测序列，RankIC 极差 max-min = 0.0048 <= 0.01，表现出高度数值稳健性)

---

## 6. 交易信号尾部分析 (Trading Signal Top Tail Analysis)

> 基于 `LightGBM Ranker` 在通用排序池上的前瞻截面收益评估：

| 尾部档位 | 标的占比 | 日均持股数 | 20D 前瞻超额收益均值 | 20D 前瞻超额收益中位数 | 正超额收益胜率 | 最差 10% 尾部均值 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 5%** | `5%` | `14.1` | **2.07%** | 0.28% | 51.55% | -9.33% |
| **Top 10%** | `10%` | `29.1` | **1.53%** | 0.09% | 50.52% | -9.47% |
| **Top 20%** | `20%` | `59.1` | **1.14%** | -0.01% | 49.93% | -9.24% |

---

## 7. 配对块 Bootstrap 显著性检验 (Paired Block Bootstrap vs Baseline)

> 采用 20 交易日块采样 (20-Day Block Bootstrap, 1,000 次重抽样, 固定随机种子 42, 共同样本日期 744 天)：

| 对比模型组合 (Candidate vs Baseline) | Mean RankIC 差值 | 95% 置信区间 (95% CI) | 提升概率 P(Diff > 0) | Bootstrap p-like 概率 | 统计显著提升 |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `double_ensemble_vs_lightgbm_clf_baseline` | `-0.01989` | `[-0.03911, 0.00123]` | `3.8%` | `0.0760` | `FALSE` |
| `lightgbm_ranker_vs_lightgbm_clf_baseline` | `-0.01238` | `[-0.03386, 0.01084]` | `14.9%` | `0.2980` | `FALSE` |
| `lightgbm_reg_baseline_vs_lightgbm_clf_baseline` | `-0.03081` | `[-0.06746, 0.00865]` | `8.2%` | `0.1640` | `FALSE` |

- **结论**: 未发现候选模型相对 Baseline 存在统计稳健的 RankIC 提升，因此按照预设模型选择规则保留 Baseline。

---

## 8. 验收矩阵与阶段准入 (Acceptance Matrix)

| 审计项目 | 结果 | 审计依据与说明 |
| :--- | :---: | :--- |
| `SOURCE_PROVENANCE` | **PASS** | 源码冻结 Commit 先行提交，无未提交污染，血缘链闭环 |
| `SEED_PROPAGATION` | **PASS** | `random_state`、`feature_fraction_seed`、`bagging_seed`、`data_random_seed` 4 重参数真实注入 |
| `SEED_ROBUSTNESS` | **PASS** | 3 独立种子完成真实重训并生成序列化 Hash 存证 |
| `COMMON_OOS_POOL` | **PASS** | 221,019 行通用池公平对比，分类二值 NaN 严格不排除排序池 |
| `NW20_CERTIFICATION` | **PASS** | Newey-West Lag 20 严格对齐 20D 标签重叠期 |
| `BOOTSTRAP_VALIDITY` | **PASS** | 候选 vs Baseline 配对检验完成，置信区间如实报告 |
| `SELF_COMPARISON_GUARD` | **PASS** | 自我对比在代码层抛出 `ValueError` 严格阻断 |
| `REPORT_SEMANTICS` | **PASS** | Q5-Q1 准确命名为算术前瞻收益差，Ranker/Reg AUC 标记 N/A |
| `TRADING_SIGNAL_ROBUSTNESS`| **PASS** | Ranker Top 5%/10%/20% 尾部超额与胜率完成统计分析 |
| `PYTEST` | **PASS** | 全量单元测试套件 100% 通过 |
| `FAST_CI` | **PASS** | 门禁全部就绪 |
| `HEAD_ORIGIN_SYNC` | **TRUE** | 本地与远程完全同步 |

---

## 9. 最终判定状态

- **PHASE_2_0_2_STATUS**: `CLOSED`
- **PHASE_2_1_READY**: `TRUE` (已具备进入 Phase 2.1 投资组合权重、Top-K 分配与执行优化的全部先决条件)
- **LIVE_TRADING_READY**: `FALSE` (严格禁止直接用于实盘交易)
