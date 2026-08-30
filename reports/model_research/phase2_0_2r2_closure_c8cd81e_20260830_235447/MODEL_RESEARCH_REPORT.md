# Phase 2.0.2r2 — Final Certification Closure Report
# A股模型研究认证真实性最终闭环与全量门禁实证报告

## 1. Git 溯源与血缘分层 (Git Provenance)

- **Run ID**: `phase2_0_2r2_closure_c8cd81e_20260830_235447`
- **Model Evidence Source Commit**: `e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48`
- **Certification Hotfix Source Commit**: `c8cd81e9ac2b78edaf0376638b2cb9e7aafd8b2c`
- **Previous Experiment Commit**: `fd01da829e9802804b7c5026b32d3e26a382c377`
- **Git Worktree Clean Before Formal Run**: `TRUE`
- **报告生成时点**: 2026-08-30 23:54:48

---

## 2. 生产数据集与特征架构 (Dataset)

- **Dataset Path**: `factor_matrix_300.parquet`
- **Dataset SHA256**: `9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42`
- **Candidate Model Families**: `4`
- **Dataset Rows**: `349,379`
- **Dataset Symbols**: `300` (全量真实 PIT 股票池)
- **Common Ranking Rows**: `221,019`
- **Common OOS Dates**: `744`
- **Feature Schema Hash**: `eb0fc8adc7538549d5399c475a38cff8f1e45a23b962fabab3d1aa67082f2eaa`
- **Model Config Hash**: `28dd57c9b671172d3abe936db2643ddb7c1523ef3625aea2248ae7ec2eef1802`
- **Reconstructed Source Model Config Hash**: `28dd57c9b671172d3abe936db2643ddb7c1523ef3625aea2248ae7ec2eef1802`
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
- **Mean Daily OOS RankIC**: **+0.0503**
- **NW20 RankICIR**: **+0.4044**
- **Q5-Q1 Annualized Arithmetic Forward Excess Spread**: **+7.17%**
- **MODEL_RESEARCH_STATUS**: `BASELINE_REMAINS_CHAMPION`

### 4.2 交易信号候选 (Trading Signal Candidate)
- **候选模型**: **LightGBM Ranker (LambdaRank)** (`lightgbm_ranker`)
- **Cost-adjusted Excess Return**: **+5.72%**
- **Strategy Sharpe**: **+0.36**
- **Strategy Max Drawdown**: **-14.35%**
- **Real Trading Fold Win Ratio**: **55.0%** (11/20 Folds 胜出)
- **TRADING_SIGNAL_STATUS**: `PROMISING_OOS_SIGNAL`

---

## 5. 真实多随机种子稳健性认证 (Seed Certification)

| 随机种子 Seed | random_state | feature_fraction_seed | bagging_seed | data_random_seed | 预测结果 SHA256 Hash | Mean Daily RankIC | NW20 RankICIR | AUC | 评估行数 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `42` | `42` | `42` | `42` | `42` | `0912c793c922a8ea` | 0.0503 | 0.4044 | 0.5319 | 221,019 |
| `2026` | `2026` | `2026` | `2026` | `2026` | `b52900d7c0fb782a` | 0.0455 | 0.3396 | 0.5290 | 221,019 |
| `3407` | `3407` | `3407` | `3407` | `3407` | `e20ce8cec27e2969` | 0.0459 | 0.3560 | 0.5305 | 221,019 |

- **SEED_ROBUSTNESS_STATUS**: `VERIFIED_STABLE` (RankIC 极差 max-min <= 0.01，表现出高度数值稳健性)

---

## 6. 真实 20 Fold 交易稳健性实证 (Trading Fold Stability)

| Fold 序号 | 测试起始 | 测试结束 | Ranker超额收益 | Baseline超额收益 | 超额收益差 (Delta) | Ranker夏普 | Baseline夏普 | Ranker胜出 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 01 | `2023-07-03` | `2023-08-25` | 1.24% | 4.26% | **-3.02%** | -1.96 | -0.67 | LOSS |
| Fold 02 | `2023-08-28` | `2023-10-30` | 2.91% | 1.57% | **+1.34%** | -0.73 | -1.60 | WIN |
| Fold 03 | `2023-10-31` | `2023-12-25` | 5.47% | -1.48% | **+6.95%** | -0.48 | -5.87 | WIN |
| Fold 04 | `2023-12-26` | `2024-02-28` | -5.31% | -5.27% | **-0.04%** | -0.65 | -0.60 | LOSS |
| Fold 05 | `2024-02-29` | `2024-04-26` | -2.72% | -0.43% | **-2.29%** | -0.59 | 0.87 | LOSS |
| Fold 06 | `2024-04-29` | `2024-06-27` | 3.37% | 4.75% | **-1.38%** | -1.16 | -0.15 | LOSS |
| Fold 07 | `2024-06-28` | `2024-08-22` | 8.65% | 5.24% | **+3.41%** | 1.55 | 0.53 | WIN |
| Fold 08 | `2024-08-23` | `2024-10-28` | 2.87% | -10.15% | **+13.02%** | 3.24 | 2.98 | WIN |
| Fold 09 | `2024-10-29` | `2024-12-23` | 3.10% | 1.52% | **+1.58%** | 1.44 | 0.78 | WIN |
| Fold 10 | `2024-12-24` | `2025-02-26` | -3.05% | -5.77% | **+2.72%** | -2.21 | -4.42 | WIN |
| Fold 11 | `2025-02-27` | `2025-04-24` | -5.53% | 5.28% | **-10.81%** | -2.86 | 0.34 | LOSS |
| Fold 12 | `2025-04-25` | `2025-06-25` | 8.01% | 0.59% | **+7.42%** | 2.54 | 4.30 | WIN |
| Fold 13 | `2025-06-26` | `2025-08-20` | 14.79% | -9.31% | **+24.10%** | 7.83 | -0.85 | WIN |
| Fold 14 | `2025-08-21` | `2025-10-23` | -8.71% | 5.35% | **-14.06%** | -0.78 | 3.36 | LOSS |
| Fold 15 | `2025-10-24` | `2025-12-18` | 6.54% | -2.83% | **+9.37%** | 2.24 | -4.36 | WIN |
| Fold 16 | `2025-12-19` | `2026-02-24` | 2.52% | 3.29% | **-0.77%** | 2.47 | 2.22 | LOSS |
| Fold 17 | `2026-02-25` | `2026-04-22` | -0.73% | -1.26% | **+0.53%** | 0.19 | -0.02 | WIN |
| Fold 18 | `2026-04-23` | `2026-06-23` | -3.03% | -10.54% | **+7.51%** | -0.19 | -3.88 | WIN |
| Fold 19 | `2026-06-24` | `2026-08-18` | 2.00% | 4.19% | **-2.19%** | -0.94 | -0.11 | LOSS |
| Fold 20 | `2026-08-19` | `2026-08-24` | -2.51% | 3.30% | **-5.81%** | -14.93 | 10.66 | LOSS |

> **注**: 末尾极短测试折 (如 Fold 20 仅数个交易日) 的单折夏普比率仅具描述性参考意义，不作为硬性门禁阻断项。
- **REAL_TRADING_FOLD_WIN_RATIO**: **55.0%** (11 / 20 Folds 胜出)

---

## 7. 交易信号尾部分析 (Trading Signal Top Tail Analysis)

| 尾部档位 | 标的占比 | 日均持股数 | 20D 前瞻超额收益均值 | 20D 前瞻超额收益中位数 | 正超额收益胜率 | 最差 10% 真实尾部均值 (`worst_decile_mean`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 5%** | `5%` | `14.1` | **2.07%** | 0.28% | 51.55% | -14.64% |
| **Top 10%** | `10%` | `29.1` | **1.53%** | 0.09% | 50.52% | -14.31% |
| **Top 20%** | `20%` | `59.1` | **1.14%** | -0.01% | 49.93% | -13.67% |

---

## 8. 配对块 Bootstrap 显著性检验 (Paired Block Bootstrap vs Baseline)

| 对比模型组合 (Candidate vs Baseline) | Mean RankIC 差值 | 95% 置信区间 (95% CI) | 提升概率 P(Diff > 0) | Bootstrap p-like 概率 | 统计显著提升 |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `double_ensemble_vs_lightgbm_clf_baseline` | `-0.01989` | `[-0.03911, 0.00123]` | `3.8%` | `0.0760` | `FALSE` |
| `lightgbm_ranker_vs_lightgbm_clf_baseline` | `-0.01238` | `[-0.03386, 0.01084]` | `14.9%` | `0.2980` | `FALSE` |
| `lightgbm_reg_baseline_vs_lightgbm_clf_baseline` | `-0.03081` | `[-0.06746, 0.00865]` | `8.2%` | `0.1640` | `FALSE` |

---

## 9. 纯证据驱动全量门禁判定矩阵 (Certification Gate Matrix)

| 审计项目 | 结果 | 证据推导依据与规则 |
| :--- | :---: | :--- |
| `SOURCE_PROVENANCE` | **PASS** | 源码冻结 Commit 先行提交，工作区干净无未暂存变更，Commit 对象在 Git 库真实存在 |
| `MODEL_CONFIG_HASH_VALIDITY` | **PASS** | 从真实 settings 生成 canonical 模型配置 SHA256，且与历史 source commit 严格匹配 |
| `ARTIFACT_REUSE_COMPATIBILITY` | **PASS** | dataset, feature schema, label horizon 及历史模型配置全要素无漂移 |
| `SEED_PROPAGATION` | **PASS** | 全部 3 个随机种子 (42, 2026, 3407) 的 4 重种子参数全部真实注入底层模型 |
| `SEED_ROBUSTNESS` | **PASS** | 3 独立种子已生成独立预测 Hash，RankIC 极差 <= 0.01 (`VERIFIED_STABLE`) |
| `COMMON_OOS_POOL` | **PASS** | 221,019 行通用池公平对比，分类二值 NaN 严格不排除排序池 |
| `NW20_CERTIFICATION` | **PASS** | Newey-West Lag 20 严格对齐 20D 标签重叠期，全模型值有效且有限 |
| `BOOTSTRAP_VALIDITY` | **PASS** | 候选 vs Baseline 配对检验完成，置信区间如实报告，概率介于 [0, 1] |
| `SELF_COMPARISON_GUARD` | **PASS** | 自我对比在代码层抛出 `ValueError` 严格阻断 |
| `REPORT_SEMANTICS` | **PASS** | Q5-Q1 准确命名为算术前瞻收益差，Ranker/Reg AUC 严格标为 N/A (无 NaN/inf) |
| `TRADING_FOLD_EVIDENCE_VALIDITY`| **PASS** | 20 Fold 交易指标独立回测计算，差值与胜负逻辑严格自洽 |
| `PYTEST` | **PASS** | 全量单元测试套件 100% 通过 (test_status.json exit_code == 0, Fail-Closed) |
| `LOCAL_PHASE_2_1_READY` | **TRUE** | 本地前置门禁全部就绪 |
| `FAST_CI` | **PENDING_POST_PUSH** | 等待 push 后外部 GitHub Actions 执行 |

---

## 10. 最终判定状态 (Final Status)

- **PHASE_2_0_2_STATUS**: `CLOSED`
- **LOCAL_PHASE_2_1_READY**: `TRUE`
- **FINAL_PHASE_2_1_READY**: `PENDING_CI` (等待 push 后 Fast CI 查询)
- **LIVE_TRADING_READY**: `FALSE` (严格禁止直接用于实盘交易)
- **NO_PHASE_2_0_2R3**: `TRUE` (本阶段认证闭环完成，无须进入 r3，直接推进 Phase 2.1)
