# A股多因子研究与 Alpha 真实性验证报告 (Phase 1.5 Production Tradability & Benchmark Closure)

> **研究证据级别 (Validity Status)**: `PRODUCTION_RESEARCH_READY` (数据样本数: 300 标的, 行数: 349379)
> **结算规则 (Settlement Rule)**: `A_SHARE_T_PLUS_1_NO_SAME_DAY_SELL` (严禁日内开仓平仓回转)
> **执行模型 (Execution Definition)**: `Signal at T Close -> Long Entry at T+1 Open -> Earliest Exit at T+2 Open (Delayed Exit on Lock/Suspension)`
> **基准时序状态 (Benchmark Timing)**: `VALID` (开盘覆盖率: 100.0%, 收盘覆盖率: 100.0%)

## 1. 核心架构与真实性闭环要点 (Phase 1.5 Integrity Highlights)
- **P0-1/P0-2 基准缺失严格 Fail-Closed**: 基准开盘价缺失或不达标时，所有超额收益标签严格置为 NaN，超额指标显示 `N/A` (`BENCHMARK_TIMING_INVALID`)，绝不进行假想平价或 0 回退；
- **P0-3 严密缓存失效与必需列校验**: Factor / Market 缓存架构升级至 v3.2，严格要求 `[date, symbol, adj_open, adj_close, benchmark_open, benchmark_close, in_universe]`，残缺缓存自动触发重构；
- **P0-4 生产交易 Schema 完全对齐**: Execution 引擎原生接入 `is_limit_up_locked`, `is_limit_down_locked`, `limit_up_price`, `limit_down_price`，准确拦截涨停买入与 ST；
- **P0-5 真实 Delayed Exit 展期机制**: 当 $T+2$ 遇到跌停或停牌无法卖出时，持仓顺延至 $T+3..T+k$ 成交，交易成本严格发生在实际成交日 `actual_exit_date`；
- **P1-1 真实物理父链 Manifest**: 绝不使用伪造哈希，缺失父链如实标记为 `null` / `MISSING`；
- **P1-3 几何复合增长率 (CAGR)**: 纯多头复合收益率严格采用 `(final_equity / initial_equity)**(252/N) - 1`，彻底消除算术均值年化误差。

## 2. 研究概览与因子分级统计
- **候选因子总数**: 10 个
- **STRONG 核心有效因子**: 0 个
- **USEFUL 次级可用因子**: 0 个
- **WEAK 弱预测因子**: 4 个
- **REJECT 淘汰因子**: 6 个
- **高相关冗余聚类群组**: 1 组
- **Walk-Forward 验证状态**: `OOS_VALIDATED` (总 Fold 数: 6)

## 3. Top 10 探索性候选因子表现 (Exploratory Candidates)

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 纯多头 CAGR | 纯多头夏普 | 日均换手 | 纯多头超额年化 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `CORR_PV_10` | `WEAK` | `WEAK` | 反向 (-1) | 20D | -0.0128 | -2.35 | 0.0467 | 78.0% | 2.36 | 25.5% | 86.5% |
| 2 | `WVMA_20` | `WEAK` | `WEAK` | 反向 (-1) | 3D | -0.0149 | -1.85 | 0.1148 | 39.9% | 1.41 | 10.4% | 62.8% |
| 3 | `TURNOVER_STD_20` | `WEAK` | `WEAK` | 反向 (-1) | 20D | -0.0181 | -2.14 | 0.0660 | 28.2% | 1.06 | 5.9% | 33.1% |
| 4 | `CORR_PV_20` | `WEAK` | `WEAK` | 反向 (-1) | 20D | -0.0134 | -2.13 | 0.0660 | 76.1% | 2.29 | 14.6% | 83.5% |
| 5 | `WVMA_10` | `REJECT` | `REJECT` | 反向 (-1) | 3D | -0.0160 | -2.21 | 0.0626 | 51.0% | 1.68 | 17.4% | 57.0% |
| 6 | `LIMIT_UP_SPACE` | `REJECT` | `REJECT` | 正向 (+1) | 1D | 0.0029 | 0.58 | 0.7208 | 130.4% | 2.45 | 37.2% | 135.4% |
| 7 | `IS_LIMIT_UP_LAG1` | `REJECT` | `REJECT` | 正向 (+1) | 3D | 0.0006 | 0.87 | 0.5471 | 31.4% | 1.23 | 40.4% | 109.8% |
| 8 | `TURNOVER_SURGE_20` | `REJECT` | `REJECT` | 正向 (+1) | 10D | 0.0022 | 0.66 | 0.6695 | 47.7% | 1.55 | 53.8% | 49.9% |
| 9 | `TURNOVER_SURGE_5` | `REJECT` | `REJECT` | 正向 (+1) | 1D | 0.0006 | 0.36 | 0.8589 | 39.8% | 1.38 | 67.5% | 51.1% |
| 10 | `AMOUNT_RATIO_5_20` | `REJECT` | `REJECT` | 正向 (+1) | 1D | 0.0010 | 0.23 | 0.9125 | 47.7% | 1.55 | 20.9% | 46.3% |

## 4. 真实截面中性化与正交化实证证据

| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | 真实正交化 RankIC | 中性化状态 | 正交化状态 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `CORR_PV_10` | -0.0128 | -0.0128 | -0.0118 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `WVMA_20` | -0.0149 | -0.0153 | -0.0044 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `TURNOVER_STD_20` | -0.0181 | -0.0178 | -0.0067 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `CORR_PV_20` | -0.0134 | -0.0130 | -0.0066 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `WVMA_10` | -0.0160 | -0.0163 | -0.0153 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `LIMIT_UP_SPACE` | 0.0029 | 0.0025 | 0.0029 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `IS_LIMIT_UP_LAG1` | 0.0006 | 0.0003 | 0.0006 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `TURNOVER_SURGE_20` | 0.0022 | 0.0024 | 0.0088 | `REAL_CALCULATED` | `REAL_CALCULATED` |

## 5. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)

### 📍 Fold 1
- **训练区间**: `2021-09-29` ~ `2023-04-21` (108630 样本, 295 标的)
- **Purge 隔离区间**: `2023-04-24` ~ `2023-06-07` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-06-08` ~ `2023-12-12` (37374 样本, 297 标的)
- **训练集选出因子数**: `1` 个
- **OOS 验证表现**: {'TURNOVER_STD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0436, 'oos_aligned_rank_ic': 0.0436, 'oos_icir': 0.4996}}
### 📍 Fold 2
- **训练区间**: `2022-04-12` ~ `2023-10-31` (110481 样本, 297 标的)
- **Purge 隔离区间**: `2023-11-01` ~ `2023-12-12` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-12-13` ~ `2024-06-24` (37397 样本, 297 标的)
- **训练集选出因子数**: `3` 个
- **OOS 验证表现**: {'CORR_PV_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0099, 'oos_aligned_rank_ic': -0.0099, 'oos_icir': -0.1622}, 'WVMA_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0024, 'oos_aligned_rank_ic': -0.0024, 'oos_icir': -0.0295}, 'TURNOVER_STD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0074, 'oos_aligned_rank_ic': -0.0074, 'oos_icir': -0.0846}}
### 📍 Fold 3
- **训练区间**: `2022-10-19` ~ `2024-05-10` (111643 样本, 297 标的)
- **Purge 隔离区间**: `2024-05-13` ~ `2024-06-24` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2024-06-25` ~ `2024-12-26` (37364 样本, 297 标的)
- **训练集选出因子数**: `0` 个
- **OOS 验证表现**: {}
### 📍 Fold 4
- **训练区间**: `2023-04-24` ~ `2024-11-14` (112075 样本, 297 标的)
- **Purge 隔离区间**: `2024-11-15` ~ `2024-12-26` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2024-12-27` ~ `2025-07-08` (37494 样本, 298 标的)
- **训练集选出因子数**: `2` 个
- **OOS 验证表现**: {'WVMA_10': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0229, 'oos_aligned_rank_ic': 0.0229, 'oos_icir': 0.3202}, 'WVMA_20': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0062, 'oos_aligned_rank_ic': 0.0062, 'oos_icir': 0.0844}}
### 📍 Fold 5
- **训练区间**: `2023-11-01` ~ `2025-05-26` (112243 样本, 298 标的)
- **Purge 隔离区间**: `2025-05-27` ~ `2025-07-08` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2025-07-09` ~ `2026-01-12` (37608 样本, 300 标的)
- **训练集选出因子数**: `0` 个
- **OOS 验证表现**: {}
### 📍 Fold 6
- **训练区间**: `2024-05-13` ~ `2025-11-27` (112422 样本, 299 标的)
- **Purge 隔离区间**: `2025-11-28` ~ `2026-01-12` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2026-01-13` ~ `2026-07-22` (37762 样本, 300 标的)
- **训练集选出因子数**: `1` 个
- **OOS 验证表现**: {'WVMA_10': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': -0.012, 'oos_aligned_rank_ic': 0.012, 'oos_icir': 0.1426}}

---
*本报告由 `research/factor_analyzer.py` 自动生成，20 份结构化证据已同步归档至 `reports/factor_research/`。*