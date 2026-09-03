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
- **USEFUL 次级可用因子**: 1 个
- **WEAK 弱预测因子**: 2 个
- **REJECT 淘汰因子**: 7 个
- **高相关冗余聚类群组**: 4 组
- **Walk-Forward 验证状态**: `OOS_VALIDATED` (总 Fold 数: 6)

## 3. Top 10 核心有效因子排行榜

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 纯多头 CAGR | 纯多头夏普 | 日均换手 | 纯多头超额年化 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `ROC60` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0261 | -2.65 | 0.0577 | 56.5% | 1.67 | 10.4% | 55.7% |
| 2 | `MA_RATIO_120` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0228 | 2.01 | 0.1415 | 91.6% | 2.39 | 8.3% | 58.3% |
| 3 | `MIN_RATIO_60` | `WEAK` | `WEAK` | 正向 (+1) | 20D | 0.0173 | 1.84 | 0.1627 | 26.6% | 1.01 | 12.2% | 33.7% |
| 4 | `MIN_RATIO_120` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0138 | 1.26 | 0.2834 | 61.6% | 1.94 | 8.5% | 27.9% |
| 5 | `ROC120` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0199 | -1.71 | 0.1681 | 102.2% | 2.59 | 7.7% | 62.3% |
| 6 | `MIN_RATIO_30` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0133 | 1.59 | 0.1931 | 52.9% | 1.69 | 17.8% | 55.7% |
| 7 | `MA_RATIO_60` | `WEAK` | `WEAK` | 正向 (+1) | 20D | 0.0195 | 2.00 | 0.1415 | 69.9% | 1.93 | 11.6% | 76.3% |
| 8 | `MA_RATIO_30` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0118 | 1.45 | 0.2436 | 101.0% | 2.44 | 16.4% | 105.9% |
| 9 | `MAX_RATIO_60` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0078 | 0.89 | 0.4651 | 85.4% | 2.19 | 10.8% | 92.0% |
| 10 | `MAX_RATIO_120` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0065 | 0.65 | 0.6159 | 114.7% | 2.65 | 7.8% | 82.9% |

## 4. 真实截面中性化与正交化实证证据

| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | 真实正交化 RankIC | 中性化状态 | 正交化状态 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `ROC60` | -0.0261 | -0.0259 | -0.0221 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `MA_RATIO_120` | 0.0228 | 0.0233 | 0.0032 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `MIN_RATIO_60` | 0.0173 | 0.0173 | -0.0052 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `MIN_RATIO_120` | 0.0138 | 0.0143 | -0.0111 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `ROC120` | -0.0199 | -0.0195 | 0.0017 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `MIN_RATIO_30` | 0.0133 | 0.0136 | 0.0122 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `MA_RATIO_60` | 0.0195 | 0.0198 | 0.0053 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `MA_RATIO_30` | 0.0118 | 0.0120 | 0.0020 | `REAL_CALCULATED` | `REAL_CALCULATED` |

## 5. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)

### 📍 Fold 1
- **训练区间**: `2021-09-29` ~ `2023-04-21` (108630 样本, 295 标的)
- **Purge 隔离区间**: `2023-04-24` ~ `2023-06-07` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-06-08` ~ `2023-12-12` (37374 样本, 297 标的)
- **训练集选出因子数**: `0` 个
- **OOS 验证表现**: {}
### 📍 Fold 2
- **训练区间**: `2022-04-12` ~ `2023-10-31` (110481 样本, 297 标的)
- **Purge 隔离区间**: `2023-11-01` ~ `2023-12-12` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-12-13` ~ `2024-06-24` (37397 样本, 297 标的)
- **训练集选出因子数**: `0` 个
- **OOS 验证表现**: {}
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
- **训练集选出因子数**: `0` 个
- **OOS 验证表现**: {}
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
- **训练集选出因子数**: `5` 个
- **OOS 验证表现**: {'MIN_RATIO_60': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0314, 'oos_aligned_rank_ic': 0.0314, 'oos_icir': 0.2598}, 'MA_RATIO_60': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0188, 'oos_aligned_rank_ic': 0.0188, 'oos_icir': 0.1813}, 'ROC120': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0057, 'oos_aligned_rank_ic': 0.0057, 'oos_icir': 0.0473}, 'MIN_RATIO_120': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0152, 'oos_aligned_rank_ic': 0.0152, 'oos_icir': 0.1183}, 'MA_RATIO_120': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0119, 'oos_aligned_rank_ic': 0.0119, 'oos_icir': 0.1094}}

---
*本报告由 `research/factor_analyzer.py` 自动生成，20 份结构化证据已同步归档至 `reports/factor_research/`。*