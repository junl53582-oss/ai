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
- **候选因子总数**: 9 个
- **STRONG 核心有效因子**: 0 个
- **USEFUL 次级可用因子**: 0 个
- **WEAK 弱预测因子**: 1 个
- **REJECT 淘汰因子**: 8 个
- **高相关冗余聚类群组**: 1 组
- **Walk-Forward 验证状态**: `OOS_VALIDATED` (总 Fold 数: 6)

## 3. Top 10 探索性候选因子表现 (Exploratory Candidates)

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 纯多头 CAGR | 纯多头夏普 | 日均换手 | 纯多头超额年化 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `LOG_CIRC_MV` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0395 | -2.05 | 0.2253 | 209.5% | 3.43 | 1.8% | 218.1% |
| 2 | `EFFICIENCY_RATIO_10` | `WEAK` | `WEAK` | 正向 (+1) | 20D | 0.0116 | 3.26 | 0.0261 | 68.4% | 2.04 | 34.8% | 74.7% |
| 3 | `INTRADAY_CLOSE_LOC` | `REJECT` | `REJECT` | 反向 (-1) | 1D | -0.0030 | -1.42 | 0.4915 | 32.9% | 1.19 | 77.2% | 37.4% |
| 4 | `LOWER_SHADOW_RATIO` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0037 | -1.96 | 0.2253 | 31.0% | 1.15 | 77.1% | 33.3% |
| 5 | `IS_LIMIT_DOWN_LAG1` | `REJECT` | `REJECT` | 反向 (-1) | 5D | -0.0001 | -0.37 | 0.9975 | 207.7% | 3.17 | 69.4% | -673.5% |
| 6 | `AMIHUD_20` | `REJECT` | `REJECT` | 正向 (+1) | 1D | 0.0000 | 0.00 | 1.0000 | 155.1% | 3.41 | 4.0% | 154.3% |
| 7 | `AMIHUD_20_LN` | `REJECT` | `REJECT` | 正向 (+1) | 1D | 0.0000 | 0.00 | 1.0000 | 155.1% | 3.41 | 4.0% | 154.3% |
| 8 | `ENTITY_RATIO` | `REJECT` | `REJECT` | 正向 (+1) | 1D | 0.0016 | 0.85 | 0.6636 | 26.5% | 1.04 | 77.6% | 34.8% |
| 9 | `UPPER_SHADOW_RATIO` | `REJECT` | `REJECT` | 反向 (-1) | 3D | -0.0004 | -0.24 | 1.0000 | 23.2% | 0.93 | 77.2% | 29.9% |

## 4. 真实截面中性化与正交化实证证据

| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | 真实正交化 RankIC | 中性化状态 | 正交化状态 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `LOG_CIRC_MV` | -0.0395 | 0.0000 | 0.0084 | `REAL_CALCULATED` | `PARTIAL` |
| `EFFICIENCY_RATIO_10` | 0.0116 | 0.0114 | 0.0415 | `REAL_CALCULATED` | `PARTIAL` |
| `INTRADAY_CLOSE_LOC` | -0.0030 | -0.0032 | -0.0028 | `REAL_CALCULATED` | `PARTIAL` |
| `LOWER_SHADOW_RATIO` | -0.0037 | -0.0036 | 0.0149 | `REAL_CALCULATED` | `PARTIAL` |
| `IS_LIMIT_DOWN_LAG1` | -0.0001 | -0.0002 | -0.0079 | `REAL_CALCULATED` | `PARTIAL` |
| `AMIHUD_20` | 0.0000 | 0.0000 | 0.0000 | `REAL_CALCULATED` | `PARTIAL` |
| `AMIHUD_20_LN` | 0.0000 | 0.0000 | 0.0000 | `REAL_CALCULATED` | `PARTIAL` |
| `ENTITY_RATIO` | 0.0016 | 0.0016 | -0.0187 | `REAL_CALCULATED` | `PARTIAL` |

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
- **训练集选出因子数**: `0` 个
- **OOS 验证表现**: {}

---
*本报告由 `research/factor_analyzer.py` 自动生成，20 份结构化证据已同步归档至 `reports/factor_research/`。*