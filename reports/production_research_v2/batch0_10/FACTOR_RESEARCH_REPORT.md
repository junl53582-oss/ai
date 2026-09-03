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
- **WEAK 弱预测因子**: 1 个
- **REJECT 淘汰因子**: 9 个
- **高相关冗余聚类群组**: 1 组
- **Walk-Forward 验证状态**: `OOS_VALIDATED` (总 Fold 数: 6)

## 3. Top 10 探索性候选因子表现 (Exploratory Candidates)

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 纯多头 CAGR | 纯多头夏普 | 日均换手 | 纯多头超额年化 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `KLEN` | `WEAK` | `WEAK` | 反向 (-1) | 3D | -0.0109 | -1.64 | 0.1368 | 16.1% | 0.71 | 53.4% | 26.9% |
| 2 | `KLOW` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0093 | -2.74 | 0.0131 | 32.9% | 1.17 | 72.3% | 38.5% |
| 3 | `MIN_RATIO_5` | `REJECT` | `REJECT` | 正向 (+1) | 1D | 0.0074 | 1.67 | 0.1336 | 57.9% | 1.75 | 45.0% | 62.4% |
| 4 | `ROC5` | `REJECT` | `REJECT` | 反向 (-1) | 1D | -0.0068 | -1.33 | 0.2259 | 73.9% | 1.99 | 34.2% | 79.4% |
| 5 | `KSFT` | `REJECT` | `REJECT` | 反向 (-1) | 1D | -0.0058 | -2.57 | 0.0186 | 59.1% | 1.70 | 75.4% | 63.2% |
| 6 | `KUP` | `REJECT` | `REJECT` | 反向 (-1) | 3D | -0.0052 | -1.25 | 0.2522 | 16.2% | 0.69 | 70.0% | 22.5% |
| 7 | `KMID` | `REJECT` | `REJECT` | 反向 (-1) | 1D | -0.0048 | -1.96 | 0.0786 | 51.5% | 1.56 | 75.8% | 48.6% |
| 8 | `MA_RATIO_5` | `REJECT` | `REJECT` | 正向 (+1) | 1D | 0.0046 | 1.07 | 0.3219 | 75.6% | 2.02 | 42.5% | 81.7% |
| 9 | `MAX_RATIO_5` | `REJECT` | `REJECT` | 反向 (-1) | 1D | -0.0046 | -1.02 | 0.3374 | 28.7% | 1.04 | 46.5% | 38.2% |
| 10 | `KMID2` | `REJECT` | `REJECT` | 反向 (-1) | 1D | -0.0014 | -0.59 | 0.5811 | 36.5% | 1.29 | 77.9% | 40.3% |

## 4. 真实截面中性化与正交化实证证据

| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | 真实正交化 RankIC | 中性化状态 | 正交化状态 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `KLEN` | -0.0109 | -0.0111 | -0.0112 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `KLOW` | -0.0093 | -0.0093 | -0.0061 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `MIN_RATIO_5` | 0.0074 | 0.0076 | 0.0043 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `ROC5` | -0.0068 | -0.0071 | -0.0033 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `KSFT` | -0.0058 | -0.0058 | -0.0010 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `KUP` | -0.0052 | -0.0054 | -0.0003 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `KMID` | -0.0048 | -0.0049 | -0.0048 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `MA_RATIO_5` | 0.0046 | 0.0050 | 0.0025 | `REAL_CALCULATED` | `REAL_CALCULATED` |

## 5. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)

### 📍 Fold 1
- **训练区间**: `2021-09-29` ~ `2023-04-21` (108630 样本, 295 标的)
- **Purge 隔离区间**: `2023-04-24` ~ `2023-06-07` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-06-08` ~ `2023-12-12` (37374 样本, 297 标的)
- **训练集选出因子数**: `2` 个
- **OOS 验证表现**: {'KMID': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0019, 'oos_aligned_rank_ic': 0.0019, 'oos_icir': 0.0252}, 'KSFT': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0058, 'oos_aligned_rank_ic': 0.0058, 'oos_icir': 0.0757}}
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