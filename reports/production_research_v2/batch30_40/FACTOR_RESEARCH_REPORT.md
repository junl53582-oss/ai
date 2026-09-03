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
- **WEAK 弱预测因子**: 1 个
- **REJECT 淘汰因子**: 8 个
- **高相关冗余聚类群组**: 3 组
- **Walk-Forward 验证状态**: `OOS_VALIDATED` (总 Fold 数: 6)

## 3. Top 10 核心有效因子排行榜

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 纯多头 CAGR | 纯多头夏普 | 日均换手 | 纯多头超额年化 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `STD30` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0229 | -2.10 | 0.0848 | 36.5% | 1.35 | 6.0% | 55.4% |
| 2 | `STD20` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0209 | -2.08 | 0.0848 | 44.6% | 1.57 | 8.5% | 63.2% |
| 3 | `STD10` | `WEAK` | `WEAK` | 反向 (-1) | 20D | -0.0186 | -2.15 | 0.0801 | 40.9% | 1.43 | 15.2% | 43.7% |
| 4 | `STD60` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0209 | -1.85 | 0.1198 | 32.4% | 1.23 | 3.4% | 56.0% |
| 5 | `MIN_RATIO_250` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0127 | 0.92 | 0.5000 | 48.9% | 1.76 | 5.4% | -27.6% |
| 6 | `STD5` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0132 | -1.86 | 0.1198 | 32.4% | 1.18 | 28.0% | 37.5% |
| 7 | `STD120` | `REJECT` | `REJECT` | 反向 (-1) | 3D | -0.0151 | -1.34 | 0.2743 | 61.1% | 2.10 | 2.2% | 33.6% |
| 8 | `MA_RATIO_250` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0122 | 0.87 | 0.5057 | 77.3% | 2.31 | 5.7% | -23.8% |
| 9 | `MAX_RATIO_250` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0045 | 0.36 | 0.8382 | 98.0% | 2.63 | 6.1% | 23.9% |
| 10 | `ROC250` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0054 | -0.39 | 0.8329 | 62.5% | 2.02 | 5.6% | -27.6% |

## 4. 真实截面中性化与正交化实证证据

| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | 真实正交化 RankIC | 中性化状态 | 正交化状态 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `STD30` | -0.0229 | -0.0233 | -0.0063 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `STD20` | -0.0209 | -0.0215 | -0.0097 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `STD10` | -0.0186 | -0.0194 | -0.0148 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `STD60` | -0.0209 | -0.0208 | -0.0058 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `MIN_RATIO_250` | 0.0127 | 0.0126 | 0.0124 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `STD5` | -0.0132 | -0.0140 | -0.0187 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `STD120` | -0.0151 | -0.0156 | 0.0050 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `MA_RATIO_250` | 0.0122 | 0.0122 | 0.0092 | `REAL_CALCULATED` | `REAL_CALCULATED` |

## 5. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)

### 📍 Fold 1
- **训练区间**: `2021-09-29` ~ `2023-04-21` (108630 样本, 295 标的)
- **Purge 隔离区间**: `2023-04-24` ~ `2023-06-07` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-06-08` ~ `2023-12-12` (37374 样本, 297 标的)
- **训练集选出因子数**: `1` 个
- **OOS 验证表现**: {'STD120': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0345, 'oos_aligned_rank_ic': 0.0345, 'oos_icir': 0.3431}}
### 📍 Fold 2
- **训练区间**: `2022-04-12` ~ `2023-10-31` (110481 样本, 297 标的)
- **Purge 隔离区间**: `2023-11-01` ~ `2023-12-12` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-12-13` ~ `2024-06-24` (37397 样本, 297 标的)
- **训练集选出因子数**: `4` 个
- **OOS 验证表现**: {'MIN_RATIO_250': {'train_direction': 1, 'train_horizon': '10D', 'oos_raw_rank_ic': -0.0315, 'oos_aligned_rank_ic': -0.0315, 'oos_icir': -0.3135}, 'STD20': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0166, 'oos_aligned_rank_ic': 0.0166, 'oos_icir': 0.1823}, 'STD30': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.013, 'oos_aligned_rank_ic': 0.013, 'oos_icir': 0.141}, 'STD120': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0373, 'oos_aligned_rank_ic': -0.0373, 'oos_icir': -0.3123}}
### 📍 Fold 3
- **训练区间**: `2022-10-19` ~ `2024-05-10` (111643 样本, 297 标的)
- **Purge 隔离区间**: `2024-05-13` ~ `2024-06-24` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2024-06-25` ~ `2024-12-26` (37364 样本, 297 标的)
- **训练集选出因子数**: `2` 个
- **OOS 验证表现**: {'STD30': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0214, 'oos_aligned_rank_ic': 0.0214, 'oos_icir': 0.2325}, 'STD120': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0222, 'oos_aligned_rank_ic': 0.0222, 'oos_icir': 0.2096}}
### 📍 Fold 4
- **训练区间**: `2023-04-24` ~ `2024-11-14` (112075 样本, 297 标的)
- **Purge 隔离区间**: `2024-11-15` ~ `2024-12-26` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2024-12-27` ~ `2025-07-08` (37494 样本, 298 标的)
- **训练集选出因子数**: `2` 个
- **OOS 验证表现**: {'STD10': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0217, 'oos_aligned_rank_ic': 0.0217, 'oos_icir': 0.2739}, 'STD20': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0039, 'oos_aligned_rank_ic': 0.0039, 'oos_icir': 0.0455}}
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
- **训练集选出因子数**: `3` 个
- **OOS 验证表现**: {'MA_RATIO_250': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0034, 'oos_aligned_rank_ic': 0.0034, 'oos_icir': 0.0269}, 'STD10': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0273, 'oos_aligned_rank_ic': 0.0273, 'oos_icir': 0.2856}, 'STD20': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': -0.0269, 'oos_aligned_rank_ic': 0.0269, 'oos_icir': 0.2641}}

---
*本报告由 `research/factor_analyzer.py` 自动生成，20 份结构化证据已同步归档至 `reports/factor_research/`。*