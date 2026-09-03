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
- **WEAK 弱预测因子**: 0 个
- **REJECT 淘汰因子**: 9 个
- **高相关冗余聚类群组**: 2 组
- **Walk-Forward 验证状态**: `OOS_VALIDATED` (总 Fold 数: 6)

## 3. Top 10 核心有效因子排行榜

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 纯多头 CAGR | 纯多头夏普 | 日均换手 | 纯多头超额年化 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `MOM_ACC_20_60` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0245 | -2.58 | 0.0455 | 75.6% | 2.04 | 4.0% | 83.0% |
| 2 | `ATR_RATIO_20` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0174 | -1.59 | 0.2454 | 33.2% | 1.29 | 5.5% | 50.3% |
| 3 | `ATR_RATIO_5` | `REJECT` | `REJECT` | 反向 (-1) | 3D | -0.0145 | -1.56 | 0.2454 | 29.5% | 1.15 | 15.6% | 47.0% |
| 4 | `MOM_ACC_60_120` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0152 | -1.36 | 0.3097 | 87.4% | 2.38 | 2.4% | 48.7% |
| 5 | `STD250` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0181 | -1.45 | 0.2733 | 82.2% | 2.73 | 1.4% | 6.2% |
| 6 | `ROC_STD_60` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0147 | -1.54 | 0.2454 | 45.9% | 1.52 | 11.0% | 43.1% |
| 7 | `ROC_STD_120` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0061 | -0.54 | 0.7170 | 69.0% | 2.13 | 7.6% | 27.0% |
| 8 | `ROC_STD_20` | `REJECT` | `REJECT` | 反向 (-1) | 3D | -0.0060 | -0.79 | 0.5376 | 63.0% | 1.91 | 18.2% | 69.5% |
| 9 | `ATR_RATIO_60` | `REJECT` | `REJECT` | 反向 (-1) | 3D | -0.0118 | -1.10 | 0.4127 | 29.1% | 1.16 | 3.9% | 44.8% |
| 10 | `ROC_STD_250` | `REJECT` | `REJECT` | 反向 (-1) | 1D | -0.0013 | -0.10 | 0.9227 | 66.6% | 2.23 | 5.4% | -41.8% |

## 4. 真实截面中性化与正交化实证证据

| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | 真实正交化 RankIC | 中性化状态 | 正交化状态 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `MOM_ACC_20_60` | -0.0245 | -0.0246 | -0.0108 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `ATR_RATIO_20` | -0.0174 | -0.0180 | -0.0030 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `ATR_RATIO_5` | -0.0145 | -0.0151 | -0.0093 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `MOM_ACC_60_120` | -0.0152 | -0.0155 | -0.0126 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `STD250` | -0.0181 | -0.0185 | -0.0181 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `ROC_STD_60` | -0.0147 | -0.0146 | -0.0134 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `ROC_STD_120` | -0.0061 | -0.0060 | 0.0087 | `REAL_CALCULATED` | `REAL_CALCULATED` |
| `ROC_STD_20` | -0.0060 | -0.0066 | 0.0012 | `REAL_CALCULATED` | `REAL_CALCULATED` |

## 5. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)

### 📍 Fold 1
- **训练区间**: `2021-09-29` ~ `2023-04-21` (108630 样本, 295 标的)
- **Purge 隔离区间**: `2023-04-24` ~ `2023-06-07` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-06-08` ~ `2023-12-12` (37374 样本, 297 标的)
- **训练集选出因子数**: `1` 个
- **OOS 验证表现**: {'ATR_RATIO_60': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0305, 'oos_aligned_rank_ic': 0.0305, 'oos_icir': 0.3028}}
### 📍 Fold 2
- **训练区间**: `2022-04-12` ~ `2023-10-31` (110481 样本, 297 标的)
- **Purge 隔离区间**: `2023-11-01` ~ `2023-12-12` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-12-13` ~ `2024-06-24` (37397 样本, 297 标的)
- **训练集选出因子数**: `2` 个
- **OOS 验证表现**: {'STD250': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0388, 'oos_aligned_rank_ic': -0.0388, 'oos_icir': -0.3003}, 'ATR_RATIO_20': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0076, 'oos_aligned_rank_ic': 0.0076, 'oos_icir': 0.0797}}
### 📍 Fold 3
- **训练区间**: `2022-10-19` ~ `2024-05-10` (111643 样本, 297 标的)
- **Purge 隔离区间**: `2024-05-13` ~ `2024-06-24` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2024-06-25` ~ `2024-12-26` (37364 样本, 297 标的)
- **训练集选出因子数**: `2` 个
- **OOS 验证表现**: {'STD250': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.019, 'oos_aligned_rank_ic': 0.019, 'oos_icir': 0.1981}, 'ATR_RATIO_20': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0209, 'oos_aligned_rank_ic': 0.0209, 'oos_icir': 0.2175}}
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
- **训练集选出因子数**: `6` 个
- **OOS 验证表现**: {'ATR_RATIO_5': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0123, 'oos_aligned_rank_ic': 0.0123, 'oos_icir': 0.1044}, 'ATR_RATIO_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0161, 'oos_aligned_rank_ic': 0.0161, 'oos_icir': 0.1375}, 'ROC_STD_60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0302, 'oos_aligned_rank_ic': 0.0302, 'oos_icir': 0.2786}, 'ROC_STD_120': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0069, 'oos_aligned_rank_ic': 0.0069, 'oos_icir': 0.0621}, 'MOM_ACC_20_60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0231, 'oos_aligned_rank_ic': 0.0231, 'oos_icir': 0.2161}, 'MOM_ACC_60_120': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0088, 'oos_aligned_rank_ic': -0.0088, 'oos_icir': -0.0986}}

---
*本报告由 `research/factor_analyzer.py` 自动生成，20 份结构化证据已同步归档至 `reports/factor_research/`。*