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
- **候选因子总数**: 79 个
- **STRONG 核心有效因子**: 2 个
- **USEFUL 次级可用因子**: 12 个
- **WEAK 弱预测因子**: 8 个
- **REJECT 淘汰因子**: 57 个
- **高相关冗余聚类群组**: 21 组
- **Walk-Forward 验证状态**: `OOS_VALIDATED` (总 Fold 数: 6)

## 3. Top 10 核心有效因子排行榜

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 纯多头 CAGR | 纯多头夏普 | 日均换手 | 纯多头超额年化 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `LOG_CIRC_MV` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0395 | -2.05 | 0.1095 | 209.5% | 3.43 | 1.8% | 218.1% |
| 2 | `MOM_ACC_20_60` | `STRONG` | `IN_SAMPLE_STRONG` | 反向 (-1) | 20D | -0.0418 | -2.86 | 0.0218 | 96.0% | 2.36 | 4.0% | 100.8% |
| 3 | `TURNOVER_STD_20` | `STRONG` | `IN_SAMPLE_STRONG` | 反向 (-1) | 20D | -0.0455 | -2.94 | 0.0199 | 31.1% | 1.23 | 5.6% | 53.2% |
| 4 | `MA_RATIO_60` | `USEFUL` | `USEFUL` | 正向 (+1) | 20D | 0.0384 | 2.60 | 0.0399 | 91.2% | 2.24 | 11.7% | 95.9% |
| 5 | `VSTD_20` | `WEAK` | `WEAK` | 反向 (-1) | 20D | -0.0194 | -2.86 | 0.0218 | 59.9% | 1.87 | 14.0% | 68.3% |
| 6 | `CORR_PV_20` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0270 | -2.41 | 0.0583 | 70.5% | 2.14 | 14.1% | 72.0% |
| 7 | `STD30` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0443 | -2.32 | 0.0697 | 44.1% | 1.79 | 5.3% | 78.3% |
| 8 | `ROC30` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0350 | -2.58 | 0.0415 | 77.1% | 2.03 | 14.5% | 82.0% |
| 9 | `STD60` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0441 | -2.20 | 0.0895 | 50.9% | 2.00 | 3.0% | 93.0% |
| 10 | `CORR_PV_10` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0255 | -2.50 | 0.0484 | 70.5% | 2.16 | 24.9% | 75.9% |

## 4. 真实截面中性化与正交化实证证据

| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | 真实正交化 RankIC | 中性化状态 | 正交化状态 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `LOG_CIRC_MV` | -0.0395 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `MOM_ACC_20_60` | -0.0418 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `TURNOVER_STD_20` | -0.0455 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `MA_RATIO_60` | 0.0384 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `VSTD_20` | -0.0194 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `CORR_PV_20` | -0.0270 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `STD30` | -0.0443 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `ROC30` | -0.0350 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |

## 5. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)

### 📍 Fold 1
- **训练区间**: `2021-09-29` ~ `2023-04-21` (108630 样本, 295 标的)
- **Purge 隔离区间**: `2023-04-24` ~ `2023-06-07` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-06-08` ~ `2023-12-12` (37374 样本, 297 标的)
- **训练集选出因子数**: `20` 个
- **OOS 验证表现**: {'KMID': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0049, 'oos_aligned_rank_ic': 0.0049, 'oos_icir': 0.0413}, 'KUP': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0549, 'oos_aligned_rank_ic': 0.0549, 'oos_icir': 0.5302}, 'KLOW': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0307, 'oos_aligned_rank_ic': 0.0307, 'oos_icir': 0.3766}, 'KSFT': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0078, 'oos_aligned_rank_ic': 0.0078, 'oos_icir': 0.0615}, 'ROC5': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0018, 'oos_aligned_rank_ic': 0.0018, 'oos_icir': 0.0134}, 'MA_RATIO_5': {'train_direction': 1, 'train_horizon': '1D', 'oos_raw_rank_ic': 0.0076, 'oos_aligned_rank_ic': 0.0076, 'oos_icir': 0.059}, 'ROC30': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0279, 'oos_aligned_rank_ic': 0.0279, 'oos_icir': 0.2332}, 'MIN_RATIO_60': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0605, 'oos_aligned_rank_ic': 0.0605, 'oos_icir': 0.4665}, 'MA_RATIO_60': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0228, 'oos_aligned_rank_ic': 0.0228, 'oos_icir': 0.1513}, 'MA_RATIO_120': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0255, 'oos_aligned_rank_ic': 0.0255, 'oos_icir': 0.1414}, 'MIN_RATIO_250': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0247, 'oos_aligned_rank_ic': 0.0247, 'oos_icir': 0.1402}, 'STD60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.1187, 'oos_aligned_rank_ic': 0.1187, 'oos_icir': 0.8283}, 'STD250': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.11, 'oos_aligned_rank_ic': 0.11, 'oos_icir': 0.8595}, 'ATR_RATIO_20': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0615, 'oos_aligned_rank_ic': 0.0615, 'oos_icir': 0.3945}, 'ATR_RATIO_60': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0544, 'oos_aligned_rank_ic': 0.0544, 'oos_icir': 0.3161}, 'MOM_ACC_20_60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0514, 'oos_aligned_rank_ic': 0.0514, 'oos_icir': 0.3273}, 'VMA_RATIO_20': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': -0.0028, 'oos_aligned_rank_ic': 0.0028, 'oos_icir': 0.0319}, 'TURNOVER_STD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.1156, 'oos_aligned_rank_ic': 0.1156, 'oos_icir': 0.9661}, 'AMOUNT_RATIO_5_20': {'train_direction': 1, 'train_horizon': '10D', 'oos_raw_rank_ic': 0.0133, 'oos_aligned_rank_ic': 0.0133, 'oos_icir': 0.15}, 'EFFICIENCY_RATIO_10': {'train_direction': 1, 'train_horizon': '10D', 'oos_raw_rank_ic': 0.0123, 'oos_aligned_rank_ic': 0.0123, 'oos_icir': 0.1374}}
### 📍 Fold 2
- **训练区间**: `2022-04-12` ~ `2023-10-31` (110481 样本, 297 标的)
- **Purge 隔离区间**: `2023-11-01` ~ `2023-12-12` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-12-13` ~ `2024-06-24` (37397 样本, 297 标的)
- **训练集选出因子数**: `13` 个
- **OOS 验证表现**: {'KUP': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.004, 'oos_aligned_rank_ic': 0.004, 'oos_icir': 0.0404}, 'KLOW': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0077, 'oos_aligned_rank_ic': -0.0077, 'oos_icir': -0.0859}, 'MIN_RATIO_10': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0139, 'oos_aligned_rank_ic': -0.0139, 'oos_icir': -0.1084}, 'ROC30': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0211, 'oos_aligned_rank_ic': -0.0211, 'oos_icir': -0.1462}, 'MIN_RATIO_30': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0179, 'oos_aligned_rank_ic': -0.0179, 'oos_icir': -0.1469}, 'MIN_RATIO_60': {'train_direction': 1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0034, 'oos_aligned_rank_ic': -0.0034, 'oos_icir': -0.0252}, 'MIN_RATIO_250': {'train_direction': 1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0148, 'oos_aligned_rank_ic': -0.0148, 'oos_icir': -0.1099}, 'STD30': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0568, 'oos_aligned_rank_ic': 0.0568, 'oos_icir': 0.3315}, 'STD120': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0682, 'oos_aligned_rank_ic': 0.0682, 'oos_icir': 0.335}, 'ATR_RATIO_5': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0402, 'oos_aligned_rank_ic': 0.0402, 'oos_icir': 0.285}, 'MOM_ACC_20_60': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': 0.007, 'oos_aligned_rank_ic': -0.007, 'oos_icir': -0.0446}, 'TURNOVER_STD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0593, 'oos_aligned_rank_ic': 0.0593, 'oos_icir': 0.5256}, 'EFFICIENCY_RATIO_10': {'train_direction': 1, 'train_horizon': '10D', 'oos_raw_rank_ic': 0.0024, 'oos_aligned_rank_ic': 0.0024, 'oos_icir': 0.0271}}
### 📍 Fold 3
- **训练区间**: `2022-10-19` ~ `2024-05-10` (111643 样本, 297 标的)
- **Purge 隔离区间**: `2024-05-13` ~ `2024-06-24` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2024-06-25` ~ `2024-12-26` (37364 样本, 297 标的)
- **训练集选出因子数**: `12` 个
- **OOS 验证表现**: {'KUP': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0257, 'oos_aligned_rank_ic': 0.0257, 'oos_icir': 0.255}, 'MAX_RATIO_5': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0117, 'oos_aligned_rank_ic': 0.0117, 'oos_icir': 0.085}, 'MAX_RATIO_10': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0283, 'oos_aligned_rank_ic': 0.0283, 'oos_icir': 0.1966}, 'MIN_RATIO_20': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0785, 'oos_aligned_rank_ic': 0.0785, 'oos_icir': 0.5651}, 'MIN_RATIO_30': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0868, 'oos_aligned_rank_ic': 0.0868, 'oos_icir': 0.5908}, 'MAX_RATIO_250': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': 0.01, 'oos_aligned_rank_ic': -0.01, 'oos_icir': -0.0587}, 'STD20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.083, 'oos_aligned_rank_ic': 0.083, 'oos_icir': 0.5118}, 'STD60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0991, 'oos_aligned_rank_ic': 0.0991, 'oos_icir': 0.5455}, 'ATR_RATIO_60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0909, 'oos_aligned_rank_ic': 0.0909, 'oos_icir': 0.634}, 'TURNOVER_STD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0499, 'oos_aligned_rank_ic': 0.0499, 'oos_icir': 0.3195}, 'LIMIT_UP_SPACE': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': -0.0048, 'oos_aligned_rank_ic': 0.0048, 'oos_icir': 0.0353}, 'EFFICIENCY_RATIO_10': {'train_direction': 1, 'train_horizon': '10D', 'oos_raw_rank_ic': -0.0105, 'oos_aligned_rank_ic': -0.0105, 'oos_icir': -0.1132}}
### 📍 Fold 4
- **训练区间**: `2023-04-24` ~ `2024-11-14` (112075 样本, 297 标的)
- **Purge 隔离区间**: `2024-11-15` ~ `2024-12-26` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2024-12-27` ~ `2025-07-08` (37494 样本, 298 标的)
- **训练集选出因子数**: `16` 个
- **OOS 验证表现**: {'KUP': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0018, 'oos_aligned_rank_ic': 0.0018, 'oos_icir': 0.0161}, 'MAX_RATIO_5': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': 0.0182, 'oos_aligned_rank_ic': -0.0182, 'oos_icir': -0.1455}, 'MIN_RATIO_5': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0127, 'oos_aligned_rank_ic': 0.0127, 'oos_icir': 0.1003}, 'MAX_RATIO_10': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': 0.0047, 'oos_aligned_rank_ic': -0.0047, 'oos_icir': -0.0336}, 'MIN_RATIO_20': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0133, 'oos_aligned_rank_ic': 0.0133, 'oos_icir': 0.091}, 'MIN_RATIO_30': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0204, 'oos_aligned_rank_ic': 0.0204, 'oos_icir': 0.1236}, 'MAX_RATIO_250': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': 0.0025, 'oos_aligned_rank_ic': -0.0025, 'oos_icir': -0.0149}, 'STD60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0073, 'oos_aligned_rank_ic': -0.0073, 'oos_icir': -0.0341}, 'STD120': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0122, 'oos_aligned_rank_ic': 0.0122, 'oos_icir': 0.0601}, 'ATR_RATIO_5': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0091, 'oos_aligned_rank_ic': 0.0091, 'oos_icir': 0.0549}, 'ATR_RATIO_60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0028, 'oos_aligned_rank_ic': -0.0028, 'oos_icir': -0.0144}, 'VMA_RATIO_60': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0193, 'oos_aligned_rank_ic': -0.0193, 'oos_icir': -0.2276}, 'WVMA_5': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0051, 'oos_aligned_rank_ic': -0.0051, 'oos_icir': -0.0388}, 'CORR_PV_10': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': -0.0197, 'oos_aligned_rank_ic': 0.0197, 'oos_icir': 0.1985}, 'WVMA_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0073, 'oos_aligned_rank_ic': -0.0073, 'oos_icir': -0.0457}, 'LIMIT_UP_SPACE': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0124, 'oos_aligned_rank_ic': 0.0124, 'oos_icir': 0.1145}}
### 📍 Fold 5
- **训练区间**: `2023-11-01` ~ `2025-05-26` (112243 样本, 298 标的)
- **Purge 隔离区间**: `2025-05-27` ~ `2025-07-08` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2025-07-09` ~ `2026-01-12` (37608 样本, 300 标的)
- **训练集选出因子数**: `14` 个
- **OOS 验证表现**: {'KLEN': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0527, 'oos_aligned_rank_ic': -0.0527, 'oos_icir': -0.4057}, 'MIN_RATIO_60': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0434, 'oos_aligned_rank_ic': -0.0434, 'oos_icir': -0.362}, 'STD30': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0757, 'oos_aligned_rank_ic': -0.0757, 'oos_icir': -0.4605}, 'STD60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0836, 'oos_aligned_rank_ic': -0.0836, 'oos_icir': -0.5048}, 'STD250': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': 0.0854, 'oos_aligned_rank_ic': -0.0854, 'oos_icir': -0.4062}, 'VSTD_10': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0283, 'oos_aligned_rank_ic': 0.0283, 'oos_icir': 0.3784}, 'VSTD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0351, 'oos_aligned_rank_ic': 0.0351, 'oos_icir': 0.4647}, 'VSTD_60': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0152, 'oos_aligned_rank_ic': 0.0152, 'oos_icir': 0.1478}, 'CORR_PV_5': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0014, 'oos_aligned_rank_ic': 0.0014, 'oos_icir': 0.0161}, 'WVMA_5': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0335, 'oos_aligned_rank_ic': -0.0335, 'oos_icir': -0.256}, 'CORR_PV_10': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': 0.0001, 'oos_aligned_rank_ic': -0.0001, 'oos_icir': -0.001}, 'CORR_PV_20': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0036, 'oos_aligned_rank_ic': 0.0036, 'oos_icir': 0.0263}, 'TURNOVER_STD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0252, 'oos_aligned_rank_ic': -0.0252, 'oos_icir': -0.1817}, 'LIMIT_UP_SPACE': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.032, 'oos_aligned_rank_ic': -0.032, 'oos_icir': -0.3109}}
### 📍 Fold 6
- **训练区间**: `2024-05-13` ~ `2025-11-27` (112422 样本, 299 标的)
- **Purge 隔离区间**: `2025-11-28` ~ `2026-01-12` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2026-01-13` ~ `2026-07-22` (37762 样本, 300 标的)
- **训练集选出因子数**: `16` 个
- **OOS 验证表现**: {'KMID': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0151, 'oos_aligned_rank_ic': 0.0151, 'oos_icir': 0.1071}, 'KSFT': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0169, 'oos_aligned_rank_ic': 0.0169, 'oos_icir': 0.1111}, 'ROC5': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': 0.0046, 'oos_aligned_rank_ic': -0.0046, 'oos_icir': -0.0295}, 'MIN_RATIO_5': {'train_direction': 1, 'train_horizon': '1D', 'oos_raw_rank_ic': 0.0046, 'oos_aligned_rank_ic': 0.0046, 'oos_icir': 0.0352}, 'MIN_RATIO_10': {'train_direction': 1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0142, 'oos_aligned_rank_ic': -0.0142, 'oos_icir': -0.1092}, 'ROC20': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': 0.0155, 'oos_aligned_rank_ic': -0.0155, 'oos_icir': -0.1109}, 'MA_RATIO_20': {'train_direction': 1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0062, 'oos_aligned_rank_ic': -0.0062, 'oos_icir': -0.0395}, 'MIN_RATIO_30': {'train_direction': 1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0148, 'oos_aligned_rank_ic': -0.0148, 'oos_icir': -0.1022}, 'MIN_RATIO_60': {'train_direction': 1, 'train_horizon': '5D', 'oos_raw_rank_ic': 0.0076, 'oos_aligned_rank_ic': 0.0076, 'oos_icir': 0.0463}, 'VSTD_10': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0246, 'oos_aligned_rank_ic': 0.0246, 'oos_icir': 0.3277}, 'VSTD_20': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0279, 'oos_aligned_rank_ic': 0.0279, 'oos_icir': 0.3935}, 'VSTD_60': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.045, 'oos_aligned_rank_ic': 0.045, 'oos_icir': 0.5503}, 'WVMA_5': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0219, 'oos_aligned_rank_ic': 0.0219, 'oos_icir': 0.1388}, 'CORR_PV_10': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0363, 'oos_aligned_rank_ic': 0.0363, 'oos_icir': 0.3245}, 'WVMA_10': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0202, 'oos_aligned_rank_ic': 0.0202, 'oos_icir': 0.1293}, 'CORR_PV_20': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0307, 'oos_aligned_rank_ic': 0.0307, 'oos_icir': 0.2769}}

---
*本报告由 `research/factor_analyzer.py` 自动生成，20 份结构化证据已同步归档至 `reports/factor_research/`。*