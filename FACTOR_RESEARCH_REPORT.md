# A股多因子研究与 Alpha 真实性验证报告 (Phase 1.5 Production Tradability & Benchmark Closure)

> **研究证据级别 (Validity Status)**: `PRODUCTION_RESEARCH_READY` (数据样本数: 300 标的, 行数: 465544)
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
- **候选因子总数**: 86 个
- **STRONG 核心有效因子**: 1 个
- **USEFUL 次级可用因子**: 14 个
- **WEAK 弱预测因子**: 8 个
- **REJECT 淘汰因子**: 63 个
- **高相关冗余聚类群组**: 22 组
- **Walk-Forward 验证状态**: `OOS_VALIDATED` (总 Fold 数: 9)

## 3. Top 10 核心有效因子排行榜

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 纯多头 CAGR | 纯多头夏普 | 日均换手 | 纯多头超额年化 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `LOG_CIRC_MV` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0382 | -1.98 | 0.1112 | 28.3% | 1.05 | 3.0% | 28.0% |
| 2 | `TURNOVER_STD_20` | `STRONG` | `IN_SAMPLE_STRONG` | 反向 (-1) | 20D | -0.0457 | -3.00 | 0.0150 | 11.8% | 0.58 | 7.8% | 11.4% |
| 3 | `VSTD_20` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0209 | -3.03 | 0.0141 | 10.5% | 0.50 | 15.7% | 9.4% |
| 4 | `MOM_ACC_20_60` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0401 | -2.77 | 0.0256 | 16.8% | 0.71 | 6.1% | 17.8% |
| 5 | `MA_RATIO_60` | `USEFUL` | `USEFUL` | 正向 (+1) | 20D | 0.0379 | 2.58 | 0.0415 | 18.9% | 0.77 | 13.7% | 18.8% |
| 6 | `CORR_PV_10` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0266 | -2.63 | 0.0368 | 11.2% | 0.55 | 27.4% | 12.2% |
| 7 | `CORR_PV_20` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0278 | -2.49 | 0.0481 | 12.8% | 0.61 | 16.6% | 13.0% |
| 8 | `ROC30` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0350 | -2.59 | 0.0402 | 13.9% | 0.60 | 16.3% | 16.5% |
| 9 | `STD30` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0443 | -2.36 | 0.0636 | 4.6% | 0.32 | 8.0% | 3.6% |
| 10 | `STD60` | `USEFUL` | `USEFUL` | 反向 (-1) | 20D | -0.0441 | -2.23 | 0.0798 | 9.0% | 0.52 | 5.8% | 7.1% |

## 4. 真实截面中性化与正交化实证证据

| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | 真实正交化 RankIC | 中性化状态 | 正交化状态 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `LOG_CIRC_MV` | -0.0382 | 0.0000 | None | `REAL_CALCULATED` | `UNAVAILABLE` |
| `TURNOVER_STD_20` | -0.0457 | -0.0394 | None | `REAL_CALCULATED` | `UNAVAILABLE` |
| `VSTD_20` | -0.0209 | -0.0217 | None | `REAL_CALCULATED` | `UNAVAILABLE` |
| `MOM_ACC_20_60` | -0.0401 | -0.0408 | None | `REAL_CALCULATED` | `UNAVAILABLE` |
| `MA_RATIO_60` | 0.0379 | 0.0380 | None | `REAL_CALCULATED` | `UNAVAILABLE` |
| `CORR_PV_10` | -0.0266 | -0.0268 | None | `REAL_CALCULATED` | `UNAVAILABLE` |
| `CORR_PV_20` | -0.0278 | -0.0304 | None | `REAL_CALCULATED` | `UNAVAILABLE` |
| `ROC30` | -0.0350 | -0.0352 | None | `REAL_CALCULATED` | `UNAVAILABLE` |

## 5. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)

### 📍 Fold 1
- **训练区间**: `2020-01-02` ~ `2021-07-23` (100960 样本, 277 标的)
- **Purge 隔离区间**: `2021-07-26` ~ `2021-09-03` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2021-09-06` ~ `2022-03-17` (35388 样本, 284 标的)
- **训练集选出因子数**: `0` 个
- **OOS 验证表现**: {}
### 📍 Fold 2
- **训练区间**: `2020-07-14` ~ `2022-01-27` (103385 样本, 283 标的)
- **Purge 隔离区间**: `2022-01-28` ~ `2022-03-17` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2022-03-18` ~ `2022-09-20` (36133 样本, 291 标的)
- **训练集选出因子数**: `0` 个
- **OOS 验证表现**: {}
### 📍 Fold 3
- **训练区间**: `2021-01-15` ~ `2022-08-08` (105484 样本, 289 标的)
- **Purge 隔离区间**: `2022-08-09` ~ `2022-09-20` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2022-09-21` ~ `2023-03-30` (36869 样本, 295 标的)
- **训练集选出因子数**: `0` 个
- **OOS 验证表现**: {}
### 📍 Fold 4
- **训练区间**: `2021-07-26` ~ `2023-02-16` (107861 样本, 295 标的)
- **Purge 隔离区间**: `2023-02-17` ~ `2023-03-30` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-03-31` ~ `2023-10-10` (37283 样本, 297 标的)
- **训练集选出因子数**: `26` 个
- **OOS 验证表现**: {'KMID': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0003, 'oos_aligned_rank_ic': 0.0003, 'oos_icir': 0.0026}, 'KUP': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0599, 'oos_aligned_rank_ic': 0.0599, 'oos_icir': 0.6388}, 'KLOW': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0395, 'oos_aligned_rank_ic': 0.0395, 'oos_icir': 0.4717}, 'KSFT': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0092, 'oos_aligned_rank_ic': 0.0092, 'oos_icir': 0.078}, 'ROC5': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0099, 'oos_aligned_rank_ic': 0.0099, 'oos_icir': 0.0713}, 'MA_RATIO_5': {'train_direction': 1, 'train_horizon': '1D', 'oos_raw_rank_ic': 0.0122, 'oos_aligned_rank_ic': 0.0122, 'oos_icir': 0.0935}, 'MIN_RATIO_10': {'train_direction': 1, 'train_horizon': '1D', 'oos_raw_rank_ic': 0.0236, 'oos_aligned_rank_ic': 0.0236, 'oos_icir': 0.1845}, 'MA_RATIO_30': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0338, 'oos_aligned_rank_ic': 0.0338, 'oos_icir': 0.306}, 'MIN_RATIO_60': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0602, 'oos_aligned_rank_ic': 0.0602, 'oos_icir': 0.4731}, 'MA_RATIO_60': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0331, 'oos_aligned_rank_ic': 0.0331, 'oos_icir': 0.2593}, 'MIN_RATIO_120': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.049, 'oos_aligned_rank_ic': 0.049, 'oos_icir': 0.2913}, 'MA_RATIO_120': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0231, 'oos_aligned_rank_ic': 0.0231, 'oos_icir': 0.1404}, 'MA_RATIO_250': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0056, 'oos_aligned_rank_ic': 0.0056, 'oos_icir': 0.0369}, 'STD5': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.1023, 'oos_aligned_rank_ic': 0.1023, 'oos_icir': 0.8866}, 'STD10': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0465, 'oos_aligned_rank_ic': 0.0465, 'oos_icir': 0.3327}, 'STD250': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.1216, 'oos_aligned_rank_ic': 0.1216, 'oos_icir': 0.8185}, 'ATR_RATIO_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.1254, 'oos_aligned_rank_ic': 0.1254, 'oos_icir': 0.7523}, 'ROC_STD_60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0114, 'oos_aligned_rank_ic': 0.0114, 'oos_icir': 0.0786}, 'MOM_ACC_20_60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0265, 'oos_aligned_rank_ic': 0.0265, 'oos_icir': 0.1802}, 'MOM_ACC_60_120': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0004, 'oos_aligned_rank_ic': -0.0004, 'oos_icir': -0.0023}, 'VMA_RATIO_20': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': -0.0063, 'oos_aligned_rank_ic': 0.0063, 'oos_icir': 0.0671}, 'CORR_PV_5': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0209, 'oos_aligned_rank_ic': 0.0209, 'oos_icir': 0.2794}, 'CORR_PV_10': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0175, 'oos_aligned_rank_ic': 0.0175, 'oos_icir': 0.2347}, 'CORR_PV_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0179, 'oos_aligned_rank_ic': 0.0179, 'oos_icir': 0.1994}, 'TURNOVER_STD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.1261, 'oos_aligned_rank_ic': 0.1261, 'oos_icir': 1.0316}, 'ALPHA_RESIDUAL_MOMENTUM_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0325, 'oos_aligned_rank_ic': 0.0325, 'oos_icir': 0.3272}}
### 📍 Fold 5
- **训练区间**: `2022-01-28` ~ `2023-08-21` (109868 样本, 297 标的)
- **Purge 隔离区间**: `2023-08-22` ~ `2023-10-10` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-10-11` ~ `2024-04-16` (37407 样本, 297 标的)
- **训练集选出因子数**: `10` 个
- **OOS 验证表现**: {'KUP': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.004, 'oos_aligned_rank_ic': 0.004, 'oos_icir': 0.0377}, 'KLOW': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0059, 'oos_aligned_rank_ic': -0.0059, 'oos_icir': -0.0656}, 'KSFT': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': 0.0001, 'oos_aligned_rank_ic': -0.0001, 'oos_icir': -0.0009}, 'MIN_RATIO_120': {'train_direction': 1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0162, 'oos_aligned_rank_ic': -0.0162, 'oos_icir': -0.112}, 'MIN_RATIO_250': {'train_direction': 1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0198, 'oos_aligned_rank_ic': -0.0198, 'oos_icir': -0.1312}, 'STD10': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0439, 'oos_aligned_rank_ic': 0.0439, 'oos_icir': 0.3043}, 'STD120': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0547, 'oos_aligned_rank_ic': 0.0547, 'oos_icir': 0.2892}, 'STD250': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0286, 'oos_aligned_rank_ic': 0.0286, 'oos_icir': 0.153}, 'ATR_RATIO_20': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0507, 'oos_aligned_rank_ic': 0.0507, 'oos_icir': 0.3048}, 'TURNOVER_STD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0503, 'oos_aligned_rank_ic': 0.0503, 'oos_icir': 0.4323}}
### 📍 Fold 6
- **训练区间**: `2022-08-09` ~ `2024-03-01` (111360 样本, 297 标的)
- **Purge 隔离区间**: `2024-03-04` ~ `2024-04-16` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2024-04-17` ~ `2024-10-24` (37365 样本, 297 标的)
- **训练集选出因子数**: `15` 个
- **OOS 验证表现**: {'KUP': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0073, 'oos_aligned_rank_ic': 0.0073, 'oos_icir': 0.0753}, 'KLOW': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.012, 'oos_aligned_rank_ic': 0.012, 'oos_icir': 0.1247}, 'MAX_RATIO_5': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0079, 'oos_aligned_rank_ic': 0.0079, 'oos_icir': 0.0585}, 'MAX_RATIO_10': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0109, 'oos_aligned_rank_ic': 0.0109, 'oos_icir': 0.0752}, 'MIN_RATIO_20': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0341, 'oos_aligned_rank_ic': 0.0341, 'oos_icir': 0.2583}, 'MIN_RATIO_30': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0233, 'oos_aligned_rank_ic': 0.0233, 'oos_icir': 0.1733}, 'MAX_RATIO_250': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0082, 'oos_aligned_rank_ic': 0.0082, 'oos_icir': 0.0433}, 'MIN_RATIO_250': {'train_direction': 1, 'train_horizon': '3D', 'oos_raw_rank_ic': 0.0158, 'oos_aligned_rank_ic': 0.0158, 'oos_icir': 0.0807}, 'STD5': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0339, 'oos_aligned_rank_ic': 0.0339, 'oos_icir': 0.2767}, 'STD60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0483, 'oos_aligned_rank_ic': 0.0483, 'oos_icir': 0.3275}, 'ATR_RATIO_60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0607, 'oos_aligned_rank_ic': 0.0607, 'oos_icir': 0.4851}, 'TURNOVER_SURGE_20': {'train_direction': 1, 'train_horizon': '10D', 'oos_raw_rank_ic': -0.027, 'oos_aligned_rank_ic': -0.027, 'oos_icir': -0.2989}, 'TURNOVER_STD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0279, 'oos_aligned_rank_ic': 0.0279, 'oos_icir': 0.1993}, 'LIMIT_UP_SPACE': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': -0.0055, 'oos_aligned_rank_ic': 0.0055, 'oos_icir': 0.0442}, 'ALPHA_TURNOVER_SURPRISE_5_20': {'train_direction': 1, 'train_horizon': '10D', 'oos_raw_rank_ic': -0.0294, 'oos_aligned_rank_ic': -0.0294, 'oos_icir': -0.3185}}
### 📍 Fold 7
- **训练区间**: `2023-02-17` ~ `2024-09-03` (112022 样本, 297 标的)
- **Purge 隔离区间**: `2024-09-04` ~ `2024-10-24` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2024-10-25` ~ `2025-04-30` (37458 样本, 298 标的)
- **训练集选出因子数**: `15` 个
- **OOS 验证表现**: {'KUP': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0432, 'oos_aligned_rank_ic': 0.0432, 'oos_icir': 0.4106}, 'MAX_RATIO_5': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0028, 'oos_aligned_rank_ic': 0.0028, 'oos_icir': 0.0207}, 'MAX_RATIO_10': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': 0.0055, 'oos_aligned_rank_ic': -0.0055, 'oos_icir': -0.0378}, 'MAX_RATIO_120': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': 0.028, 'oos_aligned_rank_ic': -0.028, 'oos_icir': -0.175}, 'MAX_RATIO_250': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0156, 'oos_aligned_rank_ic': -0.0156, 'oos_icir': -0.1307}, 'STD60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.1312, 'oos_aligned_rank_ic': 0.1312, 'oos_icir': 0.6364}, 'ATR_RATIO_5': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.1007, 'oos_aligned_rank_ic': 0.1007, 'oos_icir': 0.5906}, 'ATR_RATIO_60': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.1006, 'oos_aligned_rank_ic': 0.1006, 'oos_icir': 0.5245}, 'ROC_STD_250': {'train_direction': 1, 'train_horizon': '1D', 'oos_raw_rank_ic': 0.0071, 'oos_aligned_rank_ic': 0.0071, 'oos_icir': 0.0507}, 'VSTD_10': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0151, 'oos_aligned_rank_ic': 0.0151, 'oos_icir': 0.2009}, 'VSTD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0235, 'oos_aligned_rank_ic': 0.0235, 'oos_icir': 0.3347}, 'WVMA_5': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.054, 'oos_aligned_rank_ic': 0.054, 'oos_icir': 0.4003}, 'TURNOVER_STD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0972, 'oos_aligned_rank_ic': 0.0972, 'oos_icir': 0.6725}, 'LIMIT_UP_SPACE': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0412, 'oos_aligned_rank_ic': 0.0412, 'oos_icir': 0.3866}, 'ALPHA_MONEY_FLOW_DIV_10': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0318, 'oos_aligned_rank_ic': 0.0318, 'oos_icir': 0.3763}}
### 📍 Fold 8
- **训练区间**: `2023-08-22` ~ `2025-03-18` (112214 样本, 298 标的)
- **Purge 隔离区间**: `2025-03-19` ~ `2025-04-30` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2025-05-06` ~ `2025-11-06` (37574 样本, 299 标的)
- **训练集选出因子数**: `14` 个
- **OOS 验证表现**: {'KUP': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0473, 'oos_aligned_rank_ic': -0.0473, 'oos_icir': -0.4985}, 'KLOW': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0306, 'oos_aligned_rank_ic': -0.0306, 'oos_icir': -0.3158}, 'MIN_RATIO_5': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.027, 'oos_aligned_rank_ic': -0.027, 'oos_icir': -0.235}, 'MIN_RATIO_20': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0716, 'oos_aligned_rank_ic': -0.0716, 'oos_icir': -0.5676}, 'MIN_RATIO_60': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0741, 'oos_aligned_rank_ic': -0.0741, 'oos_icir': -0.5761}, 'STD30': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0712, 'oos_aligned_rank_ic': -0.0712, 'oos_icir': -0.4928}, 'STD250': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': 0.0701, 'oos_aligned_rank_ic': -0.0701, 'oos_icir': -0.3263}, 'ATR_RATIO_5': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0714, 'oos_aligned_rank_ic': -0.0714, 'oos_icir': -0.5197}, 'VSTD_10': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0077, 'oos_aligned_rank_ic': 0.0077, 'oos_icir': 0.0947}, 'VSTD_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0026, 'oos_aligned_rank_ic': 0.0026, 'oos_icir': 0.0308}, 'VSTD_60': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0264, 'oos_aligned_rank_ic': 0.0264, 'oos_icir': 0.2756}, 'CORR_PV_10': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': 0.0219, 'oos_aligned_rank_ic': -0.0219, 'oos_icir': -0.2087}, 'WVMA_20': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.0608, 'oos_aligned_rank_ic': -0.0608, 'oos_icir': -0.4415}, 'ALPHA_IDIO_VOL_PENALTY': {'train_direction': 1, 'train_horizon': '20D', 'oos_raw_rank_ic': -0.0675, 'oos_aligned_rank_ic': -0.0675, 'oos_icir': -0.4687}}
### 📍 Fold 9
- **训练区间**: `2024-03-04` ~ `2025-09-17` (112332 样本, 299 标的)
- **Purge 隔离区间**: `2025-09-18` ~ `2025-11-06` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2025-11-07` ~ `2026-05-19` (37696 样本, 300 标的)
- **训练集选出因子数**: `12` 个
- **OOS 验证表现**: {'KMID': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.018, 'oos_aligned_rank_ic': 0.018, 'oos_icir': 0.1379}, 'KSFT': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0182, 'oos_aligned_rank_ic': 0.0182, 'oos_icir': 0.1312}, 'MIN_RATIO_5': {'train_direction': 1, 'train_horizon': '1D', 'oos_raw_rank_ic': 0.0113, 'oos_aligned_rank_ic': 0.0113, 'oos_icir': 0.0929}, 'MIN_RATIO_10': {'train_direction': 1, 'train_horizon': '1D', 'oos_raw_rank_ic': 0.016, 'oos_aligned_rank_ic': 0.016, 'oos_icir': 0.1273}, 'ATR_RATIO_5': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0103, 'oos_aligned_rank_ic': 0.0103, 'oos_icir': 0.058}, 'VSTD_10': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0321, 'oos_aligned_rank_ic': 0.0321, 'oos_icir': 0.499}, 'VSTD_20': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0322, 'oos_aligned_rank_ic': 0.0322, 'oos_icir': 0.4832}, 'WVMA_5': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': 0.0052, 'oos_aligned_rank_ic': -0.0052, 'oos_icir': -0.0429}, 'CORR_PV_10': {'train_direction': -1, 'train_horizon': '5D', 'oos_raw_rank_ic': -0.0233, 'oos_aligned_rank_ic': 0.0233, 'oos_icir': 0.2197}, 'WVMA_10': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0036, 'oos_aligned_rank_ic': 0.0036, 'oos_icir': 0.026}, 'CORR_PV_20': {'train_direction': -1, 'train_horizon': '3D', 'oos_raw_rank_ic': -0.0195, 'oos_aligned_rank_ic': 0.0195, 'oos_icir': 0.1579}, 'WVMA_20': {'train_direction': -1, 'train_horizon': '1D', 'oos_raw_rank_ic': -0.0089, 'oos_aligned_rank_ic': 0.0089, 'oos_icir': 0.0551}}

---
*本报告由 `research/factor_analyzer.py` 自动生成，20 份结构化证据已同步归档至 `reports/factor_research/`。*