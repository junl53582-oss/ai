# A股多因子研究与 Alpha 真实性验证报告 (Phase 1.5 Production Tradability & Benchmark Closure)

> **研究证据级别 (Validity Status)**: `DEVELOPMENT_SAMPLE` (数据样本数: 5 标的, 行数: 4555)
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
- **候选因子总数**: 3 个
- **STRONG 核心有效因子**: 0 个
- **USEFUL 次级可用因子**: 0 个
- **WEAK 弱预测因子**: 0 个
- **REJECT 淘汰因子**: 3 个
- **高相关冗余聚类群组**: 1 组
- **Walk-Forward 验证状态**: `DEVELOPMENT_SAMPLE` (总 Fold 数: 3)

## 3. Top 10 探索性候选因子表现 (Exploratory Candidates)

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 纯多头 CAGR | 纯多头夏普 | 日均换手 | 纯多头超额年化 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `KLEN` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0248 | 1.47 | 0.4258 | -29.8% | -1.19 | 73.2% | -19.1% |
| 2 | `KMID` | `REJECT` | `REJECT` | 反向 (-1) | 1D | -0.0108 | -0.68 | 0.7333 | -28.1% | -1.07 | 75.2% | -21.6% |
| 3 | `KMID2` | `REJECT` | `REJECT` | 正向 (+1) | 1D | 0.0071 | 0.43 | 0.7681 | -10.4% | -0.29 | 72.4% | -1.0% |

## 4. 真实截面中性化与正交化实证证据

| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | 真实正交化 RankIC | 中性化状态 | 正交化状态 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `KLEN` | 0.0248 | None | 0.0376 | `INSUFFICIENT_CROSS_SECTION` | `REAL_CALCULATED` |
| `KMID` | -0.0108 | None | -0.0108 | `INSUFFICIENT_CROSS_SECTION` | `REAL_CALCULATED` |
| `KMID2` | 0.0071 | None | 0.0185 | `INSUFFICIENT_CROSS_SECTION` | `REAL_CALCULATED` |

## 5. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)

### 📍 Fold 1
- **训练区间**: `2021-01-01` ~ `2022-06-14` (1890 样本, 5 标的)
- **Purge 隔离区间**: `2022-06-15` ~ `2022-07-26` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2022-07-27` ~ `2023-01-18` (630 样本, 5 标的)
- **训练集选出因子数**: `0` 个
- **OOS 验证表现**: {}
### 📍 Fold 2
- **训练区间**: `2021-06-28` ~ `2022-12-07` (1890 样本, 5 标的)
- **Purge 隔离区间**: `2022-12-08` ~ `2023-01-18` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-01-19` ~ `2023-07-13` (630 样本, 5 标的)
- **训练集选出因子数**: `0` 个
- **OOS 验证表现**: {}
### 📍 Fold 3
- **训练区间**: `2021-12-21` ~ `2023-06-01` (1890 样本, 5 标的)
- **Purge 隔离区间**: `2023-06-02` ~ `2023-07-13` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-07-14` ~ `2024-01-05` (630 样本, 5 标的)
- **训练集选出因子数**: `1` 个
- **OOS 验证表现**: {'KLEN': {'train_direction': 1, 'train_horizon': '10D', 'oos_raw_rank_ic': 0.0035, 'oos_aligned_rank_ic': 0.0035, 'oos_icir': 0.0066}}

---
*本报告由 `research/factor_analyzer.py` 自动生成，20 份结构化证据已同步归档至 `reports/factor_research/`。*