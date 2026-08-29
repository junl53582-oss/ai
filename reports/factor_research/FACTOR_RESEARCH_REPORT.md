# A股多因子研究与 Alpha 真实性验证报告 (Phase 1.4 T+1 Settlement & Lineage Closure)

> **研究证据级别 (Validity Status)**: `DEVELOPMENT_SAMPLE` (数据样本数: 5 标的, 行数: 4555)
> **结算规则 (Settlement Rule)**: `A_SHARE_T_PLUS_1_NO_SAME_DAY_SELL` (严禁日内开仓平仓回转)
> **执行模型 (Execution Definition)**: `Signal at T Close -> Long Entry at T+1 Open -> Earliest Exit at T+2 Open`
> **基准时序状态 (Benchmark Timing)**: `BENCHMARK_OPEN_UNAVAILABLE` (开盘覆盖率: 0.0%, 收盘覆盖率: 100.0%)

## 1. 核心架构与真实性闭环要点 (Phase 1.4 Integrity Highlights)
- **P0-1 严格区分诊断与可交易收益**: T+1 Open->Close 明确标记为 `INTRADAY_FACTOR_DIAGNOSTIC_RETURN`；真实可执行策略遵循 `T Close Signal -> T+1 Open Entry -> T+2 Open Earliest Exit`；
- **P0-2 Benchmark Open 数据链与单维映射**: 消除基准价格直接参与相减的错误，基准指数严格按 `date` 单维映射，同一日期所有标的基准收益严格一致；
- **P0-3 Walk-Forward 全家族 Global FDR**: 每一个 Train Fold 内部严格运行完整 $79 	imes 5 = 395$ 假设的 Global BH-FDR，仅允许通过 FDR 的视界入选并在验证折评估冻结决策；
- **P0-4 Canonical Date+Symbol 组合哈希**: Factor Matrix 与 Research Input Dataset 均严格绑定 `[date, symbol]` 联合唯一索引排序后生成不可篡改 SHA-256 指纹；
- **P0-5 完整可交易性与拒单审计**: 严格审计一字涨停无法买入、一字跌停无法卖出与停牌，导出 `trade_rejection_evidence.csv`；
- **P0-6 独立 Long-Only 策略与非对称交易成本**: 买入计收佣金与滑点，卖出计收佣金、印花税与滑点，独立输出纯多头净值曲线与诊断用多空利差。

## 2. 研究概览与因子分级统计
- **候选因子总数**: 79 个
- **STRONG 核心有效因子**: 0 个
- **USEFUL 次级可用因子**: 0 个
- **WEAK 弱预测因子**: 0 个
- **REJECT 淘汰因子**: 79 个
- **高相关冗余聚类群组**: 15 组
- **Walk-Forward 验证状态**: `DEVELOPMENT_SAMPLE` (总 Fold 数: 1)

## 3. Top 10 探索性候选因子表现 (Exploratory Candidates)

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 纯多头年化 | 纯多头夏普 | 日均换手 | 纯多头超额年化 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `VSTD_60` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.1278 | 2.28 | 0.8118 | 35.5% | 1.16 | 6.4% | 37.1% |
| 2 | `STD60` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.1066 | -2.07 | 0.8295 | 24.1% | 0.81 | 7.6% | 28.3% |
| 3 | `MAX_RATIO_250` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0924 | -1.66 | 1.0000 | 33.7% | 1.11 | 4.6% | 40.8% |
| 4 | `MAX_RATIO_30` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0755 | -1.64 | 1.0000 | 26.5% | 0.88 | 25.3% | 41.0% |
| 5 | `STD10` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0769 | -1.89 | 0.9088 | 14.1% | 0.49 | 24.9% | 25.5% |
| 6 | `ROC20` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0742 | 1.63 | 1.0000 | 1.8% | 0.07 | 17.6% | 7.5% |
| 7 | `STD20` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0803 | -1.61 | 1.0000 | 4.9% | 0.17 | 15.8% | 10.4% |
| 8 | `ROC_STD_20` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0713 | 1.58 | 1.0000 | 8.8% | 0.31 | 18.3% | 13.7% |
| 9 | `MAX_RATIO_60` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0589 | -1.15 | 1.0000 | 30.2% | 0.99 | 15.9% | 37.9% |
| 10 | `STD30` | `REJECT` | `REJECT` | 反向 (-1) | 10D | -0.0631 | -1.23 | 1.0000 | 2.7% | 0.10 | 13.3% | 9.2% |

## 4. 真实截面中性化与正交化实证证据

| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | 真实正交化 RankIC | 中性化状态 | 正交化状态 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `VSTD_60` | 0.1278 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `STD60` | -0.1066 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `MAX_RATIO_250` | -0.0924 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `MAX_RATIO_30` | -0.0755 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `STD10` | -0.0769 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `ROC20` | 0.0742 | None | None | `INSUFFICIENT_CROSS_SECTION` | `UNAVAILABLE` |
| `STD20` | -0.0803 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |
| `ROC_STD_20` | 0.0713 | None | None | `UNAVAILABLE` | `UNAVAILABLE` |

## 5. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)

### 📍 Fold 1
- **训练区间**: `2021-01-01` ~ `2022-12-07` (2520 样本, 5 标的)
- **Purge 隔离区间**: `2022-12-08` ~ `2023-01-18` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-01-19` ~ `2024-01-05` (1260 样本, 5 标的)
- **训练集选出因子数**: `4` 个
- **OOS 验证表现**: {'STD10': {'train_direction': -1, 'train_horizon': '20D', 'oos_raw_rank_ic': 0.1077, 'oos_aligned_rank_ic': -0.1077, 'oos_icir': -0.2028}, 'STD20': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': 0.1567, 'oos_aligned_rank_ic': -0.1567, 'oos_icir': -0.2956}, 'STD30': {'train_direction': -1, 'train_horizon': '10D', 'oos_raw_rank_ic': 0.0374, 'oos_aligned_rank_ic': -0.0374, 'oos_icir': -0.0691}, 'CORR_PV_5': {'train_direction': 1, 'train_horizon': '10D', 'oos_raw_rank_ic': 0.0142, 'oos_aligned_rank_ic': 0.0142, 'oos_icir': 0.0276}}

---
*本报告由 `research/factor_analyzer.py` 自动生成，20 份结构化证据已同步归档至 `reports/factor_research/`。*