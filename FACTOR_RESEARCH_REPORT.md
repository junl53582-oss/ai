# A股多因子研究与 Alpha 真实性验证报告 (Phase 1.2 Empirical Validity & Execution Integrity)

> **研究证据级别 (Validity Status)**: `DEVELOPMENT_SAMPLE` (数据样本数: 5 标的, 行数: 4555)
> **交易执行模型 (Execution Definition)**: `Signal at T Close -> Earliest Entry at T+1 Open -> Realized Post-Entry Return`

## 1. 核心架构与真实性闭环要点 (Phase 1.2 Execution Integrity)
- **P0-1 严格 T+1 开盘真实执行**: 杜绝 $P[T+1]/P[T]-1$ 伪执行前视，以 $T+1$ 开盘价真实计价成交并过滤 $T+1$ 停牌/一字涨跌停；
- **P0-2 截面样本门禁**: 明确区分 `DEVELOPMENT_SAMPLE` (小样本工程测试) 与 `PRODUCTION_RESEARCH`；
- **P0-3 Fail-Closed 中性化**: 样本不足或秩亏时严格返回 `None` 与 `INSUFFICIENT_CROSS_SECTION`，绝不回退充当伪计算；
- **P0-4 Sequential Residualization 正交化**: 逐步残差正交化消除 QR 矩阵维度异常，真实评估增量 Alpha；
- **P0-5 训练折内完整选择**: 严格在 Purged 训练折内独立执行方向、最优视界、HAC-FDR 与冗余剪枝，验证折严格评估冻结决策；
- **P0-6 全要素 Manifest 绑定**: 全量 79 个因子矩阵规范排序哈希，绑定源码 Tree Hash 与测试执行凭据。

## 2. 研究概览与因子分级统计
- **候选因子总数**: 79 个
- **STRONG 核心有效因子**: 0 个
- **USEFUL 次级可用因子**: 0 个
- **WEAK 弱预测因子**: 0 个
- **REJECT 淘汰因子**: 79 个
- **高相关冗余聚类群组**: 12 组
- **Walk-Forward 验证状态**: `OOS_PRELIMINARY` (总 Fold 数: 1)

## 3. Top 10 探索性候选因子表现 (Exploratory / Research Candidates)

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 真实年化收益 | 真实夏普(10bps) | 日均换手 | 纯多头超额年化 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `VSTD_60` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.1257 | 2.23 | 0.9141 | -12.5% | -1.04 | 6.4% | -99.0% |
| 2 | `STD60` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0997 | -1.97 | 0.9141 | 6.0% | 0.44 | 7.6% | -99.0% |
| 3 | `STD10` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0780 | -1.91 | 0.9141 | -8.3% | -0.66 | 25.0% | -99.0% |
| 4 | `ROC20` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0764 | 1.68 | 0.9141 | -13.8% | -1.19 | 17.6% | -99.0% |
| 5 | `MAX_RATIO_250` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0842 | -1.53 | 0.9141 | -14.0% | -1.18 | 4.6% | -99.0% |
| 6 | `MAX_RATIO_30` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0729 | -1.59 | 0.9141 | -13.2% | -1.11 | 25.3% | -99.0% |
| 7 | `ROC_STD_20` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0738 | 1.64 | 0.9141 | -13.8% | -1.20 | 18.3% | -99.0% |
| 8 | `STD20` | `REJECT` | `REJECT` | 反向 (-1) | 10D | -0.0635 | -1.28 | 0.9141 | 3.1% | 0.23 | 15.9% | -99.0% |
| 9 | `MOM_ACC_60_120` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0347 | -0.64 | 0.9575 | 6.7% | 0.50 | 1.5% | -99.0% |
| 10 | `STD30` | `REJECT` | `REJECT` | 反向 (-1) | 10D | -0.0555 | -1.11 | 0.9141 | -0.0% | -0.00 | 13.3% | -99.0% |

## 4. Top 候选因子实证特征解析

### 📍 `VSTD_60`
- **RankIC & HAC 稳健显著性**: 20D 均值 RankIC 为 `0.1257`，Newey-West HAC t-stat 为 `2.23`，全家族 FDR p-val 为 `0.9141`；
- **分层单调性**: 截面分层相关性得分为 `0.90`；
- **真实 T+1 开盘可执行 PnL**: 日均换手率 `6.4%`，真实非重叠日度年化收益 `-12.5%`，扣除摩擦后夏普为 `-1.04`，Top 组相对于基准超额年化为 `-100.0%`；
- **市场状态表现**: 牛市 RankIC=`0.0484`，熊市 RankIC=`0.1950`，震荡市 RankIC=`0.1285`。
### 📍 `STD60`
- **RankIC & HAC 稳健显著性**: 20D 均值 RankIC 为 `-0.0997`，Newey-West HAC t-stat 为 `-1.97`，全家族 FDR p-val 为 `0.9141`；
- **分层单调性**: 截面分层相关性得分为 `-0.70`；
- **真实 T+1 开盘可执行 PnL**: 日均换手率 `7.6%`，真实非重叠日度年化收益 `6.0%`，扣除摩擦后夏普为 `0.44`，Top 组相对于基准超额年化为 `-100.0%`；
- **市场状态表现**: 牛市 RankIC=`-0.0967`，熊市 RankIC=`-0.1849`，震荡市 RankIC=`-0.0721`。
### 📍 `STD10`
- **RankIC & HAC 稳健显著性**: 20D 均值 RankIC 为 `-0.0780`，Newey-West HAC t-stat 为 `-1.91`，全家族 FDR p-val 为 `0.9141`；
- **分层单调性**: 截面分层相关性得分为 `-0.90`；
- **真实 T+1 开盘可执行 PnL**: 日均换手率 `25.0%`，真实非重叠日度年化收益 `-8.3%`，扣除摩擦后夏普为 `-0.66`，Top 组相对于基准超额年化为 `-100.0%`；
- **市场状态表现**: 牛市 RankIC=`-0.0962`，熊市 RankIC=`-0.1867`，震荡市 RankIC=`-0.0336`。
### 📍 `ROC20`
- **RankIC & HAC 稳健显著性**: 20D 均值 RankIC 为 `0.0764`，Newey-West HAC t-stat 为 `1.68`，全家族 FDR p-val 为 `0.9141`；
- **分层单调性**: 截面分层相关性得分为 `0.70`；
- **真实 T+1 开盘可执行 PnL**: 日均换手率 `17.6%`，真实非重叠日度年化收益 `-13.8%`，扣除摩擦后夏普为 `-1.19`，Top 组相对于基准超额年化为 `-100.0%`；
- **市场状态表现**: 牛市 RankIC=`0.0940`，熊市 RankIC=`-0.0287`，震荡市 RankIC=`0.1059`。
### 📍 `MAX_RATIO_250`
- **RankIC & HAC 稳健显著性**: 20D 均值 RankIC 为 `-0.0842`，Newey-West HAC t-stat 为 `-1.53`，全家族 FDR p-val 为 `0.9141`；
- **分层单调性**: 截面分层相关性得分为 `-0.80`；
- **真实 T+1 开盘可执行 PnL**: 日均换手率 `4.6%`，真实非重叠日度年化收益 `-14.0%`，扣除摩擦后夏普为 `-1.18`，Top 组相对于基准超额年化为 `-100.0%`；
- **市场状态表现**: 牛市 RankIC=`-0.1443`，熊市 RankIC=`-0.1392`，震荡市 RankIC=`-0.0422`。

## 5. 真实截面中性化实证证据 (Neutralization Evidence)

| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | Delta | 有效截面天数 | 失败天数 | 状态 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `KMID` | -0.0006 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `KLEN` | 0.0134 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `KMID2` | 0.0167 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `KUP` | -0.0103 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `KLOW` | 0.0180 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `KSFT` | 0.0254 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `ROC5` | -0.0005 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `MAX_RATIO_5` | 0.0258 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |

## 6. 真实施密特逐步正交化证据 (Orthogonalization Evidence)

| 因子名称 | Raw RankIC | 真实正交化 RankIC | Delta | 有效截面天数 | 失败天数 | 状态 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `KMID` | -0.0006 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `KLEN` | 0.0134 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `KMID2` | 0.0167 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `KUP` | -0.0103 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `KLOW` | 0.0180 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `KSFT` | 0.0254 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `ROC5` | -0.0005 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |
| `MAX_RATIO_5` | 0.0258 | None | None | 0 | 911 | `INSUFFICIENT_CROSS_SECTION` |

## 7. 淘汰因子清单及淘汰归因 (Sample Rejected Factors)

| 因子名称 | 综合得分 | 淘汰原因归因 |
| :--- | :---: | :--- |
| `VSTD_60` | 2.30 | `fdr_rejected_pval_0.914` |
| `STD60` | 2.23 | `fdr_rejected_pval_0.914` |
| `STD10` | 1.36 | `fdr_rejected_pval_0.914` |
| `ROC20` | 1.27 | `fdr_rejected_pval_0.914` |
| `MAX_RATIO_250` | 1.24 | `fdr_rejected_pval_0.914` |
| `MAX_RATIO_30` | 1.24 | `fdr_rejected_pval_0.914` |
| `ROC_STD_20` | 1.16 | `high_redundancy_in_cluster_3` |
| `STD20` | 0.93 | `fdr_rejected_pval_0.914` |
| `MOM_ACC_60_120` | 0.81 | `low_icir_stability, fdr_rejected_pval_0.958` |
| `STD30` | 0.70 | `fdr_rejected_pval_0.914` |

## 8. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)

### 📍 Fold 1
- **训练区间**: `2021-01-01` ~ `2022-12-07` (2520 样本, 5 标的)
- **Purge 隔离区间**: `2022-12-08` ~ `2023-01-18` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-01-19` ~ `2024-01-05` (1260 样本, 5 标的)
- **训练集选出因子数**: `13` 个
- **OOS 验证表现**: {'STD20': {'train_direction': -1, 'train_horizon': '10D', 'oos_rank_ic': -0.1496, 'oos_icir': -0.285}, 'STD10': {'train_direction': -1, 'train_horizon': '20D', 'oos_rank_ic': -0.1123, 'oos_icir': -0.2064}, 'STD30': {'train_direction': -1, 'train_horizon': '10D', 'oos_rank_ic': -0.04, 'oos_icir': -0.0738}, 'STD60': {'train_direction': -1, 'train_horizon': '20D', 'oos_rank_ic': -0.1016, 'oos_icir': -0.2039}, 'ROC_STD_60': {'train_direction': 1, 'train_horizon': '1D', 'oos_rank_ic': -0.043, 'oos_icir': -0.0851}, 'WVMA_20': {'train_direction': -1, 'train_horizon': '10D', 'oos_rank_ic': -0.1502, 'oos_icir': -0.3086}, 'CORR_PV_20': {'train_direction': 1, 'train_horizon': '10D', 'oos_rank_ic': 0.107, 'oos_icir': 0.2076}, 'CORR_PV_10': {'train_direction': 1, 'train_horizon': '10D', 'oos_rank_ic': 0.1159, 'oos_icir': 0.2498}, 'STD5': {'train_direction': -1, 'train_horizon': '20D', 'oos_rank_ic': -0.0926, 'oos_icir': -0.1754}, 'WVMA_5': {'train_direction': -1, 'train_horizon': '20D', 'oos_rank_ic': -0.1568, 'oos_icir': -0.3085}, 'CORR_PV_5': {'train_direction': 1, 'train_horizon': '10D', 'oos_rank_ic': 0.0355, 'oos_icir': 0.0696}, 'ENTITY_RATIO': {'train_direction': -1, 'train_horizon': '5D', 'oos_rank_ic': 0.0107, 'oos_icir': 0.0205}, 'KUP': {'train_direction': -1, 'train_horizon': '20D', 'oos_rank_ic': -0.0529, 'oos_icir': -0.1042}}

---
*本报告由 `research/factor_analyzer.py` 自动生成，16 份结构化证据已同步归档至 `reports/factor_research/`。*