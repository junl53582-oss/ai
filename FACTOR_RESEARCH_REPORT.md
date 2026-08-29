# A股多因子研究与 Alpha 真实性验证报告 (Phase 1.3 Execution & Research Provenance Closure)

> **研究证据级别 (Validity Status)**: `DEVELOPMENT_SAMPLE` (数据样本数: 5 标的, 行数: 4555)
> **交易执行模型 (Execution Definition)**: `Signal at T Close -> Earliest Entry at T+1 Open -> Realized Post-Entry Return`
> **基准收益模型 (Benchmark Definition)**: `Benchmark Entry at T+1 Open -> Benchmark Exit at T+H Close (Exact-Math Matching)`

## 1. 核心架构与真实性闭环要点 (Phase 1.3 Integrity Highlights)
- **P0-1 严格基准收益解耦**: 彻底消除基准价格直接参与相减的错误，基准日度收益与前向收益严格基于 `T+1 Open -> T+H Close` 计价计算；
- **P0-2 前向超额标签严密对齐**: 个股与基准收益享有完全一致的进入与退出时间点，超额收益精确为小数收益率之差；
- **P0-3 CI 真实可复现性**: `requirements.txt` 完整纳入 `cryptography` 与 `pytest`，工作流配置真实测试与产物上传；
- **P0-4 Manifest 干净源码绑定**: 引入两阶段提交流程，Manifest 严密绑定执行时的 clean source commit 指纹；
- **P0-5 真实可交易性过滤**: 严格审计 T+1 停牌、一字涨跌停锁死与无开盘价，阻断不可执行成交；
- **P0-6 多空换手与成本独立分离**: 独立计算多头换手与空头换手，分别核算买入滑点佣金与卖出印花税佣金滑点。

## 2. 研究概览与因子分级统计
- **候选因子总数**: 79 个
- **STRONG 核心有效因子**: 0 个
- **USEFUL 次级可用因子**: 0 个
- **WEAK 弱预测因子**: 0 个
- **REJECT 淘汰因子**: 79 个
- **高相关冗余聚类群组**: 15 组
- **Walk-Forward 验证状态**: `DEVELOPMENT_SAMPLE` (总 Fold 数: 1)

## 3. Top 10 探索性候选因子表现 (Exploratory / Research Candidates)

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 真实年化收益 | 真实夏普(10bps) | 日均换手 | 纯多头超额年化 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `VSTD_60` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.1257 | 2.23 | 1.0000 | -12.9% | -1.08 | 7.4% | -3.5% |
| 2 | `STD60` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0997 | -1.97 | 1.0000 | 5.6% | 0.42 | 8.3% | 5.4% |
| 3 | `STD10` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0780 | -1.91 | 1.0000 | -7.8% | -0.62 | 24.0% | 7.3% |
| 4 | `MAX_RATIO_250` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0842 | -1.53 | 1.0000 | -14.5% | -1.23 | 5.8% | -2.5% |
| 5 | `ROC20` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0764 | 1.68 | 1.0000 | -13.9% | -1.20 | 17.8% | -3.0% |
| 6 | `ROC_STD_20` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0738 | 1.64 | 1.0000 | -13.6% | -1.18 | 17.8% | -4.3% |
| 7 | `MAX_RATIO_30` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0729 | -1.59 | 1.0000 | -11.8% | -0.98 | 22.2% | -0.2% |
| 8 | `STD20` | `REJECT` | `REJECT` | 反向 (-1) | 10D | -0.0635 | -1.28 | 1.0000 | 2.5% | 0.19 | 17.0% | 10.1% |
| 9 | `MOM_ACC_60_120` | `REJECT` | `REJECT` | 反向 (-1) | 20D | -0.0347 | -0.64 | 1.0000 | 6.5% | 0.48 | 1.8% | 9.2% |
| 10 | `STD30` | `REJECT` | `REJECT` | 反向 (-1) | 10D | -0.0555 | -1.11 | 1.0000 | -0.4% | -0.03 | 14.1% | 7.8% |

## 4. Top 候选因子实证特征解析

### 📍 `VSTD_60`
- **RankIC & HAC 稳健显著性**: 20D 均值 RankIC 为 `0.1257`，Newey-West HAC t-stat 为 `2.23`，全家族 FDR p-val 为 `1.0000`；
- **分层单调性**: 截面分层相关性得分为 `0.90`；
- **真实 T+1 开盘可执行 PnL**: 多头换手率 `6.4%`，空头换手率 `8.4%`，综合日均换手率 `7.4%`，真实非重叠日度年化收益 `-12.9%`，扣除摩擦后夏普为 `-1.08`，Top 组相对于基准超额年化为 `-3.5%`；
- **市场状态表现**: 牛市 RankIC=`0.0484`，熊市 RankIC=`0.1950`，震荡市 RankIC=`0.1285`。
### 📍 `STD60`
- **RankIC & HAC 稳健显著性**: 20D 均值 RankIC 为 `-0.0997`，Newey-West HAC t-stat 为 `-1.97`，全家族 FDR p-val 为 `1.0000`；
- **分层单调性**: 截面分层相关性得分为 `-0.70`；
- **真实 T+1 开盘可执行 PnL**: 多头换手率 `7.6%`，空头换手率 `8.9%`，综合日均换手率 `8.3%`，真实非重叠日度年化收益 `5.6%`，扣除摩擦后夏普为 `0.42`，Top 组相对于基准超额年化为 `5.4%`；
- **市场状态表现**: 牛市 RankIC=`-0.0967`，熊市 RankIC=`-0.1849`，震荡市 RankIC=`-0.0721`。
### 📍 `STD10`
- **RankIC & HAC 稳健显著性**: 20D 均值 RankIC 为 `-0.0780`，Newey-West HAC t-stat 为 `-1.91`，全家族 FDR p-val 为 `1.0000`；
- **分层单调性**: 截面分层相关性得分为 `-0.90`；
- **真实 T+1 开盘可执行 PnL**: 多头换手率 `25.0%`，空头换手率 `23.1%`，综合日均换手率 `24.0%`，真实非重叠日度年化收益 `-7.8%`，扣除摩擦后夏普为 `-0.62`，Top 组相对于基准超额年化为 `7.3%`；
- **市场状态表现**: 牛市 RankIC=`-0.0962`，熊市 RankIC=`-0.1867`，震荡市 RankIC=`-0.0367`。
### 📍 `MAX_RATIO_250`
- **RankIC & HAC 稳健显著性**: 20D 均值 RankIC 为 `-0.0842`，Newey-West HAC t-stat 为 `-1.53`，全家族 FDR p-val 为 `1.0000`；
- **分层单调性**: 截面分层相关性得分为 `-0.80`；
- **真实 T+1 开盘可执行 PnL**: 多头换手率 `4.6%`，空头换手率 `6.9%`，综合日均换手率 `5.8%`，真实非重叠日度年化收益 `-14.5%`，扣除摩擦后夏普为 `-1.23`，Top 组相对于基准超额年化为 `-2.5%`；
- **市场状态表现**: 牛市 RankIC=`-0.1443`，熊市 RankIC=`-0.1392`，震荡市 RankIC=`-0.0422`。
### 📍 `ROC20`
- **RankIC & HAC 稳健显著性**: 20D 均值 RankIC 为 `0.0764`，Newey-West HAC t-stat 为 `1.68`，全家族 FDR p-val 为 `1.0000`；
- **分层单调性**: 截面分层相关性得分为 `0.70`；
- **真实 T+1 开盘可执行 PnL**: 多头换手率 `17.6%`，空头换手率 `17.9%`，综合日均换手率 `17.8%`，真实非重叠日度年化收益 `-13.9%`，扣除摩擦后夏普为 `-1.20`，Top 组相对于基准超额年化为 `-3.0%`；
- **市场状态表现**: 牛市 RankIC=`0.0940`，熊市 RankIC=`-0.0287`，震荡市 RankIC=`0.1059`。

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
| `VSTD_60` | 2.30 | `fdr_rejected_pval_1.000` |
| `STD60` | 2.22 | `fdr_rejected_pval_1.000` |
| `STD10` | 1.36 | `fdr_rejected_pval_1.000` |
| `MAX_RATIO_250` | 1.29 | `fdr_rejected_pval_1.000` |
| `ROC20` | 1.28 | `fdr_rejected_pval_1.000` |
| `ROC_STD_20` | 1.17 | `high_redundancy_in_cluster_6` |
| `MAX_RATIO_30` | 1.15 | `fdr_rejected_pval_1.000` |
| `STD20` | 0.89 | `fdr_rejected_pval_1.000` |
| `MOM_ACC_60_120` | 0.82 | `low_icir_stability, fdr_rejected_pval_1.000` |
| `STD30` | 0.70 | `fdr_rejected_pval_1.000` |

## 8. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)

### 📍 Fold 1
- **训练区间**: `2021-01-01` ~ `2022-12-07` (2520 样本, 5 标的)
- **Purge 隔离区间**: `2022-12-08` ~ `2023-01-18` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-01-19` ~ `2024-01-05` (1260 样本, 5 标的)
- **训练集选出因子数**: `13` 个
- **OOS 验证表现**: {'STD20': {'train_direction': -1, 'train_horizon': '10D', 'oos_rank_ic': -0.1496, 'oos_icir': -0.285}, 'STD10': {'train_direction': -1, 'train_horizon': '20D', 'oos_rank_ic': -0.1123, 'oos_icir': -0.2064}, 'STD30': {'train_direction': -1, 'train_horizon': '10D', 'oos_rank_ic': -0.04, 'oos_icir': -0.0738}, 'STD60': {'train_direction': -1, 'train_horizon': '20D', 'oos_rank_ic': -0.1016, 'oos_icir': -0.2039}, 'ROC_STD_60': {'train_direction': 1, 'train_horizon': '1D', 'oos_rank_ic': -0.043, 'oos_icir': -0.0851}, 'WVMA_20': {'train_direction': -1, 'train_horizon': '10D', 'oos_rank_ic': -0.1502, 'oos_icir': -0.3086}, 'CORR_PV_20': {'train_direction': 1, 'train_horizon': '10D', 'oos_rank_ic': 0.107, 'oos_icir': 0.2076}, 'CORR_PV_10': {'train_direction': 1, 'train_horizon': '10D', 'oos_rank_ic': 0.1159, 'oos_icir': 0.2498}, 'STD5': {'train_direction': -1, 'train_horizon': '20D', 'oos_rank_ic': -0.0926, 'oos_icir': -0.1754}, 'WVMA_5': {'train_direction': -1, 'train_horizon': '20D', 'oos_rank_ic': -0.1568, 'oos_icir': -0.3085}, 'CORR_PV_5': {'train_direction': 1, 'train_horizon': '10D', 'oos_rank_ic': 0.0355, 'oos_icir': 0.0696}, 'ENTITY_RATIO': {'train_direction': -1, 'train_horizon': '5D', 'oos_rank_ic': 0.0107, 'oos_icir': 0.0205}, 'KUP': {'train_direction': -1, 'train_horizon': '20D', 'oos_rank_ic': -0.0529, 'oos_icir': -0.1042}}

---
*本报告由 `research/factor_analyzer.py` 自动生成，18 份结构化证据已同步归档至 `reports/factor_research/`。*