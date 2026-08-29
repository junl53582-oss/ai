# A股多因子研究与 Alpha 真实性验证报告 (FACTOR RESEARCH INTEGRITY REPORT)

> **数据研究性质说明**: `RESEARCH_ONLY` (基于 Point-In-Time 严格截面清洗、无前视标签与真实日度非重叠 PnL 评估)

## 1. 核心研究架构与防作弊升级要点 (Phase 1.1 Integrity Hotfix)
- **P0-1 收益与标签解耦**: 彻底区分 Label 预测研究 (Forward H-Day Return) 与多空组合回测 (Real Non-Overlapping Daily PnL)；
- **P0-2 严格 Purged Walk-Forward**: 训练窗口与验证窗口之间硬性插入 25 个交易日 Purge Gap，彻底杜绝边界标签泄漏；
- **P0-3 真实执行时序**: 因子于 $T$ 日收盘计算，组合于 $T+1$ 执行并获得真实日度实现收益；
- **P0-4 真实中性化/正交化**: 彻底清除伪常量乘子，全量通过截面 OLS 残差回归与 Gram-Schmidt QR 分解计算真实增量 Alpha；
- **P0-5 排序置换不变性**: 禁用强制拆解 Ties 的 `method='first'`，常数因子严格判定为 0 Alpha；
- **P1-1 强化 FDR 门禁**: STRONG 因子必须满足 Benjamini-Hochberg FDR $p \le 0.05$；
- **P1-2 序列相关性校正**: 采用 Newey-West HAC 异方差自相关稳健统计量评估重叠视界显著性。

## 2. 研究概览与因子分级统计
- **候选因子总数**: 79 个
- **STRONG 核心有效因子**: 0 个
- **USEFUL 次级可用因子**: 0 个
- **WEAK 弱预测因子**: 45 个
- **REJECT 淘汰因子**: 34 个
- **高相关冗余聚类群组**: 12 组
- **Walk-Forward 验证状态**: `OOS_PRELIMINARY` (总 Fold 数: 1)

## 3. Top 10 核心有效因子排行榜 (Top Selected Factors)

| 排名 | 因子名称 | 分级状态 | 证据级别 | 推荐方向 | 最优视界 | Mean RankIC | HAC t-stat | FDR p-val | 真实年化收益 | 真实夏普(10bps) | 日均换手 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `VSTD_60` | `WEAK` | `WEAK` | 正向 (+1) | 20D | 0.1206 | 2.10 | 0.9303 | 36.4% | 0.87 | 6.3% |
| 2 | `STD60` | `WEAK` | `WEAK` | 反向 (-1) | 20D | -0.0954 | -1.84 | 0.9303 | 43.5% | 0.99 | 7.5% |
| 3 | `MAX_RATIO_250` | `WEAK` | `WEAK` | 反向 (-1) | 20D | -0.0896 | -1.60 | 0.9303 | 31.5% | 0.74 | 4.4% |
| 4 | `ROC20` | `WEAK` | `WEAK` | 正向 (+1) | 20D | 0.0790 | 1.73 | 0.9303 | -10.3% | -0.59 | 17.5% |
| 5 | `STD10` | `WEAK` | `WEAK` | 反向 (-1) | 20D | -0.0744 | -1.82 | 0.9303 | 8.0% | -0.14 | 24.9% |
| 6 | `ROC_STD_20` | `REJECT` | `REJECT` | 正向 (+1) | 20D | 0.0771 | 1.71 | 0.9303 | -9.2% | -0.55 | 18.2% |
| 7 | `MAX_RATIO_30` | `WEAK` | `WEAK` | 反向 (-1) | 20D | -0.0680 | -1.47 | 0.9303 | 22.4% | 0.22 | 25.2% |
| 8 | `MAX_RATIO_60` | `WEAK` | `WEAK` | 反向 (-1) | 20D | -0.0573 | -1.13 | 0.9303 | 48.5% | 0.95 | 15.8% |
| 9 | `STD30` | `WEAK` | `WEAK` | 反向 (-1) | 10D | -0.0501 | -0.97 | 0.9303 | 33.0% | 0.67 | 13.2% |
| 10 | `CORR_PV_20` | `WEAK` | `WEAK` | 正向 (+1) | 10D | 0.0466 | 1.02 | 0.9303 | 34.2% | 0.51 | 24.5% |

## 4. Top 因子保留原因与真实特征解析

### 🌟 `VSTD_60`
- **RankIC & 统计显著性**: 20D 均值 RankIC 为 `0.1206`，Newey-West HAC t-stat 为 `2.10`，FDR 校正 p-val 为 `0.9303`；
- **分层单调性**: 截面分层相关性得分为 `0.90`；
- **真实日度 PnL 表现**: 日均多头换手率 `6.3%`，真实非重叠日收益年化 `36.4%`，扣除 10 bps 摩擦成本后真实每日 PnL 夏普为 `0.87`；
- **市场状态表现**: 牛市 RankIC=`0.0419`，熊市 RankIC=`0.2058`，震荡市 RankIC=`0.1186`。
### 🌟 `STD60`
- **RankIC & 统计显著性**: 20D 均值 RankIC 为 `-0.0954`，Newey-West HAC t-stat 为 `-1.84`，FDR 校正 p-val 为 `0.9303`；
- **分层单调性**: 截面分层相关性得分为 `-0.70`；
- **真实日度 PnL 表现**: 日均多头换手率 `7.5%`，真实非重叠日收益年化 `43.5%`，扣除 10 bps 摩擦成本后真实每日 PnL 夏普为 `0.99`；
- **市场状态表现**: 牛市 RankIC=`-0.1060`，熊市 RankIC=`-0.1819`，震荡市 RankIC=`-0.0627`。
### 🌟 `MAX_RATIO_250`
- **RankIC & 统计显著性**: 20D 均值 RankIC 为 `-0.0896`，Newey-West HAC t-stat 为 `-1.60`，FDR 校正 p-val 为 `0.9303`；
- **分层单调性**: 截面分层相关性得分为 `-0.80`；
- **真实日度 PnL 表现**: 日均多头换手率 `4.4%`，真实非重叠日收益年化 `31.5%`，扣除 10 bps 摩擦成本后真实每日 PnL 夏普为 `0.74`；
- **市场状态表现**: 牛市 RankIC=`-0.1582`，熊市 RankIC=`-0.1521`，震荡市 RankIC=`-0.0416`。
### 🌟 `ROC20`
- **RankIC & 统计显著性**: 20D 均值 RankIC 为 `0.0790`，Newey-West HAC t-stat 为 `1.73`，FDR 校正 p-val 为 `0.9303`；
- **分层单调性**: 截面分层相关性得分为 `0.70`；
- **真实日度 PnL 表现**: 日均多头换手率 `17.5%`，真实非重叠日收益年化 `-10.3%`，扣除 10 bps 摩擦成本后真实每日 PnL 夏普为 `-0.59`；
- **市场状态表现**: 牛市 RankIC=`0.1130`，熊市 RankIC=`-0.0293`，震荡市 RankIC=`0.1043`。
### 🌟 `STD10`
- **RankIC & 统计显著性**: 20D 均值 RankIC 为 `-0.0744`，Newey-West HAC t-stat 为 `-1.82`，FDR 校正 p-val 为 `0.9303`；
- **分层单调性**: 截面分层相关性得分为 `-0.90`；
- **真实日度 PnL 表现**: 日均多头换手率 `24.9%`，真实非重叠日收益年化 `8.0%`，扣除 10 bps 摩擦成本后真实每日 PnL 夏普为 `-0.14`；
- **市场状态表现**: 牛市 RankIC=`-0.0959`，熊市 RankIC=`-0.1804`，震荡市 RankIC=`-0.0289`。

## 5. 真实中性化与正交化对照实证 (Real Neutralization & Orthogonalization Evidence)

| 因子名称 | Raw RankIC | 真实市值行业中性化 RankIC | Delta (中性化) | 真实正交化 RankIC | Delta (正交化) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `KMID` | 0.0017 | 0.0017 | +0.0000 | 0.0017 | +0.0000 |
| `KLEN` | 0.0084 | 0.0084 | +0.0000 | 0.0084 | +0.0000 |
| `KMID2` | 0.0181 | 0.0181 | +0.0000 | 0.0181 | +0.0000 |
| `KUP` | -0.0122 | -0.0122 | +0.0000 | -0.0122 | +0.0000 |
| `KLOW` | 0.0165 | 0.0165 | +0.0000 | 0.0165 | +0.0000 |
| `KSFT` | 0.0273 | 0.0273 | +0.0000 | 0.0273 | +0.0000 |
| `ROC5` | 0.0018 | 0.0018 | +0.0000 | 0.0018 | +0.0000 |
| `MAX_RATIO_5` | 0.0225 | 0.0225 | +0.0000 | 0.0225 | +0.0000 |

## 6. 淘汰因子清单及淘汰归因 (Sample Rejected Factors)

| 因子名称 | 综合得分 | 淘汰原因归因 |
| :--- | :---: | :--- |
| `ROC_STD_20` | 1.20 | `high_redundancy_in_cluster_3` |
| `MA_RATIO_60` | 0.56 | `high_redundancy_in_cluster_6` |
| `MIN_RATIO_60` | 0.49 | `high_redundancy_in_cluster_6` |
| `MA_RATIO_120` | 0.46 | `high_redundancy_in_cluster_8` |
| `MA_RATIO_250` | 0.33 | `high_redundancy_in_cluster_10` |
| `MAX_RATIO_20` | 0.31 | `high_redundancy_in_cluster_4` |
| `ROC_STD_60` | 0.23 | `high_redundancy_in_cluster_5` |
| `KSFT` | 0.06 | `high_redundancy_in_cluster_2` |
| `MIN_RATIO_250` | -0.07 | `high_redundancy_in_cluster_10` |
| `ROC120` | -0.20 | `high_redundancy_in_cluster_7` |

## 7. 严格 Purged Walk-Forward 滚动折数审计 (Fold-by-Fold Audit)

### 📍 Fold 1
- **训练区间**: `2021-01-01` ~ `2022-12-07` (2520 条样本)
- **Purge 隔离区间**: `2022-12-08` ~ `2023-01-18` (硬性隔离，无标签重叠)
- **验证区间 (OOS)**: `2023-01-19` ~ `2024-01-05` (1260 条样本)
- **训练集选出因子数**: `32` 个
- **OOS 验证表现**: {'ROC5': {'oos_rank_ic': -0.0632, 'oos_icir': -0.1279}, 'MAX_RATIO_10': {'oos_rank_ic': -0.0613, 'oos_icir': -0.1219}, 'ROC20': {'oos_rank_ic': -0.0058, 'oos_icir': -0.0116}, 'MAX_RATIO_20': {'oos_rank_ic': -0.0667, 'oos_icir': -0.1239}, 'MAX_RATIO_30': {'oos_rank_ic': -0.0189, 'oos_icir': -0.0376}, 'MIN_RATIO_30': {'oos_rank_ic': 0.0381, 'oos_icir': 0.0723}, 'ROC120': {'oos_rank_ic': -0.1242, 'oos_icir': -0.2602}, 'MAX_RATIO_120': {'oos_rank_ic': -0.1689, 'oos_icir': -0.3973}, 'ROC250': {'oos_rank_ic': -0.0127, 'oos_icir': -0.025}, 'MAX_RATIO_250': {'oos_rank_ic': -0.0884, 'oos_icir': -0.1937}, 'MIN_RATIO_250': {'oos_rank_ic': -0.0056, 'oos_icir': -0.0102}, 'STD5': {'oos_rank_ic': -0.0947, 'oos_icir': -0.1825}, 'STD10': {'oos_rank_ic': -0.1099, 'oos_icir': -0.2057}, 'STD20': {'oos_rank_ic': -0.1444, 'oos_icir': -0.2757}, 'STD30': {'oos_rank_ic': -0.0426, 'oos_icir': -0.0805}, 'STD60': {'oos_rank_ic': -0.1115, 'oos_icir': -0.2238}, 'STD120': {'oos_rank_ic': -0.0186, 'oos_icir': -0.0355}, 'STD250': {'oos_rank_ic': -0.0452, 'oos_icir': -0.0891}, 'ATR_RATIO_5': {'oos_rank_ic': -0.0043, 'oos_icir': -0.0081}, 'ATR_RATIO_20': {'oos_rank_ic': -0.0933, 'oos_icir': -0.186}, 'ROC_STD_20': {'oos_rank_ic': 0.0152, 'oos_icir': 0.0307}, 'ROC_STD_120': {'oos_rank_ic': -0.1214, 'oos_icir': -0.2547}, 'VSTD_5': {'oos_rank_ic': -0.0075, 'oos_icir': -0.0153}, 'VSTD_10': {'oos_rank_ic': -0.034, 'oos_icir': -0.0622}, 'VSTD_60': {'oos_rank_ic': 0.1639, 'oos_icir': 0.3301}, 'CORR_PV_5': {'oos_rank_ic': 0.0375, 'oos_icir': 0.0728}, 'WVMA_5': {'oos_rank_ic': -0.1564, 'oos_icir': -0.3153}, 'CORR_PV_10': {'oos_rank_ic': 0.1206, 'oos_icir': 0.2581}, 'WVMA_10': {'oos_rank_ic': -0.1636, 'oos_icir': -0.3071}, 'CORR_PV_20': {'oos_rank_ic': 0.1018, 'oos_icir': 0.1988}, 'WVMA_20': {'oos_rank_ic': -0.1625, 'oos_icir': -0.3351}, 'ENTITY_RATIO': {'oos_rank_ic': 0.0013, 'oos_icir': 0.0024}}

---
*本报告由 `research/factor_analyzer.py` 自动生成，结构化数据已同步归档至 `reports/factor_research/`。*