# Legacy vs 数据集 v2 因子值对比报告

生成时间: 2026-09-01 | 数据: factor_matrix_300.parquet (legacy 认证件) vs factor_matrix_300_v2.parquet

## 结论摘要

**97 个公共因子中 94 个相关性 < 0.9，中位相关性仅 0.71 —— 数据底座修复对因子值是质变，不是微调。**

这意味着 legacy 的因子研究结论（STRONG 2 / USEFUL 12 / REJECT 57）建立在**实质不同的因子值**之上，
v2 重认证**绝对必要**，legacy 结论不可直接沿用。

## 全量统计

| 指标 | 值 |
|---|---|
| 公共因子数 | 97（列集合完全一致） |
| 相关性中位 | 0.7082 |
| 相关性均值 | 0.7138 |
| 相关性最小 | 0.0235（IS_LIMIT_DOWN_LAG1） |
| 相关性 < 0.9 的因子数 | 94 / 97 |

## 漂移最严重的 10 个因子

| 因子 | 相关性 | 相对漂移 |
|---|---|---|
| IS_LIMIT_DOWN_LAG1 | 0.0235 | 0.37 |
| IS_LIMIT_UP_LAG1 | 0.0585 | 0.61 |
| STD250 | 0.6535 | 0.64 |
| STD120 | 0.6605 | 0.63 |
| ATR_RATIO_60 | 0.6681 | 0.62 |
| STD60 | 0.6701 | 0.62 |
| ATR_RATIO_20 | 0.6745 | 0.61 |
| MA_RATIO_250 | 0.6764 | 0.59 |
| YANG_ZHANG_VOL_20 | 0.6772 | 0.61 |
| STD30 | 0.6822 | 0.61 |

## 变化最小的因子（高相关）

| 因子 | 相关性 |
|---|---|
| AMIHUD_20 / AMIHUD_20_LN | 1.0000 |
| LOG_CIRC_MV | 0.9999 |
| AMIHUD_ILLIQUIDITY_20 | 0.7910 |
| ENTITY_RATIO | 0.7871 |

## 变化来源归因（v2 重建是复合变更，注意不能全归因于单一因素）

1. **LOG_CIRC_MV 原始值修复**（血缘污染 → 原始对数市值）：该列是中性化回归的市值控制变量，
   其值漂移 → 全部 97 个残差化因子连锁变化。
2. **in_universe 掩码差异**：legacy 构建在 538 个交易日使用"全成分"掩码（构建期 PIT 状态未入库），
   v2 使用保存的 PIT 掩码 → 逐日标准化/中性化统计量的截面不同。
3. **代码漂移**：IS_LIMIT_UP_LAG1/IS_LIMIT_DOWN_LAG1 在 legacy 构建期（08-30 09:26）为旧逻辑
   （is_limit_up.shift），当前代码用 is_limit_up_locked → 这两个因子相关性仅 0.02-0.06。
4. **波动类因子（STD/ATR/MA_RATIO）漂移大**：对标准化截面的统计量变化敏感。

## 建议

- 等 8 批 v2 因子研究完成后，用 v2 的 factor_ic/factor_summary/factor_selection 重新评定因子分级，
  与 legacy 的 14 个 STRONG/USEFUL 对比，输出"哪些因子结论翻转"清单。
- 全因子明细见 artifacts/legacy_vs_v2_factor_corr.csv。
