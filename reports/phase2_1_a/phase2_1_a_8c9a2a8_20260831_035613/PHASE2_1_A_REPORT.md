# Phase 2.1-A r2 — Execution-Aligned Label A/B Study Report
# 实盘执行对齐标签严格受控 A/B 实验研究报告 (r2 可复现性闭环版本)

> **研究结论 (Scientific Verdict)**: **`NO_IMPROVEMENT`**
> **实盘许可声明 (Live Trading Guard)**: `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 1. 实验控制变量与血缘规范 (Controlled Variables & Provenance)

- **基准代码提交 (Source Code Commit)**: `8c9a2a84191945a7e7a91b99f840d392fdfae567` (Tree Clean: `True`)
- **基准模型族**: `LightGBM Classification` (二分类概率预测)
- **特征集与顺序**: 严格相同 (79 因子, Feature Hash = `82dfd3e9643ae1352829e736b9c8b89d1d648b98d16ef59153f261bf7a453460`)
- **有效模型参数**: 严格逐字段相同 (Effective Hash = `8112506c58f4714da99fbc1075061d74fdcd17caa06a9ca55894deae24663d2b`, `scale_pos_weight = 1.0`)
- **随机种子**: `42` (严格传播至 feature_fraction_seed, bagging_seed, data_random_seed)
- **时序划分**: 严格相同 Walk-Forward 滚动折划分 (Purge Gap = 25 天 >= Label Horizon 20 天)
- **训练准入池**: 严格相同的共同准入交集 (194,854 行, Common Train Pool Hash = `bfff9a2d0a9b52a0d4924ea36d643923d1e42ab6bd48ed60017d020d22c42bcb`)
- **唯一自变量 (Primary Change)**: **训练目标标签定义** (Legacy `ab_label_legacy` vs Execution-Aligned `ab_label_execution`)

---

## 2. 统一公平评估目标 (Common Execution OOS Target)

两个 Arm 均在完全相同的实盘执行对齐前瞻净超额收益目标 `label_net_alpha_20d` 上进行 OOS 评价：

- **共同 OOS 样本行数**: 220,913
- **共同 OOS 交易日数**: 743
- **共同 OOS 股票池数**: 300
- **共同 OOS 评价池 SHA256**: `a464b29fd12a50891ef68777791ebcb0c7c4f9fc96b59137174387100ca5fd1c`

---

## 3. 核心对比结果 (Controlled A/B Evaluation Results)

| 评价维度 / 指标 | Arm A (Legacy Label) | Arm B (Execution-Aligned Label) | 差异 (Delta B - A) |
| :--- | :---: | :---: | :---: |
| **Mean Daily OOS RankIC** | **0.045205** | **0.043682** | **-0.001524** |
| **NW20 RankICIR (年化)** | **0.348391** | **0.352015** | **+0.003624** |
| **RankIC > 0 交易日占比** | 60.30% | 60.03% | -0.27% |
| **Q5-Q1 年化超额收益差 (pct points)** | 3.31 pts | 5.16 pts | +1.85 pts |
| **Top 10% 20日平均真实执行净超额** | 0.3900% | 0.5054% | +0.1154% |
| **分组单调性得分** | 0.7000 | 0.4000 | -0.3000 |

---

## 4. 统计检验与折稳定性 (Statistical Significance & Fold Stability)

- **20-Day Paired Block Bootstrap (2,000 Resamples)**:
  - **Mean RankIC Delta**: **-0.001524**
  - **95% 置信区间 (95% CI)**: `[-0.011169, +0.009657]`
  - **提升概率 P(Delta > 0)**: **33.25%**
  - **统计显著提升 (CI Lower > 0)**: **`False`**
- **Fold-Level 胜率实证 (已排除 0 样本无效折)**:
  - **滚动折总数 (Total Folds)**: 20
  - **有效对比折数 (Valid Folds)**: 19
  - **0 交易日无效折数 (Excluded Folds)**: 1
  - **Execution Arm 胜出折数**: 9
  - **Legacy Arm 胜出折数**: 10
  - **平局折数 (Ties)**: 0
  - **Fold 胜率 (Fold Win Ratio)**: **47.37%**

---

## 5. 生产模型物理隔离审计 (Production Model Isolation Audit)

- **Production Models Dir**: `saved_models`
- **Production Model Path**: `saved_models/latest_lightgbm.pkl`
- **Exists Before Experiment**: `True`
- **SHA256 Before Experiment**: `93575e2dfd644ee2701ef50376ac612537fb52c07f9c119379bd3b17bea74900`
- **Exists After Experiment**: `True`
- **SHA256 After Experiment**: `93575e2dfd644ee2701ef50376ac612537fb52c07f9c119379bd3b17bea74900`
- **SHA256 Unchanged**: **`True`**
- **Production Models Dir Mutated**: **`False`**
- **Legacy Experiment Model Dir**: `reports/phase2_1_a/phase2_1_a_8c9a2a8_20260831_035613/models/legacy`
- **Execution Experiment Model Dir**: `reports/phase2_1_a/phase2_1_a_8c9a2a8_20260831_035613/models/execution`

---

## 6. 大文件管理与本地证据治理 (Large Artifact Policy)

- **Common Execution OOS Parquet**: `reports/phase2_1_a/phase2_1_a_8c9a2a8_20260831_035613/common_execution_oos.parquet` (Size: 10,352,394 bytes, SHA256: `22f387d73c1bdb48262e4d71e877ffc09d0ba2aef51726ac55ef15f7fbf22750`, Storage: `local_not_git_tracked`)
- **Legacy Model PKL**: `reports/phase2_1_a/phase2_1_a_8c9a2a8_20260831_035613/models/legacy/latest_lightgbm.pkl` (SHA256: `43ec3449cfbab41a457ed8a74b1c9105625b54d712ab57c00f2fbd9f9ee3661f`, Storage: `local_not_git_tracked`)
- **Execution Model PKL**: `reports/phase2_1_a/phase2_1_a_8c9a2a8_20260831_035613/models/execution/latest_lightgbm.pkl` (SHA256: `71c8cdc3af0c0649dfc16dc5c09ae809231ac5bfa18334efbabc217225015cad`, Storage: `local_not_git_tracked`)

---

## 7. 科学判定与结论说明 (Scientific Finding & Next Step)

- **判定状态**: **`NO_IMPROVEMENT`**
- **结论阐述**: 
  实证表明：Execution-Aligned Labels 在本受控实验中未能超越 Legacy Labels。更真实的标签并不必然等同于更高的 OOS 预测能力。
