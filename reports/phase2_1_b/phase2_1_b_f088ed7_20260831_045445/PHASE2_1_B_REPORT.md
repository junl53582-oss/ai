# Phase 2.1-B — Model Objective Study Report
# A股模型学习目标函数严格受控三臂 OOS 研究报告 (Classification vs Regression vs LambdaRank)

> **研究结论 (Scientific Verdict)**: **`MIXED_EVIDENCE`**
> **实盘许可声明 (Live Trading Guard)**: `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 1. 实验控制变量与血缘规范 (Controlled Variables & Provenance)

- **基准代码提交 (Source Code Commit)**: `f088ed735d196311aabd9479c5b6a0849e0e670d` (Tree Clean: `True`)
- **真实远程跟踪 SHA (True Remote SHA)**: `f088ed735d196311aabd9479c5b6a0849e0e670d`
- **基准特征集 (Feature Set)**: 严格相同 79 因子 (Feature Hash = `82dfd3e9643ae1352829e736b9c8b89d1d648b98d16ef59153f261bf7a453460`)
- **数据集哈希 (Dataset SHA256)**: `9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42`
- **共同训练准入池 (Common Train Pool)**: 严格相同 (194,854 行, Hash = `bfff9a2d0a9b52a0d4924ea36d643923d1e42ab6bd48ed60017d020d22c42bcb`)
- **唯一自变量 (Primary Independent Variable)**: **模型学习目标函数 (Model Objective)**
  - **Arm A (Classification Baseline)**: Binary Logloss Classification (复现 Phase 2.1-A Execution Arm)
  - **Arm B (Continuous Regression)**: L2 / RMSE Regression on continuous `label_net_alpha_20d`
  - **Arm C (True LambdaRank)**: LightGBM LambdaRank on 10 relevance grades (0..9) with NDCG@30

---

## 2. 基准复现门禁核验 (Classification Baseline Reproduction)

- **Phase 2.1-A 预期 RankIC**: `0.043682`
- **Phase 2.1-B 实际 RankIC**: `0.043682` (Diff: `0.00e+00`)
- **OOS 评价池哈希匹配**: **`True`**
- **复现门禁状态**: **`PASS`**

---

## 3. 三臂核心实证对比 (Three-Arm Evaluation Results)

| 评价指标 | Arm A (Classification) | Arm B (Regression) | Arm C (LambdaRank) | Delta (Reg - Clf) | Delta (Rank - Clf) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Daily OOS RankIC** | **0.043682** | **0.046707** | **0.036246** | **+0.003026** | **-0.007435** |
| **NW20 RankICIR (年化)** | **0.352015** | **0.450744** | **0.305322** | **+0.098729** | **-0.046692** |
| **RankIC > 0 占比** | 60.03% | 70.12% | 59.49% | +10.09% | -0.54% |
| **Q5-Q1 年化超额 (pct pts)** | 5.16 pts | 15.38 pts | 8.30 pts | +10.22 pts | +3.14 pts |
| **Top 10% 20日平均净超额** | 0.5054% | 1.5314% | 0.1586% | +1.0261% | -0.3468% |
| **分组单调性得分** | 0.4000 | 0.9000 | 0.4000 | +0.5000 | +0.0000 |

---

## 4. 统计检验与 Fold 胜率 (Paired Block Bootstrap & Fold Stability)

- **Regression vs Classification**:
  - Mean RankIC Delta: **`+0.003026`**
  - 95% 置信区间: `[-0.024592, +0.032127]`
  - 97.5% 置信区间 (保守门禁): `[-0.028369, +0.035591]`
  - 提升概率 P(Delta > 0): **`61.15%`**
  - 有效 Fold 胜率: **`36.84%`** (7/19)
  - 达到 +0.0020 实效门禁: **`True`**
  - 满足全部 Robust Gate: **`False`**

- **LambdaRank vs Classification**:
  - Mean RankIC Delta: **`-0.007435`**
  - 95% 置信区间: `[-0.025563, +0.012254]`
  - 97.5% 置信区间 (保守门禁): `[-0.028553, +0.014843]`
  - 提升概率 P(Delta > 0): **`24.80%`**
  - 有效 Fold 胜率: **`47.37%`** (9/19)
  - 达到 +0.0020 实效门禁: **`False`**
  - 满足全部 Robust Gate: **`False`**

---

## 5. 生产模型隔离与大文件治理

- **生产模型文件**: `saved_models/latest_lightgbm.pkl` (SHA256: `93575e2dfd644ee2701ef50376ac612537fb52c07f9c119379bd3b17bea74900`)
- **生产模型 SHA 未修改**: **`True`**
- **生产目录无意外变动**: **`True`**
- **大文件存储模式**: `common_objective_oos.parquet` 与各 Arm 实验模型均采用 `local_not_git_tracked` 存储。

---

## 6. 科学判定与结论说明 (Scientific Verdict)

- **判定状态**: **`MIXED_EVIDENCE`**
