# Phase 2.1-B r2 — LambdaRank Target-Scope & Evidence Closure Report
# A股模型学习目标函数严格受控三臂 OOS 研究报告 (r2 截面范围修正与证据闭环)

> **最终科学判定 (Scientific Verdict)**: **`MIXED_EVIDENCE`**
> **方法修正类型 (Method Correction)**: `lambdarank_relevance_common_train_scope`
> **实盘许可声明 (Live Trading Guard)**: `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 1. 5 层完整 Git 血缘链 (5-Tier Git Provenance)

| 阶段 / 提交层级 | Commit SHA | 说明与治理模式 |
| :--- | :--- | :--- |
| **Phase 2.1-B Initial Code** | [`40b7bb8bc5c8b297ca5e259ac54f1e6df8bc5af9`](https://github.com/junl53582-oss/ai/commit/40b7bb8bc5c8b297ca5e259ac54f1e6df8bc5af9) | 初版三臂 Runner 与单测 |
| **Phase 2.1-B Pre-Run Bugfix** | [`f088ed735d196311aabd9479c5b6a0849e0e670d`](https://github.com/junl53582-oss/ai/commit/f088ed735d196311aabd9479c5b6a0849e0e670d) | 运行前修复 float NaN 类型转换 |
| **Phase 2.1-B v1 Evidence** | [`5bb6ff21707ae484d1659c4a575645dd6e4c98ce`](https://github.com/junl53582-oss/ai/commit/5bb6ff21707ae484d1659c4a575645dd6e4c98ce) | v1 证据：确立 Regression `MIXED_EVIDENCE` 结论，定位 LambdaRank 截面范围问题 |
| **Phase 2.1-B r2 Code** | [`ae65aabdee9951b5e7bbd42049594556296b0520`](https://github.com/junl53582-oss/ai/commit/ae65aabdee9951b5e7bbd42049594556296b0520) | r2 修复：仅 common_train 样本参与分位数计算，添加极端样本隔离单测与真 Scipy 版本 |
| **Phase 2.1-B r2 Evidence** | *待 Stage B 提交* | r2 证据：双重复现门禁通过，LambdaRank r2 认证指标入库 |

---

## 2. 双重严格复现门禁核验 (Dual Reproduction Gates)

### A. Classification Baseline Reproduction
- **Phase 2.1-A 预期 RankIC**: `0.043682`
- **Phase 2.1-B r2 实际 RankIC**: `0.043682` (Diff: `0.00e+00`)
- **OOS Pool Hash 匹配**: **`True`**
- **门禁状态**: **`PASS`**

### B. Regression Reproduction Gate
- **Phase 2.1-B v1 预期 RankIC**: `0.046707`
- **Phase 2.1-B r2 实际 RankIC**: `0.046707` (Diff: `0.00e+00`)
- **预期 NW20**: `0.450744` | 实际: `0.450744`
- **预期 Top10 Alpha**: `1.5314%` | 实际: `1.5314%`
- **门禁状态**: **`PASS`**
- **Regression 科学结论维持**: **`MIXED_EVIDENCE`** (大实效提升信号，但统计显著与跨折胜率证据不足)

---

## 3. 三臂核心实证结果对比 (Three-Arm Evaluation Results)

| 评价指标 | Arm A (Classification) | Arm B (Regression) | Arm C (LambdaRank r2) | Delta (Reg - Clf) | Delta (Rank - Clf) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Daily OOS RankIC** | **0.043682** | **0.046707** | **0.030628** | **+0.003026** | **-0.013054** |
| **NW20 RankICIR (年化)** | **0.352015** | **0.450744** | **0.259945** | **+0.098729** | **-0.092069** |
| **RankIC > 0 占比** | 60.03% | 70.12% | 59.08% | +10.09% | -0.94% |
| **Q5-Q1 年化超额 (pct pts)** | 5.16 pts | 15.38 pts | 3.03 pts | +10.22 pts | -2.13 pts |
| **Top 10% 20日平均净超额** | 0.5054% | 1.5314% | -0.0991% | +1.0261% | -0.6045% |
| **分组单调性得分** | 0.4000 | 0.9000 | 0.4000 | +0.5000 | +0.0000 |

---

## 4. LambdaRank Target-Scope 修正与诊断

- **Eligible 训练样本数**: `194,854`
- **Ineligible 标记样本数**: `154,525`
- **Ineligible 样本中非空相关性等级数**: **`0`** (完全隔离)
- **单日 Eligible 样本数量分布**: Min=`134`, Max=`176`, Mean=`167.1`
- **v1 LambdaRank 状态**: **`SUPERSEDED_METHOD_SCOPE_ISSUE`** (v1 结果由 r2 正式取代)

---

## 5. 生产模型物理隔离审计

- **生产模型路径**: `saved_models/latest_lightgbm.pkl` (SHA256: `93575e2dfd644ee2701ef50376ac612537fb52c07f9c119379bd3b17bea74900`)
- **生产模型 SHA 未修改**: **`True`**
- **生产目录无变动**: **`True`**
- **大文件存储模式**: `common_objective_oos.parquet` 与各 Arm 实验模型均采用 `local_not_git_tracked` 存储。
