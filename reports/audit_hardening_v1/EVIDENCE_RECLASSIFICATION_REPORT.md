# Phase 2.1-B r3 — Research Integrity Hardening & Evidence Reclassification Report
# 科研真实性、研究基础设施加固与历史证据重新分类报告

> **执行基线 (Baseline Commit)**: `ddbbf7fa92e1c9218681a1dc257cd813a0b3cd7d`
> **数据治理状态 (Data Governance Status)**: `FINAL_HOLDOUT_AVAILABLE = FALSE`
> **实盘许可状态 (Live Trading Guard)**: `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 一、历史证据治理与重新分类 (Evidence Reclassification Matrix)

| 历史产物 / 证据项 | 原有状态宣称 | 重新分类状态 (Audit Hardening v1) | 真实科研定论与治理说明 |
| :--- | :--- | :--- | :--- |
| **Phase 2.0.2 trading_fold_stability.csv** | `VERIFIED_STABLE` | **`SUPERSEDED_INVALID_EVIDENCE`** | 历史代码循环 1..20 硬编码 `POSITIVE` 并重复全局 CAGR，属于合成伪证据。保留物理文件以维持审计历史，正式认证废止。 |
| **历史 2023-2026 Walk-Forward OOS 数据集** | `FINAL_UNTOUCHED_OOS` | **`HISTORICAL_RESEARCH_OOS`** | 因历史已多次用于特征筛选 (ENABLE_REGISTRY_FACTORS) 与超参/持仓参数选择，不可再称为未触碰盲测集。 |
| **Phase 2.1-B r2 Classification & Regression 实证数值** | `CERTIFIED_BENCHMARK` | **`VALID_HISTORICAL_RESEARCH_EVIDENCE`** | 严格冻结并认证：Classification (0.043682), Regression (0.046707), Delta (+0.003026)。 |
| **Phase 2.1-B r2 Continuous Regression 稳健改进结论** | `ROBUST_CANDIDATE` | **`MIXED_EVIDENCE_NOT_ROBUST`** | 数值与经济提升显著，但配对 Bootstrap 97.5% CI 包含 0 且折胜率 (36.8%) 未达标，维持科学结论 `MIXED_EVIDENCE`。 |
| **Phase 2.1-B r2 True LambdaRank 实证结论** | `LAMBDARANK_R2` | **`VALID_NEGATIVE_RESULT`** | 严密确认横截面 Spearman RankIC 下降 (-0.013054) 属于有效负向科学结论，严禁删除。 |

---

## 二、P0 / P1 基础设施加固成果

1. **P0: 生产模型物理隔离 (Production Model Isolation Guard)**
   - `WalkForwardTrainer` 默认 `save_model=False`，禁止向 `settings.MODELS_DIR` 直接持久化。
   - 所有研究 Runner 强制使用 run-scoped 独立模型目录。
   - 正式实验前后对 `saved_models/` 进行 SHA256 与全目录快照比对，任何变动立即触发 Fail-Closed。

2. **P0: 废止虚假 Fold 交易稳定性证据**
   - 彻底删除 `candidate_excess_advantage = "POSITIVE"` 等硬编码。
   - `tools/run_model_research.py` 改造为针对每 Fold 的 OOS 预测记录调用独立 `BacktestEngine`，计算真实逐折收益与胜负。

3. **P0: 所有认证状态严格基于证据推导 (Evidence-Derived Gates)**
   - 彻底移除 `phase_2_1_ready = True`, `git_worktree_clean = True`, `seed_robustness = "VERIFIED_STABLE"` 等假象。
   - 严格基于 Git porcelain、ls-remote、Bootstrap CI、多 Seed 统计方差推导认证状态。

4. **P0: 基本面 Point-In-Time 严格化 (Strict PIT Fundamentals)**
   - 明确区分 `OFFICIAL_ANNOUNCEMENT_DATE` 与 `SYNTHETIC_DELAY_ESTIMATE`。
   - 严格 PIT 模式下仅官方公告日期允许作为财务因子暴露，严禁通过 +110 天自动变成 verified PIT。
   - 通过未来函数攻击测试 (T+150d 公告在 T+110d 绝对不可见)。

5. **P1: Walk-Forward 全面 Fail-Closed**
   - 标签缺失时直接抛出 `KeyError`，禁止静默回退。
   - Purge 窗口不足时直接抛出 `RuntimeError`，禁止无 Purge 训练。
   - 消除 Python `assert`，使用显式门禁。

6. **P1: 分位数评估确定性与跨日等权聚合 (Quantile Determinism & Daily Equal-Weighting)**
   - 采用 `rank(method='first', pct=True)` 杜绝 pd.qcut 异常坍塌至 Q1。
   - 每日等权计算各组收益再跨日求平均，杜绝大截面交易日过度支配权重。

7. **P1: 延期退出上限策略 (Max Exit Deferral Policy)**
   - 增加 `max_exit_defer_trading_days`，超出策略上限的延期样本严格置 `label_valid=False` 并标记 `EXIT_DEFER_EXCEEDS_POLICY`。

8. **P1: 训练池治理分离 (Phase 2.1-C Two Training Pools)**
   - `OBJECTIVE_COMMON_TRAIN_POOL`: 用于三臂受控公平比较。
   - `REGRESSION_NATIVE_TRAIN_POOL`: 用于未来 Phase 2.1-C 释放连续回归完整学习能力。
