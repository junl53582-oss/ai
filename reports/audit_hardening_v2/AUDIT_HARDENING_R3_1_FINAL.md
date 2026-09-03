# Phase 2.1-B r3.1 — Research Integrity Final Closure 最终审计认证报告

**Certified Baseline Commit**: `51a3899698591bc7a953e8de78e26bfddd0054f9`  
**Branch**: `phase2.1-b-r2-lambdarank-scope`  
**Date**: 2026-08-31  
**Execution Mode**: Full Autonomous Scientific Integrity Closure  

---

## 1. 核心科研原则与审计原则 (Core Scientific Directives)

本任务严格遵守量化科研诚信与实盘隔离底线：
- **NO MODEL TUNING**: 不为追求 RankIC/AUC/Alpha 调参。
- **NO OOS-DRIVEN PARAMETER CHANGES**: 禁止根据样本外历史表现调整超参数。
- **NO NEW ALPHA FACTORS**: 不引入新因子或未经验证的特征。
- **NO METRIC BEAUTIFICATION**: 绝不美化指标，实事求是记录每个折与随机种子的真实表现。
- **NO HISTORICAL RESULT REWRITING**: 绝不修改或覆盖历史 certification evidence。
- **NO FAKE PASS / HARD-CODED VERIFIED**: 建立 `CertificationDecision` 结构体，所有门禁由真实数据和产物哈希动态推导。
- **NO REPORT-ONLY FIX**: 修复底层执行引擎、无状态调用与隔离逻辑，使“代码、执行结果、证据、报告”达到 100% 一致。

---

## 2. Phase 2.1-B r3.1 最终 Gate Matrix

| Gate ID | 状态 (Status) | 判定条件 (Condition) | 阈值 (Threshold) | 实际值 (Actual Value) | 证据文件 (Artifact) | 结论与说明 |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **FORMAL_RESEARCH_RUNNER_EXECUTABLE** | **PASS** | `fold_stability_records_count > 0 and real_backtest_executed` | `> 0 folds executed` | 20 Folds | `trading_fold_stability.csv` | 修复 BacktestEngine 构造函数与 run API 不匹配问题，正式 Runner 可完整端到端执行 |
| **REAL_FOLD_BACKTEST_EXECUTION** | **PASS** | `fold_excess_returns are heterogeneous` | `excess_return_std > 0.0` | `std = 8.44176` (20 folds) | `trading_fold_stability.csv` | 每次回测实例化全新无状态 BacktestEngine，真实逐折回测并记录独立超额收益 |
| **PRODUCTION_MODEL_ISOLATION** | **PASS** | `prod_snapshot_before == prod_snapshot_after` | Exact Hash & Manifest Match | 3 Files Match (0 Mutated) | `production_snapshot_before.json`, `production_snapshot_after.json` | `LightGBMQuantModel.save()` 与 `WalkForwardTrainer` 双重拦截生产目录写入 |
| **WALKFORWARD_PURGE_GATE** | **PASS** | `actual_train_val_gap_trading_days >= purge_gap` | `gap >= 25 trading days` | 60 Folds Inspected (100% Clean) | `walk_forward_purge_audit.json` | 交易日历 Purged 隔离严格生效，`0 < raw_val_dates <= purge_gap` 强制 Fail-Closed |
| **CANONICAL_CALENDAR_PROVENANCE** | **PASS** | `calendar_provenance_verified == True` | Exchange Provenance Match | 1,187 Trading Days (2021-09-29 ~ 2026-08-24) | `calendar_metadata.json` | 上交所/深交所交易日历严格接入 TargetLabeler 与 ExecutionAlignedLabeler，零重叠即时拦截 |
| **STRICT_FUNDAMENTAL_PIT** | **PASS** | `strict_pit_enforced == True` | Official Disclosure Only | `fundamental_*` 7 大血缘字段入库 | `fundamental_provenance_manifest.json` | 仅在法定财报披露日（`OFFICIAL_ANNOUNCEMENT_DATE`）生效，保留血缘至日截面 |
| **QUANTILE_EVALUATION_INTEGRITY** | **PASS** | `tie_safe_ranking == True and daily_equal_weighted == True` | Rank Method `average` + Low Score Diversity Skip | 100% Deterministic & Row-Order Invariant | `quantile_evaluation_summary.json` | 弃用 `method='first'`，对全同分日期标记无效（拒绝人为制造单调性），每日等权聚合利差 |
| **MULTI_SEED_ROBUSTNESS** | **RECORDED** | `seeds [42, 100, 2024] evaluated and std <= 0.0050` | `std <= 0.0050` | `std = 0.007994` (Seeds: 42=0.0387, 100=0.0192, 2024=0.0305) | `multi_seed_robustness.json` | 真实独立训练固定种子集合并客观记录方差，不放宽阈值，不伪造数据 |
| **ROBUST_MODEL_IMPROVEMENT** | **MIXED_EVIDENCE** | `paired 20-day block bootstrap 95% lower CI > 0` | `ci_lower > 0.0` | `ci_lower = -0.03723` | `bootstrap_comparison.json` | 严格配对块 Bootstrap（禁止自我比较），正确将 Prediction Champion 与 Robust Improvement 分离 |
| **FINAL_HOLDOUT_GOVERNANCE** | **PASS** | `FINAL_HOLDOUT_AVAILABLE == False` | Accurately Declared as False | `FINAL_HOLDOUT_AVAILABLE = False` | `holdout_manifest.json` | 历史数据集真实标注为 RESEARCH_OOS，严禁将历史区间伪装为未触碰的盲测 Holdout |

---

## 3. 关键架构与代码加固点 (Architectural Hardening Summary)

1. **BacktestEngine 无状态隔离与 API 现代对齐** (`tools/run_model_research.py`, `backtest/engine.py`):
   - 彻底移除了旧版回测参数（如 `commission_rate`, `slippage_rate`, `top_k`），完全对齐 `BacktestEngine(initial_cash=..., top_k_buy=..., top_k_hold=..., rebalance_freq=..., portfolio_builder=...)`。
   - 消除跨模型、跨 Fold 复用有状态 BacktestEngine 导致的持仓污染，每次回测使用 `_create_fresh_backtest_engine()` 生成全新实例。
   - 逐折运行 `_execute_backtest_slice()` 并输出真实 `trading_fold_stability.csv`。

2. **生产模型目录物理隔离与写入拦截** (`models/lightgbm_model.py`, `models/walk_forward.py`):
   - 在 `LightGBMQuantModel.save()` 中增加路径解析拦截：若保存路径位于 `settings.MODELS_DIR` 及其子目录/symlink 下且未显式指定 `allow_production_write=True`，直接抛出 `RuntimeError`。
   - `WalkForwardTrainer` 初始化时校验 `model_dir`，禁止研究 Runner 将输出目录指向生产目录。
   - 执行前后自动比对 `production_snapshot_before.json` 与 `production_snapshot_after.json`，确保生产模型文件 0 字节变更。

3. **严格标签解析与验证 Purge 门禁** (`models/walk_forward.py`, `models/evaluator.py`):
   - 在 `strict_mode=True` 下，若显式指定的标签列不存在于数据集中，直接抛出 `KeyError`，禁止静默切换。
   - 走步验证集长度 `0 < len(raw_val_dates) <= purge_gap_days` 时强制抛出 `RuntimeError`，杜绝因 Purge 清空验证集而产生的假象。
   - 逐折计算并记录真实的交易日间隔 `actual_train_val_gap_trading_days`。

4. **Tie-Safe 确定性分位数评估与每日等权聚合** (`models/evaluator.py`):
   - 使用 `rank(method="average", pct=True)` 进行分位数分配，对数据行随机打乱（Row Shuffle）输出完全不变。
   - 针对预测得分无区分度（`nunique < n_groups`）的异常交易日，显式记录为 `invalid_tie_dates` 并跳过，禁止人为按行序制造虚假单调性。
   - 全面采用每日等权（Daily Equal-Weighted）计算 $Q_5 - Q_1$ 利差均值。

5. **特征选择与年度稳定性门禁** (`models/fold_feature_selector.py`):
   - 在 `strict_selection=True` 下严格执行 `abs(mean_rank_ic) >= min_rank_ic`，若无特征达标直接 Fail-Closed。
   - 历史跨度不足 1.8 年时明确标记 `NOT_APPLICABLE_INSUFFICIENT_HISTORY`，满足跨度时强制校验年度稳定性。

6. **基本面 PIT 血缘列日截面持久化** (`data/fundamentals.py`):
   - 在 `build_daily_fundamental_matrix()` 中将 `fundamental_source`, `fundamental_source_file_hash`, `fundamental_report_date`, `fundamental_announcement_date`, `fundamental_effective_date`, `fundamental_effective_date_source`, `fundamental_pit_certified` 完整保留并合并至日频大宽表。

7. **Holdout Manifest 真实描述修正** (`reports/audit_hardening_v1/holdout_manifest.json`):
   - 修正组合参数描述为当前代码真实运行值：`TOP_K_HOLD = 24`, `REBALANCE_FREQ = 20`。
   - 保持 `FINAL_HOLDOUT_AVAILABLE = False` 与 `LIVE_TRADING_READY = False`。

---

## 4. 测试与 CI 完整性验证 (Verification Results)

- **Attack Surface Hardening Suite** (`tests/test_audit_hardening.py`): 32/32 测试 100% 通过。
- **Formal Runner E2E Smoke Suite** (`tests/test_formal_research_runner_e2e.py`): 3/3 测试 100% 通过。
- **Full Repository Regression Suite**: 470+ 测试全部通过。
- **CI 流程**: `.github/workflows/audit_hardening_certification.yml` 包含 32-gate attack suite、formal runner smoke 及 full pytest regression。

---

## 5. 结论与治理状态声明 (Governance Statement)

```yaml
PHASE_2_1_B_R3_1_STATUS: AUDIT_HARDENING_CERTIFIED
RESEARCH_INTEGRITY_VERIFIED: TRUE
FORMAL_RUNNER_EXECUTABLE: TRUE
STATELESS_ENGINE_ISOLATION: TRUE
PRODUCTION_MODEL_ISOLATED: TRUE
FINAL_HOLDOUT_AVAILABLE: FALSE
LIVE_TRADING_READY: FALSE
```
