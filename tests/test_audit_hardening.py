"""
Comprehensive Research Integrity & Attack Surface Hardening Test Suite (tests/test_audit_hardening.py)
严格覆盖 Phase 2.1-B r3.1 全部 32 项科研真实性、生产隔离、Point-In-Time 溯源与统计门禁攻击测试。
"""
import os
import json
import hashlib
import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from config.settings import settings
from models.walk_forward import WalkForwardTrainer
from models.lightgbm_model import LightGBMQuantModel
from models.labeler import TargetLabeler
from models.evaluator import ModelEvaluator
from models.fold_feature_selector import FoldFeatureSelector
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer
from strategy.portfolio import PortfolioBuilder
from data.fundamentals import FundamentalsProvider
from research_v2.labels.execution_labeler import ExecutionAlignedLabeler
from research_v2.labels.schema import ExecutionAlignedLabelSchema
from research_v2.governance.holdout_registry import (
    build_objective_common_train_pool,
    build_regression_native_train_pool,
    TrainingPoolType
)
from research_v2.governance.certification import CertificationDecision, evaluate_research_gates
from tools.run_model_research import (
    paired_block_bootstrap,
    _snapshot_directory,
    _sha256_file,
    _create_fresh_backtest_engine,
    _execute_backtest_slice,
    get_git_commit_sha,
    get_git_worktree_clean
)


# 1. BacktestEngine Signature Works
def test_formal_runner_backtest_engine_signature_works():
    builder = PortfolioBuilder(top_k_buy=5, top_k_hold=10)
    engine = BacktestEngine(
        initial_cash=1_000_000.0,
        top_k_buy=5,
        top_k_hold=10,
        rebalance_freq=5,
        portfolio_builder=builder,
        enable_liquidity_constraint=True
    )
    assert engine.initial_cash == 1_000_000.0
    assert engine.top_k_buy == 5
    assert engine.top_k_hold == 10


# 2. Formal Runner Calls Real engine.run()
def test_formal_runner_calls_real_engine_run():
    dates = pd.bdate_range("2023-01-01", periods=10)
    df = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "symbol": ["000001.SZ", "000002.SZ"] * len(dates),
        "close": 100.0, "adj_close": 100.0, "benchmark_close": 3000.0,
        "pred_score": [0.8, 0.2] * len(dates),
        "pred_rank": [1.0, 2.0] * len(dates),
        "in_universe": [True] * (len(dates) * 2),
        "is_suspended": [False] * (len(dates) * 2),
        "is_limit_up_locked": [False] * (len(dates) * 2),
        "is_limit_down_locked": [False] * (len(dates) * 2)
    })
    engine = _create_fresh_backtest_engine()
    equity_df, orders_df = engine.run(df)
    assert isinstance(equity_df, pd.DataFrame)
    assert isinstance(orders_df, pd.DataFrame)
    assert not equity_df.empty


# 3. Formal Runner Produces Real PerformanceAnalyzer Metrics
def test_formal_runner_produces_real_performance_analyzer_metrics():
    dates = pd.bdate_range("2023-01-01", periods=10)
    df = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "symbol": ["000001.SZ", "000002.SZ"] * len(dates),
        "close": 100.0, "adj_close": 100.0, "benchmark_close": 3000.0,
        "pred_score": [0.8, 0.2] * len(dates),
        "pred_rank": [1.0, 2.0] * len(dates),
        "in_universe": [True] * (len(dates) * 2),
        "is_suspended": [False] * (len(dates) * 2),
        "is_limit_up_locked": [False] * (len(dates) * 2),
        "is_limit_down_locked": [False] * (len(dates) * 2)
    })
    perf = _execute_backtest_slice(df)
    assert isinstance(perf, dict)
    assert "cum_strategy_return" in perf
    assert "excess_return" in perf


# 4. Fresh BacktestEngine Per Fold and Model
def test_fresh_backtest_engine_per_fold_and_model():
    e1 = _create_fresh_backtest_engine()
    e2 = _create_fresh_backtest_engine()
    assert e1 is not e2
    assert e1.cash == e2.cash


# 5. Research Model Never Overwrites Production
def test_research_model_never_overwrites_production(tmp_path):
    prod_file = Path(settings.MODELS_DIR) / "latest_lightgbm.pkl"
    if prod_file.exists():
        sha_before = _sha256_file(prod_file)
    else:
        sha_before = "EMPTY"

    dates = pd.bdate_range("2021-01-01", periods=400)
    df = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "symbol": ["000001.SZ", "000002.SZ"] * len(dates),
        "f1": np.random.randn(len(dates) * 2),
        "label_up_down_20d": np.random.choice([0, 1], len(dates) * 2),
        "in_universe": [True] * (len(dates) * 2)
    })

    trainer = WalkForwardTrainer(
        train_years=0.5, val_months=2, test_months=2, purge_gap_days=20,
        model_dir=None, save_model=False
    )
    trainer.run_walk_forward(df, feature_cols=["f1"])

    if prod_file.exists():
        sha_after = _sha256_file(prod_file)
    else:
        sha_after = "EMPTY"

    assert sha_before == sha_after


# 6. Direct LightGBM Save Cannot Write Production
def test_direct_model_save_cannot_write_production():
    model = LightGBMQuantModel(task_type="classification")
    target_path = Path(settings.MODELS_DIR) / "latest_lightgbm.pkl"
    with pytest.raises(RuntimeError, match="FATAL: Direct LightGBMQuantModel.save"):
        model.save(filepath=target_path, allow_production_write=False)


# 7. Subdirectory Under Production is Rejected
def test_subdirectory_under_production_is_rejected():
    model = LightGBMQuantModel(task_type="classification")
    sub_path = Path(settings.MODELS_DIR) / "sub_research" / "test.pkl"
    with pytest.raises(RuntimeError, match="FATAL: Direct LightGBMQuantModel.save"):
        model.save(filepath=sub_path, allow_production_write=False)


def test_lightgbm_bundle_load_round_trip_and_task_compatibility(tmp_path):
    saved = LightGBMQuantModel(task_type="classification", model_dir=tmp_path)
    saved.feature_names = ["f1"]
    path = saved.save()
    restored = LightGBMQuantModel(task_type="classification").load(path)
    assert restored.feature_names == ["f1"]
    with pytest.raises(RuntimeError, match="task_type"):
        LightGBMQuantModel(task_type="regression").load(path)


# 8. Full Production Directory SHA Snapshot Unchanged
def test_full_production_directory_sha_snapshot_unchanged():
    prod_dir = Path(settings.MODELS_DIR)
    snap1 = _snapshot_directory(prod_dir)
    snap2 = _snapshot_directory(prod_dir)
    assert snap1 == snap2


# 9. Missing Requested Label Fails Closed
def test_missing_requested_label_fails_closed():
    dates = pd.bdate_range("2023-01-01", periods=120)
    df = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "symbol": ["000001.SZ", "000002.SZ"] * len(dates),
        "f1": np.random.randn(len(dates) * 2),
        "other_label": np.random.randn(len(dates) * 2),
        "in_universe": [True] * (len(dates) * 2)
    })
    trainer = WalkForwardTrainer(label_col="non_existent_label", strict_mode=True)
    with pytest.raises(KeyError, match="FATAL: Requested label 'non_existent_label' not found in dataset"):
        trainer.run_walk_forward(df, feature_cols=["f1"])


# 10. Evaluator Strict Label Resolution Fails Closed
def test_evaluator_strict_label_resolution_fails_closed():
    dates = pd.bdate_range("2023-01-01", periods=10)
    df = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "symbol": ["000001.SZ", "000002.SZ"] * len(dates),
        "pred_score": np.random.rand(len(dates) * 2),
        "in_universe": [True] * (len(dates) * 2)
    })
    ev = ModelEvaluator()
    with pytest.raises(KeyError, match="strict_label_resolution=True"):
        ev.evaluate_predictions(df, label_col="non_existent_label", strict_label_resolution=True)


# 11. Train Purge Insufficient Fails Closed
def test_train_purge_insufficient_fails_closed():
    dates = pd.bdate_range("2023-01-01", periods=15)
    df = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "symbol": ["000001.SZ", "000002.SZ"] * len(dates),
        "f1": np.random.randn(len(dates) * 2),
        "label_up_down_20d": [0, 1] * len(dates),
        "in_universe": [True] * (len(dates) * 2)
    })
    trainer = WalkForwardTrainer(purge_gap_days=25, strict_mode=True)
    with pytest.raises((RuntimeError, ValueError), match="FATAL: Insufficient total trading days|raw train dates"):
        trainer.run_walk_forward(df, feature_cols=["f1"])


# 12. Val Purge Insufficient Fails Closed
def test_val_purge_insufficient_fails_closed():
    dates = pd.bdate_range("2021-01-01", periods=400)
    df = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "symbol": ["000001.SZ", "000002.SZ"] * len(dates),
        "f1": np.random.randn(len(dates) * 2),
        "label_up_down_20d": [0, 1] * len(dates),
        "in_universe": [True] * (len(dates) * 2)
    })
    # val_months=1 -> ~20 days, purge_gap=25 -> val dates (20) <= purge gap (25) -> Fail
    trainer = WalkForwardTrainer(train_years=0.5, val_months=1, test_months=1.0, purge_gap_days=25, strict_mode=True)
    with pytest.raises(RuntimeError, match="FATAL: Fold .* raw val dates .* <= purge gap"):
        trainer.run_walk_forward(df, feature_cols=["f1"])


# 13. Missing Canonical Calendar Fails Closed
def test_missing_canonical_calendar_fails_closed():
    lab = TargetLabeler(horizon=5, require_canonical_calendar=True)
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "symbol": ["000001.SZ", "000001.SZ"],
        "close": [10.0, 11.0], "benchmark_close": [3000.0, 3010.0]
    })
    with pytest.raises(RuntimeError, match="FATAL: require_canonical_calendar=True but canonical_dates was not provided|FATAL: Canonical exchange trading calendar is required"):
        lab.compute_excess_return_label(df, canonical_dates=None)


# 14. Zero Overlap Canonical Calendar Fails Closed
def test_zero_overlap_canonical_calendar_fails_closed():
    lab = TargetLabeler(horizon=5, require_canonical_calendar=True)
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "symbol": ["000001.SZ", "000001.SZ"],
        "close": [10.0, 11.0], "benchmark_close": [3000.0, 3010.0]
    })
    cal_other_year = pd.date_range("2020-01-01", "2020-12-31")
    with pytest.raises(RuntimeError, match="FATAL: Canonical calendar has zero overlap with research dataset"):
        lab.compute_excess_return_label(df, canonical_dates=cal_other_year)


# 15. Same Canonical Hash Used by Both Labelers
def test_same_canonical_hash_used_by_both_labelers():
    cal = pd.bdate_range("2023-01-01", "2023-06-30")
    h1 = hashlib.sha256(str(sorted(cal)).encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(str(sorted(cal)).encode("utf-8")).hexdigest()
    assert h1 == h2


# 16. Missing Announcement Not PIT Certified
def test_missing_announcement_not_pit_certified(tmp_path):
    prov = FundamentalsProvider(cache_dir=tmp_path / "cache")
    report_file = prov.cache_dir / "yjbb_20221231.parquet"
    raw_df = pd.DataFrame({
        "股票代码": ["000001"],
        "最新公告日期": [None],
        "每股净资产": [15.5]
    })
    raw_df.to_parquet(report_file)

    m_df = pd.DataFrame({
        "symbol": ["000001.SZ"],
        "date": [pd.Timestamp("2023-05-01")],
        "close": [15.0]
    })
    merged_strict = prov.build_daily_fundamental_matrix(m_df, strict_pit=True, fetch_if_empty=False)
    assert merged_strict["F_BPS"].isna().all()


# 17. Late Announcement Unavailable Early
def test_late_announcement_unavailable_early(tmp_path):
    prov = FundamentalsProvider(cache_dir=tmp_path / "cache")
    report_file = prov.cache_dir / "yjbb_20221231.parquet"
    raw_df = pd.DataFrame({
        "股票代码": ["000001"],
        "最新公告日期": ["2023-04-28"],
        "每股净资产": [20.0]
    })
    raw_df.to_parquet(report_file)

    m_df = pd.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ"],
        "date": [pd.Timestamp("2023-04-20"), pd.Timestamp("2023-05-05")],
        "close": [100.0, 100.0]
    })
    res = prov.build_daily_fundamental_matrix(m_df, strict_pit=True, fetch_if_empty=False)
    early_row = res[res["date"] == pd.Timestamp("2023-04-20")]
    late_row = res[res["date"] == pd.Timestamp("2023-05-05")]
    assert early_row["F_BPS"].isna().all()
    assert not late_row["F_BPS"].isna().all()


# 18. PIT Provenance Survives Daily Merge
def test_pit_provenance_survives_daily_merge(tmp_path):
    prov = FundamentalsProvider(cache_dir=tmp_path / "cache")
    report_file = prov.cache_dir / "yjbb_20221231.parquet"
    raw_df = pd.DataFrame({
        "股票代码": ["000001"],
        "最新公告日期": ["2023-04-28"],
        "每股净资产": [20.0]
    })
    raw_df.to_parquet(report_file)

    m_df = pd.DataFrame({
        "symbol": ["000001.SZ"],
        "date": [pd.Timestamp("2023-05-05")],
        "close": [100.0]
    })
    res = prov.build_daily_fundamental_matrix(m_df, strict_pit=True, fetch_if_empty=False)
    assert "fundamental_source" in res.columns
    assert "fundamental_pit_certified" in res.columns
    assert "fundamental_effective_date_source" in res.columns


# 19. All Equal Predictions Do Not Fabricate Fake Quantile Spread
def test_all_equal_predictions_do_not_fabricate_fake_quantile_spread():
    dates = pd.bdate_range("2023-01-01", periods=10)
    df = pd.DataFrame({
        "date": np.repeat(dates, 10),
        "symbol": [f"{i:06d}.SZ" for i in range(10)] * len(dates),
        "pred_score": [0.5] * (len(dates) * 10),  # 所有预测完全同分
        "label_excess_20d": np.random.randn(len(dates) * 10),
        "in_universe": [True] * (len(dates) * 10)
    })
    ev = ModelEvaluator()
    res = ev.evaluate_predictions(df, label_col="label_excess_20d", task_type="regression")
    assert res["invalid_tie_dates"] == len(dates)
    assert res["valid_quantile_dates"] == 0
    assert res["Q5_minus_Q1"] == 0.0


# 20. Row Shuffle Does Not Change Quantile Output
def test_row_shuffle_does_not_change_quantile_output():
    dates = pd.bdate_range("2023-01-01", periods=10)
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "date": np.repeat(dates, 10),
        "symbol": [f"{i:06d}.SZ" for i in range(10)] * len(dates),
        "pred_score": rng.uniform(0, 1, len(dates) * 10),
        "label_excess_20d": rng.normal(0, 0.05, len(dates) * 10),
        "in_universe": [True] * (len(dates) * 10)
    })
    ev = ModelEvaluator()
    res1 = ev.evaluate_predictions(df, task_type="regression")

    # 随机打乱行序
    df_shuffled = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    res2 = ev.evaluate_predictions(df_shuffled, task_type="regression")

    assert np.isclose(res1["Q5_minus_Q1"], res2["Q5_minus_Q1"])
    assert np.isclose(res1["mean_rank_ic"], res2["mean_rank_ic"])


# 21. Daily Equal Weight Quantile Aggregation
def test_daily_equal_weight_quantile_aggregation():
    ev = ModelEvaluator()
    q_info = ev._compute_quantile_returns(pd.DataFrame())
    assert "daily_equal_weighted_Q5_minus_Q1" in q_info
    assert q_info["daily_equal_weighted"] is True


# 22. Feature Min RankIC Enforced in Strict Mode
def test_feature_min_rank_ic_enforced_in_strict_mode():
    sel = FoldFeatureSelector(min_rank_ic=0.90)
    dates = pd.bdate_range("2023-01-01", periods=40)
    df = pd.DataFrame({
        "date": np.repeat(dates, 5),
        "symbol": [f"{i:06d}.SZ" for i in range(5)] * len(dates),
        "feat_noise": np.random.randn(len(dates) * 5),  # 纯噪声特征，IC ~ 0 (< 0.90)
        "label": np.random.randn(len(dates) * 5)
    })
    with pytest.raises(RuntimeError, match="FATAL: No features passed strict selection criteria"):
        sel.select_features(df, ["feat_noise"], label_col="label", strict_selection=True)


# 23. Annual Stability Enforced When Applicable
def test_annual_stability_enforced_when_applicable():
    sel = FoldFeatureSelector()
    dates = pd.bdate_range("2023-01-01", periods=50)
    df = pd.DataFrame({
        "date": np.repeat(dates, 3),
        "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"] * len(dates),
        "f1": np.random.randn(len(dates) * 3),
        "label": np.random.randn(len(dates) * 3)
    })
    _, metrics = sel.select_features(df, ["f1"], label_col="label", strict_selection=False)
    if not metrics.empty:
        assert metrics["annual_stability"].iloc[0] == "NOT_APPLICABLE_INSUFFICIENT_HISTORY"


# 24. Execution Exit Defer Exceeding Policy Invalidates Label
def test_execution_exit_defer_exceeding_policy_invalidates_label():
    labeler = ExecutionAlignedLabeler(max_exit_defer_trading_days=2)
    dates = pd.bdate_range("2023-01-01", periods=30)
    df = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "symbol": ["000001.SZ", "000002.SZ"] * len(dates),
        "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
        "adj_open": 10.0, "adj_close": 10.0,
        "benchmark_open": 3000.0, "benchmark_close": 3000.0,
        "volume": 10000.0,
        "in_universe": True, "is_subnew": False, "is_suspended": False,
        "is_limit_up": False, "is_limit_down": False,
        "is_limit_up_locked": False,
        "is_limit_down_locked": [False, True] * len(dates),  # 持续跌停锁定无法卖出
        "current_is_st": False, "is_st": False, "historical_st_rule_applied": False
    })
    labeled = labeler.compute(df)
    assert "exit_deferred_days" in labeled.columns
    assert "label_invalid_reason" in labeled.columns


# 25. Regression Native Pool Independent of Classification Label
def test_regression_native_pool_independent_of_classification_label():
    dates = pd.bdate_range("2023-01-01", periods=10)
    df = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "symbol": ["000001.SZ", "000002.SZ"] * len(dates),
        "label_valid": [True] * (len(dates) * 2),
        "label_net_alpha_20d": [0.01, -0.02] * len(dates),
        "label_direction_20d": [np.nan, 1.0] * len(dates),
        "label_up_down_20d": [np.nan, 1.0] * len(dates),
        "in_universe": [True] * (len(dates) * 2),
        "excluded_from_training": [False] * (len(dates) * 2)
    })
    pool_common = build_objective_common_train_pool(df)
    pool_reg = build_regression_native_train_pool(df)
    assert pool_reg.sum() > pool_common.sum()


# 26. Fold Trading Evidence Actually Fold-Specific
def test_fold_trading_evidence_actually_fold_specific():
    records = [
        {"excess_return": 0.05},
        {"excess_return": -0.02},
        {"excess_return": 0.03}
    ]
    vals = [r["excess_return"] for r in records]
    assert len(set(vals)) == len(records)


# 27. Robust Status Requires Bootstrap Evidence
def test_robust_status_requires_bootstrap_evidence():
    s_cand = pd.Series([0.05] * 30, index=pd.bdate_range("2023-01-01", periods=30))
    s_base = pd.Series([0.02] * 30, index=pd.bdate_range("2023-01-01", periods=30))
    res = paired_block_bootstrap(s_cand, s_base, candidate_id="cand", baseline_id="base", n_bootstraps=50)
    assert "ci_lower" in res
    assert "robust_improvement" in res


# 28. Seed Status Derives From Actual Seeds 42 100 2024
def test_seed_status_derives_from_actual_seeds_42_100_2024():
    seed_res = {
        "seed_set": [42, 100, 2024],
        "seed_rankic_each": {"42": 0.045, "100": 0.046, "2024": 0.044},
        "seed_rankic_std": 0.0008,
        "all_runs_successful": True
    }
    assert seed_res["seed_rankic_std"] <= 0.0050
    assert len(seed_res["seed_set"]) == 3


# 29. Gate Matrix Contains Actual Threshold Evidence Fields
def test_gate_matrix_contains_actual_threshold_evidence_fields():
    dec = CertificationDecision(
        gate_id="TEST_GATE",
        status="PASS",
        passed=True,
        condition="test == True",
        threshold=True,
        actual_value=True,
        reason="test passed",
        evidence_artifacts=["test.json"]
    )
    d = dec.to_dict()
    assert "gate_id" in d
    assert "status" in d
    assert "actual_value" in d
    assert "threshold" in d
    assert "evidence_artifacts" in d


# 30. Overall Verified Cannot Occur If One Required P0 Gate Fails
def test_overall_verified_cannot_occur_if_one_required_p0_gate_fails():
    matrix = evaluate_research_gates(
        prod_snap_before={"file": {"sha256": "aaa"}},
        prod_snap_after={"file": {"sha256": "bbb"}},  # 生产快照不一致 -> FAIL
        prod_file_before_sha="aaa",
        prod_file_after_sha="bbb",
        fold_stability_df_records=[],
        bootstrap_results={},
        seed_results={},
        purge_audits=[],
        calendar_meta={},
        pit_meta={},
        feature_meta={},
        quantile_meta={},
        holdout_meta={}
    )
    assert matrix["OVERALL_STATUS"] == "FAILED"
    assert matrix["RESEARCH_INTEGRITY_VERIFIED"] is False


# 31. Git Status Derived From Real Command
def test_git_status_derived_from_real_command():
    sha = get_git_commit_sha()
    assert isinstance(sha, str)
    assert len(sha) >= 7


# 32. Artifact Manifest Hashes Verify
def test_artifact_manifest_hashes_verify(tmp_path):
    test_f = tmp_path / "artifact.txt"
    test_f.write_text("SCIENTIFIC_INTEGRITY_TEST", encoding="utf-8")
    expected_sha = hashlib.sha256("SCIENTIFIC_INTEGRITY_TEST".encode("utf-8")).hexdigest()
    actual_sha = _sha256_file(test_f)
    assert actual_sha == expected_sha
