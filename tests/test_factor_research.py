"""
因子研究、A股 T+1 结算与 Alpha 证据链严密验证测试套件 (tests/test_factor_research.py)
Phase 1.4 核心强化:
1. P0-1: 验证 A 股 T+1 结算规则 (禁止日内买卖回转，Earliest Exit at T+2)
2. P0-2: 验证 Benchmark Open 数据链、Date-Level 唯一映射与缺失 Fail-Closed
3. P0-3: 验证 Walk-Forward Train Fold 内全家族 Factor x Horizon Global BH-FDR 门禁
4. P0-4: 验证 Canonical Date+Symbol 组合哈希与 Research Run Manifest 证据绑定
5. P0-5: 验证可交易性过滤 (一字涨跌停、停牌、ST) 与卖出锁死拒单记录
6. P0-6: 验证 Long-Only 纯多头策略与非对称交易摩擦成本核算
7. P1: 验证年度稳定性无有效年份 Fail-Closed 为 None、有序交易日日历索引隔离等
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from research.config import ResearchConfig, default_research_config
from research.factor_metrics import FactorMetricsEngine, FactorEvaluationMetrics, TradabilityStatus
from research.factor_decay import FactorDecayEngine
from research.factor_stability import FactorStabilityEngine, FactorStabilityResult
from research.factor_correlation import CorrelationAnalysisResult, FactorCorrelationEngine
from research.factor_selection import FactorSelectionEngine, FactorSelectionResult
from research.factor_analyzer import FactorResearchEngine


@pytest.fixture
def synthetic_panel_data():
    """生成 3 年多资产截面面板测试数据 (包含基准开盘价与收盘价)"""
    dates = pd.date_range("2021-01-01", periods=600, freq="B")
    symbols = [f"{i:06d}.SZ" for i in range(1, 21)]
    
    rows = []
    rng = np.random.default_rng(42)
    
    for d in dates:
        base_ret = rng.normal(0.0005, 0.01)
        bench_open = 4000.0 * (1.0 + rng.normal(0, 0.02))
        bench_close = bench_open * (1.0 + base_ret)
        
        for s in symbols:
            s_idx = int(s.split(".")[0])
            stock_ret = base_ret + rng.normal(0.0, 0.015)
            open_p = 10.0 * (1.0 + rng.normal(0, 0.05))
            close_p = open_p * (1.0 + stock_ret)
            
            f_alpha = stock_ret * 50.0 + rng.normal(0, 0.5)
            f_neg = -stock_ret * 50.0 + rng.normal(0, 0.5)
            f_constant = 0.0
            f_mv = 10.0 + s_idx * 0.5 + rng.normal(0, 0.1)
            f_redundant = f_alpha * 0.98 + rng.normal(0, 0.05)
            
            rows.append({
                "date": d,
                "symbol": s,
                "open": open_p,
                "close": close_p,
                "adj_open": open_p,
                "adj_close": close_p,
                "in_universe": True,
                "is_suspended": False,
                "is_st": False,
                "industry": f"IND_{s_idx % 3}",
                "LOG_CIRC_MV": f_mv,
                "F_ALPHA": f_alpha,
                "F_NEG": f_neg,
                "F_CONSTANT": f_constant,
                "F_REDUNDANT": f_redundant,
                "benchmark_open": bench_open,
                "benchmark_close": bench_close
            })
            
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ------------------------------------------------------------------------------
# Test 1: P0-1 A-share T+1 settlement cannot capture same-day intraday return
# ------------------------------------------------------------------------------
def test_a_share_t_plus_1_settlement_cannot_sell_same_day():
    """
    构造:
    T 日 (2021-01-04) 信号
    T+1 (2021-01-05): Open = 10, Close = 12 (+20% 假想日内收益)
    T+2 (2021-01-06): Open = 10 (最早可卖出开盘价)
    要求:
    真实 A 股可执行收益率不能为 +20%，按 T+2 Open 卖出结算收益率为 10/10 - 1 = 0%
    """
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 10.0, "close": 12.0, "adj_open": 10.0, "adj_close": 12.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-06", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-06", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
    ])
    df["date"] = pd.to_datetime(df["date"])

    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df, "F_TEST", direction=1, n_quantiles=2)
    pnl_df = pnl_res["daily_pnl_df"]
    assert not pnl_df.empty

    row0 = pnl_df.iloc[0]
    # 真实持仓从 T+1 Open (10) 到 T+2 Open (10)，收益为 0%
    assert np.isclose(row0["long_gross_return"], 0.0, atol=1e-5)
    assert not np.isclose(row0["long_gross_return"], 0.20, atol=1e-3)


# ------------------------------------------------------------------------------
# Test 2: P0-1 Long-only excess uses benchmark return not benchmark price
# ------------------------------------------------------------------------------
def test_long_only_excess_uses_benchmark_return_not_benchmark_price():
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False, "benchmark_open": 100.0, "benchmark_close": 100.0},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False, "benchmark_open": 100.0, "benchmark_close": 100.0},
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False, "benchmark_open": 100.0, "benchmark_close": 100.0},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False, "benchmark_open": 100.0, "benchmark_close": 100.0},
        {"date": "2021-01-06", "symbol": "000001.SZ", "open": 10.2, "close": 10.2, "adj_open": 10.2, "adj_close": 10.2, "F_TEST": 1.0, "in_universe": True, "is_suspended": False, "benchmark_open": 101.0, "benchmark_close": 101.0},
        {"date": "2021-01-06", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False, "benchmark_open": 101.0, "benchmark_close": 101.0},
    ])
    df["date"] = pd.to_datetime(df["date"])

    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df, "F_TEST", direction=1, n_quantiles=2)
    pnl_df = pnl_res["daily_pnl_df"]
    assert not pnl_df.empty

    row0 = pnl_df.iloc[0]
    # stock top return: (10.2 / 10.0 - 1) = 0.02
    assert np.isclose(row0["long_gross_return"], 0.02, atol=1e-5)
    # benchmark return: (101.0 / 100.0 - 1) = 0.01
    assert np.isclose(row0["benchmark_return"], 0.01, atol=1e-5)
    # long-only excess return: 0.02 - 0.01 = 0.01
    assert np.isclose(row0["long_excess_return"], 0.01, atol=1e-5)
    assert not np.isclose(row0["long_excess_return"], -100.98, atol=1.0)


# ------------------------------------------------------------------------------
# Test 3: P0-2 Benchmark missing blocks certification in strict mode
# ------------------------------------------------------------------------------
def test_missing_benchmark_open_fails_closed_in_strict_mode(synthetic_panel_data):
    df_no_bench = synthetic_panel_data.drop(columns=["benchmark_open"]).copy()
    cfg = ResearchConfig(ALLOW_BENCHMARK_FALLBACK_FOR_TESTS=False, USE_EXCESS_RETURN=True)
    engine = FactorResearchEngine(config=cfg)
    
    engine._audit_benchmark_evidence(df_no_bench)
    assert engine.benchmark_evidence["benchmark_timing_status"] == "BENCHMARK_OPEN_UNAVAILABLE"


# ------------------------------------------------------------------------------
# Test 4: P0-3 Walk-Forward Factor x Horizon Global FDR inside Train fold
# ------------------------------------------------------------------------------
def test_walk_forward_global_fdr_rejects_spurious_horizons(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=25, HORIZONS=[1, 5, 20])
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=cfg.HORIZONS)
    
    res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_CONSTANT"], config=cfg)
    assert "wf_horizon_significance" in res
    assert len(res["wf_horizon_significance"]) > 0
    # 常数因子全假设 global_fdr_p 应为 1.0，绝不入选
    for r in res["wf_horizon_significance"]:
        assert r["train_global_fdr_p"] > 0.05
        assert r["selected"] == False


# ------------------------------------------------------------------------------
# Test 5: P0-4 Canonical factor matrix hash binds date and symbol identity
# ------------------------------------------------------------------------------
def test_factor_matrix_hash_binds_date_symbol_identity(synthetic_panel_data):
    engine = FactorResearchEngine()
    factors = ["F_ALPHA", "F_NEG"]
    df1 = synthetic_panel_data.copy()
    
    engine._build_research_run_manifest(df1, factors)
    hash1 = engine.run_manifest["factor_matrix_hash"]
    
    # 对调第一天两只股票的因子值 (值的多重集不变，但 symbol identity 变动)
    df2 = df1.copy()
    row0_val = df2.loc[0, "F_ALPHA"]
    row1_val = df2.loc[1, "F_ALPHA"]
    df2.loc[0, "F_ALPHA"] = row1_val
    df2.loc[1, "F_ALPHA"] = row0_val
    
    engine._build_research_run_manifest(df2, factors)
    hash2 = engine.run_manifest["factor_matrix_hash"]
    
    assert hash1 != hash2, "Factor matrix hash did not change when symbol identities were swapped!"


# ------------------------------------------------------------------------------
# Test 6: P0-5 Limit-up locked cannot enter long and records trade rejection
# ------------------------------------------------------------------------------
def test_limit_up_locked_cannot_enter_long_and_records_rejection():
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        # T+1 日 000001.SZ 一字涨停锁死
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 11.0, "close": 11.0, "high": 11.0, "low": 11.0, "limit_up": 11.0, "adj_open": 11.0, "adj_close": 11.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "high": 10.0, "low": 10.0, "limit_up": 11.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-06", "symbol": "000001.SZ", "open": 11.0, "close": 11.0, "adj_open": 11.0, "adj_close": 11.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-06", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
    ])
    df["date"] = pd.to_datetime(df["date"])
    
    pnl = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df, "F_TEST", n_quantiles=2)
    assert len(pnl["trade_rejections"]) > 0
    rej = pnl["trade_rejections"][0]
    assert rej["reject_reason"] == TradabilityStatus.LIMIT_UP_LOCKED.value
    assert rej["symbol"] == "000001.SZ"


# ------------------------------------------------------------------------------
# Test 7: P0-6 Exit commission, stamp duty, and slippage applied correctly
# ------------------------------------------------------------------------------
def test_exit_stamp_duty_applied_only_to_turnover():
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-06", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-06", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
    ])
    df["date"] = pd.to_datetime(df["date"])
    
    pnl = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df, "F_TEST", n_quantiles=2)
    pnl_df = pnl["daily_pnl_df"]
    row0 = pnl_df.iloc[0]
    
    # 印花税必须为 万5 (0.0005) x turnover (1.0) = 0.0005
    assert np.isclose(row0["stamp_duty"], 0.0005, atol=1e-6)


# ------------------------------------------------------------------------------
# Test 8: P1 Annual stability fails closed when no valid year
# ------------------------------------------------------------------------------
def test_annual_stability_fails_closed_when_no_valid_year():
    cfg = ResearchConfig(MIN_VALID_DAYS_PER_YEAR=60)
    # 仅有 5 个交易日
    dates = pd.date_range("2021-01-01", periods=5, freq="B")
    rows = []
    for d in dates:
        for s in ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]:
            s_val = int(s[5])
            rows.append({"date": d, "symbol": s, "F_TEST": s_val, "future_return_20d": s_val * 0.01, "in_universe": True, "is_suspended": False})
            
    df = pd.DataFrame(rows)
    stab = FactorStabilityEngine.evaluate_stability(df, "F_TEST", "future_return_20d", config=cfg)
    
    assert stab.sign_consistency_ratio is None
    assert stab.annual_stability_status == "INSUFFICIENT_DATA"


# ------------------------------------------------------------------------------
# Test 9: P1-6 Benchmark alignment is date-only without cross-symbol leakage
# ------------------------------------------------------------------------------
def test_benchmark_open_close_alignment_is_date_only(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    grouped = df_labeled.dropna(subset=["future_benchmark_return_20d"]).groupby("date")["future_benchmark_return_20d"].nunique()
    assert (grouped == 1).all(), "Benchmark return differed across symbols on the same date!"


# ------------------------------------------------------------------------------
# Test 10: Constant factor does not generate fake long-short alpha
# ------------------------------------------------------------------------------
def test_constant_factor_does_not_generate_fake_long_short_alpha(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    ic_s = FactorMetricsEngine.compute_daily_ic(df_labeled, "F_CONSTANT", "future_return_20d")
    assert (ic_s == 0.0).all() or ic_s.empty
    
    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(synthetic_panel_data, factor_col="F_CONSTANT", direction=1, n_quantiles=5)
    assert pnl_res["long_only_cagr"] == 0.0
    assert pnl_res["long_only_sharpe"] == 0.0


# ------------------------------------------------------------------------------
# Test 11: Neutralization insufficient cross section returns None fail-closed
# ------------------------------------------------------------------------------
def test_neutralization_insufficient_cross_section_returns_null_not_raw_ic(synthetic_panel_data):
    engine = FactorResearchEngine(config=ResearchConfig(MIN_NEUTRALIZATION_CROSS_SECTION=10))
    sub_df = synthetic_panel_data[synthetic_panel_data["symbol"].isin(["000001.SZ", "000002.SZ", "000003.SZ"])].copy()
    df_labeled = engine.generate_future_return_labels(sub_df, horizons=[20])

    neu_res = engine._run_real_neutralization_comparison(df_labeled, ["F_ALPHA"], "future_return_20d")
    item = neu_res["F_ALPHA"]
    
    assert item["neutralized_rank_ic"] is None
    assert item["status"] == "INSUFFICIENT_CROSS_SECTION"


# ------------------------------------------------------------------------------
# Test 12: Orthogonalizer insufficient samples returns None fail closed
# ------------------------------------------------------------------------------
def test_orthogonalizer_insufficient_samples_returns_null_fail_closed(synthetic_panel_data):
    engine = FactorResearchEngine()
    sub_df = synthetic_panel_data[synthetic_panel_data["symbol"].isin(["000001.SZ", "000002.SZ", "000003.SZ"])].copy()
    df_labeled = engine.generate_future_return_labels(sub_df, horizons=[20])
    
    factors = ["F_ALPHA", "F_NEG", "F_CONSTANT", "F_REDUNDANT", "LOG_CIRC_MV"]
    ortho_res = engine._run_real_orthogonalization_comparison(df_labeled, factors, "future_return_20d")
    
    for f in factors[:2]:
        assert ortho_res[f]["orthogonalized_rank_ic"] is None
        assert ortho_res[f]["status"] == "INSUFFICIENT_CROSS_SECTION"


# ------------------------------------------------------------------------------
# Test 13: Development sample cannot be OOS certified
# ------------------------------------------------------------------------------
def test_development_sample_cannot_be_oos_certified(synthetic_panel_data):
    cfg = ResearchConfig(MIN_RESEARCH_SYMBOLS=50)
    engine = FactorResearchEngine(config=cfg)
    sub_df = synthetic_panel_data[synthetic_panel_data["symbol"].isin(["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"])].copy()
    
    engine._build_research_run_manifest(sub_df, ["F_ALPHA", "F_NEG"])
    assert engine.run_manifest["research_validity_status"] == "DEVELOPMENT_SAMPLE"


# ------------------------------------------------------------------------------
# Test 14: DataFrame row permutation must not change quantile results
# ------------------------------------------------------------------------------
def test_dataframe_row_permutation_must_not_change_quantile_results(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    q_orig, mono_orig, _ = FactorMetricsEngine.compute_quantile_returns(df_labeled, "F_ALPHA", "future_return_20d", n_quantiles=5)
    
    df_shuf = df_labeled.sample(frac=1.0, random_state=42).reset_index(drop=True)
    q_shuf, mono_shuf, _ = FactorMetricsEngine.compute_quantile_returns(df_shuf, "F_ALPHA", "future_return_20d", n_quantiles=5)
    
    assert mono_orig == mono_shuf
    for k in q_orig:
        assert np.isclose(q_orig[k], q_shuf[k], atol=1e-6)


# ------------------------------------------------------------------------------
# Test 15: Train label end must not cross validation boundary
# ------------------------------------------------------------------------------
def test_train_label_end_must_not_cross_validation_boundary(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=25, HORIZONS=[1, 5, 20])
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=cfg.HORIZONS)
    
    res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_ALPHA", "F_NEG"], config=cfg)
    for f in res["folds_detail"]:
        t_end = pd.to_datetime(f["train_end"])
        v_start = pd.to_datetime(f["validation_start"])
        # 严格按日历索引比较
        assert t_end < v_start


# ------------------------------------------------------------------------------
# Test 16: Changing validation prices cannot affect train factor selection
# ------------------------------------------------------------------------------
def test_changing_validation_prices_cannot_affect_train_factor_selection(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=20, MIN_WF_FOLDS_FOR_CERTIFICATION=1)
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[1, 5, 20])
    
    res1 = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_ALPHA", "F_NEG", "F_CONSTANT"], config=cfg)
    fold1_selected = res1["folds_detail"][0]["selected_factors"]
    
    val_start = pd.to_datetime(res1["folds_detail"][0]["validation_start"])
    df_mutated = df_labeled.copy()
    val_mask = df_mutated["date"] >= val_start
    df_mutated.loc[val_mask, "adj_close"] *= 10.0
    df_mutated.loc[val_mask, "F_ALPHA"] *= -10.0
    
    res2 = FactorSelectionEngine.run_purged_walk_forward(df_mutated, ["F_ALPHA", "F_NEG", "F_CONSTANT"], config=cfg)
    fold2_selected = res2["folds_detail"][0]["selected_factors"]
    
    assert fold1_selected == fold2_selected
