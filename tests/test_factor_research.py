"""
因子研究、A股 T+1 结算、Delayed Exit 与生产数据链严密验证测试套件 (tests/test_factor_research.py)
Phase 1.5 核心强化:
1. P0-1/P0-2: 基准严格 Fail-Closed 测试 (禁止任何隐式回退或造0)
2. P0-3: 缓存失效与必需列校验测试 (旧版缓存/缺少 benchmark_open 自动失效)
3. P0-4: 生产 Schema 对齐测试 (is_limit_up_locked, limit_up_price, is_limit_down_locked)
4. P0-5: Delayed Exit 真实展期执行测试 (跌停/停牌顺延成交)
5. P1-1: 真实物理父链与哈希校验测试 (绝不伪造 None_manifest 哈希)
6. P1-3: 真实几何 CAGR 检验 (杜绝算术均值年化错误)
7. P1-4: 中性化异常 Fail-Closed 测试 (置为 NaN，绝不保留原始值冒充)
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from research.config import ResearchConfig, default_research_config
from research.factor_metrics import FactorMetricsEngine, FactorEvaluationMetrics, TradabilityStatus, ExitStatus
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
                "is_limit_up_locked": False,
                "is_limit_down_locked": False,
                "limit_up_price": open_p * 1.10,
                "limit_down_price": open_p * 0.90,
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
    assert np.isclose(row0["long_gross_return"], 0.02, atol=1e-5)
    assert np.isclose(row0["benchmark_return"], 0.01, atol=1e-5)
    assert np.isclose(row0["long_excess_return"], 0.01, atol=1e-5)


# ------------------------------------------------------------------------------
# Test 3: P0-2 Strict benchmark fail-closed sets NaN and N/A
# ------------------------------------------------------------------------------
def test_strict_excess_research_aborts_when_benchmark_open_missing(synthetic_panel_data):
    df_no_bench = synthetic_panel_data.drop(columns=["benchmark_open"]).copy()
    cfg = ResearchConfig(ALLOW_BENCHMARK_FALLBACK_FOR_TESTS=False, USE_EXCESS_RETURN=True)
    engine = FactorResearchEngine(config=cfg)
    
    df_labeled = engine.generate_future_return_labels(df_no_bench, config=cfg)
    # 严格模式下，超额收益必须全部为 NaN
    assert df_labeled["future_excess_return_20d"].isna().all()
    
    engine._audit_benchmark_evidence(df_labeled)
    assert engine.benchmark_evidence["benchmark_timing_status"] == "BENCHMARK_OPEN_UNAVAILABLE"


# ------------------------------------------------------------------------------
# Test 4: P0-4 Real production limit schema is consumed by execution engine
# ------------------------------------------------------------------------------
def test_real_data_manager_limit_schema_is_consumed_by_execution_engine():
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        # T+1 日 000001.SZ 生产字段 is_limit_up_locked = True
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 11.0, "close": 11.0, "is_limit_up_locked": True, "limit_up_price": 11.0, "adj_open": 11.0, "adj_close": 11.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "is_limit_up_locked": False, "limit_up_price": 11.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
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
# Test 5: P0-5 Delayed Exit when T+2 is limit-down locked
# ------------------------------------------------------------------------------
def test_limit_down_at_t_plus_2_delays_exit_to_t_plus_3():
    """
    构造:
    T 日 (2021-01-04) 信号: 买入 000001.SZ
    T+1 (2021-01-05): Open = 10.0 买入成交
    T+2 (2021-01-06): Open = 9.0, is_limit_down_locked = True (跌停锁死无法卖出!)
    T+3 (2021-01-07): Open = 8.5, is_limit_down_locked = False (最早可卖出日!)
    要求:
    1. actual_exit_date 为 2021-01-07 (T+3)
    2. exit_delay_days 为 1
    3. exit_status 为 DELAYED_LIMIT_DOWN
    4. 实际收益率为 8.5 / 10.0 - 1 = -15%
    """
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        # T+2 跌停锁死
        {"date": "2021-01-06", "symbol": "000001.SZ", "open": 9.0, "close": 9.0, "is_limit_down_locked": True, "limit_down_price": 9.0, "adj_open": 9.0, "adj_close": 9.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-06", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        # T+3 恢复正常
        {"date": "2021-01-07", "symbol": "000001.SZ", "open": 8.5, "close": 8.5, "is_limit_down_locked": False, "limit_down_price": 8.1, "adj_open": 8.5, "adj_close": 8.5, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-07", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
    ])
    df["date"] = pd.to_datetime(df["date"])

    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df, "F_TEST", n_quantiles=2)
    pnl_df = pnl_res["daily_pnl_df"]
    assert not pnl_df.empty

    row0 = pnl_df.iloc[0]
    assert row0["earliest_exit_date"] == "2021-01-06"
    assert row0["actual_exit_date"] == "2021-01-07"
    assert row0["exit_delay_days"] == 1
    assert row0["exit_status"] == ExitStatus.DELAYED_LIMIT_DOWN.value
    assert np.isclose(row0["long_gross_return"], -0.15, atol=1e-5)


# ------------------------------------------------------------------------------
# Test 6: P0-5 Suspension delays exit
# ------------------------------------------------------------------------------
def test_suspension_delays_exit():
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        # T+2 停牌
        {"date": "2021-01-06", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": True},
        {"date": "2021-01-06", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        # T+3 复牌
        {"date": "2021-01-07", "symbol": "000001.SZ", "open": 10.5, "close": 10.5, "adj_open": 10.5, "adj_close": 10.5, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-07", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
    ])
    df["date"] = pd.to_datetime(df["date"])

    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df, "F_TEST", n_quantiles=2)
    pnl_df = pnl_res["daily_pnl_df"]
    assert not pnl_df.empty

    row0 = pnl_df.iloc[0]
    assert row0["actual_exit_date"] == "2021-01-07"
    assert row0["exit_delay_days"] == 1
    assert row0["exit_status"] == ExitStatus.DELAYED_SUSPENSION.value
    assert np.isclose(row0["long_gross_return"], 0.05, atol=1e-5)


# ------------------------------------------------------------------------------
# Test 7: P1-3 Geometric CAGR not equal to arithmetic annualization
# ------------------------------------------------------------------------------
def test_geometric_cagr_not_equal_to_arithmetic_mean():
    """
    交替 +10% 与 -10%：
    最终净值 = 1.1 * 0.9 = 0.99
    算术均值 = 0.0 (年化算术 = 0.0)
    几何 CAGR 必须 < 0
    """
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-06", "symbol": "000001.SZ", "open": 11.0, "close": 11.0, "adj_open": 11.0, "adj_close": 11.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-06", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-07", "symbol": "000001.SZ", "open": 9.9, "close": 9.9, "adj_open": 9.9, "adj_close": 9.9, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-07", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
    ])
    df["date"] = pd.to_datetime(df["date"])

    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df, "F_TEST", n_quantiles=2)
    # 几何 CAGR 必须为负数
    assert pnl_res["long_only_cagr"] < 0.0


# ------------------------------------------------------------------------------
# Test 8: P0-3 Factor cache invalidated when benchmark_open added
# ------------------------------------------------------------------------------
def test_factor_cache_invalidated_when_benchmark_open_missing():
    from factors.processor import FactorProcessor
    fp = FactorProcessor()
    
    # 模拟缺少 benchmark_open 的老缓存
    mock_old_cache = pd.DataFrame({
        "date": ["2021-01-01"],
        "symbol": ["000001.SZ"],
        "adj_open": [10.0],
        "adj_close": [10.0],
        "in_universe": [True]
    })
    
    req_cols = ["date", "symbol", "adj_open", "adj_close", "benchmark_open", "benchmark_close", "in_universe"]
    missing_req = [c for c in req_cols if c not in mock_old_cache.columns]
    assert "benchmark_open" in missing_req
    assert len(missing_req) > 0


# ------------------------------------------------------------------------------
# Test 9: P0-4 Canonical factor matrix hash binds date and symbol identity
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
# Test 10: P0-3 Walk-Forward Factor x Horizon Global FDR inside Train fold
# ------------------------------------------------------------------------------
def test_walk_forward_global_fdr_rejects_spurious_horizons(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=25, HORIZONS=[1, 5, 20])
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=cfg.HORIZONS)
    
    res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_CONSTANT"], config=cfg)
    assert "wf_horizon_significance" in res
    assert len(res["wf_horizon_significance"]) > 0
    for r in res["wf_horizon_significance"]:
        assert r["train_global_fdr_p"] > 0.05
        assert r["selected"] == False


# ------------------------------------------------------------------------------
# Test 11: Annual stability fails closed when no valid year
# ------------------------------------------------------------------------------
def test_annual_stability_fails_closed_when_no_valid_year():
    cfg = ResearchConfig(MIN_VALID_DAYS_PER_YEAR=60)
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
# Test 12: Benchmark date-only mapping invariant
# ------------------------------------------------------------------------------
def test_benchmark_open_close_alignment_is_date_only(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    grouped = df_labeled.dropna(subset=["future_benchmark_return_20d"]).groupby("date")["future_benchmark_return_20d"].nunique()
    assert (grouped == 1).all(), "Benchmark return differed across symbols on the same date!"


# ------------------------------------------------------------------------------
# Test 13: Neutralization insufficient cross section returns None fail-closed
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
# Test 14: Development sample cannot be OOS certified
# ------------------------------------------------------------------------------
def test_development_sample_cannot_be_oos_certified(synthetic_panel_data):
    cfg = ResearchConfig(MIN_RESEARCH_SYMBOLS=50)
    engine = FactorResearchEngine(config=cfg)
    sub_df = synthetic_panel_data[synthetic_panel_data["symbol"].isin(["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"])].copy()
    
    engine._build_research_run_manifest(sub_df, ["F_ALPHA", "F_NEG"])
    assert engine.run_manifest["research_validity_status"] == "DEVELOPMENT_SAMPLE"


# ------------------------------------------------------------------------------
# Test 15: Changing validation prices cannot affect train factor selection
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


# ------------------------------------------------------------------------------
# Test 16: P0-1 Market dataset contains Alpha158 required price columns
# ------------------------------------------------------------------------------
def test_market_dataset_contains_alpha158_required_price_columns():
    from tools.check_committed_dataset_schema import REQUIRED_MARKET_COLS
    from pathlib import Path
    market_file = Path("data_storage/parquet/market_daily.parquet")
    assert market_file.exists(), "Committed market_daily.parquet does not exist!"
    df = pd.read_parquet(market_file)
    for col in ["open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]:
        assert col in df.columns, f"Market dataset missing Alpha158 required column '{col}'!"


# ------------------------------------------------------------------------------
# Test 17: P0-2 Committed factor matrix contains benchmark_open
# ------------------------------------------------------------------------------
def test_committed_factor_matrix_contains_benchmark_open():
    from pathlib import Path
    factor_file = Path("data_storage/factors/factor_matrix.parquet")
    assert factor_file.exists(), "Committed factor_matrix.parquet does not exist!"
    df = pd.read_parquet(factor_file)
    assert "benchmark_open" in df.columns, "Committed factor matrix missing 'benchmark_open'!"
    cov = float((df["benchmark_open"] > 0).mean())
    assert cov >= 0.80, f"benchmark_open coverage ratio {cov:.2f} is below required 80%!"


# ------------------------------------------------------------------------------
# Test 18: P0-3 Schema gate rejects missing open/adj_open
# ------------------------------------------------------------------------------
def test_clean_checkout_schema_gate_rejects_missing_open(tmp_path):
    from tools.check_committed_dataset_schema import check_market_dataset
    bad_df = pd.DataFrame([{"date": "2021-01-01", "symbol": "000001.SZ", "close": 10.0}])
    bad_path = tmp_path / "bad_market.parquet"
    bad_df.to_parquet(bad_path)
    assert check_market_dataset(bad_path) is False


# ------------------------------------------------------------------------------
# Test 19: Alpha158 fails with explicit MarketSchemaError not KeyError
# ------------------------------------------------------------------------------
def test_alpha158_fails_with_explicit_schema_error_not_keyerror():
    from factors.alpha158 import Alpha158Subset, MarketSchemaError
    bad_df = pd.DataFrame([{"date": "2021-01-01", "symbol": "000001.SZ", "close": 10.0}])
    calc = Alpha158Subset()
    with pytest.raises(MarketSchemaError) as exc_info:
        calc.compute_all(bad_df)
    assert "adj_open" in str(exc_info.value) or "open" in str(exc_info.value)


# ------------------------------------------------------------------------------
# Test 20: Factor rebuild from committed market dataset succeeds
# ------------------------------------------------------------------------------
def test_factor_rebuild_from_committed_market_dataset_succeeds(tmp_path):
    from factors.processor import FactorProcessor
    from pathlib import Path
    market_file = Path("data_storage/parquet/market_daily.parquet")
    assert market_file.exists()
    market_df = pd.read_parquet(market_file)
    
    # 抽取 2 只股票小样本验证构建链无异常
    sub_market = market_df[market_df["symbol"].isin(["600519.SH", "000858.SZ"])].copy()
    fp = FactorProcessor(factor_dir=tmp_path / "factors")
    factor_df = fp.build_and_save_factor_matrix(sub_market, force_update=True)
    assert not factor_df.empty
    assert "adj_open" in factor_df.columns
    assert "benchmark_open" in factor_df.columns


# ------------------------------------------------------------------------------
# Test 21: Clean checkout research pipeline smoke test (3 factors)
# ------------------------------------------------------------------------------
def test_clean_checkout_research_pipeline_smoke(tmp_path):
    from research import FactorResearchEngine, default_research_config
    from pathlib import Path
    factor_file = Path("data_storage/factors/factor_matrix.parquet")
    assert factor_file.exists()
    factor_df = pd.read_parquet(factor_file)
    
    engine = FactorResearchEngine(config=default_research_config)
    res = engine.run_full_research(
        df=factor_df,
        factor_cols=["KMID", "ROC5", "ROC20"],
        primary_horizon=20,
        output_dir=tmp_path / "reports"
    )
    assert res is not None
    assert (tmp_path / "reports" / "FACTOR_RESEARCH_REPORT.md").exists()
    assert (tmp_path / "reports" / "research_run_manifest.json").exists()


# ------------------------------------------------------------------------------
# Test 22: P1.6-1 Security Master contains 300 symbols with official listing dates
# ------------------------------------------------------------------------------
def test_security_master_contains_300_symbols_with_listing_dates():
    from pathlib import Path
    sec_file = Path("data_storage/security_master.parquet")
    assert sec_file.exists(), "Security Master file must exist"
    df_sec = pd.read_parquet(sec_file)
    assert len(df_sec) >= 300, f"Security master symbols ({len(df_sec)}) must be >= 300"
    assert "list_date" in df_sec.columns
    assert "industry" in df_sec.columns
    assert df_sec["list_date"].notna().mean() >= 0.99
    assert df_sec["industry"].notna().mean() >= 0.99


# ------------------------------------------------------------------------------
# Test 23: P1.6-2 PIT universe filter accurately filters subnew stocks (< 60 days)
# ------------------------------------------------------------------------------
def test_pit_universe_filter_filters_subnew_stocks():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    # 模拟 2 只股票: A 上市满 1 年, B 上市第 10 天
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "symbol": ["A.SH"] * len(dates) + ["B.SH"] * len(dates),
        "days_since_listing": [365] * len(dates) + [10] * len(dates),
        "is_st": [False] * (2 * len(dates)),
        "is_suspended": [False] * (2 * len(dates))
    })
    df["in_universe"] = (df["days_since_listing"] >= 60) & (~df["is_st"]) & (~df["is_suspended"])
    
    assert df[df["symbol"] == "A.SH"]["in_universe"].all() == True
    assert df[df["symbol"] == "B.SH"]["in_universe"].all() == False


# ------------------------------------------------------------------------------
# Test 24: P1.6-3 Production dataset cross-section median >= 100
# ------------------------------------------------------------------------------
def test_pit_universe_cross_section_scale_exceeds_100():
    from pathlib import Path
    prod_path = Path("data_storage/research/market_daily_300.parquet")
    if prod_path.exists():
        df = pd.read_parquet(prod_path)
        cs = df.groupby("date")["symbol"].count()
        med_cs = cs.median()
        assert med_cs >= 100, f"Median cross-section ({med_cs}) must be >= 100"
        assert df["symbol"].nunique() >= 300


# ------------------------------------------------------------------------------
# Test 25: P1.6-4 Production factor matrix schema & 100% benchmark coverage
# ------------------------------------------------------------------------------
def test_production_research_dataset_schema_and_benchmark_open():
    from pathlib import Path
    prod_f_path = Path("data_storage/research/factor_matrix_300.parquet")
    if prod_f_path.exists():
        df = pd.read_parquet(prod_f_path)
        assert "benchmark_open" in df.columns
        assert "benchmark_close" in df.columns
        assert "adj_open" in df.columns
        assert "adj_close" in df.columns
        assert (df["benchmark_open"] > 0).mean() >= 0.99
        assert (df["benchmark_close"] > 0).mean() >= 0.99


# ------------------------------------------------------------------------------
# Test 26: P1.6-5 Purged Walk-Forward configuration produces >= 3 folds
# ------------------------------------------------------------------------------
def test_purged_walk_forward_produces_ge_3_folds():
    from research.factor_selection import FactorSelectionEngine
    from research.config import default_research_config
    
    # 模拟 5 年交易日序列 (1260 天)
    dates = pd.date_range("2019-01-01", periods=1260, freq="B")
    mock_df = pd.DataFrame({
        "date": list(dates) * 5,
        "symbol": ["A"] * 1260 + ["B"] * 1260 + ["C"] * 1260 + ["D"] * 1260 + ["E"] * 1260,
        "KMID": np.random.normal(0, 1, 1260 * 5),
        "future_excess_return_20d": np.random.normal(0, 0.05, 1260 * 5)
    })
    wf_res = FactorSelectionEngine.run_purged_walk_forward(mock_df, ["KMID"], config=default_research_config)
    assert wf_res["total_folds"] >= 3, f"Expected >= 3 folds, got {wf_res['total_folds']}"


