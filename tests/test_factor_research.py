"""
因子研究与 Alpha 真实性严密验证测试套件 (tests/test_factor_research.py)
Phase 1.3 核心强化:
1. P0-1: 验证基准收益率数学准确性 (杜绝 benchmark_close 价格参与相减导致 -3400% 荒谬超额)
2. P0-2: 验证 1D/20D 前向超额收益标签严密一致性 (Exact Math)
3. P0-5: 验证一字涨停锁死无法买入、一字跌停锁死无法卖出
4. P0-6: 验证多空换手率与成本独立核算
5. P1-1: 验证 Factor x Horizon 全家族 Global FDR 多重检验
6. P1-3: 验证年度稳定性排除样本不足年份
7. P1-6: 验证基准按 date 单维映射杜绝跨股泄漏
8. 包含全部 15 项防作弊、时点因果律与时序硬隔离测试
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from research.config import ResearchConfig, default_research_config
from research.factor_metrics import FactorMetricsEngine, FactorEvaluationMetrics
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
# Test 1: P0-1 Long-only excess uses benchmark return not benchmark price
# ------------------------------------------------------------------------------
def test_long_only_excess_uses_benchmark_return_not_benchmark_price():
    """
    构造:
    benchmark_open[T+1] = 100, benchmark_close[T+1] = 101 -> Benchmark return = 1%
    stock open[T+1] = 10, stock close[T+1] = 10.2 -> Stock return = 2%
    要求:
    long_only_excess = 2% - 1% = 1% = 0.01 (而不是 2% - 101 = -100.98)
    """
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False, "benchmark_open": 100.0, "benchmark_close": 100.0},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False, "benchmark_open": 100.0, "benchmark_close": 100.0},
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 10.0, "close": 10.2, "adj_open": 10.0, "adj_close": 10.2, "F_TEST": 1.0, "in_universe": True, "is_suspended": False, "benchmark_open": 100.0, "benchmark_close": 101.0},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False, "benchmark_open": 100.0, "benchmark_close": 101.0},
    ])
    df["date"] = pd.to_datetime(df["date"])

    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df, "F_TEST", direction=1, n_quantiles=2)
    pnl_df = pnl_res["daily_pnl_df"]
    assert not pnl_df.empty

    row0 = pnl_df.iloc[0]
    # stock top return: (10.2 / 10.0 - 1) = 0.02
    assert np.isclose(row0["top_quantile_return"], 0.02, atol=1e-5)
    # benchmark return: (101.0 / 100.0 - 1) = 0.01
    assert np.isclose(row0["benchmark_return"], 0.01, atol=1e-5)
    # long-only excess return: 0.02 - 0.01 = 0.01
    assert np.isclose(row0["long_only_excess_return"], 0.01, atol=1e-5)
    assert not np.isclose(row0["long_only_excess_return"], -100.98, atol=1.0)


# ------------------------------------------------------------------------------
# Test 2: P0-1 1D excess return benchmark is not forced to zero
# ------------------------------------------------------------------------------
def test_1d_excess_return_benchmark_is_not_forced_to_zero(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[1])
    bench_1d = df_labeled["future_benchmark_return_1d"].dropna()
    assert not bench_1d.empty
    # 基准日内收益绝不能全为 0
    assert not (bench_1d == 0.0).all(), "1D benchmark return was wrongly forced to zero!"


# ------------------------------------------------------------------------------
# Test 3: P0-1 Stock and benchmark forward return share same entry timestamp
# ------------------------------------------------------------------------------
def test_stock_and_benchmark_forward_return_share_same_entry_timestamp():
    """验证个股与基准都从 T+1 Open 开始买入计价"""
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "benchmark_open": 4000.0, "benchmark_close": 4000.0},
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 11.0, "close": 12.0, "adj_open": 11.0, "adj_close": 12.0, "benchmark_open": 4100.0, "benchmark_close": 4200.0},
    ])
    df["date"] = pd.to_datetime(df["date"])
    df_labeled = FactorResearchEngine.generate_future_return_labels(df, horizons=[1])
    
    # 000001.SZ 在 2021-01-04 的 1D 收益应为 12/11 - 1
    # benchmark 在 2021-01-04 的 1D 收益应为 4200/4100 - 1
    row0 = df_labeled.iloc[0]
    expected_stock = (12.0 / 11.0) - 1.0
    expected_bench = (4200.0 / 4100.0) - 1.0
    expected_excess = expected_stock - expected_bench

    assert np.isclose(row0["future_return_1d"], expected_stock, atol=1e-5)
    assert np.isclose(row0["future_benchmark_return_1d"], expected_bench, atol=1e-5)
    assert np.isclose(row0["future_excess_return_1d"], expected_excess, atol=1e-5)


# ------------------------------------------------------------------------------
# Test 4: P0-2 Future excess return exact math 1D and 20D
# ------------------------------------------------------------------------------
def test_future_excess_return_exact_math_1d_and_20d():
    dates = pd.date_range("2021-01-01", periods=25, freq="B")
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date": d,
            "symbol": "000001.SZ",
            "open": 10.0 + i,
            "close": 10.5 + i,
            "adj_open": 10.0 + i,
            "adj_close": 10.5 + i,
            "benchmark_open": 100.0 + i * 2,
            "benchmark_close": 102.0 + i * 2,
            "in_universe": True,
            "is_suspended": False
        })
    df = pd.DataFrame(rows)
    df_labeled = FactorResearchEngine.generate_future_return_labels(df, horizons=[1, 20])
    
    # 手算 T=0 (2021-01-01) 的 20D 前向收益:
    # Stock entry: T+1 open = open[1] = 11.0, Stock exit: T+20 close = close[20] = 30.5
    # Stock ret = 30.5 / 11.0 - 1
    # Bench entry: T+1 open = bench_open[1] = 102.0, Bench exit: T+20 close = bench_close[20] = 142.0
    # Bench ret = 142.0 / 102.0 - 1
    expected_stock_20d = (30.5 / 11.0) - 1.0
    expected_bench_20d = (142.0 / 102.0) - 1.0
    expected_excess_20d = expected_stock_20d - expected_bench_20d

    row0 = df_labeled.iloc[0]
    assert np.isclose(row0["future_return_20d"], expected_stock_20d, atol=1e-5)
    assert np.isclose(row0["future_benchmark_return_20d"], expected_bench_20d, atol=1e-5)
    assert np.isclose(row0["future_excess_return_20d"], expected_excess_20d, atol=1e-5)


# ------------------------------------------------------------------------------
# Test 5: P0-5 Limit-up locked cannot enter long at T+1
# ------------------------------------------------------------------------------
def test_limit_up_locked_cannot_enter_long_at_t_plus_1():
    """T+1 日一字涨停锁死，多头标的无法买入"""
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        # T+1 日 000001.SZ 一字涨停锁死 (open == limit_up)
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 11.0, "close": 11.0, "high": 11.0, "low": 11.0, "limit_up": 11.0, "adj_open": 11.0, "adj_close": 11.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "high": 10.0, "low": 10.0, "limit_up": 11.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
    ])
    df["date"] = pd.to_datetime(df["date"])
    
    pnl = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df, "F_TEST", n_quantiles=2)
    # 多头标的被锁死阻断，无法构成完整多空成交
    assert pnl["daily_pnl_df"].empty or len(pnl["daily_pnl_df"]) == 0


# ------------------------------------------------------------------------------
# Test 6: P0-6 Bottom leg turnover changes cost even when top unchanged
# ------------------------------------------------------------------------------
def test_bottom_leg_turnover_changes_cost_even_when_top_unchanged():
    """
    Top 组持仓不变 (000001.SZ)，但 Bottom 组从 000002.SZ 换成 000003.SZ
    验证: short_turnover > 0，且产生独立的 short leg 交易成本
    """
    df = pd.DataFrame([
        # Day 1
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 2.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -2.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000003.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 0.0, "in_universe": True, "is_suspended": False},
        # Day 2
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 2.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 0.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000003.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -2.0, "in_universe": True, "is_suspended": False},
        # Day 3
        {"date": "2021-01-06", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 2.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-06", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 0.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-06", "symbol": "000003.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -2.0, "in_universe": True, "is_suspended": False},
    ])
    df["date"] = pd.to_datetime(df["date"])
    
    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df, "F_TEST", n_quantiles=3)
    pnl_df = pnl_res["daily_pnl_df"]
    
    row1 = pnl_df.iloc[1]
    assert row1["long_turnover"] == 0.0, "Top leg changed when it should have been identical!"
    assert row1["short_turnover"] == 1.0, "Bottom leg turnover was not captured!"
    assert row1["total_cost"] > 0.0, "Cost was zero despite 100% bottom leg replacement!"


# ------------------------------------------------------------------------------
# Test 7: P1-1 Factor x Horizon Global FDR Multi-Hypothesis Family
# ------------------------------------------------------------------------------
def test_factor_horizon_global_fdr_family(synthetic_panel_data):
    engine = FactorResearchEngine(config=ResearchConfig(HORIZONS=[1, 5, 20]))
    res = engine.run_full_research(synthetic_panel_data, factor_cols=["F_ALPHA", "F_NEG", "F_CONSTANT"], primary_horizon=20)
    
    sig_df = engine.horizon_significance_df
    assert not sig_df.empty
    # 3 factors x 3 horizons = 9 hypotheses
    assert len(sig_df) == 9
    assert "global_fdr_p" in sig_df.columns
    assert (sig_df["global_fdr_p"] >= 0.0).all() and (sig_df["global_fdr_p"] <= 1.0).all()


# ------------------------------------------------------------------------------
# Test 8: P1-3 Annual stability ignores year with insufficient valid days
# ------------------------------------------------------------------------------
def test_annual_stability_ignores_year_with_insufficient_days():
    cfg = ResearchConfig(MIN_VALID_DAYS_PER_YEAR=60)
    # 2021 年有 100 天 (VALID, IC=-0.05), 2022 年仅有 5 天 (INSUFFICIENT, IC=+0.10)
    dates_2021 = pd.date_range("2021-01-01", periods=100, freq="B")
    dates_2022 = pd.date_range("2022-01-01", periods=5, freq="B")
    
    rows = []
    for d in dates_2021:
        for s in ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]:
            s_val = int(s[5])
            rows.append({"date": d, "symbol": s, "F_TEST": s_val, "future_return_20d": -s_val * 0.01, "in_universe": True, "is_suspended": False})
    for d in dates_2022:
        for s in ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]:
            s_val = int(s[5])
            rows.append({"date": d, "symbol": s, "F_TEST": s_val, "future_return_20d": s_val * 0.01, "in_universe": True, "is_suspended": False})
            
    df = pd.DataFrame(rows)
    stab = FactorStabilityEngine.evaluate_stability(df, "F_TEST", "future_return_20d", config=cfg)
    
    assert stab.annual_details["2021"]["status"] == "VALID"
    assert stab.annual_details["2022"]["status"] == "INSUFFICIENT_YEAR_SAMPLE"
    # 2022 年不应破坏 2021 年的 100% 符号一致性
    assert stab.sign_consistency_ratio == 1.0


# ------------------------------------------------------------------------------
# Test 9: P1-6 Benchmark alignment is date-only without cross-symbol leakage
# ------------------------------------------------------------------------------
def test_benchmark_open_close_alignment_is_date_only(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    
    # 同一日期下所有股票对应的 future_benchmark_return_20d 必须严格相同
    grouped = df_labeled.dropna(subset=["future_benchmark_return_20d"]).groupby("date")["future_benchmark_return_20d"].nunique()
    assert (grouped == 1).all(), "Benchmark return leaked or differed across symbols on the same date!"


# ------------------------------------------------------------------------------
# Test 10: T close signal cannot capture Close[T] to Close[T+1] return
# ------------------------------------------------------------------------------
def test_t_close_signal_cannot_capture_close_t_to_close_t_plus_1_return():
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 20.0, "close": 21.0, "adj_open": 20.0, "adj_close": 21.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 20.0, "close": 20.0, "adj_open": 20.0, "adj_close": 20.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
    ])
    df["date"] = pd.to_datetime(df["date"])
    
    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df, factor_col="F_TEST", direction=1, n_quantiles=2)
    top_ret = float(pnl_res["daily_pnl_df"]["top_quantile_return"].iloc[0])
    assert np.isclose(top_ret, 0.05, atol=1e-4)
    assert not np.isclose(top_ret, 1.10, atol=1e-4)


# ------------------------------------------------------------------------------
# Test 11: Constant factor does not generate fake long-short alpha
# ------------------------------------------------------------------------------
def test_constant_factor_does_not_generate_fake_long_short_alpha(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    ic_s = FactorMetricsEngine.compute_daily_ic(df_labeled, "F_CONSTANT", "future_return_20d")
    assert (ic_s == 0.0).all() or ic_s.empty
    
    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(synthetic_panel_data, factor_col="F_CONSTANT", direction=1, n_quantiles=5)
    assert pnl_res["annualized_return"] == 0.0
    assert pnl_res["sharpe_ratio"] == 0.0


# ------------------------------------------------------------------------------
# Test 12: Neutralization insufficient cross section returns None not raw IC
# ------------------------------------------------------------------------------
def test_neutralization_insufficient_cross_section_returns_null_not_raw_ic(synthetic_panel_data):
    engine = FactorResearchEngine(config=ResearchConfig(MIN_NEUTRALIZATION_CROSS_SECTION=10))
    sub_df = synthetic_panel_data[synthetic_panel_data["symbol"].isin(["000001.SZ", "000002.SZ", "000003.SZ"])].copy()
    df_labeled = engine.generate_future_return_labels(sub_df, horizons=[20])

    neu_res = engine._run_real_neutralization_comparison(df_labeled, ["F_ALPHA"], "future_return_20d")
    item = neu_res["F_ALPHA"]
    
    assert item["neutralized_rank_ic"] is None
    assert item["delta_rank_ic"] is None
    assert item["status"] == "INSUFFICIENT_CROSS_SECTION"


# ------------------------------------------------------------------------------
# Test 13: Orthogonalizer insufficient samples returns None fail closed
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
# Test 14: Manifest full factor hash covers all factor columns
# ------------------------------------------------------------------------------
def test_manifest_full_factor_hash_covers_all_factor_columns(synthetic_panel_data):
    engine = FactorResearchEngine()
    factors = [f"FACTOR_{i:02d}" for i in range(20)]
    df1 = synthetic_panel_data.copy()
    for f in factors:
        df1[f] = np.random.randn(len(df1))
        
    engine._build_research_run_manifest(df1, factors)
    hash1 = engine.run_manifest["factor_matrix_hash"]
    
    df2 = df1.copy()
    df2.loc[0, "FACTOR_19"] += 999.0
    
    engine._build_research_run_manifest(df2, factors)
    hash2 = engine.run_manifest["factor_matrix_hash"]
    assert hash1 != hash2


# ------------------------------------------------------------------------------
# Test 15: Development sample cannot be OOS certified
# ------------------------------------------------------------------------------
def test_development_sample_cannot_be_oos_certified(synthetic_panel_data):
    cfg = ResearchConfig(MIN_RESEARCH_SYMBOLS=50)
    engine = FactorResearchEngine(config=cfg)
    sub_df = synthetic_panel_data[synthetic_panel_data["symbol"].isin(["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"])].copy()
    
    engine._build_research_run_manifest(sub_df, ["F_ALPHA", "F_NEG"])
    assert engine.run_manifest["research_validity_status"] == "DEVELOPMENT_SAMPLE"


# ------------------------------------------------------------------------------
# Test 16: DataFrame row permutation must not change quantile results
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
# Test 17: Train label end must not cross validation boundary
# ------------------------------------------------------------------------------
def test_train_label_end_must_not_cross_validation_boundary(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=25, HORIZONS=[1, 5, 20])
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=cfg.HORIZONS)
    
    res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_ALPHA", "F_NEG"], config=cfg)
    for f in res["folds_detail"]:
        t_end = pd.to_datetime(f["train_end"])
        v_start = pd.to_datetime(f["validation_start"])
        max_label_reach = t_end + pd.Timedelta(days=20)
        assert max_label_reach <= v_start


# ------------------------------------------------------------------------------
# Test 18: Changing validation prices cannot affect train factor selection
# ------------------------------------------------------------------------------
def test_changing_validation_prices_cannot_affect_train_factor_selection(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=20, MIN_WF_FOLDS_FOR_CERTIFICATION=1)
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[1, 5, 20])
    
    res1 = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_ALPHA", "F_NEG", "F_CONSTANT"], config=cfg)
    fold1_selected = res1["folds_detail"][0]["selected_factors"]
    
    val_start = pd.to_datetime(res1["folds_detail"][0]["validation_start"])
    df_mutated = df_labeled.copy()
    val_mask = df_mutated["date"] >= val_start
    df_mutated.loc[val_mask, "adj_close"] *= 100.0
    df_mutated.loc[val_mask, "F_ALPHA"] *= -10.0
    
    res2 = FactorSelectionEngine.run_purged_walk_forward(df_mutated, ["F_ALPHA", "F_NEG", "F_CONSTANT"], config=cfg)
    fold2_selected = res2["folds_detail"][0]["selected_factors"]
    
    assert fold1_selected == fold2_selected
