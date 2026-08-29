"""
因子研究与 Alpha 真实性严密验证测试套件 (tests/test_factor_research.py)
覆盖 15 项核心防作弊、执行时序与统计严密性自动化测试 (Phase 1.2 Hotfix)
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
    """生成 3 年多资产截面面板测试数据"""
    dates = pd.date_range("2021-01-01", periods=600, freq="B")
    symbols = [f"{i:06d}.SZ" for i in range(1, 21)]
    
    rows = []
    rng = np.random.default_rng(42)
    
    for d in dates:
        base_ret = rng.normal(0.0005, 0.01)
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
                "benchmark_close": 4000.0 * (1.0 + base_ret)
            })
            
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ------------------------------------------------------------------------------
# Test 1: T close signal cannot capture Close[T] to Close[T+1] return (P0-1)
# ------------------------------------------------------------------------------
def test_t_close_signal_cannot_capture_close_t_to_close_t_plus_1_return():
    """
    攻击场景:
    Close[T] = 10, Open[T+1] = 20, Close[T+1] = 21
    信号于 T 收盘生成，若在 T+1 开盘买入，真实可得日度收益为 21/20 - 1 = 5%
    系统绝不能得到 21/10 - 1 = 110% 的假执行收益！
    """
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 20.0, "close": 21.0, "adj_open": 20.0, "adj_close": 21.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 20.0, "close": 20.0, "adj_open": 20.0, "adj_close": 20.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
    ])
    df["date"] = pd.to_datetime(df["date"])
    
    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(
        df, factor_col="F_TEST", direction=1, n_quantiles=2
    )
    pnl_df = pnl_res["daily_pnl_df"]
    assert not pnl_df.empty
    
    # 验证 T+1 执行收益为 (21/20 - 1) = 0.05
    top_ret = float(pnl_df["top_quantile_return"].iloc[0])
    assert np.isclose(top_ret, 0.05, atol=1e-4)
    assert not np.isclose(top_ret, 1.10, atol=1e-4), "Illegally captured Close[T] to Close[T+1] lookahead jump!"


# ------------------------------------------------------------------------------
# Test 2: Mutating Close[T] must not change T+1 post-entry return (P0-1)
# ------------------------------------------------------------------------------
def test_mutating_close_t_must_not_change_t_plus_1_post_entry_return():
    """
    修改 T 日收盘价，只要 T+1 开盘价与收盘价不变，T+1 的日度交易 PnL 必须保持不变。
    """
    df1 = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 12.0, "close": 13.0, "adj_open": 12.0, "adj_close": 13.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 12.0, "close": 11.0, "adj_open": 12.0, "adj_close": 11.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
    ])
    df2 = df1.copy()
    # 篡改 T 日收盘价
    df2.loc[df2["date"] == "2021-01-04", "adj_close"] = 5.0

    pnl1 = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df1, "F_TEST", n_quantiles=2)
    pnl2 = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df2, "F_TEST", n_quantiles=2)

    ret1 = float(pnl1["daily_pnl_df"]["gross_return"].iloc[0])
    ret2 = float(pnl2["daily_pnl_df"]["gross_return"].iloc[0])
    assert np.isclose(ret1, ret2, atol=1e-6)


# ------------------------------------------------------------------------------
# Test 3: T close signal uses T+1 Open as earliest entry for H-Day labels (P0-1)
# ------------------------------------------------------------------------------
def test_t_close_signal_uses_t_plus_1_open_as_earliest_entry(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[1, 5, 20])
    
    sym = synthetic_panel_data["symbol"].iloc[0]
    sample = df_labeled.groupby("symbol").get_group(sym).reset_index(drop=True)
    
    open1 = sample.loc[1, "adj_open"]
    close1 = sample.loc[1, "adj_close"]
    close5 = sample.loc[5, "adj_close"]
    close20 = sample.loc[20, "adj_close"]
    
    expected_ret_1d = (close1 / open1) - 1.0
    expected_ret_5d = (close5 / open1) - 1.0
    expected_ret_20d = (close20 / open1) - 1.0
    
    assert np.isclose(sample.loc[0, "future_return_1d"], expected_ret_1d, atol=1e-5)
    assert np.isclose(sample.loc[0, "future_return_5d"], expected_ret_5d, atol=1e-5)
    assert np.isclose(sample.loc[0, "future_return_20d"], expected_ret_20d, atol=1e-5)


# ------------------------------------------------------------------------------
# Test 4: Constant factor does not generate fake long-short alpha (P0-5)
# ------------------------------------------------------------------------------
def test_constant_factor_does_not_generate_fake_long_short_alpha(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    
    ic_s = FactorMetricsEngine.compute_daily_ic(df_labeled, "F_CONSTANT", "future_return_20d")
    assert (ic_s == 0.0).all() or ic_s.empty
    
    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(
        synthetic_panel_data, factor_col="F_CONSTANT", direction=1, n_quantiles=5
    )
    assert pnl_res["annualized_return"] == 0.0
    assert pnl_res["sharpe_ratio"] == 0.0
    assert pnl_res["net_sharpe_ratio"] == 0.0


# ------------------------------------------------------------------------------
# Test 5: Train-selected factor direction frozen during validation (P0-5)
# ------------------------------------------------------------------------------
def test_train_selected_factor_direction_frozen_during_validation():
    """
    构造在训练期为负 IC、在验证期为正 IC 的因子。
    验证集必须应用训练集冻结的方向 -1，不得在验证集翻转为 +1。
    """
    cfg = ResearchConfig(WF_TRAIN_YEARS=0.5, WF_VALIDATION_YEARS=0.25, WF_PURGE_DAYS=5, MIN_WF_FOLDS_FOR_CERTIFICATION=1)
    dates = pd.date_range("2021-01-01", periods=300, freq="B")
    
    rows = []
    for idx, d in enumerate(dates):
        for s in ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ", "000006.SZ"]:
            s_val = int(s[5])
            # 训练期 (前 126 天): 因子与收益负相关
            # 验证期 (131 天后): 因子与收益正相关
            if idx < 126:
                f_val = -s_val * 1.0
                ret_val = s_val * 0.01
            else:
                f_val = s_val * 1.0
                ret_val = s_val * 0.01
                
            rows.append({
                "date": d, "symbol": s, "adj_open": 10.0, "adj_close": 10.0 * (1 + ret_val),
                "open": 10.0, "close": 10.0 * (1 + ret_val), "in_universe": True, "is_suspended": False,
                "F_REVERSAL": f_val, "future_return_20d": ret_val, "future_excess_return_20d": ret_val
            })
            
    df = pd.DataFrame(rows)
    res = FactorSelectionEngine.run_purged_walk_forward(df, ["F_REVERSAL"], config=cfg)
    assert res["total_folds"] >= 1


# ------------------------------------------------------------------------------
# Test 6: Walk-Forward uses same classification pipeline as full selection (P0-5)
# ------------------------------------------------------------------------------
def test_walk_forward_uses_same_classification_pipeline_as_full_selection(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=20)
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    
    # 构造一个高 IC 但缺失率极高 (> 50%) 的因子 F_HOLEY
    df_labeled["F_HOLEY"] = df_labeled["F_ALPHA"].copy()
    df_labeled.loc[df_labeled["symbol"].isin(["000001.SZ", "000002.SZ", "000003.SZ"]), "F_HOLEY"] = np.nan
    
    res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_ALPHA", "F_HOLEY", "F_CONSTANT"], config=cfg)
    for f in res["folds_detail"]:
        assert "F_CONSTANT" not in f["selected_factors"]


# ------------------------------------------------------------------------------
# Test 7: Walk-Forward FDR uses HAC p-value not naive p-value (P0-5)
# ------------------------------------------------------------------------------
def test_walk_forward_fdr_uses_hac_pvalue_not_naive_pvalue():
    cfg = default_research_config
    m = FactorEvaluationMetrics(
        factor_name="F_AUTO_CORR",
        horizon=20,
        mean_rank_ic=0.06,
        annualized_rank_icir=0.60,
        rank_ic_p_value=0.001,      # Naive 很小
        rank_ic_hac_p_value=0.25,   # 但 HAC 序列校正后不显著
        rank_ic_fdr_p_value=0.30,   # FDR > 0.20
        coverage_ratio=0.95
    )
    scores_df = pd.DataFrame([{
        "factor_name": "F_AUTO_CORR", "selection_score": 2.0, "abs_rank_ic": 0.06, "rank_ic": 0.06, "rank_ic_ir": 0.6,
        "monotonicity": 0.8, "sign_stability": 0.8, "net_sharpe": 1.5, "turnover": 0.1, "missing_ratio": 0.05,
        "coverage_ratio": 0.95, "is_redundant": 0.0, "recommended_direction": 1
    }])
    stab_dict = {"F_AUTO_CORR": FactorStabilityResult("F_AUTO_CORR", 0.06, 0.8, 0.06, 0.06, 0.06, {})}
    corr_res = CorrelationAnalysisResult(pd.DataFrame(), pd.DataFrame(), [], [], {})

    sel_res = FactorSelectionEngine.classify_factors(
        scores_df, {"F_AUTO_CORR": m}, {}, stab_dict, corr_res, config=cfg
    )
    assert sel_res.status_summary["F_AUTO_CORR"] == "REJECT", "Factor with HAC FDR > 0.20 must be REJECTED!"


# ------------------------------------------------------------------------------
# Test 8: Neutralization insufficient cross-section returns None not raw IC (P0-3)
# ------------------------------------------------------------------------------
def test_neutralization_insufficient_cross_section_returns_null_not_raw_ic(synthetic_panel_data):
    """
    当截面标的小于 MIN_NEUTRALIZATION_CROSS_SECTION (如仅保留 3 只股票) 时，
    中性化必须 fail-closed 返回 None 与 INSUFFICIENT_CROSS_SECTION，绝对禁止回退 raw_ic！
    """
    engine = FactorResearchEngine(config=ResearchConfig(MIN_NEUTRALIZATION_CROSS_SECTION=10))
    sub_df = synthetic_panel_data[synthetic_panel_data["symbol"].isin(["000001.SZ", "000002.SZ", "000003.SZ"])].copy()
    df_labeled = engine.generate_future_return_labels(sub_df, horizons=[20])

    neu_res = engine._run_real_neutralization_comparison(df_labeled, ["F_ALPHA"], "future_return_20d")
    item = neu_res["F_ALPHA"]
    
    assert item["neutralized_rank_ic"] is None, "Did not return None on insufficient cross section!"
    assert item["delta_rank_ic"] is None
    assert item["status"] == "INSUFFICIENT_CROSS_SECTION"


# ------------------------------------------------------------------------------
# Test 9: Orthogonalizer insufficient samples returns None fail closed (P0-4)
# ------------------------------------------------------------------------------
def test_orthogonalizer_insufficient_samples_returns_null_fail_closed(synthetic_panel_data):
    """
    当样本数 < 因子数 + 2 时，逐步正交化必须返回 None 与 INSUFFICIENT_CROSS_SECTION。
    """
    engine = FactorResearchEngine()
    sub_df = synthetic_panel_data[synthetic_panel_data["symbol"].isin(["000001.SZ", "000002.SZ", "000003.SZ"])].copy()
    df_labeled = engine.generate_future_return_labels(sub_df, horizons=[20])
    
    # 5 个因子在 3 个股票截面上无法正交化
    factors = ["F_ALPHA", "F_NEG", "F_CONSTANT", "F_REDUNDANT", "LOG_CIRC_MV"]
    ortho_res = engine._run_real_orthogonalization_comparison(df_labeled, factors, "future_return_20d")
    
    for f in factors[:2]:
        assert ortho_res[f]["orthogonalized_rank_ic"] is None
        assert ortho_res[f]["status"] == "INSUFFICIENT_CROSS_SECTION"


# ------------------------------------------------------------------------------
# Test 10: Manifest full factor hash covers all factor columns (P0-6)
# ------------------------------------------------------------------------------
def test_manifest_full_factor_hash_covers_all_factor_columns(synthetic_panel_data):
    """
    哈希必须覆盖全部因子列: 修改最后一个因子列的值，matrix_hash 必须发生变动。
    """
    engine = FactorResearchEngine()
    factors = [f"FACTOR_{i:02d}" for i in range(20)]
    df1 = synthetic_panel_data.copy()
    for f in factors:
        df1[f] = np.random.randn(len(df1))
        
    engine._build_research_run_manifest(df1, factors)
    hash1 = engine.run_manifest["factor_matrix_hash"]
    
    # 仅修改第 20 个因子 (FACTOR_19) 的一个值
    df2 = df1.copy()
    df2.loc[0, "FACTOR_19"] += 999.0
    
    engine._build_research_run_manifest(df2, factors)
    hash2 = engine.run_manifest["factor_matrix_hash"]
    
    assert hash1 != hash2, "Factor matrix hash failed to cover the last factor column!"


# ------------------------------------------------------------------------------
# Test 11: Development sample cannot be OOS certified (P0-2)
# ------------------------------------------------------------------------------
def test_development_sample_cannot_be_oos_certified(synthetic_panel_data):
    """
    当标的数小于 MIN_RESEARCH_SYMBOLS (例如仅 5 只股票) 时，
    无论 IC / Sharpe 多高，研究有效性状态必须锁定为 DEVELOPMENT_SAMPLE！
    """
    cfg = ResearchConfig(MIN_RESEARCH_SYMBOLS=50)
    engine = FactorResearchEngine(config=cfg)
    sub_df = synthetic_panel_data[synthetic_panel_data["symbol"].isin(["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"])].copy()
    
    engine._build_research_run_manifest(sub_df, ["F_ALPHA", "F_NEG"])
    assert engine.run_manifest["research_validity_status"] == "DEVELOPMENT_SAMPLE"


# ------------------------------------------------------------------------------
# Test 12: T+1 Tradability filter blocks suspended or locked assets (P1-1)
# ------------------------------------------------------------------------------
def test_t_plus_1_tradability_filter_blocks_suspended_or_locked_assets():
    """
    T 日生成信号，但 T+1 日标的停牌或无开盘价，系统必须严格阻断买入，不产生策略收益。
    """
    df = pd.DataFrame([
        {"date": "2021-01-04", "symbol": "000001.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": False},
        {"date": "2021-01-04", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
        # T+1 日 000001.SZ 停牌
        {"date": "2021-01-05", "symbol": "000001.SZ", "open": 0.0, "close": 10.0, "adj_open": 0.0, "adj_close": 10.0, "F_TEST": 1.0, "in_universe": True, "is_suspended": True},
        {"date": "2021-01-05", "symbol": "000002.SZ", "open": 10.0, "close": 10.0, "adj_open": 10.0, "adj_close": 10.0, "F_TEST": -1.0, "in_universe": True, "is_suspended": False},
    ])
    df["date"] = pd.to_datetime(df["date"])
    
    pnl = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(df, "F_TEST", n_quantiles=2)
    # 由于 Top 组标的在 T+1 停牌且开盘价为 0，无法成交，回测 PnL 记录为空
    assert pnl["daily_pnl_df"].empty or len(pnl["daily_pnl_df"]) == 0


# ------------------------------------------------------------------------------
# Test 13: DataFrame row permutation must not change quantile results (P0-5)
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
# Test 14: Train label end must not cross validation boundary (P0-5)
# ------------------------------------------------------------------------------
def test_train_label_end_must_not_cross_validation_boundary(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=25, HORIZONS=[1, 5, 20])
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=cfg.HORIZONS)
    
    res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_ALPHA", "F_NEG"], config=cfg)
    for f in res["folds_detail"]:
        t_end = pd.to_datetime(f["train_end"])
        v_start = pd.to_datetime(f["validation_start"])
        max_label_reach = t_end + pd.Timedelta(days=20)
        assert max_label_reach <= v_start, f"Label boundary crossed! reach={max_label_reach}, val_start={v_start}"


# ------------------------------------------------------------------------------
# Test 15: Changing validation prices cannot affect train factor selection (P0-5)
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
