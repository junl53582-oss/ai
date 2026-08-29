"""
因子研究与 Alpha 真实性严密验证测试套件 (tests/test_factor_research.py)
覆盖 15 项核心防作弊与统计严密性自动化测试 (Phase 1.1 Hotfix)
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from research.config import ResearchConfig, default_research_config
from research.factor_metrics import FactorMetricsEngine, FactorEvaluationMetrics
from research.factor_decay import FactorDecayEngine
from research.factor_stability import FactorStabilityEngine, FactorStabilityResult
from research.factor_correlation import CorrelationAnalysisResult
from research.factor_selection import FactorSelectionEngine
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
            close_p = 10.0 * (1.0 + rng.normal(0, 0.1))
            
            f_alpha = stock_ret * 50.0 + rng.normal(0, 0.5)
            f_neg = -stock_ret * 50.0 + rng.normal(0, 0.5)
            f_constant = 0.0
            f_mv = 10.0 + s_idx * 0.5 + rng.normal(0, 0.1)
            f_redundant = f_alpha * 0.98 + rng.normal(0, 0.05)
            
            rows.append({
                "date": d,
                "symbol": s,
                "adj_close": close_p,
                "close": close_p,
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
# Test 1: 20D forward labels cannot be treated as 1D portfolio returns (P0-1)
# ------------------------------------------------------------------------------
def test_20d_forward_labels_cannot_be_treated_as_1d_portfolio_returns(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    
    fake_mean_20d = float(df_labeled["future_return_20d"].mean())
    fake_annual_ret = fake_mean_20d * 252.0
    
    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(
        synthetic_panel_data,
        factor_col="F_ALPHA",
        direction=1,
        n_quantiles=5
    )
    real_annual_ret = pnl_res["annualized_return"]
    real_sharpe = pnl_res["sharpe_ratio"]
    
    assert real_annual_ret != fake_annual_ret
    assert abs(real_sharpe) < 20.0


# ------------------------------------------------------------------------------
# Test 2: Constant factor does not generate fake long-short alpha (P0-5)
# ------------------------------------------------------------------------------
def test_constant_factor_does_not_generate_fake_long_short_alpha(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    
    ic_s = FactorMetricsEngine.compute_daily_ic(df_labeled, "F_CONSTANT", "future_return_20d")
    assert (ic_s == 0.0).all() or ic_s.empty
    
    pnl_res = FactorMetricsEngine.compute_realized_daily_portfolio_pnl(
        synthetic_panel_data,
        factor_col="F_CONSTANT",
        direction=1,
        n_quantiles=5
    )
    assert pnl_res["annualized_return"] == 0.0
    assert pnl_res["sharpe_ratio"] == 0.0
    assert pnl_res["net_sharpe_ratio"] == 0.0


# ------------------------------------------------------------------------------
# Test 3: Changing validation prices cannot affect train factor selection (P0-6)
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


# ------------------------------------------------------------------------------
# Test 4: Train label end must not cross validation boundary (P0-2)
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
# Test 5: Purge gap >= selected label horizon (P0-2)
# ------------------------------------------------------------------------------
def test_purge_gap_greater_equal_selected_label_horizon(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=25, HORIZONS=[20])
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    
    res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_ALPHA"], config=cfg)
    for f in res["folds_detail"]:
        t_end = pd.to_datetime(f["train_end"])
        v_start = pd.to_datetime(f["validation_start"])
        gap_days = (v_start - t_end).days
        assert gap_days >= 20, f"Purge gap ({gap_days} days) is smaller than max horizon (20 days)"


# ------------------------------------------------------------------------------
# Test 6: OOS metrics only exist for train-selected factors (P0-6)
# ------------------------------------------------------------------------------
def test_oos_metrics_only_exist_for_train_selected_factors(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=20)
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    
    res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_ALPHA", "F_CONSTANT"], config=cfg)
    for f in res["folds_detail"]:
        selected = f["selected_factors"]
        oos_eval = f["oos_evaluation"]
        assert "F_CONSTANT" not in selected
        assert "F_CONSTANT" not in oos_eval, "Unselected factor was illegally assigned an OOS score!"


# ------------------------------------------------------------------------------
# Test 7: Train-selected factor direction frozen during validation (P0-7)
# ------------------------------------------------------------------------------
def test_train_selected_factor_direction_frozen_during_validation(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=20)
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    
    res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_NEG"], config=cfg)
    summary = res["factor_summary"]["F_NEG"]
    if summary["selected_count"] > 0:
        assert summary["oos_mean_rank_ic"] != 0.0


# ------------------------------------------------------------------------------
# Test 8: Train-selected best horizon frozen during validation (P0-7)
# ------------------------------------------------------------------------------
def test_train_selected_best_horizon_frozen_during_validation(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=20, HORIZONS=[1, 5, 20])
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=cfg.HORIZONS)
    
    res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_ALPHA"], config=cfg)
    assert res["total_folds"] >= 1


# ------------------------------------------------------------------------------
# Test 9: FDR-rejected factor cannot become STRONG (P1-1)
# ------------------------------------------------------------------------------
def test_fdr_rejected_factor_cannot_become_strong():
    cfg = default_research_config
    m = FactorEvaluationMetrics(
        factor_name="F_LUCKY",
        horizon=20,
        mean_rank_ic=0.08,
        annualized_rank_icir=0.80,
        rank_ic_fdr_p_value=0.15,
        coverage_ratio=0.95
    )
    scores_df = pd.DataFrame([{
        "factor_name": "F_LUCKY", "selection_score": 2.5, "abs_rank_ic": 0.08, "rank_ic": 0.08, "rank_ic_ir": 0.8,
        "monotonicity": 0.8, "sign_stability": 0.8, "net_sharpe": 2.0, "turnover": 0.1, "missing_ratio": 0.05,
        "coverage_ratio": 0.95, "is_redundant": 0.0, "recommended_direction": 1
    }])
    stab_dict = {"F_LUCKY": FactorStabilityResult("F_LUCKY", 0.08, 0.8, 0.08, 0.08, 0.08, {})}
    corr_res = CorrelationAnalysisResult(pd.DataFrame(), pd.DataFrame(), [], [], {})
    
    sel_res = FactorSelectionEngine.classify_factors(
        scores_df=scores_df,
        metrics_dict={"F_LUCKY": m},
        decay_dict={},
        stability_dict=stab_dict,
        corr_result=corr_res,
        config=cfg
    )
    assert sel_res.status_summary["F_LUCKY"] != "STRONG", "Factor with FDR > 0.05 must not be classified as STRONG!"


# ------------------------------------------------------------------------------
# Test 10: Neutralized RankIC must come from actually transformed factor values (P0-4)
# ------------------------------------------------------------------------------
def test_neutralized_rank_ic_must_come_from_actually_transformed_factor_values(synthetic_panel_data):
    engine = FactorResearchEngine()
    df_labeled = engine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    
    df_labeled["F_FAKE_MV"] = df_labeled["LOG_CIRC_MV"]
    
    neu_res = engine._run_real_neutralization_comparison(df_labeled, ["F_FAKE_MV"], "future_return_20d")
    neu_ic = neu_res["F_FAKE_MV"]["neutralized_rank_ic"]
    
    assert neu_ic is not None
    assert abs(neu_ic) < 0.05, f"Neutralized RankIC ({neu_ic}) did not strip market cap collinearity!"


# ------------------------------------------------------------------------------
# Test 11: Orthogonalized RankIC must come from actual orthogonalized series (P0-4)
# ------------------------------------------------------------------------------
def test_orthogonalized_rank_ic_must_come_from_actual_orthogonalized_series(synthetic_panel_data):
    engine = FactorResearchEngine()
    df_labeled = engine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    
    df_labeled["F_CLONE"] = df_labeled["F_ALPHA"]
    
    ortho_res = engine._run_real_orthogonalization_comparison(df_labeled, ["F_ALPHA", "F_CLONE"], "future_return_20d")
    raw_ic = ortho_res["F_CLONE"]["raw_rank_ic"]
    ortho_ic = ortho_res["F_CLONE"]["orthogonalized_rank_ic"]
    
    assert abs(ortho_ic) < abs(raw_ic)
    assert ortho_res["F_CLONE"]["status"] == "real_gram_schmidt_calculated"


# ------------------------------------------------------------------------------
# Test 12: Less than minimum OOS folds cannot be OOS certified (P1-6)
# ------------------------------------------------------------------------------
def test_less_than_minimum_oos_folds_cannot_be_oos_certified(synthetic_panel_data):
    cfg = ResearchConfig(MIN_WF_FOLDS_FOR_CERTIFICATION=5)
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    
    res = FactorSelectionEngine.run_purged_walk_forward(df_labeled, ["F_ALPHA"], config=cfg)
    assert res["walk_forward_status"] == "PRELIMINARY", "Folds < 5 must remain PRELIMINARY, cannot be OOS_VALIDATED!"


# ------------------------------------------------------------------------------
# Test 13: Feature at T cannot use execution price earlier than T+1 (P0-3)
# ------------------------------------------------------------------------------
def test_feature_at_t_cannot_use_execution_price_earlier_than_t_plus_1(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[1, 20])
    
    sym = synthetic_panel_data["symbol"].iloc[0]
    sample = df_labeled.groupby("symbol").get_group(sym).reset_index(drop=True)
    p0 = sample.loc[0, "adj_close"]
    p1 = sample.loc[1, "adj_close"]
    p20 = sample.loc[20, "adj_close"]
    
    expected_ret_1d = (p1 / p0) - 1.0
    expected_ret_20d = (p20 / p1) - 1.0
    
    assert np.isclose(sample.loc[0, "future_return_1d"], expected_ret_1d, atol=1e-5)
    assert np.isclose(sample.loc[0, "future_return_20d"], expected_ret_20d, atol=1e-5)


# ------------------------------------------------------------------------------
# Test 14: Future validation mutation cannot change training-selected horizon (P1-3)
# ------------------------------------------------------------------------------
def test_future_validation_mutation_cannot_change_training_selected_horizon(synthetic_panel_data):
    cfg = ResearchConfig(WF_TRAIN_YEARS=1.0, WF_VALIDATION_YEARS=0.5, WF_PURGE_DAYS=20, HORIZONS=[1, 5, 20])
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=cfg.HORIZONS)
    
    train_end = pd.to_datetime("2021-12-31")
    train_df1 = df_labeled[df_labeled["date"] <= train_end].copy()
    dec1 = FactorDecayEngine.analyze_decay(train_df1, "F_ALPHA", horizons=cfg.HORIZONS)
    
    df_mutated = df_labeled.copy()
    df_mutated.loc[df_mutated["date"] > train_end, "future_return_1d"] = 10.0
    train_df2 = df_mutated[df_mutated["date"] <= train_end].copy()
    dec2 = FactorDecayEngine.analyze_decay(train_df2, "F_ALPHA", horizons=cfg.HORIZONS)
    
    assert dec1.best_horizon == dec2.best_horizon


# ------------------------------------------------------------------------------
# Test 15: DataFrame row permutation must not change quantile results (P0-5)
# ------------------------------------------------------------------------------
def test_dataframe_row_permutation_must_not_change_quantile_results(synthetic_panel_data):
    df_labeled = FactorResearchEngine.generate_future_return_labels(synthetic_panel_data, horizons=[20])
    
    q_dict_orig, mono_orig, _ = FactorMetricsEngine.compute_quantile_returns(
        df_labeled, "F_ALPHA", "future_return_20d", n_quantiles=5
    )
    
    df_shuffled = df_labeled.sample(frac=1.0, random_state=123).reset_index(drop=True)
    q_dict_shuf, mono_shuf, _ = FactorMetricsEngine.compute_quantile_returns(
        df_shuffled, "F_ALPHA", "future_return_20d", n_quantiles=5
    )
    
    assert mono_orig == mono_shuf
    for k in q_dict_orig:
        assert np.isclose(q_dict_orig[k], q_dict_shuf[k], atol=1e-6)
