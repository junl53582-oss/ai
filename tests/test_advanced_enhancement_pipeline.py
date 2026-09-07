"""
进阶增强方案单元测试 (tests/test_advanced_enhancement_pipeline.py)
覆盖:
1. 方案五: 日内与集合竞价微观结构 Alpha (compute_microstructure_alphas)
2. 方案七: Barra 多风格正交残差化 (BarraStyleOrthogonalizer)
3. 方案九: 宏观自适应动态总仓位与 ATR 风险平价 (MacroAdaptiveExposureManager)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from research_v2.features.microstructure_alphas import compute_microstructure_alphas
from research_v2.features.barra_orthogonalizer import BarraStyleOrthogonalizer
from strategy.macro_adaptive_sizer import MacroAdaptiveExposureManager


@pytest.fixture
def sample_market_panel():
    """生成包含量价与 Barra 风格特征的标准测试截面"""
    dates = pd.date_range("2026-01-01", periods=40)
    symbols = [f"{i:06d}.SZ" for i in range(1, 21)]
    industries = ["Bank"] * 6 + ["Tech"] * 7 + ["Pharma"] * 7
    rows = []
    np.random.seed(42)

    for d in dates:
        for i, s in enumerate(symbols):
            p = 20.0 + np.random.randn()
            rows.append({
                "date": d,
                "symbol": s,
                "open": p * 0.99,
                "high": p * 1.03,
                "low": p * 0.97,
                "close": p,
                "volume": 20000.0 + np.random.rand() * 5000,
                "amount": 400000.0 + np.random.rand() * 100000,
                "turnover": 2.5 + np.random.rand(),
                "LOG_CIRC_MV": 22.0 + np.random.randn(),
                "ROC20": np.random.randn() * 0.05,
                "STD20": 0.02 + np.random.rand() * 0.01,
                "industry": industries[i],
                "pred_score": np.random.randn(),
                "target_ret_1d": np.random.randn() * 0.02
            })
    return pd.DataFrame(rows)


# =====================================================================
# 方案五测试: 微观结构与竞价 Alpha
# =====================================================================
def test_plan5_microstructure_alphas(sample_market_panel):
    df = compute_microstructure_alphas(sample_market_panel)
    expected_cols = [
        "ALPHA_AUCTION_GAP_THRUST",
        "ALPHA_OPEN_ABSORPTION_RATIO",
        "ALPHA_TAIL_VWAP_TWAP_SPREAD",
        "ALPHA_INTRADAY_VOLATILITY_ASYMMETRY",
        "ALPHA_CLOSE_THRUST_INTENSITY"
    ]
    for col in expected_cols:
        assert col in df.columns
        assert not df[col].isna().all()
        assert np.isfinite(df[col]).all()


# =====================================================================
# 方案七测试: Barra 多风格正交化
# =====================================================================
def test_plan7_barra_orthogonalizer(sample_market_panel):
    orth = BarraStyleOrthogonalizer(risk_factor_cols=["LOG_CIRC_MV", "ROC20", "STD20", "turnover"])
    # 构造一个人为与 LOG_CIRC_MV 高度相关的特征
    sample_market_panel["test_feature"] = sample_market_panel["LOG_CIRC_MV"] * 2.5 + np.random.randn(len(sample_market_panel))
    
    out_df = orth.orthogonalize_dataframe(sample_market_panel, ["test_feature"])
    assert "test_feature" in out_df.columns

    eval_res = orth.evaluate_orthogonality(out_df, ["test_feature"])
    assert eval_res["is_orthogonal"] is True
    assert eval_res["mean_abs_correlation"] < 0.05


# =====================================================================
# 方案九测试: 宏观自适应动态总仓位与 ATR 风险平价
# =====================================================================
def test_plan9_macro_adaptive_sizer(sample_market_panel):
    mgr = MacroAdaptiveExposureManager(min_exposure=0.30, max_exposure=1.00)
    exp_df = mgr.compute_daily_macro_exposure(sample_market_panel)

    assert "target_gross_exposure" in exp_df.columns
    assert "cash_ratio" in exp_df.columns
    # 仓位严格在 [0.30, 1.00] 之间
    assert (exp_df["target_gross_exposure"] >= 0.30).all()
    assert (exp_df["target_gross_exposure"] <= 1.00).all()
    # 股票仓位 + 现金比例 = 1.0
    tot = exp_df["target_gross_exposure"] + exp_df["cash_ratio"]
    np.testing.assert_allclose(tot, 1.0, atol=1e-5)

    # 测试单日 ATR 权重分配
    d1 = sample_market_panel[sample_market_panel["date"] == sample_market_panel["date"].iloc[0]].head(8)
    weighted = mgr.allocate_portfolio_weights(d1, target_gross_exposure=0.80)
    assert len(weighted) == 8
    # 权重总和等于 0.80 或受行业上限裁剪而略小于 0.80
    assert weighted["target_weight"].sum() <= 0.80001
    # 行业上限严格小于等于 0.30
    max_sec = weighted.groupby("industry")["target_weight"].sum().max()
    assert max_sec <= 0.30001


def test_plan9_simulate_adaptive_backtest(sample_market_panel):
    mgr = MacroAdaptiveExposureManager()
    exp_df = mgr.compute_daily_macro_exposure(sample_market_panel)
    sim_res = mgr.simulate_adaptive_backtest(sample_market_panel, exp_df, returns_col="target_ret_1d", top_k=8)

    assert "cumulative_net" in sim_res.columns
    assert "drawdown" in sim_res.columns
    assert "cash_ratio" in sim_res.columns
    assert (sim_res["drawdown"] <= 0.0001).all()
