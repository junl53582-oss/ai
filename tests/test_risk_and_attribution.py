"""
Barra 因子归因与动态风控体系测试套件 (tests/test_risk_and_attribution.py)
"""
import pytest
import pandas as pd
import numpy as np

from factors.attribution import BarraFactorAttribution, AlphaFactorExplainer, BARRA_STYLE_MAP
from strategy.risk_manager import (
    MarketRegime,
    MarketRegimeDetector,
    VolatilityTargetingEngine,
    DynamicDrawdownController
)


class TestBarraFactorAttribution:

    def test_compute_portfolio_style_exposure(self):
        target_df = pd.DataFrame({
            "symbol": ["600519.SH", "300750.SZ"],
            "target_weight": [0.60, 0.40]
        })
        factor_df = pd.DataFrame({
            "symbol": ["600519.SH", "300750.SZ"],
            "LOG_CIRC_MV": [1.5, 0.8],
            "ROC20": [0.5, 1.2],
            "STD20": [0.2, 0.9],
            "F_ROE": [2.0, 1.5],
            "KAUFMAN_EFFICIENCY_20": [0.8, 0.6],
            "FLOW_NET_BUY_RATIO_5D": [0.3, 0.7]
        })

        exposures = BarraFactorAttribution.compute_portfolio_style_exposure(target_df, factor_df)
        assert len(exposures) == len(BARRA_STYLE_MAP)
        assert exposures["Size (市值风格)"] == pytest.approx(0.6 * 1.5 + 0.4 * 0.8, abs=1e-3)
        assert exposures["Quality (基本面质量)"] > 0

    def test_decompose_returns(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        np.random.seed(42)
        b_ret = pd.Series(np.random.normal(0.0005, 0.01, size=100), index=dates)
        # 具有正 Alpha 的组合收益
        p_ret = 1.2 * b_ret + 0.0003 + np.random.normal(0, 0.005, size=100)

        decomp = BarraFactorAttribution.decompose_returns(p_ret, b_ret)
        assert "total_return" in decomp
        assert "beta" in decomp
        assert "market_beta_component" in decomp
        assert "specific_alpha_component" in decomp
        assert decomp["beta"] > 0

    def test_alpha_factor_explainer(self):
        stock_factors = pd.Series({
            "ROC20": 2.5,
            "YANG_ZHANG_VOL_20": -1.2,
            "F_ROE": 1.8,
            "symbol": "600519.SH",
            "in_universe": True
        })
        explanations = AlphaFactorExplainer.explain_stock_prediction(stock_factors, top_n=3)
        assert len(explanations) == 3
        # 绝对影响最大的应该是 ROC20 (2.5)
        assert explanations[0]["factor"] == "ROC20"
        assert explanations[0]["direction"] == "positive"


class TestDynamicRiskManager:

    def test_market_regime_detection(self):
        # 模拟牛市价格序列 (稳步上涨)
        bull_prices = pd.Series(np.linspace(3000, 4200, 100))
        res_bull = MarketRegimeDetector.detect_regime(bull_prices)
        assert res_bull["regime"] == MarketRegime.BULL_TREND.value
        assert res_bull["recommended_gross_exposure"] == 1.0

        # 模拟熊市破位序列 (持续下跌)
        bear_prices = pd.Series(np.linspace(4000, 2800, 100))
        res_bear = MarketRegimeDetector.detect_regime(bear_prices)
        assert res_bear["regime"] == MarketRegime.BEAR_CRISIS.value
        assert res_bear["recommended_gross_exposure"] <= 0.30

    def test_volatility_targeting(self):
        engine = VolatilityTargetingEngine(target_vol_annual=0.15, max_leverage=1.0)
        # 低波环境 -> 仓位系数 1.0
        low_vol_ret = pd.Series(np.random.normal(0, 0.003, size=30))
        scale_low = engine.compute_vol_scaling_factor(low_vol_ret)
        assert scale_low == 1.0

        # 高波环境 -> 仓位系数自动下调
        high_vol_ret = pd.Series(np.random.normal(0, 0.03, size=30))
        scale_high = engine.compute_vol_scaling_factor(high_vol_ret)
        assert scale_high < 0.60

    def test_dynamic_drawdown_circuit_breaker(self):
        # 初始平稳
        eq_normal = pd.Series([1.0, 1.02, 1.05, 1.04])
        lim_normal, _ = DynamicDrawdownController.evaluate_drawdown_exposure_limit(eq_normal)
        assert lim_normal == 1.0

        # 深度回撤 (峰值 1.05 -> 0.90，回撤 ~14.3%)
        eq_deep_dd = pd.Series([1.0, 1.05, 0.90])
        lim_deep, _ = DynamicDrawdownController.evaluate_drawdown_exposure_limit(eq_deep_dd)
        assert lim_deep == 0.50

        # 极端熔断 (峰值 1.05 -> 0.80，回撤 ~23.8%)
        eq_extreme = pd.Series([1.0, 1.05, 0.80])
        lim_ext, _ = DynamicDrawdownController.evaluate_drawdown_exposure_limit(eq_extreme)
        assert lim_ext == 0.0
