"""
增强预测能力四阶方案综合单元测试 (tests/test_enhanced_prediction_pipeline.py)
覆盖:
1. 方案一: 非同质化信息源采集 (AlternativeDataFetcher) 与特征计算 (compute_alternative_alphas)
2. 方案二: 宏观环境门控网络 (MacroRegimeGatingNetwork) 与混合专家架构 (MoEPredictor)
3. 方案三: 头部加权目标 (TopHeavyWeightedObjective) 与 LambdaRank 排序模型 (TopHeavyLambdaRankModel)
4. 方案四: 动态双阈值缓冲带 (BufferedPortfolioBuilder) 与行业上限硬约束 (max_sector_exposure)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from data.analyst_and_moneyflow_fetcher import AlternativeDataFetcher
from research_v2.features.alternative_alphas import compute_alternative_alphas
from models.moe_gating_model import MacroRegimeGatingNetwork, AggressiveExpert, DefensiveExpert, MoEPredictor
from models.top_heavy_ranker import TopHeavyWeightedObjective, TopHeavyLambdaRankModel
from strategy.buffered_portfolio_builder import BufferedPortfolioBuilder


@pytest.fixture
def mock_market_data():
    """生成 50 天、20 只股票的标准合成量价数据"""
    dates = pd.date_range("2026-01-01", periods=50)
    symbols = [f"{i:06d}.SZ" for i in range(1, 21)]
    industries = ["Bank"] * 6 + ["Tech"] * 7 + ["Pharma"] * 7
    rows = []
    np.random.seed(42)

    for d in dates:
        for i, s in enumerate(symbols):
            p = 10.0 + np.random.randn()
            rows.append({
                "date": d,
                "symbol": s,
                "open": p * 0.99,
                "high": p * 1.02,
                "low": p * 0.98,
                "close": p,
                "volume": 10000.0 + np.random.rand() * 5000,
                "amount": 100000.0 + np.random.rand() * 50000,
                "LOG_CIRC_MV": 22.5,
                "industry": industries[i],
                "f1": np.random.randn(),
                "f2": np.random.randn(),
                "target_ret_1d": np.random.randn() * 0.02,
                "target_label": np.random.randn() * 0.05
            })
    return pd.DataFrame(rows)


# =====================================================================
# 方案一测试: 预期与资金流
# =====================================================================
def test_plan1_alternative_alphas_generation(mock_market_data):
    df = compute_alternative_alphas(mock_market_data)
    expected_alphas = [
        "ALPHA_ANALYST_FY2_REVISION_20D",
        "ALPHA_ANALYST_COVERAGE_SURGE_60D",
        "ALPHA_MAIN_CAPITAL_NET_RATIO_5D",
        "ALPHA_MAIN_FLOW_DIVERGENCE_10D",
        "ALPHA_LARGE_ORDER_ACCUMULATION_20D"
    ]
    for alpha in expected_alphas:
        assert alpha in df.columns
        assert not df[alpha].isna().all(), f"{alpha} all nan"
        # 确认已做截面标准化，标准差不为无穷大且有限
        assert np.isfinite(df[alpha]).all()


def test_plan1_data_fetcher_offline_resilience(tmp_path):
    fetcher = AlternativeDataFetcher(cache_dir=tmp_path, timeout=1)
    # 测试异常标的不会抛出致命崩溃
    df_empty = fetcher.fetch_analyst_reports("999999.SZ", max_pages=1)
    assert isinstance(df_empty, pd.DataFrame)
    assert "symbol" in df_empty.columns


# =====================================================================
# 方案二测试: 宏观门控混合专家 (MoE)
# =====================================================================
def test_plan2_macro_gating_weights(mock_market_data):
    gating = MacroRegimeGatingNetwork()
    regime_df = gating.compute_regime_weights(mock_market_data)
    assert "weight_aggressive" in regime_df.columns
    assert "weight_defensive" in regime_df.columns
    assert "market_regime" in regime_df.columns
    # 权重和为 1.0
    w_sum = regime_df["weight_aggressive"] + regime_df["weight_defensive"]
    np.testing.assert_allclose(w_sum, 1.0, atol=1e-5)
    # 权重边界在 [0.10, 0.90]
    assert (regime_df["weight_aggressive"] >= 0.10).all()
    assert (regime_df["weight_aggressive"] <= 0.90).all()


def test_plan2_moe_predictor_fit_predict(mock_market_data):
    moe = MoEPredictor()
    split = int(len(mock_market_data) * 0.6)
    train_df = mock_market_data.iloc[:split]
    test_df = mock_market_data.iloc[split:]

    moe.fit(train_df, ["f1", "f2"], "target_label")
    preds = moe.predict(test_df, ["f1", "f2"])

    assert "pred_score" in preds.columns
    assert "pred_aggressive" in preds.columns
    assert "pred_defensive" in preds.columns
    assert "w_agg" in preds.columns
    assert len(preds) == len(test_df)


# =====================================================================
# 方案三测试: 头部加权与 LambdaRank
# =====================================================================
def test_plan3_top_heavy_weighted_objective():
    obj = TopHeavyWeightedObjective(top_quantile=0.85, top_weight_multiplier=5.0)
    preds = np.array([0.5, -0.2, 0.8, -0.1])
    labels = np.array([0.6, -0.1, -0.3, 0.0])  # 第3个是 False Positive (preds>0, labels<0)
    grad, hess = obj(preds, labels)
    assert len(grad) == 4
    assert len(hess) == 4
    assert (hess > 0).all()


def test_plan3_top_heavy_lambdarank(mock_market_data):
    ranker = TopHeavyLambdaRankModel(n_estimators=10, learning_rate=0.1)
    split = int(len(mock_market_data) * 0.6)
    train_df = mock_market_data.iloc[:split]
    test_df = mock_market_data.iloc[split:]

    ranker.fit(train_df, ["f1", "f2"], "target_label")
    preds = ranker.predict(test_df, ["f1", "f2"])
    assert len(preds) == len(test_df)
    assert np.isfinite(preds).all()


# =====================================================================
# 方案四测试: 动态缓冲带与行业上限
# =====================================================================
def test_plan4_buffered_portfolio_hysteresis():
    builder = BufferedPortfolioBuilder(enter_top_k=8, exit_top_k=20, max_sector_exposure=0.30, target_positions=8)
    symbols = [f"{i:06d}.SZ" for i in range(1, 31)]
    industries = ["Bank"] * 10 + ["Tech"] * 10 + ["Pharma"] * 10

    # Day 1: 首次建仓
    df1 = pd.DataFrame({"symbol": symbols, "pred_score": np.linspace(10, 1, 30), "industry": industries})
    p1 = builder.select_portfolio(df1, set())
    holdings1 = set(p1["symbol"])
    assert len(holdings1) == 8

    # 行业硬上限约束: 单行业权重不得超过 30%
    max_sec_w1 = p1.groupby("industry")["weight"].sum().max()
    assert max_sec_w1 <= 0.30001

    # Day 2: 制造第 7~9 名排名微调 (处于缓冲带 [9, 20])
    df2 = df1.copy()
    scores = df2["pred_score"].values.copy()
    scores[7] = 8.4  # 原第 8 名降至第 9 名
    scores[8] = 8.6  # 原第 9 名升至第 8 名
    df2["pred_score"] = scores

    p2 = builder.select_portfolio(df2, holdings1)
    holdings2 = set(p2["symbol"])

    # 缓冲带生效: 原持仓未跌破 exit_top_k (20)，全量保留，避免无谓换手
    assert holdings1 == holdings2, "Buffer policy should retain existing holdings within exit_top_k"


def test_plan4_turnover_reduction_simulation(mock_market_data):
    mock_market_data["pred_score"] = mock_market_data["f1"]
    
    # 无缓冲带 (8 进 8 出)
    unbuf = BufferedPortfolioBuilder(enter_top_k=8, exit_top_k=8)
    res_unbuf = unbuf.simulate_buffered_backtest(mock_market_data)

    # 动态缓冲带 (8 进 20 出)
    buf = BufferedPortfolioBuilder(enter_top_k=8, exit_top_k=20)
    res_buf = buf.simulate_buffered_backtest(mock_market_data)

    mean_to_unbuf = res_unbuf["turnover"].mean()
    mean_to_buf = res_buf["turnover"].mean()

    assert mean_to_buf < mean_to_unbuf, f"Buffered turnover ({mean_to_buf:.4f}) must be lower than unbuffered ({mean_to_unbuf:.4f})"
