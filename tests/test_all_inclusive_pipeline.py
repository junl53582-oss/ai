"""
第四代全景旗舰量化增强方案单元测试 (tests/test_all_inclusive_pipeline.py)
覆盖:
1. 方案 A: 产业链与板块图关联龙头传导 Alpha (compute_graph_spillover_alphas)
2. 方案 B: PyTorch 时序注意力波形深度编码器 (TemporalWaveformExpert)
3. 方案 C: 差异化动态持仓生命周期管理器 (AdaptiveHoldingPortfolioManager)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from research_v2.features.graph_spillover_alphas import compute_graph_spillover_alphas
from models.temporal_waveform_expert import TemporalWaveformExpert
from strategy.adaptive_horizon_portfolio import AdaptiveHoldingPortfolioManager


@pytest.fixture
def panel_data_for_flagship():
    dates = pd.date_range("2026-01-01", periods=30)
    symbols = [f"{i:06d}.SZ" for i in range(1, 15)]
    industries = ["Bank"] * 5 + ["Tech"] * 5 + ["Auto"] * 4
    rows = []
    np.random.seed(42)

    for d in dates:
        for i, s in enumerate(symbols):
            p = 15.0 + np.random.randn()
            rows.append({
                "date": d,
                "symbol": s,
                "open": p * 0.99,
                "high": p * 1.02,
                "low": p * 0.98,
                "close": p,
                "volume": 15000.0 + np.random.rand() * 5000,
                "amount": 225000.0 + np.random.rand() * 50000,
                "turnover": 2.0,
                "industry": industries[i],
                "f1": np.random.randn(),
                "f2": np.random.randn(),
                "pred_score": np.random.randn(),
                "target_ret_1d": np.random.randn() * 0.02,
                "target_label": np.random.randn() * 0.05
            })
    return pd.DataFrame(rows)


def test_plan_a_graph_spillover_generation(panel_data_for_flagship):
    out = compute_graph_spillover_alphas(panel_data_for_flagship)
    expected = [
        "GRAPH_LEADER_SPILLOVER_1D",
        "GRAPH_CLUSTER_BREADTH_SURGE_5D",
        "GRAPH_CLUSTER_FLOW_SPILLOVER_10D"
    ]
    for col in expected:
        assert col in out.columns
        assert not out[col].isna().all()
        assert np.isfinite(out[col]).all()


def test_plan_b_temporal_waveform_expert_fit_predict(panel_data_for_flagship):
    expert = TemporalWaveformExpert(seq_len=8, d_model=16, n_heads=2, epochs=2, batch_size=64)
    split = int(len(panel_data_for_flagship) * 0.7)
    train_df = panel_data_for_flagship.iloc[:split]
    test_df = panel_data_for_flagship.iloc[split:]

    expert.fit(train_df, ["f1", "f2"], "target_label")
    preds = expert.predict(test_df, ["f1", "f2"])

    assert len(preds) == len(test_df)
    assert np.isfinite(preds).all()


def test_plan_c_adaptive_horizon_classification(panel_data_for_flagship):
    mgr = AdaptiveHoldingPortfolioManager(enter_top_k=5, exit_top_k=10)
    
    # 构造测试行
    row_leader = pd.Series({"pred_score": 1.2, "GRAPH_LEADER_SPILLOVER_1D": 0.8, "STD20": 0.02})
    row_volatile = pd.Series({"pred_score": -0.5, "GRAPH_LEADER_SPILLOVER_1D": 0.0, "STD20": 0.045})
    row_standard = pd.Series({"pred_score": 0.3, "GRAPH_LEADER_SPILLOVER_1D": 0.1, "STD20": 0.02})

    assert mgr.classify_holding_horizon(row_leader) == 40
    assert mgr.classify_holding_horizon(row_volatile) == 8
    assert mgr.classify_holding_horizon(row_standard) == 20


def test_plan_c_adaptive_horizon_backtest(panel_data_for_flagship):
    mgr = AdaptiveHoldingPortfolioManager(enter_top_k=5, exit_top_k=10)
    res_df, stats = mgr.simulate_adaptive_horizon_backtest(panel_data_for_flagship)

    assert "cumulative_net" in res_df.columns
    assert "drawdown" in res_df.columns
    assert "profit_loss_ratio" in stats
    assert stats["profit_loss_ratio"] > 0
