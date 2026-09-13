"""宏观闸门波动率自适应风控测试: ATR 止盈止损 + 逆波动率仓位 + 静态回退"""
import numpy as np
import pandas as pd
import pytest

import strategy.macro_regime_gate as gate
from strategy.macro_regime_gate import MacroRegimeGate


def _top_df():
    return pd.DataFrame({
        "symbol": ["600000.SH", "600001.SH"],
        "name": ["低波股", "高波股"],
        "close": [10.0, 10.0],
        "weight": [0.5, 0.5],
    })


def test_atr_scaled_tp_sl_bracket(monkeypatch):
    monkeypatch.setattr(gate, "_load_atr14_map", lambda: {"600000.SH": 0.2, "600001.SH": 1.0})
    snap = {"regime_state": "Neutral (结构平衡分化)", "macro_regime_index": 0.5,
            "suggested_total_position": 0.8,
            "overseas_tech_resonance": {"nvda_change_pct": 0.0}}
    out = MacroRegimeGate.apply_macro_regime_adjustment(_top_df(), snap)

    low = out[out["symbol"] == "600000.SH"].iloc[0]
    high = out[out["symbol"] == "600001.SH"].iloc[0]
    # TP1 = close + 3.0*ATR, SL = close - 2.2*ATR (Neutral)
    assert low["dynamic_tp1"] == pytest.approx(10.0 + 3.0 * 0.2, abs=0.011)
    assert low["dynamic_sl"] == pytest.approx(10.0 - 2.2 * 0.2, abs=0.011)
    assert high["dynamic_tp1"] == pytest.approx(10.0 + 3.0 * 1.0, abs=0.011)
    assert high["dynamic_sl"] == pytest.approx(10.0 - 2.2 * 1.0, abs=0.011)
    # 高波动股止损距离更宽
    assert (high["close"] - high["dynamic_sl"]) > (low["close"] - low["dynamic_sl"])
    # 逆波动率: 低波股权重更大, 且单票不超上限
    assert low["adjusted_weight"] > high["adjusted_weight"]
    assert (out["adjusted_weight"] <= 0.30 + 1e-9).all()


def test_fallback_to_static_when_no_atr(monkeypatch):
    monkeypatch.setattr(gate, "_load_atr14_map", lambda: {})
    snap = {"regime_state": "Neutral (结构平衡分化)", "macro_regime_index": 0.5,
            "suggested_total_position": 0.8,
            "overseas_tech_resonance": {"nvda_change_pct": 0.0}}
    out = MacroRegimeGate.apply_macro_regime_adjustment(_top_df(), snap)
    row = out.iloc[0]
    # 无 ATR 回退静态比例: Neutral TP +8.5% / SL -4.5%
    assert row["dynamic_tp1"] == pytest.approx(10.0 * 1.085, abs=0.011)
    assert row["dynamic_sl"] == pytest.approx(10.0 * 0.955, abs=0.011)
    assert "静态回退" in row["macro_execution_rationale"]


def test_risk_off_tighter_structures(monkeypatch):
    monkeypatch.setattr(gate, "_load_atr14_map", lambda: {"600000.SH": 0.5})
    snap = {"regime_state": "Risk-Off (避险收敛)", "macro_regime_index": 0.2,
            "suggested_total_position": 0.55,
            "overseas_tech_resonance": {"nvda_change_pct": -2.0}}
    out = MacroRegimeGate.apply_macro_regime_adjustment(_top_df().head(1), snap)
    row = out.iloc[0]
    # Risk-Off: k_tp=2.2 / k_sl=1.6, 且总权重贴近 55% 目标
    assert row["dynamic_tp1"] == pytest.approx(10.0 + 2.2 * 0.5, abs=0.011)
    assert row["dynamic_sl"] == pytest.approx(10.0 - 1.6 * 0.5, abs=0.011)
