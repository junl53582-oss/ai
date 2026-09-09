"""模型失效检测器单元测试: 价差计算 + 状态机 + 风控降档接线"""
import numpy as np
import pandas as pd
import pytest

import strategy.signal_monitor as sm
import strategy.macro_regime_gate as gate_mod
from strategy.signal_monitor import get_signal_health


def _make_spread(values) -> pd.Series:
    idx = pd.bdate_range('2026-01-01', periods=len(values))
    return pd.Series(values, index=idx)


def test_health_state_machine():
    # 健康期: 价差全正
    h = get_signal_health(window=20, spread_series=_make_spread([1.0] * 30))
    assert h['state'] == 'HEALTHY' and h['position_multiplier'] == 1.0
    # 退化期: 价差在 -0.5% ~ 0 之间
    h = get_signal_health(window=20, spread_series=_make_spread([-0.2] * 30))
    assert h['state'] == 'DEGRADED' and h['position_multiplier'] == 0.8
    # 反转期: 价差 < -0.5%
    h = get_signal_health(window=20, spread_series=_make_spread([-1.5] * 30))
    assert h['state'] == 'REVERSED' and h['position_multiplier'] == 0.6
    # 无数据 Fail-Open
    h = get_signal_health(window=20, spread_series=_make_spread([]))
    assert h['state'] == 'NO_DATA' and h['position_multiplier'] == 1.0


def test_real_oos_data_returns_state():
    """真实 OOS 数据源应返回明确状态 (数据截至 09-04, 含 2024-2026 失效期)"""
    h = get_signal_health(window=60)
    assert h['state'] in ('HEALTHY', 'DEGRADED', 'REVERSED', 'NO_DATA', 'ERROR')
    assert 0 < h['position_multiplier'] <= 1.0
    if h['state'] not in ('NO_DATA', 'ERROR'):
        assert isinstance(h['spread'], float)


def _gate_with_health(monkeypatch, health, n=4):
    """确定性闸门测试: 固定 ATR (等权拆分), n 只股票"""
    import strategy.signal_monitor as smmod
    from strategy.macro_regime_gate import MacroRegimeGate
    monkeypatch.setattr(smmod, 'get_signal_health', lambda **kw: health)
    monkeypatch.setattr(gate_mod, '_load_atr14_map', lambda: {f'60000{i}.SH': 1.0 for i in range(n)})
    top = pd.DataFrame({'symbol': [f'60000{i}.SH' for i in range(n)],
                        'name': [f'测试{i}' for i in range(n)],
                        'close': [10.0] * n})
    snap = {'regime_state': 'Neutral (结构平衡分化)', 'macro_regime_index': 0.5,
            'suggested_total_position': 0.8,
            'overseas_tech_resonance': {'nvda_change_pct': 0.0}}
    return MacroRegimeGate.apply_macro_regime_adjustment(top, snap)


def test_gate_position_degrades_when_model_reversed(monkeypatch):
    """闸门接线测试: 检测器报告 REVERSED 时, 总仓位应按系数 0.6 降档 (0.8*0.6=0.48)"""
    out = _gate_with_health(monkeypatch, {'state': 'REVERSED', 'position_multiplier': 0.6,
                                          'spread': -1.2, 'window': 60}, n=4)
    assert out['adjusted_weight'].sum() == pytest.approx(0.48, abs=0.02)
    assert '模型健康度[REVERSED]' in out['macro_execution_rationale'].iloc[0]


def test_gate_untouched_when_healthy(monkeypatch):
    """HEALTHY 时仓位不受影响 (0.8 总仓位均分)"""
    out = _gate_with_health(monkeypatch, {'state': 'HEALTHY', 'position_multiplier': 1.0,
                                          'spread': 0.3, 'window': 60}, n=4)
    assert out['adjusted_weight'].sum() == pytest.approx(0.80, abs=0.02)
    assert '模型健康度' not in out['macro_execution_rationale'].iloc[0]
