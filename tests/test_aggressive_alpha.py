import pytest
import pandas as pd
import numpy as np
from strategy.aggressive_alpha_engine import AggressiveAlphaEngine

def test_aggressive_portfolio_generation():
    dates = pd.date_range('2026-09-01', periods=3, freq='D')
    rows = []
    for d in dates:
        rows.append({'date': d, 'symbol': '300308.SZ', 'close': 800.0, 'pct_change': 0.02})
        rows.append({'date': d, 'symbol': '603986.SH', 'close': 380.0, 'pct_change': 0.01})
    df = pd.DataFrame(rows)
    picks = AggressiveAlphaEngine.generate_aggressive_portfolio(df, '2026-09-03')
    assert len(picks) > 0
    assert 'pred_score' in picks.columns
    assert 'target_weight' in picks.columns
    assert picks['target_weight'].sum() <= 1.0
    # 验证第一重仓大于 15% (进攻型重仓)
    assert picks['target_weight'].iloc[0] >= 0.15
