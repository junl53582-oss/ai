import pytest
import pandas as pd
import numpy as np
from research_v2.labels.label_registry import LabelRegistry, LabelVersion

def test_label_registry_monotonicity_and_costs():
    dates = pd.date_range('2023-01-01', periods=40, freq='B')
    symbols = ['000001.SZ', '000002.SZ']
    
    rows = []
    for d in dates:
        for s in symbols:
            rows.append({
                'date': d,
                'symbol': s,
                'open': 10.0,
                'close': 10.0,
                'adj_open': 10.0,
                'adj_close': 10.0,
                'benchmark_open': 3000.0,
                'benchmark_close': 3000.0,
                'industry': 'BANK',
                'is_limit_up_locked': False,
                'is_suspended': False
            })
    df = pd.DataFrame(rows)
    
    # 模拟未来价格上涨 10%
    df.loc[(df['date'] >= dates[25]) & (df['symbol'] == '000001.SZ'), 'adj_open'] = 11.0
    df.loc[(df['date'] >= dates[25]) & (df['symbol'] == '000001.SZ'), 'adj_close'] = 11.0

    v1 = LabelRegistry.compute_label_v1(df, horizon=20)
    v2 = LabelRegistry.compute_label_v2(df, horizon=20)
    v3 = LabelRegistry.compute_label_v3(df, horizon=20)
    v5 = LabelRegistry.compute_label_v5(df, horizon=20, round_trip_cost_bps=20.0)

    assert len(v1) == len(df)
    assert len(v2) == len(df)
    assert len(v3) == len(df)
    assert len(v5) == len(df)

    # 验证 V5 (扣成本) 严格小于 V3 (毛超额)
    valid_mask = v3.notna() & v5.notna()
    assert (v5[valid_mask] < v3[valid_mask]).all()
    np.testing.assert_allclose((v3[valid_mask] - v5[valid_mask]), 0.0020, atol=1e-6)

def test_suspended_or_limit_up_entry_becomes_nan():
    dates = pd.date_range('2023-01-01', periods=30, freq='B')
    rows = []
    for d in dates:
        rows.append({
            'date': d,
            'symbol': '000001.SZ',
            'adj_open': 10.0,
            'adj_close': 10.0,
            'benchmark_open': 3000.0,
            'benchmark_close': 3000.0,
            'industry': 'BANK',
            'is_limit_up_locked': False,
            'is_suspended': False
        })
    df = pd.DataFrame(rows)
    
    # 在第 1 天 (T+1) 遭遇涨停板无法买入
    df.loc[1, 'is_limit_up_locked'] = True
    v2 = LabelRegistry.compute_label_v2(df, horizon=5)
    
    # T=0 的信号在 T+1 无法买入，因此 T=0 的标签必须为 NaN (Fail-Closed)
    assert np.isnan(v2.iloc[0])
