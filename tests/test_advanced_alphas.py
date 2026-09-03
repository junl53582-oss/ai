import pytest
import pandas as pd
import numpy as np
from research_v2.alphas.novel_alphas import NovelAlphaFactory

def test_novel_alphas_extended():
    n_days = 50
    symbols = ['000001.SZ', '600000.SH']
    records = []
    
    dates = pd.date_range('2025-01-01', periods=n_days)
    for sym in symbols:
        price = 10.0
        for dt in dates:
            ret = np.random.normal(0, 0.02)
            price *= (1 + ret)
            vol = np.random.uniform(1000, 5000)
            records.append({
                'date': dt,
                'symbol': sym,
                'adj_close': price,
                'close': price,
                'volume': vol,
                'amount': price * vol,
                'benchmark_close': 3000.0 * (1 + np.random.normal(0, 0.01)),
                'pct_change': ret
            })
    df = pd.DataFrame(records)

    rev = NovelAlphaFactory.calc_short_term_reversal(df, window=5)
    assert len(rev) == len(df)
    assert not rev.dropna().empty

    idio = NovelAlphaFactory.calc_idio_vol_penalty(df, window=10)
    assert len(idio) == len(df)
    assert not idio.dropna().empty

    div = NovelAlphaFactory.calc_money_flow_divergence(df, window=5)
    assert len(div) == len(df)
    assert not div.dropna().empty

    tail_bias = NovelAlphaFactory.calc_tail_liquidity_bias(df, window=5)
    assert len(tail_bias) == len(df)
    assert not tail_bias.dropna().empty
