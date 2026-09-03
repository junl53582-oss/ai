import pytest
import pandas as pd
import numpy as np
from research_v2.alphas.novel_alphas import NovelAlphaFactory
from research_v2.alphas.alpha_evaluator import AlphaEvaluator

def test_novel_alphas_computation_and_evaluation():
    dates = pd.date_range('2023-01-01', periods=50, freq='B')
    symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH']
    
    rows = []
    rng = np.random.RandomState(42)
    for d in dates:
        for s in symbols:
            ret = rng.normal(0.001, 0.02)
            rows.append({
                'date': d,
                'symbol': s,
                'close': 10.0,
                'pct_change': ret,
                'benchmark_close': 3000.0 * (1 + ret * 0.5),
                'volume': 10000.0,
                'turnover': 0.02,
                'amount': 100000.0,
                'LOG_CIRC_MV': 20.0,
                'label_excess_20d': ret * 2.0
            })
    df = pd.DataFrame(rows)

    # 1. 测试残差动量
    res_mom = NovelAlphaFactory.calc_residual_momentum(df, window=10)
    assert len(res_mom) == len(df)

    # 2. 测试换手率冲击
    to_surp = NovelAlphaFactory.calc_turnover_surprise(df, short_w=5, long_w=15)
    assert len(to_surp) == len(df)

    # 3. 测试复合质量 x 动量
    q_mom = NovelAlphaFactory.calc_quality_x_momentum(df)
    assert len(q_mom) == len(df)

    # 4. 测试单因子独立评估器
    df['novel_alpha'] = res_mom
    metrics = AlphaEvaluator.evaluate_factor(df, factor_col='novel_alpha', label_col='label_excess_20d', round_trip_bps=20.0)
    assert 'standalone_rank_ic' in metrics
    assert 'icir' in metrics
    assert 'cost_adjusted_alpha_20bps' in metrics
    assert 'year_by_year_ic' in metrics
