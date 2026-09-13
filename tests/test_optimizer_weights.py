"""inv_vol 优化器负权重回归测试 (2026-09-11 修复)

背景: 数据管线的波动率列均为标准化 z-score (约半数取负), 原实现 1/(vol+1e-6)
产生负权重 (隐式做空, 对多头清单非法)。修复: 存在非正值时用 exp(z) 单调正变换。
"""
import numpy as np
import pandas as pd
import pytest

from strategy.optimizer import InverseVolOptimizer


def test_standardized_vol_never_negative():
    """标准化波动率 (含负值) → 权重全部非负且归一"""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        'STD20': rng.randn(20),           # 标准化 z-score, 含负值
        'symbol': [f'{i:06d}.SZ' for i in range(20)],
    })
    w = InverseVolOptimizer().optimize(df)
    assert (w >= 0).all(), f'出现负权重: {w[w < 0].to_dict()}'
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_lower_vol_gets_higher_weight():
    """单调性: 波动率越低权重越高 (含负 z 输入)"""
    df = pd.DataFrame({
        'STD20': [-2.0, -1.0, 0.0, 1.0, 2.0],
        'symbol': ['a', 'b', 'c', 'd', 'e'],
    })
    w = InverseVolOptimizer().optimize(df)
    assert w.iloc[0] > w.iloc[1] > w.iloc[2] > w.iloc[3] > w.iloc[4]
    assert (w >= 0).all()


def test_positive_vol_unchanged_behavior():
    """纯正波动率输入: 保持经典倒数加权语义"""
    df = pd.DataFrame({'std': [0.01, 0.02, 0.04], 'symbol': ['a', 'b', 'c']})
    w = InverseVolOptimizer().optimize(df)
    # 0.01/0.02/0.04 的倒数为 100/50/25 → 权重比 4:2:1
    assert w.iloc[0] == pytest.approx(4 / 7, abs=1e-6)
    assert w.iloc[2] == pytest.approx(1 / 7, abs=1e-6)


def test_all_nan_falls_back_to_equal():
    df = pd.DataFrame({'STD20': [np.nan] * 4, 'symbol': list('abcd')})
    w = InverseVolOptimizer().optimize(df)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert all(abs(v - 0.25) < 1e-9 for v in w)
