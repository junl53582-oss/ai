"""G2 期中考自动裁决器单元测试"""
import numpy as np
import pandas as pd
import pytest

from tools.g2_gate_verdict import evaluate_g2_gate, MIN_DAYS

RNG = np.random.RandomState(7)


def _ledger(n, ic_a=0.06, ic_b=0.10, ic_c=0.12):
    return pd.DataFrame({
        'snap_date': [f'2026{i:04d}' for i in range(n)],
        'reconcile_date': [f'2026{i:04d}' for i in range(n)],
        'n_symbols': [300] * n,
        'ic_A': RNG.normal(ic_a, 0.05, n),
        'hi_lo_spread_A': RNG.normal(0.002, 0.01, n),
        'ic_B': RNG.normal(ic_b, 0.05, n),
        'hi_lo_spread_B': RNG.normal(0.004, 0.01, n),
        'ic_C': RNG.normal(ic_c, 0.05, n),
        'hi_lo_spread_C': RNG.normal(0.005, 0.01, n),
    })


def test_not_ready_before_min_days():
    v = evaluate_g2_gate(_ledger(MIN_DAYS - 1))
    assert v['ready'] is False and v['days'] == MIN_DAYS - 1
    assert v['tracks'] == {}


def test_verdict_after_min_days():
    v = evaluate_g2_gate(_ledger(MIN_DAYS))
    assert v['ready'] is True
    assert set(v['tracks']) == {'A', 'B', 'C'}
    for t in v['tracks'].values():
        assert t['verdict'] in ('晋级候选', '观察', '淘汰')
        assert 0.0 <= t['win_rate'] <= 1.0
        assert t['max_drawdown_spread'] <= 0.0
    # C 的均值最高 → 应排第一
    assert v['ranking'][0] == 'C'


def test_strong_track_promoted_and_weak_eliminated():
    n = MIN_DAYS
    led = _ledger(n)
    led['ic_A'] = -0.01 + RNG.normal(0, 0.001, n)      # 稳定负 → 淘汰
    led['ic_C'] = 0.15 + RNG.normal(0, 0.001, n)       # 强正且稳 → 晋级
    v = evaluate_g2_gate(led)
    assert v['tracks']['A']['verdict'] == '淘汰'
    assert v['tracks']['C']['verdict'] == '晋级候选'
    assert 'C' in v['promotion_candidates']


def test_incumbent_kept_when_nothing_passes():
    n = MIN_DAYS
    led = _ledger(n)
    led['ic_A'] = 0.005 + RNG.normal(0, 0.01, n)   # 正但不达标
    led['ic_B'] = 0.004 + RNG.normal(0, 0.01, n)
    led['ic_C'] = 0.003 + RNG.normal(0, 0.01, n)
    v = evaluate_g2_gate(led)
    assert v['promotion_candidates'] == []
    assert '维持现役模型' in v['decision']


def test_missing_track_column_tolerated():
    led = _ledger(MIN_DAYS).drop(columns=['ic_C', 'hi_lo_spread_C'])
    v = evaluate_g2_gate(led)
    assert v['ready'] is True
    assert set(v['tracks']) == {'A', 'B'}
