"""影子打分器测试: 合成数据验证 Train-Only 纪律与 Fail-Closed 行为"""
import numpy as np
import pandas as pd
import pytest

from research_v2.labels.label_registry import LabelRegistry
from strategy.shadow_scorer import compute_shadow_scores

N_SYMS = 40
N_DAYS = 400


@pytest.fixture
def synthetic_matrix():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2025-01-01', periods=N_DAYS)
    frames = []
    for i in range(N_SYMS):
        steps = rng.normal(0, 0.015, N_DAYS)
        close = 20 * np.exp(np.cumsum(steps))
        f_mom = pd.Series(close).pct_change(5).fillna(0).values
        frames.append(pd.DataFrame({
            'date': dates,
            'symbol': f'{600000 + i:06d}.SH',
            'close': close,
            'in_universe': True,
            'f_momentum': f_mom,
            'f_reversal': -f_mom,
            'f_noise': rng.normal(0, 1, N_DAYS),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def patched_label(monkeypatch):
    """注入合成标签: 未来 20 日收益方向 (有效信号 = 短期动量, 供选择器有据可选)"""
    def fake_label(df, horizon=20):
        fut = df.groupby('symbol')['close'].pct_change(horizon).shift(-horizon)
        return (fut > 0).astype(float).mask(fut.isna(), np.nan)
    monkeypatch.setattr(LabelRegistry, 'compute_label_v2', staticmethod(fake_label))


def test_shadow_scores_success_train_only(synthetic_matrix, patched_label, tmp_path):
    matrix_path = tmp_path / 'matrix.parquet'
    cache_path = tmp_path / 'cache.json'
    synthetic_matrix.to_parquet(matrix_path)

    latest_date = synthetic_matrix['date'].max()
    latest_syms = synthetic_matrix[synthetic_matrix['date'] == latest_date]['symbol'].tolist()
    top_df = pd.DataFrame({'symbol': latest_syms[:10], 'name': 'x', 'close': 20.0, 'pred_score': 0.5})

    out, meta = compute_shadow_scores(top_df, matrix_path=matrix_path, top_k=3, cache_path=cache_path)

    assert meta.get('error') is None, f'Fail-Closed 被误触发: {meta}'
    assert meta.get('as_of') == str(latest_date)[:10]
    # Train-Only: 训练截止必须早于打分截面 (purge 纪律)
    assert str(meta.get('train_end')) < str(meta.get('as_of'))
    assert out['shadow_score'].notna().sum() >= 8
    assert out['shadow_score'].between(0, 1).all()
    assert len(meta.get('factors', [])) > 0

    # 磁盘缓存: 模拟进程重启 (清空内存缓存) 后, 第二次调用从磁盘读, 不重训, 结果一致
    assert meta.get('from_cache') is False
    assert cache_path.exists()
    from strategy import shadow_scorer as ss
    ss._CACHE.clear()
    out2, meta2 = compute_shadow_scores(top_df, matrix_path=matrix_path, top_k=3, cache_path=cache_path)
    assert meta2.get('from_cache') == 'disk'
    assert meta2.get('error') is None
    pd.testing.assert_series_equal(
        out['shadow_score'].sort_index(), out2['shadow_score'].sort_index()
    )


def test_shadow_scores_fail_closed_on_missing_matrix(tmp_path):
    top_df = pd.DataFrame({'symbol': ['600000.SH'], 'pred_score': [0.5]})
    out, meta = compute_shadow_scores(top_df, matrix_path=tmp_path / 'nope.parquet',
                                      cache_path=tmp_path / 'cache.json')
    assert meta.get('error'), '缺失矩阵必须给出 Fail-Closed 原因'
    assert out['shadow_score'].isna().all()
