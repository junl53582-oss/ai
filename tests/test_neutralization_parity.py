"""
中性化串行/并行数值一致性测试 (tests/test_neutralization_parity.py)
保证 _neutralize_one_day 抽取重构后，单线程与进程池两条路径输出完全一致，
且 EMPTY_UNIVERSE 日、逐日模式与覆盖率审计统计在两条路径下均等价。
"""
import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from factors.processor import FactorProcessor

FACTOR_COLS = [f"F{k}" for k in range(6)]


def _make_panel(n_days: int = 36, n_symbols: int = 12, seed: int = 7) -> pd.DataFrame:
    """构造含 NaN / UNKNOWN 行业 / 非成分股 / 全空截面的合成面板"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    symbols = [f"60000{i}.SH" for i in range(n_symbols)]
    rows = []
    for dt in dates:
        for i, sym in enumerate(symbols):
            rows.append({
                "date": dt,
                "symbol": sym,
                "industry": None if i % 5 == 0 else f"IND_{i % 3}",
                "in_universe": bool(i % 7 != 0),
                "LOG_CIRC_MV": float(rng.normal(24.0, 1.0)),
            })
    df = pd.DataFrame(rows)
    for k in range(len(FACTOR_COLS)):
        base = rng.normal(size=len(df))
        noisy = np.where(rng.random(len(df)) < 0.05, np.nan, 0.0)
        df[FACTOR_COLS[k]] = base + noisy
    # 制造一天全非成分股的空截面 (EMPTY_UNIVERSE 路径)
    empty_date = dates[5]
    df.loc[df["date"] == empty_date, "in_universe"] = False
    return df


@pytest.fixture
def panel(tmp_path):
    return _make_panel()


def _run_neutralize(panel_df, n_jobs, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "NEUTRALIZATION_N_JOBS", n_jobs)
    proc = FactorProcessor(factor_dir=tmp_path)
    out = proc.neutralize_cross_section(panel_df.copy(), factor_cols=FACTOR_COLS)
    return out, proc


def test_serial_parallel_neutralization_identical(panel, tmp_path, monkeypatch):
    """并行(2进程)与串行输出 DataFrame 与审计统计必须完全一致"""
    out_s, p_serial = _run_neutralize(panel, 1, tmp_path, monkeypatch)
    out_p, p_par = _run_neutralize(panel, 2, tmp_path, monkeypatch)

    pd.testing.assert_frame_equal(out_s, out_p)
    assert p_serial.neutralization_mode_by_date == p_par.neutralization_mode_by_date
    assert p_serial.industry_coverage_by_date == p_par.industry_coverage_by_date
    assert p_serial.empty_universe_day_count == p_par.empty_universe_day_count


def test_neutralization_preserves_empty_universe_and_nan(panel, tmp_path, monkeypatch):
    """空截面日因子必须全为 NaN 且被计入审计，Warmup 原生 NaN 不被填充为 0"""
    out, proc = _run_neutralize(panel, 1, tmp_path, monkeypatch)

    # 找到空截面日
    empty_keys = [k for k, m in proc.neutralization_mode_by_date.items() if m == "EMPTY_UNIVERSE"]
    assert len(empty_keys) == 1
    assert proc.empty_universe_day_count == 1

    empty_day = pd.Timestamp(empty_keys[0])
    day_slice = out[out["date"] == empty_day]
    for col in FACTOR_COLS:
        assert day_slice[col].isna().all(), f"空截面日因子 {col} 应保持 NaN"

    # 非空日仍应保留原生 NaN (禁止盲目 fillna)
    non_empty = out[out["date"] != empty_day]
    assert non_empty[FACTOR_COLS].isna().any().any()

    # 逐日 mode 与覆盖率的键数量与天数一致
    assert len(proc.neutralization_mode_by_date) == out["date"].nunique()
