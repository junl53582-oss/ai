"""
数据拉取层 PIT 复权接线测试 (tests/test_data_fetcher_pit_adjustment.py)

锁定 data_fetcher._standardize_sina_stock_pit 的语义:
1. adj_price_t = raw_price_t × f_t / f_base (f_base 固定为 settings.START_DATE 处因子)
2. PIT 核心性质: 追加未来除权事件不改变历史复权价 (qfq 快照违反此性质)
3. 因子表缺失时抛 DataFetchError (fail-closed, 不静默产出非 PIT 数据)
"""
import pandas as pd
import pytest

from config.settings import settings
from data.data_fetcher import DataFetcher, DataFetchError


@pytest.fixture()
def fetcher():
    return DataFetcher(allow_synthetic=False)


def _raw_df(dates, close=10.0):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": [close] * len(dates),
        "high": [close * 1.1] * len(dates),
        "low": [close * 0.9] * len(dates),
        "close": [close] * len(dates),
        "volume": [1_000_000.0] * len(dates),
        "amount": [1e7] * len(dates),
        "turnover": [0.01] * len(dates),
    })


def _events(pairs):
    return pd.DataFrame({
        "date": pd.to_datetime([p[0] for p in pairs]),
        "hfq_factor": [p[1] for p in pairs],
    })


class TestPitAdjustment:
    def test_ratio_formula(self, fetcher):
        """除权后价格放大 = 因子比, 基准为 START_DATE 处因子"""
        dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        raw = _raw_df(dates, close=10.0)
        # 事件: 1/2 前因子 1.0 (基准), 1/4 起因子 2.0
        ev = _events([("2024-01-02", 1.0), ("2024-01-04", 2.0)])
        out = fetcher._standardize_sina_stock_pit(raw, ev, "600519.SH")
        assert list(out["adj_close"].round(6)) == [10.0, 10.0, 20.0, 20.0]

    def test_future_event_does_not_change_history(self, fetcher):
        """核心 PIT: 追加 2025 年除权事件, 2024 年复权价必须逐值不变"""
        dates = pd.bdate_range("2024-01-02", periods=10)
        raw = _raw_df(list(dates), close=15.0)
        ev_past = _events([("2024-01-02", 1.0), ("2024-03-01", 1.5)])
        ev_future = _events([("2024-01-02", 1.0), ("2024-03-01", 1.5), ("2025-06-01", 3.0)])
        a = fetcher._standardize_sina_stock_pit(raw, ev_past, "600519.SH")
        b = fetcher._standardize_sina_stock_pit(raw, ev_future, "600519.SH")
        pd.testing.assert_series_equal(a["adj_close"], b["adj_close"])
        pd.testing.assert_series_equal(a["adj_open"], b["adj_open"])

    def test_base_is_start_date_factor(self, fetcher):
        """f_base 固定为 settings.START_DATE 处因子, 与数据窗口无关 (增量拉取一致)"""
        raw_full = _raw_df(list(pd.bdate_range("2024-01-02", periods=6)), close=20.0)
        raw_tail = _raw_df(list(pd.bdate_range("2024-01-05", periods=3)), close=20.0)
        ev = _events([("2023-06-01", 2.0), ("2024-01-04", 4.0)])
        full = fetcher._standardize_sina_stock_pit(raw_full, ev, "600519.SH")
        tail = fetcher._standardize_sina_stock_pit(raw_tail, ev, "600519.SH")
        # 同一日期在两种窗口下复权值必须一致
        merged = full.merge(tail, on="date", suffixes=("_full", "_tail"))
        assert (merged["adj_close_full"] - merged["adj_close_tail"]).abs().max() < 1e-9

    def test_invalid_factor_table_fails_closed(self, fetcher):
        """因子表为空或含非正值时必须抛错, 严禁静默产出非 PIT 数据"""
        raw = _raw_df(["2024-01-02"], close=10.0)
        with pytest.raises(DataFetchError):
            fetcher._standardize_sina_stock_pit(raw, _events([]), "600519.SH")
        with pytest.raises(DataFetchError):
            fetcher._standardize_sina_stock_pit(raw, _events([("2024-01-02", 0.0)]), "600519.SH")

    def test_columns_and_metadata(self, fetcher):
        raw = _raw_df(list(pd.bdate_range("2024-01-02", periods=3)), close=10.0)
        out = fetcher._standardize_sina_stock_pit(raw, _events([("2024-01-02", 1.0)]), "600519.SH")
        for c in ["adj_open", "adj_high", "adj_low", "adj_close", "adj_pct_change"]:
            assert c in out.columns
        assert out["adjustment_mode"].eq("hfq_factor_pit").all()
        assert out["symbol"].eq("600519.SH").all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
