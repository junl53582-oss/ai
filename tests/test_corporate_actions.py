"""
公司行为 PIT 事件提供器测试 (tests/test_corporate_actions.py)

锁定 Phase A 公司行为数据源 (data/corporate_actions.py):
1. 标准化: cninfo 原始列映射到 REQUIRED_COLS, 日期/比例数值化
2. Fail-Closed 校验: 未来除权日 / 负比例必须被标记
3. 缓存与面板: 全池面板 manifest 统计
4. 真数据交叉验证: 600519 分红除权日 == hfq-factor 事件日 (两源互证)
"""
import json

import numpy as np
import pandas as pd
import pytest

from data.corporate_actions import CorporateActionProvider


@pytest.fixture()
def provider(tmp_path):
    return CorporateActionProvider(cache_dir=tmp_path)


class TestFetchNormalization:
    def test_normalize_cninfo_columns(self, provider, monkeypatch):
        """cninfo 原始列必须映射为 REQUIRED_COLS 结构"""
        raw = pd.DataFrame({
            "实施方案公告日期": ["2024-06-01", "2024-12-01"],
            "分红类型": ["年度分红", "季度分红"],
            "送股比例": ["0.0", "0.5"],
            "转增比例": ["0.0", "0.0"],
            "派息比例": ["30.0", "20.0"],
            "股权登记日": ["2024-06-05", "2024-12-05"],
            "除权日": ["2024-06-06", "2024-12-06"],
            "派息日": ["2024-06-07", "2024-12-07"],
            "实施方案分红说明": ["A", "B"],
        })
        import akshare as ak
        monkeypatch.setattr(ak, "stock_dividend_cninfo", lambda symbol: raw)
        ev = provider.fetch_events("600519.SH")
        assert {"ex_date", "announce_date", "send_ratio", "transfer_ratio", "cash_ratio"} <= set(ev.columns)
        assert ev["send_ratio"].iloc[1] == pytest.approx(0.5)
        assert ev["ex_date"].iloc[0] == pd.Timestamp("2024-06-06")
        assert ev["symbol"].eq("600519.SH").all()

    def test_empty_result(self, provider, monkeypatch):
        import akshare as ak
        monkeypatch.setattr(ak, "stock_dividend_cninfo", lambda symbol: pd.DataFrame())
        ev = provider.fetch_events("000001.SZ")
        assert ev.empty


class TestValidation:
    def test_future_ex_date_flagged(self, provider):
        ev = pd.DataFrame({
            "ex_date": [pd.Timestamp("2099-01-01")],
            "announce_date": [pd.Timestamp("2099-01-01")],
            "send_ratio": [0.0], "transfer_ratio": [0.0], "cash_ratio": [1.0],
        })
        viol = provider.validate_events(ev)
        assert any("未来" in v for v in viol)

    def test_negative_ratio_flagged(self, provider):
        ev = pd.DataFrame({
            "ex_date": [pd.Timestamp("2024-01-01")],
            "announce_date": [pd.Timestamp("2024-01-01")],
            "send_ratio": [-0.5], "transfer_ratio": [0.0], "cash_ratio": [1.0],
        })
        viol = provider.validate_events(ev)
        assert any("负" in v for v in viol)

    def test_valid_events_pass(self, provider):
        ev = pd.DataFrame({
            "ex_date": [pd.Timestamp("2024-01-01")],
            "announce_date": [pd.Timestamp("2023-12-20")],
            "send_ratio": [0.0], "transfer_ratio": [0.0], "cash_ratio": [2.5],
        })
        assert provider.validate_events(ev) == []


class TestPanelAndCache:
    def test_panel_manifest(self, provider, tmp_path, monkeypatch):
        import akshare as ak
        raw = pd.DataFrame({
            "实施方案公告日期": ["2024-06-01"], "分红类型": ["年度分红"],
            "送股比例": ["0.0"], "转增比例": ["0.0"], "派息比例": ["30.0"],
            "股权登记日": ["2024-06-05"], "除权日": ["2024-06-06"],
            "派息日": ["2024-06-07"], "实施方案分红说明": ["A"],
        })
        monkeypatch.setattr(ak, "stock_dividend_cninfo", lambda symbol: raw)
        panel = provider.build_universe_panel(["600519.SH", "000001.SZ"])
        assert panel["symbol"].nunique() == 2
        manifest = json.loads((tmp_path / "corporate_actions_manifest.json").read_text(encoding="utf-8"))
        assert manifest["symbol_count"] == 2
        assert manifest["event_count"] == 2

    def test_cache_reuse(self, provider, tmp_path, monkeypatch):
        import akshare as ak
        calls = {"n": 0}
        raw = pd.DataFrame({
            "实施方案公告日期": ["2024-06-01"], "分红类型": ["年度分红"],
            "送股比例": ["0.0"], "转增比例": ["0.0"], "派息比例": ["30.0"],
            "股权登记日": ["2024-06-05"], "除权日": ["2024-06-06"],
            "派息日": ["2024-06-07"], "实施方案分红说明": ["A"],
        })
        def fake(symbol):
            calls["n"] += 1
            return raw
        monkeypatch.setattr(ak, "stock_dividend_cninfo", fake)
        provider.get_event_stream("600519.SH")
        provider.get_event_stream("600519.SH")  # 第二次走缓存
        assert calls["n"] == 1
        assert (tmp_path / "600519.parquet").exists()


class TestCrossSourceConsistency:
    def test_cninfo_ex_dates_match_hfq_factor_events(self):
        """真数据交叉验证 (网络): 600519 的 cninfo 除权日 == hfq-factor 事件日"""
        import akshare as ak
        prov = CorporateActionProvider()
        ev = prov.fetch_events("600519.SH")
        fac = ak.stock_zh_a_daily(symbol="sh600519", adjust="hfq-factor")
        fac["date"] = pd.to_datetime(fac["date"])
        factor_dates = set(fac["date"].dt.date)
        cninfo_dates = set(ev["ex_date"].dropna().dt.date)
        overlap = cninfo_dates & factor_dates
        assert len(overlap) >= max(1, len(cninfo_dates) - 2), \
            f"cninfo 除权日与 hfq 因子事件日几乎无重叠 (overlap={len(overlap)})"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
