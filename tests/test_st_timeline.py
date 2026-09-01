"""
ST 时间线提供器测试 (tests/test_st_timeline.py)

锁定 data/st_timeline.py:
1. 深市简称变更表带日期
2. ST 周期推导: 名称含 ST 的状态切换正确折叠为周期
3. 单标的查询: 深市返回周期, 沪市返回空并告警
4. 覆盖统计: 诚实记录深市/沪市差异
"""
import pandas as pd
import pytest

from data.st_timeline import STTimelineProvider


@pytest.fixture()
def provider(tmp_path):
    return STTimelineProvider(cache_dir=tmp_path)


class TestSZPeriods:
    def test_period_folding(self, provider, monkeypatch):
        """名称切换应折叠为 ST 周期 (进入/退出日期)"""
        fake = pd.DataFrame({
            "变更日期": ["2020-01-01", "2021-06-01", "2022-01-01", "2023-01-01"],
            "证券代码": ["000711", "000711", "000711", "000711"],
            "证券简称": ["京蓝科技", "*ST京蓝", "京蓝科技", "ST京蓝"],
            "变更前简称": ["A", "京蓝科技", "*ST京蓝", "京蓝科技"],
            "变更后简称": ["京蓝科技", "*ST京蓝", "京蓝科技", "ST京蓝"],
        })
        monkeypatch.setattr(provider, "fetch_sz_name_change_table",
                            lambda refresh=False: fake)
        periods = provider.fetch_sz_st_periods()
        p711 = periods[periods["证券代码"] == "000711"]
        assert len(p711) == 2
        row = p711.iloc[0]
        assert row["st_start"] == pd.Timestamp("2021-06-01")
        assert row["st_end"] == pd.Timestamp("2022-01-01")

    def test_get_periods_sz(self, provider, monkeypatch):
        fake = pd.DataFrame({
            "变更日期": ["2020-01-01", "2021-06-01", "2022-01-01"],
            "证券代码": ["000711", "000711", "000711"],
            "证券简称": ["A", "B", "C"],
            "变更前简称": ["", "A", "B"],
            "变更后简称": ["京蓝科技", "*ST京蓝", "京蓝科技"],
        })
        monkeypatch.setattr(provider, "fetch_sz_name_change_table",
                            lambda refresh=False: fake)
        res = provider.get_st_periods("000711.SZ")
        assert res == [("2021-06-01", "2022-01-01", "ST")]

    def test_get_periods_sh_returns_empty(self, provider):
        """沪市无日期源: 返回空 (缺口文档化, 不伪造)"""
        assert provider.get_st_periods("600519.SH") == []


class TestCoverage:
    def test_coverage_counts(self, provider, monkeypatch):
        fake = pd.DataFrame({
            "变更日期": ["2021-06-01"],
            "证券代码": ["000711"],
            "证券简称": ["X"],
            "变更前简称": ["A"],
            "变更后简称": ["*ST京蓝"],
        })
        monkeypatch.setattr(provider, "fetch_sz_name_change_table",
                            lambda refresh=False: fake)
        cov = provider.build_universe_coverage(["000711.SZ", "600519.SH", "000001.SZ"])
        assert cov["requested_symbols"] == 3
        assert cov["sz_covered_with_dates"] == 1
        assert cov["sh_uncovered_no_date_source"] == 1
        assert cov["coverage_ratio"] == pytest.approx(1 / 3)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
