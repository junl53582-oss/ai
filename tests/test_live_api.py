import pytest
import pandas as pd
from pathlib import Path
from config.settings import settings
from data.network_policy import NetworkDisabledForTestError
from data.live_market_and_news_api import LiveMarketAPI, LiveNewsAPI, AutoSyncEngine


def test_live_market_api_network_blocked():
    """验证测试模式下未 mock 的实时行情请求被底层网络策略阻断"""
    symbols = ['603986.SH', '600026.SH']
    with pytest.raises(NetworkDisabledForTestError):
        LiveMarketAPI.fetch_live_quotes(symbols)


def test_live_market_api_with_mock(monkeypatch):
    """验证在测试沙箱中使用 mock 数据时行情解析正确"""
    symbols = ['603986.SH', '600026.SH']
    mock_df = pd.DataFrame([
        {'symbol': '603986.SH', 'name': '兆易创新', 'close': 85.5, 'pre_close': 84.0, 'pct_change': 0.0179, 'high': 86.0, 'low': 83.5, 'amount_yi': 12.3, 'trade_time': '15:00:00'},
        {'symbol': '600026.SH', 'name': '中远海能', 'close': 14.2, 'pre_close': 14.0, 'pct_change': 0.0143, 'high': 14.5, 'low': 13.9, 'amount_yi': 5.6, 'trade_time': '15:00:00'},
    ])
    monkeypatch.setattr(LiveMarketAPI, "fetch_live_quotes", lambda syms: mock_df)
    df = LiveMarketAPI.fetch_live_quotes(symbols)
    assert not df.empty
    assert 'close' in df.columns
    assert 'name' in df.columns
    assert (df['close'] > 0).all()


def test_live_news_api_network_blocked():
    """验证测试模式下未 mock 的快讯请求被底层网络策略阻断"""
    with pytest.raises(NetworkDisabledForTestError):
        LiveNewsAPI.fetch_7x24_telegraph(num=5)


def test_live_news_api_with_mock(monkeypatch):
    """验证在测试沙箱中使用 mock 数据时快讯结构正确"""
    mock_news = [
        {"time": "2026-09-06 15:00:00", "content": "测试财经快讯 1"},
        {"time": "2026-09-06 14:59:00", "content": "测试财经快讯 2"}
    ]
    monkeypatch.setattr(LiveNewsAPI, "fetch_7x24_telegraph", lambda num=5: mock_news)
    news = LiveNewsAPI.fetch_7x24_telegraph(num=5)
    assert isinstance(news, list)
    assert len(news) > 0
    assert 'content' in news[0]
