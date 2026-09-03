import pytest
from pathlib import Path
from data.live_market_and_news_api import LiveMarketAPI, LiveNewsAPI, AutoSyncEngine

def test_live_market_api():
    symbols = ['603986.SH', '600026.SH']
    df = LiveMarketAPI.fetch_live_quotes(symbols)
    assert not df.empty
    assert 'close' in df.columns
    assert 'name' in df.columns
    assert (df['close'] > 0).all()

def test_live_news_api():
    news = LiveNewsAPI.fetch_7x24_telegraph(num=5)
    assert isinstance(news, list)
    assert len(news) > 0
    assert 'content' in news[0]
