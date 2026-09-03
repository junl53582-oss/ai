import pytest
import pandas as pd
from factors.sentiment_engine import MarketSentimentDetector, NewsCatalystScorer

def test_market_sentiment_detector():
    dates = pd.date_range('2026-09-01', periods=2, freq='D')
    rows = []
    for d in dates:
        rows.append({'date': d, 'symbol': '000001.SZ', 'pct_change': 0.05})
        rows.append({'date': d, 'symbol': '000002.SZ', 'pct_change': -0.01})
    df = pd.DataFrame(rows)
    res = MarketSentimentDetector.evaluate_market_temperature(df, '2026-09-02')
    assert 'temperature' in res
    assert 'stage' in res
    assert 0 <= res['temperature'] <= 100

def test_news_catalyst_scorer():
    info = NewsCatalystScorer.get_stock_catalyst('600026.SH')
    assert 'headline' in info
    assert 'VLCC' in info['headline']
    assert info['sentiment_score'] >= 90
