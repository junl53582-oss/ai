"""模型失效检测器 (strategy/signal_monitor.py)

原理 (Alphalens 式分位价差监控):
  每日按预测分分 5 组 → 各组 20 日前瞻收益 → Q5-Q1 日价差 → 滚动均值。
  当滚动价差转负 (模型头部组合持续跑输尾部) = 模型进入失效期 → 风控自动降档。

状态机:
  HEALTHY  滚动价差 >= 0          → 仓位系数 1.00
  DEGRADED -0.5% <= 价差 < 0      → 仓位系数 0.80
  REVERSED  价差 < -0.5%          → 仓位系数 0.60
  NO_DATA   无可用 OOS 数据        → 仓位系数 1.00 (Fail-Open, 不误伤)

数据源: 认证模型 OOS 预测 (滚动账本积累成熟后可切换为影子账本增量数据)。
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / 'reports' / 'model_research' / 'rolling_sel_20260907_163910' / 'oos_predictions_rolling_sel.parquet'

_STATE_TABLE = [
    ('REVERSED', -0.5, 0.60),   # 价差 < -0.5%
    ('DEGRADED', 0.0, 0.80),    # -0.5% <= 价差 < 0
    ('HEALTHY', float('inf'), 1.00),
]


def compute_daily_spread(source: Path, n_quantile: int = 5, horizon: int = 20) -> pd.Series:
    """按日计算 Q最高-Q最低 分组的 20 日前瞻收益价差 (%)"""
    cols = ['date', 'symbol', 'adj_close', 'pred_score']
    df = pd.read_parquet(source, columns=cols)
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    g = df.groupby('symbol', sort=False)['adj_close']
    df['fwd'] = (g.shift(-(1 + horizon)) / g.shift(-1) - 1.0) * 100
    ev = df.dropna(subset=['pred_score', 'fwd']).copy()
    ev['q'] = ev.groupby('date')['pred_score'].transform(
        lambda s: pd.qcut(s.rank(method='first'), n_quantile, labels=False) + 1)
    daily = ev.groupby(['date', 'q'])['fwd'].mean().unstack()
    spread = daily[n_quantile] - daily[1]
    return spread.dropna()


def get_signal_health(window: int = 60, source: Optional[Path] = None,
                      spread_series: Optional[pd.Series] = None) -> Dict[str, Any]:
    """返回当前模型健康状态与仓位系数 (Fail-Open: 无数据时 1.0)"""
    try:
        s = spread_series
        if s is None:
            src = Path(source) if source else DEFAULT_SOURCE
            if not src.exists():
                return {'state': 'NO_DATA', 'position_multiplier': 1.0,
                        'spread': None, 'window': window}
            s = compute_daily_spread(src)
        if len(s) < 20:
            return {'state': 'NO_DATA', 'position_multiplier': 1.0,
                    'spread': None, 'window': window}
        rolling = s.rolling(window).mean().dropna()
        if rolling.empty:
            return {'state': 'NO_DATA', 'position_multiplier': 1.0,
                    'spread': None, 'window': window}
        cur = float(rolling.iloc[-1])
        for state, threshold, mult in _STATE_TABLE:
            if cur < threshold:
                return {'state': state, 'position_multiplier': mult,
                        'spread': round(cur, 4), 'window': window,
                        'as_of': str(rolling.index[-1])[:10]}
        return {'state': 'HEALTHY', 'position_multiplier': 1.0,
                'spread': round(cur, 4), 'window': window}
    except Exception as e:
        logger.warning(f'[SignalMonitor] Fail-Open: {e}')
        return {'state': 'ERROR', 'position_multiplier': 1.0, 'spread': None,
                'window': window, 'error': str(e)[:100]}
