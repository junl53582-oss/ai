"""资金流代理因子构造与 IC 筛选 (Phase A)

因子 (全部可由现有 OHLCV 全历史构造, 无外部数据依赖):
  MFI14        14 日资金流量指数 (典型价 × 量的多空比)
  CMF20        20 日 Chaikin 资金流 (聚散度)
  OBV_SLOPE20  OBV 20 日线性斜率 (量能趋势方向)
  VWAP_DEV20   收盘价相对 20 日成交均价偏离
  VOL_RATIO20  量比 (当日量 / 20 日均量)
  AD_SLOPE20   A/D 线 20 日斜率 (吸筹/派发)
先算这些因子的逐日 RankIC (全期/分年), 看有无信号, 再决定是否进走步池。
"""
import io
import sys
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

LABEL = 'label_up_down_20d'


def load_market() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / 'data_storage' / 'parquet' / 'market_daily.parquet',
                         columns=['date', 'symbol', 'high', 'low', 'close', 'volume', 'amount', 'pct_change'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    return df


def add_moneyflow_factors(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby('symbol', sort=False)
    tp = (df['high'] + df['low'] + df['close']) / 3.0

    # MFI14
    pos_flow = ((tp > tp.shift(1)) * tp * df['volume']).where(lambda x: x > 0, 0.0)
    neg_flow = ((tp < tp.shift(1)) * tp * df['volume']).where(lambda x: x > 0, 0.0)
    pos14 = pos_flow.groupby(df['symbol'], sort=False).rolling(14).sum().reset_index(level=0, drop=True)
    neg14 = neg_flow.groupby(df['symbol'], sort=False).rolling(14).sum().reset_index(level=0, drop=True)
    mfr = pos14 / neg14.replace(0, np.nan)
    df['MFI14'] = (100 - 100 / (1 + mfr)).clip(0, 100)

    # CMF20
    rng = (df['high'] - df['low']).replace(0, np.nan)
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / rng
    mfv = mfm * df['volume']
    df['CMF20'] = (mfv.groupby(df['symbol'], sort=False).rolling(20).sum().reset_index(level=0, drop=True)
                   / df['volume'].groupby(df['symbol'], sort=False).rolling(20).sum().reset_index(level=0, drop=True))

    # OBV 斜率20 (OBV 的 20 日线性回归斜率 / 收盘价, 归一)
    obv = (np.sign(df['close'].diff()) * df['volume']).groupby(df['symbol'], sort=False).cumsum()
    df['OBV_SLOPE20'] = (obv.groupby(df['symbol'], sort=False).rolling(20).apply(
        lambda s: np.polyfit(np.arange(len(s)), s, 1)[0], raw=True
    ).reset_index(level=0, drop=True) / df['close'])

    # VWAP 偏离 20
    vwap20 = (df['amount'].groupby(df['symbol'], sort=False).rolling(20).sum().reset_index(level=0, drop=True)
              / df['volume'].groupby(df['symbol'], sort=False).rolling(20).sum().reset_index(level=0, drop=True))
    df['VWAP_DEV20'] = df['close'] / vwap20 - 1.0

    # 量比
    df['VOL_RATIO20'] = df['volume'] / df['volume'].groupby(df['symbol'], sort=False).rolling(20).mean().reset_index(level=0, drop=True)

    # A/D 斜率20
    ad = (mfm * df['volume']).groupby(df['symbol'], sort=False).cumsum()
    df['AD_SLOPE20'] = (ad.groupby(df['symbol'], sort=False).rolling(20).apply(
        lambda s: np.polyfit(np.arange(len(s)), s, 1)[0], raw=True
    ).reset_index(level=0, drop=True) / df['close'])

    return df


def main() -> int:
    print('>>> 加载行情 (OHLCV)...', flush=True)
    mkt = load_market()
    print(f'    {len(mkt):,} 行 | {mkt["symbol"].nunique()} 标的 | {mkt["date"].min().date()} ~ {mkt["date"].max().date()}', flush=True)

    print('>>> 构造资金流因子...', flush=True)
    mkt = add_moneyflow_factors(mkt)
    factors = ['MFI14', 'CMF20', 'OBV_SLOPE20', 'VWAP_DEV20', 'VOL_RATIO20', 'AD_SLOPE20']

    # 标签: 20 日前瞻收益方向 (v2 可执行口径)
    # close 口径前瞻收益 (T+1收盘 -> T+21收盘), 因子筛选用途足够
    raw = (mkt.groupby('symbol')['close'].shift(-21)
           / mkt.groupby('symbol')['close'].shift(-1) - 1.0)
    mkt[LABEL] = (raw > 0).astype(float).mask(raw.isna(), np.nan)

    print('>>> 逐日 RankIC 筛选...', flush=True)
    rows = []
    for f in factors:
        daily = {}
        for dt, grp in mkt.dropna(subset=[f, LABEL]).groupby('date'):
            if len(grp) < 30:
                continue
            r = stats.spearmanr(grp[f], grp[LABEL])[0]
            if np.isfinite(r):
                daily[dt] = r
        s = pd.Series(daily).sort_index()
        if s.empty:
            rows.append({'factor': f, 'n_days': 0})
            continue
        yearly = s.groupby(s.index.year).mean()
        rows.append({
            'factor': f,
            'n_days': len(s),
            'full_ic': round(s.mean(), 5),
            'icir': round(s.mean() / (s.std() + 1e-9) * np.sqrt(242), 3),
            'pos_ratio': round((s > 0).mean(), 3),
            'ic_2023': round(yearly.get(2023, np.nan), 5),
            'ic_2024': round(yearly.get(2024, np.nan), 5),
            'ic_2025': round(yearly.get(2025, np.nan), 5),
            'ic_2026': round(yearly.get(2026, np.nan), 5),
        })

    t = pd.DataFrame(rows)
    out = ROOT / 'reports' / 'model_research' / f'moneyflow_factor_screen_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv'
    t.to_csv(out, index=False, encoding='utf-8-sig')
    print(t.to_string(index=False), flush=True)
    print(f'\n>>> 已落盘: {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
