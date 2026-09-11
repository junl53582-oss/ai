"""诊断: inv_vol 优化器负权重 bug 复现 (STD20 为标准化因子, 负值取倒数 → 负权重)"""
import io
import sys
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from strategy.portfolio import PortfolioBuilder

df = pd.read_parquet(ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet')
df['date'] = pd.to_datetime(df['date'])
latest = df[df['date'] == df['date'].max()].copy()
print(f"截面: {latest['date'].iloc[0].date()} | {len(latest)} 标的")

vol_candidates = [c for c in ['STD20', 'ATR_RATIO_14', 'ATR_RATIO_20', 'YANG_ZHANG_VOL_20', 'ATR_RATIO_5'] if c in latest.columns]
print('可用波动率列:', vol_candidates)
for c in vol_candidates:
    v = pd.to_numeric(latest[c], errors='coerce')
    print(f"  {c}: min={v.min():.4f} max={v.max():.4f} 负值占比={(v < 0).mean():.1%}")

latest['pred_score'] = np.random.RandomState(42).rand(len(latest))
builder = PortfolioBuilder(top_k_buy=8, top_k_hold=15, weight_method='inv_vol')
out = builder.build_target_portfolio(latest, current_holdings=set(), date=latest['date'].iloc[0])
w = out['target_weight']
print(f"\n复现结果: n={len(w)} | sum={w.sum():.4f} | min={w.min():.4f} | 负权重数={(w < 0).sum()}")
print(out[['symbol', 'target_weight']].head(8).to_string(index=False))
