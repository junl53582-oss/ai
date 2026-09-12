"""业绩预告 PIT 因子矩阵验证"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from data.fundamentals_forecast import build_forecast_daily_matrix, forecast_cache_dir

market = pd.read_parquet(ROOT / 'data_storage' / 'parquet' / 'market_daily.parquet',
                         columns=['date', 'symbol'])
m = build_forecast_daily_matrix(market, start_year='2021')
out = []
out.append(f"矩阵行数: {len(m):,} | 覆盖标的: {m['symbol'].nunique()}")
cov = m['FF_TYPE_SCORE'].notna().mean()
out.append(f"因子覆盖率: {cov:.1%} (有权重内的预告覆盖日)")

# PIT 验证: 抽一只近期有预告的股票, 检查生效日 = 公告日+1
snaps = pd.concat([pd.read_parquet(f) for f in sorted(forecast_cache_dir().glob('forecast_*.parquet'))])
snaps = snaps.dropna(subset=['notice_date'])
_univ = set(market['symbol'].astype(str).unique())
snaps = snaps[snaps['symbol'].isin(_univ)]           # 只抽池内标的
latest = snaps.sort_values('notice_date').iloc[-1]
sym, nd = latest['symbol'], latest['notice_date']
sub = m[(m['symbol'] == sym) & (m['date'] >= nd - pd.Timedelta(days=3)) & (m['date'] <= nd + pd.Timedelta(days=3))].sort_values('date')
out.append(f"\nPIT 抽样: {sym} 公告日 {nd.date()} 类型 {latest['forecast_type']}")
out.append(sub.to_string(index=False))
out.append(f"\n因子分布: FF_TYPE_SCORE 取值 {sorted(m['FF_TYPE_SCORE'].dropna().unique())}")
out.append(f"FF_AMP_MID: 中位 {m['FF_AMP_MID'].median():.1f}% | FF_DAYS_SINCE 范围 {m['FF_DAYS_SINCE'].min():.0f}~{m['FF_DAYS_SINCE'].max():.0f} 天")

out.append("\n=== 分年覆盖 (非空类型分占比) ===")
m['year'] = m['date'].dt.year
out.append(m.groupby('year')['FF_TYPE_SCORE'].apply(lambda s: f"{s.notna().mean():.1%}").to_string())

Path(ROOT / 'artifacts' / 'fc_verify.txt').write_text("\n".join(str(x) for x in out), encoding='utf-8')
print('done')
