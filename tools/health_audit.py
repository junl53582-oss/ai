"""项目健康不变量审计: 关键安全配置与数据完整性"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

out = []
from config.settings import settings

# 1. 安全不变量
out.append("=== 安全不变量 ===")
out.append(f"LIVE_TRADING_READY = {settings.LIVE_TRADING_READY}  (必须为 False)")
out.append(f"PURGE_GAP_DAYS = {settings.PURGE_GAP_DAYS} | LABEL_HORIZON = {settings.LABEL_HORIZON} "
           f"(purge >= horizon+1 才安全: {'OK' if settings.PURGE_GAP_DAYS >= settings.LABEL_HORIZON + 1 else '⚠️'})")
out.append(f"FUNDAMENTAL_DELAY_DAYS = {settings.FUNDAMENTAL_DELAY_DAYS} (PIT 保守延迟)")

# 2. 影子三轨隔离带
out.append("\n=== 影子三轨隔离带 (实时) ===")
import strategy.shadow_scorer as ss
out.append("A: label 20 → purge max(25, 21) = 25")
out.append("B: label 40 → purge max(25, 41) = 41")
out.append("C: label 40 → purge max(25, 41) = 41")
for f, tag in [(ss.CACHE_FILE, 'A'), (ss.CACHE_FILE_B, 'B'), (ss.CACHE_FILE_C, 'C')]:
    if f.exists():
        d = json.loads(f.read_text(encoding='utf-8'))
        out.append(f"  缓存{tag}: as_of={d.get('as_of')} | n={len(d.get('scores', {}))}")

# 3. 数据完整性
out.append("\n=== 数据完整性 ===")
import pandas as pd
mat = pd.read_parquet(ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet', columns=['date', 'symbol'])
out.append(f"因子矩阵: {len(mat):,} 行 | date_max={pd.to_datetime(mat['date']).max().date()} | {mat['symbol'].nunique()} 标的")
mkt = pd.read_parquet(ROOT / 'data_storage' / 'parquet' / 'market_daily.parquet', columns=['date'])
out.append(f"行情底座: {len(mkt):,} 行 | date_max={pd.to_datetime(mkt['date']).max().date()}")

# 4. 台账与自动化产物
led = ROOT / 'data_storage' / 'research' / 'shadow_reconciliation_ledger.csv'
if led.exists():
    l = pd.read_csv(led)
    out.append(f"\n对账台账: {len(l)} 天 | 列={list(l.columns)}")
out.append(f"资金流文件: {len(list((ROOT/'data_storage'/'fundflow').glob('*.parquet')))} 天")
out.append(f"预告缓存: {len(list((ROOT/'data_storage'/'fundamentals_forecast').glob('*.parquet')))} 期")
out.append(f"财报缓存: {len(list((ROOT/'data_storage'/'cache' if False else ROOT/'data_storage').rglob('yjbb_*.parquet')))} 期")

# 5. 清单权重不变量
picks = pd.read_csv(ROOT / 'artifacts' / 'latest_stock_picks.csv')
neg = int((picks['target_weight'] < 0).sum())
out.append(f"\n清单权重: n={len(picks)} | 负权重={neg} (必须 0) | sum={picks['target_weight'].sum():.4f}")
out.append(f"清单含影子列: {[c for c in picks.columns if 'shadow' in c] or '(面板打开时注入)'}")

Path(ROOT / 'artifacts' / 'health_audit.txt').write_text("\n".join(out), encoding='utf-8')
print('done')
