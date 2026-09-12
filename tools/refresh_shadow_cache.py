"""影子缓存净化重训: 用加严隔离带 (B=41天) 重新训练并刷新 A/B 打分缓存"""
import io
import sys
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from strategy.shadow_scorer import compute_shadow_scores

top = pd.read_csv(ROOT / 'artifacts' / 'latest_stock_picks.csv').head(10)[['symbol', 'name', 'close', 'pred_score']]
out, meta = compute_shadow_scores(top, force_refresh=True)
b = meta.get('variant_b', {})
c = meta.get('variant_c', {})
print(f"[结果] A: as_of={meta.get('as_of')} err={meta.get('error')} | "
      f"B: as_of={b.get('as_of')} err={b.get('error')} | "
      f"C: as_of={c.get('as_of')} err={c.get('error')}")
print(f"[样本] A非空 {int(out['shadow_score'].notna().sum())}/{len(out)} | "
      f"B非空 {int(out['shadow_score_b'].notna().sum())}/{len(out)} | "
      f"C非空 {int(out['shadow_score_c'].notna().sum())}/{len(out)}")
print(f"[C因子] {c.get('factors')}")
