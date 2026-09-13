"""每日影子对账器: 让"验收报告"每天被消费一次

流程 (每日 20:00 计划任务调用):
  1. 快照: 把当日影子缓存 (A/B 分) 存档到 snapshots/ 目录
  2. 对账: 对所有"次日实际涨跌已就绪"的历史快照, 计算
     IC (分-实际涨跌 Spearman) + 高低组差 → 追加到对账台账
  3. 台账: data_storage/research/shadow_reconciliation_ledger.csv
     (G2 期中考的原始数据源: 20-30 天后按此裁决 A/B 晋级)
幂等: 已对账的日期自动跳过。
"""
import io
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats as st

CACHE_A = ROOT / 'data_storage' / 'cache' / 'shadow_scores_latest.json'
CACHE_B = ROOT / 'data_storage' / 'cache' / 'shadow_scores_b_latest.json'
CACHE_C = ROOT / 'data_storage' / 'cache' / 'shadow_scores_c_latest.json'
SNAP_DIR = ROOT / 'data_storage' / 'research' / 'shadow_score_snapshots'
LEDGER = ROOT / 'data_storage' / 'research' / 'shadow_reconciliation_ledger.csv'
MARKET = ROOT / 'data_storage' / 'parquet' / 'market_daily.parquet'


def main() -> int:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 快照当日缓存
    snap_paths = {}
    for label, src in [('A', CACHE_A), ('B', CACHE_B), ('C', CACHE_C)]:
        if src.exists():
            dst = SNAP_DIR / f"scores_{label}_{datetime.now():%Y%m%d}.json"
            if not dst.exists():
                shutil.copy2(src, dst)
            snap_paths[label] = dst
    print(f"[快照] {list(snap_paths.keys())} 已就绪", flush=True)

    # 2. 收集所有快照日
    snap_dates = sorted({p.stem.split('_')[-1] for p in SNAP_DIR.glob('scores_*.json')})
    if not snap_dates:
        print("[-] 无快照", flush=True)
        return 0

    mkt = pd.read_parquet(MARKET, columns=['date', 'symbol', 'pct_change'])
    mkt['date'] = pd.to_datetime(mkt['date'])
    mkt_by_date = {d: g.set_index('symbol')['pct_change'] for d, g in mkt.groupby('date')}
    all_dates = sorted(mkt['date'].dt.strftime('%Y%m%d').unique())

    # 3. 逐快照日对账 (次日涨跌已就绪 且 未入台账)
    done_dates = set()
    if LEDGER.exists():
        try:
            done_dates = set(pd.read_csv(LEDGER, dtype={'snap_date': str})['snap_date'].astype(str))
        except Exception:
            done_dates = set()

    records = []
    for d in snap_dates:
        if d in done_dates:
            continue
        d_idx = all_dates.index(d) if d in all_dates else None
        if d_idx is None or d_idx + 1 >= len(all_dates):
            continue  # 次日行情未就绪
        next_d = all_dates[d_idx + 1]
        realized = mkt_by_date.get(pd.Timestamp(next_d))
        if realized is None:
            continue

        row = {'snap_date': d, 'reconcile_date': next_d, 'n_symbols': 0}
        for label in ['A', 'B', 'C']:
            f = SNAP_DIR / f"scores_{label}_{d}.json"
            if not f.exists():
                continue
            scores = pd.Series({k: float(v) for k, v in
                                json.loads(f.read_text(encoding='utf-8'))['scores'].items()})
            cmp = pd.concat([scores.rename('s'), realized.rename('r')], axis=1).dropna()
            if len(cmp) < 30:
                continue
            ic = float(st.spearmanr(cmp['s'], cmp['r'])[0])
            hi = cmp[cmp['s'] >= cmp['s'].quantile(0.8)]['r'].mean()
            lo = cmp[cmp['s'] <= cmp['s'].quantile(0.2)]['r'].mean()
            row[f'ic_{label}'] = round(ic, 4)
            row[f'hi_lo_spread_{label}'] = round(float(hi - lo), 4)
            row['n_symbols'] = len(cmp)
        if f'ic_A' in row or f'ic_B' in row or f'ic_C' in row:
            records.append(row)
            print(f"[对账] {d} -> {next_d}: IC_A={row.get('ic_A')} IC_B={row.get('ic_B')} "
                  f"IC_C={row.get('ic_C')} | 价差A={row.get('hi_lo_spread_A')} "
                  f"价差B={row.get('hi_lo_spread_B')} 价差C={row.get('hi_lo_spread_C')}", flush=True)

    if not records:
        print("[skip] 无新增可对账日期", flush=True)
        return 0

    new = pd.DataFrame(records)
    old = pd.read_csv(LEDGER, dtype={'snap_date': str}) if LEDGER.exists() else pd.DataFrame()
    ledger = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=['snap_date'])
    ledger = ledger.sort_values('snap_date')
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER, index=False, encoding='utf-8-sig')

    # 4. 累计摘要
    if len(ledger) >= 3:
        print(f"\n[累计摘要] 已对账 {len(ledger)} 天", flush=True)
        print(f"  A赛道 平均IC: {ledger['ic_A'].mean():+.4f} | 正向天数占比: "
              f"{(ledger['ic_A'] > 0).mean()*100:.0f}%", flush=True)
        if 'ic_B' in ledger.columns:
            print(f"  B赛道 平均IC: {ledger['ic_B'].mean():+.4f} | 正向天数占比: "
                  f"{(ledger['ic_B'] > 0).mean()*100:.0f}%", flush=True)
        if 'ic_C' in ledger.columns and ledger['ic_C'].notna().any():
            print(f"  C赛道 平均IC: {ledger['ic_C'].mean():+.4f} | 正向天数占比: "
                  f"{(ledger['ic_C'] > 0).mean()*100:.0f}%", flush=True)
    print(f"\n[+] 台账已更新: {LEDGER}", flush=True)

    # 5. G2 期中考自动裁决 (满 20 个交易日后自动出裁决书; Fail-Open)
    try:
        from tools.g2_gate_verdict import run_verdict
        v = run_verdict(write=True)
        if v.get('ready'):
            print(f"[G2] 裁决书已生成: " + " | ".join(
                f"{t['label'].split(' ')[0]} {t['verdict']}" for t in v['tracks'].values()), flush=True)
        else:
            print(f"[G2] 观察期 {v.get('days')}/{v.get('min_days')} 天, 未到期", flush=True)
    except Exception as e:
        print(f"[G2] 裁决器 Fail-Open: {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
