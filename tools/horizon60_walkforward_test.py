"""业绩预告因子走步实验: 60日标签 + 正交化F + 预告因子 (purge=41 净化协议)

对比基线:
  干净 20d 基线:        full IC +0.0272 | 2026 -0.0020
  40d + 正交化F (净化): full IC +0.0477 | 2026 +0.0433   ← 现有最佳 (影子B配置)
本实验新增: FF_TYPE_SCORE / FF_AMP_MID / FF_DAYS_SINCE (公告日+1 生效, 真实 PIT)
判定: 预告因子能否在现有最佳之上再提供增量。
"""
import io
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats as st

from config.settings import settings
from data.fundamentals import FundamentalsProvider, FUNDAMENTAL_FACTOR_NAMES
from data.fundamentals_forecast import build_forecast_daily_matrix
from research_v2.labels.label_registry import LabelRegistry
from models.walk_forward import WalkForwardTrainer

HORIZON = 60
PURGE = HORIZON + 1
LABEL_COL = 'label_exp60fc'
WINDOWS = {'full_oos': ('2021-09-01', '2026-09-07'), 'y2026': ('2026-01-01', '2026-09-07')}
FC_FACTORS = ['FF_TYPE_SCORE', 'FF_AMP_MID', 'FF_DAYS_SINCE']
POOL_FILE = ROOT / 'reports' / 'model_research' / 'candidate_pool_top40.txt'
LOG = ROOT / 'artifacts' / 'horizon60_walkforward.log'


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def load_base():
    df = pd.read_parquet(ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet')
    df['date'] = pd.to_datetime(df['date'])
    market_df = pd.read_parquet(ROOT / 'data_storage' / 'parquet' / 'market_daily.parquet',
                                columns=['date', 'symbol', 'high', 'low', 'close', 'volume', 'amount', 'pct_change'])
    market_df['date'] = pd.to_datetime(market_df['date'])
    # 正式财报因子 (110天延迟 PIT)
    fp = FundamentalsProvider(delay_days=settings.FUNDAMENTAL_DELAY_DAYS)
    fund_daily = fp.build_daily_fundamental_matrix(market_df, start_year=settings.FUNDAMENTAL_START_YEAR,
                                                   fetch_if_empty=False)
    fund_daily['date'] = pd.to_datetime(fund_daily['date'])
    facs = [f for f in FUNDAMENTAL_FACTOR_NAMES if f in fund_daily.columns]
    df = df.merge(fund_daily[['symbol', 'date'] + facs], on=['symbol', 'date'], how='left')

    # 业绩预告因子 (公告日+1 生效)
    log("构建业绩预告日频矩阵...")
    fc = build_forecast_daily_matrix(market_df, start_year='2021')
    fc['date'] = pd.to_datetime(fc['date'])
    df = df.merge(fc, on=['symbol', 'date'], how='left')
    log(f"预告因子覆盖: {int(df['FF_TYPE_SCORE'].notna().sum()):,} 行")

    raw = LabelRegistry.compute_label_v2(df, horizon=HORIZON)
    df[LABEL_COL] = (raw > 0).astype(float).mask(raw.isna(), np.nan)
    pool = [x.strip() for x in POOL_FILE.read_text(encoding='utf-8').splitlines() if x.strip()]
    pool = [f for f in pool if f in df.columns]
    return df, facs, pool


def orthogonalize_fundamentals(df, facs, pool):
    """F 因子对 Top40 逐日残差化 (2026 修复胜出配置 V2)"""
    d = df.copy()
    regs = [f for f in pool if f in d.columns and f not in facs]
    for dt, grp in d.groupby('date'):
        if len(grp) < 40:
            continue
        X = grp[regs].apply(pd.to_numeric, errors='coerce')
        X = (X - X.mean()) / (X.std() + 1e-12)
        X = X.fillna(0.0).values
        X = np.column_stack([np.ones(len(X)), X])
        for f in facs:
            y = pd.to_numeric(d.loc[grp.index, f], errors='coerce')
            valid = y.notna().values
            if valid.sum() < 40:
                continue
            beta, *_ = np.linalg.lstsq(X[valid], y.values[valid], rcond=None)
            d.loc[grp.index, f] = np.where(valid, y.values - X @ beta, np.nan)
    return d


def window_metrics(oos_df, start, end, tag):
    sub = oos_df[(oos_df['date'] >= start) & (oos_df['date'] <= end)]
    ics = {}
    for dt, grp in sub.groupby('date'):
        g = grp[['pred_score', LABEL_COL]].dropna()
        if len(g) < 8:
            continue
        r = st.spearmanr(g['pred_score'], g[LABEL_COL])[0]
        if np.isfinite(r):
            ics[dt] = float(r)
    if not ics:
        return {'window': tag, 'n_days': 0}
    s = pd.Series(ics).sort_index()
    return {'window': tag, 'n_days': int(len(s)), 'mean_rank_ic': round(float(s.mean()), 5),
            'positive_ratio': round(float((s > 0).mean()), 3)}


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    log(">>> 业绩预告因子走步实验 (60d标签 + 正交化F + 预告因子, purge=41)")
    df, facs, pool = load_base()
    d = orthogonalize_fundamentals(df, facs, pool)
    log("正交化完成, 训练开始...")
    t0 = time.time()
    trainer = WalkForwardTrainer(
        model_type='lightgbm_ranker', task_type='classification', label_col=LABEL_COL,
        feature_selection_method='rank_ic_pruned', top_k_features=10,
        strict_mode=False, save_model=False, purge_gap_days=PURGE)
    feats = pool + facs + FC_FACTORS
    oos_df, _ = trainer.run_walk_forward(d, feature_cols=feats)
    log(f"完成 {len(trainer.models)} 折, {time.time()-t0:.0f}s")
    res = [window_metrics(oos_df, s, e, t) for t, (s, e) in WINDOWS.items()]
    log("[结果] " + json.dumps(res, ensure_ascii=False))
    out = {'generated_at': datetime.now().isoformat(timespec='seconds'), 'purge_days': PURGE,
           'baseline_clean_20d': {'full': 0.0272, 'y2026': -0.0020},
           'best_so_far_40d_orthoF_forecast': {'full': 0.0491, 'y2026': 0.0464},
           'variant_60d_orthoF_forecast': res}
    p = ROOT / 'reports' / 'model_research' / f'horizon60_walkforward_{datetime.now():%Y%m%d_%H%M%S}.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f">>> 完成, 落盘: {p}")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(f"CRASH:\n{traceback.format_exc()}\n")
        raise
