"""泄漏审计: 40 日标签实验在严格隔离 (purge=41) 下的净化重测

背景: 40 日标签的前瞻收益窗口 (40 日) 超过了默认隔离带 (PURGE_GAP_DAYS=25,
为 20 日标签设计), 存在 15 天的标签/测试期重叠 = 标签前视泄漏。
本实验以 purge=41 (>= 40+1) 重跑两个 40 日变体, 与已污染数字对比:
  - 40d pure  (污染值: full +0.0452, 2026 +0.0126)
  - 40d + 正交化F (污染值: full +0.0557, 2026 +0.0446)  ← 影子B当前配置
基线 (20d, purge=25, 无泄漏): full +0.0272, 2026 -0.0020
判定: 若净化后 40d 优势消失 → 泄漏作弊实锤, 回退 20d; 若保留 → 改进真实。
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
from research_v2.labels.label_registry import LabelRegistry
from models.walk_forward import WalkForwardTrainer

HORIZON = 40
PURGE = HORIZON + 1  # 41 天严格隔离 (>= 前瞻窗口)
LABEL_COL = 'label_exp40'
WINDOWS = {'full_oos': ('2021-09-01', '2026-09-07'), 'y2026': ('2026-01-01', '2026-09-07')}
POOL_FILE = ROOT / 'reports' / 'model_research' / 'candidate_pool_top40.txt'
LOG = ROOT / 'artifacts' / 'leakage_audit_40d.log'


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
    fp = FundamentalsProvider(delay_days=settings.FUNDAMENTAL_DELAY_DAYS)
    fund_daily = fp.build_daily_fundamental_matrix(market_df, start_year=settings.FUNDAMENTAL_START_YEAR,
                                                   fetch_if_empty=False)
    fund_daily['date'] = pd.to_datetime(fund_daily['date'])
    facs = [f for f in FUNDAMENTAL_FACTOR_NAMES if f in fund_daily.columns]
    df = df.merge(fund_daily[['symbol', 'date'] + facs], on=['symbol', 'date'], how='left')
    raw = LabelRegistry.compute_label_v2(df, horizon=HORIZON)
    df[LABEL_COL] = (raw > 0).astype(float).mask(raw.isna(), np.nan)
    pool = [x.strip() for x in POOL_FILE.read_text(encoding='utf-8').splitlines() if x.strip()]
    pool = [f for f in pool if f in df.columns]
    return df, facs, pool


def orthogonalize(df, facs, pool):
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
            resid = y.values - X @ beta
            d.loc[grp.index, f] = np.where(valid, resid, np.nan)
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
    m, sd = float(s.mean()), float(s.std())
    return {'window': tag, 'n_days': int(len(s)), 'mean_rank_ic': round(m, 5),
            'icir_annualized': round(m / (sd + 1e-9) * np.sqrt(6), 3),
            'positive_ratio': round(float((s > 0).mean()), 3)}


def run(df, pool, tag):
    log(f"[{tag}] 训练开始 (池 {len(pool)}, purge={PURGE})")
    t0 = time.time()
    trainer = WalkForwardTrainer(
        model_type='lightgbm_ranker', task_type='classification', label_col=LABEL_COL,
        feature_selection_method='rank_ic_pruned', top_k_features=10,
        strict_mode=False, save_model=False, purge_gap_days=PURGE)
    oos_df, _ = trainer.run_walk_forward(df, feature_cols=pool)
    log(f"[{tag}] 完成 30 折, {time.time()-t0:.0f}s")
    return [window_metrics(oos_df, s, e, t) for t, (s, e) in WINDOWS.items()]


def main() -> int:
    log(">>> 泄漏审计: 40 日标签净化重测 (purge=41)")
    df, facs, pool = load_base()

    d_orth = orthogonalize(df, facs, pool)
    r_orth = run(d_orth, pool + facs, '40d_orthoF_purge41')
    log(f"[40d_orthoF结果] " + json.dumps(r_orth, ensure_ascii=False))

    out = {'generated_at': datetime.now().isoformat(timespec='seconds'),
           'purge_days': PURGE,
           'contaminated_numbers': {'40d_pure': {'full': 0.0452, 'y2026': 0.0126},
                                    '40d_orthoF': {'full': 0.0557, 'y2026': 0.0446}},
           'baseline_clean_20d': {'full': 0.0272, 'y2026': -0.0020},
           'ortho': r_orth}
    out_path = ROOT / 'reports' / 'model_research' / f'leakage_audit_40d_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f">>> 完成, 落盘: {out_path}")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(f"CRASH:\n{traceback.format_exc()}\n")
        raise
