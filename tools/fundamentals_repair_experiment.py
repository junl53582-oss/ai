"""F 因子 2026 修复实验: 两个修复变体 vs 组合基线 (40日+基本面, full +0.0568/ICIR 8.0, 2026 -0.0172)

V1 行业中性化: F 因子逐日按行业内去均值 (剥离行业风格暴露)
V2 正交化:     F 因子对 Top40 量价因子逐日残差化 (剥离与量价的拥挤重叠)
判定: 2026 窗口能否从 -0.0172 修复, 同时保住全期 ICIR 8.0 的大头
"""
import io
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import settings
from data.fundamentals import FundamentalsProvider, FUNDAMENTAL_FACTOR_NAMES
from research_v2.labels.label_registry import LabelRegistry
from models.walk_forward import WalkForwardTrainer

LABEL_COL = 'label_up_down_20d_raw40'
HORIZON = 40
WINDOWS = {'full_oos': ('2021-09-01', '2026-09-07'), 'y2026': ('2026-01-01', '2026-09-07')}
POOL_FILE = ROOT / 'reports' / 'model_research' / 'candidate_pool_top40.txt'
LOG = ROOT / 'artifacts' / 'fundamentals_repair_experiment.log'


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


def neutralize_by_industry(df, facs):
    d = df.copy()
    for f in facs:
        d[f] = d[f] - d.groupby(['date', 'industry'])[f].transform('mean')
    return d


def orthogonalize(df, facs, pool):
    """逐日: F 因子对 Top40 量价因子回归取残差 (剥离拥挤重叠)"""
    d = df.copy()
    regs = [f for f in pool if f in d.columns and f not in facs]
    n_done = 0
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
        n_done += 1
    log(f"  正交化完成: {n_done} 个交易日")
    return d


def window_metrics(oos_df, start, end, tag):
    sub = oos_df[(oos_df['date'] >= start) & (oos_df['date'] <= end)]
    ics = {}
    for dt, grp in sub.groupby('date'):
        g = grp[['pred_score', LABEL_COL]].dropna()
        if len(g) < 8:
            continue
        r = stats.spearmanr(g['pred_score'], g[LABEL_COL])[0]
        if np.isfinite(r):
            ics[dt] = float(r)
    if not ics:
        return {'window': tag, 'n_days': 0}
    s = pd.Series(ics).sort_index()
    m, sd = float(s.mean()), float(s.std())
    return {'window': tag, 'n_days': int(len(s)), 'mean_rank_ic': round(m, 5),
            'icir_annualized': round(m / (sd + 1e-9) * np.sqrt(6), 3),  # 40日重叠→年化折减
            'positive_ratio': round(float((s > 0).mean()), 3)}


def run_variant(df, facs, pool, tag):
    log(f"[{tag}] 训练开始 (池 {len(pool)})")
    t0 = time.time()
    trainer = WalkForwardTrainer(
        model_type='lightgbm_ranker', task_type='classification', label_col=LABEL_COL,
        feature_selection_method='rank_ic_pruned', top_k_features=10,
        strict_mode=False, save_model=False)
    oos_df, _ = trainer.run_walk_forward(df, feature_cols=pool)
    log(f"[{tag}] 完成 30 折, {time.time()-t0:.0f}s")
    return [window_metrics(oos_df, s, e, t) for t, (s, e) in WINDOWS.items()]


def main() -> int:
    log(">>> F 因子 2026 修复实验开始 (40日标签)")
    df, facs, pool = load_base()
    log(f"基本面因子 {len(facs)} 个 | 标签 horizon={HORIZON}")

    out = {'generated_at': datetime.now().isoformat(timespec='seconds'),
           'baseline_combo_same_data': {'full': 0.05684, 'icir': 8.0036, 'y2026': -0.01724},
           'variants': {}}

    d_v1 = neutralize_by_industry(df, facs)
    r1 = run_variant(d_v1, facs, pool + facs, 'V1行业中性化')
    out['variants']['v1_industry_neutral'] = r1
    log(f"[V1结果] " + json.dumps(r1, ensure_ascii=False))

    d_v2 = orthogonalize(df, facs, pool)
    r2 = run_variant(d_v2, facs, pool + facs, 'V2正交化')
    out['variants']['v2_orthogonalized'] = r2
    log(f"[V2结果] " + json.dumps(r2, ensure_ascii=False))

    out_path = ROOT / 'reports' / 'model_research' / f'fundamentals_repair_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f">>> 完成, 落盘: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
