"""基本面因子走步实验 (Phase: 真实财报数据 PIT 注入)

数据: 东财 datacenter 业绩报表 (含公告日期 PIT 锚) -> FundamentalsProvider
     的 build_daily_fundamental_matrix 按 effective_date 展开成日频因子。
因子: F_ROE / F_GROSS_MARGIN / F_REV_GROWTH / F_PROFIT_GROWTH / F_EPS /
     F_BPS / F_OCF_PS (8 个真实基本面因子)
对比基线: Top40 单独 (今晨: full +0.0272 | 2026 -0.0020)
"""
import io
import json
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

LABEL_COL = 'label_up_down_20d'
WINDOWS = {
    'full_oos': ('2021-09-01', '2026-09-07'),
    'y2026': ('2026-01-01', '2026-09-07'),
}
POOL_FILE = ROOT / 'reports' / 'model_research' / 'candidate_pool_top40.txt'


def window_metrics(oos_df: pd.DataFrame, start, end, tag) -> dict:
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
    mean_ic, std_ic = float(s.mean()), float(s.std())
    return {'window': tag, 'n_days': int(len(s)), 'mean_rank_ic': round(mean_ic, 5),
            'icir_annualized': round(mean_ic / (std_ic + 1e-9) * np.sqrt(242), 4),
            'positive_ratio': round(float((s > 0).mean()), 4)}


def main() -> int:
    log_path = ROOT / 'artifacts' / 'fundamentals_walkforward.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import logging
    logging.basicConfig(level=logging.WARNING)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log(">>> 基本面因子走步实验开始")
    df = pd.read_parquet(ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet')
    df['date'] = pd.to_datetime(df['date'])

    # 基本面日频因子 (PIT: 公告日/110天延迟 -> effective_date 展开)
    market_df = pd.read_parquet(ROOT / 'data_storage' / 'parquet' / 'market_daily.parquet',
                                columns=['date', 'symbol', 'high', 'low', 'close', 'volume', 'amount', 'pct_change'])
    market_df['date'] = pd.to_datetime(market_df['date'])
    fp = FundamentalsProvider(delay_days=settings.FUNDAMENTAL_DELAY_DAYS)
    log("构建基本面日频矩阵 (PIT 展开)...")
    fund_daily = fp.build_daily_fundamental_matrix(market_df, start_year=settings.FUNDAMENTAL_START_YEAR)
    if fund_daily is None or fund_daily.empty:
        log("[-] 基本面矩阵为空, 实验终止")
        return 1
    fund_daily['date'] = pd.to_datetime(fund_daily['date'])
    facs = [f for f in FUNDAMENTAL_FACTOR_NAMES if f in fund_daily.columns]
    log(f"基本面因子 {len(facs)} 个 | 覆盖 {fund_daily['symbol'].nunique()} 标的 | "
        f"非空行 {int(fund_daily[facs[0]].notna().sum()):,}")

    df = df.merge(fund_daily[['symbol', 'date'] + facs], on=['symbol', 'date'], how='left')
    raw = LabelRegistry.compute_label_v2(df, horizon=20)
    df[LABEL_COL] = (raw > 0).astype(float).mask(raw.isna(), np.nan)

    pool = [x.strip() for x in POOL_FILE.read_text(encoding='utf-8').splitlines() if x.strip()]
    pool = [f for f in pool if f in df.columns]
    pool += [f for f in facs if f in df.columns]
    log(f"候选池: Top40 + 基本面 = {len(pool)}")

    t0 = time.time()
    trainer = WalkForwardTrainer(
        model_type='lightgbm_ranker', task_type='classification', label_col=LABEL_COL,
        feature_selection_method='rank_ic_pruned', top_k_features=10,
        strict_mode=False, save_model=False,
    )
    oos_df, _ = trainer.run_walk_forward(df, feature_cols=pool)
    log(f"完成 {len(trainer.models)} 折, {time.time()-t0:.0f}s")

    results = [window_metrics(oos_df, s, e, t) for t, (s, e) in WINDOWS.items()]
    out = {'generated_at': datetime.now().isoformat(timespec='seconds'),
           'variant': 'Top40 + PIT fundamental factors',
           'baseline_same_data': {'full': 0.02724, 'y2026': -0.00197},
           'results': results}
    out_path = ROOT / 'reports' / 'model_research' / f'fundamentals_walkforward_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f"[结果] " + json.dumps(results, ensure_ascii=False))
    log(f">>> 完成, 落盘: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
