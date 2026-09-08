"""资金流候选因子进池决胜实验: Top40 + 4 个资金流候选 vs 上午 v2 基线

唯一变量 = 候选池扩充 (VOL_RATIO20 / VWAP_DEV20 / NEG_OBV_SLOPE20 / MFI14)。
基线 (今晨同数据): full +0.0272 | 2025 +0.0105 | 2026 -0.0020
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
from research_v2.labels.label_registry import LabelRegistry
from models.walk_forward import WalkForwardTrainer

LABEL_COL = 'label_up_down_20d'
WINDOWS = {
    'full_oos': ('2021-09-01', '2026-09-07'),
    'recent_2025_2026': ('2025-01-01', '2026-09-07'),
    'y2026': ('2026-01-01', '2026-09-07'),
}
NEW_FACTORS = ['VOL_RATIO20', 'VWAP_DEV20', 'NEG_OBV_SLOPE20', 'MFI14']


def add_moneyflow(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    pos_flow = ((tp > tp.shift(1)) * tp * df['volume']).where(lambda x: x > 0, 0.0)
    neg_flow = ((tp < tp.shift(1)) * tp * df['volume']).where(lambda x: x > 0, 0.0)
    pos14 = pos_flow.groupby(df['symbol'], sort=False).rolling(14).sum().reset_index(level=0, drop=True)
    neg14 = neg_flow.groupby(df['symbol'], sort=False).rolling(14).sum().reset_index(level=0, drop=True)
    mfr = pos14 / neg14.replace(0, np.nan)
    df['MFI14'] = (100 - 100 / (1 + mfr)).clip(0, 100)
    obv = (np.sign(df['close'].diff()) * df['volume']).groupby(df['symbol'], sort=False).cumsum()
    df['NEG_OBV_SLOPE20'] = -(obv.groupby(df['symbol'], sort=False).rolling(20).apply(
        lambda s: np.polyfit(np.arange(len(s)), s, 1)[0], raw=True
    ).reset_index(level=0, drop=True) / df['close'])
    vwap20 = (df['amount'].groupby(df['symbol'], sort=False).rolling(20).sum().reset_index(level=0, drop=True)
              / df['volume'].groupby(df['symbol'], sort=False).rolling(20).sum().reset_index(level=0, drop=True))
    df['VWAP_DEV20'] = df['close'] / vwap20 - 1.0
    df['VOL_RATIO20'] = df['volume'] / df['volume'].groupby(df['symbol'], sort=False).rolling(20).mean().reset_index(level=0, drop=True)
    return df


def window_metrics(oos_df: pd.DataFrame, start: str, end: str, tag: str) -> dict:
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
            'positive_ratio': round(float((s > 0).mean()), 4),
            'per_year': {str(y): round(float(v), 5) for y, v in s.groupby(s.index.year).mean().items()}}


def main() -> int:
    log_path = ROOT / 'artifacts' / 'moneyflow_walkforward.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import logging
    logging.basicConfig(level=logging.WARNING)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log(">>> 资金流因子进池决胜实验开始")
    df = pd.read_parquet(ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = add_moneyflow(df)
    raw = LabelRegistry.compute_label_v2(df, horizon=20)
    df[LABEL_COL] = (raw > 0).astype(float).mask(raw.isna(), np.nan)

    pool = [x.strip() for x in (ROOT / 'reports' / 'model_research' / 'candidate_pool_top40.txt')
            .read_text(encoding='utf-8').splitlines() if x.strip()]
    pool = [f for f in pool if f in df.columns]
    pool += [f for f in NEW_FACTORS if f in df.columns]
    log(f"候选池: Top40 + 资金流候选 = {len(pool)}")

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
           'variant': 'Top40_pool + moneyflow_candidates',
           'baseline_same_data': {'full': 0.02724, 'y2025': 0.01047, 'y2026': -0.00197},
           'results': results}
    out_path = ROOT / 'reports' / 'model_research' / f'moneyflow_walkforward_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f"[结果] " + json.dumps(results, ensure_ascii=False))
    log(f">>> 完成, 落盘: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
