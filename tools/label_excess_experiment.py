"""标签口径对照实验: v2 绝对涨跌 vs v3 基准超额收益 (滚动折内筛选, 消除前视)

假设: v2 标签混入大盘 beta, 2025-2026 大盘震荡导致信号衰减;
     v3 超额标签让模型专注"跑赢大盘"的结构性因子, IC 稳定性应提升。
协议: 与 rolling_selection_walkforward 完全一致 (lightgbm_ranker + Train-Only
     折内筛选 top_k=10, Top40 池, purge 25, 30 折), 唯一变量 = 标签口径。
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

LABEL_COL = 'label_exp'
WINDOWS = {
    'full_oos': ('2021-09-01', '2026-09-07'),
    'recent_2025_2026': ('2025-01-01', '2026-09-07'),
    'y2026': ('2026-01-01', '2026-09-07'),
}
POOL_FILE = ROOT / 'reports' / 'model_research' / 'candidate_pool_top40.txt'


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
    mean_ic = float(s.mean())
    std_ic = float(s.std())
    return {
        'window': tag,
        'n_days': int(len(s)),
        'mean_rank_ic': round(mean_ic, 5),
        'icir_annualized': round(mean_ic / (std_ic + 1e-9) * np.sqrt(242), 4),
        'positive_ratio': round(float((s > 0).mean()), 4),
        'per_year': {str(y): round(float(v), 5) for y, v in s.groupby(s.index.year).mean().items()},
    }


def run_variant(df: pd.DataFrame, raw_label: pd.Series, tag: str, log) -> dict:
    d = df.copy()
    d[LABEL_COL] = (raw_label > 0).astype(float).mask(raw_label.isna(), np.nan)
    pool = [x.strip() for x in POOL_FILE.read_text(encoding='utf-8').splitlines() if x.strip()]
    pool = [f for f in pool if f in d.columns]
    log(f"[{tag}] 走步训练开始 (池 {len(pool)}, 标签非空 {int(d[LABEL_COL].notna().sum()):,})")
    t0 = time.time()
    trainer = WalkForwardTrainer(
        model_type='lightgbm_ranker', task_type='classification', label_col=LABEL_COL,
        feature_selection_method='rank_ic_pruned', top_k_features=10,
        strict_mode=False, save_model=False,
    )
    oos_df, _ = trainer.run_walk_forward(d, feature_cols=pool)
    log(f"[{tag}] 完成 {len(trainer.models)} 折, {time.time()-t0:.0f}s")
    return {'results': [window_metrics(oos_df, s, e, t) for t, (s, e) in WINDOWS.items()],
            'n_oos': int(len(oos_df))}


def main() -> int:
    log_path = ROOT / 'artifacts' / 'label_excess_experiment.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    import logging
    logging.basicConfig(level=logging.WARNING)

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log(">>> 标签口径对照实验开始")
    df = pd.read_parquet(ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet')
    df['date'] = pd.to_datetime(df['date'])

    v2 = LabelRegistry.compute_label_v2(df, horizon=20)
    v3 = LabelRegistry.compute_label_v3(df, horizon=20)

    out = {'generated_at': datetime.now().isoformat(timespec='seconds'),
           'protocol': 'lightgbm_ranker + Train-Only rolling selection, top_k=10, Top40 pool, purge 25, 30 folds',
           'variants': {}}

    r2 = run_variant(df, v2, 'v2绝对涨跌(基线)', log)
    out['variants']['v2_absolute'] = r2
    log(f"[v2结果] " + json.dumps(r2['results'], ensure_ascii=False))

    r3 = run_variant(df, v3, 'v3基准超额', log)
    out['variants']['v3_excess'] = r3
    log(f"[v3结果] " + json.dumps(r3['results'], ensure_ascii=False))

    out_path = ROOT / 'reports' / 'model_research' / f'label_excess_experiment_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f">>> 实验完成, 结果落盘: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
