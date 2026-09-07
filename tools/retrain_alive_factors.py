"""存活因子池重训: 用 2025-2026 regime 中仍然有效的因子重训 ranker, 专测近期 OOS

来源: factor_regime_screen 筛出的 2025-2026 存活 alpha (IC>0.01, ICIR>1.5, 正比例>53%)
对照: 全因子池 ranker 全期 IC +0.0282 但 2026 转负 (-0.0142)
"""
import io
import json
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import settings
from research_v2.labels.label_registry import LabelRegistry
from models.walk_forward import WalkForwardTrainer

LABEL_COL = 'label_up_down_20d'
MIN_CROSS_SECTION = 8

ALIVE_FACTORS = [
    'turnover',
    'MA_RATIO_60',
    'KAUFMAN_EFFICIENCY_20',
    'ALPHA_LIQUIDITY_X_VOL',
    'STD250',
    'IS_LIMIT_UP_LAG1',
    'STD120',
    'EFFICIENCY_RATIO_10',
    'ATR_RATIO_60',
    'STD60',
]

OUT_DIR = PROJECT_ROOT / 'reports' / 'model_research' / f'retrain_alive_{datetime.now():%Y%m%d_%H%M%S}'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def window_metrics(oos_df: pd.DataFrame, start: str, end: str, tag: str) -> dict:
    sub = oos_df[(oos_df['date'] >= start) & (oos_df['date'] <= end)]
    ics = {}
    for dt, grp in sub.groupby('date'):
        g = grp[['pred_score', LABEL_COL]].dropna()
        if len(g) < MIN_CROSS_SECTION:
            continue
        r = stats.spearmanr(g['pred_score'], g[LABEL_COL])[0]
        if not np.isnan(r):
            ics[dt] = float(r)
    if not ics:
        return {'window': tag, 'n_days': 0}
    s = pd.Series(ics)
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


def main() -> int:
    print('>> 加载 v3 因子矩阵...')
    df = pd.read_parquet(PROJECT_ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet')
    df['date'] = pd.to_datetime(df['date'])
    raw = LabelRegistry.compute_label_v2(df, horizon=settings.LABEL_HORIZON)
    df[LABEL_COL] = (raw > 0).astype(float).mask(raw.isna(), np.nan)

    missing = [f for f in ALIVE_FACTORS if f not in df.columns]
    if missing:
        print(f'FATAL 缺失因子: {missing}')
        return 2
    print(f'存活因子池 ({len(ALIVE_FACTORS)}): {ALIVE_FACTORS}')

    trainer = WalkForwardTrainer(
        model_type='lightgbm_ranker',
        task_type='classification',
        label_col=LABEL_COL,
        strict_mode=True,
        save_model=False,
    )
    print('>> 走步训练 (lightgbm_ranker on alive factors)...')
    oos_df, _ = trainer.run_walk_forward(df, feature_cols=ALIVE_FACTORS)
    oos_df.to_parquet(OUT_DIR / 'oos_predictions_alive_ranker.parquet', index=False)
    print(f'OOS 记录: {len(oos_df):,} 行 | {len(trainer.models)} 折')

    windows = {
        'full_oos': ('2021-09-01', '2026-09-04'),
        'recent_2025_2026': ('2025-01-01', '2026-09-04'),
        'y2026': ('2026-01-01', '2026-09-04'),
        'q3_2026': ('2026-07-01', '2026-09-04'),
    }
    results = [window_metrics(oos_df, s, e, t) for t, (s, e) in windows.items()]
    summary = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'config': 'lightgbm_ranker on 2025-2026 alive factors',
        'factors': ALIVE_FACTORS,
        'n_folds': len(trainer.models),
        'n_oos_rows': int(len(oos_df)),
        'purge_gap_days': int(trainer.purge_gap_days),
        'windows': results,
    }
    (OUT_DIR / 'retrain_alive_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('\n===== 存活因子池 ranker 诚实评估 =====')
    for r in results:
        if r.get('n_days'):
            print(f"{r['window']}: mean IC={r['mean_rank_ic']:+.5f} | ICIR={r['icir_annualized']:+.3f} "
                  f"| 正比例={r['positive_ratio']:.1%} | 分年={r.get('per_year')}")
        else:
            print(f"{r['window']}: 无有效样本")
    print(f'\n产物目录: {OUT_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
