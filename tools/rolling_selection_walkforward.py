"""滚动因子筛选版走步 (消除前视偏差的干净实验)

背景: 上一轮用 2025-2026 窗口筛选因子、又在同一窗口评估 -> 筛选前视偏差 (selection
look-ahead bias), 导致近期 IC 反而更差。本实验改用仓库已认证的 Train-Only 折内筛选:
WalkForwardTrainer(feature_selection_method=...) 在每一折内调用 models/fold_feature_selector
.FoldFeatureSelector, 仅用该折训练期数据计算 RankIC / 相关性剪枝 / 年度稳定性, 杜绝跨折泄漏。

对照基线 (同一 v3 矩阵, 同一标签, 同一 30 折走步):
  - 全因子池 ranker: 全期 +0.0282 | 2025-2026 ~0 | 2026 -0.0142
  - 存活因子池 ranker: 全期 +0.0349 | 2025-2026 -0.0064 | 2026 -0.0181 (有前视偏差)
"""
import io
import json
import sys
import time
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
OUT_DIR = PROJECT_ROOT / 'reports' / 'model_research' / f'rolling_sel_{datetime.now():%Y%m%d_%H%M%S}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = {
    'full_oos': ('2021-09-01', '2026-09-04'),
    'recent_2025_2026': ('2025-01-01', '2026-09-04'),
    'y2026': ('2026-01-01', '2026-09-04'),
    'q3_2026': ('2026-07-01', '2026-09-04'),
}


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


def main() -> int:
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    method = sys.argv[2] if len(sys.argv) > 2 else 'rank_ic_pruned'
    # strict_mode=False 仅放宽 FoldFeatureSelector 的门禁 (回退为按 |RankIC| Top-N),
    # 仍然是 Train-Only 折内筛选; purge gap 25 天照常施加, 无前视偏差。
    strict = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False

    print('>> 加载 v3 因子矩阵...')
    df = pd.read_parquet(PROJECT_ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet')
    df['date'] = pd.to_datetime(df['date'])
    raw = LabelRegistry.compute_label_v2(df, horizon=settings.LABEL_HORIZON)
    df[LABEL_COL] = (raw > 0).astype(float).mask(raw.isna(), np.nan)

    print(f'>> 走步训练: lightgbm_ranker + Train-Only 折内筛选 ({method}, top_k={top_k})')
    t0 = time.time()
    trainer = WalkForwardTrainer(
        model_type='lightgbm_ranker',
        task_type='classification',
        label_col=LABEL_COL,
        feature_selection_method=method,
        top_k_features=top_k,
        strict_mode=strict,
        save_model=False,
    )
    pool_file = PROJECT_ROOT / 'reports' / 'model_research' / 'candidate_pool_top40.txt'
    if pool_file.exists():
        pool = [x.strip() for x in pool_file.read_text(encoding='utf-8').splitlines() if x.strip()]
        pool = [f for f in pool if f in df.columns]
        print(f'>> 候选池: {len(pool)} 个因子 (Top40 by |full-period IC|, 仅限制池, 折内仍 Train-Only 选择)')
    else:
        pool = None
    oos_df, _ = trainer.run_walk_forward(df, feature_cols=pool)
    elapsed = time.time() - t0
    oos_df.to_parquet(OUT_DIR / 'oos_predictions_rolling_sel.parquet', index=False)
    print(f'>> {len(trainer.models)} 折完成, 耗时 {elapsed:.0f}s, OOS {len(oos_df):,} 行')

    results = [window_metrics(oos_df, s, e, t) for t, (s, e) in WINDOWS.items()]
    summary = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'config': f'lightgbm_ranker + Train-Only fold selection ({method}, top_k={top_k}, strict={strict})',
        'dataset': 'v3 factor_matrix_300.parquet (sha=bb1593ed)',
        'label': LABEL_COL,
        'n_folds': len(trainer.models),
        'n_oos_rows': int(len(oos_df)),
        'purge_gap_days': int(trainer.purge_gap_days),
        'elapsed_seconds': round(elapsed, 1),
        'baseline_comparison': {
            'full_factor_pool_ranker': {'full': 0.0282, 'y2026': -0.0142},
            'alive_factor_pool_ranker': {'full': 0.0349, 'y2026': -0.0181},
        },
        'windows': results,
    }
    (OUT_DIR / 'rolling_selection_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('\n===== 滚动筛选版诚实评估 =====')
    for r in results:
        if r.get('n_days'):
            print(f"{r['window']}: mean IC={r['mean_rank_ic']:+.5f} | ICIR={r['icir_annualized']:+.3f} "
                  f"| 正比例={r['positive_ratio']:.1%} | 分年={r.get('per_year')}")
        else:
            print(f"{r['window']}: 无有效样本")
    print(f'\n产物: {OUT_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
