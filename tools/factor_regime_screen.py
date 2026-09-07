"""子窗口因子筛查: 定位 2025+ regime 中仍然有效的因子 (v3 重训后续方向 ①)

背景: 全历史走步显示模型 2025 起 IC 衰减、2026 转负。全期统计的因子筛选会淹没
"只在近期 regime 有效"的因子。本脚本按时间子窗口逐因子计算截面 RankIC, 找出
在 2025-2026 仍稳定为正的因子, 为下一轮重训提供候选因子池。

输出:
  - 各窗口逐因子 mean IC / IC 标准差 / ICIR / 正比例
  - 全期正 vs 近期正的对比表 (区分"过期 alpha"与"活 alpha")
  - 近期窗口仍显著为正的因子清单 -> factor_regime_screen_<ts>.json
诚实口径: 不做任何美化, 正负如实记录; 本脚本仅做筛查, 不晋升模型。
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

LABEL_COL = 'label_up_down_20d'
MIN_CROSS_SECTION = 8

WINDOWS = {
    'full_2021_2026': ('2021-09-01', '2026-09-04'),
    'recent_2025_2026': ('2025-01-01', '2026-09-04'),
    'h1_2026': ('2026-01-01', '2026-06-30'),
    'q3_2026': ('2026-07-01', '2026-09-04'),
}

# 非因子列 (结构与标签), 不参与筛选
NON_FEATURE = {
    'date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount',
    'outstanding_share', 'pct_change', 'adj_open', 'adj_high', 'adj_low',
    'adj_close', 'adj_pct_change', 'data_source', 'adjustment_mode', 'name',
    'industry', 'board', 'current_is_st', 'st_status_known',
    'historical_st_rule_applied', 'is_st', 'is_st_unknown',
    'excluded_from_training', 'is_nontradable', 'is_subnew', 'is_suspended',
    'pre_close', 'limit_up_ratio', 'limit_down_ratio', 'limit_up_price',
    'limit_down_price', 'price_limit_rule_id', 'is_limit_up', 'is_limit_down',
    'is_limit_up_locked', 'is_limit_down_locked', 'circ_mv', 'circ_mv_raw',
    'benchmark_open', 'benchmark_close', 'benchmark_pct_change',
    'in_universe', LABEL_COL,
}


def window_ic(df: pd.DataFrame, factor: str, start: str, end: str) -> dict:
    sub = df[(df['date'] >= start) & (df['date'] <= end)]
    ics = {}
    for dt, grp in sub.groupby('date'):
        g = grp[[factor, LABEL_COL]].dropna()
        if len(g) < MIN_CROSS_SECTION:
            continue
        r = stats.spearmanr(g[factor], g[LABEL_COL])[0]
        if not np.isnan(r):
            ics[dt] = float(r)
    if not ics:
        return {'n_days': 0}
    s = pd.Series(ics)
    mean_ic = float(s.mean())
    std_ic = float(s.std())
    return {
        'n_days': int(len(s)),
        'mean_ic': round(mean_ic, 5),
        'ic_std': round(std_ic, 5),
        'icir_annualized': round(mean_ic / (std_ic + 1e-9) * np.sqrt(242), 3),
        'positive_ratio': round(float((s > 0).mean()), 3),
    }


def main() -> int:
    print('>> 加载 v3 因子矩阵...')
    df = pd.read_parquet(PROJECT_ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet')
    df['date'] = pd.to_datetime(df['date'])
    raw_label = LabelRegistry.compute_label_v2(df, horizon=settings.LABEL_HORIZON)
    df[LABEL_COL] = (raw_label > 0).astype(float).mask(raw_label.isna(), np.nan)
    if 'in_universe' in df.columns:
        df = df[df['in_universe']]
    df = df.dropna(subset=[LABEL_COL])
    print(f'样本: {len(df):,} 行 | {df["date"].min():%Y-%m-%d} -> {df["date"].max():%Y-%m-%d}')

    factors = [c for c in df.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]
    print(f'待筛查因子数: {len(factors)}')

    results = {}
    for fname, (s, e) in WINDOWS.items():
        print(f'>> 窗口 {fname} ({s} ~ {e})...')
        results[fname] = {f: window_ic(df, f, s, e) for f in factors}

    rows = []
    for f in factors:
        row = {'factor': f}
        for w in WINDOWS:
            r = results[w][f]
            row[f'{w}_ic'] = r.get('mean_ic')
            row[f'{w}_icir'] = r.get('icir_annualized')
            row[f'{w}_pos'] = r.get('positive_ratio')
        rows.append(row)
    table = pd.DataFrame(rows)

    recent = 'recent_2025_2026'
    alive = table[
        (table[f'{recent}_ic'] > 0.01) &
        (table[f'{recent}_icir'] > 1.5) &
        (table[f'{recent}_pos'] > 0.53)
    ].sort_values(f'{recent}_ic', ascending=False)

    out_dir = PROJECT_ROOT / 'reports' / 'model_research'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    table.sort_values('full_2021_2026_ic', ascending=False).to_csv(
        out_dir / f'factor_regime_screen_table_{ts}.csv', index=False
    )
    summary = {
        'generated_at': ts,
        'dataset': 'v3 factor_matrix_300.parquet (sha=bb1593ed)',
        'label': LABEL_COL,
        'windows': WINDOWS,
        'n_factors_screened': len(factors),
        'alive_in_2025_2026': {
            'criteria': 'mean_ic > 0.01 and icir > 1.5 and positive_ratio > 0.53',
            'factors': [
                {
                    'factor': r['factor'],
                    'recent_ic': r[f'{recent}_ic'],
                    'recent_icir': r[f'{recent}_icir'],
                    'recent_positive_ratio': r[f'{recent}_pos'],
                    'full_period_ic': r['full_2021_2026_ic'],
                }
                for _, r in alive.iterrows()
            ],
        },
        'top10_full_period': table.head(10)[['factor', 'full_2021_2026_ic']].to_dict('records'),
    }
    (out_dir / f'factor_regime_screen_{ts}.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    print('\n===== 2025-2026 仍有效的因子 (存活 alpha) =====')
    if alive.empty:
        print('(无因子通过筛选 —— 说明衰减是体系性的, 需构造新因子/换标签)')
    else:
        print(alive[['factor', f'{recent}_ic', f'{recent}_icir', f'{recent}_pos',
                     'full_2021_2026_ic']].to_string(index=False))
    print('\n===== 全期 Top10 (对照: 过期 alpha 可能列此) =====')
    print(table.sort_values('full_2021_2026_ic', ascending=False)
          .head(10)[['factor', 'full_2021_2026_ic', f'{recent}_ic']].to_string(index=False))
    print(f'\n产物: factor_regime_screen_{ts}.json / factor_regime_screen_table_{ts}.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
