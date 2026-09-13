"""信号质量深诊 (借鉴 Alphalens / Qlib 的评判方法) — 对认证模型的 OOS 预测做:

1. 分位数组合分析: 每日按 pred_score 分 5 组, 看各组 20 日前瞻收益是否单调递增
   (Q1<Q2<Q3<Q4<Q5 = 信号真有梯度; Q5-Q1 多空价差 = 信号的可交易强度)
2. IC 衰减曲线: 预测在 1/5/10/20/40 日前瞻收益上的 Rank IC (信号保质期)
3. 头部组合模拟: 每日 Top 20% 等权组合的 20 日收益 vs 全市场均值 (超额)
"""
import io
import sys
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

SRC = ROOT / 'reports' / 'model_research' / 'rolling_sel_20260907_163910' / 'oos_predictions_rolling_sel.parquet'
HORIZONS = [1, 5, 10, 20, 40]
N_QUANTILE = 5


def main() -> int:
    print('>>> 加载 OOS 预测 (352K 行)...', flush=True)
    df = pd.read_parquet(SRC, columns=['date', 'symbol', 'adj_close', 'pred_score'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    print(f'    {len(df):,} 行 | {df["date"].min().date()} ~ {df["date"].max().date()}', flush=True)

    # 前瞻收益 (T+1 收盘进入, T+1+N 收盘退出 — 与标签撮合口径一致)
    g = df.groupby('symbol', sort=False)['adj_close']
    for n in HORIZONS:
        df[f'fwd_{n}'] = (g.shift(-(1 + n)) / g.shift(-1) - 1.0) * 100
    # 头部组合模拟需要逐日全截面: 去掉无前瞻收益的尾部行
    eval_df = df.dropna(subset=['pred_score'] + [f'fwd_{n}' for n in HORIZONS])
    print(f'    有效评估行: {len(eval_df):,}', flush=True)

    # ---- 1. 分位数组合分析 (20 日前瞻) ----
    print('\n===== [1] 分位数组合分析 (20 日前瞻收益, 每日5分组) =====', flush=True)
    eval_df['q'] = eval_df.groupby('date')['pred_score'].transform(
        lambda s: pd.qcut(s.rank(method='first'), N_QUANTILE, labels=False) + 1)
    q_stats = eval_df.groupby('q')['fwd_20'].agg(['mean', 'std', 'count'])
    q_daily_mean = eval_df.groupby(['date', 'q'])['fwd_20'].mean().unstack()
    spread_daily = q_daily_mean[N_QUANTILE] - q_daily_mean[1]
    spread_icir = spread_daily.mean() / (spread_daily.std() + 1e-9) * np.sqrt(12)  # 20日重叠→月频近似
    monotonic = all(q_stats['mean'].iloc[i] < q_stats['mean'].iloc[i + 1] for i in range(N_QUANTILE - 1))
    q_stats['mean'] = q_stats['mean'].round(3)
    print(q_stats.round(3).to_string(), flush=True)
    print(f'    单调递增 (Q1→Q5): {"✓ 是" if monotonic else "✗ 否"}', flush=True)
    print(f'    Q5-Q1 多空价差: 均值 {spread_daily.mean():+.3f}%/20日 | t={spread_daily.mean()/(spread_daily.std()+1e-9):.2f} | '
          f'年化 Sharpe 近似 {spread_icir:+.2f}', flush=True)

    # 分年价差
    yr = spread_daily.groupby(spread_daily.index.year).mean().round(3)
    print(f'    分年 Q5-Q1 价差: {yr.to_dict()}', flush=True)

    # ---- 2. IC 衰减曲线 ----
    print('\n===== [2] IC 衰减曲线 (Rank IC @ 各前瞻期) =====', flush=True)
    decay = {}
    for n in HORIZONS:
        ics = []
        sub = eval_df[['date', 'pred_score', f'fwd_{n}']].dropna()
        for dt, grp in sub.groupby('date'):
            if len(grp) < 30:
                continue
            r = stats.spearmanr(grp['pred_score'], grp[f'fwd_{n}'])[0]
            if np.isfinite(r):
                ics.append(r)
        s = pd.Series(ics)
        decay[n] = {'mean_ic': round(s.mean(), 5), 'pos_ratio': round((s > 0).mean(), 3)}
        print(f'    @{n:>2}日: mean IC {s.mean():+.5f} | 正日占比 {(s > 0).mean():.3f}', flush=True)

    # ---- 3. 头部组合超额 (Top 20% vs 全市场均值, 20 日) ----
    top_daily = eval_df[eval_df['q'] == N_QUANTILE].groupby('date')['fwd_20'].mean()
    mkt_daily = eval_df.groupby('date')['fwd_20'].mean()
    excess = top_daily - mkt_daily
    yr_ex = excess.groupby(excess.index.year).mean().round(3)
    print('\n===== [3] 头部组合 (Top 20% 等权) 超额 =====', flush=True)
    print(f'    日均超额: {excess.mean():+.3f}%/20日 | 分年: {yr_ex.to_dict()}', flush=True)

    # 落盘
    out = ROOT / 'reports' / 'model_research' / f'signal_quality_analysis_{pd.Timestamp.now():%Y%m%d_%H%M%S}.json'
    import json
    out.write_text(json.dumps({
        'quantile_mean_fwd20': {int(k): round(float(v), 4) for k, v in q_stats['mean'].items()},
        'monotonic_q1_to_q5': bool(monotonic),
        'spread_mean_pct': round(float(spread_daily.mean()), 4),
        'spread_t_stat': round(float(spread_daily.mean() / (spread_daily.std() + 1e-9)), 3),
        'spread_by_year': {str(k): float(v) for k, v in yr.items()},
        'ic_decay': decay,
        'top20_excess_mean_pct': round(float(excess.mean()), 4),
        'top20_excess_by_year': {str(k): float(v) for k, v in yr_ex.items()},
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n>>> 已落盘: {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
