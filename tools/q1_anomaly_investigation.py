"""Q1 反常组特征调查: 为什么模型最低分组的 20 日收益不差?

假设检验: Q1 组是否集中了低波动/大市值/高股息属性 (2024-2026 红利风格受益者)?
输出: Q1 vs Q5 的特征画像对比 + 分年表现
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

SRC = ROOT / 'reports' / 'model_research' / 'rolling_sel_20260907_163910' / 'oos_predictions_rolling_sel.parquet'


def main() -> int:
    df = pd.read_parquet(SRC, columns=['date', 'symbol', 'name', 'industry', 'adj_close',
                                       'circ_mv', 'STD20', 'turnover', 'pred_score'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    g = df.groupby('symbol', sort=False)['adj_close']
    df['fwd_20'] = (g.shift(-21) / g.shift(-1) - 1.0) * 100
    ev = df.dropna(subset=['pred_score', 'fwd_20']).copy()
    ev['q'] = ev.groupby('date')['pred_score'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1)

    print('===== Q1(最低分) vs Q5(最高分) 特征画像 =====', flush=True)
    prof_cols = {'circ_mv': '流通市值(亿)', 'STD20': '20日波动率', 'turnover': '换手率', 'fwd_20': '20日收益%'}
    rows = {}
    for q in [1, 5]:
        sub = ev[ev['q'] == q]
        rows[f'Q{q}'] = {
            '流通市值(亿)': round(sub['circ_mv'].mean() / 1e8, 0),
            '20日波动率': round(sub['STD20'].mean(), 4),
            '换手率': round(sub['turnover'].mean(), 4),
            '20日收益%': round(sub['fwd_20'].mean(), 3),
            '样本数': len(sub),
        }
    print(pd.DataFrame(rows).to_string(), flush=True)

    # 行业分布: Q1 的 Top 行业占比 vs Q5
    print('\n===== Q1 行业 Top8 (占比) vs Q5 =====', flush=True)
    q1_ind = ev[ev['q'] == 1]['industry'].value_counts(normalize=True).head(8)
    q5_ind = ev[ev['q'] == 5]['industry'].value_counts(normalize=True)
    for ind, pct in q1_ind.items():
        print(f'  {ind}: Q1 {pct*100:.1f}% | Q5 {q5_ind.get(ind, 0)*100:.1f}%', flush=True)

    # 分年: Q1 表现是否集中于近两年 (反常是否 = 风格切换产物)
    print('\n===== 分年 20日收益: Q1 vs Q5 =====', flush=True)
    ev['year'] = ev['date'].dt.year
    piv = ev.pivot_table(index='year', columns='q', values='fwd_20', aggfunc='mean').round(3)
    print(piv[[1, 3, 5]].to_string(), flush=True)

    # 关键洞察: Q1 里表现最好的股票特征 (2024+)
    print('\n===== 2024+ Q1 组内分化: 表现最好 20% 的特征 =====', flush=True)
    q1_recent = ev[(ev['q'] == 1) & (ev['year'] >= 2024)]
    top20_cut = q1_recent['fwd_20'].quantile(0.8)
    winners = q1_recent[q1_recent['fwd_20'] >= top20_cut]
    losers = q1_recent[q1_recent['fwd_20'] < q1_recent['fwd_20'].quantile(0.2)]
    for label, sub in [('赢家(前20%)', winners), ('输家(后20%)', losers)]:
        print(f"  {label}: 流通市值 {sub['circ_mv'].mean()/1e8:.0f}亿 | 波动率 {sub['STD20'].mean():.4f} | "
              f"换手 {sub['turnover'].mean():.4f} | 均收益 {sub['fwd_20'].mean():.2f}%", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
