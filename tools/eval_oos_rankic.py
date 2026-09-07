"""独立 OOS 评估器: 对已落盘的 oos_predictions parquet 计算诚实指标 (不重训)"""
import io
import json
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import stats

LABEL_COL = 'label_up_down_20d'
MIN_CROSS_SECTION = 8


def evaluate(parquet_path: str, config_name: str, out_dir: Path) -> dict:
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date'])

    ic = {}
    spreads = []
    for dt, grp in df.groupby('date'):
        if len(grp) < MIN_CROSS_SECTION:
            continue
        r = stats.spearmanr(grp['pred_score'], grp[LABEL_COL])[0]
        if not np.isnan(r):
            ic[dt] = float(r)
        if len(grp) >= MIN_CROSS_SECTION * 3:
            q_hi = grp['pred_score'].quantile(0.8)
            q_lo = grp['pred_score'].quantile(0.2)
            top = grp.loc[grp['pred_score'] >= q_hi, LABEL_COL].mean()
            bot = grp.loc[grp['pred_score'] <= q_lo, LABEL_COL].mean()
            if np.isfinite(top) and np.isfinite(bot):
                spreads.append(float(top - bot))

    ic = pd.Series(ic, name='rank_ic').dropna().sort_index()
    mean_ic = float(ic.mean())
    std_ic = float(ic.std())
    per_year = {str(y): round(float(v), 4) for y, v in ic.groupby(ic.index.year).mean().items()}
    result = {
        'config': config_name,
        'source': str(parquet_path),
        'n_oos_rows': int(len(df)),
        'n_ic_days': int(len(ic)),
        'ic_first_date': str(ic.index.min())[:10],
        'ic_last_date': str(ic.index.max())[:10],
        'mean_rank_ic': round(mean_ic, 5),
        'ic_std': round(std_ic, 5),
        'icir_annualized': round(mean_ic / (std_ic + 1e-9) * np.sqrt(242), 4),
        'ic_positive_win_rate': round(float((ic > 0).mean()), 4),
        'per_year_mean_ic': per_year,
        'long_short_daily_spread_20pct': round(float(np.mean(spreads)), 5) if spreads else None,
        'evaluated_at': datetime.now().isoformat(timespec='seconds'),
    }
    ic.rename('rank_ic').to_frame().assign(date=lambda x: x.index).to_csv(
        out_dir / f'daily_rank_ic_{config_name}.csv', index=False
    )
    return result


def main():
    parquet = sys.argv[1]
    config_name = sys.argv[2]
    out_dir = Path(parquet).parent
    r = evaluate(parquet, config_name, out_dir)
    out_json = out_dir / f'eval_{config_name}.json'
    out_json.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
