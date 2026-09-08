"""真实资金流因子走步实验 (Phase B): Top40 池 + 真实东财资金流因子

数据: data_storage/fundflow/fundflow_20260908.parquet (186 标的 x ~120 交易日, 09-07 截止)
因子 (全部来自真实主力资金数据):
  FF_MAIN_RATIO_MA5/MA20   主力净占比 5/20 日均值 (持续吸筹/派发)
  FF_SUPER_RATIO_MA5       超大单净占比 5 日均值 (大机构动向)
  FF_CONTIN_DAYS           连续主力净流入天数 (带符号的连续性)
  FF_MAIN_SUM5_RATIO       近 5 日主力净流入合计 / 近 20 日日均成交额 (强度归一)
基线 (今晨同数据同协议): full +0.0272 | 2025 +0.0105 | 2026 -0.0020
注意: 资金流历史仅 ~120 交易日 (2026-03 起), 早期折无此数据属预期, 只影响近期折。
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

from research_v2.labels.label_registry import LabelRegistry
from models.walk_forward import WalkForwardTrainer

LABEL_COL = 'label_up_down_20d'
WINDOWS = {
    'full_oos': ('2021-09-01', '2026-09-07'),
    'y2026': ('2026-01-01', '2026-09-07'),
    'since_2026_03 (资金流可用)': ('2026-03-01', '2026-09-07'),
}
FF_FACTORS = ['FF_MAIN_RATIO_MA5', 'FF_MAIN_RATIO_MA20', 'FF_SUPER_RATIO_MA5',
              'FF_CONTIN_DAYS', 'FF_MAIN_SUM5_RATIO']
POOL_FILE = ROOT / 'reports' / 'model_research' / 'candidate_pool_top40.txt'


def build_ff_factors(ff: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    ff = ff.sort_values(['symbol', 'date']).reset_index(drop=True)
    g = ff.groupby('symbol', sort=False)
    ratio = ff['main_net_ratio']
    ff['FF_MAIN_RATIO_MA5'] = g['main_net_ratio'].rolling(5).mean().reset_index(level=0, drop=True)
    ff['FF_MAIN_RATIO_MA20'] = g['main_net_ratio'].rolling(20).mean().reset_index(level=0, drop=True)
    ff['FF_SUPER_RATIO_MA5'] = g['super_net_ratio'].rolling(5).mean().reset_index(level=0, drop=True)

    # 连续净流入天数 (带符号: 连续为正计数+, 连续为负计数-)
    sign = np.sign(ff['main_net_ratio'])
    streak = sign.groupby((sign != sign.shift()).cumsum()).cumsum() * sign
    ff['FF_CONTIN_DAYS'] = streak

    # 近 5 日主力净流入合计 / 近 20 日日均成交额 (强度归一, 需要 market amount)
    amt = mkt[['symbol', 'date', 'amount']].copy()
    amt['AMT_MA20'] = amt.groupby('symbol')['amount'].rolling(20).mean().reset_index(level=0, drop=True)
    sum5 = g['main_net_inflow'].rolling(5).sum().reset_index(level=0, drop=True)
    ff = ff.merge(amt[['symbol', 'date', 'AMT_MA20']], on=['symbol', 'date'], how='left')
    ff['FF_MAIN_SUM5_RATIO'] = sum5 / ff['AMT_MA20'].replace(0, np.nan)
    return ff[['symbol', 'date'] + FF_FACTORS]


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
    log_path = ROOT / 'artifacts' / 'fundflow_walkforward.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import logging
    logging.basicConfig(level=logging.WARNING)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log(">>> 真实资金流因子走步实验开始")
    df = pd.read_parquet(ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet')
    df['date'] = pd.to_datetime(df['date'])
    ff = pd.read_parquet(ROOT / 'data_storage' / 'fundflow' / 'fundflow_20260908.parquet')
    ff['date'] = pd.to_datetime(ff['date'])
    log(f"资金流底座: {ff['symbol'].nunique()} 标的 | {ff['date'].min().date()} ~ {ff['date'].max().date()}")

    # 市场成交额用于强度归一
    mkt = pd.read_parquet(ROOT / 'data_storage' / 'parquet' / 'market_daily.parquet',
                          columns=['date', 'symbol', 'amount'])
    mkt['date'] = pd.to_datetime(mkt['date'])

    ffd = build_ff_factors(ff, mkt)
    df = df.merge(ffd, on=['symbol', 'date'], how='left')

    raw = LabelRegistry.compute_label_v2(df, horizon=20)
    df[LABEL_COL] = (raw > 0).astype(float).mask(raw.isna(), np.nan)

    pool = [x.strip() for x in POOL_FILE.read_text(encoding='utf-8').splitlines() if x.strip()]
    pool = [f for f in pool if f in df.columns]
    pool += [f for f in FF_FACTORS if f in df.columns]
    log(f"候选池: Top40 + 真实资金流 = {len(pool)} | 资金流因子覆盖: "
        f"{int(df['FF_MAIN_RATIO_MA5'].notna().sum()):,} 行")

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
           'variant': 'Top40 + REAL fundflow factors (186 syms, 120d history)',
           'baseline_same_data': {'full': 0.02724, 'y2026': -0.00197},
           'results': results}
    out_path = ROOT / 'reports' / 'model_research' / f'fundflow_walkforward_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f"[结果] " + json.dumps(results, ensure_ascii=False))
    log(f">>> 完成, 落盘: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
