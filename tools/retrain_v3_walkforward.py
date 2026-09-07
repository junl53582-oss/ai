"""v3 数据集全历史 Walk-Forward 重训与诚实评估 (2026-09-07 重训计划 Step 1)

协议:
- 数据: data_storage/research/factor_matrix_300.parquet (v3, sha=bb1593ed..., 465,544 x 152)
- 标签: LabelRegistry.compute_label_v2, horizon=20 (label_up_down_20d), 与现役模型一致
- 走步: WalkForwardTrainer strict_mode=True (PURGE_GAP=25 >= LABEL_HORIZON=20, 禁未来函数)
- 配置 A: bagging_ensemble (现役模型同族 MultiSeedBaggingModel)
- 配置 B: lightgbm_ranker (截面排序目标)
- 指标: 逐日截面 RankIC (spearman, >=8 只样本), mean/ICIR/胜率/分年度 — 不做任何美化
- 裁决: mean OOS RankIC > 0 且分年度大体为正才考虑后续晋升流程; 否则如实报告 FAIL
- 本脚本只产出研究制品, 不触碰生产注册表 PRODUCTION 状态 (LIVE_TRADING_READY=False 铁律不动)
"""
import io
import json
import logging
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('retrain_v3')

RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_DIR = PROJECT_ROOT / 'reports' / 'model_research' / f'retrain_v3_{RUN_TS}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = 'label_up_down_20d'
MIN_CROSS_SECTION = 8


def daily_rank_ic(oos_df: pd.DataFrame, pred_col: str, label_col: str) -> pd.Series:
    ics = {}
    for dt, grp in oos_df.groupby('date'):
        if len(grp) < MIN_CROSS_SECTION:
            continue
        r = stats.spearmanr(grp[pred_col], grp[label_col])[0]
        if not np.isnan(r):
            ics[dt] = float(r)
    return pd.Series(ics, name='rank_ic')


def evaluate(oos_df: pd.DataFrame, pred_col: str, config_name: str) -> dict:
    ic = daily_rank_ic(oos_df, pred_col, LABEL_COL)
    ic = ic.dropna()
    mean_ic = float(ic.mean())
    std_ic = float(ic.std())
    icir_ann = float(mean_ic / (std_ic + 1e-9) * np.sqrt(242))
    win_rate = float((ic > 0).mean())
    per_year = {str(y): float(v) for y, v in ic.groupby(ic.index.year).mean().items()}

    # 多空分位价差 (Top20% - Bottom20%, 等权日均, 仅排序不做交易成本模拟)
    spreads = []
    for dt, grp in oos_df.groupby('date'):
        if len(grp) < MIN_CROSS_SECTION * 3:
            continue
        q_hi = grp[pred_col].quantile(0.8)
        q_lo = grp[pred_col].quantile(0.2)
        top = grp.loc[grp[pred_col] >= q_hi, LABEL_COL].mean()
        bot = grp.loc[grp[pred_col] <= q_lo, LABEL_COL].mean()
        if np.isfinite(top) and np.isfinite(bot):
            spreads.append(float(top - bot))
    ls_spread = float(np.mean(spreads)) if spreads else float('nan')

    result = {
        'config': config_name,
        'n_oos_rows': int(len(oos_df)),
        'n_ic_days': int(len(ic)),
        'ic_first_date': str(ic.index.min())[:10] if len(ic) else None,
        'ic_last_date': str(ic.index.max())[:10] if len(ic) else None,
        'mean_rank_ic': mean_ic,
        'ic_std': std_ic,
        'icir_annualized': icir_ann,
        'ic_positive_win_rate': win_rate,
        'per_year_mean_ic': per_year,
        'long_short_daily_spread_20pct': ls_spread,
    }
    ic.rename('rank_ic').to_frame().assign(date=lambda x: x.index).to_csv(
        OUT_DIR / f'daily_rank_ic_{config_name}.csv', index=False
    )
    return result


def run_config(model_type: str, df: pd.DataFrame, config_name: str) -> dict:
    logger.info(f'===== 配置 {config_name} (model_type={model_type}) 开始走步训练 =====')
    t0 = time.time()
    trainer = WalkForwardTrainer(
        model_type=model_type,
        task_type='classification',
        label_col=LABEL_COL,
        strict_mode=True,
        save_model=False,
    )
    oos_df, latest_model = trainer.run_walk_forward(df)
    elapsed = time.time() - t0
    logger.info(f'配置 {config_name}: {len(trainer.models)} 折完成, 耗时 {elapsed:.0f}s')

    oos_df.to_parquet(OUT_DIR / f'oos_predictions_{config_name}.parquet', index=False)
    result = evaluate(oos_df, 'pred_score', config_name)
    result['elapsed_seconds'] = round(elapsed, 1)
    result['n_folds'] = len(trainer.models)
    result['warmup_rows_excluded'] = int(trainer.warmup_rows_excluded)
    result['purge_gap_days'] = int(trainer.purge_gap_days)
    logger.info(
        f"配置 {config_name} 诚实评估: mean RankIC={result['mean_rank_ic']:+.4f} "
        f"ICIR(年化)={result['icir_annualized']:+.3f} 胜率={result['ic_positive_win_rate']:.1%} "
        f"分年度={result['per_year_mean_ic']}"
    )
    return result


def main() -> int:
    config_filter = None
    if len(sys.argv) > 1:
        config_filter = sys.argv[1]
        logger.info(f'仅运行配置: {config_filter}')

    logger.info('>> 加载 v3 因子矩阵...')
    df = pd.read_parquet(PROJECT_ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet')
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f'矩阵: {len(df)} 行 x {len(df.columns)} 列, {df["date"].min():%Y-%m-%d} -> {df["date"].max():%Y-%m-%d}')

    logger.info('>> 计算标签 label_up_down_20d (LabelRegistry v2, horizon=%d)...', settings.LABEL_HORIZON)
    raw_label = LabelRegistry.compute_label_v2(df, horizon=settings.LABEL_HORIZON)
    df[LABEL_COL] = (raw_label > 0).astype(float).mask(raw_label.isna(), np.nan)
    valid_ratio = float(df[LABEL_COL].notna().mean())
    logger.info(f'标签有效率: {valid_ratio:.1%}')
    if valid_ratio < 0.5:
        logger.error('FATAL: 标签有效率过低, 终止 (fail-closed)')
        return 2

    results = []
    all_configs = [('bagging_ensemble', 'bagging_ensemble'), ('lightgbm_ranker', 'lgbm_ranker')]
    for model_type, name in all_configs:
        if config_filter and name != config_filter:
            continue
        try:
            results.append(run_config(model_type, df, name))
        except Exception:
            logger.exception(f'配置 {name} 训练失败 (fail-closed 记录, 不美化)')
            results.append({'config': name, 'error': 'training_failed'})

    summary = {
        'run_timestamp': RUN_TS,
        'dataset': 'data_storage/research/factor_matrix_300.parquet (v3)',
        'dataset_sha256_prefix': 'bb1593ed91bec4a8',
        'label': LABEL_COL,
        'label_horizon_trading_days': int(settings.LABEL_HORIZON),
        'purge_gap_days': int(settings.PURGE_GAP_DAYS),
        'protocol': 'walk-forward strict, train 1.5y / val 3m / test 2m, purged gap 25d',
        'verdict_rule': 'mean OOS RankIC 显著为正且分年度大体为正才允许进入晋升流程; 否则 FAIL 不晋升',
        'results': results,
    }
    (OUT_DIR / 'retrain_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    logger.info(f'全部完成, 产物目录: {OUT_DIR}')
    for r in results:
        if 'error' in r:
            logger.info(f"  [{r['config']}] 训练失败")
        else:
            logger.info(
                f"  [{r['config']}] mean RankIC={r['mean_rank_ic']:+.4f} | ICIR={r['icir_annualized']:+.3f} | "
                f"胜率={r['ic_positive_win_rate']:.1%} | 多空价差={r['long_short_daily_spread_20pct']:+.5f}"
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
