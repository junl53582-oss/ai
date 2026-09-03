import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('diagnose_seed_variance')

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import settings

def run_seed_diagnosis():
    # 找到最新的运行目录
    runs_dir = PROJECT_ROOT / 'reports' / 'audit_hardening_v3' / 'runs'
    latest_run = max(runs_dir.glob('research_*'), key=lambda p: p.stat().st_mtime)
    logger.info(f'分析目标运行目录: {latest_run}')

    seed_file = latest_run / 'multi_seed_robustness.json'
    if not seed_file.exists():
        logger.error(f'缺失种子文件: {seed_file}')
        return

    with open(seed_file, encoding='utf-8') as f:
        seed_data = json.load(f)

    logger.info('=' * 60)
    logger.info('Stage 4: Seed Variance Root-Cause Diagnosis (种子方差根因深度诊断)')
    logger.info('=' * 60)

    s_set = seed_data.get("seed_set")
    s_each = seed_data.get("seed_rankic_each")
    s_std = seed_data.get("seed_rankic_std", 0.0)
    logger.info(f"种子集合: {s_set}")
    logger.info(f"各种子全样本 RankIC: {s_each}")
    logger.info(f"种子方差 STD: {s_std:.6f} (门禁阈值: 0.005000) -> FAIL")

    # 读取数据集以进行精确时序分解
    dataset_path = PROJECT_ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet'
    if not dataset_path.exists():
        logger.warning(f'数据集文件不存在: {dataset_path}')
        return

    df = pd.read_parquet(dataset_path)
    from models.labeler import TargetLabeler
    labeler = TargetLabeler(horizon=settings.LABEL_HORIZON)
    df = labeler.compute_excess_return_label(df)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    from models.walk_forward import WalkForwardTrainer

    feature_cols = [c for c in df.columns if c not in [
        'date', 'symbol', 'in_universe', 'label_excess_20d', 'label_up_down_20d',
        'is_suspended', 'is_limit_up_locked', 'is_limit_down_locked', 'benchmark_open',
        'benchmark_close', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change'
    ] and np.issubdtype(df[c].dtype, np.number)]

    seeds = [42, 100, 2024]
    fold_ic_matrix = {s: [] for s in seeds}
    year_ic_matrix = {s: {} for s in seeds}
    seed_preds = {}

    for s in seeds:
        logger.info(f'--> 重新提取 Seed {s} 的逐折与逐年细粒度表现...')
        trainer = WalkForwardTrainer(
            train_years=settings.TRAIN_WINDOW_YEARS,
            val_months=settings.VAL_WINDOW_MONTHS,
            test_months=settings.TEST_WINDOW_MONTHS,
            purge_gap_days=settings.PURGE_GAP_DAYS,
            random_state=s,
            strict_mode=False
        )
        oos_df, _ = trainer.run_walk_forward(df, feature_cols=feature_cols[:25])
        oos_df['year'] = pd.to_datetime(oos_df['date']).dt.year
        seed_preds[s] = oos_df[['date', 'symbol', 'pred_score', 'label_up_down_20d', 'year']].copy()

        # 逐年 RankIC
        for yr, group in oos_df.groupby('year'):
            ics = []
            for dt, sub in group.groupby('date'):
                sub_val = sub.dropna(subset=['pred_score', 'label_up_down_20d'])
                if len(sub_val) >= 5:
                    r = stats.spearmanr(sub_val['pred_score'], sub_val['label_up_down_20d'])[0]
                    if not np.isnan(r):
                        ics.append(r)
            year_ic_matrix[s][yr] = float(np.mean(ics)) if ics else 0.0

    # 1. 输出 Seed × Year RankIC 矩阵
    year_df = pd.DataFrame(year_ic_matrix).round(5)
    year_df['Year_STD'] = year_df.std(axis=1).round(5)
    logger.info('\n[1] Seed × Year RankIC 矩阵:')
    print(year_df.to_string())

    # 2. 输出 Seed 间预测值相关性矩阵
    p_df = seed_preds[42][['date', 'symbol']].copy()
    for s in seeds:
        p_df[f'pred_{s}'] = seed_preds[s]['pred_score']
    
    corr_matrix = p_df[[f'pred_{s}' for s in seeds]].corr().round(4)
    logger.info('\n[2] Seed 间预测分截面相关性矩阵 (Prediction Correlation):')
    print(corr_matrix.to_string())

    # 保存诊断报告
    diag_dir = latest_run / 'seed_variance_diagnosis'
    diag_dir.mkdir(exist_ok=True)
    year_df.to_csv(diag_dir / 'seed_year_rankic_matrix.csv')
    corr_matrix.to_csv(diag_dir / 'seed_prediction_correlation.csv')

    logger.info(f'\n诊断结果已归档至: {diag_dir}')

if __name__ == '__main__':
    run_seed_diagnosis()
