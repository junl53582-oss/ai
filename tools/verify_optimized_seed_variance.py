import logging
import sys
import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import numpy as np
from scipy import stats
from config.settings import settings
from research_v2.alphas.novel_alphas import NovelAlphaFactory
from research_v2.labels.label_registry import LabelRegistry
from models.bagging_ensemble import MultiSeedBaggingModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('verify_seed_variance')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def verify():
    print('>> 正在验证优化后模型的多种子稳定性 (MULTI_SEED_ROBUSTNESS)...')
    df = pd.read_parquet(PROJECT_ROOT / 'data_storage' / 'research' / 'factor_matrix_300.parquet')
    df['date'] = pd.to_datetime(df['date'])

    # 注入新特征
    df['ALPHA_RESIDUAL_MOMENTUM_20'] = NovelAlphaFactory.calc_residual_momentum(df, window=20)
    df['ALPHA_QUALITY_X_MOMENTUM'] = NovelAlphaFactory.calc_quality_x_momentum(df)
    df['label_up_down_20d'] = LabelRegistry.compute_label_v2(df, horizon=settings.LABEL_HORIZON)
    df['label_up_down_20d'] = (df['label_up_down_20d'] > 0).astype(float).mask(df['label_up_down_20d'].isna(), np.nan)

    # 取最近 2 年的样本快速测试 3 个不同 Bagging 组合的方差
    recent_df = df[df['date'] >= '2024-01-01'].copy()
    features = ['ALPHA_RESIDUAL_MOMENTUM_20', 'ALPHA_QUALITY_X_MOMENTUM', 'LOG_CIRC_MV', 'ROC30', 'turnover']

    test_runs = [
        [42, 100, 2024],
        [100, 2024, 7],
        [2024, 7, 999]
    ]

    rank_ics = []
    # 划分训练集和测试集
    split_dt = '2025-01-01'
    train_sub = recent_df[recent_df['date'] < split_dt].dropna(subset=features + ['label_up_down_20d'])
    test_sub = recent_df[recent_df['date'] >= split_dt].dropna(subset=features + ['label_up_down_20d'])

    for idx, s_group in enumerate(test_runs):
        m = MultiSeedBaggingModel(seeds=s_group, n_estimators=60, num_leaves=15)
        m.fit(train_sub[features], train_sub['label_up_down_20d'])
        preds = m.predict(test_sub[features])
        test_sub_copy = test_sub.copy()
        test_sub_copy['pred'] = preds
        
        ics = []
        for dt, grp in test_sub_copy.groupby('date'):
            if len(grp) >= 10:
                r = stats.spearmanr(grp['pred'], grp['label_up_down_20d'])[0]
                if not np.isnan(r):
                    ics.append(r)
        mean_ic = float(np.mean(ics))
        rank_ics.append(mean_ic)
        print(f'   Run {idx+1} (Seeds: {s_group}) -> OOS RankIC: {mean_ic:+.4f}')

    std = float(np.std(rank_ics))
    print(f'[*] 多种子 STD: {std:.6f} (门禁阈值: 0.005000)')
    if std <= 0.0050:
        print('[PASS] MULTI_SEED_ROBUSTNESS: PASS! 种子方差成功攻克！')
    else:
        print('[FAIL] MULTI_SEED_ROBUSTNESS: FAIL')

if __name__ == '__main__':
    verify()
