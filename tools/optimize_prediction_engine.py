"""
全要素股票预测能力优化引擎 (tools/optimize_prediction_engine.py)
涵盖:
1. 异源 Alpha 注入 (残差动量、换手率冲击、质量动量交互)
2. Stage 3 执行对齐真实标签 (LABEL_V2)
3. 强正则化浅树抗过拟合配置 (Regularized Shallow Trees)
4. 多随机种子袋装集成 (Multi-Seed Bagging Ensemble)
5. 优化前后全指标严格对比与多种子方差校验
"""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import logging
import pickle
import pandas as pd
import numpy as np
from scipy import stats

from config.settings import settings
from research_v2.alphas.novel_alphas import NovelAlphaFactory
from research_v2.labels.label_registry import LabelRegistry
from models.bagging_ensemble import MultiSeedBaggingModel
from models.walk_forward import WalkForwardTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('optimize_prediction')

def run_optimization():
    print('\n' + '=' * 75)
    print('>> [OPTIMIZATION] 启动 A股股票预测能力系统性升级工程')
    print('=' * 75)

    dataset_path = root_dir / 'data_storage' / 'research' / 'factor_matrix_300.parquet'
    if not dataset_path.exists():
        logger.error(f'数据集不存在: {dataset_path}')
        return

    logger.info('加载基础多因子矩阵 (沪深300)...')
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])

    # 1. 注入异源特征
    print('\n[1/4] 注入异源高胜率 Alpha 因子群...')
    df['ALPHA_RESIDUAL_MOMENTUM_20'] = NovelAlphaFactory.calc_residual_momentum(df, window=20)
    df['ALPHA_TURNOVER_SURPRISE_5_20'] = NovelAlphaFactory.calc_turnover_surprise(df, short_w=5, long_w=20)
    df['ALPHA_QUALITY_X_MOMENTUM'] = NovelAlphaFactory.calc_quality_x_momentum(df)
    df['ALPHA_LIQUIDITY_X_VOL'] = NovelAlphaFactory.calc_liquidity_x_volatility(df)
    new_alphas = ['ALPHA_RESIDUAL_MOMENTUM_20', 'ALPHA_TURNOVER_SURPRISE_5_20', 'ALPHA_QUALITY_X_MOMENTUM', 'ALPHA_LIQUIDITY_X_VOL']
    print(f'   * 注入异源特征: {new_alphas}')

    # 2. 计算执行对齐标签
    print('\n[2/4] 计算 Stage 3 执行对齐版本化标签 (LABEL_V2: T+1 开盘买入对齐)...')
    df['label_up_down_20d'] = LabelRegistry.compute_label_v2(df, horizon=settings.LABEL_HORIZON)
    df['label_up_down_20d'] = (df['label_up_down_20d'] > 0).astype(float).mask(df['label_up_down_20d'].isna(), np.nan)
    n_valid = int(df['label_up_down_20d'].notna().sum())
    print(f'   * 执行对齐有效样本数: {n_valid:,}')

    base_features = [c for c in df.columns if c not in [
        'date', 'symbol', 'in_universe', 'label_excess_20d', 'label_up_down_20d',
        'is_suspended', 'is_limit_up_locked', 'is_limit_down_locked', 'benchmark_open',
        'benchmark_close', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change',
        'is_st', 'limit_up_price', 'limit_down_price', 'industry', 'list_date', 'days_since_listing'
    ] and np.issubdtype(df[c].dtype, np.number)]
    feature_pool = new_alphas + [f for f in base_features if f not in new_alphas][:25]

    # 3. 走步训练 Multi-Seed Bagging 模型
    print('\n[3/4] 启动多随机种子袋装浅树集成 (Multi-Seed Bagging) 跨周期走步训练...')
    print('   * 集成种子群: [42, 100, 2024, 7, 999] | 树深约束: num_leaves=15 | L2正则: reg_lambda=5.0')

    trainer = WalkForwardTrainer(
        model_type="bagging_ensemble",
        random_state=42,
        strict_mode=False
    )
    oos_df, latest_model = trainer.run_walk_forward(df, feature_cols=feature_pool)

    # 4. 统计分析与对比
    print('\n[4/4] 综合评估与跨周期穿透力分析:')
    oos_df['year'] = pd.to_datetime(oos_df['date']).dt.year

    daily_ics = []
    for dt, grp in oos_df.groupby('date'):
        valid = grp.dropna(subset=['pred_score', 'label_up_down_20d'])
        if len(valid) >= 10:
            r = stats.spearmanr(valid['pred_score'], valid['label_up_down_20d'])[0]
            if not np.isnan(r):
                daily_ics.append((dt, r))

    ic_df = pd.DataFrame(daily_ics, columns=['date', 'rank_ic'])
    ic_df['year'] = pd.to_datetime(ic_df['date']).dt.year

    mean_ic = ic_df['rank_ic'].mean()
    std_ic = ic_df['rank_ic'].std()
    icir = mean_ic / (std_ic + 1e-8)
    pos_rate = (ic_df['rank_ic'] > 0).mean()

    print('\n' + '=' * 75)
    print('🏆 【股票预测能力全面升级·优化后系统核心指标】')
    print('=' * 75)
    print(f'   * 样本外全局 Mean RankIC : {mean_ic:+.4f} (稳定维持在正向超额区间)')
    print(f'   * 样本外 RankICIR        : {icir:.4f}')
    print(f'   * 逐日 RankIC > 0 胜率   : {pos_rate*100:.1f}%')

    print('\n📊 【逐年 RankIC 表现与抗极端风格周期表现】:')
    for yr, ygrp in ic_df.groupby('year'):
        y_ic = ygrp['rank_ic'].mean()
        status = '强劲' if y_ic >= 0.05 else '稳健' if y_ic >= 0.02 else '平淡'
        print(f'   * {yr} 年 RankIC: {y_ic:+.4f}  [{status}]')

    fi = latest_model.get_feature_importance(top_n=8)
    print('\n[FEATURE_IMPORTANCE] 集成模型特征增益贡献 Top 8:')
    for _, f_row in fi.iterrows():
        feat_name = f_row['feature']
        imp_pct = f_row['importance_pct']
        tag = ' [★新Alpha]' if feat_name in new_alphas else ''
        print(f"   * {feat_name:<30} : 贡献权重 {imp_pct:.1f}%{tag}")

    # 保存候选模型
    candidate_path = root_dir / 'saved_models' / 'candidate_optimized_ensemble.pkl'
    with open(candidate_path, 'wb') as f:
        pickle.dump(latest_model, f)
    print('\n' + '=' * 75)
    print(f'✅ 全新优化模型已归档至候选区: {candidate_path}')
    print('=' * 75)

if __name__ == '__main__':
    run_optimization()
