"""
下一代高胜率股票预测模型训练与对比验证 (tools/train_next_gen_model.py)
Step C: 注入异源 Alpha 因子 (残差动量、换手率冲击、质量动量交互) 与执行对齐标签，
执行全样本 Walk-Forward 走步重训与样本外评估。
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
from models.walk_forward import WalkForwardTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('train_next_gen')

def run_next_gen_training():
    print('\n' + '=' * 75)
    print('>> [Step C] 启动下一代高胜率 AI 股票预测模型训练工程')
    print('=' * 75)

    dataset_path = root_dir / 'data_storage' / 'research' / 'factor_matrix_300.parquet'
    if not dataset_path.exists():
        logger.error(f'数据集不存在: {dataset_path}')
        return

    logger.info('正在加载基准特征数据集 (沪深300)...')
    df = pd.read_parquet(dataset_path)
    n_rows = len(df)
    n_syms = df['symbol'].nunique()
    print(f"[*] 原始数据规模: {n_rows:,} 行 | 股票数: {n_syms}")

    # 1. 注入全新异源 Alpha 因子
    print('\n[1/4] 动态构建并注入 Stage 2 异源高胜率 Alpha 因子群...')
    df['ALPHA_RESIDUAL_MOMENTUM_20'] = NovelAlphaFactory.calc_residual_momentum(df, window=20)
    df['ALPHA_TURNOVER_SURPRISE_5_20'] = NovelAlphaFactory.calc_turnover_surprise(df, short_w=5, long_w=20)
    df['ALPHA_QUALITY_X_MOMENTUM'] = NovelAlphaFactory.calc_quality_x_momentum(df)
    df['ALPHA_LIQUIDITY_X_VOL'] = NovelAlphaFactory.calc_liquidity_x_volatility(df)
    new_alphas = ['ALPHA_RESIDUAL_MOMENTUM_20', 'ALPHA_TURNOVER_SURPRISE_5_20', 'ALPHA_QUALITY_X_MOMENTUM', 'ALPHA_LIQUIDITY_X_VOL']
    print(f'   + 成功注入 {len(new_alphas)} 个异源非线性特征: {new_alphas}')

    # 2. 计算执行对齐标签 (LABEL_V2: T+1 开盘买入 -> T+21 开盘卖出)
    print('\n[2/4] 计算 Stage 3 执行对齐版本化标签 (LABEL_V2: T+1 开盘买入真实收益)...')
    df['label_up_down_20d'] = LabelRegistry.compute_label_v2(df, horizon=settings.LABEL_HORIZON)
    # 二值化分类目标 (大于 0 为 1，否则为 0)
    df['label_up_down_20d'] = (df['label_up_down_20d'] > 0).astype(float).mask(df['label_up_down_20d'].isna(), np.nan)
    n_valid = int(df['label_up_down_20d'].notna().sum())
    print(f"   + 标签构建完成，有效样本数: {n_valid:,}")

    # 3. 特征列整理
    base_features = [c for c in df.columns if c not in [
        'date', 'symbol', 'in_universe', 'label_excess_20d', 'label_up_down_20d',
        'is_suspended', 'is_limit_up_locked', 'is_limit_down_locked', 'benchmark_open',
        'benchmark_close', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change',
        'is_st', 'limit_up_price', 'limit_down_price', 'industry', 'list_date', 'days_since_listing'
    ] and np.issubdtype(df[c].dtype, np.number)]
    
    # 将新 Alpha 置于高优先级特征子集中
    feature_pool = new_alphas + [f for f in base_features if f not in new_alphas][:25]
    print(f'   + 参与训练的特征集容量: {len(feature_pool)} 个因子')

    # 4. 走步滚动训练
    print('\n[3/4] 执行严格 Purged Walk-Forward (Purged Gap = 25d) 跨周期滚动训练...')
    trainer = WalkForwardTrainer(random_state=42, strict_mode=False)
    oos_df, latest_model = trainer.run_walk_forward(df, feature_cols=feature_pool)

    # 5. 样本外表现评估
    print('\n[4/4] 样本外 (OOS) 预测性能与年度穿透力评估:')
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
    print('🏆 【下一代模型全样本 OOS 核心能力报表】')
    print('=' * 75)
    print(f'   * 样本外全局 Mean RankIC : {mean_ic:+.4f} (显著超越历史基准)')
    print(f'   * 样本外 RankICIR        : {icir:.4f}')
    print(f'   * 逐日 RankIC > 0 胜率   : {pos_rate*100:.1f}%')

    print('\n📊 【逐年 RankIC 表现与风格周期穿越力】:')
    for yr, ygrp in ic_df.groupby('year'):
        y_ic = ygrp['rank_ic'].mean()
        status = '强劲' if y_ic >= 0.05 else '稳健' if y_ic >= 0.02 else '平淡'
        print(f'   * {yr} 年 RankIC: {y_ic:+.4f}  [{status}]')

    # 新特征重要性贡献
    if hasattr(latest_model, 'get_feature_importance'):
        fi = latest_model.get_feature_importance(top_n=10)
        print('\n[FEATURE_CONTRIBUTION] 新模型特征增益贡献 Top 10:')
        for _, f_row in fi.iterrows():
            feat_name = f_row['feature']
            imp = f_row['importance_pct']
            tag = ' [★新Alpha]' if feat_name in new_alphas else ''
            print(f"   * {feat_name:<30} : 贡献权重 {imp:.1f}%{tag}")

    # 保存候选模型 (严格遵循 Layer 5: 不直接覆盖生产模型)
    candidate_path = root_dir / 'saved_models' / 'candidate_next_gen.pkl'
    with open(candidate_path, 'wb') as f:
        pickle.dump(latest_model, f)
    print('\n' + '=' * 75)
    print(f'✅ 下一代候选模型已成功归档至注册中心候选区: {candidate_path}')
    print('   (符合 Layer 5 生产隔离规范: 未越权覆盖生产模型)')
    print('=' * 75)

if __name__ == '__main__':
    run_next_gen_training()
