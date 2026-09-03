"""
生产级即时股票预测与决策清单生成工具 (tools/predict_stocks.py)
用于 Step A: 快速完成最新截面的股票预测、风控过滤与目标仓位构建。
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

from config.settings import settings
from strategy.portfolio import PortfolioBuilder
from data.universe_provider import create_universe_provider

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('predict_stocks')

def run_latest_prediction():
    print('\n' + '=' * 75)
    print('>> [Step A] 启动 A股多因子 AI 股票预测与选股决策引擎')
    print('=' * 75)

    dataset_path = root_dir / 'data_storage' / 'research' / 'factor_matrix_300.parquet'
    if not dataset_path.exists():
        logger.error(f'未找到基础因子数据集: {dataset_path}')
        return

    logger.info('正在加载多因子全量数据矩阵 (沪深300)...')
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])

    # 动态注入 7 大高胜率异源 Alpha
    logger.info('正在注入 7 大高胜率异源 Alpha 因子群...')
    from research_v2.alphas.novel_alphas import NovelAlphaFactory
    df['ALPHA_RESIDUAL_MOMENTUM_20'] = NovelAlphaFactory.calc_residual_momentum(df, window=20)
    df['ALPHA_TURNOVER_SURPRISE_5_20'] = NovelAlphaFactory.calc_turnover_surprise(df, short_w=5, long_w=20)
    df['ALPHA_QUALITY_X_MOMENTUM'] = NovelAlphaFactory.calc_quality_x_momentum(df)
    df['ALPHA_LIQUIDITY_X_VOL'] = NovelAlphaFactory.calc_liquidity_x_volatility(df)
    df['ALPHA_SHORT_REVERSAL_5'] = NovelAlphaFactory.calc_short_term_reversal(df, window=5)
    df['ALPHA_IDIO_VOL_PENALTY'] = NovelAlphaFactory.calc_idio_vol_penalty(df, window=20)
    df['ALPHA_MONEY_FLOW_DIV_10'] = NovelAlphaFactory.calc_money_flow_divergence(df, window=10)

    latest_date = df['date'].max()
    dt_str = latest_date.strftime("%Y-%m-%d")
    n_syms = df['symbol'].nunique()
    print(f"[*] 数据集最新时点: {dt_str} (覆盖股票: {n_syms} 支)")

    # 加载股票中文名称字典
    sec_master_path = root_dir / 'data_storage' / 'security_master.parquet'
    name_map = {}
    if sec_master_path.exists():
        sm_df = pd.read_parquet(sec_master_path)
        if 'name' in sm_df.columns:
            name_map = dict(zip(sm_df['symbol'], sm_df['name']))

    # 加载生产注册模型
    prod_model_path = root_dir / 'saved_models' / 'latest_lightgbm.pkl'
    latest_model = None
    if prod_model_path.exists():
        try:
            with open(prod_model_path, 'rb') as f:
                latest_model = pickle.load(f)
            logger.info('成功加载最新生产模型: latest_lightgbm.pkl')
        except Exception as e:
            logger.warning(f'加载生产模型失败: {e}')

    # 获取特征列 (若生产模型已明确指定特征列则严格对齐)
    if latest_model is not None and hasattr(latest_model, 'feature_names') and latest_model.feature_names:
        feature_cols = [f for f in latest_model.feature_names if f in df.columns]
    else:
        feature_cols = [c for c in df.columns if c not in [
            'date', 'symbol', 'in_universe', 'label_excess_20d', 'label_up_down_20d',
            'is_suspended', 'is_limit_up_locked', 'is_limit_down_locked', 'benchmark_open',
            'benchmark_close', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change',
            'is_st', 'limit_up_price', 'limit_down_price', 'industry', 'list_date', 'days_since_listing'
        ] and np.issubdtype(df[c].dtype, np.number)]

    latest_slice = df[df['date'] == latest_date].copy()
    X_latest = latest_slice[feature_cols].fillna(0.0)

    if latest_model is not None and hasattr(latest_model, 'predict'):
        preds = latest_model.predict(X_latest)
        latest_slice['pred_score'] = preds
    else:
        # 使用时序走步训练器快速产出
        from models.walk_forward import WalkForwardTrainer
        from models.labeler import TargetLabeler
        logger.info('正在执行 Walk-Forward 最新折模型预测...')
        labeler = TargetLabeler(horizon=settings.LABEL_HORIZON)
        df = labeler.compute_excess_return_label(df)
        trainer = WalkForwardTrainer(random_state=42)
        oos_df, latest_model = trainer.run_walk_forward(df, feature_cols=feature_cols[:25])
        latest_slice = oos_df[oos_df['date'] == latest_date].copy()

    # 计算截面百分比排名
    latest_slice['pred_rank'] = latest_slice['pred_score'].rank(pct=True, ascending=False)
    latest_slice['name'] = latest_slice['symbol'].map(lambda s: name_map.get(s, ''))

    # 构建组合优化器
    univ_provider = create_universe_provider(settings)
    builder = PortfolioBuilder(
        top_k_buy=settings.TOP_K_BUY,
        top_k_hold=settings.TOP_K_HOLD,
        weight_method='inv_vol',
        universe_provider=univ_provider
    )
    target_portfolio = builder.build_target_portfolio(latest_slice, current_holdings=set(), date=latest_date)

    print('\n' + '=' * 75)
    print(f"[PREDICTION] A股最新多因子预测结果与选股决策清单 (信号日期: {dt_str})")
    print('=' * 75)
    print("   选股股票池: 沪深300核心成分股 | 组合优化: 倒波动率加权 (Inverse Volatility)")
    print(f"   预测目标: 未来 {settings.LABEL_HORIZON} 个交易日预期超额收益率\n")

    rows = []
    print(f"{'排名':<4} | {'代码':<9} | {'股票名称':<8} | {'所属行业':<10} | {'收盘价':<8} | {'预测超额':<10} | {'建议配置权重':<10}")
    print('-' * 75)
    for idx, r in target_portfolio.reset_index().iterrows():
        s_name = r['name'] if pd.notna(r['name']) and r['name'] else 'N/A'
        ind = r.get('industry', '未知')
        close_p = f"{r['close']:.2f}元"
        pred_pct = f"{r['pred_score']*100:+.2f}%"
        w_pct = f"{r['target_weight']*100:.2f}%"
        print(f"{idx+1:<4} | {r['symbol']:<9} | {s_name:<8} | {ind:<10} | {close_p:<8} | {pred_pct:<10} | {w_pct:<10}")
        rows.append({
            'rank': idx + 1,
            'symbol': r['symbol'],
            'name': s_name,
            'industry': ind,
            'close': r['close'],
            'pred_score': r['pred_score'],
            'target_weight': r['target_weight']
        })

    # 特征重要性
    if hasattr(latest_model, 'get_feature_importance'):
        fi = latest_model.get_feature_importance(top_n=8)
        print('\n' + '-' * 75)
        print('[ALPHA_CONTRIBUTION] 驱动本次预测的核心有效 Alpha 因子 Top 8:')
        for _, f_row in fi.iterrows():
            print(f"   * {f_row['feature']:<25} : 贡献权重 {f_row['importance_pct']:.1f}%")

    # 保存产物
    art_dir = root_dir / 'artifacts'
    art_dir.mkdir(exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(art_dir / 'latest_stock_picks.csv', index=False, encoding='utf-8-sig')
    print('\n' + '=' * 75)
    print(f"[DONE] 预测报告已成功持久化落盘: {art_dir / 'latest_stock_picks.csv'}")
    print('=' * 75)

if __name__ == '__main__':
    run_latest_prediction()
