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
    logger.info('正在注入 7 大高胜率异源 Alpha 因子群 (通过统一 FactorProcessor)...')
    from factors.processor import FactorProcessor
    df = FactorProcessor.compute_advanced_alpha_features(df)

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

    # 严格使用已注册生产模型执行批量推理 (单一生产链 Fail-Closed)
    from models.inference import BatchInference, InferenceError
    logger.info('正在加载 ModelRegistry 已上线的生产模型...')
    try:
        engine = BatchInference()
        logger.info(f"成功载入生产模型: {engine.model_id} (Schema Hash: {engine.expected_schema_hash[:12]}...)")
        latest_slice = engine.predict(df, date=latest_date)
    except Exception as e:
        logger.error(f"生产模型推理失败 (Fail-Closed: 拒绝无生产模型或特征缺失时退化重训): {e}")
        raise

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
    print(f"   生产模型: {engine.model_id} | 预测任务: 未来 {settings.LABEL_HORIZON} 个交易日超额概率\n")

    rows = []
    print(f"{'排名':<4} | {'代码':<9} | {'股票名称':<8} | {'所属行业':<10} | {'收盘价':<8} | {'预测概率/得分':<14} | {'建议配置权重':<10}")
    print('-' * 75)
    for idx, r in target_portfolio.reset_index().iterrows():
        s_name = r['name'] if pd.notna(r['name']) and r['name'] else 'N/A'
        ind = r.get('industry', '未知')
        close_p = f"{r['close']:.2f}元"
        pred_pct = f"{r['pred_score']:.4f}"
        w_pct = f"{r['target_weight']*100:.2f}%"
        print(f"{idx+1:<4} | {r['symbol']:<9} | {s_name:<8} | {ind:<10} | {close_p:<8} | {pred_pct:<14} | {w_pct:<10}")
        rows.append({
            'rank': idx + 1,
            'symbol': r['symbol'],
            'name': s_name,
            'industry': ind,
            'close': r['close'],
            'pred_score': r['pred_score'],
            'target_weight': r['target_weight'],
            'model_id': engine.model_id,
            'model_state': engine.record.state,
            'data_as_of': dt_str,
            'feature_schema_hash': engine.expected_schema_hash,
            'is_synthetic_demo': False
        })

    # 特征重要性
    if hasattr(engine.model, 'get_feature_importance'):
        fi = engine.model.get_feature_importance(top_n=8)
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
