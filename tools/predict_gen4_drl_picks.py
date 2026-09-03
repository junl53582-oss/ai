import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import settings
from research_v2.alphas.novel_alphas import NovelAlphaFactory

print('=== 启动第四代 DRL 强化模型 (Mega-Alpha + 策略梯度智能体) 最新选股与权重优化 ===')

# 1. 加载第四代强化大模型
cand_path = settings.MODELS_DIR / 'candidate_gen4_drl_model.pkl'
drl_model = joblib.load(cand_path)
print(f'成功加载第四代强化大模型: {cand_path}')

# 2. 读取全量特征矩阵
matrix_path = settings.DATA_DIR / 'research' / 'factor_matrix_300.parquet'
df = pd.read_parquet(matrix_path)
df['date'] = pd.to_datetime(df['date'])

df['ALPHA_RESIDUAL_MOM_20'] = NovelAlphaFactory.calc_residual_momentum(df, window=20)
df['ALPHA_TURNOVER_SURPRISE_5_20'] = NovelAlphaFactory.calc_turnover_surprise(df, short_w=5, long_w=20)
df['ALPHA_QUALITY_MOM'] = NovelAlphaFactory.calc_quality_x_momentum(df)
df['ALPHA_LIQ_VOL_CROSS'] = NovelAlphaFactory.calc_liquidity_x_volatility(df)
df['ALPHA_SHORT_REV_5'] = NovelAlphaFactory.calc_short_term_reversal(df, window=5)
df['ALPHA_IDIO_VOL_PENALTY'] = NovelAlphaFactory.calc_idio_vol_penalty(df, window=20)
df['ALPHA_MONEY_FLOW_DIV_10'] = NovelAlphaFactory.calc_money_flow_divergence(df, window=10)
df['ALPHA_TAIL_LIQUIDITY_BIAS'] = NovelAlphaFactory.calc_tail_liquidity_bias(df, window=15)

latest_dt = df['date'].max()
dt_str = latest_dt.strftime('%Y-%m-%d')
print(f'最新截面交易日: {dt_str}')

cross_df = df[df['date'] == latest_dt].copy()

# 3. 预测基础 Alpha
feats = drl_model.feature_names
X_latest = cross_df[feats].fillna(0.0)
cross_df['pred_score'] = drl_model.predict_alpha(X_latest)

if 'volatility' not in cross_df.columns:
    cross_df['volatility'] = 0.02

# 4. 排序筛选 Top 10 并由 DRL 智能体执行强化动态权重优化
top_candidates = cross_df.sort_values(by='pred_score', ascending=False).head(10).copy()
drl_optimized = drl_model.optimize_portfolio(top_candidates)

out_cols = ['date', 'symbol', 'name', 'industry', 'close', 'pred_score', 'drl_target_weight']
res_df = drl_optimized[[c for c in out_cols if c in drl_optimized.columns]].copy()
res_df.rename(columns={'drl_target_weight': 'target_weight'}, inplace=True)

out_path = settings.BASE_DIR / 'artifacts' / 'gen4_drl_stock_picks.csv'
res_df.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'[+] 第四代强化模型最新选股与动态权重已落盘: {out_path}')

print('\nTop 8 DRL 强化模型动态仓位推荐:')
for i, (_, row) in enumerate(res_df.head(8).iterrows(), 1):
    sym = row['symbol']
    name = row.get('name', 'N/A')
    ind = row.get('industry', 'N/A')
    p = row.get('close', 0.0)
    sc = row['pred_score']
    w = row['target_weight']
    print(f'  {i}. {sym} | {name:<6} | {ind:<6} | 2026-09-03收盘: {p:6.2f}元 | 预测分: {sc:.4f} | DRL最优仓位: {w*100:4.1f}%')
