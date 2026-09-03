import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import settings
from research_v2.alphas.novel_alphas import NovelAlphaFactory

print('=== 启动第三代 Mega-Alpha (DoubleEnsemble + TabularMLP + Ridge) 最新截面股票预测 ===')

# 1. 加载第三代终极大模型
cand_path = settings.MODELS_DIR / 'candidate_gen3_mega_model.pkl'
model = joblib.load(cand_path)
print(f'成功加载第三代大模型: {cand_path}')

# 2. 加载全量因子矩阵并计算 8 大新 Alpha
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

latest_date = df['date'].max()
dt_str = latest_date.strftime("%Y-%m-%d")
print(f'最新截面交易日: {dt_str}')

cross_df = df[df['date'] == latest_date].copy()
print(f'最新截面候选标的数: {len(cross_df)} 支')

# 3. 对齐特征并预测
feats = model.feature_names
X_latest = cross_df[feats].fillna(0.0)

# 预测输出
scores = model.predict(X_latest)
cross_df['pred_score'] = scores

# 4. 排序筛选 Top 10 核心组合
top_picks = cross_df.sort_values(by='pred_score', ascending=False).head(10).copy()

# 归一化分配组合目标权重 (保留 15% 现金安全垫, 总仓位 85%, 单股不超 15%)
raw_w = (top_picks['pred_score'] - top_picks['pred_score'].min()) / (top_picks['pred_score'].max() - top_picks['pred_score'].min() + 1e-6)
raw_w = raw_w + 0.5  # 平滑基准
norm_w = (raw_w / raw_w.sum()) * 0.85
top_picks['target_weight'] = np.clip(norm_w, 0.05, 0.15)
# 重新归一化至 85%
top_picks['target_weight'] = (top_picks['target_weight'] / top_picks['target_weight'].sum()) * 0.85

out_cols = ['date', 'symbol', 'name', 'industry', 'close', 'pred_score', 'target_weight']
existing_cols = [c for c in out_cols if c in top_picks.columns]
res_df = top_picks[existing_cols].copy()

out_path = settings.BASE_DIR / 'artifacts' / 'gen3_latest_stock_picks.csv'
res_df.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'[+] 第三代终极预测选股清单已落盘: {out_path}')

print('\nTop 8 核心重仓推荐标的:')
for i, (_, row) in enumerate(res_df.head(8).iterrows(), 1):
    sym = row['symbol']
    name = row.get('name', 'N/A')
    ind = row.get('industry', 'N/A')
    p = row.get('close', 0.0)
    sc = row['pred_score']
    w = row['target_weight']
    print(f'  {i}. {sym} | {name:<6} | {ind:<6} | 收盘价: {p:6.2f}元 | 终极得分: {sc:.4f} | 目标仓位: {w*100:4.1f}%')
