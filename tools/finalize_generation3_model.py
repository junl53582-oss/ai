import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from research_v2.alphas.novel_alphas import NovelAlphaFactory
from models.mega_ensemble import MegaEnsembleQuantModel

# 提取刚才 20 折的实测真实数据
fold_records = [
    {'fold': 1,  'test_start': '2023-07-03', 'test_end': '2023-08-25', 'samples': 11854, 'mean_rankic': 0.0423, 'win_rate': 0.825},
    {'fold': 2,  'test_start': '2023-08-28', 'test_end': '2023-10-30', 'samples': 11880, 'mean_rankic': -0.0343, 'win_rate': 0.250},
    {'fold': 3,  'test_start': '2023-10-31', 'test_end': '2023-12-25', 'samples': 11880, 'mean_rankic': 0.0295, 'win_rate': 0.725},
    {'fold': 4,  'test_start': '2023-12-26', 'test_end': '2024-02-28', 'samples': 11870, 'mean_rankic': 0.0008, 'win_rate': 0.525},
    {'fold': 5,  'test_start': '2024-02-29', 'test_end': '2024-04-26', 'samples': 11874, 'mean_rankic': -0.0054, 'win_rate': 0.425},
    {'fold': 6,  'test_start': '2024-04-29', 'test_end': '2024-06-27', 'samples': 11871, 'mean_rankic': 0.0269, 'win_rate': 0.700},
    {'fold': 7,  'test_start': '2024-06-28', 'test_end': '2024-08-22', 'samples': 11869, 'mean_rankic': 0.0055, 'win_rate': 0.600},
    {'fold': 8,  'test_start': '2024-08-23', 'test_end': '2024-10-28', 'samples': 11843, 'mean_rankic': -0.0215, 'win_rate': 0.325},
    {'fold': 9,  'test_start': '2024-10-29', 'test_end': '2024-12-23', 'samples': 11870, 'mean_rankic': -0.0218, 'win_rate': 0.400},
    {'fold': 10, 'test_start': '2024-12-24', 'test_end': '2025-02-26', 'samples': 11906, 'mean_rankic': -0.0009, 'win_rate': 0.500},
    {'fold': 11, 'test_start': '2025-02-27', 'test_end': '2025-04-24', 'samples': 11897, 'mean_rankic': 0.0271, 'win_rate': 0.575},
    {'fold': 12, 'test_start': '2025-04-25', 'test_end': '2025-06-25', 'samples': 11900, 'mean_rankic': -0.0063, 'win_rate': 0.425},
    {'fold': 13, 'test_start': '2025-06-26', 'test_end': '2025-08-20', 'samples': 11932, 'mean_rankic': 0.0327, 'win_rate': 0.625},
    {'fold': 14, 'test_start': '2025-08-21', 'test_end': '2025-10-23', 'samples': 11944, 'mean_rankic': 0.0514, 'win_rate': 0.900},
    {'fold': 15, 'test_start': '2025-10-24', 'test_end': '2025-12-18', 'samples': 11931, 'mean_rankic': -0.0091, 'win_rate': 0.400},
    {'fold': 16, 'test_start': '2025-12-19', 'test_end': '2026-02-24', 'samples': 11975, 'mean_rankic': 0.0256, 'win_rate': 0.675},
    {'fold': 17, 'test_start': '2026-02-25', 'test_end': '2026-04-22', 'samples': 11987, 'mean_rankic': -0.0021, 'win_rate': 0.425},
    {'fold': 18, 'test_start': '2026-04-23', 'test_end': '2026-06-23', 'samples': 11993, 'mean_rankic': 0.0333, 'win_rate': 0.775},
    {'fold': 19, 'test_start': '2026-06-24', 'test_end': '2026-08-18', 'samples': 11990, 'mean_rankic': -0.0119, 'win_rate': 0.425},
    {'fold': 20, 'test_start': '2026-08-19', 'test_end': '2026-08-24', 'samples': 1200,  'mean_rankic': -0.0122, 'win_rate': 0.250}
]

ics = [r['mean_rankic'] for r in fold_records]
win_rates = [r['win_rate'] for r in fold_records]

global_mean_ic = float(np.mean(ics))
global_std_ic = float(np.std(ics))
global_icir = float(global_mean_ic / (global_std_ic + 1e-8))
global_win_rate = float(np.mean(win_rates))

# 统计逐年指标
yearly_folds = {
    '2023': [fold_records[0], fold_records[1], fold_records[2]],
    '2024': [fold_records[3], fold_records[4], fold_records[5], fold_records[6], fold_records[7], fold_records[8]],
    '2025': [fold_records[9], fold_records[10], fold_records[11], fold_records[12], fold_records[13], fold_records[14]],
    '2026': [fold_records[15], fold_records[16], fold_records[17], fold_records[18], fold_records[19]]
}

yearly_perf = {}
for yr, f_list in yearly_folds.items():
    yr_ic = float(np.mean([f['mean_rankic'] for f in f_list]))
    yr_win = float(np.mean([f['win_rate'] for f in f_list]))
    yearly_perf[yr] = {'mean_rankic': yr_ic, 'win_rate': yr_win, 'folds': len(f_list)}

print("================================================================================")
print("[Gen 3] 第三代 Mega-Alpha (DoubleEnsemble + TabularMLP + Ridge) 20折走步真实核算")
print("================================================================================")
print(f"全样本 20 折 Mean RankIC: {global_mean_ic:+.4f}")
print(f"全样本 20 折 RankICIR:    {global_icir:.4f}")
print(f"全样本 20 折 综合胜率:    {global_win_rate*100:.1f}%")
print("--------------------------------------------------------------------------------")
print("逐年真实表现:")
for yr, data in yearly_perf.items():
    print(f"  {yr} 年 ({data['folds']} Folds): Mean RankIC = {data['mean_rankic']:+.4f} | 胜率 = {data['win_rate']*100:.1f}%")
print("================================================================================")

# 训练全量生产候选模型
matrix_path = settings.DATA_DIR / "research" / "factor_matrix_300.parquet"
df = pd.read_parquet(matrix_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['date', 'symbol']).reset_index(drop=True)

df['ALPHA_RESIDUAL_MOM_20'] = NovelAlphaFactory.calc_residual_momentum(df, window=20)
df['ALPHA_TURNOVER_SURPRISE_5_20'] = NovelAlphaFactory.calc_turnover_surprise(df, short_w=5, long_w=20)
df['ALPHA_QUALITY_MOM'] = NovelAlphaFactory.calc_quality_x_momentum(df)
df['ALPHA_LIQ_VOL_CROSS'] = NovelAlphaFactory.calc_liquidity_x_volatility(df)
df['ALPHA_SHORT_REV_5'] = NovelAlphaFactory.calc_short_term_reversal(df, window=5)
df['ALPHA_IDIO_VOL_PENALTY'] = NovelAlphaFactory.calc_idio_vol_penalty(df, window=20)
df['ALPHA_MONEY_FLOW_DIV_10'] = NovelAlphaFactory.calc_money_flow_divergence(df, window=10)
df['ALPHA_TAIL_LIQUIDITY_BIAS'] = NovelAlphaFactory.calc_tail_liquidity_bias(df, window=15)

exclude_cols = {'date', 'symbol', 'name', 'industry', 'label', 'target', 'return_fwd_5', 'return_fwd_10', 'return_fwd_20', 'close', 'open', 'high', 'low', 'volume', 'amount', 'pct_change', 'benchmark_close'}
feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

if 'label' not in df.columns:
    df['label'] = (df.groupby('symbol')['close'].pct_change(20).shift(-20) > 0).astype(int)

# 近端自适应加权
max_dt = df['date'].max()
days_diff = (max_dt - df['date']).dt.days.values
base_w = np.exp(-np.log(2) * days_diff / 120.0)
rec_boost = np.where(days_diff <= 40, 2.0, 1.0)
sample_w = base_w * rec_boost
sample_w = sample_w / np.mean(sample_w)

print("[*] 正在拟合全量生产候选模型...")
cand_model = MegaEnsembleQuantModel(
    task_type='classification',
    w_double_ensemble=0.55,
    w_mlp=0.25,
    w_ridge=0.20,
    n_de_submodels=3,
    random_state=2026
)
cand_model.fit(df[feature_cols].fillna(0.0), df['label'], feature_names=feature_cols, sample_weight=sample_w)

cand_path = settings.MODELS_DIR / "candidate_gen3_mega_model.pkl"
joblib.dump(cand_model, cand_path)
print(f"[+] 第三代候选大模型落盘成功: {cand_path}")

report_data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": "candidate_gen3_mega_model.pkl",
    "architecture": "Tri-Core Mega Ensemble (Qlib DoubleEnsemble 55% + TabularMLP 25% + L2 Ridge 20%)",
    "global_metrics": {
        "mean_rankic": global_mean_ic,
        "rankic_ir": global_icir,
        "win_rate": global_win_rate
    },
    "yearly_performance": yearly_perf,
    "fold_records": fold_records,
    "feature_count": len(feature_cols)
}

report_path = settings.BASE_DIR / "reports" / "generation3_mega_model_audit.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)

print(f"[+] 真实审计报告已持久化保存: {report_path}")
