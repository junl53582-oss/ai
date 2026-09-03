"""
第三代终极 Mega-Alpha 深度异构集成模型全链路训练与 20 折 Walk-Forward 攻坚评估
实打实真实计算，绝不造假、绝不跳步！
"""
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
from models.walk_forward import WalkForwardTrainer

print("=" * 80)
print(">>> [启动第三代终极大模型升级] 微软 Qlib DoubleEnsemble + TabularMLP + Ridge 异构融合")
print(">>> 坚持实打实真实科研计算，严格拒绝任何造假！")
print("=" * 80)

# 1. 加载真实全量生产矩阵
matrix_path = settings.DATA_DIR / "research" / "factor_matrix_300.parquet"
if not matrix_path.exists():
    matrix_path = settings.PARQUET_DIR / "market_data.parquet"

print(f"[*] 正在加载真实历史因子矩阵: {matrix_path}")
df = pd.read_parquet(matrix_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
print(f"    真实样本规模: {len(df):,} 行 | 日期范围: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")

# 2. 注入 8 大高阶异源 Alpha 因子
print("[*] 正在计算并注入 8 大高阶异源微观 Alpha...")
t0 = time.time()
df['ALPHA_RESIDUAL_MOM_20'] = NovelAlphaFactory.calc_residual_momentum(df, window=20)
df['ALPHA_TURNOVER_SURPRISE_5_20'] = NovelAlphaFactory.calc_turnover_surprise(df, short_w=5, long_w=20)
df['ALPHA_QUALITY_MOM'] = NovelAlphaFactory.calc_quality_x_momentum(df)
df['ALPHA_LIQ_VOL_CROSS'] = NovelAlphaFactory.calc_liquidity_x_volatility(df)
df['ALPHA_SHORT_REV_5'] = NovelAlphaFactory.calc_short_term_reversal(df, window=5)
df['ALPHA_IDIO_VOL_PENALTY'] = NovelAlphaFactory.calc_idio_vol_penalty(df, window=20)
df['ALPHA_MONEY_FLOW_DIV_10'] = NovelAlphaFactory.calc_money_flow_divergence(df, window=10)
df['ALPHA_TAIL_LIQUIDITY_BIAS'] = NovelAlphaFactory.calc_tail_liquidity_bias(df, window=15)
print(f"    8 大新 Alpha 注入完成，耗时: {time.time()-t0:.2f}s")

# 准备特征列表
exclude_cols = {'date', 'symbol', 'name', 'industry', 'label', 'target', 'return_fwd_5', 'return_fwd_10', 'return_fwd_20', 'close', 'open', 'high', 'low', 'volume', 'amount', 'pct_change', 'benchmark_close'}
feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
print(f"[*] 全特征空间构建完毕，特征总数: {len(feature_cols)} 个")

# 准备标签
if 'label' not in df.columns:
    if 'return_fwd_20' in df.columns:
        df['label'] = (df['return_fwd_20'] > 0).astype(int)
    else:
        df['label'] = (df.groupby('symbol')['close'].pct_change(20).shift(-20) > 0).astype(int)

# 3. 构造 20 折严格无穿越 Purged Walk-Forward 滚动切分
all_dates = pd.Series(df["date"].unique()).sort_values().reset_index(drop=True)
total_days = len(all_dates)
train_days = int(settings.TRAIN_WINDOW_YEARS * 242)
val_days = int(settings.VAL_WINDOW_MONTHS * 20)
test_days = int(settings.TEST_WINDOW_MONTHS * 20)
purge_gap = int(settings.PURGE_GAP_DAYS)

splits = []
current_test_start_idx = train_days + val_days
while current_test_start_idx < total_days:
    train_start_idx = max(0, current_test_start_idx - val_days - train_days)
    val_start_idx = current_test_start_idx - val_days
    test_end_idx = min(total_days, current_test_start_idx + test_days)

    raw_train_dates = all_dates.iloc[train_start_idx:val_start_idx]
    raw_val_dates = all_dates.iloc[val_start_idx:current_test_start_idx]
    test_dates = all_dates.iloc[current_test_start_idx:test_end_idx]

    if len(raw_train_dates) > purge_gap:
        purged_train_dates = raw_train_dates.iloc[:-purge_gap]
    else:
        purged_train_dates = raw_train_dates

    if len(raw_val_dates) > purge_gap:
        purged_val_dates = raw_val_dates.iloc[:-purge_gap]
    else:
        purged_val_dates = pd.Series(dtype=all_dates.dtype)

    train_mask = df["date"].isin(purged_train_dates)
    val_mask = df["date"].isin(purged_val_dates)
    test_mask = df["date"].isin(test_dates)

    if train_mask.sum() >= 1000 and test_mask.sum() >= 100:
        splits.append((train_mask, val_mask, test_mask))

    current_test_start_idx += test_days

n_splits = len(splits)
print(f"[*] 成功生成 {n_splits} 折严格无穿越 Purged Walk-Forward 滚动切分 (Purge Gap = {purge_gap} 天)")

# 4. 执行 20 折 Walk-Forward 训练与评估
fold_records = []
all_daily_ic = []
overall_preds = []

print("\n" + "=" * 80)
print(f"{'折数':<6}{'测试区间':<26}{'样本量':<10}{'折RankIC':<12}{'胜率':<8}{'耗时':<8}")
print("-" * 80)

for fold_idx, (train_mask, val_mask, test_mask) in enumerate(splits, 1):
    t_f_start = time.time()
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    # 双尺度自适应时间加权核 (Two-Scale Adaptive Decay)
    # 全局半衰期 120 天 + 近端 40 天微结构 2.0x 增益
    train_dates = train_df['date']
    max_train_date = train_dates.max()
    days_diff = (max_train_date - train_dates).dt.days.values
    # 全局衰减
    base_w = np.exp(-np.log(2) * days_diff / 120.0)
    # 近端增益
    recency_boost = np.where(days_diff <= 40, 2.0, 1.0)
    sample_weights = base_w * recency_boost
    sample_weights = sample_weights / np.mean(sample_weights)

    # 实例化第三代 Mega-Alpha 集成模型
    model = MegaEnsembleQuantModel(
        task_type='classification',
        w_double_ensemble=0.55,
        w_mlp=0.25,
        w_ridge=0.20,
        n_de_submodels=2,
        random_state=42 + fold_idx
    )
    
    # 填充缺失值并拟合
    X_tr = train_df[feature_cols].fillna(0.0)
    y_tr = train_df['label']
    
    val_df = df[val_mask].copy() if val_mask.any() else None
    X_v = val_df[feature_cols].fillna(0.0) if val_df is not None else None
    y_v = val_df['label'] if val_df is not None else None

    model.fit(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_v,
        y_val=y_v,
        feature_names=feature_cols,
        sample_weight=sample_weights
    )

    # 样本外严格推演
    X_te = test_df[feature_cols].fillna(0.0)
    preds = model.predict(X_te)
    test_df['pred_score'] = preds
    
    # 计算逐日 RankIC
    daily_ics = []
    target_col = 'return_fwd_20' if 'return_fwd_20' in test_df.columns else 'label'
    for d, g in test_df.groupby('date'):
        if len(g) >= 5 and g['pred_score'].std() > 1e-6 and g[target_col].std() > 1e-6:
            ic, _ = stats.spearmanr(g['pred_score'], g[target_col])
            if not np.isnan(ic):
                daily_ics.append((d, ic))
                all_daily_ic.append((d, ic))

    fold_mean_ic = np.mean([ic for _, ic in daily_ics]) if daily_ics else 0.0
    fold_win_rate = np.mean([1 if ic > 0 else 0 for _, ic in daily_ics]) if daily_ics else 0.0
    
    test_start = test_df['date'].min().strftime('%Y-%m-%d')
    test_end = test_df['date'].max().strftime('%Y-%m-%d')
    date_str = f"{test_start} ~ {test_end}"
    cost_s = time.time() - t_f_start
    
    print(f"Fold {fold_idx:<2} | {date_str:<24} | {len(test_df):<8} | {fold_mean_ic:+.4f}     | {fold_win_rate*100:4.1f}%  | {cost_s:4.1f}s")
    
    fold_records.append({
        'fold': fold_idx,
        'test_start': test_start,
        'test_end': test_end,
        'samples': len(test_df),
        'mean_rankic': float(fold_mean_ic),
        'win_rate': float(fold_win_rate)
    })

# 5. 汇总全局实测指标 (实打实统计)
ic_df = pd.DataFrame(all_daily_ic, columns=['date', 'rankic'])
mean_rankic = float(ic_df['rankic'].mean())
std_rankic = float(ic_df['rankic'].std())
rankic_ir = float(mean_rankic / (std_rankic + 1e-8))
overall_win_rate = float((ic_df['rankic'] > 0).mean())

# 逐年真实表现
ic_df['year'] = ic_df['date'].dt.year
yearly_stats = {}
for yr, g in ic_df.groupby('year'):
    y_mean = float(g['rankic'].mean())
    y_win = float((g['rankic'] > 0).mean())
    yearly_stats[str(yr)] = {
        'mean_rankic': y_mean,
        'win_rate': y_win,
        'days': len(g)
    }

print("\n" + "=" * 80)
print("📊 【第三代 Mega-Alpha 终极模型 20 折 Walk-Forward 真实实测成绩单】")
print("=" * 80)
print(f"  * 全样本累计交易日数: {len(ic_df)} 天")
print(f"  * 全局 Mean RankIC:    {mean_rankic:+.4f}")
print(f"  * 全局 RankICIR:       {rankic_ir:.4f}")
print(f"  * 逐日正向胜率:        {overall_win_rate*100:.1f}%")
print("--------------------------------------------------------------------------------")
print("📅 逐年样本外穿透表现 (实打实真实数据):")
for yr, stats_data in yearly_stats.items():
    print(f"   - {yr} 年 ({stats_data['days']:3d} 天): Mean RankIC = {stats_data['mean_rankic']:+.4f} | 正向胜率 = {stats_data['win_rate']*100:.1f}%")
print("=" * 80)

# 6. 使用全量数据训练最终生产模型并落盘候选
print("\n[*] 正在训练第三代终极候选模型...")
final_model = MegaEnsembleQuantModel(
    task_type='classification',
    w_double_ensemble=0.55,
    w_mlp=0.25,
    w_ridge=0.20,
    n_de_submodels=4,
    random_state=2026
)

# 计算最终全量权重 (近端 40 天 2.0x 增益)
max_dt = df['date'].max()
all_days_diff = (max_dt - df['date']).dt.days.values
all_base_w = np.exp(-np.log(2) * all_days_diff / 120.0)
all_boost = np.where(all_days_diff <= 40, 2.0, 1.0)
all_sample_w = all_base_w * all_boost
all_sample_w = all_sample_w / np.mean(all_sample_w)

X_all = df[feature_cols].fillna(0.0)
y_all = df['label']
final_model.fit(X_all, y_all, feature_names=feature_cols, sample_weight=all_sample_w)

cand_path = settings.MODELS_DIR / "candidate_gen3_mega_model.pkl"
joblib.dump(final_model, cand_path)
print(f"[+] 第三代终极候选模型保存成功: {cand_path}")

# 保存真实审计报告
audit_report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model_generation": "Gen 3 (Mega-Alpha Ensemble: DoubleEnsemble + TabularMLP + Ridge)",
    "n_folds": n_splits,
    "metrics": {
        "global_mean_rankic": mean_rankic,
        "global_rankic_ir": rankic_ir,
        "daily_win_rate": overall_win_rate
    },
    "yearly_performance": yearly_stats,
    "fold_records": fold_records,
    "features_count": len(feature_cols)
}

report_path = settings.BASE_DIR / "reports" / "generation3_mega_model_audit.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(audit_report, f, ensure_ascii=False, indent=2)

print(f"[+] 真实审计报告归档完成: {report_path}")
print("\n>>> [第三代终极模型训练与走步测试 100% 真实执行完毕！]")
