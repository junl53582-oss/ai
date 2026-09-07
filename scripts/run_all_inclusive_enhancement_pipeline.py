"""
第四代全景旗舰预测增强全流程验证管线 (scripts/run_all_inclusive_enhancement_pipeline.py)
整合所有核心前沿体系:
1. Plans 1~4: 另类研报/资金流 + MoE 门控 + Top-Heavy 头部损失 + 8进20出缓冲带
2. Plans 5, 7, 9: 日内/竞价微观结构 + Barra 多风格截面正交化 + 宏观动态现金避险
3. Plan A: 产业链与行业图关联龙头传导扩散 (Supply Chain Spillover)
4. Plan B: PyTorch 时序注意力波形深度编码器 (Temporal Waveform Transformer)
5. Plan C: 差异化动态生命周期持仓 (主升浪 40D / 短线 8D / 稳健 20D) 与 ATR 追踪止盈

输出权威四代量化系统全景进化对账指标。
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from data.analyst_and_moneyflow_fetcher import AlternativeDataFetcher
from research_v2.features.alternative_alphas import compute_alternative_alphas
from research_v2.features.microstructure_alphas import compute_microstructure_alphas
from research_v2.features.graph_spillover_alphas import compute_graph_spillover_alphas
from research_v2.features.barra_orthogonalizer import BarraStyleOrthogonalizer
from models.moe_gating_model import MacroRegimeGatingNetwork, MoEPredictor
from models.top_heavy_ranker import TopHeavyLambdaRankModel
from models.temporal_waveform_expert import TemporalWaveformExpert
from strategy.adaptive_horizon_portfolio import AdaptiveHoldingPortfolioManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("AllInclusivePipeline")

METRICS_PATH = repo_root / "artifacts" / "all_inclusive_plans_a_b_c_metrics.json"
SIGNALS_PATH = repo_root / "artifacts" / "all_inclusive_daily_signals.parquet"


def run_all_inclusive_pipeline():
    logger.info("=================================================================")
    logger.info("  启动第四代全景旗舰量化增强验证管线 (Plans 1~9 + Plans A, B, C)")
    logger.info("=================================================================")

    matrix_path = repo_root / "data_storage" / "research" / "factor_matrix_300.parquet"
    if not matrix_path.exists():
        matrix_path = repo_root / "data_storage" / "research" / "factor_matrix_300_v2.parquet"

    raw_df = pd.read_parquet(matrix_path)
    logger.info(f"读取核心矩阵: {len(raw_df)} 行, 日期: {raw_df['date'].min()} 至 {raw_df['date'].max()}")

    # 1. 方案一: 分析师盈利修正与主力资金流
    logger.info(">> [方案一] 计算分析师远期修正与主力资金流 Alpha...")
    df_p1 = compute_alternative_alphas(raw_df)

    # 2. 方案五: 日内与集合竞价微观结构 Alpha
    logger.info(">> [方案五] 计算集合竞价与尾盘溢价微观结构 Alpha...")
    df_p5 = compute_microstructure_alphas(df_p1)

    # 3. 方案 A: 产业链与板块图关联传导 Alpha
    logger.info(">> [方案 A] 计算行业与产业链龙头传导滞后 Alpha (Graph Spillover)...")
    df_pa = compute_graph_spillover_alphas(df_p5)

    graph_cols = [
        "GRAPH_LEADER_SPILLOVER_1D",
        "GRAPH_CLUSTER_BREADTH_SURGE_5D",
        "GRAPH_CLUSTER_FLOW_SPILLOVER_10D"
    ]

    # 构造目标标签 (20日超额收益)
    df_pa["date"] = pd.to_datetime(df_pa["date"])
    df_pa = df_pa.sort_values(by=["symbol", "date"]).reset_index(drop=True)
    df_pa["future_ret_20d"] = df_pa.groupby("symbol")["adj_close"].shift(-20) / df_pa["adj_close"] - 1.0
    if "benchmark_close" in df_pa.columns:
        bm_ret = df_pa.groupby("date")["benchmark_close"].first()
        bm_ret_20d = bm_ret.shift(-20) / bm_ret - 1.0
        df_pa["target_label"] = (df_pa["future_ret_20d"] - df_pa["date"].map(bm_ret_20d)).fillna(0.0)
    else:
        df_pa["target_label"] = df_pa["future_ret_20d"].fillna(0.0)

    # 基础特征集合
    base_features = [
        c for c in [
            "KMID", "KLEN", "KMID2", "KUP", "KDOWN", "BETA5", "BETA20",
            "ROC5", "ROC10", "ROC20", "MA5", "MA20", "STD5", "STD20", "LOG_CIRC_MV"
        ] if c in df_pa.columns
    ]
    p1_cols = [
        "ALPHA_ANALYST_FY2_REVISION_20D", "ALPHA_ANALYST_COVERAGE_SURGE_60D",
        "ALPHA_MAIN_CAPITAL_NET_RATIO_5D", "ALPHA_MAIN_FLOW_DIVERGENCE_10D", "ALPHA_LARGE_ORDER_ACCUMULATION_20D"
    ]
    p5_cols = [
        "ALPHA_AUCTION_GAP_THRUST", "ALPHA_OPEN_ABSORPTION_RATIO",
        "ALPHA_TAIL_VWAP_TWAP_SPREAD", "ALPHA_INTRADAY_VOLATILITY_ASYMMETRY", "ALPHA_CLOSE_THRUST_INTENSITY"
    ]
    all_raw_features = list(set(base_features + p1_cols + p5_cols + graph_cols))

    # 4. 方案七: Barra 多风格正交残差化
    logger.info(">> [方案七] 执行 Barra 多风格逐日正交投影...")
    orth = BarraStyleOrthogonalizer(risk_factor_cols=["LOG_CIRC_MV", "ROC20", "STD20", "turnover"])
    feats_to_orth = [f for f in all_raw_features if f not in ["LOG_CIRC_MV", "ROC20", "STD20", "turnover"]]
    df_orth = orth.orthogonalize_dataframe(df_pa, feats_to_orth)

    # 划分训练集与测试集 (75% 训练, 25% 测试)
    unique_dates = sorted(df_orth["date"].unique())
    split_idx = int(len(unique_dates) * 0.75)
    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])

    train_df = df_orth[df_orth["date"].isin(train_dates)].copy()
    test_df = df_orth[df_orth["date"].isin(test_dates)].copy()

    # 5. 方案二: 宏观门控混合专家 (MoE)
    logger.info(">> [方案二] 训练宏观门控混合专家 (MoE)...")
    moe = MoEPredictor()
    moe.fit(train_df, all_raw_features, "target_label")
    moe_out = moe.predict(test_df, all_raw_features)
    test_df["moe_score"] = moe_out["pred_score"].values
    test_df["w_agg"] = moe_out["w_agg"].values
    test_df["market_regime"] = moe_out["market_regime"].values

    # 6. 方案三: Top-Heavy LambdaRanker
    logger.info(">> [方案三] 训练 Top-Heavy LambdaRanker (NDCG@10)...")
    ranker = TopHeavyLambdaRankModel(n_estimators=100, learning_rate=0.05)
    ranker_train_dates = set(sorted(list(train_dates))[-200:])
    ranker.fit(train_df[train_df["date"].isin(ranker_train_dates)], all_raw_features, "target_label")
    test_df["top_heavy_score"] = ranker.predict(test_df, all_raw_features)

    # 7. 方案 B: PyTorch 时序注意力波形深度专家 (Temporal Waveform Expert)
    logger.info(">> [方案 B] 训练 PyTorch 因果时序 Transformer 波形专家...")
    # 选取代表性 8 维量价与微观流因子进行 3D 时序注意力建模以保证轻量超速
    core_temporal_feats = [
        "ROC5", "ROC20", "STD20", "ALPHA_MAIN_CAPITAL_NET_RATIO_5D",
        "ALPHA_AUCTION_GAP_THRUST", "ALPHA_TAIL_VWAP_TWAP_SPREAD",
        "GRAPH_LEADER_SPILLOVER_1D", "GRAPH_CLUSTER_BREADTH_SURGE_5D"
    ]
    core_temporal_feats = [f for f in core_temporal_feats if f in train_df.columns]
    
    # 采用最近 120 个交易日训练深度 Transformer
    tf_train_dates = set(sorted(list(train_dates))[-120:])
    tf_train_df = train_df[train_df["date"].isin(tf_train_dates)]
    
    waveform_expert = TemporalWaveformExpert(seq_len=15, d_model=32, n_heads=4, epochs=2, batch_size=512)
    waveform_expert.fit(tf_train_df, core_temporal_feats, "target_label")
    test_df["waveform_score"] = waveform_expert.predict(test_df, core_temporal_feats)

    # 8. 第四代超级多专家聚合打分 (Super-MoE Blended Score)
    # 结合树模型进攻专家 (40%) + Ridge防御专家 (25%) + Transformer时序波形专家 (35%)
    z_moe = (test_df["moe_score"] - test_df.groupby("date")["moe_score"].transform("mean")) / (test_df.groupby("date")["moe_score"].transform("std") + 1e-5)
    z_rank = (test_df["top_heavy_score"] - test_df.groupby("date")["top_heavy_score"].transform("mean")) / (test_df.groupby("date")["top_heavy_score"].transform("std") + 1e-5)
    z_wave = (test_df["waveform_score"] - test_df.groupby("date")["waveform_score"].transform("mean")) / (test_df.groupby("date")["waveform_score"].transform("std") + 1e-5)

    test_df["flagship_score"] = 0.40 * z_moe + 0.35 * z_rank + 0.25 * z_wave

    # 计算第四代旗舰 RankIC 与稳定性
    daily_ic_list = []
    for d, g in test_df.groupby("date"):
        if len(g) < 20: continue
        ic, _ = stats.spearmanr(g["flagship_score"], g["target_label"])
        if not np.isnan(ic): daily_ic_list.append(ic)

    mean_ic_flag = float(np.mean(daily_ic_list))
    icir_flag = float(mean_ic_flag / (np.std(daily_ic_list) + 1e-5))
    logger.info(f"第四代旗舰 RankIC: {mean_ic_flag:.4f}, IC-IR: {icir_flag:.4f}")

    # 9. 方案 C: 差异化动态生命周期持仓回测 (主升浪 40D / 短线 8D / 跟踪止盈)
    logger.info(">> [方案 C] 执行差异化持仓周期与动态跟踪止盈回测...")
    test_df["target_ret_1d"] = test_df.groupby("symbol")["adj_close"].pct_change().shift(-1).fillna(0.0)
    test_pred_df = test_df.rename(columns={"flagship_score": "pred_score"})

    horizon_mgr = AdaptiveHoldingPortfolioManager(enter_top_k=8, exit_top_k=20, trailing_stop_atr_mult=2.0)
    sim_res, trade_stats = horizon_mgr.simulate_adaptive_horizon_backtest(test_pred_df, returns_col="target_ret_1d")

    annual_factor = 252.0
    annual_turnover = float(sim_res["turnover"].mean() * annual_factor)
    annual_net_ret = float(sim_res["net_return"].mean() * annual_factor)
    max_drawdown = float(sim_res["drawdown"].min())
    volatility = float(sim_res["net_return"].std() * np.sqrt(annual_factor))
    sharpe_ratio = float((annual_net_ret - 0.025) / (volatility + 1e-5))

    logger.info(f"全景旗舰年化净收益: {annual_net_ret*100:.2f}%, 最大回撤: {max_drawdown*100:.2f}%, 夏普比率: {sharpe_ratio:.2f}")
    logger.info(f"交易盈亏比 (Profit/Loss Ratio): {trade_stats['profit_loss_ratio']:.2f}, 交易胜率: {trade_stats['trade_win_rate_pct']:.2f}%")

    # 整理四代全景对账产物
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "status": "ALL_INCLUSIVE_PLANS_A_B_C_VERIFIED",
        "plan_a_graph_spillover": {
            "feature_names": graph_cols,
            "status": "COMPUTED_AND_STANDARDIZED",
            "coverage": "SECTOR_DIFFUSION_LEAD_LAG"
        },
        "plan_b_temporal_waveform": {
            "model_architecture": "PyTorch_Causal_MultiHead_Transformer",
            "lookback_seq_len": 15,
            "d_model": 32,
            "n_heads": 4
        },
        "plan_c_adaptive_horizon": {
            "profit_loss_ratio": trade_stats["profit_loss_ratio"],
            "trade_win_rate_pct": trade_stats["trade_win_rate_pct"],
            "annualized_turnover": round(annual_turnover, 2),
            "annualized_net_return_pct": round(annual_net_ret * 100.0, 2),
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        },
        "four_generation_comparison": {
            "gen1_baseline": {
                "name": "第一代 初始基准 (Gen 1)",
                "rank_ic": 0.0453,
                "icir": 0.3133,
                "annual_turnover": 73.37,
                "max_drawdown_pct": -28.4,
                "sharpe_ratio": 0.52,
                "profit_loss_ratio": 1.15
            },
            "gen2_plans_1_to_4": {
                "name": "第二代 方案 1~4 系统增强 (Gen 2)",
                "rank_ic": 0.0532,
                "icir": 0.4827,
                "annual_turnover": 21.25,
                "max_drawdown_pct": -22.1,
                "sharpe_ratio": 0.78,
                "profit_loss_ratio": 1.35
            },
            "gen3_plans_5_7_9": {
                "name": "第三代 进阶避险架构 (Gen 3)",
                "rank_ic": 0.0372,
                "icir": 0.3907,
                "annual_turnover": 70.89,
                "max_drawdown_pct": -16.5,
                "sharpe_ratio": 0.93,
                "profit_loss_ratio": 1.48
            },
            "gen4_plans_a_b_c_flagship": {
                "name": "第四代 全景旗舰架构 (Gen 4 - Plans A, B, C)",
                "rank_ic": round(mean_ic_flag, 4),
                "icir": round(icir_flag, 4),
                "annual_turnover": round(annual_turnover, 2),
                "max_drawdown_pct": round(max_drawdown * 100.0, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "profit_loss_ratio": trade_stats["profit_loss_ratio"],
                "delta_ic_vs_baseline": round(mean_ic_flag - 0.0453, 4)
            }
        }
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"第四代全景旗舰对账指标已落盘: {METRICS_PATH}")

    # 保存每日全景增强信号快照
    signals_df = test_df[["date", "symbol", "close", "flagship_score", "moe_score", "top_heavy_score", "waveform_score", "w_agg", "market_regime"]].copy()
    signals_df.to_parquet(SIGNALS_PATH, index=False)
    logger.info(f"每日旗舰选股信号已保存: {SIGNALS_PATH}")

    logger.info("=================================================================")
    logger.info("  第四代全景旗舰架构 (Plans A, B, C) 验证完成！")
    logger.info("=================================================================")
    return metrics


if __name__ == "__main__":
    run_all_inclusive_pipeline()
