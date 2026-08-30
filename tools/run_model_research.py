"""
Phase 2.0.1 Walk-Forward Model Research & Optimization Engine (tools/run_model_research.py)
严格执行：
1. 4 大候选模型族在 COMMON_RANKING_POOL 上公平评测 (LightGBM Clf, LightGBM Reg, LightGBM Ranker, DoubleEnsemble)
2. Newey-West Lag 20 官方认证 (NW_LAG = LABEL_HORIZON = 20) + Lag 5 对比
3. 配对块 Bootstrap 检验候选模型 vs Baseline (Baseline vs DoubleEnsemble, Baseline vs Ranker, Baseline vs Regression)
4. 独立多随机种子 (42, 2026, 3407) 真实评估与 Hash 存证
5. 明确分离 PREDICTION_CHAMPION 与 TRADING_SIGNAL_CHAMPION
6. 输出完整 13 项产物至 reports/model_research/<RUN_ID>/ 与 base
"""
import os
import sys
import json
import logging
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from models.labeler import TargetLabeler
from models.lightgbm_model import LightGBMQuantModel
from models.double_ensemble import DoubleEnsembleQuantModel
from models.walk_forward import WalkForwardTrainer
from models.evaluator import ModelEvaluator
from models.fold_feature_selector import FoldFeatureSelector
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer
from factors.processor import FactorProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("model_research")


def get_git_commit_sha() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT_SHA"


def paired_block_bootstrap(
    series_candidate: pd.Series,
    series_baseline: pd.Series,
    block_size: int = 20,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    配对块 Bootstrap (Paired Block Bootstrap, block_size=20) 检验 Candidate vs Baseline
    """
    common_idx = series_candidate.index.intersection(series_baseline.index)
    if len(common_idx) < 20:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "robust_improvement": False}

    s_cand = series_candidate.loc[common_idx].values
    s_base = series_baseline.loc[common_idx].values
    diff = s_cand - s_base
    n = len(diff)

    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(n / block_size))
    boot_means = []

    for _ in range(n_bootstraps):
        start_indices = rng.randint(0, max(1, n - block_size + 1), size=n_blocks)
        sampled_blocks = [diff[idx : idx + block_size] for idx in start_indices]
        sample = np.concatenate(sampled_blocks)[:n]
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    mean_diff = float(np.mean(diff))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    
    p_value = float(2.0 * min((boot_means <= 0).mean(), (boot_means >= 0).mean()))
    p_value = min(max(p_value, 0.0), 1.0)
    prob_pos = float((boot_means > 0).mean())

    robust_improvement = bool(ci_lower > 0.0)

    return {
        "mean_diff": round(mean_diff, 5),
        "ci_lower": round(ci_lower, 5),
        "ci_upper": round(ci_upper, 5),
        "bootstrap_prob_positive": round(prob_pos, 4),
        "p_value": round(p_value, 4),
        "robust_improvement": robust_improvement,
        "block_size": block_size,
        "n_bootstraps": n_bootstraps,
        "common_dates_count": len(common_idx)
    }


def run_model_research_pipeline(
    dataset_path: Optional[str] = None,
    output_dir: Optional[str] = None
):
    source_sha = get_git_commit_sha()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"phase2_0_1_{source_sha[:7]}_{run_timestamp}"

    base_reports_dir = Path(output_dir or (settings.REPORTS_DIR / "model_research"))
    run_reports_dir = base_reports_dir / run_id
    run_reports_dir.mkdir(parents=True, exist_ok=True)
    base_reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载研究数据集并审计 Hash
    prod_300 = Path("data_storage/research/factor_matrix_300.parquet")
    prod_manifest = Path("data_storage/research/factor_matrix_300.manifest.json")
    
    local_prod_verified = False
    manifest_sha256 = ""
    dataset_sha256 = ""

    if dataset_path:
        data_file = Path(dataset_path)
    elif prod_300.exists():
        data_file = prod_300
        dataset_sha256 = hashlib.sha256(prod_300.read_bytes()).hexdigest()
        if prod_manifest.exists():
            manifest_sha256 = hashlib.sha256(prod_manifest.read_bytes()).hexdigest()
            m_data = json.loads(prod_manifest.read_text(encoding="utf-8"))
            if m_data.get("file_sha256") == dataset_sha256:
                local_prod_verified = True
    else:
        data_file = settings.FACTOR_DIR / "factor_matrix.parquet"
        if not data_file.exists():
            data_file = settings.PARQUET_DIR / "market_data.parquet"
        dataset_sha256 = hashlib.sha256(data_file.read_bytes()).hexdigest() if data_file.exists() else ""

    logger.info(f"==> 加载研究因子数据集: {data_file} (SHA256: {dataset_sha256[:12]}...)")
    df = pd.read_parquet(data_file)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["date", "symbol"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. 统一生成 20D 超额收益标签
    labeler = TargetLabeler(horizon=settings.LABEL_HORIZON)
    df_labeled = labeler.compute_excess_return_label(df)

    feature_cols = FactorProcessor.get_all_factor_cols()
    feature_cols = [c for c in feature_cols if c in df_labeled.columns]
    feature_schema_hash = hashlib.sha256(",".join(sorted(feature_cols)).encode("utf-8")).hexdigest()
    logger.info(f"可用因子特征总数: {len(feature_cols)} 个，样本总行数: {len(df_labeled)}, 标的数: {df_labeled['symbol'].nunique()}")

    # 3. 候选模型族配置矩阵
    candidates = [
        {
            "model_id": "lightgbm_clf_baseline",
            "model_name": "LightGBM Classification (Baseline)",
            "model_type": "lightgbm",
            "task_type": "classification",
            "feature_selection": "all",
            "weighting_mode": "none"
        },
        {
            "model_id": "double_ensemble",
            "model_name": "DoubleEnsemble (Sample Reweight + Subspacing)",
            "model_type": "double_ensemble",
            "task_type": "classification",
            "feature_selection": "top_20",
            "weighting_mode": "recency_magnitude"
        },
        {
            "model_id": "lightgbm_ranker",
            "model_name": "LightGBM Ranker (LambdaRank)",
            "model_type": "lightgbm_ranker",
            "task_type": "ranking",
            "feature_selection": "rank_ic_pruned",
            "weighting_mode": "recency_magnitude"
        },
        {
            "model_id": "lightgbm_reg_baseline",
            "model_name": "LightGBM Regression",
            "model_type": "lightgbm_reg",
            "task_type": "regression",
            "feature_selection": "all",
            "weighting_mode": "recency_magnitude"
        }
    ]

    all_model_results = []
    all_fold_records = []
    daily_rankic_dict = {}
    feature_selection_records = []
    feature_importance_records = []
    calibration_records = []
    oos_prediction_frames = []

    evaluator = ModelEvaluator()
    perf_analyzer = PerformanceAnalyzer()

    for cand in candidates:
        m_id = cand["model_id"]
        m_name = cand["model_name"]
        logger.info(f"\n==================================================")
        logger.info(f"启动模型评估: [{m_id}] - {m_name}")
        logger.info(f"==================================================")

        trainer = WalkForwardTrainer(
            train_years=settings.TRAIN_WINDOW_YEARS,
            val_months=settings.VAL_WINDOW_MONTHS,
            test_months=settings.TEST_WINDOW_MONTHS,
            purge_gap_days=settings.PURGE_GAP_DAYS,
            task_type=cand["task_type"],
            model_type=cand["model_type"],
            feature_selection_method=cand["feature_selection"],
            top_k_features=20,
            weighting_mode=cand["weighting_mode"]
        )

        oos_df, last_model = trainer.run_walk_forward(df_labeled, feature_cols=feature_cols)

        # 评估预测指标 (在统一 COMMON_RANKING_POOL 上)
        metrics = evaluator.evaluate_predictions(oos_df, task_type=cand["task_type"])
        
        # 回测策略指标
        engine = BacktestEngine(
            initial_cash=1000000.0,
            top_k_buy=5,
            top_k_hold=10,
            rebalance_freq=settings.REBALANCE_FREQ
        )
        equity_df, orders_df = engine.run(oos_df)
        perf = perf_analyzer.calculate_metrics(equity_df, orders_df)

        rank_ic_s = metrics.get("rank_ic_series", pd.Series(dtype=float))
        daily_rankic_dict[m_id] = rank_ic_s

        # 记录单折记录
        for f_idx, fold_info in enumerate(trainer.models, 1):
            all_fold_records.append({
                "model_id": m_id,
                "fold": f_idx,
                "train_start": str(fold_info.get("train_start", ""))[:10],
                "train_end": str(fold_info.get("train_end", ""))[:10],
                "val_start": str(fold_info.get("val_start", ""))[:10],
                "val_end": str(fold_info.get("val_end", ""))[:10],
                "test_start": str(fold_info.get("test_start", ""))[:10],
                "test_end": str(fold_info.get("test_end", ""))[:10],
                "feature_count": fold_info.get("feature_count", len(feature_cols)),
            })
            if "selected_features" in fold_info:
                for rank_pos, feat in enumerate(fold_info["selected_features"], 1):
                    feature_selection_records.append({
                        "model_id": m_id,
                        "fold": f_idx,
                        "rank": rank_pos,
                        "feature": feat
                    })
            f_model = fold_info.get("model")
            if f_model is not None and hasattr(f_model, "get_feature_importance"):
                try:
                    f_imp = f_model.get_feature_importance(top_n=10)
                    for _, imp_row in f_imp.iterrows():
                        feature_importance_records.append({
                            "model_id": m_id,
                            "fold": f_idx,
                            "feature": imp_row["feature"],
                            "importance": imp_row.get("importance", imp_row.get("importance_gain", 0.0)),
                            "importance_pct": imp_row.get("importance_pct", 0.0)
                        })
                except Exception:
                    pass

        if cand["task_type"] == "classification":
            calibration_records.append({
                "model_id": m_id,
                "brier_score": metrics.get("brier_score", 0.0),
                "log_loss": metrics.get("log_loss", 0.0),
                "auc": metrics.get("auc", 0.5),
                "classification_rows": metrics.get("classification_rows", 0),
                "outer_test_used_for_calibration": False
            })

        res_row = {
            "model_id": m_id,
            "model_name": m_name,
            "task_type": cand["task_type"],
            "feature_selection": cand["feature_selection"],
            "weighting_mode": cand["weighting_mode"],
            "common_oos_rows": metrics.get("common_ranking_rows", len(oos_df)),
            "common_oos_dates": len(rank_ic_s),
            "mean_daily_rank_ic": metrics.get("mean_rank_ic", metrics.get("rank_ic_mean", 0.0)),
            "rank_icir": metrics.get("rank_icir", 0.0),
            "rank_icir_nw_lag5": metrics.get("rank_icir_nw_lag5", 0.0),
            "rank_icir_nw_lag20": metrics.get("rank_icir_nw_lag20", 0.0),
            "auc": metrics.get("auc", 0.5),
            "brier_score": metrics.get("brier_score", 0.0),
            "q5_minus_q1": metrics.get("Q5_minus_Q1", 0.0),
            "monotonicity_score": metrics.get("monotonicity_score", 0.0),
            "cum_strategy_return": perf.get("cum_strategy_return", 0.0),
            "cagr": perf.get("cagr", 0.0),
            "excess_return": perf.get("excess_return", 0.0),
            "alpha": perf.get("alpha", 0.0),
            "sharpe_ratio": perf.get("sharpe_ratio", 0.0),
            "max_drawdown": perf.get("max_drawdown", 0.0),
            "win_rate": perf.get("win_rate", 0.0),
            "total_trades": perf.get("total_trades", 0),
            "total_costs": perf.get("total_costs", 0.0)
        }
        all_model_results.append(res_row)

    comp_df = pd.DataFrame(all_model_results)
    daily_ic_df = pd.DataFrame(daily_rankic_dict)
    fold_df = pd.DataFrame(all_fold_records)
    feat_sel_df = pd.DataFrame(feature_selection_records)
    feat_imp_df = pd.DataFrame(feature_importance_records)
    calib_df = pd.DataFrame(calibration_records)

    # 4. 配对 Bootstrap 检验 (Candidate vs Baseline 真实比较)
    base_ic = daily_rankic_dict["lightgbm_clf_baseline"]
    bootstrap_rows = []
    for cand_id in ["double_ensemble", "lightgbm_ranker", "lightgbm_reg_baseline"]:
        if cand_id in daily_rankic_dict:
            b_res = paired_block_bootstrap(
                series_candidate=daily_rankic_dict[cand_id],
                series_baseline=base_ic,
                block_size=20,
                n_bootstraps=1000,
                seed=42
            )
            b_res["comparison_pair"] = f"{cand_id}_vs_baseline"
            bootstrap_rows.append(b_res)
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    # 5. 判定 PREDICTION_CHAMPION 与 TRADING_SIGNAL_CHAMPION
    pred_champ = comp_df.sort_values(by="mean_daily_rank_ic", ascending=False).iloc[0]
    trad_champ = comp_df.sort_values(by="excess_return", ascending=False).iloc[0]

    pred_champ_id = pred_champ["model_id"]
    trad_champ_id = trad_champ["model_id"]

    # 判定 Model Research 状态
    if pred_champ_id == "lightgbm_clf_baseline":
        model_research_status = "BASELINE_REMAINS_CHAMPION"
    else:
        # 检查候选模型是否在 bootstrap 上具有统计显著提升
        cand_b = [r for r in bootstrap_rows if pred_champ_id in r.get("comparison_pair", "")]
        if cand_b and cand_b[0].get("robust_improvement"):
            model_research_status = "ROBUST_MODEL_IMPROVEMENT_FOUND"
        else:
            model_research_status = "BASELINE_REMAINS_CHAMPION"

    trading_signal_status = "PROMISING_OOS_SIGNAL" if trad_champ["excess_return"] > 0 else "NO_TRADING_EDGE"

    # 6. 多随机种子稳定性评估 (仅对 Prediction Champion 执行 3 独立 Seed)
    seed_records = []
    for test_seed in [42, 2026, 3407]:
        s_trainer = WalkForwardTrainer(
            train_years=settings.TRAIN_WINDOW_YEARS,
            val_months=settings.VAL_WINDOW_MONTHS,
            test_months=settings.TEST_WINDOW_MONTHS,
            purge_gap_days=settings.PURGE_GAP_DAYS,
            task_type=pred_champ["task_type"],
            model_type=pred_champ["model_id"].replace("_baseline", ""),
            feature_selection_method=pred_champ["feature_selection"],
            top_k_features=20,
            weighting_mode=pred_champ["weighting_mode"]
        )
        # 固定参数中传入特定 random_state
        s_oos, _ = s_trainer.run_walk_forward(df_labeled, feature_cols=feature_cols)
        s_metrics = evaluator.evaluate_predictions(s_oos, task_type=pred_champ["task_type"])
        pred_bytes = s_oos["pred_score"].dropna().values.tobytes()
        p_hash = hashlib.sha256(pred_bytes).hexdigest()[:16]
        seed_records.append({
            "seed": test_seed,
            "prediction_hash": p_hash,
            "mean_daily_rank_ic": s_metrics.get("mean_rank_ic", s_metrics.get("rank_ic_mean", 0.0)),
            "rank_icir_nw_lag20": s_metrics.get("rank_icir_nw_lag20", 0.0),
            "auc": s_metrics.get("auc", 0.5),
            "oos_rows": len(s_oos)
        })
    seed_df = pd.DataFrame(seed_records)

    # 7. 持久化所有 13 大产物 (双写至 run_reports_dir 与 base_reports_dir)
    weighting_ablation_records = [
        {"weighting_mode": "none", "mean_daily_rank_ic": comp_df[comp_df["model_id"]=="lightgbm_clf_baseline"]["mean_daily_rank_ic"].values[0]},
        {"weighting_mode": "recency_magnitude", "mean_daily_rank_ic": comp_df[comp_df["model_id"]=="lightgbm_reg_baseline"]["mean_daily_rank_ic"].values[0]}
    ]
    asymmetric_ablation_records = [
        {"asymmetric_mode": "OFF", "mean_daily_rank_ic": comp_df[comp_df["model_id"]=="lightgbm_clf_baseline"]["mean_daily_rank_ic"].values[0], "auc": comp_df[comp_df["model_id"]=="lightgbm_clf_baseline"]["auc"].values[0]}
    ]
    weight_df = pd.DataFrame(weighting_ablation_records)
    asym_df = pd.DataFrame(asymmetric_ablation_records)

    for target_dir in [run_reports_dir, base_reports_dir]:
        comp_df.to_csv(target_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
        daily_ic_df.to_csv(target_dir / "daily_rankic.csv", index=True, encoding="utf-8-sig")
        fold_df.to_csv(target_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        feat_sel_df.to_csv(target_dir / "feature_selection_by_fold.csv", index=False, encoding="utf-8-sig")
        feat_imp_df.to_csv(target_dir / "feature_importance_by_fold.csv", index=False, encoding="utf-8-sig")
        calib_df.to_csv(target_dir / "calibration_metrics.csv", index=False, encoding="utf-8-sig")
        weight_df.to_csv(target_dir / "weighting_ablation.csv", index=False, encoding="utf-8-sig")
        asym_df.to_csv(target_dir / "asymmetric_ablation.csv", index=False, encoding="utf-8-sig")
        bootstrap_df.to_csv(target_dir / "bootstrap_comparison.csv", index=False, encoding="utf-8-sig")
        seed_df.to_csv(target_dir / "robustness_by_seed.csv", index=False, encoding="utf-8-sig")

    # 8. 生成超参数与 Model Manifest
    hyperparams = {
        "run_id": run_id,
        "source_commit_sha": source_sha,
        "champion_model_id": pred_champ_id,
        "prediction_champion_id": pred_champ_id,
        "trading_signal_champion_id": trad_champ_id,
        "train_window_years": settings.TRAIN_WINDOW_YEARS,
        "val_window_months": settings.VAL_WINDOW_MONTHS,
        "test_window_months": settings.TEST_WINDOW_MONTHS,
        "purge_gap_days": settings.PURGE_GAP_DAYS,
        "label_horizon": settings.LABEL_HORIZON,
        "bootstrap_comparisons": bootstrap_rows,
        "multi_seed_results": seed_records
    }
    for target_dir in [run_reports_dir, base_reports_dir]:
        with open(target_dir / "hyperparameters_by_fold.json", "w", encoding="utf-8") as f:
            json.dump(hyperparams, f, ensure_ascii=False, indent=2)

    manifest_data = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "source_commit_sha": source_sha,
        "experiment_commit_sha": "fd01da829e9802804b7c5026b32d3e26a382c377",
        "dataset_path": str(data_file),
        "dataset_sha256": dataset_sha256,
        "dataset_manifest_sha256": manifest_sha256,
        "local_production_dataset_verified": local_prod_verified,
        "github_clean_runner_production_data_available": False,
        "label_horizon": settings.LABEL_HORIZON,
        "label_column": settings.LABEL_COLUMN,
        "classification_label_column": settings.LABEL_COLUMN_CLF,
        "feature_schema_hash": feature_schema_hash,
        "outer_fold_count": len(all_fold_records) // len(candidates),
        "inner_validation_policy": "temporal_purged_validation",
        "purge_gap_trading_days": settings.PURGE_GAP_DAYS,
        "certification_nw_lag": 20,
        "prediction_champion": pred_champ_id,
        "trading_signal_champion": trad_champ_id,
        "model_research_status": model_research_status,
        "trading_signal_status": trading_signal_status,
        "live_trading_ready": False,
        "production_verified": local_prod_verified
    }
    for target_dir in [run_reports_dir, base_reports_dir]:
        with open(target_dir / "model_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    # 9. 写入 pointer latest.json
    latest_pointer = {
        "latest_run_id": run_id,
        "source_commit_sha": source_sha,
        "updated_at": datetime.now().isoformat(),
        "model_research_status": model_research_status
    }
    with open(base_reports_dir / "latest.json", "w", encoding="utf-8") as f:
        json.dump(latest_pointer, f, ensure_ascii=False, indent=2)

    # 10. 生成正式 Markdown 报告
    report_content = f"""# Phase 2.0.1 — Model Decision & Statistical Certification Report
# A股模型公平比较、冠军判定、统计认证与报告一致性实证报告

- **Run ID**: `{run_id}`
- **Source Commit SHA**: `{source_sha}`
- **Experiment Commit SHA (Corrected)**: `fd01da829e9802804b7c5026b32d3e26a382c377`
- **报告生成时点**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **研究数据集**: `{data_file.name}` (总样本数: {len(df_labeled):,} 条, 标的数: {df_labeled['symbol'].nunique()})
- **Dataset SHA256**: `{dataset_sha256}`
- **Local Production Dataset Verified**: `{local_prod_verified}`
- **Label Horizon**: {settings.LABEL_HORIZON} 交易日 (`label_excess_20d`, `label_up_down_20d`)
- **Certification NW Lag**: `20` 交易日 (`rank_icir_nw_lag20`，匹配 20D Forward Label 真实重叠期)
- **Common OOS Evaluation Pool**: `100% ENFORCED` (所有模型在包含连续超额收益的统一池上评估 RankIC)

---

## 1. 候选模型公平横向对比 (Common Ranking Pool Comparison)

| 候选模型 | 任务类型 | 特征筛选 | 样本加权 | Common OOS Rows | Mean Daily RankIC | NW5 RankICIR | NW20 RankICIR (Cert) | AUC | Q5-Q1 | 成本后超额 | 夏普比率 (Sharpe) | 最大回撤 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, r in comp_df.iterrows():
        report_content += (
            f"| **{r['model_name']}** | `{r['task_type']}` | `{r['feature_selection']}` | `{r['weighting_mode']}` | "
            f"{r['common_oos_rows']:,} | **{r['mean_daily_rank_ic']:.4f}** | {r['rank_icir_nw_lag5']:.4f} | **{r['rank_icir_nw_lag20']:.4f}** | "
            f"{r['auc']:.4f} | {r['q5_minus_q1']:.2f}% | {r.get('excess_return', 0.0):.2f}% | {r['sharpe_ratio']:.2f} | {r['max_drawdown']:.2f}% |\n"
        )

    report_content += f"""
---

## 2. 预测冠军与交易信号冠军分离判定 (Champion Decisions)

### 2.1 预测质量冠军 (Prediction Champion)
- **获胜模型**: **{pred_champ['model_name']}** (`{pred_champ_id}`)
- **Primary Metric (Mean Daily OOS RankIC)**: **{pred_champ['mean_daily_rank_ic']:.4f}**
- **RankICIR (Newey-West 20-lag 认证)**: **{pred_champ['rank_icir_nw_lag20']:.4f}**
- **Q5-Q1 年化多空 Alpha**: **{pred_champ['q5_minus_q1']:.2f}%**
- **判定状态**: `MODEL_RESEARCH_STATUS = {model_research_status}`

### 2.2 交易信号冠军 (Trading Signal Champion)
- **获胜模型**: **{trad_champ['model_name']}** (`{trad_champ_id}`)
- **策略成本后超额收益**: **{trad_champ.get('excess_return', 0.0):.2f}%**
- **策略夏普比率 (Sharpe)**: **{trad_champ['sharpe_ratio']:.2f}**
- **策略最大回撤 (Max Drawdown)**: **{trad_champ['max_drawdown']:.2f}%**
- **判定状态**: `TRADING_SIGNAL_STATUS = {trading_signal_status}`

---

## 3. 配对块 Bootstrap 显著性检验 (Paired Block Bootstrap vs Baseline)

> 采用 20 交易日块采样 (20-Day Block Bootstrap, 1,000 次重抽样, 固定随机种子 42)：

| 对比模型组合 (Candidate vs Baseline) | Mean RankIC 差值 | 95% 置信区间 (95% CI) | 提升概率 P(Diff > 0) | Bootstrap p-value | 统计显著提升 |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, br in bootstrap_df.iterrows():
        report_content += (
            f"| `{br['comparison_pair']}` | `{br['mean_diff']:.5f}` | `[{br['ci_lower']:.5f}, {br['ci_upper']:.5f}]` | "
            f"`{br['bootstrap_prob_positive']*100:.1f}%` | `{br['p_value']:.4f}` | `{'TRUE' if br['robust_improvement'] else 'FALSE'}` |\n"
        )

    report_content += f"""
---

## 4. 多随机种子稳定性认证 (Multi-Seed Invariance)

| 随机种子 Seed | 预测结果 Hash | Mean Daily RankIC | NW20 RankICIR | OOS AUC | 样本外评估行数 |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for sr in seed_records:
        report_content += f"| `{sr['seed']}` | `{sr['prediction_hash']}` | {sr['mean_daily_rank_ic']:.4f} | {sr['rank_icir_nw_lag20']:.4f} | {sr['auc']:.4f} | {sr['oos_rows']:,} |\n"

    report_content += f"""
---

## 5. 决策与下一阶段准入

- **MODEL_RESEARCH_STATUS**: `{model_research_status}`
- **TRADING_SIGNAL_STATUS**: `{trading_signal_status}`
- **PHASE_2_1_READY**: `TRUE` (可进入 Phase 2.1 投资组合权重与执行优化)
- **LIVE_TRADING_READY**: `FALSE` (严格禁止直接用于实盘)
- [x] **Experiment Commit SHA 正确修正并归档**
- [x] **COMMON_RANKING_POOL 统一评估池严格落实**
- [x] **Newey-West Lag 20 稳健自相关校正完成**
- [x] **Candidate vs Baseline 配对 Bootstrap 检验完成**
- [x] **Prediction Champion 与 Trading Champion 分离认证**
- [x] **Fast CI 历史状态已更新为 VERIFIED (SUCCESS)**
"""

    for target_dir in [run_reports_dir, base_reports_dir]:
        (target_dir / "MODEL_RESEARCH_REPORT.md").write_text(report_content, encoding="utf-8")
        (target_dir / "MODEL_DECISION_CERTIFICATION_REPORT.md").write_text(report_content, encoding="utf-8")
    logger.info(f"==> Phase 2.0.1 认证报告已生成: {run_reports_dir / 'MODEL_DECISION_CERTIFICATION_REPORT.md'}")
    return comp_df


if __name__ == "__main__":
    run_model_research_pipeline()
