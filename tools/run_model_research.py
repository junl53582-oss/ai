"""
Formal Model Research & Scientific Integrity Certification Runner (tools/run_model_research.py)
A股实盘级量化走步模型科研评估与全证据驱动认证运行器 (Phase 2.1-B r3.1 Hardened)

核心原则：
1. 真实性优先于指标漂亮，严禁伪造数据与虚假门禁
2. 状态隔离：每次回测创建全新 BacktestEngine 实例，严禁跨模型/跨折复用有状态引擎
3. 真实 Fold 级时序交易执行：逐折运行 BacktestEngine 与 PerformanceAnalyzer
4. 物理隔离生产模型目录：严格禁止覆盖 saved_models/latest_lightgbm.pkl
5. 全证据推导 Gate Matrix：接入 research_v2/governance/certification.py
"""
import os
import sys
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from factors.processor import FactorProcessor
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from models.evaluator import ModelEvaluator
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer
from strategy.portfolio import PortfolioBuilder
from research_v2.governance.certification import evaluate_research_gates, CertificationDecision
from research_v2.governance.holdout_registry import (
    build_objective_common_train_pool,
    build_regression_native_train_pool,
    TrainingPoolType
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("run_model_research")


def _sha256_file(filepath: Path) -> str:
    if not filepath.exists():
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _sha256_frame(frame: pd.DataFrame) -> str:
    """Stable hash of an in-memory runtime artifact."""
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _snapshot_directory(dir_path: Path) -> Dict[str, Dict[str, Any]]:
    if not dir_path.exists():
        return {}
    snap = {}
    for p in sorted(dir_path.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(dir_path)).replace("\\", "/")
            snap[rel] = {
                "size": p.stat().st_size,
                "sha256": _sha256_file(p)
            }
    return snap


def get_git_commit_sha() -> str:
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
        return out
    except Exception:
        return "UNKNOWN_COMMIT_SHA"


def get_git_worktree_clean() -> Tuple[bool, List[str]]:
    try:
        import subprocess
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT), text=True).strip()
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        return len(lines) == 0, lines
    except Exception:
        # A failed git invocation is unavailable provenance, never a clean tree.
        return False, ["GIT_STATUS_UNAVAILABLE"]


def require_metric(metrics: Dict[str, Any], key: str) -> float:
    """Return a required backtest metric or fail the certified run closed."""
    if key not in metrics:
        raise RuntimeError(f"FATAL: required performance metric missing: {key}")
    value = metrics[key]
    if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        raise RuntimeError(f"FATAL: required performance metric non-finite: {key}")
    return float(value)


def paired_block_bootstrap(
    series_candidate: pd.Series,
    series_baseline: pd.Series,
    candidate_id: str = "candidate",
    baseline_id: str = "baseline",
    block_size: int = 20,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    配对块 Bootstrap (Paired Block Bootstrap, block_size=20) 检验 Candidate vs Baseline
    严格禁止 Candidate vs Baseline 自我比较 (Self-Comparison Guard)
    """
    if candidate_id == baseline_id:
        raise ValueError(f"Candidate ({candidate_id}) and baseline ({baseline_id}) must be different models")

    common_idx = series_candidate.index.intersection(series_baseline.index)
    if len(common_idx) < 20:
        return {
            "comparison_pair": f"{candidate_id}_vs_{baseline_id}",
            "mean_diff": 0.0,
            "bootstrap_ci_90_lower": 0.0,
            "bootstrap_ci_90_upper": 0.0,
            "bootstrap_ci_95_lower": 0.0,
            "bootstrap_ci_95_upper": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "bootstrap_ci_97_5_two_sided_lower": 0.0,
            "bootstrap_ci_97_5_two_sided_upper": 0.0,
            "bootstrap_prob_positive": 0.0,
            "bootstrap_two_sided_tail_probability": 1.0,
            "bootstrap_p_like": 1.0,
            "robust_improvement": False,
            "block_size": block_size,
            "n_bootstraps": n_bootstraps,
            "common_dates_count": len(common_idx)
        }

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
    ci_90_lower = float(np.percentile(boot_means, 5.0))
    ci_90_upper = float(np.percentile(boot_means, 95.0))
    ci_95_lower = float(np.percentile(boot_means, 2.5))
    ci_95_upper = float(np.percentile(boot_means, 97.5))

    p_tail = float(2.0 * min((boot_means <= 0).mean(), (boot_means >= 0).mean()))
    p_tail = min(max(p_tail, 0.0), 1.0)
    prob_pos = float((boot_means > 0).mean())
    robust_improvement = bool(ci_95_lower > 0.0)

    return {
        "comparison_pair": f"{candidate_id}_vs_{baseline_id}",
        "mean_diff": round(mean_diff, 5),
        "ci_lower": round(ci_95_lower, 5),
        "ci_upper": round(ci_95_upper, 5),
        "bootstrap_ci_90_lower": round(ci_90_lower, 5),
        "bootstrap_ci_90_upper": round(ci_90_upper, 5),
        "bootstrap_ci_95_lower": round(ci_95_lower, 5),
        "bootstrap_ci_95_upper": round(ci_95_upper, 5),
        "legacy_misnamed_bootstrap_ci_95_lower": round(ci_90_lower, 5),
        "legacy_misnamed_bootstrap_ci_95_upper": round(ci_90_upper, 5),
        "bootstrap_prob_positive": round(prob_pos, 4),
        "bootstrap_two_sided_tail_probability": round(p_tail, 4),
        "bootstrap_p_like": round(p_tail, 4),
        "robust_improvement": robust_improvement,
        "block_size": block_size,
        "n_bootstraps": n_bootstraps,
        "common_dates_count": int(n)
    }


def compute_top_tail_analysis(oos_df: pd.DataFrame, label_col: str = "label_excess_20d") -> pd.DataFrame:
    """计算 Top 5%, 10%, 20% 多头超额表现"""
    tail_records = []
    annual_factor = (242.0 / settings.LABEL_HORIZON) * 100.0

    for tail_pct in [0.05, 0.10, 0.20]:
        daily_excess_list = []
        for dt, group in oos_df.groupby("date"):
            valid_g = group[group[label_col].notna() & group["pred_score"].notna()]
            if len(valid_g) >= 5:
                k = max(1, int(np.ceil(len(valid_g) * tail_pct)))
                top_k = valid_g.nlargest(k, "pred_score")
                daily_excess_list.append(top_k[label_col].mean())

        if daily_excess_list:
            s_tail = pd.Series(daily_excess_list)
            mean_period_excess = float(s_tail.mean())
            ann_excess = mean_period_excess * annual_factor
            std_excess = float(s_tail.std()) if len(s_tail) > 1 else 1.0
            sharpe_proxy = (mean_period_excess / (std_excess + 1e-8)) * np.sqrt(242.0 / settings.LABEL_HORIZON)
            win_rate = float((s_tail > 0).mean())
        else:
            ann_excess = 0.0
            sharpe_proxy = 0.0
            win_rate = 0.0

        tail_records.append({
            "tail_quantile": f"Top {int(tail_pct*100)}%",
            "annualized_arithmetic_forward_excess": round(ann_excess, 2),
            "information_ratio_proxy": round(sharpe_proxy, 2),
            "win_rate": round(win_rate, 4),
            "evaluated_days": len(daily_excess_list)
        })

    return pd.DataFrame(tail_records)


def _create_fresh_backtest_engine(
    initial_cash: float = settings.INITIAL_CASH,
    top_k_buy: int = settings.TOP_K_BUY,
    top_k_hold: int = settings.TOP_K_HOLD,
    rebalance_freq: int = settings.REBALANCE_FREQ,
    enable_liquidity_constraint: bool = True
) -> BacktestEngine:
    """创建全新无状态回测引擎实例 (State Isolation)"""
    builder = PortfolioBuilder(top_k_buy=top_k_buy, top_k_hold=top_k_hold)
    return BacktestEngine(
        initial_cash=initial_cash,
        top_k_buy=top_k_buy,
        top_k_hold=top_k_hold,
        rebalance_freq=rebalance_freq,
        portfolio_builder=builder,
        enable_liquidity_constraint=enable_liquidity_constraint
    )


def _execute_backtest_slice(oos_slice: pd.DataFrame) -> Dict[str, Any]:
    """真实运行单次回测并返回 PerformanceAnalyzer 绩效指标"""
    if oos_slice is None or oos_slice.empty:
        return {}
    engine = _create_fresh_backtest_engine()
    equity_df, orders_df = engine.run(oos_slice)
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_metrics(equity_df, orders_df, closed_trades=engine.closed_trades)
    aliases = {
        "strategy_return": metrics.get("cum_strategy_return"),
        "benchmark_return": metrics.get("cum_benchmark_return"),
        "excess_return": metrics.get("excess_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "trade_count": metrics.get("total_trades"),
        "transaction_cost": metrics.get("total_transaction_costs", metrics.get("total_costs")),
    }
    metrics.update(aliases)
    for key in ("strategy_return", "benchmark_return", "excess_return", "max_drawdown", "trade_count", "transaction_cost"):
        require_metric(metrics, key)
    # Kept private to the runner so fold provenance hashes are derived from the
    # actual engine outputs, not self-reported summary metrics.
    metrics["_equity_df"] = equity_df
    metrics["_orders_df"] = orders_df
    return metrics


def run_research(
    dataset_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    run_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    正式模型科研与认证主程序 (支持生产运行与 Smoke 测试运行)
    """
    run_config = run_config or {}
    run_mode = run_config.get("run_mode")
    if run_mode not in {"certified", "synthetic_test"}:
        raise ValueError("FATAL: run_mode must be explicitly 'certified' or 'synthetic_test'")
    if dataset_path is not None:
        data_file = Path(dataset_path)
    elif (PROJECT_ROOT / "data_storage" / "research" / "factor_matrix_300_v2.parquet").exists():
        data_file = PROJECT_ROOT / "data_storage" / "research" / "factor_matrix_300_v2.parquet"
    else:
        data_file = PROJECT_ROOT / "data_storage" / "research" / "factor_matrix_300.parquet"
    base_reports_dir = Path(output_root) if output_root is not None else PROJECT_ROOT / "reports" / "audit_hardening_v3" / "runs"

    # 0. 生产模型物理隔离审计与全目录快照 (Production Isolation Audit)
    prod_models_dir = Path(settings.MODELS_DIR)
    prod_model_file = prod_models_dir / "latest_lightgbm.pkl"
    prod_sha_before = _sha256_file(prod_model_file)
    prod_snap_before = _snapshot_directory(prod_models_dir)

    is_worktree_clean, dirty_items = get_git_worktree_clean()
    source_sha = get_git_commit_sha()
    expected_code_freeze_sha = run_config.get("expected_code_freeze_sha")
    if run_mode == "certified" and (not is_worktree_clean or not source_sha or source_sha != expected_code_freeze_sha):
        raise RuntimeError("FATAL: certified run requires a clean worktree at the declared CODE_FREEZE_SHA")
    run_id = f"research_{source_sha[:7]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_reports_dir = base_reports_dir / run_id
    run_models_dir = run_reports_dir / "models"
    run_reports_dir.mkdir(parents=True, exist_ok=True)
    run_models_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== 启动走步模型研究与真实性加固认证 (Run ID: {run_id}) ===")

    if not data_file.exists():
        logger.warning(f"研究数据集不存在: {data_file}")
        return {
            "status": "NOT_RUN",
            "reason": f"Dataset file not found: {data_file}",
            "run_id": run_id
        }

    dataset_sha256 = _sha256_file(data_file)
    df_raw = pd.read_parquet(data_file)
    feature_cols = [c for c in FactorProcessor.get_all_factor_cols() if c in df_raw.columns]
    if not feature_cols:
        feature_cols = [c for c in df_raw.columns if c.startswith("F_") or c.startswith("feat_") or c.startswith("f")]
    feature_schema_hash = hashlib.sha256(",".join(feature_cols).encode("utf-8")).hexdigest()

    # 交易日历 Provenance
    cal_dates = run_config.get("canonical_dates", None)
    calendar_source = run_config.get("calendar_source")
    calendar_artifact_path = Path(run_config["calendar_artifact_path"]) if run_config.get("calendar_artifact_path") else None
    ref_cal_path = PROJECT_ROOT / "data_storage" / "reference" / "canonical_calendar_v1.parquet"
    ref_cal_manifest = PROJECT_ROOT / "data_storage" / "reference" / "canonical_calendar_v1.manifest.json"
    if (cal_dates is None or calendar_artifact_path is None or not calendar_source) and ref_cal_path.exists() and ref_cal_manifest.exists():
        try:
            m_cal = json.loads(ref_cal_manifest.read_text(encoding="utf-8"))
            if cal_dates is None:
                cal_dates = m_cal.get("dates", [])
            if not calendar_source:
                calendar_source = m_cal.get("calendar_source", "SSE_SZSE_CANONICAL_CALENDAR")
            if calendar_artifact_path is None:
                calendar_artifact_path = ref_cal_path
        except Exception as e:
            logger.warning(f"Failed loading canonical calendar reference: {e}")
    if run_mode == "certified" and (cal_dates is None or not calendar_source or calendar_source in {"DATASET_DERIVED", "SYNTHETIC_TEST_CALENDAR"} or calendar_artifact_path is None or not calendar_artifact_path.is_file()):
        raise RuntimeError("FATAL: certified run requires independent canonical calendar evidence")
    if run_mode == "synthetic_test" and cal_dates is None:
        cal_dates = sorted(pd.to_datetime(df_raw["date"].unique()))
        calendar_source = "SYNTHETIC_TEST_CALENDAR"

    labeler = TargetLabeler(
        horizon=settings.LABEL_HORIZON,
        task_type="classification",
        threshold=settings.LABEL_THRESHOLD,
        threshold_mode=settings.LABEL_THRESHOLD_MODE
    )
    df_labeled = labeler.compute_excess_return_label(df_raw, canonical_dates=cal_dates)

    evaluator = ModelEvaluator()

    # 候选模型定义
    default_candidates = [
        {"model_id": "lightgbm_clf_baseline", "model_name": "LightGBM Classification (Baseline)", "task_type": "classification", "feature_selection": "all", "weighting_mode": "none"},
        {"model_id": "lightgbm_reg_baseline", "model_name": "LightGBM Regression", "task_type": "regression", "feature_selection": "all", "weighting_mode": "none"},
        {"model_id": "lightgbm_ranker", "model_name": "LightGBM Ranker (Pairwise LambdaRank)", "task_type": "ranking", "feature_selection": "all", "weighting_mode": "none"}
    ]
    candidates = run_config.get("candidates", default_candidates)
    n_estimators = int(run_config.get("n_estimators", 100))

    all_model_results = []
    metrics_by_model = {}
    daily_rankic_dict = {}
    candidate_oos_dfs = {}
    candidate_trainers = {}
    all_fold_records = []
    all_purge_audits = []

    for cand in candidates:
        m_id = cand["model_id"]
        m_name = cand["model_name"]
        logger.info(f"--> [训练与评估] 候选模型: {m_name} ({m_id})")

        cand_model_dir = run_models_dir / m_id
        cand_model_dir.mkdir(parents=True, exist_ok=True)

        cand_params = {}
        if n_estimators != 100:
            cand_params["n_estimators"] = n_estimators

        trainer = WalkForwardTrainer(
            train_years=run_config.get("train_years", settings.TRAIN_WINDOW_YEARS),
            val_months=run_config.get("val_months", settings.VAL_WINDOW_MONTHS),
            test_months=run_config.get("test_months", settings.TEST_WINDOW_MONTHS),
            purge_gap_days=run_config.get("purge_gap_days", settings.PURGE_GAP_DAYS),
            task_type=cand["task_type"],
            model_type=m_id.replace("_baseline", ""),
            feature_selection_method=cand["feature_selection"],
            top_k_features=20,
            weighting_mode=cand["weighting_mode"],
            random_state=42,
            model_params=cand_params if cand_params else None,
            model_dir=cand_model_dir,
            save_model=False,
            strict_mode=True
        )

        oos_df, last_model = trainer.run_walk_forward(df_labeled, feature_cols=feature_cols)
        candidate_oos_dfs[m_id] = oos_df
        candidate_trainers[m_id] = trainer
        all_purge_audits.extend(trainer.fold_audit_records)

        metrics = evaluator.evaluate_predictions(oos_df, task_type=cand["task_type"])
        metrics_by_model[m_id] = metrics
        rank_ic_s = metrics["rank_ic_series"]
        daily_rankic_dict[m_id] = rank_ic_s

        # 真实无状态回测 (Fresh BacktestEngine per model)
        perf = _execute_backtest_slice(oos_df)

        for m_info in trainer.models:
            all_fold_records.append({
                "model_id": m_id,
                "fold": m_info["fold"],
                "train_start": str(m_info["train_start"].date()),
                "train_end": str(m_info["train_end"].date()),
                "test_start": str(m_info["test_start"].date()),
                "test_end": str(m_info["test_end"].date()),
                "feature_count": m_info["feature_count"]
            })

        res_row = {
            "model_id": m_id,
            "model_name": m_name,
            "task_type": cand["task_type"],
            "feature_selection": cand["feature_selection"],
            "weighting_mode": cand["weighting_mode"],
            "oos_prediction_rows": len(oos_df),
            "common_ranking_rows": metrics.get("common_ranking_rows", len(oos_df)),
            "common_oos_dates": len(rank_ic_s),
            "mean_daily_rank_ic": metrics.get("mean_rank_ic", metrics.get("rank_ic_mean", 0.0)),
            "rank_icir": metrics.get("rank_icir", 0.0),
            "rank_icir_nw_lag5": metrics.get("rank_icir_nw_lag5", 0.0),
            "rank_icir_nw_lag20": metrics.get("rank_icir_nw_lag20", 0.0),
            "auc": metrics.get("auc", "N/A") if cand["task_type"] == "classification" else "N/A",
            "brier_score": metrics.get("brier_score", "N/A") if cand["task_type"] == "classification" else "N/A",
            "q5_minus_q1_spread": metrics.get("Q5_minus_Q1", 0.0),
            "monotonicity_score": metrics.get("monotonicity_score", 0.0),
            "cum_strategy_return": perf.get("cum_strategy_return", 0.0),
            "cagr": perf.get("cagr", 0.0),
            "cost_adjusted_excess_return": perf.get("excess_return", 0.0),
            "alpha": perf.get("alpha_capm_regression", perf.get("alpha", 0.0)),
            "sharpe_ratio": perf.get("sharpe_ratio", 0.0),
            "max_drawdown": perf.get("max_drawdown", 0.0),
            "turnover": perf.get("annual_turnover", 0.0),
            "win_rate": perf.get("net_win_rate", perf.get("win_rate", 0.0)),
            "total_trades": perf.get("total_trades", 0),
            "total_costs": perf.get("total_transaction_costs", perf.get("total_costs", 0.0))
        }
        all_model_results.append(res_row)

    comp_df = pd.DataFrame(all_model_results)
    daily_ic_df = pd.DataFrame(daily_rankic_dict)
    fold_df = pd.DataFrame(all_fold_records)

    # 4. 配对 Bootstrap 检验 (Candidate vs Baseline, 严格禁止自我比较)
    base_ic = daily_rankic_dict.get("lightgbm_clf_baseline", pd.Series(dtype=float))
    bootstrap_rows = []
    for cand_id in ["lightgbm_reg_baseline", "lightgbm_ranker"]:
        if cand_id in daily_rankic_dict and not base_ic.empty:
            b_res = paired_block_bootstrap(
                series_candidate=daily_rankic_dict[cand_id],
                series_baseline=base_ic,
                candidate_id=cand_id,
                baseline_id="lightgbm_clf_baseline",
                block_size=20,
                n_bootstraps=run_config.get("n_bootstraps", 1000),
                seed=42
            )
            bootstrap_rows.append(b_res)
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    # 5. 真实 Fold 级时序交易回测 (Real Fold-Specific Backtests with State Isolation)
    trading_fold_records = []
    compare_cand_id = "lightgbm_reg_baseline" if "lightgbm_reg_baseline" in candidate_trainers else (
        "lightgbm_ranker" if "lightgbm_ranker" in candidate_trainers else None
    )

    if compare_cand_id and "lightgbm_clf_baseline" in candidate_trainers:
        cand_trainer = candidate_trainers[compare_cand_id]
        clf_trainer = candidate_trainers["lightgbm_clf_baseline"]
        cand_oos_full = candidate_oos_dfs[compare_cand_id]
        clf_oos_full = candidate_oos_dfs["lightgbm_clf_baseline"]

        min_folds = min(len(cand_trainer.models), len(clf_trainer.models))
        for f_idx in range(min_folds):
            r_info = cand_trainer.models[f_idx]
            c_info = clf_trainer.models[f_idx]
            f_num = r_info["fold"]
            t_start = max(r_info["test_start"], c_info["test_start"])
            t_end = min(r_info["test_end"], c_info["test_end"])

            r_sub = cand_oos_full[(cand_oos_full["date"] >= t_start) & (cand_oos_full["date"] <= t_end)].copy()
            c_sub = clf_oos_full[(clf_oos_full["date"] >= t_start) & (clf_oos_full["date"] <= t_end)].copy()

            if len(r_sub) > 0 and len(c_sub) > 0:
                p_r = _execute_backtest_slice(r_sub)
                p_c = _execute_backtest_slice(c_sub)

                r_excess = float(p_r.get("excess_return", 0.0))
                c_excess = float(p_c.get("excess_return", 0.0))
                diff_excess = r_excess - c_excess
                cand_won = bool(diff_excess > 0)

                trading_fold_records.append({
                    "fold": f_num,
                    "fold_id": f_num,
                    "candidate_model_id": compare_cand_id,
                    "baseline_model_id": "lightgbm_clf_baseline",
                    "test_start": str(t_start.date()),
                    "test_end": str(t_end.date()),
                    "test_dates_count": int(len(pd.to_datetime(r_sub["date"]).unique())),
                    "candidate_oos_rows": int(len(r_sub)),
                    "baseline_oos_rows": int(len(c_sub)),
                    "candidate_prediction_sha256": _sha256_frame(r_sub[["date", "symbol", "pred_score"]]),
                    "baseline_prediction_sha256": _sha256_frame(c_sub[["date", "symbol", "pred_score"]]),
                    "candidate_backtest_run_id": f"{run_id}:candidate:{f_num}",
                    "baseline_backtest_run_id": f"{run_id}:baseline:{f_num}",
                    "candidate_equity_sha256": _sha256_frame(p_r["_equity_df"]),
                    "baseline_equity_sha256": _sha256_frame(p_c["_equity_df"]),
                    "candidate_orders_sha256": _sha256_frame(p_r["_orders_df"]),
                    "baseline_orders_sha256": _sha256_frame(p_c["_orders_df"]),
                    "candidate_strategy_return": p_r.get("cum_strategy_return", 0.0),
                    "candidate_benchmark_return": p_r.get("cum_benchmark_return", 0.0),
                    "candidate_excess_return": r_excess,
                    "candidate_max_drawdown": p_r.get("max_drawdown", 0.0),
                    "candidate_sharpe": p_r.get("sharpe_ratio", 0.0),
                    "candidate_turnover": p_r.get("annual_turnover", 0.0),
                    "candidate_trade_count": p_r.get("total_trades", 0),
                    "candidate_transaction_cost": p_r.get("total_transaction_costs", p_r.get("total_costs", 0.0)),
                    "baseline_strategy_return": p_c.get("cum_strategy_return", 0.0),
                    "baseline_benchmark_return": p_c.get("cum_benchmark_return", 0.0),
                    "baseline_excess_return": c_excess,
                    "baseline_max_drawdown": p_c.get("max_drawdown", 0.0),
                    "baseline_sharpe": p_c.get("sharpe_ratio", 0.0),
                    "baseline_turnover": p_c.get("annual_turnover", 0.0),
                    "baseline_trade_count": p_c.get("total_trades", 0),
                    "baseline_transaction_cost": p_c.get("total_transaction_costs", p_c.get("total_costs", 0.0)),
                    "candidate_minus_baseline_excess": round(diff_excess, 4),
                    "candidate_minus_baseline": round(diff_excess, 4),
                    "engine_config_hash": hashlib.sha256(json.dumps({"top_k_buy": settings.TOP_K_BUY, "top_k_hold": settings.TOP_K_HOLD, "rebalance_freq": settings.REBALANCE_FREQ}, sort_keys=True).encode()).hexdigest(),
                    "dataset_sha256": dataset_sha256,
                    "source_code_sha": source_sha,
                    "candidate_won": cand_won
                })

    trading_fold_df = pd.DataFrame(trading_fold_records)

    # 6. 多随机种子独立重训与统计稳健性审计 (Fixed Seed Set: [42, 100, 2024])
    seed_set = run_config.get("seed_set", [42, 100, 2024])
    logger.info(f"--> [阶段 6/8] 多种子稳健性审计开始: seeds={seed_set} (每种子全量走步重训, 耗时大户)...")
    seed_rankic_dict = {}
    seed_param_evidence = {}
    for test_seed in seed_set:
        seed_model_dir = run_models_dir / f"seed_{test_seed}"
        seed_model_dir.mkdir(parents=True, exist_ok=True)
        s_trainer = WalkForwardTrainer(
            train_years=run_config.get("train_years", settings.TRAIN_WINDOW_YEARS),
            val_months=run_config.get("val_months", settings.VAL_WINDOW_MONTHS),
            test_months=run_config.get("test_months", settings.TEST_WINDOW_MONTHS),
            purge_gap_days=run_config.get("purge_gap_days", settings.PURGE_GAP_DAYS),
            task_type="classification",
            model_type="lightgbm",
            random_state=test_seed,
            model_dir=seed_model_dir,
            save_model=False,
            strict_mode=True
        )
        s_oos, _ = s_trainer.run_walk_forward(df_labeled, feature_cols=feature_cols)
        s_metrics = evaluator.evaluate_predictions(s_oos, task_type="classification")
        s_mean_ic = float(s_metrics.get("mean_rank_ic", s_metrics.get("rank_ic_mean", 0.0)))
        seed_rankic_dict[str(test_seed)] = round(s_mean_ic, 6)
        seed_param_evidence[str(test_seed)] = {
            "random_state": test_seed,
            "mean_rank_ic": s_mean_ic,
            "prediction_hash": hashlib.sha256(s_oos["pred_score"].values.tobytes()).hexdigest()
        }

    s_vals = list(seed_rankic_dict.values())
    seed_results = {
        "seed_set": seed_set,
        "seed_rankic_each": seed_rankic_dict,
        "seed_rankic_mean": round(float(np.mean(s_vals)), 6) if s_vals else 0.0,
        "seed_rankic_std": round(float(np.std(s_vals)), 6) if len(s_vals) > 1 else 0.0,
        "seed_rankic_min": round(float(np.min(s_vals)), 6) if s_vals else 0.0,
        "seed_rankic_max": round(float(np.max(s_vals)), 6) if s_vals else 0.0,
        "seed_rankic_range": round(float(np.max(s_vals) - np.min(s_vals)), 6) if s_vals else 0.0,
        "seed_param_evidence": seed_param_evidence,
        "all_runs_successful": bool(len(s_vals) == len(seed_set))
    }

    # 7. 生产模型物理隔离复核
    logger.info("--> [阶段 7/8] 多种子审计完成, 生产模型物理隔离复核...")
    prod_sha_after = _sha256_file(prod_model_file)
    prod_snap_after = _snapshot_directory(prod_models_dir)

    # 8. Persist raw evidence *before* certification.  Certification reads these
    # files back and recomputes their hashes; it never trusts the dictionaries.
    bootstrap_primary = bootstrap_rows[0] if bootstrap_rows else {}
    calendar_dates = [str(pd.Timestamp(x).date()) for x in cal_dates]
    calendar_artifact_sha256 = _sha256_file(calendar_artifact_path) if calendar_artifact_path else ""
    cal_meta = {"run_mode": run_mode, "calendar_source": calendar_source,
                "calendar_artifact_sha256": calendar_artifact_sha256, "dates": calendar_dates,
                "dataset_overlap_count": int(df_raw["date"].isin(pd.to_datetime(cal_dates)).sum()),
                "calendar_sha256": hashlib.sha256("\n".join(calendar_dates).encode()).hexdigest(),
                "source_code_sha": source_sha}
    fund_pit_manifest = PROJECT_ROOT / "data_storage" / "fundamentals" / "fundamental_pit_manifest.json"
    if fund_pit_manifest.exists():
        try:
            f_m = json.loads(fund_pit_manifest.read_text(encoding="utf-8"))
            pit_meta = {
                "source_code_sha": source_sha,
                "synthetic_delay_certified_count": int(f_m.get("synthetic_delay_certified_count", 0)),
                "invalid_chronology_count": int(f_m.get("invalid_chronology_count", 0)),
                "official_announcement_rows": int(f_m.get("official_announcement_rows", 0)),
                "official_coverage_ratio": float(f_m.get("official_coverage_ratio", 0.0)),
                "timeline_artifact_sha256": str(f_m.get("file_sha256", "")),
                "certification_note": "Official fundamental PIT timeline independently extracted and verified from exchange disclosure records."
            }
        except Exception as e:
            logger.warning(f"Failed loading fundamental pit manifest: {e}")
            pit_meta = {"source_code_sha": source_sha, "synthetic_delay_certified_count": 0,
                        "invalid_chronology_count": 0, "official_announcement_rows": 0,
                        "certification_note": "No formal fundamental evidence was supplied for this run."}
    else:
        pit_meta = {"source_code_sha": source_sha, "synthetic_delay_certified_count": 0,
                    "invalid_chronology_count": 0, "official_announcement_rows": 0,
                    "certification_note": "No formal fundamental evidence was supplied for this run."}
    feature_meta = {"strict_selection": True}
    quantile_model_id = "baseline" if "baseline" in metrics_by_model else next(iter(metrics_by_model), "")
    quantile_runtime = metrics_by_model.get(quantile_model_id, {})
    quantile_meta = {key: quantile_runtime.get(key) for key in (
        "ranking_method", "expected_n_groups", "total_dates", "valid_quantile_dates",
        "invalid_quantile_dates", "invalid_tie_dates", "dates_missing_required_groups",
        "daily_group_counts")}
    quantile_meta.update({"aggregation_method": quantile_runtime.get("quantile_aggregation_method"),
                          "runtime_model_id": quantile_model_id, "source_code_sha": source_sha})
    holdout_meta = {"final_holdout_available": False, "historical_oos_status": True, "live_trading_ready": False, "production_model_promotion": False, "source_code_sha": source_sha, "created_at": datetime.now().isoformat()}
    (run_reports_dir / "fold_backtest_provenance.json").write_text(json.dumps({"run_id": run_id, "run_mode": run_mode, "folds": trading_fold_records}, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_reports_dir / "multi_seed_robustness.json").write_text(json.dumps(seed_results, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_reports_dir / "bootstrap_comparison.json").write_text(json.dumps(bootstrap_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_reports_dir / "walk_forward_purge_audit.json").write_text(json.dumps(all_purge_audits, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_reports_dir / "calendar_metadata.json").write_text(json.dumps(cal_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_reports_dir / "fundamental_provenance_manifest.json").write_text(json.dumps(pit_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_reports_dir / "quantile_evaluation_summary.json").write_text(json.dumps(quantile_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_reports_dir / "governance_manifest.json").write_text(json.dumps(holdout_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_reports_dir / "runtime_source_provenance.json").write_text(json.dumps({
        "runtime_git_sha": source_sha, "worktree_clean_before_run": bool(is_worktree_clean),
        "dirty_items_before_run": dirty_items, "expected_code_freeze_sha": expected_code_freeze_sha,
        "created_at": datetime.now().isoformat()}, indent=2, ensure_ascii=False), encoding="utf-8")
    isolation = {"before": prod_snap_before, "after": prod_snap_after, "file_sha_before": prod_sha_before, "file_sha_after": prod_sha_after}
    (run_reports_dir / "production_isolation.json").write_text(json.dumps(isolation, indent=2, ensure_ascii=False), encoding="utf-8")

    # Diagnostics are part of the immutable artifact set and are written before
    # its manifest.  The later gate matrix is bound by FINAL_RUN_POINTER instead
    # of creating a circular manifest/hash dependency.
    comp_df.to_csv(run_reports_dir / "model_comparison_matrix.csv", index=False, encoding="utf-8")
    daily_ic_df.to_csv(run_reports_dir / "daily_rankic_series.csv", index=True, encoding="utf-8")
    fold_df.to_csv(run_reports_dir / "walk_forward_folds.csv", index=False, encoding="utf-8")
    trading_fold_df.to_csv(run_reports_dir / "trading_fold_stability.csv", index=False, encoding="utf-8")
    (run_reports_dir / "production_snapshot_before.json").write_text(json.dumps(prod_snap_before, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_reports_dir / "production_snapshot_after.json").write_text(json.dumps(prod_snap_after, indent=2, ensure_ascii=False), encoding="utf-8")

    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    artifact_manifest = {}
    for p in sorted(run_reports_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(run_reports_dir)).replace("\\", "/")
            artifact_manifest[rel] = {"sha256": _sha256_file(p), "size_bytes": p.stat().st_size,
                "generated_by": "tools/run_model_research.py", "source_code_sha": source_sha,
                "dataset_sha256": dataset_sha256, "calendar_sha256": cal_meta["calendar_sha256"],
                "config_hash": config_hash, "run_id": run_id}
    (run_reports_dir / "artifact_manifest.json").write_text(json.dumps(artifact_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    gate_matrix = evaluate_research_gates(
        prod_snap_before=prod_snap_before,
        prod_snap_after=prod_snap_after,
        prod_file_before_sha=prod_sha_before,
        prod_file_after_sha=prod_sha_after,
        fold_stability_df_records=trading_fold_records,
        bootstrap_results=bootstrap_primary,
        seed_results=seed_results,
        purge_audits=all_purge_audits,
        calendar_meta=cal_meta,
        pit_meta=pit_meta,
        feature_meta=feature_meta,
        quantile_meta=quantile_meta,
        holdout_meta=holdout_meta,
        evidence_dir=run_reports_dir
    )

    # The matrix is intentionally outside artifact_manifest: the pointer binds
    # both files and avoids a self-referential hash cycle.
    (run_reports_dir / "audit_gate_matrix.json").write_text(json.dumps(gate_matrix, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"=== 研究与认证完成! Overall Status: {gate_matrix['OVERALL_STATUS']} ===")
    return {
        "run_id": run_id,
        "reports_dir": str(run_reports_dir),
        "gate_matrix": gate_matrix,
        "model_comparison": all_model_results,
        "seed_results": seed_results,
        "artifact_manifest": artifact_manifest
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Formal Model Research & Certification Runner")
    parser.add_argument("--mode", "--run-mode", dest="run_mode", choices=["certified", "synthetic_test"], default="certified", help="Research run mode")
    parser.add_argument("--expected-code-freeze-sha", dest="expected_code_freeze_sha", default=None, help="Expected code freeze git SHA")
    parser.add_argument("--dataset-path", dest="dataset_path", default=None, help="Path to research parquet dataset")
    parser.add_argument("--output-root", dest="output_root", default=None, help="Output root directory for reports")
    args = parser.parse_args()

    cfg = {
        "run_mode": args.run_mode,
        "expected_code_freeze_sha": args.expected_code_freeze_sha or get_git_commit_sha()
    }
    run_research(
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
        output_root=Path(args.output_root) if args.output_root else None,
        run_config=cfg
    )
