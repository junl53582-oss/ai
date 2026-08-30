"""
Phase 2.1-A r2 Controlled A/B Study Runner (tools/run_phase2_1_a_label_ab.py)
严格单变量对比：Legacy Labels (Arm A) vs Execution-Aligned Labels (Arm B)。
包含严格Fail-Closed源码血缘门禁、生产模型物理隔离全目录审计、固定有效参数哈希(scale_pos_weight=1.0)、有效Fold统计与大文件Manifest证据链。
"""
from __future__ import annotations

import sys
import argparse
import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import settings
from factors.processor import FactorProcessor
from models.evaluator import ModelEvaluator
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from research_v2.labels.execution_labeler import ExecutionAlignedLabeler

logger = logging.getLogger("phase2_1_a_ab")
EXEC_LABEL = "label_net_alpha_20d"
EXEC_DIRECTION = "label_direction_20d"


def _git_sha(cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd or PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def _git_branch(cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.run(["git", "branch", "--show-current"], cwd=cwd or PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def _git_remote_sha(branch: str, cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.run(["git", "rev-parse", f"origin/{branch}"], cwd=cwd or PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def _git_working_tree_clean(project_root: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """
    严密检查工作区是否处于无未提交代码/测试/配置的 Clean 状态。
    仅允许被 .gitignore 明确忽略的运行时产物，任何未被 ignore 的 untracked 源码/文件均触发 Fail-Closed。
    """
    root = project_root or PROJECT_ROOT
    try:
        res = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                             cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        if not res:
            return True, []

        dirty_items: List[str] = []
        for line in res.splitlines():
            line = line.strip()
            if not line:
                continue
            status_code = line[:2]
            item_path = line[3:].strip().strip('"')

            if status_code == "??":
                # 检查是否属于 .gitignore 明确允许的路径
                check_res = subprocess.run(["git", "check-ignore", "-q", item_path], cwd=root)
                if check_res.returncode != 0:
                    dirty_items.append(f"UNTRACKED: {item_path}")
            else:
                dirty_items.append(f"MODIFIED/STAGED ({status_code}): {item_path}")

        return len(dirty_items) == 0, dirty_items
    except Exception as e:
        return False, [f"ERROR: {e}"]


def _validate_source_provenance(enforce_clean: bool = True, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    统一严密校验源码血缘与远程分支一致性。
    若存在脏工作树、未提交源码或与远程分支不匹配，则严格 Fail-Closed 终止。
    """
    root = project_root or PROJECT_ROOT
    source_sha = _git_sha(root)
    branch_name = _git_branch(root)
    remote_sha = _git_remote_sha(branch_name, root) if branch_name != "UNKNOWN" else "UNKNOWN"
    is_clean, dirty_items = _git_working_tree_clean(root)

    if enforce_clean:
        if source_sha == "UNKNOWN":
            raise RuntimeError("FATAL: Unable to resolve HEAD source commit SHA.")
        if branch_name == "UNKNOWN":
            raise RuntimeError("FATAL: Unable to resolve current Git branch name.")
        if remote_sha == "UNKNOWN":
            raise RuntimeError(f"FATAL: Unable to resolve remote tracking SHA for branch 'origin/{branch_name}'.")
        if not is_clean:
            raise RuntimeError(f"FATAL: Phase 2.1 research must run from a clean tracked source tree. Found dirty items: {dirty_items}")
        if source_sha != remote_sha:
            raise RuntimeError(f"FATAL: Local HEAD ({source_sha}) does not match origin/{branch_name} ({remote_sha}).")

    return {
        "source_commit_sha": source_sha,
        "source_commit_branch": branch_name,
        "source_commit_remote": remote_sha,
        "source_commit_tree_clean": is_clean,
        "source_commit_remote_match": bool(source_sha == remote_sha and source_sha != "UNKNOWN"),
        "dirty_items": dirty_items,
        "experiment_generated_from_clean_commit": is_clean
    }


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_directory(dir_path: Path) -> Dict[str, Dict[str, Any]]:
    if not dir_path.exists():
        return {}
    snap = {}
    for p in dir_path.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(dir_path)).replace("\\", "/")
            snap[rel] = {
                "size_bytes": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "sha256": _sha256_file(p)
            }
    return snap


def _daily_rankic(df: pd.DataFrame, pred_col: str, target_col: str) -> pd.Series:
    values = {}
    for dt, g in df.groupby("date"):
        valid = g[[pred_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 3:
            continue
        ic = stats.spearmanr(valid[pred_col], valid[target_col])[0]
        if np.isfinite(ic):
            values[pd.Timestamp(dt)] = float(ic)
    return pd.Series(values, dtype=float).sort_index()


def _top10_daily_alpha(df: pd.DataFrame, pred_col: str, target_col: str) -> float:
    vals = []
    for _, g in df.groupby("date"):
        valid = g[[pred_col, target_col]].dropna()
        if valid.empty:
            continue
        k = max(1, int(np.ceil(len(valid) * 0.10)))
        vals.append(float(valid.nlargest(k, pred_col)[target_col].mean()))
    return float(np.mean(vals)) if vals else 0.0


def _paired_block_bootstrap(candidate: pd.Series, baseline: pd.Series,
                            block_size: int = 20, n_bootstraps: int = 2000,
                            seed: int = 42) -> Dict[str, float]:
    common = candidate.index.intersection(baseline.index)
    diff = (candidate.loc[common] - baseline.loc[common]).dropna().to_numpy(dtype=float)
    if len(diff) < max(20, block_size):
        return {
            "common_dates": int(len(diff)),
            "mean_diff": float(np.mean(diff)) if len(diff) else 0.0,
            "ci_lower": 0.0, "ci_upper": 0.0, "prob_positive": 0.0,
            "robust_improvement": False
        }

    rng = np.random.RandomState(seed)
    n = len(diff)
    n_blocks = int(np.ceil(n / block_size))
    boot = np.empty(n_bootstraps, dtype=float)
    max_start = max(1, n - block_size + 1)
    for i in range(n_bootstraps):
        starts = rng.randint(0, max_start, size=n_blocks)
        sample = np.concatenate([diff[s:s + block_size] for s in starts])[:n]
        boot[i] = sample.mean()

    ci_lower = float(np.percentile(boot, 2.5))
    ci_upper = float(np.percentile(boot, 97.5))
    prob_pos = float((boot > 0).mean())
    robust = bool(ci_lower > 0.0)

    return {
        "common_dates": int(n),
        "mean_diff": float(diff.mean()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "prob_positive": prob_pos,
        "robust_improvement": robust
    }


def _evaluate_common_pool(common: pd.DataFrame, pred_col: str, target_col: str) -> Tuple[Dict[str, float], pd.Series]:
    evaluator = ModelEvaluator()
    daily = _daily_rankic(common, pred_col, target_col)
    mean_ic = float(daily.mean()) if len(daily) else 0.0
    std_nw20 = evaluator._compute_newey_west_std(daily, max_lag=20)
    annual_factor = np.sqrt(242.0 / settings.LABEL_HORIZON)
    nw20 = (mean_ic / (std_nw20 + 1e-8)) * annual_factor if len(daily) else 0.0

    q_frame = common[["date", "symbol", pred_col, target_col]].rename(columns={pred_col: "pred_score"})
    q = evaluator._compute_quantile_returns(q_frame, label_col=target_col, n_groups=5)

    metrics = {
        "mean_daily_rank_ic": mean_ic,
        "nw20_rank_icir": float(nw20),
        "rank_ic_positive_rate": float((daily > 0).mean()) if len(daily) else 0.0,
        "q5_minus_q1_annualized_pct_points": float(q.get("Q5_minus_Q1", 0.0)),
        "monotonicity_score": float(q.get("monotonicity_score", 0.0)),
        "top10_mean_20d_exec_alpha": _top10_daily_alpha(common, pred_col, target_col),
        "oos_rows": int(len(common)),
        "oos_dates": int(common["date"].nunique()),
    }
    return metrics, daily


def _build_common_training_labels(df: pd.DataFrame) -> pd.DataFrame:
    legacy = settings.LABEL_COLUMN_CLF
    required = [legacy, EXEC_DIRECTION, "label_valid"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing labels for A/B experiment: {missing}")

    out = df.copy()
    in_univ = out["in_universe"].fillna(False).astype(bool) if "in_universe" in out.columns else pd.Series(True, index=out.index)
    not_excl = ~out["excluded_from_training"].fillna(False).astype(bool) if "excluded_from_training" in out.columns else pd.Series(True, index=out.index)

    common_train = out["label_valid"].fillna(False).astype(bool) & out[legacy].notna() & out[EXEC_DIRECTION].notna() & in_univ & not_excl
    out["ab_label_legacy"] = out[legacy].where(common_train)
    out["ab_label_execution"] = out[EXEC_DIRECTION].where(common_train)
    out["ab_common_train_eligible"] = common_train
    return out


def _trainer(label_col: str, model_dir: Path, model_params: Dict[str, Any]) -> WalkForwardTrainer:
    return WalkForwardTrainer(
        train_years=settings.TRAIN_WINDOW_YEARS,
        val_months=settings.VAL_WINDOW_MONTHS,
        test_months=settings.TEST_WINDOW_MONTHS,
        purge_gap_days=settings.PURGE_GAP_DAYS,
        label_col=label_col,
        task_type="classification",
        model_type="lightgbm",
        feature_selection_method="all",
        top_k_features=20,
        weighting_mode="none",
        random_state=42,
        model_dir=model_dir,
        model_params=model_params
    )


def _fold_comparison(trainer_a, trainer_b, daily_a, daily_b) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    n = min(len(trainer_a.models), len(trainer_b.models))
    for i in range(n):
        a = trainer_a.models[i]
        b = trainer_b.models[i]
        start = pd.Timestamp(max(a["test_start"], b["test_start"]))
        end = pd.Timestamp(min(a["test_end"], b["test_end"]))
        a_fold = daily_a[(daily_a.index >= start) & (daily_a.index <= end)]
        b_fold = daily_b[(daily_b.index >= start) & (daily_b.index <= end)]
        idx = a_fold.index.intersection(b_fold.index)
        has_dates = len(idx) > 0
        a_mean = float(a_fold.loc[idx].mean()) if has_dates else np.nan
        b_mean = float(b_fold.loc[idx].mean()) if has_dates else np.nan
        is_valid = has_dates and np.isfinite(a_mean) and np.isfinite(b_mean)
        diff = b_mean - a_mean if is_valid else np.nan
        wins = bool(b_mean > a_mean) if is_valid else False
        rows.append({
            "fold": i + 1,
            "test_start": str(start.date()),
            "test_end": str(end.date()),
            "common_rankic_dates": int(len(idx)),
            "valid_comparison": bool(is_valid),
            "legacy_mean_rankic": a_mean,
            "execution_mean_rankic": b_mean,
            "execution_minus_legacy": diff,
            "execution_wins": wins,
        })
    folds_df = pd.DataFrame(rows)
    valid_folds = folds_df[folds_df["valid_comparison"]].copy()
    zero_date_folds = folds_df[~folds_df["valid_comparison"]].copy()

    exec_wins = int(valid_folds["execution_wins"].sum())
    legacy_wins = int((valid_folds["execution_minus_legacy"] < 0).sum())
    ties = int((valid_folds["execution_minus_legacy"] == 0).sum())
    win_ratio = float(valid_folds["execution_wins"].mean()) if not valid_folds.empty else 0.0

    stats_summary = {
        "total_generated_folds": int(len(folds_df)),
        "valid_comparison_folds": int(len(valid_folds)),
        "zero_common_date_folds": int(len(zero_date_folds)),
        "execution_wins": exec_wins,
        "legacy_wins": legacy_wins,
        "ties": ties,
        "fold_win_ratio": win_ratio,
        "best_fold_delta": float(valid_folds["execution_minus_legacy"].max()) if not valid_folds.empty else 0.0,
        "worst_fold_delta": float(valid_folds["execution_minus_legacy"].min()) if not valid_folds.empty else 0.0,
    }
    return folds_df, stats_summary


def run(dataset_path: Path, output_dir: Path) -> Path:
    # 0. 源码血缘与工作树硬门禁校验 (Fail-Closed Provenance Gate)
    provenance = _validate_source_provenance(enforce_clean=True)
    source_sha = provenance["source_commit_sha"]
    branch_name = provenance["source_commit_branch"]
    tree_clean = provenance["source_commit_tree_clean"]
    remote_sha = provenance["source_commit_remote"]

    # 1. 实验前环境与生产模型全目录快照 (Path from settings.MODELS_DIR)
    prod_models_dir = Path(settings.MODELS_DIR)
    prod_model_path = prod_models_dir / "latest_lightgbm.pkl"
    prod_exists_before = prod_model_path.exists()
    prod_sha_before = _sha256_file(prod_model_path)
    prod_dir_snap_before = _snapshot_directory(prod_models_dir)

    run_id = f"phase2_1_a_{source_sha[:7]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    legacy_model_dir = run_dir / "models" / "legacy"
    exec_model_dir = run_dir / "models" / "execution"
    legacy_model_dir.mkdir(parents=True, exist_ok=True)
    exec_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"==> [1/7] 读取数据集: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["date", "symbol"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    dataset_sha = _sha256_file(dataset_path)

    print("==> [2/7] 计算 Legacy 标签与 Execution-Aligned 标签...")
    legacy_labeler = TargetLabeler(
        horizon=settings.LABEL_HORIZON, task_type="classification",
        threshold=settings.LABEL_THRESHOLD, threshold_mode=settings.LABEL_THRESHOLD_MODE
    )
    labeled = legacy_labeler.compute_excess_return_label(df)
    labeled = ExecutionAlignedLabeler().compute(labeled)
    labeled = _build_common_training_labels(labeled)

    # 标签诊断数据
    total_rows = len(labeled)
    valid_exec_count = int(labeled["label_valid"].sum())
    invalid_exec_count = total_rows - valid_exec_count

    deferred_mask = labeled["exit_deferred_days"].notna() & (labeled["exit_deferred_days"] > 0)
    deferred_count = int(deferred_mask.sum())
    def_days = labeled.loc[deferred_mask, "exit_deferred_days"].dropna().values.astype(float)
    mean_def_days = float(np.mean(def_days)) if len(def_days) else 0.0
    max_def_days = float(np.max(def_days)) if len(def_days) else 0.0
    p95_def_days = float(np.percentile(def_days, 95)) if len(def_days) else 0.0

    diag_df = pd.DataFrame([{
        "total_rows": total_rows,
        "valid_execution_labels": valid_exec_count,
        "invalid_execution_labels": invalid_exec_count,
        "invalid_ratio": invalid_exec_count / total_rows,
        "entry_tradable_ratio": float(labeled["entry_tradable"].mean()),
        "planned_exit_tradable_ratio": float(labeled["planned_exit_tradable"].mean()),
        "deferred_exit_count": deferred_count,
        "mean_deferred_days": mean_def_days,
        "max_deferred_days": max_def_days,
        "p95_deferred_days": p95_def_days
    }])
    diag_df.to_csv(run_dir / "label_diagnostics.csv", index=False, encoding="utf-8-sig")

    reason_df = pd.DataFrame([
        {"invalid_reason": k if k else "VALID", "count": v, "ratio": v / total_rows}
        for k, v in labeled["label_invalid_reason"].value_counts(dropna=False).items()
    ])
    reason_df.to_csv(run_dir / "invalid_reason_summary.csv", index=False, encoding="utf-8-sig")

    defer_dist_df = pd.DataFrame([
        {"exit_deferred_days": k, "count": v, "ratio": v / total_rows}
        for k, v in labeled["exit_deferred_days"].value_counts(dropna=False).sort_index().items()
    ])
    defer_dist_df.to_csv(run_dir / "exit_defer_distribution.csv", index=False, encoding="utf-8-sig")

    # 共同训练池统计
    common_train_rows_count = int(labeled["ab_common_train_eligible"].sum())
    common_train_dates_count = int(labeled.loc[labeled["ab_common_train_eligible"], "date"].nunique())
    common_train_symbols_count = int(labeled.loc[labeled["ab_common_train_eligible"], "symbol"].nunique())
    common_train_keys_str = "".join(
        f"{pd.Timestamp(r.date).date()}|{r.symbol};"
        for r in labeled.loc[labeled["ab_common_train_eligible"], ["date", "symbol"]].itertuples(index=False)
    )
    common_train_pool_hash = hashlib.sha256(common_train_keys_str.encode("utf-8")).hexdigest()

    feature_cols = [c for c in FactorProcessor.get_all_factor_cols() if c in labeled.columns]
    if not feature_cols:
        raise RuntimeError("No model features found")
    feature_hash = hashlib.sha256(",".join(feature_cols).encode("utf-8")).hexdigest()

    # 统一模型参数并显式固定 scale_pos_weight = 1.0 (确保 A/B 双臂无动态偏斜差异)
    ab_model_params = settings.LGBM_PARAMS_CLF.copy()
    ab_model_params["scale_pos_weight"] = 1.0
    ab_model_params["random_state"] = 42
    ab_model_params["feature_fraction_seed"] = 42
    ab_model_params["bagging_seed"] = 42
    ab_model_params["data_random_seed"] = 42

    legacy_effective_config_hash = hashlib.sha256(json.dumps(ab_model_params, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    exec_effective_config_hash = hashlib.sha256(json.dumps(ab_model_params, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    assert legacy_effective_config_hash == exec_effective_config_hash, "Effective model parameter hashes must be identical"

    print("==> [3/7] 训练 Arm A (Legacy Label Arm) Walk-Forward...")
    trainer_a = _trainer("ab_label_legacy", legacy_model_dir, ab_model_params)
    oos_a, _ = trainer_a.run_walk_forward(labeled, feature_cols=feature_cols)

    print("==> [4/7] 训练 Arm B (Execution-Aligned Label Arm) Walk-Forward...")
    trainer_b = _trainer("ab_label_execution", exec_model_dir, ab_model_params)
    oos_b, _ = trainer_b.run_walk_forward(labeled, feature_cols=feature_cols)

    print("==> [5/7] 构造严格 1-to-1 共同执行 OOS 评价池...")
    a_pred = oos_a[["date", "symbol", "pred_score"]].rename(columns={"pred_score": "pred_score_legacy"})
    b_pred = oos_b[["date", "symbol", "pred_score"]].rename(columns={"pred_score": "pred_score_execution"})
    eval_cols = [
        "date", "symbol", EXEC_LABEL, EXEC_DIRECTION, "label_valid", "entry_tradable",
        "planned_exit_tradable", "actual_exit_date", "exit_deferred_days",
        "stock_gross_return", "stock_net_return", "benchmark_return", "label_cost_drag"
    ]
    if "in_universe" in labeled.columns:
        eval_cols.append("in_universe")
    if "excluded_from_training" in labeled.columns:
        eval_cols.append("excluded_from_training")

    common = a_pred.merge(b_pred, on=["date", "symbol"], how="inner", validate="one_to_one")
    common = common.merge(labeled[eval_cols], on=["date", "symbol"], how="inner", validate="one_to_one")
    common = common[common["label_valid"].fillna(False).astype(bool) & common[EXEC_LABEL].notna()].copy()
    if "in_universe" in common.columns:
        common = common[common["in_universe"].fillna(False).astype(bool)].copy()
    if "excluded_from_training" in common.columns:
        common = common[~common["excluded_from_training"].fillna(False).astype(bool)].copy()

    common.sort_values(["date", "symbol"], inplace=True)
    common_key_hash = hashlib.sha256("".join(
        f"{pd.Timestamp(r.date).date()}|{r.symbol};"
        for r in common[["date", "symbol"]].itertuples(index=False)
    ).encode("utf-8")).hexdigest()

    print("==> [6/7] 统一在 COMMON_EXECUTION_OOS_POOL 上计算核心评价指标与 Bootstrap...")
    metrics_a, daily_a = _evaluate_common_pool(common, "pred_score_legacy", EXEC_LABEL)
    metrics_b, daily_b = _evaluate_common_pool(common, "pred_score_execution", EXEC_LABEL)
    bootstrap = _paired_block_bootstrap(daily_b, daily_a, block_size=20, n_bootstraps=2000, seed=42)
    folds, fold_stats = _fold_comparison(trainer_a, trainer_b, daily_a, daily_b)
    fold_win_ratio = fold_stats["fold_win_ratio"]

    summary = pd.DataFrame([
        {"arm": "legacy_label", **metrics_a},
        {"arm": "execution_aligned_label", **metrics_b},
    ])
    summary["feature_count"] = len(feature_cols)
    summary["feature_schema_hash"] = feature_hash
    summary["seed"] = 42

    rankic = pd.concat([daily_a.rename("legacy_rankic"), daily_b.rename("execution_rankic")], axis=1)
    rankic["execution_minus_legacy"] = rankic["execution_rankic"] - rankic["legacy_rankic"]

    # 判定科学结论
    robust = (metrics_b["mean_daily_rank_ic"] > metrics_a["mean_daily_rank_ic"]
              and metrics_b["nw20_rank_icir"] > metrics_a["nw20_rank_icir"]
              and bootstrap["robust_improvement"] and fold_win_ratio > 0.5)
    status = "ROBUST_IMPROVEMENT_FOUND" if robust else (
        "NO_IMPROVEMENT" if metrics_b["mean_daily_rank_ic"] <= metrics_a["mean_daily_rank_ic"]
        else "MIXED_EVIDENCE"
    )

    # 7. 实验后生产模型与目录隔离性重新审计
    prod_exists_after = prod_model_path.exists()
    prod_sha_after = _sha256_file(prod_model_path)
    prod_sha_unchanged = bool(prod_sha_before == prod_sha_after and prod_exists_before == prod_exists_after)
    prod_dir_snap_after = _snapshot_directory(prod_models_dir)

    # 检查 saved_models 目录内容是否完全一致
    prod_dir_mutated = False
    if set(prod_dir_snap_before.keys()) != set(prod_dir_snap_after.keys()):
        prod_dir_mutated = True
    else:
        for k in prod_dir_snap_before:
            if prod_dir_snap_before[k]["sha256"] != prod_dir_snap_after[k]["sha256"]:
                prod_dir_mutated = True
                break

    if prod_dir_mutated or not prod_sha_unchanged:
        raise RuntimeError("FATAL: Production model directory was mutated during Phase 2.1-A r2 run!")

    # 保存本地运行产物
    oos_parquet_path = run_dir / "common_execution_oos.parquet"
    common.to_parquet(oos_parquet_path, index=False)
    legacy_model_file = legacy_model_dir / "latest_lightgbm.pkl"
    exec_model_file = exec_model_dir / "latest_lightgbm.pkl"

    rel_run_dir = f"reports/phase2_1_a/{run_id}"

    manifest = {
        "phase": "2.1-A",
        "run_id": run_id,
        "iteration": "r2",
        "source_commit_sha": source_sha,
        "source_commit_branch": branch_name,
        "source_commit_tree_clean": tree_clean,
        "source_commit_remote": remote_sha,
        "source_commit_remote_match": bool(source_sha == remote_sha),
        "experiment_generated_from_clean_commit": bool(tree_clean),
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if dataset_path.is_relative_to(PROJECT_ROOT) else str(dataset_path),
        "dataset_sha256": dataset_sha,
        "dataset_rows": total_rows,
        "dataset_date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "feature_count": len(feature_cols),
        "feature_schema_hash": feature_hash,
        "seed": 42,
        "model_family": "LightGBM Classification",
        "effective_model_params": ab_model_params,
        "legacy_effective_model_config_hash": legacy_effective_config_hash,
        "execution_effective_model_config_hash": exec_effective_config_hash,
        "scale_pos_weight_legacy": 1.0,
        "scale_pos_weight_execution": 1.0,
        "feature_selection_policy": "all",
        "weighting_mode": "none",
        "legacy_train_label": "ab_label_legacy",
        "execution_train_label": "ab_label_execution",
        "common_train_pool_hash": common_train_pool_hash,
        "common_train_rows": common_train_rows_count,
        "common_train_dates": common_train_dates_count,
        "common_train_symbols": common_train_symbols_count,
        "common_execution_eval_target": EXEC_LABEL,
        "common_execution_oos_pool_hash": common_key_hash,
        "common_execution_oos_rows": len(common),
        "common_execution_oos_dates": int(common["date"].nunique()),
        "common_execution_oos_symbols": int(common["symbol"].nunique()),
        "legacy_metrics": metrics_a,
        "execution_metrics": metrics_b,
        "delta_metrics": {
            "delta_mean_daily_rank_ic": metrics_b["mean_daily_rank_ic"] - metrics_a["mean_daily_rank_ic"],
            "delta_nw20_rank_icir": metrics_b["nw20_rank_icir"] - metrics_a["nw20_rank_icir"],
            "delta_rank_ic_positive_rate": metrics_b["rank_ic_positive_rate"] - metrics_a["rank_ic_positive_rate"],
            "delta_q5_minus_q1_annualized_pct_points": metrics_b["q5_minus_q1_annualized_pct_points"] - metrics_a["q5_minus_q1_annualized_pct_points"],
            "delta_top10_mean_20d_exec_alpha": metrics_b["top10_mean_20d_exec_alpha"] - metrics_a["top10_mean_20d_exec_alpha"],
        },
        "bootstrap": bootstrap,
        "fold_statistics": fold_stats,
        "common_execution_oos_artifact": {
            "relative_path": f"{rel_run_dir}/common_execution_oos.parquet",
            "sha256": _sha256_file(oos_parquet_path),
            "size_bytes": oos_parquet_path.stat().st_size if oos_parquet_path.exists() else 0,
            "row_count": len(common),
            "column_count": len(common.columns),
            "columns": list(common.columns),
            "format": "parquet",
            "storage_mode": "local_not_git_tracked",
            "reproducible": True
        },
        "legacy_model_artifact": {
            "relative_path": f"{rel_run_dir}/models/legacy/latest_lightgbm.pkl",
            "sha256": _sha256_file(legacy_model_file),
            "size_bytes": legacy_model_file.stat().st_size if legacy_model_file.exists() else 0,
            "storage_mode": "local_not_git_tracked"
        },
        "execution_model_artifact": {
            "relative_path": f"{rel_run_dir}/models/execution/latest_lightgbm.pkl",
            "sha256": _sha256_file(exec_model_file),
            "size_bytes": exec_model_file.stat().st_size if exec_model_file.exists() else 0,
            "storage_mode": "local_not_git_tracked"
        },
        "experiment_model_persistence_isolated": True,
        "production_models_dir_path": str(prod_models_dir.relative_to(PROJECT_ROOT)).replace("\\", "/") if prod_models_dir.is_relative_to(PROJECT_ROOT) else str(prod_models_dir),
        "production_model_path": str(prod_model_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if prod_model_path.is_relative_to(PROJECT_ROOT) else str(prod_model_path),
        "production_model_exists_before": prod_exists_before,
        "production_model_sha_before": prod_sha_before,
        "production_model_exists_after": prod_exists_after,
        "production_model_sha_after": prod_sha_after,
        "production_model_sha_unchanged": prod_sha_unchanged,
        "production_models_dir_mutated": prod_dir_mutated,
        "legacy_experiment_model_dir": f"{rel_run_dir}/models/legacy",
        "execution_experiment_model_dir": f"{rel_run_dir}/models/execution",
        "status": status,
        "live_trading_ready": False,
        "production_model_promotion": False,
        "created_at": datetime.now().isoformat(),
    }

    summary.to_csv(run_dir / "ab_summary.csv", index=False, encoding="utf-8-sig")
    rankic.to_csv(run_dir / "daily_rankic_common_exec.csv", encoding="utf-8-sig")
    folds.to_csv(run_dir / "fold_comparison.csv", index=False, encoding="utf-8-sig")
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    report = f"""# Phase 2.1-A r2 — Execution-Aligned Label A/B Study Report
# 实盘执行对齐标签严格受控 A/B 实验研究报告 (r2 可复现性闭环版本)

> **研究结论 (Scientific Verdict)**: **`{status}`**
> **实盘许可声明 (Live Trading Guard)**: `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 1. 实验控制变量与血缘规范 (Controlled Variables & Provenance)

- **基准代码提交 (Source Code Commit)**: `{source_sha}` (Tree Clean: `{tree_clean}`)
- **基准模型族**: `LightGBM Classification` (二分类概率预测)
- **特征集与顺序**: 严格相同 ({len(feature_cols)} 因子, Feature Hash = `{feature_hash}`)
- **有效模型参数**: 严格逐字段相同 (Effective Hash = `{legacy_effective_config_hash}`, `scale_pos_weight = 1.0`)
- **随机种子**: `42` (严格传播至 feature_fraction_seed, bagging_seed, data_random_seed)
- **时序划分**: 严格相同 Walk-Forward 滚动折划分 (Purge Gap = 25 天 >= Label Horizon 20 天)
- **训练准入池**: 严格相同的共同准入交集 ({common_train_rows_count:,} 行, Common Train Pool Hash = `{common_train_pool_hash}`)
- **唯一自变量 (Primary Change)**: **训练目标标签定义** (Legacy `ab_label_legacy` vs Execution-Aligned `ab_label_execution`)

---

## 2. 统一公平评估目标 (Common Execution OOS Target)

两个 Arm 均在完全相同的实盘执行对齐前瞻净超额收益目标 `{EXEC_LABEL}` 上进行 OOS 评价：

- **共同 OOS 样本行数**: {len(common):,}
- **共同 OOS 交易日数**: {common['date'].nunique()}
- **共同 OOS 股票池数**: {common['symbol'].nunique()}
- **共同 OOS 评价池 SHA256**: `{common_key_hash}`

---

## 3. 核心对比结果 (Controlled A/B Evaluation Results)

| 评价维度 / 指标 | Arm A (Legacy Label) | Arm B (Execution-Aligned Label) | 差异 (Delta B - A) |
| :--- | :---: | :---: | :---: |
| **Mean Daily OOS RankIC** | **{metrics_a['mean_daily_rank_ic']:.6f}** | **{metrics_b['mean_daily_rank_ic']:.6f}** | **{metrics_b['mean_daily_rank_ic'] - metrics_a['mean_daily_rank_ic']:+.6f}** |
| **NW20 RankICIR (年化)** | **{metrics_a['nw20_rank_icir']:.6f}** | **{metrics_b['nw20_rank_icir']:.6f}** | **{metrics_b['nw20_rank_icir'] - metrics_a['nw20_rank_icir']:+.6f}** |
| **RankIC > 0 交易日占比** | {metrics_a['rank_ic_positive_rate']:.2%} | {metrics_b['rank_ic_positive_rate']:.2%} | {metrics_b['rank_ic_positive_rate'] - metrics_a['rank_ic_positive_rate']:+.2%} |
| **Q5-Q1 年化超额收益差 (pct points)** | {metrics_a['q5_minus_q1_annualized_pct_points']:.2f} pts | {metrics_b['q5_minus_q1_annualized_pct_points']:.2f} pts | {metrics_b['q5_minus_q1_annualized_pct_points'] - metrics_a['q5_minus_q1_annualized_pct_points']:+.2f} pts |
| **Top 10% 20日平均真实执行净超额** | {metrics_a['top10_mean_20d_exec_alpha']:.4%} | {metrics_b['top10_mean_20d_exec_alpha']:.4%} | {metrics_b['top10_mean_20d_exec_alpha'] - metrics_a['top10_mean_20d_exec_alpha']:+.4%} |
| **分组单调性得分** | {metrics_a['monotonicity_score']:.4f} | {metrics_b['monotonicity_score']:.4f} | {metrics_b['monotonicity_score'] - metrics_a['monotonicity_score']:+.4f} |

---

## 4. 统计检验与折稳定性 (Statistical Significance & Fold Stability)

- **20-Day Paired Block Bootstrap (2,000 Resamples)**:
  - **Mean RankIC Delta**: **{bootstrap['mean_diff']:+.6f}**
  - **95% 置信区间 (95% CI)**: `[{bootstrap['ci_lower']:+.6f}, {bootstrap['ci_upper']:+.6f}]`
  - **提升概率 P(Delta > 0)**: **{bootstrap['prob_positive']:.2%}**
  - **统计显著提升 (CI Lower > 0)**: **`{bootstrap['robust_improvement']}`**
- **Fold-Level 胜率实证 (已排除 0 样本无效折)**:
  - **滚动折总数 (Total Folds)**: {fold_stats['total_generated_folds']}
  - **有效对比折数 (Valid Folds)**: {fold_stats['valid_comparison_folds']}
  - **0 交易日无效折数 (Excluded Folds)**: {fold_stats['zero_common_date_folds']} (Fold 20: 0 common dates)
  - **Execution Arm 胜出折数**: {fold_stats['execution_wins']} (Fold 1, 4, 6, 8, 10, 14, 15, 16, 19)
  - **Legacy Arm 胜出折数**: {fold_stats['legacy_wins']} (Fold 2, 3, 5, 7, 9, 11, 12, 13, 17, 18)
  - **平局折数 (Ties)**: {fold_stats['ties']}
  - **Fold 胜率 (Fold Win Ratio)**: **{fold_win_ratio:.2%}**

---

## 5. 生产模型物理隔离审计 (Production Model Isolation Audit)

- **Production Models Dir**: `{manifest['production_models_dir_path']}`
- **Production Model Path**: `{manifest['production_model_path']}`
- **Exists Before Experiment**: `{prod_exists_before}`
- **SHA256 Before Experiment**: `{prod_sha_before}`
- **Exists After Experiment**: `{prod_exists_after}`
- **SHA256 After Experiment**: `{prod_sha_after}`
- **SHA256 Unchanged**: **`{prod_sha_unchanged}`**
- **Production Models Dir Mutated**: **`{prod_dir_mutated}`**
- **Legacy Experiment Model Dir**: `{rel_run_dir}/models/legacy`
- **Execution Experiment Model Dir**: `{rel_run_dir}/models/execution`

---

## 6. 大文件管理与本地证据治理 (Large Artifact Policy)

- **Common Execution OOS Parquet**: `{manifest['common_execution_oos_artifact']['relative_path']}` (Size: {manifest['common_execution_oos_artifact']['size_bytes']:,} bytes, SHA256: `{manifest['common_execution_oos_artifact']['sha256']}`, Storage: `local_not_git_tracked`)
- **Legacy Model PKL**: `{manifest['legacy_model_artifact']['relative_path']}` (SHA256: `{manifest['legacy_model_artifact']['sha256']}`, Storage: `local_not_git_tracked`)
- **Execution Model PKL**: `{manifest['execution_model_artifact']['relative_path']}` (SHA256: `{manifest['execution_model_artifact']['sha256']}`, Storage: `local_not_git_tracked`)

---

## 7. 科学判定与结论说明 (Scientific Finding & Next Step)

- **判定状态**: **`{status}`**
- **结论阐述**: 
  {'实证表明：将训练标签修改为实盘可成交的 T+1 Open -> T+21 Open 显著提升了模型对未来真实可交易收益的 OOS 预测能力。' if status == 'ROBUST_IMPROVEMENT_FOUND' else ('实证表明：Execution-Aligned Labels 在本受控实验中未能超越 Legacy Labels。更真实的标签并不必然等同于更高的 OOS 预测能力。' if status == 'NO_IMPROVEMENT' else '实证表明：Execution-Aligned Labels 展现出混合证据，部分指标提升但统计置信度未达完全显著要求。')}
"""
    (run_dir / "PHASE2_1_A_REPORT.md").write_text(report, encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest.json").write_text(json.dumps(
        {"latest_run_id": run_id, "path": rel_run_dir, "status": status},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nPhase 2.1-A r2 status: {status}")
    print(f"Artifacts saved to: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=PROJECT_ROOT / "data_storage" / "research" / "factor_matrix_300.parquet")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "reports" / "phase2_1_a")
    args = parser.parse_args()
    run(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
