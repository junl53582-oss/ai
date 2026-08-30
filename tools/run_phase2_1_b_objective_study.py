"""
Phase 2.1-B — Model Objective Study Runner (tools/run_phase2_1_b_objective_study.py)
严格受控三臂 OOS 研究：Classification (Arm A) vs Regression (Arm B) vs True LambdaRank (Arm C)。
包含：
1. 真实远程 SHA (git ls-remote) 与干净工作树硬门禁
2. 冻结数据集 SHA、特征顺序敏感哈希与共同训练池哈希硬校验
3. Arm A Classification 基准复现门禁 (严格校验与 Phase 2.1-A Execution Arm 数值一致)
4. 共享模型超参数严格对齐 (seed=42, learning_rate=0.02, num_leaves=63, max_depth=8, etc.)
5. True LambdaRank 10 档相关性等级构造与横截面分组校验
6. 严格 1-to-1 共同执行 OOS 评价池
7. 20-Day 配对块 Bootstrap (4,000 resamples, 95% & 97.5% CI) 与有效 Fold 统计 (排除 0-date fold)
8. 生产模型物理隔离全目录快照审计与大文件治理
"""
from __future__ import annotations

import sys
import argparse
import hashlib
import json
import logging
import platform
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

logger = logging.getLogger("phase2_1_b_objectives")
EXEC_LABEL = "label_net_alpha_20d"
EXEC_DIRECTION = "label_direction_20d"

# 预期 Phase 2.1-A 认证哈希
CERTIFIED_DATASET_SHA = "9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42"
CERTIFIED_FEATURE_SCHEMA_HASH = "82dfd3e9643ae1352829e736b9c8b89d1d648b98d16ef59153f261bf7a453460"
CERTIFIED_COMMON_TRAIN_POOL_HASH = "bfff9a2d0a9b52a0d4924ea36d643923d1e42ab6bd48ed60017d020d22c42bcb"
CERTIFIED_COMMON_OOS_POOL_HASH = "a464b29fd12a50891ef68777791ebcb0c7c4f9fc96b59137174387100ca5fd1c"
CERTIFIED_PHASE2_1_A_EXEC_MEAN_RANKIC = 0.043681739629648594
CERTIFIED_PHASE2_1_A_EXEC_NW20_RANKICIR = 0.3520146818106218


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


def _git_true_remote_sha(branch: str, cwd: Optional[Path] = None) -> str:
    try:
        res = subprocess.run(
            ["git", "ls-remote", "--heads", "origin", branch],
            cwd=cwd or PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        if not res:
            return "UNKNOWN"
        lines = [l.strip() for l in res.splitlines() if l.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                sha = parts[0].strip()
                ref = parts[1].strip()
                if ref in (f"refs/heads/{branch}", branch) and len(sha) == 40 and all(c in "0123456789abcdefABCDEF" for c in sha):
                    return sha.lower()
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _git_working_tree_clean(project_root: Optional[Path] = None) -> Tuple[bool, List[str]]:
    root = project_root or PROJECT_ROOT
    try:
        res = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                             cwd=root, capture_output=True, text=True, check=True).stdout
        if not res or not res.strip():
            return True, []

        dirty_items: List[str] = []
        for raw_line in res.splitlines():
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            if len(line) < 3:
                dirty_items.append(f"MALFORMED_STATUS_LINE: {line}")
                continue

            status_code = line[:2]
            item_path = line[3:].strip().strip('"')

            if status_code == "??":
                check_res = subprocess.run(["git", "check-ignore", "-q", item_path], cwd=root)
                if check_res.returncode != 0:
                    dirty_items.append(f"UNTRACKED: {item_path}")
            else:
                dirty_items.append(f"MODIFIED/STAGED ({status_code}): {item_path}")

        return len(dirty_items) == 0, dirty_items
    except Exception as e:
        return False, [f"ERROR: {e}"]


def _validate_source_provenance(enforce_clean: bool = True, project_root: Optional[Path] = None) -> Dict[str, Any]:
    root = project_root or PROJECT_ROOT
    source_sha = _git_sha(root)
    branch_name = _git_branch(root)
    local_remote_tracking_sha = _git_remote_sha(branch_name, root) if branch_name != "UNKNOWN" else "UNKNOWN"
    true_remote_sha = _git_true_remote_sha(branch_name, root) if branch_name != "UNKNOWN" else "UNKNOWN"
    is_clean, dirty_items = _git_working_tree_clean(root)

    if enforce_clean:
        if source_sha == "UNKNOWN":
            raise RuntimeError("FATAL: Unable to resolve HEAD source commit SHA.")
        if branch_name == "UNKNOWN":
            raise RuntimeError("FATAL: Unable to resolve current Git branch name.")
        if true_remote_sha == "UNKNOWN":
            raise RuntimeError(f"FATAL: Unable to resolve true remote branch SHA for 'refs/heads/{branch_name}' via git ls-remote.")
        if not is_clean:
            raise RuntimeError(f"FATAL: Phase 2.1-B research must run from a clean tracked source tree. Found dirty items: {dirty_items}")
        if source_sha != true_remote_sha:
            raise RuntimeError(f"FATAL: Local HEAD ({source_sha}) does not match true remote origin/{branch_name} ({true_remote_sha}).")

    return {
        "source_commit_sha": source_sha,
        "source_commit_branch": branch_name,
        "local_remote_tracking_sha": local_remote_tracking_sha,
        "true_remote_sha": true_remote_sha,
        "source_commit_tree_clean": is_clean,
        "source_commit_remote_match": bool(source_sha == true_remote_sha and source_sha != "UNKNOWN"),
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
                            block_size: int = 20, n_bootstraps: int = 4000,
                            seed: int = 42) -> Dict[str, Any]:
    common = candidate.index.intersection(baseline.index)
    diff = (candidate.loc[common] - baseline.loc[common]).dropna().to_numpy(dtype=float)
    if len(diff) < max(20, block_size):
        return {
            "common_dates": int(len(diff)),
            "mean_diff": float(np.mean(diff)) if len(diff) else 0.0,
            "ci_95_lower": 0.0, "ci_95_upper": 0.0,
            "ci_97_5_lower": 0.0, "ci_97_5_upper": 0.0,
            "prob_positive": 0.0,
            "robust_improvement_97_5": False
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

    ci_95_lower = float(np.percentile(boot, 2.5))
    ci_95_upper = float(np.percentile(boot, 97.5))
    ci_97_5_lower = float(np.percentile(boot, 1.25))
    ci_97_5_upper = float(np.percentile(boot, 98.75))
    prob_pos = float((boot > 0).mean())
    robust = bool(ci_97_5_lower > 0.0)

    return {
        "common_dates": int(n),
        "mean_diff": float(diff.mean()),
        "ci_95_lower": ci_95_lower,
        "ci_95_upper": ci_95_upper,
        "ci_97_5_lower": ci_97_5_lower,
        "ci_97_5_upper": ci_97_5_upper,
        "prob_positive": prob_pos,
        "robust_improvement_97_5": robust
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


def _fold_comparison_three_arms(trainer_clf, trainer_reg, trainer_rank,
                                daily_clf, daily_reg, daily_rank) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    n = min(len(trainer_clf.models), len(trainer_reg.models), len(trainer_rank.models))
    for i in range(n):
        clf_m = trainer_clf.models[i]
        reg_m = trainer_reg.models[i]
        rank_m = trainer_rank.models[i]

        start = pd.Timestamp(max(clf_m["test_start"], reg_m["test_start"], rank_m["test_start"]))
        end = pd.Timestamp(min(clf_m["test_end"], reg_m["test_end"], rank_m["test_end"]))

        clf_f = daily_clf[(daily_clf.index >= start) & (daily_clf.index <= end)]
        reg_f = daily_reg[(daily_reg.index >= start) & (daily_reg.index <= end)]
        rank_f = daily_rank[(daily_rank.index >= start) & (daily_rank.index <= end)]

        idx = clf_f.index.intersection(reg_f.index).intersection(rank_f.index)
        has_dates = len(idx) > 0

        clf_mean = float(clf_f.loc[idx].mean()) if has_dates else np.nan
        reg_mean = float(reg_f.loc[idx].mean()) if has_dates else np.nan
        rank_mean = float(rank_f.loc[idx].mean()) if has_dates else np.nan

        is_valid = has_dates and np.isfinite(clf_mean) and np.isfinite(reg_mean) and np.isfinite(rank_mean)

        reg_diff = reg_mean - clf_mean if is_valid else np.nan
        rank_diff = rank_mean - clf_mean if is_valid else np.nan

        reg_wins = bool(reg_mean > clf_mean) if is_valid else False
        rank_wins = bool(rank_mean > clf_mean) if is_valid else False

        rows.append({
            "fold": i + 1,
            "test_start": str(start.date()),
            "test_end": str(end.date()),
            "common_rankic_dates": int(len(idx)),
            "valid_comparison": bool(is_valid),
            "clf_mean_rankic": clf_mean,
            "reg_mean_rankic": reg_mean,
            "rank_mean_rankic": rank_mean,
            "reg_minus_clf": reg_diff,
            "rank_minus_clf": rank_diff,
            "reg_wins_vs_clf": reg_wins,
            "rank_wins_vs_clf": rank_wins,
        })

    folds_df = pd.DataFrame(rows)
    valid_folds = folds_df[folds_df["valid_comparison"]].copy()
    zero_date_folds = folds_df[~folds_df["valid_comparison"]].copy()

    reg_win_ratio = float(valid_folds["reg_wins_vs_clf"].mean()) if not valid_folds.empty else 0.0
    rank_win_ratio = float(valid_folds["rank_wins_vs_clf"].mean()) if not valid_folds.empty else 0.0

    stats_summary = {
        "total_generated_folds": int(len(folds_df)),
        "valid_comparison_folds": int(len(valid_folds)),
        "zero_common_date_folds": int(len(zero_date_folds)),
        "regression_wins_vs_clf": int(valid_folds["reg_wins_vs_clf"].sum()),
        "lambdarank_wins_vs_clf": int(valid_folds["rank_wins_vs_clf"].sum()),
        "regression_fold_win_ratio": reg_win_ratio,
        "lambdarank_fold_win_ratio": rank_win_ratio,
        "regression_best_fold_delta": float(valid_folds["reg_minus_clf"].max()) if not valid_folds.empty else 0.0,
        "regression_worst_fold_delta": float(valid_folds["reg_minus_clf"].min()) if not valid_folds.empty else 0.0,
        "lambdarank_best_fold_delta": float(valid_folds["rank_minus_clf"].max()) if not valid_folds.empty else 0.0,
        "lambdarank_worst_fold_delta": float(valid_folds["rank_minus_clf"].min()) if not valid_folds.empty else 0.0,
    }
    return folds_df, stats_summary


def _get_environment_info() -> Dict[str, Any]:
    import sklearn
    import lightgbm
    import joblib

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": stats.__name__,
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lightgbm.__version__,
        "joblib_version": joblib.__version__,
        "reproducibility_statement": "reproducible under compatible pinned environment"
    }


def run(dataset_path: Path, output_dir: Path) -> Path:
    # 0. 源码血缘与工作树硬门禁校验 (Fail-Closed True Remote Provenance Gate)
    provenance = _validate_source_provenance(enforce_clean=True)
    source_sha = provenance["source_commit_sha"]
    branch_name = provenance["source_commit_branch"]
    tree_clean = provenance["source_commit_tree_clean"]
    true_remote_sha = provenance["true_remote_sha"]
    local_remote_tracking_sha = provenance["local_remote_tracking_sha"]

    # 1. 实验前环境与生产模型全目录快照 (Path from settings.MODELS_DIR)
    prod_models_dir = Path(settings.MODELS_DIR)
    prod_model_path = prod_models_dir / "latest_lightgbm.pkl"
    prod_exists_before = prod_model_path.exists()
    prod_sha_before = _sha256_file(prod_model_path)
    prod_dir_snap_before = _snapshot_directory(prod_models_dir)

    run_id = f"phase2_1_b_{source_sha[:7]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    clf_model_dir = run_dir / "models" / "classification"
    reg_model_dir = run_dir / "models" / "regression"
    rank_model_dir = run_dir / "models" / "lambdarank"
    clf_model_dir.mkdir(parents=True, exist_ok=True)
    reg_model_dir.mkdir(parents=True, exist_ok=True)
    rank_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"==> [1/8] 读取并严密验证数据集: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["date", "symbol"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    dataset_sha = _sha256_file(dataset_path)

    if dataset_sha != CERTIFIED_DATASET_SHA:
        raise RuntimeError(f"FATAL: Dataset SHA mismatch! Expected {CERTIFIED_DATASET_SHA}, got {dataset_sha}")

    feature_cols = [c for c in FactorProcessor.get_all_factor_cols() if c in df.columns]
    feature_hash = hashlib.sha256(",".join(feature_cols).encode("utf-8")).hexdigest()
    if feature_hash != CERTIFIED_FEATURE_SCHEMA_HASH:
        raise RuntimeError(f"FATAL: Feature Schema Hash mismatch! Expected {CERTIFIED_FEATURE_SCHEMA_HASH}, got {feature_hash}")
    if len(feature_cols) != 79:
        raise RuntimeError(f"FATAL: Expected 79 features, got {len(feature_cols)}")

    print("==> [2/8] 构建标签与严密校验共同训练池...")
    legacy_labeler = TargetLabeler(
        horizon=settings.LABEL_HORIZON, task_type="classification",
        threshold=settings.LABEL_THRESHOLD, threshold_mode=settings.LABEL_THRESHOLD_MODE
    )
    labeled = legacy_labeler.compute_excess_return_label(df)
    labeled = ExecutionAlignedLabeler().compute(labeled)

    in_univ = labeled["in_universe"].fillna(False).astype(bool) if "in_universe" in labeled.columns else pd.Series(True, index=labeled.index)
    not_excl = ~labeled["excluded_from_training"].fillna(False).astype(bool) if "excluded_from_training" in labeled.columns else pd.Series(True, index=labeled.index)

    common_train = (
        labeled["label_valid"].fillna(False).astype(bool)
        & labeled[settings.LABEL_COLUMN_CLF].notna()
        & labeled[EXEC_DIRECTION].notna()
        & in_univ
        & not_excl
    )
    labeled["ab_common_train_eligible"] = common_train

    common_train_rows_count = int(common_train.sum())
    common_train_dates_count = int(labeled.loc[common_train, "date"].nunique())
    common_train_symbols_count = int(labeled.loc[common_train, "symbol"].nunique())
    common_train_keys_str = "".join(
        f"{pd.Timestamp(r.date).date()}|{r.symbol};"
        for r in labeled.loc[common_train, ["date", "symbol"]].itertuples(index=False)
    )
    common_train_pool_hash = hashlib.sha256(common_train_keys_str.encode("utf-8")).hexdigest()

    if common_train_pool_hash != CERTIFIED_COMMON_TRAIN_POOL_HASH:
        raise RuntimeError(
            f"FATAL: Common Train Pool Hash mismatch! Expected {CERTIFIED_COMMON_TRAIN_POOL_HASH}, got {common_train_pool_hash}"
        )

    # 构造三臂训练标签
    labeled["ab_label_classification"] = labeled[EXEC_DIRECTION].where(common_train)
    labeled["ab_label_regression"] = labeled[EXEC_LABEL].where(common_train)

    # Arm C 10档相关性等级构造 (严格同日截面，不跨日，0~9分位数等级)
    # relevance = clip(ceil(pct_rank * 10) - 1, 0, 9)
    pct_ranks = labeled.groupby("date")[EXEC_LABEL].rank(method="average", pct=True)
    relevance_grades = np.clip(np.ceil(pct_ranks * 10.0) - 1.0, 0.0, 9.0)
    labeled["ab_label_lambdarank"] = relevance_grades.where(common_train)

    # 3. 共享基础结构超参数
    base_lgbm_params = settings.LGBM_PARAMS_CLF.copy()
    base_lgbm_params["random_state"] = 42
    base_lgbm_params["feature_fraction_seed"] = 42
    base_lgbm_params["bagging_seed"] = 42
    base_lgbm_params["data_random_seed"] = 42
    base_lgbm_params["verbose"] = -1
    base_lgbm_params["n_jobs"] = -1

    # Arm A 参数 (Classification)
    clf_params = base_lgbm_params.copy()
    clf_params["objective"] = "binary"
    clf_params["metric"] = ["binary_logloss", "auc"]
    clf_params["scale_pos_weight"] = 1.0

    # Arm B 参数 (Regression)
    reg_params = base_lgbm_params.copy()
    reg_params["objective"] = "regression"
    reg_params["metric"] = ["l2", "rmse"]
    if "scale_pos_weight" in reg_params:
        del reg_params["scale_pos_weight"]

    # Arm C 参数 (LambdaRank)
    rank_params = base_lgbm_params.copy()
    rank_params["objective"] = "lambdarank"
    rank_params["metric"] = "ndcg"
    rank_params["eval_at"] = [30]
    rank_params["label_gain"] = list(range(10))
    if "scale_pos_weight" in rank_params:
        del rank_params["scale_pos_weight"]

    clf_config_hash = hashlib.sha256(json.dumps(clf_params, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    reg_config_hash = hashlib.sha256(json.dumps(reg_params, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    rank_config_hash = hashlib.sha256(json.dumps(rank_params, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    print("==> [3/8] 训练 Arm A (Classification Baseline) Walk-Forward...")
    trainer_clf = WalkForwardTrainer(
        train_years=settings.TRAIN_WINDOW_YEARS, val_months=settings.VAL_WINDOW_MONTHS,
        test_months=settings.TEST_WINDOW_MONTHS, purge_gap_days=settings.PURGE_GAP_DAYS,
        label_col="ab_label_classification", task_type="classification", model_type="lightgbm",
        feature_selection_method="all", top_k_features=20, weighting_mode="none",
        random_state=42, model_dir=clf_model_dir, model_params=clf_params
    )
    oos_clf, _ = trainer_clf.run_walk_forward(labeled, feature_cols=feature_cols)

    # 4. 执行 Baseline Reproduction Gate 校验
    print("==> [4/8] 执行 Arm A Classification Baseline Reproduction Gate 严格核验...")
    eval_cols = [
        "date", "symbol", EXEC_LABEL, EXEC_DIRECTION, "label_valid", "entry_tradable",
        "planned_exit_tradable", "actual_exit_date", "exit_deferred_days",
        "stock_gross_return", "stock_net_return", "benchmark_return", "label_cost_drag"
    ]
    if "in_universe" in labeled.columns:
        eval_cols.append("in_universe")
    if "excluded_from_training" in labeled.columns:
        eval_cols.append("excluded_from_training")

    clf_eval = oos_clf[["date", "symbol", "pred_score"]].rename(columns={"pred_score": "pred_score_classification"})
    clf_eval = clf_eval.merge(labeled[eval_cols], on=["date", "symbol"], how="inner", validate="one_to_one")
    clf_eval = clf_eval[clf_eval["label_valid"].fillna(False).astype(bool) & clf_eval[EXEC_LABEL].notna()].copy()
    if "in_universe" in clf_eval.columns:
        clf_eval = clf_eval[clf_eval["in_universe"].fillna(False).astype(bool)].copy()
    if "excluded_from_training" in clf_eval.columns:
        clf_eval = clf_eval[~clf_eval["excluded_from_training"].fillna(False).astype(bool)].copy()

    metrics_clf, daily_clf = _evaluate_common_pool(clf_eval, "pred_score_classification", EXEC_LABEL)

    clf_oos_key_hash = hashlib.sha256("".join(
        f"{pd.Timestamp(r.date).date()}|{r.symbol};"
        for r in clf_eval[["date", "symbol"]].itertuples(index=False)
    ).encode("utf-8")).hexdigest()

    reprod_metrics_diff = abs(metrics_clf["mean_daily_rank_ic"] - CERTIFIED_PHASE2_1_A_EXEC_MEAN_RANKIC)
    reprod_nw20_diff = abs(metrics_clf["nw20_rank_icir"] - CERTIFIED_PHASE2_1_A_EXEC_NW20_RANKICIR)

    reprod_passed = bool(
        clf_oos_key_hash == CERTIFIED_COMMON_OOS_POOL_HASH
        and reprod_metrics_diff <= 1e-6
        and reprod_nw20_diff <= 1e-6
        and len(clf_eval) == 220913
    )

    baseline_reprod_df = pd.DataFrame([{
        "phase": "2.1-B",
        "expected_mean_daily_rankic": CERTIFIED_PHASE2_1_A_EXEC_MEAN_RANKIC,
        "actual_mean_daily_rankic": metrics_clf["mean_daily_rank_ic"],
        "rankic_abs_diff": reprod_metrics_diff,
        "expected_nw20_rankicir": CERTIFIED_PHASE2_1_A_EXEC_NW20_RANKICIR,
        "actual_nw20_rankicir": metrics_clf["nw20_rank_icir"],
        "nw20_abs_diff": reprod_nw20_diff,
        "expected_oos_pool_hash": CERTIFIED_COMMON_OOS_POOL_HASH,
        "actual_oos_pool_hash": clf_oos_key_hash,
        "pool_hash_matched": bool(clf_oos_key_hash == CERTIFIED_COMMON_OOS_POOL_HASH),
        "expected_oos_rows": 220913,
        "actual_oos_rows": len(clf_eval),
        "reproduction_passed": reprod_passed
    }])
    baseline_reprod_df.to_csv(run_dir / "baseline_reproduction.csv", index=False, encoding="utf-8-sig")

    if not reprod_passed:
        raise RuntimeError(
            f"FATAL: Classification Baseline Reproduction Failed! Diff: {reprod_metrics_diff:.2e}, "
            f"Pool Hash Match: {bool(clf_oos_key_hash == CERTIFIED_COMMON_OOS_POOL_HASH)}"
        )
    print(f"   -> Classification Baseline Reproduction 100% PASS! (Diff = {reprod_metrics_diff:.2e})")

    print("==> [5/8] 训练 Arm B (Continuous Regression) Walk-Forward...")
    trainer_reg = WalkForwardTrainer(
        train_years=settings.TRAIN_WINDOW_YEARS, val_months=settings.VAL_WINDOW_MONTHS,
        test_months=settings.TEST_WINDOW_MONTHS, purge_gap_days=settings.PURGE_GAP_DAYS,
        label_col="ab_label_regression", task_type="regression", model_type="regression",
        feature_selection_method="all", top_k_features=20, weighting_mode="none",
        random_state=42, model_dir=reg_model_dir, model_params=reg_params
    )
    oos_reg, _ = trainer_reg.run_walk_forward(labeled, feature_cols=feature_cols)

    print("==> [6/8] 训练 Arm C (True LambdaRank) Walk-Forward...")
    trainer_rank = WalkForwardTrainer(
        train_years=settings.TRAIN_WINDOW_YEARS, val_months=settings.VAL_WINDOW_MONTHS,
        test_months=settings.TEST_WINDOW_MONTHS, purge_gap_days=settings.PURGE_GAP_DAYS,
        label_col="ab_label_lambdarank", task_type="ranking", model_type="ranking",
        feature_selection_method="all", top_k_features=20, weighting_mode="none",
        random_state=42, model_dir=rank_model_dir, model_params=rank_params
    )
    oos_rank, _ = trainer_rank.run_walk_forward(labeled, feature_cols=feature_cols)

    print("==> [7/8] 构造严格 1-to-1 共同三臂 OOS 评价池...")
    a_pred = oos_clf[["date", "symbol", "pred_score"]].rename(columns={"pred_score": "pred_score_classification"})
    b_pred = oos_reg[["date", "symbol", "pred_score"]].rename(columns={"pred_score": "pred_score_regression"})
    c_pred = oos_rank[["date", "symbol", "pred_score"]].rename(columns={"pred_score": "pred_score_lambdarank"})

    common = a_pred.merge(b_pred, on=["date", "symbol"], how="inner", validate="one_to_one")
    common = common.merge(c_pred, on=["date", "symbol"], how="inner", validate="one_to_one")
    common = common.merge(labeled[eval_cols], on=["date", "symbol"], how="inner", validate="one_to_one")
    common = common[common["label_valid"].fillna(False).astype(bool) & common[EXEC_LABEL].notna()].copy()
    if "in_universe" in common.columns:
        common = common[common["in_universe"].fillna(False).astype(bool)].copy()
    if "excluded_from_training" in common.columns:
        common = common[~common["excluded_from_training"].fillna(False).astype(bool)].copy()

    common.sort_values(["date", "symbol"], inplace=True)
    common_oos_pool_hash = hashlib.sha256("".join(
        f"{pd.Timestamp(r.date).date()}|{r.symbol};"
        for r in common[["date", "symbol"]].itertuples(index=False)
    ).encode("utf-8")).hexdigest()

    if common_oos_pool_hash != CERTIFIED_COMMON_OOS_POOL_HASH:
        raise RuntimeError(
            f"FATAL: Phase 2.1-B Common OOS Pool Hash mismatch! Expected {CERTIFIED_COMMON_OOS_POOL_HASH}, got {common_oos_pool_hash}"
        )

    print("==> [8/8] 统一在 COMMON_OBJECTIVE_OOS_POOL 上计算核心评价指标、Bootstrap 与折稳定性...")
    metrics_clf, daily_clf = _evaluate_common_pool(common, "pred_score_classification", EXEC_LABEL)
    metrics_reg, daily_reg = _evaluate_common_pool(common, "pred_score_regression", EXEC_LABEL)
    metrics_rank, daily_rank = _evaluate_common_pool(common, "pred_score_lambdarank", EXEC_LABEL)

    # 4,000 Resamples Paired Block Bootstrap
    boot_reg_vs_clf = _paired_block_bootstrap(daily_reg, daily_clf, block_size=20, n_bootstraps=4000, seed=42)
    boot_rank_vs_clf = _paired_block_bootstrap(daily_rank, daily_clf, block_size=20, n_bootstraps=4000, seed=42)
    boot_rank_vs_reg = _paired_block_bootstrap(daily_rank, daily_reg, block_size=20, n_bootstraps=4000, seed=42)

    folds_df, fold_stats = _fold_comparison_three_arms(trainer_clf, trainer_reg, trainer_rank, daily_clf, daily_reg, daily_rank)

    # Delta 指标计算
    delta_reg_ic = metrics_reg["mean_daily_rank_ic"] - metrics_clf["mean_daily_rank_ic"]
    delta_rank_ic = metrics_rank["mean_daily_rank_ic"] - metrics_clf["mean_daily_rank_ic"]

    # 预注册 Robust Improvement 门禁判定 (+0.0020 practical effect threshold & 97.5% CI lower > 0)
    reg_robust = bool(
        delta_reg_ic >= 0.0020
        and metrics_reg["nw20_rank_icir"] > metrics_clf["nw20_rank_icir"]
        and boot_reg_vs_clf["robust_improvement_97_5"]
        and fold_stats["regression_fold_win_ratio"] > 0.50
        and metrics_reg["top10_mean_20d_exec_alpha"] >= metrics_clf["top10_mean_20d_exec_alpha"]
    )
    rank_robust = bool(
        delta_rank_ic >= 0.0020
        and metrics_rank["nw20_rank_icir"] > metrics_clf["nw20_rank_icir"]
        and boot_rank_vs_clf["robust_improvement_97_5"]
        and fold_stats["lambdarank_fold_win_ratio"] > 0.50
        and metrics_rank["top10_mean_20d_exec_alpha"] >= metrics_clf["top10_mean_20d_exec_alpha"]
    )

    if reg_robust or rank_robust:
        status = "ROBUST_OBJECTIVE_IMPROVEMENT_FOUND"
    elif delta_reg_ic > 0 or delta_rank_ic > 0 or (metrics_rank["top10_mean_20d_exec_alpha"] > metrics_clf["top10_mean_20d_exec_alpha"]):
        status = "MIXED_EVIDENCE"
    else:
        status = "NO_IMPROVEMENT"

    # 生产模型隔离审计
    prod_exists_after = prod_model_path.exists()
    prod_sha_after = _sha256_file(prod_model_path)
    prod_sha_unchanged = bool(prod_sha_before == prod_sha_after and prod_exists_before == prod_exists_after)
    prod_dir_snap_after = _snapshot_directory(prod_models_dir)

    prod_dir_mutated = False
    if set(prod_dir_snap_before.keys()) != set(prod_dir_snap_after.keys()):
        prod_dir_mutated = True
    else:
        for k in prod_dir_snap_before:
            if prod_dir_snap_before[k]["sha256"] != prod_dir_snap_after[k]["sha256"]:
                prod_dir_mutated = True
                break

    if prod_dir_mutated or not prod_sha_unchanged:
        raise RuntimeError("FATAL: Production model directory was mutated during Phase 2.1-B run!")

    # 保存产物
    oos_parquet_path = run_dir / "common_objective_oos.parquet"
    common.to_parquet(oos_parquet_path, index=False)

    clf_model_file = clf_model_dir / "latest_lightgbm.pkl"
    reg_model_file = reg_model_dir / "latest_lightgbm.pkl"
    rank_model_file = rank_model_dir / "latest_lightgbm.pkl"

    rel_run_dir = f"reports/phase2_1_b/{run_id}"

    env_info = _get_environment_info()
    with (run_dir / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(env_info, f, ensure_ascii=False, indent=2)

    summary_df = pd.DataFrame([
        {"arm": "classification_baseline", "task_type": "classification", "objective": "binary", **metrics_clf},
        {"arm": "continuous_regression", "task_type": "regression", "objective": "regression", **metrics_reg},
        {"arm": "true_lambdarank", "task_type": "ranking", "objective": "lambdarank", **metrics_rank},
    ])
    summary_df["feature_count"] = len(feature_cols)
    summary_df["feature_schema_hash"] = feature_hash
    summary_df["seed"] = 42
    summary_df.to_csv(run_dir / "objective_summary.csv", index=False, encoding="utf-8-sig")

    rankic_df = pd.concat([
        daily_clf.rename("classification_rankic"),
        daily_reg.rename("regression_rankic"),
        daily_rank.rename("lambdarank_rankic")
    ], axis=1)
    rankic_df["reg_minus_clf"] = rankic_df["regression_rankic"] - rankic_df["classification_rankic"]
    rankic_df["rank_minus_clf"] = rankic_df["lambdarank_rankic"] - rankic_df["classification_rankic"]
    rankic_df.to_csv(run_dir / "daily_rankic_common_exec.csv", encoding="utf-8-sig")

    folds_df.to_csv(run_dir / "fold_comparison.csv", index=False, encoding="utf-8-sig")

    boot_df = pd.DataFrame([
        {"comparison": "regression_vs_classification", "type": "primary_candidate", **boot_reg_vs_clf},
        {"comparison": "lambdarank_vs_classification", "type": "primary_candidate", **boot_rank_vs_clf},
        {"comparison": "lambdarank_vs_regression", "type": "exploratory", **boot_rank_vs_reg},
    ])
    boot_df.to_csv(run_dir / "bootstrap_comparison.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "phase": "2.1-B",
        "run_id": run_id,
        "source_commit_sha": source_sha,
        "source_commit_branch": branch_name,
        "source_commit_tree_clean": tree_clean,
        "local_remote_tracking_sha": local_remote_tracking_sha,
        "true_remote_sha": true_remote_sha,
        "source_commit_remote": true_remote_sha,
        "source_commit_remote_match": bool(source_sha == true_remote_sha and source_sha != "UNKNOWN"),
        "experiment_generated_from_clean_commit": bool(tree_clean),
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if dataset_path.is_relative_to(PROJECT_ROOT) else str(dataset_path),
        "dataset_sha256": dataset_sha,
        "dataset_rows": len(df),
        "dataset_date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "feature_count": len(feature_cols),
        "feature_schema_hash": feature_hash,
        "seed": 42,
        "model_family": "LightGBM Quant",
        "baseline_reproduction": {
            "passed": reprod_passed,
            "mean_daily_rankic_diff": reprod_metrics_diff,
            "nw20_rankicir_diff": reprod_nw20_diff,
            "oos_pool_hash_matched": bool(clf_oos_key_hash == CERTIFIED_COMMON_OOS_POOL_HASH)
        },
        "effective_model_params": {
            "classification": clf_params,
            "regression": reg_params,
            "lambdarank": rank_params
        },
        "effective_model_config_hashes": {
            "classification": clf_config_hash,
            "regression": reg_config_hash,
            "lambdarank": rank_config_hash
        },
        "common_train_pool_hash": common_train_pool_hash,
        "common_train_rows": common_train_rows_count,
        "common_train_dates": common_train_dates_count,
        "common_train_symbols": common_train_symbols_count,
        "common_execution_eval_target": EXEC_LABEL,
        "common_execution_oos_pool_hash": common_oos_pool_hash,
        "common_execution_oos_rows": len(common),
        "common_execution_oos_dates": int(common["date"].nunique()),
        "common_execution_oos_symbols": int(common["symbol"].nunique()),
        "classification_metrics": metrics_clf,
        "regression_metrics": metrics_reg,
        "lambdarank_metrics": metrics_rank,
        "delta_metrics": {
            "regression_minus_classification": {
                "delta_mean_daily_rank_ic": delta_reg_ic,
                "delta_nw20_rank_icir": metrics_reg["nw20_rank_icir"] - metrics_clf["nw20_rank_icir"],
                "delta_rank_ic_positive_rate": metrics_reg["rank_ic_positive_rate"] - metrics_clf["rank_ic_positive_rate"],
                "delta_q5_minus_q1_annualized_pct_points": metrics_reg["q5_minus_q1_annualized_pct_points"] - metrics_clf["q5_minus_q1_annualized_pct_points"],
                "delta_top10_mean_20d_exec_alpha": metrics_reg["top10_mean_20d_exec_alpha"] - metrics_clf["top10_mean_20d_exec_alpha"],
            },
            "lambdarank_minus_classification": {
                "delta_mean_daily_rank_ic": delta_rank_ic,
                "delta_nw20_rank_icir": metrics_rank["nw20_rank_icir"] - metrics_clf["nw20_rank_icir"],
                "delta_rank_ic_positive_rate": metrics_rank["rank_ic_positive_rate"] - metrics_clf["rank_ic_positive_rate"],
                "delta_q5_minus_q1_annualized_pct_points": metrics_rank["q5_minus_q1_annualized_pct_points"] - metrics_clf["q5_minus_q1_annualized_pct_points"],
                "delta_top10_mean_20d_exec_alpha": metrics_rank["top10_mean_20d_exec_alpha"] - metrics_clf["top10_mean_20d_exec_alpha"],
            }
        },
        "bootstrap": {
            "regression_vs_classification": boot_reg_vs_clf,
            "lambdarank_vs_classification": boot_rank_vs_clf,
            "lambdarank_vs_regression": boot_rank_vs_reg
        },
        "fold_statistics": fold_stats,
        "robust_improvement_gates": {
            "regression_passed": reg_robust,
            "lambdarank_passed": rank_robust
        },
        "common_objective_oos_artifact": {
            "relative_path": f"{rel_run_dir}/common_objective_oos.parquet",
            "sha256": _sha256_file(oos_parquet_path),
            "size_bytes": oos_parquet_path.stat().st_size if oos_parquet_path.exists() else 0,
            "row_count": len(common),
            "column_count": len(common.columns),
            "columns": list(common.columns),
            "format": "parquet",
            "storage_mode": "local_not_git_tracked",
            "reproducible": True
        },
        "classification_model_artifact": {
            "relative_path": f"{rel_run_dir}/models/classification/latest_lightgbm.pkl",
            "sha256": _sha256_file(clf_model_file),
            "size_bytes": clf_model_file.stat().st_size if clf_model_file.exists() else 0,
            "storage_mode": "local_not_git_tracked"
        },
        "regression_model_artifact": {
            "relative_path": f"{rel_run_dir}/models/regression/latest_lightgbm.pkl",
            "sha256": _sha256_file(reg_model_file),
            "size_bytes": reg_model_file.stat().st_size if reg_model_file.exists() else 0,
            "storage_mode": "local_not_git_tracked"
        },
        "lambdarank_model_artifact": {
            "relative_path": f"{rel_run_dir}/models/lambdarank/latest_lightgbm.pkl",
            "sha256": _sha256_file(rank_model_file),
            "size_bytes": rank_model_file.stat().st_size if rank_model_file.exists() else 0,
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
        "status": status,
        "live_trading_ready": False,
        "production_model_promotion": False,
        "environment": env_info,
        "created_at": datetime.now().isoformat(),
    }

    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    report = f"""# Phase 2.1-B — Model Objective Study Report
# A股模型学习目标函数严格受控三臂 OOS 研究报告 (Classification vs Regression vs LambdaRank)

> **研究结论 (Scientific Verdict)**: **`{status}`**
> **实盘许可声明 (Live Trading Guard)**: `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 1. 实验控制变量与血缘规范 (Controlled Variables & Provenance)

- **基准代码提交 (Source Code Commit)**: `{source_sha}` (Tree Clean: `{tree_clean}`)
- **真实远程跟踪 SHA (True Remote SHA)**: `{true_remote_sha}`
- **基准特征集 (Feature Set)**: 严格相同 79 因子 (Feature Hash = `{feature_hash}`)
- **数据集哈希 (Dataset SHA256)**: `{dataset_sha}`
- **共同训练准入池 (Common Train Pool)**: 严格相同 ({common_train_rows_count:,} 行, Hash = `{common_train_pool_hash}`)
- **唯一自变量 (Primary Independent Variable)**: **模型学习目标函数 (Model Objective)**
  - **Arm A (Classification Baseline)**: Binary Logloss Classification (复现 Phase 2.1-A Execution Arm)
  - **Arm B (Continuous Regression)**: L2 / RMSE Regression on continuous `label_net_alpha_20d`
  - **Arm C (True LambdaRank)**: LightGBM LambdaRank on 10 relevance grades (0..9) with NDCG@30

---

## 2. 基准复现门禁核验 (Classification Baseline Reproduction)

- **Phase 2.1-A 预期 RankIC**: `{CERTIFIED_PHASE2_1_A_EXEC_MEAN_RANKIC:.6f}`
- **Phase 2.1-B 实际 RankIC**: `{metrics_clf['mean_daily_rank_ic']:.6f}` (Diff: `{reprod_metrics_diff:.2e}`)
- **OOS 评价池哈希匹配**: **`{bool(clf_oos_key_hash == CERTIFIED_COMMON_OOS_POOL_HASH)}`**
- **复现门禁状态**: **`{'PASS' if reprod_passed else 'FAIL'}`**

---

## 3. 三臂核心实证对比 (Three-Arm Evaluation Results)

| 评价指标 | Arm A (Classification) | Arm B (Regression) | Arm C (LambdaRank) | Delta (Reg - Clf) | Delta (Rank - Clf) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Daily OOS RankIC** | **{metrics_clf['mean_daily_rank_ic']:.6f}** | **{metrics_reg['mean_daily_rank_ic']:.6f}** | **{metrics_rank['mean_daily_rank_ic']:.6f}** | **{delta_reg_ic:+.6f}** | **{delta_rank_ic:+.6f}** |
| **NW20 RankICIR (年化)** | **{metrics_clf['nw20_rank_icir']:.6f}** | **{metrics_reg['nw20_rank_icir']:.6f}** | **{metrics_rank['nw20_rank_icir']:.6f}** | **{metrics_reg['nw20_rank_icir'] - metrics_clf['nw20_rank_icir']:+.6f}** | **{metrics_rank['nw20_rank_icir'] - metrics_clf['nw20_rank_icir']:+.6f}** |
| **RankIC > 0 占比** | {metrics_clf['rank_ic_positive_rate']:.2%} | {metrics_reg['rank_ic_positive_rate']:.2%} | {metrics_rank['rank_ic_positive_rate']:.2%} | {metrics_reg['rank_ic_positive_rate'] - metrics_clf['rank_ic_positive_rate']:+.2%} | {metrics_rank['rank_ic_positive_rate'] - metrics_clf['rank_ic_positive_rate']:+.2%} |
| **Q5-Q1 年化超额 (pct pts)** | {metrics_clf['q5_minus_q1_annualized_pct_points']:.2f} pts | {metrics_reg['q5_minus_q1_annualized_pct_points']:.2f} pts | {metrics_rank['q5_minus_q1_annualized_pct_points']:.2f} pts | {metrics_reg['q5_minus_q1_annualized_pct_points'] - metrics_clf['q5_minus_q1_annualized_pct_points']:+.2f} pts | {metrics_rank['q5_minus_q1_annualized_pct_points'] - metrics_clf['q5_minus_q1_annualized_pct_points']:+.2f} pts |
| **Top 10% 20日平均净超额** | {metrics_clf['top10_mean_20d_exec_alpha']:.4%} | {metrics_reg['top10_mean_20d_exec_alpha']:.4%} | {metrics_rank['top10_mean_20d_exec_alpha']:.4%} | {metrics_reg['top10_mean_20d_exec_alpha'] - metrics_clf['top10_mean_20d_exec_alpha']:+.4%} | {metrics_rank['top10_mean_20d_exec_alpha'] - metrics_clf['top10_mean_20d_exec_alpha']:+.4%} |
| **分组单调性得分** | {metrics_clf['monotonicity_score']:.4f} | {metrics_reg['monotonicity_score']:.4f} | {metrics_rank['monotonicity_score']:.4f} | {metrics_reg['monotonicity_score'] - metrics_clf['monotonicity_score']:+.4f} | {metrics_rank['monotonicity_score'] - metrics_clf['monotonicity_score']:+.4f} |

---

## 4. 统计检验与 Fold 胜率 (Paired Block Bootstrap & Fold Stability)

- **Regression vs Classification**:
  - Mean RankIC Delta: **`{boot_reg_vs_clf['mean_diff']:+.6f}`**
  - 95% 置信区间: `[{boot_reg_vs_clf['ci_95_lower']:+.6f}, {boot_reg_vs_clf['ci_95_upper']:+.6f}]`
  - 97.5% 置信区间 (保守门禁): `[{boot_reg_vs_clf['ci_97_5_lower']:+.6f}, {boot_reg_vs_clf['ci_97_5_upper']:+.6f}]`
  - 提升概率 P(Delta > 0): **`{boot_reg_vs_clf['prob_positive']:.2%}`**
  - 有效 Fold 胜率: **`{fold_stats['regression_fold_win_ratio']:.2%}`** ({fold_stats['regression_wins_vs_clf']}/{fold_stats['valid_comparison_folds']})
  - 达到 +0.0020 实效门禁: **`{bool(delta_reg_ic >= 0.0020)}`**
  - 满足全部 Robust Gate: **`{reg_robust}`**

- **LambdaRank vs Classification**:
  - Mean RankIC Delta: **`{boot_rank_vs_clf['mean_diff']:+.6f}`**
  - 95% 置信区间: `[{boot_rank_vs_clf['ci_95_lower']:+.6f}, {boot_rank_vs_clf['ci_95_upper']:+.6f}]`
  - 97.5% 置信区间 (保守门禁): `[{boot_rank_vs_clf['ci_97_5_lower']:+.6f}, {boot_rank_vs_clf['ci_97_5_upper']:+.6f}]`
  - 提升概率 P(Delta > 0): **`{boot_rank_vs_clf['prob_positive']:.2%}`**
  - 有效 Fold 胜率: **`{fold_stats['lambdarank_fold_win_ratio']:.2%}`** ({fold_stats['lambdarank_wins_vs_clf']}/{fold_stats['valid_comparison_folds']})
  - 达到 +0.0020 实效门禁: **`{bool(delta_rank_ic >= 0.0020)}`**
  - 满足全部 Robust Gate: **`{rank_robust}`**

---

## 5. 生产模型隔离与大文件治理

- **生产模型文件**: `saved_models/latest_lightgbm.pkl` (SHA256: `{prod_sha_after}`)
- **生产模型 SHA 未修改**: **`{prod_sha_unchanged}`**
- **生产目录无意外变动**: **`{not prod_dir_mutated}`**
- **大文件存储模式**: `common_objective_oos.parquet` 与各 Arm 实验模型均采用 `local_not_git_tracked` 存储。

---

## 6. 科学判定与结论说明 (Scientific Verdict)

- **判定状态**: **`{status}`**
"""
    (run_dir / "PHASE2_1_B_REPORT.md").write_text(report, encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest.json").write_text(json.dumps(
        {"latest_run_id": run_id, "path": rel_run_dir, "status": status},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nPhase 2.1-B status: {status}")
    print(f"Artifacts saved to: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=PROJECT_ROOT / "data_storage" / "research" / "factor_matrix_300.parquet")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "reports" / "phase2_1_b")
    args = parser.parse_args()
    run(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
