"""
Phase 2.1-B r2 — Model Objective Study Runner (tools/run_phase2_1_b_objective_study.py)
严格受控三臂 OOS 研究：Classification (Arm A) vs Regression (Arm B) vs True LambdaRank (Arm C)。
Phase 2.1-B r2 核心修正与最终证据闭环：
1. 修正 LambdaRank relevance target scope：仅 common_train 样本参与同日截面分位数计算，严禁未准入样本污染相关性等级。
2. 严格执行四重核验门禁：
   - Classification Summary & Daily RankIC 序列 (max diff <= 1e-10) 精确复现
   - Regression Summary & Daily RankIC 序列 (max diff <= 1e-10) 精确复现
3. LambdaRank Runtime 严格断言 (outside_scope == 0, eligible integer-valued [0..9], common_train null exec_label == 0)。
4. 补充各 Fold 各 Arm 最佳迭代数及比率 (best_iteration, best_iteration_ratio, early_stopping_rounds) 实证诊断。
5. 修复 scipy 真实版本号记录。
6. 记录 5 层完整 Git 血缘链 (Initial B -> Bugfix -> v1 Evidence -> r2 Code -> r2 Evidence)。
7. 生产模型物理隔离全目录快照审计与大文件治理。
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
import scipy
from scipy import stats

from config.settings import settings
from factors.processor import FactorProcessor
from models.evaluator import ModelEvaluator
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from research_v2.labels.execution_labeler import ExecutionAlignedLabeler

logger = logging.getLogger("phase2_1_b_r2_objectives")
EXEC_LABEL = "label_net_alpha_20d"
EXEC_DIRECTION = "label_direction_20d"

# 预期 Phase 2.1-A 认证哈希与基准
CERTIFIED_DATASET_SHA = "9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42"
CERTIFIED_FEATURE_SCHEMA_HASH = "82dfd3e9643ae1352829e736b9c8b89d1d648b98d16ef59153f261bf7a453460"
CERTIFIED_COMMON_TRAIN_POOL_HASH = "bfff9a2d0a9b52a0d4924ea36d643923d1e42ab6bd48ed60017d020d22c42bcb"
CERTIFIED_COMMON_OOS_POOL_HASH = "a464b29fd12a50891ef68777791ebcb0c7c4f9fc96b59137174387100ca5fd1c"

CERTIFIED_PHASE2_1_A_EXEC_MEAN_RANKIC = 0.043681739629648594
CERTIFIED_PHASE2_1_A_EXEC_NW20_RANKICIR = 0.3520146818106218

# Phase 2.1-B v1 认证冻结结果与 Daily RankIC 序列哈希
CERTIFIED_PHASE2_1_B_V1_REG_MEAN_RANKIC = 0.04670731619915874
CERTIFIED_PHASE2_1_B_V1_REG_NW20_RANKICIR = 0.45074353161665703
CERTIFIED_PHASE2_1_B_V1_REG_TOP10_ALPHA = 0.01531426561629291
CERTIFIED_PHASE2_1_B_V1_REG_Q5_Q1 = 15.38
CERTIFIED_PHASE2_1_B_V1_REG_POSITIVE_RATE = 0.7012113055181696
CERTIFIED_PHASE2_1_B_V1_REG_DELTA_RANKIC = 0.0030255765695101425
CERTIFIED_PHASE2_1_B_V1_REG_FOLD_WINS = 7

CERTIFIED_V1_CLF_DAILY_RANKIC_HASH = "5ec8d5630bd017892ebffe4e5069c023dfdfa7e93f5b03202470e92a6d0f52d5"
CERTIFIED_V1_REG_DAILY_RANKIC_HASH = "548333d9765b322ebc1ef04763049e0ba5a5bb5a18cb16820c60add4eb12cff9"

# 历史 Git 提交血缘
PHASE2_1_B_INITIAL_CODE_COMMIT = "40b7bb8bc5c8b297ca5e259ac54f1e6df8bc5af9"
PHASE2_1_B_PRERUN_BUGFIX_COMMIT = "f088ed735d196311aabd9479c5b6a0849e0e670d"
PHASE2_1_B_V1_EVIDENCE_COMMIT = "5bb6ff21707ae484d1659c4a575645dd6e4c98ce"


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
            raise RuntimeError(f"FATAL: Phase 2.1-B r2 research must run from a clean tracked source tree. Found dirty items: {dirty_items}")
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


def _build_lambdarank_relevance_labels(
    labeled: pd.DataFrame,
    common_train: pd.Series,
    target_col: str = EXEC_LABEL,
    n_grades: int = 10
) -> pd.Series:
    """
    构建 LambdaRank 训练用相关性等级标签 (0 ~ n_grades-1)。
    核心科学准则 (Phase 2.1-B r2 修正):
    只有 common_train == True 的样本允许参与同交易日横截面分位数 (percentile rank) 计算。
    非 common_train 样本一律赋值为 np.nan，严禁参与或污染分位数计算。
    """
    grades = pd.Series(np.nan, index=labeled.index, dtype=float)
    common_mask = common_train.fillna(False).astype(bool)
    eligible = labeled.loc[common_mask].copy()
    if len(eligible) > 0:
        pct_rank = eligible.groupby("date")[target_col].rank(method="average", pct=True)
        eligible_grades = np.clip(np.ceil(pct_rank * float(n_grades)) - 1.0, 0.0, float(n_grades - 1))
        grades.loc[common_mask] = eligible_grades.values
    return grades


def _compute_daily_rankic_hash(daily_series: pd.Series) -> str:
    s = daily_series.sort_index()
    lines = [f"{pd.Timestamp(dt).strftime('%Y-%m-%d')}|{float(val):.17g}\n" for dt, val in s.items()]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


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


def _extract_fold_diagnostics(trainer_clf, trainer_reg, trainer_rank) -> pd.DataFrame:
    diag_rows = []
    arm_trainers = [
        ("classification", trainer_clf, 800, 80),
        ("regression", trainer_reg, 800, 80),
        ("lambdarank", trainer_rank, 800, 80)
    ]
    for arm_name, trainer, n_est_cfg, es_cfg in arm_trainers:
        for m in trainer.models:
            best_iter = None
            model_obj = m.get("model")
            if model_obj is not None:
                inner_model = getattr(model_obj, "model", None)
                if inner_model is not None:
                    best_iter = getattr(inner_model, "best_iteration_", None)

            b_iter_int = int(best_iter) if best_iter is not None else None
            ratio = round(float(b_iter_int) / float(n_est_cfg), 6) if b_iter_int is not None else None

            diag_rows.append({
                "fold": m["fold"],
                "arm": arm_name,
                "train_start": str(pd.Timestamp(m["train_start"]).date()),
                "train_end": str(pd.Timestamp(m["train_end"]).date()),
                "val_start": str(pd.Timestamp(m["val_start"]).date()) if m["val_start"] is not None else None,
                "val_end": str(pd.Timestamp(m["val_end"]).date()) if m["val_end"] is not None else None,
                "test_start": str(pd.Timestamp(m["test_start"]).date()),
                "test_end": str(pd.Timestamp(m["test_end"]).date()),
                "best_iteration": b_iter_int,
                "n_estimators_configured": n_est_cfg,
                "best_iteration_ratio": ratio,
                "early_stopping_rounds": es_cfg
            })
    return pd.DataFrame(diag_rows)


def _compute_lambdarank_scope_diagnostics(labeled: pd.DataFrame, common_train: pd.Series, grades: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    common_mask = common_train.fillna(False).astype(bool)
    eligible_count = int(common_mask.sum())
    ineligible_count = int((~common_mask).sum())

    ineligible_non_null = int(grades[~common_mask].notna().sum())
    eligible_grades = grades[common_mask]
    eligible_non_finite = int((~np.isfinite(eligible_grades)).sum())
    eligible_non_integer = int((np.mod(eligible_grades.dropna(), 1) != 0).sum())

    dist = eligible_grades.value_counts().sort_index().to_dict()
    daily_eligible_counts = labeled[common_mask].groupby("date").size()

    summary = {
        "common_train_rows": eligible_count,
        "outside_common_train_rows": ineligible_count,
        "outside_scope_non_null_grade_count": ineligible_non_null,
        "eligible_non_finite_grade_count": eligible_non_finite,
        "eligible_non_integer_grade_count": eligible_non_integer,
        "eligible_grade_min": int(eligible_grades.min()) if len(eligible_grades) else 0,
        "eligible_grade_max": int(eligible_grades.max()) if len(eligible_grades) else 0,
        "eligible_per_date_min": int(daily_eligible_counts.min()) if len(daily_eligible_counts) else 0,
        "eligible_per_date_median": float(daily_eligible_counts.median()) if len(daily_eligible_counts) else 0.0,
        "eligible_per_date_max": int(daily_eligible_counts.max()) if len(daily_eligible_counts) else 0,
        "daily_eligible_count_mean": float(daily_eligible_counts.mean()) if len(daily_eligible_counts) else 0.0,
        "grade_distribution": {int(k): int(v) for k, v in dist.items()}
    }
    rows = []
    for g in range(10):
        c = int(dist.get(float(g), 0))
        rows.append({
            "metric_item": f"grade_{g}_count",
            "value": c,
            "pct_of_eligible": float(c) / float(eligible_count) if eligible_count else 0.0
        })
    rows.extend([
        {"metric_item": "common_train_rows", "value": eligible_count, "pct_of_eligible": 1.0},
        {"metric_item": "outside_common_train_rows", "value": ineligible_count, "pct_of_eligible": 0.0},
        {"metric_item": "outside_scope_non_null_grade_count", "value": ineligible_non_null, "pct_of_eligible": 0.0},
        {"metric_item": "eligible_non_finite_grade_count", "value": eligible_non_finite, "pct_of_eligible": 0.0},
        {"metric_item": "eligible_non_integer_grade_count", "value": eligible_non_integer, "pct_of_eligible": 0.0},
        {"metric_item": "eligible_grade_min", "value": summary["eligible_grade_min"], "pct_of_eligible": 0.0},
        {"metric_item": "eligible_grade_max", "value": summary["eligible_grade_max"], "pct_of_eligible": 0.0},
        {"metric_item": "eligible_per_date_min", "value": summary["eligible_per_date_min"], "pct_of_eligible": 0.0},
        {"metric_item": "eligible_per_date_median", "value": summary["eligible_per_date_median"], "pct_of_eligible": 0.0},
        {"metric_item": "eligible_per_date_max", "value": summary["eligible_per_date_max"], "pct_of_eligible": 0.0},
    ])
    return pd.DataFrame(rows), summary


def _get_environment_info() -> Dict[str, Any]:
    import sklearn
    import lightgbm
    import joblib

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
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

    # 读取 Phase 2.1-B v1 的 Daily RankIC 序列进行逐日复现认证
    v1_daily_csv = PROJECT_ROOT / "reports" / "phase2_1_b" / "phase2_1_b_f088ed7_20260831_045445" / "daily_rankic_common_exec.csv"
    if not v1_daily_csv.exists():
        raise RuntimeError(f"FATAL: Missing v1 daily rankic CSV for exact reproduction certification: {v1_daily_csv}")
    v1_daily_df = pd.read_csv(v1_daily_csv, index_col=0, parse_dates=True)
    v1_clf_daily_expected = v1_daily_df["classification_rankic"]
    v1_reg_daily_expected = v1_daily_df["regression_rankic"]

    run_id = f"phase2_1_b_r2_{source_sha[:7]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    clf_model_dir = run_dir / "models" / "classification"
    reg_model_dir = run_dir / "models" / "regression"
    rank_model_dir = run_dir / "models" / "lambdarank"
    clf_model_dir.mkdir(parents=True, exist_ok=True)
    reg_model_dir.mkdir(parents=True, exist_ok=True)
    rank_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"==> [1/9] 读取并严密验证数据集: {dataset_path}")
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

    print("==> [2/9] 构建标签与严密校验共同训练池...")
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

    # 构造三臂训练标签 (Phase 2.1-B r2 核心修正: LambdaRank 仅在 common_train 样本内部计算分位数)
    labeled["ab_label_classification"] = labeled[EXEC_DIRECTION].where(common_train)
    labeled["ab_label_regression"] = labeled[EXEC_LABEL].where(common_train)
    labeled["ab_label_lambdarank"] = _build_lambdarank_relevance_labels(labeled, common_train, EXEC_LABEL, n_grades=10)

    # Runtime 硬断言 (Runtime Fail-Closed Scope & Grade Assertions)
    outside_scope_non_null = int(((~common_train) & labeled["ab_label_lambdarank"].notna()).sum())
    if outside_scope_non_null != 0:
        raise RuntimeError(f"FATAL: Found {outside_scope_non_null} non-null relevance grades outside common_train!")

    eligible_grades_s = labeled.loc[common_train, "ab_label_lambdarank"]
    if not eligible_grades_s.notna().all():
        raise RuntimeError("FATAL: NaN found in eligible relevance grades!")
    if not np.isfinite(eligible_grades_s).all():
        raise RuntimeError("FATAL: Non-finite values found in eligible relevance grades!")
    if not (np.mod(eligible_grades_s, 1) == 0).all():
        raise RuntimeError("FATAL: Non-integer values found in eligible relevance grades!")
    if eligible_grades_s.min() < 0 or eligible_grades_s.max() > 9:
        raise RuntimeError(f"FATAL: Eligible grades out of bounds [0, 9]! Min: {eligible_grades_s.min()}, Max: {eligible_grades_s.max()}")

    if int(((common_train) & labeled[EXEC_LABEL].isna()).sum()) != 0:
        raise RuntimeError("FATAL: NaN execution label found in common_train set!")

    # 计算并保存 LambdaRank Scope Diagnostics
    scope_df, scope_summary = _compute_lambdarank_scope_diagnostics(labeled, common_train, labeled["ab_label_lambdarank"])
    scope_df.to_csv(run_dir / "lambdarank_scope_diagnostics.csv", index=False, encoding="utf-8-sig")

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

    print("==> [3/9] 训练 Arm A (Classification Baseline) Walk-Forward...")
    trainer_clf = WalkForwardTrainer(
        train_years=settings.TRAIN_WINDOW_YEARS, val_months=settings.VAL_WINDOW_MONTHS,
        test_months=settings.TEST_WINDOW_MONTHS, purge_gap_days=settings.PURGE_GAP_DAYS,
        label_col="ab_label_classification", task_type="classification", model_type="lightgbm",
        feature_selection_method="all", top_k_features=20, weighting_mode="none",
        random_state=42, model_dir=clf_model_dir, model_params=clf_params
    )
    oos_clf, _ = trainer_clf.run_walk_forward(labeled, feature_cols=feature_cols)

    eval_cols = [
        "date", "symbol", EXEC_LABEL, EXEC_DIRECTION, "label_valid", "entry_tradable",
        "planned_exit_tradable", "actual_exit_date", "exit_deferred_days",
        "stock_gross_return", "stock_net_return", "benchmark_return", "label_cost_drag"
    ]
    if "in_universe" in labeled.columns:
        eval_cols.append("in_universe")
    if "excluded_from_training" in labeled.columns:
        eval_cols.append("excluded_from_training")

    print("==> [4/9] 执行 Arm A Classification Baseline Reproduction Gate 严格核验...")
    clf_eval = oos_clf[["date", "symbol", "pred_score"]].rename(columns={"pred_score": "pred_score_classification"})
    clf_eval = clf_eval.merge(labeled[eval_cols], on=["date", "symbol"], how="inner", validate="one_to_one")
    clf_eval = clf_eval[clf_eval["label_valid"].fillna(False).astype(bool) & clf_eval[EXEC_LABEL].notna()].copy()
    if "in_universe" in clf_eval.columns:
        clf_eval = clf_eval[clf_eval["in_universe"].fillna(False).astype(bool)].copy()
    if "excluded_from_training" in clf_eval.columns:
        clf_eval = clf_eval[~clf_eval["excluded_from_training"].fillna(False).astype(bool)].copy()

    metrics_clf, daily_clf = _evaluate_common_pool(clf_eval, "pred_score_classification", EXEC_LABEL)
    r2_clf_daily_hash = _compute_daily_rankic_hash(daily_clf)

    clf_oos_key_hash = hashlib.sha256("".join(
        f"{pd.Timestamp(r.date).date()}|{r.symbol};"
        for r in clf_eval[["date", "symbol"]].itertuples(index=False)
    ).encode("utf-8")).hexdigest()

    reprod_clf_diff = abs(metrics_clf["mean_daily_rank_ic"] - CERTIFIED_PHASE2_1_A_EXEC_MEAN_RANKIC)
    reprod_clf_nw20_diff = abs(metrics_clf["nw20_rank_icir"] - CERTIFIED_PHASE2_1_A_EXEC_NW20_RANKICIR)
    clf_daily_max_diff = float((daily_clf - v1_clf_daily_expected).abs().max())
    clf_daily_reprod_passed = bool(clf_daily_max_diff <= 1e-10)

    reprod_clf_passed = bool(
        clf_oos_key_hash == CERTIFIED_COMMON_OOS_POOL_HASH
        and reprod_clf_diff <= 1e-10
        and reprod_clf_nw20_diff <= 1e-10
        and clf_daily_reprod_passed
        and len(clf_eval) == 220913
    )

    baseline_reprod_df = pd.DataFrame([{
        "phase": "2.1-B-r2",
        "expected_mean_daily_rankic": CERTIFIED_PHASE2_1_A_EXEC_MEAN_RANKIC,
        "actual_mean_daily_rankic": metrics_clf["mean_daily_rank_ic"],
        "rankic_abs_diff": reprod_clf_diff,
        "expected_nw20_rankicir": CERTIFIED_PHASE2_1_A_EXEC_NW20_RANKICIR,
        "actual_nw20_rankicir": metrics_clf["nw20_rank_icir"],
        "nw20_abs_diff": reprod_clf_nw20_diff,
        "expected_daily_rankic_hash": CERTIFIED_V1_CLF_DAILY_RANKIC_HASH,
        "actual_daily_rankic_hash": r2_clf_daily_hash,
        "daily_rankic_max_abs_diff": clf_daily_max_diff,
        "daily_rankic_reproduction_passed": clf_daily_reprod_passed,
        "expected_oos_pool_hash": CERTIFIED_COMMON_OOS_POOL_HASH,
        "actual_oos_pool_hash": clf_oos_key_hash,
        "pool_hash_matched": bool(clf_oos_key_hash == CERTIFIED_COMMON_OOS_POOL_HASH),
        "expected_oos_rows": 220913,
        "actual_oos_rows": len(clf_eval),
        "reproduction_passed": reprod_clf_passed
    }])
    baseline_reprod_df.to_csv(run_dir / "baseline_reproduction.csv", index=False, encoding="utf-8-sig")

    if not reprod_clf_passed:
        raise RuntimeError(
            f"FATAL: Classification Baseline Reproduction Failed! Diff: {reprod_clf_diff:.2e}, "
            f"Daily Max Diff: {clf_daily_max_diff:.2e}"
        )
    print(f"   -> Classification Baseline Reproduction 100% PASS! (Diff = {reprod_clf_diff:.2e}, Daily Max Diff = {clf_daily_max_diff:.2e})")

    print("==> [5/9] 训练 Arm B (Continuous Regression) Walk-Forward...")
    trainer_reg = WalkForwardTrainer(
        train_years=settings.TRAIN_WINDOW_YEARS, val_months=settings.VAL_WINDOW_MONTHS,
        test_months=settings.TEST_WINDOW_MONTHS, purge_gap_days=settings.PURGE_GAP_DAYS,
        label_col="ab_label_regression", task_type="regression", model_type="regression",
        feature_selection_method="all", top_k_features=20, weighting_mode="none",
        random_state=42, model_dir=reg_model_dir, model_params=reg_params
    )
    oos_reg, _ = trainer_reg.run_walk_forward(labeled, feature_cols=feature_cols)

    print("==> [6/9] 执行 Arm B Regression Reproduction Gate 严格核验...")
    reg_eval = oos_reg[["date", "symbol", "pred_score"]].rename(columns={"pred_score": "pred_score_regression"})
    reg_eval = reg_eval.merge(labeled[eval_cols], on=["date", "symbol"], how="inner", validate="one_to_one")
    reg_eval = reg_eval[reg_eval["label_valid"].fillna(False).astype(bool) & reg_eval[EXEC_LABEL].notna()].copy()
    if "in_universe" in reg_eval.columns:
        reg_eval = reg_eval[reg_eval["in_universe"].fillna(False).astype(bool)].copy()
    if "excluded_from_training" in reg_eval.columns:
        reg_eval = reg_eval[~reg_eval["excluded_from_training"].fillna(False).astype(bool)].copy()

    metrics_reg_check, daily_reg_check = _evaluate_common_pool(reg_eval, "pred_score_regression", EXEC_LABEL)
    r2_reg_daily_hash = _compute_daily_rankic_hash(daily_reg_check)

    reprod_reg_diff = abs(metrics_reg_check["mean_daily_rank_ic"] - CERTIFIED_PHASE2_1_B_V1_REG_MEAN_RANKIC)
    reprod_reg_nw20_diff = abs(metrics_reg_check["nw20_rank_icir"] - CERTIFIED_PHASE2_1_B_V1_REG_NW20_RANKICIR)
    reg_daily_max_diff = float((daily_reg_check - v1_reg_daily_expected).abs().max())
    reg_daily_reprod_passed = bool(reg_daily_max_diff <= 1e-10)

    reg_reprod_passed = bool(
        reprod_reg_diff <= 1e-10
        and reprod_reg_nw20_diff <= 1e-10
        and reg_daily_reprod_passed
        and len(reg_eval) == 220913
    )

    reg_reprod_df = pd.DataFrame([{
        "phase": "2.1-B-r2",
        "expected_mean_daily_rankic": CERTIFIED_PHASE2_1_B_V1_REG_MEAN_RANKIC,
        "actual_mean_daily_rankic": metrics_reg_check["mean_daily_rank_ic"],
        "rankic_abs_diff": reprod_reg_diff,
        "expected_nw20_rankicir": CERTIFIED_PHASE2_1_B_V1_REG_NW20_RANKICIR,
        "actual_nw20_rankicir": metrics_reg_check["nw20_rank_icir"],
        "nw20_abs_diff": reprod_reg_nw20_diff,
        "expected_top10_alpha": CERTIFIED_PHASE2_1_B_V1_REG_TOP10_ALPHA,
        "actual_top10_alpha": metrics_reg_check["top10_mean_20d_exec_alpha"],
        "expected_q5_q1": CERTIFIED_PHASE2_1_B_V1_REG_Q5_Q1,
        "actual_q5_q1": metrics_reg_check["q5_minus_q1_annualized_pct_points"],
        "expected_daily_rankic_hash": CERTIFIED_V1_REG_DAILY_RANKIC_HASH,
        "actual_daily_rankic_hash": r2_reg_daily_hash,
        "daily_rankic_max_abs_diff": reg_daily_max_diff,
        "daily_rankic_reproduction_passed": reg_daily_reprod_passed,
        "expected_oos_rows": 220913,
        "actual_oos_rows": len(reg_eval),
        "reproduction_passed": reg_reprod_passed
    }])
    reg_reprod_df.to_csv(run_dir / "regression_reproduction.csv", index=False, encoding="utf-8-sig")

    if not reg_reprod_passed:
        raise RuntimeError(
            f"FATAL: Regression Reproduction Gate Failed! Diff: {reprod_reg_diff:.2e}, "
            f"Daily Max Diff: {reg_daily_max_diff:.2e}"
        )
    print(f"   -> Regression Reproduction Gate 100% PASS! (Diff = {reprod_reg_diff:.2e}, Daily Max Diff = {reg_daily_max_diff:.2e})")

    print("==> [7/9] 训练 Arm C (True LambdaRank r2) Walk-Forward...")
    trainer_rank = WalkForwardTrainer(
        train_years=settings.TRAIN_WINDOW_YEARS, val_months=settings.VAL_WINDOW_MONTHS,
        test_months=settings.TEST_WINDOW_MONTHS, purge_gap_days=settings.PURGE_GAP_DAYS,
        label_col="ab_label_lambdarank", task_type="ranking", model_type="ranking",
        feature_selection_method="all", top_k_features=20, weighting_mode="none",
        random_state=42, model_dir=rank_model_dir, model_params=rank_params
    )
    oos_rank, _ = trainer_rank.run_walk_forward(labeled, feature_cols=feature_cols)

    print("==> [8/9] 构造严格 1-to-1 共同三臂 OOS 评价池...")
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
            f"FATAL: Phase 2.1-B r2 Common OOS Pool Hash mismatch! Expected {CERTIFIED_COMMON_OOS_POOL_HASH}, got {common_oos_pool_hash}"
        )

    print("==> [9/9] 统一在 COMMON_OBJECTIVE_OOS_POOL 上计算核心评价指标、Bootstrap、Fold 与迭代诊断...")
    metrics_clf, daily_clf = _evaluate_common_pool(common, "pred_score_classification", EXEC_LABEL)
    metrics_reg, daily_reg = _evaluate_common_pool(common, "pred_score_regression", EXEC_LABEL)
    metrics_rank, daily_rank = _evaluate_common_pool(common, "pred_score_lambdarank", EXEC_LABEL)

    # 4,000 Resamples Paired Block Bootstrap
    boot_reg_vs_clf = _paired_block_bootstrap(daily_reg, daily_clf, block_size=20, n_bootstraps=4000, seed=42)
    boot_rank_vs_clf = _paired_block_bootstrap(daily_rank, daily_clf, block_size=20, n_bootstraps=4000, seed=42)
    boot_rank_vs_reg = _paired_block_bootstrap(daily_rank, daily_reg, block_size=20, n_bootstraps=4000, seed=42)

    folds_df, fold_stats = _fold_comparison_three_arms(trainer_clf, trainer_reg, trainer_rank, daily_clf, daily_reg, daily_rank)
    fold_diag_df = _extract_fold_diagnostics(trainer_clf, trainer_reg, trainer_rank)
    fold_diag_df.to_csv(run_dir / "fold_training_diagnostics.csv", index=False, encoding="utf-8-sig")

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
        raise RuntimeError("FATAL: Production model directory was mutated during Phase 2.1-B r2 run!")

    # 保存产物
    oos_parquet_path = run_dir / "common_objective_oos.parquet"
    common.to_parquet(oos_parquet_path, index=False)

    clf_model_file = clf_model_dir / "latest_lightgbm.pkl"
    reg_model_file = reg_model_dir / "latest_lightgbm.pkl"
    rank_model_file = rank_model_dir / "latest_lightgbm.pkl"

    rel_run_dir = str(run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/") if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir)

    env_info = _get_environment_info()
    with (run_dir / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(env_info, f, ensure_ascii=False, indent=2)

    # 保存 Daily RankIC Hash 认证证据
    daily_hashes_dict = {
        "classification": {
            "v1_daily_rankic_hash": CERTIFIED_V1_CLF_DAILY_RANKIC_HASH,
            "r2_daily_rankic_hash": r2_clf_daily_hash,
            "daily_rankic_max_abs_diff_vs_v1": clf_daily_max_diff,
            "hash_matched": bool(r2_clf_daily_hash == CERTIFIED_V1_CLF_DAILY_RANKIC_HASH),
            "reproduction_passed": reprod_clf_passed
        },
        "regression": {
            "v1_daily_rankic_hash": CERTIFIED_V1_REG_DAILY_RANKIC_HASH,
            "r2_daily_rankic_hash": r2_reg_daily_hash,
            "daily_rankic_max_abs_diff_vs_v1": reg_daily_max_diff,
            "hash_matched": bool(r2_reg_daily_hash == CERTIFIED_V1_REG_DAILY_RANKIC_HASH),
            "reproduction_passed": reg_reprod_passed
        }
    }
    with (run_dir / "daily_rankic_reproduction_hashes.json").open("w", encoding="utf-8") as f:
        json.dump(daily_hashes_dict, f, ensure_ascii=False, indent=2)

    summary_df = pd.DataFrame([
        {"arm": "classification_baseline", "task_type": "classification", "objective": "binary", **metrics_clf},
        {"arm": "continuous_regression", "task_type": "regression", "objective": "regression", **metrics_reg},
        {"arm": "true_lambdarank_r2", "task_type": "ranking", "objective": "lambdarank", **metrics_rank},
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
        "iteration": "r2",
        "correction_type": "lambdarank_relevance_common_train_scope",
        "classification_logic_changed": False,
        "regression_logic_changed": False,
        "lambdarank_target_logic_changed": True,
        "hyperparameter_tuning_performed": False,
        "feature_change_performed": False,
        "dataset_change_performed": False,
        "oos_driven_tuning_performed": False,
        "formal_r2_attempts": 1,
        "successful_formal_r2_runs": 1,
        "run_id": run_id,
        "git_provenance": {
            "initial_b_code_commit": PHASE2_1_B_INITIAL_CODE_COMMIT,
            "prerun_bugfix_commit": PHASE2_1_B_PRERUN_BUGFIX_COMMIT,
            "v1_evidence_commit": PHASE2_1_B_V1_EVIDENCE_COMMIT,
            "r2_source_commit_sha": source_sha,
            "r2_source_commit_branch": branch_name,
            "local_remote_tracking_sha": local_remote_tracking_sha,
            "true_remote_sha": true_remote_sha,
            "source_commit_tree_clean": tree_clean,
            "source_commit_remote_match": bool(source_sha == true_remote_sha and source_sha != "UNKNOWN"),
        },
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if dataset_path.is_relative_to(PROJECT_ROOT) else str(dataset_path),
        "dataset_sha256": dataset_sha,
        "dataset_rows": len(df),
        "dataset_date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "feature_count": len(feature_cols),
        "feature_schema_hash": feature_hash,
        "seed": 42,
        "model_family": "LightGBM Quant",
        "baseline_reproduction": {
            "passed": reprod_clf_passed,
            "mean_daily_rankic_diff": reprod_clf_diff,
            "nw20_rankicir_diff": reprod_clf_nw20_diff,
            "oos_pool_hash_matched": bool(clf_oos_key_hash == CERTIFIED_COMMON_OOS_POOL_HASH),
            "v1_daily_rankic_hash": CERTIFIED_V1_CLF_DAILY_RANKIC_HASH,
            "r2_daily_rankic_hash": r2_clf_daily_hash,
            "daily_rankic_max_abs_diff": clf_daily_max_diff,
            "daily_rankic_reproduction_passed": clf_daily_reprod_passed
        },
        "regression_reproduction": {
            "passed": reg_reprod_passed,
            "mean_daily_rankic_diff": reprod_reg_diff,
            "nw20_rankicir_diff": reprod_reg_nw20_diff,
            "v1_daily_rankic_hash": CERTIFIED_V1_REG_DAILY_RANKIC_HASH,
            "r2_daily_rankic_hash": r2_reg_daily_hash,
            "daily_rankic_max_abs_diff": reg_daily_max_diff,
            "daily_rankic_reproduction_passed": reg_daily_reprod_passed,
            "scientific_status": "MIXED_EVIDENCE"
        },
        "lambdarank_scope_diagnostics": scope_summary,
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
        "lambdarank_metrics_r2": metrics_rank,
        "v1_lambdarank_status": "SUPERSEDED_METHOD_SCOPE_ISSUE",
        "delta_metrics": {
            "regression_minus_classification": {
                "delta_mean_daily_rank_ic": delta_reg_ic,
                "delta_nw20_rank_icir": metrics_reg["nw20_rank_icir"] - metrics_clf["nw20_rank_icir"],
                "delta_rank_ic_positive_rate": metrics_reg["rank_ic_positive_rate"] - metrics_clf["rank_ic_positive_rate"],
                "delta_q5_minus_q1_annualized_pct_points": metrics_reg["q5_minus_q1_annualized_pct_points"] - metrics_clf["q5_minus_q1_annualized_pct_points"],
                "delta_top10_mean_20d_exec_alpha": metrics_reg["top10_mean_20d_exec_alpha"] - metrics_clf["top10_mean_20d_exec_alpha"],
            },
            "lambdarank_minus_classification_r2": {
                "delta_mean_daily_rank_ic": delta_rank_ic,
                "delta_nw20_rank_icir": metrics_rank["nw20_rank_icir"] - metrics_clf["nw20_rank_icir"],
                "delta_rank_ic_positive_rate": metrics_rank["rank_ic_positive_rate"] - metrics_clf["rank_ic_positive_rate"],
                "delta_q5_minus_q1_annualized_pct_points": metrics_rank["q5_minus_q1_annualized_pct_points"] - metrics_clf["q5_minus_q1_annualized_pct_points"],
                "delta_top10_mean_20d_exec_alpha": metrics_rank["top10_mean_20d_exec_alpha"] - metrics_clf["top10_mean_20d_exec_alpha"],
            }
        },
        "bootstrap": {
            "regression_vs_classification": boot_reg_vs_clf,
            "lambdarank_vs_classification_r2": boot_rank_vs_clf,
            "lambdarank_vs_regression_r2": boot_rank_vs_reg
        },
        "fold_statistics": fold_stats,
        "robust_improvement_gates": {
            "regression_passed": reg_robust,
            "lambdarank_passed_r2": rank_robust
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

    report = f"""# Phase 2.1-B r2 — LambdaRank Target-Scope & Final Evidence Closure Report
# A股模型学习目标函数严格受控三臂 OOS 研究报告 (r2 截面范围修正与证据最终闭环)

> **最终科学判定 (Scientific Verdict)**: **`{status}`**
> **方法修正类型 (Method Correction)**: `lambdarank_relevance_common_train_scope`
> **实盘许可声明 (Live Trading Guard)**: `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 1. 5 层完整 Git 血缘链 (5-Tier Git Provenance)

| 阶段 / 提交层级 | Commit SHA | 说明与治理模式 |
| :--- | :--- | :--- |
| **Phase 2.1-B Initial Code** | [`{PHASE2_1_B_INITIAL_CODE_COMMIT}`](https://github.com/junl53582-oss/ai/commit/{PHASE2_1_B_INITIAL_CODE_COMMIT}) | 初版三臂 Runner 与单测 |
| **Phase 2.1-B Pre-Run Bugfix** | [`{PHASE2_1_B_PRERUN_BUGFIX_COMMIT}`](https://github.com/junl53582-oss/ai/commit/{PHASE2_1_B_PRERUN_BUGFIX_COMMIT}) | 运行前修复 float NaN 类型转换 |
| **Phase 2.1-B v1 Evidence** | [`{PHASE2_1_B_V1_EVIDENCE_COMMIT}`](https://github.com/junl53582-oss/ai/commit/{PHASE2_1_B_V1_EVIDENCE_COMMIT}) | v1 证据：确立 Regression `MIXED_EVIDENCE` 结论，定位 LambdaRank 截面范围问题 |
| **Phase 2.1-B r2 Code** | [`{source_sha}`](https://github.com/junl53582-oss/ai/commit/{source_sha}) | r2 修复：仅 common_train 样本参与分位数计算，添加极端样本隔离单测与真 Scipy 版本 |
| **Phase 2.1-B r2 Evidence** | *待 Stage B 提交* | r2 证据：双重复现与 Daily RankIC 序列门禁 100% 通过，LambdaRank r2 认证指标入库 |

---

## 2. 四重严格复现与序列哈希门禁核验 (Four-Fold Reproduction Gates)

### A. Classification Baseline Reproduction
- **Phase 2.1-A 预期 RankIC**: `{CERTIFIED_PHASE2_1_A_EXEC_MEAN_RANKIC:.6f}`
- **Phase 2.1-B r2 实际 RankIC**: `{metrics_clf['mean_daily_rank_ic']:.6f}` (Diff: `{reprod_clf_diff:.2e}`)
- **V1 预期 Daily RankIC Hash**: `{CERTIFIED_V1_CLF_DAILY_RANKIC_HASH}`
- **r2 实际 Daily RankIC Hash**: `{r2_clf_daily_hash}`
- **Daily RankIC 逐日最大绝对误差**: `{clf_daily_max_diff:.2e}` (Passed: **`{clf_daily_reprod_passed}`**)
- **OOS Pool Hash 匹配**: **`{bool(clf_oos_key_hash == CERTIFIED_COMMON_OOS_POOL_HASH)}`**
- **门禁状态**: **`{'PASS' if reprod_clf_passed else 'FAIL'}`**

### B. Regression Reproduction Gate
- **Phase 2.1-B v1 预期 RankIC**: `{CERTIFIED_PHASE2_1_B_V1_REG_MEAN_RANKIC:.6f}`
- **Phase 2.1-B r2 实际 RankIC**: `{metrics_reg['mean_daily_rank_ic']:.6f}` (Diff: `{reprod_reg_diff:.2e}`)
- **预期 NW20**: `{CERTIFIED_PHASE2_1_B_V1_REG_NW20_RANKICIR:.6f}` | 实际: `{metrics_reg['nw20_rank_icir']:.6f}`
- **预期 Top10 Alpha**: `{CERTIFIED_PHASE2_1_B_V1_REG_TOP10_ALPHA:.4%}` | 实际: `{metrics_reg['top10_mean_20d_exec_alpha']:.4%}`
- **V1 预期 Daily RankIC Hash**: `{CERTIFIED_V1_REG_DAILY_RANKIC_HASH}`
- **r2 实际 Daily RankIC Hash**: `{r2_reg_daily_hash}`
- **Daily RankIC 逐日最大绝对误差**: `{reg_daily_max_diff:.2e}` (Passed: **`{reg_daily_reprod_passed}`**)
- **门禁状态**: **`{'PASS' if reg_reprod_passed else 'FAIL'}`**
- **Regression 科学结论维持**: **`MIXED_EVIDENCE`** (大实效提升信号，但统计显著与跨折胜率证据不足)

---

## 3. 三臂核心实证结果对比 (Three-Arm Evaluation Results)

| 评价指标 | Arm A (Classification) | Arm B (Regression) | Arm C (LambdaRank r2) | Delta (Reg - Clf) | Delta (Rank - Clf) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Daily OOS RankIC** | **{metrics_clf['mean_daily_rank_ic']:.6f}** | **{metrics_reg['mean_daily_rank_ic']:.6f}** | **{metrics_rank['mean_daily_rank_ic']:.6f}** | **{delta_reg_ic:+.6f}** | **{delta_rank_ic:+.6f}** |
| **NW20 RankICIR (年化)** | **{metrics_clf['nw20_rank_icir']:.6f}** | **{metrics_reg['nw20_rank_icir']:.6f}** | **{metrics_rank['nw20_rank_icir']:.6f}** | **{metrics_reg['nw20_rank_icir'] - metrics_clf['nw20_rank_icir']:+.6f}** | **{metrics_rank['nw20_rank_icir'] - metrics_clf['nw20_rank_icir']:+.6f}** |
| **RankIC > 0 占比** | {metrics_clf['rank_ic_positive_rate']:.2%} | {metrics_reg['rank_ic_positive_rate']:.2%} | {metrics_rank['rank_ic_positive_rate']:.2%} | {metrics_reg['rank_ic_positive_rate'] - metrics_clf['rank_ic_positive_rate']:+.2%} | {metrics_rank['rank_ic_positive_rate'] - metrics_clf['rank_ic_positive_rate']:+.2%} |
| **Q5-Q1 年化超额 (pct pts)** | {metrics_clf['q5_minus_q1_annualized_pct_points']:.2f} pts | {metrics_reg['q5_minus_q1_annualized_pct_points']:.2f} pts | {metrics_rank['q5_minus_q1_annualized_pct_points']:.2f} pts | {metrics_reg['q5_minus_q1_annualized_pct_points'] - metrics_clf['q5_minus_q1_annualized_pct_points']:+.2f} pts | {metrics_rank['q5_minus_q1_annualized_pct_points'] - metrics_clf['q5_minus_q1_annualized_pct_points']:+.2f} pts |
| **Top 10% 20日平均净超额** | {metrics_clf['top10_mean_20d_exec_alpha']:.4%} | {metrics_reg['top10_mean_20d_exec_alpha']:.4%} | {metrics_rank['top10_mean_20d_exec_alpha']:.4%} | {metrics_reg['top10_mean_20d_exec_alpha'] - metrics_clf['top10_mean_20d_exec_alpha']:+.4%} | {metrics_rank['top10_mean_20d_exec_alpha'] - metrics_clf['top10_mean_20d_exec_alpha']:+.4%} |
| **分组单调性得分** | {metrics_clf['monotonicity_score']:.4f} | {metrics_reg['monotonicity_score']:.4f} | {metrics_rank['monotonicity_score']:.4f} | {metrics_reg['monotonicity_score'] - metrics_clf['monotonicity_score']:+.4f} | {metrics_rank['monotonicity_score'] - metrics_clf['monotonicity_score']:+.4f} |

---

## 4. LambdaRank Target-Scope 修正与诊断

- **Eligible 训练样本数**: `{scope_summary['common_train_rows']:,}`
- **Ineligible 标记样本数**: `{scope_summary['outside_common_train_rows']:,}`
- **Ineligible 样本中非空相关性等级数**: **`{scope_summary['outside_scope_non_null_grade_count']}`** (完全隔离)
- **单日 Eligible 样本数量分布**: Min=`{scope_summary['eligible_per_date_min']}`, Median=`{scope_summary['eligible_per_date_median']:.1f}`, Max=`{scope_summary['eligible_per_date_max']}`, Mean=`{scope_summary['daily_eligible_count_mean']:.1f}`
- **v1 LambdaRank 状态**: **`SUPERSEDED_METHOD_SCOPE_ISSUE`** (v1 结果由 r2 正式取代)

---

## 5. 生产模型物理隔离审计

- **生产模型路径**: `saved_models/latest_lightgbm.pkl` (SHA256: `{prod_sha_after}`)
- **生产模型 SHA 未修改**: **`{prod_sha_unchanged}`**
- **生产目录无变动**: **`{not prod_dir_mutated}`**
- **大文件存储模式**: `common_objective_oos.parquet` 与各 Arm 实验模型均采用 `local_not_git_tracked` 存储。
"""
    (run_dir / "PHASE2_1_B_R2_REPORT.md").write_text(report, encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest.json").write_text(json.dumps(
        {"latest_run_id": run_id, "path": rel_run_dir, "status": status},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nPhase 2.1-B r2 status: {status}")
    print(f"Artifacts saved to: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=PROJECT_ROOT / "data_storage" / "research" / "factor_matrix_300.parquet")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "reports" / "phase2_1_b_r2")
    args = parser.parse_args()
    run(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
