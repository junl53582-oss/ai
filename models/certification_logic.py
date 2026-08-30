"""
Phase 2.0.2r2 Final Certification Logic & Runtime State Derivation (models/certification_logic.py)
严格执行 Fail-Closed 运行时状态推导、真实指标计算与纯证据驱动多重门禁判定。
"""
import os
import re
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


def build_canonical_model_config(settings_obj=None) -> Dict[str, Any]:
    """从 settings 构建标准 canonical 模型研究配置字典"""
    if settings_obj is None:
        from config.settings import settings as current_settings
        settings_obj = current_settings

    return {
        "walk_forward": {
            "train_window_years": float(getattr(settings_obj, "TRAIN_WINDOW_YEARS", 1.5)),
            "val_window_months": int(getattr(settings_obj, "VAL_WINDOW_MONTHS", 3)),
            "test_window_months": int(getattr(settings_obj, "TEST_WINDOW_MONTHS", 2)),
            "purge_gap_days": int(getattr(settings_obj, "PURGE_GAP_DAYS", 25)),
        },
        "labeling": {
            "label_horizon": int(getattr(settings_obj, "LABEL_HORIZON", 20)),
            "label_threshold_mode": str(getattr(settings_obj, "LABEL_THRESHOLD_MODE", "cross_sectional_extreme")),
            "label_extreme_quantile": float(getattr(settings_obj, "LABEL_EXTREME_QUANTILE", 0.30)),
        },
        "models": {
            "lightgbm_clf_baseline": {
                "task_type": "classification",
                "model_type": "lightgbm",
                "feature_selection_method": "all",
                "weighting_mode": "none",
                "random_state": 42
            },
            "double_ensemble": {
                "task_type": "classification",
                "model_type": "double_ensemble",
                "feature_selection_method": "top_20",
                "weighting_mode": "recency_magnitude",
                "top_k_features": 20,
                "random_state": 42
            },
            "lightgbm_ranker": {
                "task_type": "ranking",
                "model_type": "lightgbm_ranker",
                "feature_selection_method": "rank_ic_pruned",
                "weighting_mode": "recency_magnitude",
                "top_k_features": 20,
                "random_state": 42
            },
            "lightgbm_reg_baseline": {
                "task_type": "regression",
                "model_type": "lightgbm_regressor",
                "feature_selection_method": "all",
                "weighting_mode": "recency_magnitude",
                "random_state": 42
            }
        }
    }


def compute_model_config_hash(cfg: Optional[Dict[str, Any]] = None) -> str:
    """生成标准模型研究配置的确定性 SHA256 哈希"""
    if cfg is None:
        cfg = build_canonical_model_config()
    raw_json = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()


def reconstruct_source_model_config_from_git(
    source_commit: str = "e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48",
    project_root: Optional[Path] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    从 Git 历史提交只读读取当时的 config/settings.py 并重建历史模型配置 (不改写工作区)
    """
    try:
        cmd = ["git", "show", f"{source_commit}:config/settings.py"]
        res = subprocess.run(cmd, cwd=project_root, capture_output=True, encoding="utf-8", errors="replace", check=True)
        txt = res.stdout

        tw = float(re.search(r"TRAIN_WINDOW_YEARS:\s*float\s*=\s*([0-9\.]+)", txt).group(1))
        vm = int(re.search(r"VAL_WINDOW_MONTHS:\s*int\s*=\s*([0-9]+)", txt).group(1))
        tm = int(re.search(r"TEST_WINDOW_MONTHS:\s*int\s*=\s*([0-9]+)", txt).group(1))
        pg = int(re.search(r"PURGE_GAP_DAYS:\s*int\s*=\s*([0-9]+)", txt).group(1))
        lh = int(re.search(r"LABEL_HORIZON:\s*int\s*=\s*([0-9]+)", txt).group(1))
        ltm = re.search(r'LABEL_THRESHOLD_MODE:\s*str\s*=\s*["\']([^"\']+)["\']', txt).group(1)
        leq = float(re.search(r"LABEL_EXTREME_QUANTILE:\s*float\s*=\s*([0-9\.]+)", txt).group(1))

        historical_cfg = {
            "walk_forward": {
                "train_window_years": tw,
                "val_window_months": vm,
                "test_window_months": tm,
                "purge_gap_days": pg,
            },
            "labeling": {
                "label_horizon": lh,
                "label_threshold_mode": ltm,
                "label_extreme_quantile": leq,
            },
            "models": {
                "lightgbm_clf_baseline": {
                    "task_type": "classification",
                    "model_type": "lightgbm",
                    "feature_selection_method": "all",
                    "weighting_mode": "none",
                    "random_state": 42
                },
                "double_ensemble": {
                    "task_type": "classification",
                    "model_type": "double_ensemble",
                    "feature_selection_method": "top_20",
                    "weighting_mode": "recency_magnitude",
                    "top_k_features": 20,
                    "random_state": 42
                },
                "lightgbm_ranker": {
                    "task_type": "ranking",
                    "model_type": "lightgbm_ranker",
                    "feature_selection_method": "rank_ic_pruned",
                    "weighting_mode": "recency_magnitude",
                    "top_k_features": 20,
                    "random_state": 42
                },
                "lightgbm_reg_baseline": {
                    "task_type": "regression",
                    "model_type": "lightgbm_regressor",
                    "feature_selection_method": "all",
                    "weighting_mode": "recency_magnitude",
                    "random_state": 42
                }
            }
        }
        hist_hash = compute_model_config_hash(historical_cfg)
        return True, hist_hash, historical_cfg
    except Exception as e:
        return False, f"Reconstruction failed: {str(e)}", None


def validate_artifact_reuse_compatibility(
    current_meta: Dict[str, Any],
    source_meta: Dict[str, Any],
    source_commit: str = "e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48"
) -> Tuple[bool, str]:
    """
    严格验证产物复用兼容性 (Fail-Closed):
    比对 dataset_sha256, feature_schema_hash, label_horizon 以及从 Git 历史真实重建的 source_model_config_hash
    """
    for key in ["dataset_sha256", "feature_schema_hash", "label_horizon"]:
        c_val = current_meta.get(key)
        s_val = source_meta.get(key)
        if c_val != s_val:
            return False, f"Mismatch in {key}: current={c_val} vs source={s_val}"

    # 从 Git 历史重建 source_model_config_hash
    reconstruct_ok, hist_hash_or_err, _ = reconstruct_source_model_config_from_git(source_commit=source_commit)
    if not reconstruct_ok:
        return False, f"Source model config reconstruction error: {hist_hash_or_err}"

    current_hash = current_meta.get("model_config_hash")
    if current_hash != hist_hash_or_err:
        return False, f"Mismatch in model_config_hash: current={current_hash} vs reconstructed_source={hist_hash_or_err}"

    return True, "COMPATIBLE"


def derive_seed_status(records: List[Dict[str, Any]]) -> str:
    """
    根据多随机种子评估证据运行时动态推导 SEED_ROBUSTNESS_STATUS:
    - 如果证据缺失或少于 2 个种子: NOT_VERIFIED
    - 如果所有种子 hash 完全相同且已传入不同 seed: DETERMINISTIC_IDENTICAL
    - 如果种子 hash 不同且 RankIC 极差 max-min <= 0.01: VERIFIED_STABLE
    - 如果 RankIC 极差 max-min > 0.02: UNSTABLE
    - 否则: PARTIAL
    """
    if not records or len(records) < 2:
        return "NOT_VERIFIED"

    hashes = [r.get("prediction_hash") for r in records if r.get("prediction_hash")]
    rank_ics = [float(r.get("mean_daily_rank_ic", r.get("mean_rank_ic", 0.0))) for r in records]

    if len(hashes) < len(records):
        return "NOT_VERIFIED"

    unique_hashes = set(hashes)
    ic_spread = max(rank_ics) - min(rank_ics)

    if len(unique_hashes) == 1:
        return "DETERMINISTIC_IDENTICAL"
    elif ic_spread <= 0.01:
        return "VERIFIED_STABLE"
    elif ic_spread > 0.02:
        return "UNSTABLE"
    else:
        return "PARTIAL"


def derive_trading_signal_status(
    overall_excess: float,
    fold_win_ratio: float,
    top5_mean: float,
    top10_mean: float
) -> str:
    """
    根据整体成本后超额、真实 Fold 胜率和 Top Tail 前瞻收益推导交易信号状态:
    - overall_excess > 0 且 fold_win_ratio >= 0.50 且 top5_mean > 0 且 top10_mean > 0: PROMISING_OOS_SIGNAL
    - overall_excess > 0 但 (fold_win_ratio < 0.50 或 top tail <= 0): UNSTABLE_OOS_SIGNAL
    - 否则: NO_TRADING_EDGE
    """
    if overall_excess > 0 and fold_win_ratio >= 0.50 and top5_mean > 0 and top10_mean > 0:
        return "PROMISING_OOS_SIGNAL"
    elif overall_excess > 0:
        return "UNSTABLE_OOS_SIGNAL"
    else:
        return "NO_TRADING_EDGE"


def derive_phase_2_1_ready(required_local_gates: Dict[str, str]) -> bool:
    """
    Fail-Closed 准入判定:
    只有全部前置门禁为 PASS 时才允许进入 Phase 2.1
    """
    if not required_local_gates:
        return False
    return all(v == "PASS" for v in required_local_gates.values())


def compute_top_tail_analysis(oos_df: pd.DataFrame, label_col: str = "label_excess_20d") -> pd.DataFrame:
    """
    计算预测得分 Top 5%, 10%, 20% 截面多头前瞻收益与胜率
    严格使用 bottom 10% 真实均值作为 worst_decile_mean
    """
    valid = oos_df[oos_df[label_col].notna() & oos_df["pred_score"].notna()].copy()
    if "in_universe" in valid.columns:
        valid = valid[valid["in_universe"].fillna(False).astype(bool)].copy()

    records = []
    for pct, label in [(0.05, "Top 5%"), (0.10, "Top 10%"), (0.20, "Top 20%")]:
        tail_excess = []
        tail_sizes = []
        for dt, g in valid.groupby("date"):
            k = max(1, int(len(g) * pct))
            top_g = g.sort_values(by="pred_score", ascending=False).head(k)
            tail_excess.extend(top_g[label_col].values)
            tail_sizes.append(len(top_g))

        s = pd.Series(tail_excess)
        mean_excess = float(s.mean()) if len(s) > 0 else 0.0
        median_excess = float(s.median()) if len(s) > 0 else 0.0
        hit_rate = float((s > 0).mean() * 100.0) if len(s) > 0 else 0.0
        
        # 真正计算最差 10% 样本的算术均值
        bottom_n = max(1, int(np.ceil(len(s) * 0.10))) if len(s) > 0 else 0
        worst_decile_mean = float(s.nsmallest(bottom_n).mean()) if bottom_n > 0 else 0.0

        records.append({
            "tail_tier": label,
            "quantile_pct": pct,
            "tail_size_avg": round(float(np.mean(tail_sizes)), 1) if tail_sizes else 0.0,
            "mean_forward_20d_excess": round(mean_excess * 100.0, 2),
            "median_forward_20d_excess": round(median_excess * 100.0, 2),
            "positive_excess_hit_rate": round(hit_rate, 2),
            "worst_decile_mean": round(worst_decile_mean * 100.0, 2)
        })

    return pd.DataFrame(records)


def evaluate_all_gates(
    worktree_clean: bool,
    source_commits_valid: bool,
    dataset_and_schema_valid: bool,
    model_config_hash_valid: bool,
    artifact_reuse_valid: bool,
    seed_params_valid: bool,
    derived_seed_status: str,
    common_rows_equal: bool,
    common_dates_equal: bool,
    nw20_valid: bool,
    bootstrap_valid: bool,
    self_comp_guard_passed: bool,
    report_semantics_passed: bool,
    trading_fold_valid: bool,
    test_exit_code_zero: bool
) -> Dict[str, str]:
    """
    纯证据驱动全量 Gate 状态推导 (Fail-Closed)
    """
    return {
        "SOURCE_PROVENANCE": "PASS" if (worktree_clean and source_commits_valid and dataset_and_schema_valid) else "FAIL",
        "MODEL_CONFIG_HASH_VALIDITY": "PASS" if model_config_hash_valid else "FAIL",
        "ARTIFACT_REUSE_COMPATIBILITY": "PASS" if artifact_reuse_valid else "FAIL",
        "SEED_PROPAGATION": "PASS" if seed_params_valid else "FAIL",
        "SEED_ROBUSTNESS": "PASS" if derived_seed_status in ("VERIFIED_STABLE", "DETERMINISTIC_IDENTICAL") else "FAIL",
        "COMMON_OOS_POOL": "PASS" if (common_rows_equal and common_dates_equal) else "FAIL",
        "NW20_CERTIFICATION": "PASS" if nw20_valid else "FAIL",
        "BOOTSTRAP_VALIDITY": "PASS" if bootstrap_valid else "FAIL",
        "SELF_COMPARISON_GUARD": "PASS" if self_comp_guard_passed else "FAIL",
        "REPORT_SEMANTICS": "PASS" if report_semantics_passed else "FAIL",
        "TRADING_FOLD_EVIDENCE_VALIDITY": "PASS" if trading_fold_valid else "FAIL",
        "PYTEST": "PASS" if test_exit_code_zero else "FAIL"
    }
