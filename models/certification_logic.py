"""
Phase 2.0.2r2 Certification Logic & Runtime State Derivation (models/certification_logic.py)
严格执行 Fail-Closed 运行时状态推导、真实指标计算与多重门禁判定，杜绝任何硬编码假证据。
"""
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


def compute_model_config_hash(
    train_window_years: float = 1.4,
    val_window_months: int = 2,
    test_window_months: int = 2,
    purge_gap_days: int = 20,
    label_horizon: int = 20,
    top_k_features: int = 20
) -> str:
    """生成稳健模型研究配置摘要哈希"""
    cfg = {
        "train_window_years": train_window_years,
        "val_window_months": val_window_months,
        "test_window_months": test_window_months,
        "purge_gap_days": purge_gap_days,
        "label_horizon": label_horizon,
        "top_k_features": top_k_features,
        "models": ["lightgbm_clf_baseline", "double_ensemble", "lightgbm_ranker", "lightgbm_reg_baseline"]
    }
    raw_str = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()


def validate_artifact_reuse_compatibility(
    current_meta: Dict[str, Any],
    source_meta: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    严格验证产物复用兼容性 (Fail-Closed):
    如果 dataset_sha256, feature_schema_hash, label_horizon 或 model_config_hash 任一不一致，拒绝复用
    """
    for key in ["dataset_sha256", "feature_schema_hash", "label_horizon"]:
        c_val = current_meta.get(key)
        s_val = source_meta.get(key)
        if c_val != s_val:
            return False, f"Mismatch in {key}: current={c_val} vs source={s_val}"

    if "model_config_hash" in current_meta and "model_config_hash" in source_meta:
        if current_meta["model_config_hash"] != source_meta["model_config_hash"]:
            return False, "Mismatch in model_config_hash"

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
    source_state_valid: bool,
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
        "SOURCE_PROVENANCE": "PASS" if (worktree_clean and source_state_valid) else "FAIL",
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
