"""
Phase 2.0.2r1 Certification Logic & Runtime State Derivation (models/certification_logic.py)
严格执行 Fail-Closed 运行时状态推导与真实指标计算，杜绝任何硬编码假证据。
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


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
