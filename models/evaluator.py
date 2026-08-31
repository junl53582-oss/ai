"""
模型预测性能评估器 (models/evaluator.py)
严格在 COMMON_RANKING_POOL (in_universe == True 的有效样本) 上执行 OOS 预测质量评估
Research Integrity Hardened:
- 分位数评估 (Quantile Evaluation): 采用确定性 rank(pct=True) 映射，严禁将异常日全部坍塌塞入 Q1
- 跨日等权聚合 (Daily Equal-Weighting): 先计算每日等权 Quantile Return，再跨日等权求平均，杜绝样本数更多交易日主导权重
- 区分 Signal Metric (annualized_arithmetic_forward_label_spread) 与真实 Portfolio Return
- 补充未年化与年化 HAC RankICIR 多维度指标 (rankicir_raw_unannualized, rankicir_nw20_unannualized, rankicir_hac_v2)
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, brier_score_loss, log_loss
)

from config.settings import settings

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """量化预测模型表现评估引擎 (支持二分类与连续回归/排序任务，严格在成分股共同池上评估)"""

    def __init__(self):
        pass

    def evaluate_predictions(
        self,
        df: pd.DataFrame,
        label_col: str = settings.LABEL_COLUMN,
        task_type: str = settings.TASK_TYPE
    ) -> Dict[str, Any]:
        """
        评估样本外 (OOS) 预测结果：
        严格在 in_universe == True 样本上评估横截面 RankIC 与分层收益
        """
        if df is None or len(df) == 0:
            raise ValueError("评估数据集为空！")

        oos_total_rows = len(df)
        if "in_universe" in df.columns:
            df_ranking = df[df["in_universe"].fillna(False).astype(bool)].copy()
        else:
            df_ranking = df.copy()

        if "excluded_from_training" in df_ranking.columns:
            df_ranking = df_ranking[~df_ranking["excluded_from_training"].fillna(False).astype(bool)].copy()

        if label_col is None:
            label_col = settings.LABEL_COLUMN_CLF if task_type == "classification" else settings.LABEL_COLUMN

        if task_type == "classification":
            clf_col = label_col if (label_col in df_ranking.columns and not label_col.startswith("label_excess_") and not label_col.startswith("label_net_alpha_")) else None
            if clf_col is None:
                clf_cands = [c for c in df_ranking.columns if c.startswith("label_up_down_") or c.startswith("label_direction_") or c.startswith("ab_label_")]
                if clf_cands:
                    clf_col = clf_cands[0]
                elif label_col in df_ranking.columns:
                    clf_col = label_col

            if clf_col and clf_col in df_ranking.columns:
                clf_member_rows = len(df_ranking[df_ranking[clf_col].notna()])
                df_clf = df_ranking[df_ranking[clf_col].notna()].copy()
                label_col = clf_col
            else:
                clf_member_rows = 0
                df_clf = pd.DataFrame()

            cont_col = None
            for cand in (settings.LABEL_COLUMN, "label_excess_20d", "label_net_alpha_20d", "label_excess_5d", "label_net_alpha_5d"):
                if cand in df_ranking.columns:
                    cont_col = cand
                    break
            if cont_col is None:
                reg_cands = [c for c in df_ranking.columns if c.startswith("label_excess_") or c.startswith("label_net_alpha_")]
                if reg_cands:
                    cont_col = reg_cands[0]
        else:
            clf_member_rows = 0
            df_clf = pd.DataFrame()
            cont_col = label_col
            if cont_col not in df_ranking.columns:
                reg_cands = [c for c in df_ranking.columns if c.startswith("label_excess_") or c.startswith("label_net_alpha_")]
                if reg_cands:
                    cont_col = reg_cands[0]

        evaluated_member_rows = len(df_ranking)
        oos_excluded_nonmember_rows = oos_total_rows - evaluated_member_rows
        evaluation_coverage_ratio = round(evaluated_member_rows / max(oos_total_rows, 1), 4)

        if len(df_clf) == 0:
            auc, accuracy, precision, recall, f1 = 0.5, 0.0, 0.0, 0.0, 0.0
            cm, brier, logloss, positive_rate = [[0, 0], [0, 0]], 0.0, 0.0, 0.0
        else:
            y_true = df_clf[label_col].astype(int).values
            y_prob = df_clf["pred_score"].values
            try:
                auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
            except Exception:
                auc = 0.5
            y_pred = (y_prob >= 0.5).astype(int)
            accuracy = float(accuracy_score(y_true, y_pred))
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            recall = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
            brier = float(brier_score_loss(y_true, y_prob))
            y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
            try:
                logloss = float(log_loss(y_true, y_prob_clipped))
            except Exception:
                logloss = 0.0
            positive_rate = float(y_true.mean())

        # 分层单调性 (在 COMMON_RANKING_POOL 上评估)
        quantile_info = self._compute_quantile_returns(df_ranking, label_col=cont_col or label_col, n_groups=5)

        # 计算每日横截面 RankIC 与 RankICIR
        daily_rank_ic_list = []
        rank_ic_dates = []
        if cont_col is not None:
            for dt, group in df_ranking.groupby("date"):
                valid_g = group[group[cont_col].notna() & group["pred_score"].notna()]
                if len(valid_g) >= 3:
                    s_ic = stats.spearmanr(valid_g["pred_score"], valid_g[cont_col])[0]
                    if np.isfinite(s_ic):
                        daily_rank_ic_list.append(float(s_ic))
                        rank_ic_dates.append(pd.Timestamp(dt))

        rank_ic_series = pd.Series(daily_rank_ic_list, index=rank_ic_dates).sort_index()
        rank_ic_mean = float(rank_ic_series.mean()) if len(rank_ic_series) > 0 else 0.0
        rank_ic_std = float(rank_ic_series.std()) if len(rank_ic_series) > 0 else 1.0

        annual_factor = np.sqrt(242.0 / settings.LABEL_HORIZON)
        rankicir_raw_unannualized = float(rank_ic_mean / (rank_ic_std + 1e-8))
        rank_icir = rankicir_raw_unannualized * annual_factor

        std_nw5 = self._compute_newey_west_std(rank_ic_series, max_lag=5)
        std_nw20 = self._compute_newey_west_std(rank_ic_series, max_lag=20)
        rankicir_nw5_unannualized = float(rank_ic_mean / (std_nw5 + 1e-8))
        rankicir_nw20_unannualized = float(rank_ic_mean / (std_nw20 + 1e-8))

        rank_icir_nw_lag5 = rankicir_nw5_unannualized * annual_factor
        rank_icir_nw_lag20 = rankicir_nw20_unannualized * annual_factor
        rankicir_hac_v2 = rankicir_nw20_unannualized * np.sqrt(242.0)

        metrics = {
            "task_type": "classification" if task_type == "classification" else "regression",
            "auc": round(auc, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(logloss, 4),
            "confusion_matrix": cm,
            "positive_rate": round(positive_rate, 4),
            "evaluated_member_rows": evaluated_member_rows,
            "common_ranking_rows": evaluated_member_rows,
            "classification_rows": clf_member_rows,
            "oos_total_rows": oos_total_rows,
            "oos_excluded_nonmember_rows": oos_excluded_nonmember_rows,
            "evaluation_coverage_ratio": evaluation_coverage_ratio,
            "quantile_returns": quantile_info["annualized_arithmetic_forward_excess_return"],
            "annualized_arithmetic_forward_excess_return": quantile_info["annualized_arithmetic_forward_excess_return"],
            "annualized_arithmetic_forward_label_spread": quantile_info["Q5_minus_Q1"],
            "Q5_minus_Q1": quantile_info["Q5_minus_Q1"],
            "mean_daily_q5_minus_q1": quantile_info.get("mean_daily_q5_minus_q1", quantile_info["Q5_minus_Q1"]),
            "legacy_row_weighted_q5_minus_q1": quantile_info.get("legacy_row_weighted_q5_minus_q1", quantile_info["Q5_minus_Q1"]),
            "monotonicity_score": quantile_info["monotonicity_score"],
            "quantile_observation_count": quantile_info["quantile_observation_count"],
            "rank_ic_mean": round(rank_ic_mean, 6),
            "mean_rank_ic": round(rank_ic_mean, 6),
            "mean_daily_rank_ic": round(rank_ic_mean, 6),
            "rank_ic_std": round(rank_ic_std, 6),
            "rankic_std_raw": round(rank_ic_std, 6),
            "rankic_long_run_std_nw5": round(std_nw5, 6),
            "rankic_long_run_std_nw20": round(std_nw20, 6),
            "rankicir_raw_unannualized": round(rankicir_raw_unannualized, 6),
            "rankicir_nw20_unannualized": round(rankicir_nw20_unannualized, 6),
            "rank_icir": round(rank_icir, 6),
            "rank_icir_nw_lag5": round(rank_icir_nw_lag5, 6),
            "rank_icir_nw_lag20": round(rank_icir_nw_lag20, 6),
            "rankicir_hac_v2": round(rankicir_hac_v2, 6),
            "rank_ic_series": rank_ic_series
        }
        return metrics

    def _compute_newey_west_std(self, series: pd.Series, max_lag: int = 5) -> float:
        """计算 Newey-West (Bartlett 权重) 稳健标准差"""
        if len(series) < max_lag + 2:
            return float(series.std()) if len(series) > 0 else 1.0

        y = series.values - series.mean()
        T = len(y)
        gamma_0 = np.sum(y**2) / T
        v_nw = gamma_0
        for lag in range(1, max_lag + 1):
            weight = 1.0 - lag / (max_lag + 1.0)
            gamma_l = np.sum(y[lag:] * y[:-lag]) / T
            v_nw += 2.0 * weight * gamma_l

        return float(np.sqrt(max(1e-8, v_nw)))

    def _compute_quantile_returns(
        self,
        df: pd.DataFrame,
        label_col: str = settings.LABEL_COLUMN,
        n_groups: int = 5
    ) -> Dict[str, Any]:
        """
        将股票每日按预测得分分为 5 组 (Q1~Q5)，评估分层收益的单调性与利差：
        Research Integrity Hardened:
        - 确定性 percentile rank 映射 (rank(method='first', pct=True))，杜绝异常日全部塞 Q1
        - 每日横截面有效样本不足 n_groups 时显式标记无效并跳过，禁止坍塌
        - 跨交易日等权聚合 (Daily Equal-Weighting): 先算每日各组平均收益，再跨日求平均
        - 保留旧版 row-weighted 结果并显式命名为 legacy_row_weighted_q5_minus_q1
        """
        df = df.copy()
        valid_mask = df[label_col].notna() & df["pred_score"].notna()
        df_valid = df[valid_mask].copy()

        if len(df_valid) == 0:
            return {
                "annualized_arithmetic_forward_excess_return": {},
                "Q5_minus_Q1": 0.0,
                "mean_daily_q5_minus_q1": 0.0,
                "legacy_row_weighted_q5_minus_q1": 0.0,
                "monotonicity_score": 0.0,
                "valid_quantile_dates": 0,
                "invalid_quantile_dates": 0,
                "quantile_observation_count": 0
            }

        daily_groups = []
        invalid_dates_count = 0

        daily_sizes = df_valid.groupby("date").size()
        med_size = daily_sizes.median() if not daily_sizes.empty else 0
        effective_n_groups = int(med_size) if (2 <= med_size < n_groups) else n_groups

        for dt, g in df_valid.groupby("date"):
            n_obs = len(g)
            if n_obs < effective_n_groups:
                invalid_dates_count += 1
                continue

            # 确定性分位数映射
            pct = g["pred_score"].rank(method="first", pct=True)
            q_bins = np.clip(np.ceil(pct * float(effective_n_groups)), 1, effective_n_groups).astype(int)
            g_assigned = g.copy()
            g_assigned["group"] = [f"Q{b}" for b in q_bins]
            daily_groups.append(g_assigned)

        if not daily_groups:
            return {
                "annualized_arithmetic_forward_excess_return": {},
                "Q5_minus_Q1": 0.0,
                "mean_daily_q5_minus_q1": 0.0,
                "legacy_row_weighted_q5_minus_q1": 0.0,
                "monotonicity_score": 0.0,
                "valid_quantile_dates": 0,
                "invalid_quantile_dates": invalid_dates_count,
                "quantile_observation_count": 0
            }

        grouped_df = pd.concat(daily_groups, ignore_index=True)
        annual_factor = (242.0 / settings.LABEL_HORIZON) * 100.0

        # 1. 每日等权聚合 (Daily Equal-Weighted Aggregation)
        daily_q_means = grouped_df.groupby(["date", "group"])[label_col].mean().unstack("group")
        mean_daily_group_returns = (daily_q_means.mean() * annual_factor).to_dict()
        returns_dict = {str(k): round(float(v), 2) for k, v in mean_daily_group_returns.items()}

        q1_ret = returns_dict.get("Q1", 0.0)
        q5_ret = returns_dict.get("Q5", 0.0)
        mean_daily_q5_minus_q1 = round(q5_ret - q1_ret, 2)

        # 2. 传统行加权聚合 (Legacy Row-Weighted Aggregation)
        legacy_group_means = (grouped_df.groupby("group", observed=False)[label_col].mean() * annual_factor).to_dict()
        legacy_q1 = legacy_group_means.get("Q1", 0.0)
        legacy_q5 = legacy_group_means.get("Q5", 0.0)
        legacy_q5_minus_q1 = round(float(legacy_q5 - legacy_q1), 2)

        # 3. 单调性得分 (Spearman Rank Corr)
        if len(returns_dict) >= 3:
            ranks = list(range(1, len(returns_dict) + 1))
            rets = [returns_dict[f"Q{i}"] for i in ranks if f"Q{i}" in returns_dict]
            if len(rets) == len(ranks):
                monotonicity = round(float(stats.spearmanr(ranks, rets)[0]), 4)
            else:
                monotonicity = 0.0
        else:
            monotonicity = 0.0

        return {
            "annualized_arithmetic_forward_excess_return": returns_dict,
            "Q5_minus_Q1": mean_daily_q5_minus_q1,
            "mean_daily_q5_minus_q1": mean_daily_q5_minus_q1,
            "legacy_row_weighted_q5_minus_q1": legacy_q5_minus_q1,
            "monotonicity_score": monotonicity,
            "valid_quantile_dates": int(len(daily_q_means)),
            "invalid_quantile_dates": int(invalid_dates_count),
            "quantile_observation_count": int(len(grouped_df))
        }
