"""
量化预测模型与 Alpha 因子绩效评估器
- 回归模式: 每日 IC、RankIC、Mean IC、Mean RankIC、ICIR、RankICIR、滚动 20D/60D RankIC 与 5 分层组合收益单调性
- 分类模式 (涨跌预测): AUC、Accuracy、Precision、Recall、F1、混淆矩阵、Brier Score、对数损失与 5 分层单调性
"""
import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats

from config.settings import settings

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型与信号质量评估引擎 (支持 PIT 成员过滤、Horizon 对齐分层收益与 Newey-West 稳健 ICIR)"""

    def evaluate_predictions(
        self,
        oos_df: pd.DataFrame,
        label_col: Optional[str] = None,
        task_type: str = settings.TASK_TYPE
    ) -> Dict[str, Any]:
        """
        评估样本外 (Out-of-Sample) 预测结果质量 (按任务类型分发)：
        - classification: AUC/Accuracy/Precision/Recall/F1/混淆矩阵/Brier/对数损失 + 分层单调性
        - regression: IC/RankIC/ICIR/RankICIR/滚动 RankIC/IC 胜率 + 分层单调性
        """
        if label_col is None:
            # 自动探测在 oos_df 中实际存在的有效标签列
            clf_cands = [c for c in oos_df.columns if c.startswith("label_up_down_")]
            reg_cands = [c for c in oos_df.columns if c.startswith("label_excess_")]
            if task_type == "classification":
                if settings.LABEL_COLUMN_CLF in oos_df.columns:
                    label_col = settings.LABEL_COLUMN_CLF
                elif clf_cands:
                    label_col = clf_cands[0]
                else:
                    label_col = settings.LABEL_COLUMN_CLF
            else:
                if settings.LABEL_COLUMN in oos_df.columns:
                    label_col = settings.LABEL_COLUMN
                elif reg_cands:
                    label_col = reg_cands[0]
                else:
                    label_col = settings.LABEL_COLUMN

        if task_type == "classification":
            return self._evaluate_classification(oos_df, label_col=label_col)
        return self._evaluate_regression(oos_df, label_col=label_col)

    def _evaluate_classification(self, oos_df: pd.DataFrame, label_col: str = settings.LABEL_COLUMN_CLF) -> Dict[str, Any]:
        """
        分类模式评估 (涨跌二分类)：
        1. 严格仅在 in_universe == True 且标签与概率均有效的样本上评估
        2. AUC-ROC (核心)、Accuracy、Precision、Recall、F1、混淆矩阵
        3. Brier Score 与对数损失 (概率质量)
        4. Q1 ~ Q5 分层单调性 (基于概率排序，用连续超额收益标签衡量真实分层收益)
        """
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix, brier_score_loss, log_loss
        )

        logger.info("开始计算分类模型样本外评估指标 (AUC / Accuracy / F1 / Brier ...)...")
        oos_total_rows = len(oos_df)
        df_valid = oos_df[oos_df[label_col].notna() & oos_df["pred_score"].notna()].copy()

        # 过滤未纳入股票池与严格 ST 排除行
        if "excluded_from_training" in df_valid.columns:
            not_excl = ~df_valid["excluded_from_training"].fillna(False).astype(bool)
            df_valid = df_valid[not_excl].copy()

        if "in_universe" in df_valid.columns:
            in_univ_mask = df_valid["in_universe"].fillna(False).astype(bool)
            df = df_valid[in_univ_mask].copy()
        else:
            df = df_valid.copy()

        evaluated_member_rows = len(df)
        oos_excluded_nonmember_rows = oos_total_rows - evaluated_member_rows
        evaluation_coverage_ratio = round(evaluated_member_rows / max(oos_total_rows, 1), 4)

        if len(df) == 0:
            return {
                "task_type": "classification",
                "auc": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "brier_score": 0.0, "log_loss": 0.0,
                "confusion_matrix": [[0, 0], [0, 0]],
                "positive_rate": 0.0, "evaluated_member_rows": 0,
                "oos_total_rows": oos_total_rows,
                "oos_excluded_nonmember_rows": oos_excluded_nonmember_rows,
                "evaluation_coverage_ratio": evaluation_coverage_ratio,
                "quantile_returns": {}, "Q5_minus_Q1": 0.0,
                "monotonicity_score": 0.0, "quantile_observation_count": 0
            }

        y_true = df[label_col].astype(int).values
        y_prob = df["pred_score"].values

        # AUC (处理单类别样本边缘情况)
        try:
            auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            auc = 0.5

        # 0.5 阈值二值化预测
        y_pred = (y_prob >= 0.5).astype(int)

        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        brier = float(brier_score_loss(y_true, y_prob))

        # 对数损失 (概率截断避免 log(0))
        y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        try:
            logloss = float(log_loss(y_true, y_prob_clipped))
        except Exception:
            logloss = 0.0

        positive_rate = float(y_true.mean())

        # 分层单调性 (按概率排序，用连续超额收益标签衡量真实分层收益)
        cont_col = settings.LABEL_COLUMN if settings.LABEL_COLUMN in df.columns else None
        if cont_col is None:
            cand_cols = [c for c in df.columns if "excess" in c and c != label_col]
            cont_col = cand_cols[0] if cand_cols else None

        quantile_info = self._compute_quantile_returns(df, label_col=cont_col or label_col, n_groups=5)

        # 计算每日横截面 RankIC 与 RankICIR
        daily_rank_ic_list = []
        rank_ic_dates = []
        if cont_col is not None:
            for dt, group in df.groupby("date"):
                valid_g = group[group[cont_col].notna() & group["pred_score"].notna()]
                if len(valid_g) >= 3:
                    s_ic = stats.spearmanr(valid_g["pred_score"], valid_g[cont_col])[0]
                    if not np.isnan(s_ic):
                        daily_rank_ic_list.append(s_ic)
                        rank_ic_dates.append(dt)

        rank_ic_series = pd.Series(daily_rank_ic_list, index=rank_ic_dates)
        rank_ic_mean = float(rank_ic_series.mean()) if len(rank_ic_series) > 0 else 0.0
        rank_ic_std = float(rank_ic_series.std()) if len(rank_ic_series) > 0 else 1.0
        annual_factor = np.sqrt(242.0 / settings.LABEL_HORIZON)
        rank_icir = (rank_ic_mean / (rank_ic_std + 1e-8)) * annual_factor
        std_nw = self._compute_newey_west_std(rank_ic_series, max_lag=5)
        rank_icir_newey_west = (rank_ic_mean / (std_nw + 1e-8)) * annual_factor

        metrics = {
            "task_type": "classification",
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
            "oos_total_rows": oos_total_rows,
            "oos_excluded_nonmember_rows": oos_excluded_nonmember_rows,
            "evaluation_coverage_ratio": evaluation_coverage_ratio,
            "quantile_returns": quantile_info["annualized_arithmetic_forward_excess_return"],
            "annualized_arithmetic_forward_excess_return": quantile_info["annualized_arithmetic_forward_excess_return"],
            "Q5_minus_Q1": quantile_info["Q5_minus_Q1"],
            "monotonicity_score": quantile_info["monotonicity_score"],
            "quantile_observation_count": quantile_info["quantile_observation_count"],
            "mean_rank_ic": round(rank_ic_mean, 4),
            "rank_ic_std": round(rank_ic_std, 4),
            "rank_icir": round(rank_icir, 4),
            "rank_icir_newey_west": round(rank_icir_newey_west, 4),
            "rank_ic_series": rank_ic_series
        }

        logger.info(
            f"分类 OOS 评估: AUC={metrics['auc']} | RankIC={metrics['mean_rank_ic']} | "
            f"RankICIR(NW)={metrics['rank_icir_newey_west']} | Accuracy={metrics['accuracy']} | "
            f"F1={metrics['f1']} | Brier={metrics['brier_score']} | "
            f"上涨样本占比={positive_rate*100:.1f}% | 评估行数={evaluated_member_rows}/{oos_total_rows}"
        )
        return metrics

    def _evaluate_regression(self, oos_df: pd.DataFrame, label_col: str = settings.LABEL_COLUMN) -> Dict[str, Any]:
        logger.info("开始计算样本外模型预测质量与因子 IC 体系...")
        oos_total_rows = len(oos_df)
        df_valid = oos_df[oos_df[label_col].notna() & oos_df["pred_score"].notna()].copy()

        # 过滤未纳入股票池与严格 ST 排除行 (P0-5, P0-7, P1-3)
        if "excluded_from_training" in df_valid.columns:
            not_excl = ~df_valid["excluded_from_training"].fillna(False).astype(bool)
            df_valid = df_valid[not_excl].copy()

        if "in_universe" in df_valid.columns:
            in_univ_mask = df_valid["in_universe"].fillna(False).astype(bool)
            df = df_valid[in_univ_mask].copy()
        else:
            df = df_valid.copy()

        evaluated_member_rows = len(df)
        oos_excluded_nonmember_rows = oos_total_rows - evaluated_member_rows
        evaluation_coverage_ratio = round(evaluated_member_rows / max(oos_total_rows, 1), 4)

        daily_ic_list = []
        daily_rank_ic_list = []
        dates = []

        # 按交易日循环计算截面相关性
        for dt, group in df.groupby("date"):
            if len(group) < 3:
                continue
            pred = group["pred_score"]
            true_y = group[label_col]

            # Pearson IC
            p_ic = stats.pearsonr(pred, true_y)[0]
            # Spearman RankIC
            s_ic = stats.spearmanr(pred, true_y)[0]

            if not np.isnan(p_ic) and not np.isnan(s_ic):
                daily_ic_list.append(p_ic)
                daily_rank_ic_list.append(s_ic)
                dates.append(dt)

        ic_series = pd.Series(daily_ic_list, index=dates)
        rank_ic_series = pd.Series(daily_rank_ic_list, index=dates)

        ic_mean = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
        ic_std = float(ic_series.std()) if len(ic_series) > 0 else 1.0
        rank_ic_mean = float(rank_ic_series.mean()) if len(rank_ic_series) > 0 else 0.0
        rank_ic_std = float(rank_ic_series.std()) if len(rank_ic_series) > 0 else 1.0

        # 年化 ICIR 与 RankICIR (以 5 日预测为周期折算)
        annual_factor = np.sqrt(242.0 / settings.LABEL_HORIZON)
        icir = (ic_mean / (ic_std + 1e-8)) * annual_factor
        rank_icir = (rank_ic_mean / (rank_ic_std + 1e-8)) * annual_factor

        # Newey-West 自相关调整 RankICIR (P1-4 Lag=5)
        std_nw = self._compute_newey_west_std(rank_ic_series, max_lag=5)
        rank_icir_newey_west = (rank_ic_mean / (std_nw + 1e-8)) * annual_factor

        ic_win_rate = float((ic_series > 0).mean() * 100.0) if len(ic_series) > 0 else 0.0
        rank_ic_win_rate = float((rank_ic_series > 0).mean() * 100.0) if len(rank_ic_series) > 0 else 0.0

        # 20D / 60D 滚动 RankIC
        rolling_rank_ic_20 = float(rank_ic_series.rolling(20).mean().dropna().mean()) if len(rank_ic_series) >= 20 else rank_ic_mean
        rolling_rank_ic_60 = float(rank_ic_series.rolling(60).mean().dropna().mean()) if len(rank_ic_series) >= 60 else rank_ic_mean

        # 5 分层组合收益 (基于未来超额收益标签 P0-6, P1-1)
        quantile_info = self._compute_quantile_returns(df, label_col=label_col, n_groups=5)

        metrics = {
            "task_type": "regression",
            "ic_mean": round(ic_mean, 4),
            "mean_ic": round(ic_mean, 4),
            "rank_ic_mean": round(rank_ic_mean, 4),
            "mean_rank_ic": round(rank_ic_mean, 4),
            "ic_std": round(ic_std, 4),
            "rank_ic_std": round(rank_ic_std, 4),
            "icir": round(float(icir), 4),
            "rank_icir": round(float(rank_icir), 4),
            "rank_icir_newey_west": round(float(rank_icir_newey_west), 4),
            "ic_win_rate": round(ic_win_rate, 2),
            "rank_ic_win_rate": round(rank_ic_win_rate, 2),
            "rolling_rank_ic_20d": round(rolling_rank_ic_20, 4),
            "rolling_rank_ic_60d": round(rolling_rank_ic_60, 4),
            "total_evaluated_days": len(dates),
            "oos_total_rows": oos_total_rows,
            "evaluated_member_rows": evaluated_member_rows,
            "oos_excluded_nonmember_rows": oos_excluded_nonmember_rows,
            "evaluation_coverage_ratio": evaluation_coverage_ratio,
            "quantile_returns": quantile_info["annualized_arithmetic_forward_excess_return"],
            "annualized_arithmetic_forward_excess_return": quantile_info["annualized_arithmetic_forward_excess_return"],
            "Q5_minus_Q1": quantile_info["Q5_minus_Q1"],
            "monotonicity_score": quantile_info["monotonicity_score"],
            "quantile_observation_count": quantile_info["quantile_observation_count"],
            "rank_ic_series": rank_ic_series
        }

        logger.info(
            f"OOS 评估结果: Mean IC={metrics['ic_mean']} | "
            f"Mean RankIC={metrics['rank_ic_mean']} | "
            f"RankICIR={metrics['rank_icir']} (Newey-West: {metrics['rank_icir_newey_west']}) | "
            f"RankIC>0 胜率={metrics['rank_ic_win_rate']}% | 评估成分股行数={evaluated_member_rows}/{oos_total_rows}"
        )
        return metrics

    def _compute_newey_west_std(self, series: pd.Series, max_lag: int = 5) -> float:
        """计算 Newey-West (Bartlett 权重) 稳健标准差 (P1-4)"""
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
        将股票每日按预测得分分为 5 组 (Q1~Q5)，评估分层收益的单调性 (P0-6, P1-1)：
        使用模型预测周期的真实未来超额收益率标签 label_col，并折算年化算术超额收益
        """
        df = df.copy()

        def _assign_group(group: pd.DataFrame) -> pd.DataFrame:
            if len(group) >= n_groups:
                try:
                    group["group"] = pd.qcut(
                        group["pred_score"],
                        q=n_groups,
                        labels=[f"Q{i+1}" for i in range(n_groups)],
                        duplicates="drop"
                    )
                except Exception:
                    group["group"] = "Q1"
            else:
                group["group"] = "Q1"
            return group

        grouped_df = df.groupby("date", group_keys=False).apply(_assign_group)
        annual_factor = (242.0 / settings.LABEL_HORIZON) * 100.0
        group_means = grouped_df.groupby("group", observed=False)[label_col].mean() * annual_factor
        
        returns_dict = {str(k): round(float(v), 2) for k, v in group_means.items()}
        q1_ret = returns_dict.get("Q1", 0.0)
        q5_ret = returns_dict.get("Q5", 0.0)
        q5_minus_q1 = round(q5_ret - q1_ret, 2)

        # 单调性得分 (Spearman Rank Corr)
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
            "Q5_minus_Q1": q5_minus_q1,
            "monotonicity_score": monotonicity,
            "quantile_observation_count": int(grouped_df["group"].count())
        }
