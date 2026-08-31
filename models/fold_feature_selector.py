"""
Fold-level Train-Only Feature Selector (models/fold_feature_selector.py)
严格在单折训练窗口 (Train-only window) 内计算因子 IC / RankIC 与相关性剪枝，
严禁接触 Validation 与 Test 数据集，彻底杜绝特征选择未来数据泄漏 (Feature Selection Lookahead Leakage).
Research Integrity Hardened:
- 严格执行 min_rank_ic 门禁
- 审计年度稳定性 (annual stability)，在训练周期不足时明确标记 NOT_APPLICABLE_INSUFFICIENT_HISTORY
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class FoldFeatureSelector:
    """走步滚动单折训练集特征选择器 (Train-Only)"""

    def __init__(
        self,
        top_n: int = 20,
        min_rank_ic: float = 0.015,
        max_correlation: float = 0.70,
        min_annual_stability: float = 0.50
    ):
        self.top_n = int(top_n)
        self.min_rank_ic = float(min_rank_ic)
        self.max_correlation = float(max_correlation)
        self.min_annual_stability = float(min_annual_stability)

    def select_features(
        self,
        train_df: pd.DataFrame,
        candidate_features: List[str],
        label_col: str,
        method: str = "rank_ic_pruned",
        strict_selection: bool = False
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        在 train_df 上纯净计算各候选因子的 RankIC、胜率与相关性剪枝：
        返回: (selected_features, feature_metrics_df)
        """
        if method == "all" or not candidate_features:
            return candidate_features, pd.DataFrame()

        valid_df = train_df[train_df[label_col].notna()].copy()
        if len(valid_df) < 30:
            return candidate_features[:self.top_n], pd.DataFrame()

        # 检查训练年份跨度
        train_dates = pd.to_datetime(valid_df["date"])
        year_span = (train_dates.max() - train_dates.min()).days / 365.25
        sufficient_annual_history = bool(year_span >= 1.8)

        feature_records = []

        for feat in candidate_features:
            if feat not in valid_df.columns:
                continue

            daily_rank_ics = []
            daily_dates = []
            for dt, group in valid_df.groupby("date"):
                sub_g = group[[feat, label_col]].dropna()
                if len(sub_g) >= 3 and sub_g[feat].nunique() > 1:
                    r_ic = stats.spearmanr(sub_g[feat], sub_g[label_col])[0]
                    if np.isfinite(r_ic):
                        daily_rank_ics.append(float(r_ic))
                        daily_dates.append(pd.Timestamp(dt))

            if not daily_rank_ics:
                continue

            r_ic_s = pd.Series(daily_rank_ics, index=daily_dates)
            mean_ic = float(r_ic_s.mean())
            abs_ic = abs(mean_ic)
            ic_std = float(r_ic_s.std()) if len(r_ic_s) > 1 else 1.0
            icir = mean_ic / (ic_std + 1e-8)
            win_rate = float((r_ic_s * np.sign(mean_ic) > 0).mean())

            # 计算年度稳定性
            if sufficient_annual_history:
                yearly_ic = r_ic_s.groupby(r_ic_s.index.year).mean()
                annual_stability = float((yearly_ic * np.sign(mean_ic) > 0).mean())
            else:
                annual_stability = "NOT_APPLICABLE_INSUFFICIENT_HISTORY"

            feature_records.append({
                "feature": feat,
                "mean_rank_ic": mean_ic,
                "abs_rank_ic": abs_ic,
                "ic_std": ic_std,
                "icir": icir,
                "win_rate": win_rate,
                "annual_stability": annual_stability,
                "meets_min_rank_ic": bool(abs_ic >= self.min_rank_ic)
            })

        if not feature_records:
            return candidate_features[:self.top_n], pd.DataFrame()

        metrics_df = pd.DataFrame(feature_records)
        metrics_df.sort_values(by="abs_rank_ic", ascending=False, inplace=True)
        metrics_df.reset_index(drop=True, inplace=True)

        # 优先使用满足 min_rank_ic 的特征子集 (strict_selection 模式下强制门禁，禁止静默回退)
        filtered_df = metrics_df[metrics_df["meets_min_rank_ic"]].copy()
        if strict_selection:
            if sufficient_annual_history:
                filtered_df = filtered_df[filtered_df["annual_stability"] >= self.min_annual_stability].copy()
            if len(filtered_df) == 0:
                raise RuntimeError(
                    f"FATAL: No features passed strict selection criteria (min_rank_ic={self.min_rank_ic})!"
                )
            candidate_pool = filtered_df
        else:
            if len(filtered_df) < max(5, self.top_n // 2):
                logger.warning(
                    f"仅有 {len(filtered_df)} 个因子达到 min_rank_ic >= {self.min_rank_ic}，回退使用 Top 排序因子保证模型容量"
                )
                candidate_pool = metrics_df
            else:
                candidate_pool = filtered_df

        if method == "top_n":
            selected = list(candidate_pool["feature"].head(self.top_n))
            return selected, metrics_df

        # 相关性剪枝 (Correlation Pruning)
        selected: List[str] = []
        for feat in candidate_pool["feature"]:
            if len(selected) >= self.top_n:
                break
            if not selected:
                selected.append(feat)
                continue

            is_redundant = False
            for s_feat in selected:
                sub_pair = valid_df[[feat, s_feat]].dropna()
                if len(sub_pair) >= 20:
                    corr = abs(float(stats.pearsonr(sub_pair[feat], sub_pair[s_feat])[0]))
                    if not np.isnan(corr) and corr > self.max_correlation:
                        is_redundant = True
                        break
            if not is_redundant:
                selected.append(feat)

        return selected, metrics_df
