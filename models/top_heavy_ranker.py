"""
Top-Heavy 头部加权排序与 LambdaMART 优化目标 (models/top_heavy_ranker.py)

核心原理:
标准均方误差 (MSE) 和二分类对截面所有 300 只股票等权惩罚，导致模型把 97% 的拟合能力浪费在
排名第 50~250 名我们根本不买的中庸股票上。
本模块提供两大约束头部排序的强化算法:
1. TopHeavyWeightedObjective (非对称头部加权自定义目标):
   - 截面真实超额处于 Top 15% 的个股赋予 5.0x 的高阶权重梯度
   - 预期做多但实际暴跌的 False Positive 施加 2.5x 惩罚
2. TopHeavyLambdaRankModel (每日横截面 Pairwise LambdaRank 模型):
   - 按交易日作为 query 分组，以 NDCG@10 / NDCG@5 作为优化目标函数
   - 直接针对组合实际持仓的 Top 8~10 只股票的排序正确率优化
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)


class TopHeavyWeightedObjective:
    """头部聚焦与非对称下行惩罚自定义回归损失"""

    def __init__(
        self,
        top_quantile: float = 0.85,
        top_weight_multiplier: float = 5.0,
        false_positive_penalty: float = 2.5
    ):
        self.top_q = top_quantile
        self.top_mult = top_weight_multiplier
        self.fp_penalty = false_positive_penalty

    def __call__(self, preds: np.ndarray, train_data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        LightGBM 自定义目标函数接口
        preds: raw predictions
        train_data: lgb.Dataset 或 numpy array
        """
        if hasattr(train_data, "get_label"):
            labels = train_data.get_label()
        else:
            labels = train_data

        residual = preds - labels

        # 计算样本基础权重: 真实收益率排名前 15% 的核心样本赋予 5.0x 权重
        sample_weight = np.ones_like(labels, dtype=float)
        # 若 labels 包含变异度，取 85% 分位数
        if len(labels) > 20:
            threshold = np.quantile(labels, self.top_q)
            sample_weight = np.where(labels >= threshold, self.top_mult, 1.0)

        # 误判惩罚: 预测买入 (preds > 0) 但实际暴跌 (labels < 0)
        is_false_positive = (preds > 0.0) & (labels < 0.0)
        penalty_mult = np.where(is_false_positive, self.fp_penalty, 1.0)
        
        effective_weight = sample_weight * penalty_mult

        grad = effective_weight * residual
        hess = np.maximum(effective_weight, 1e-4)

        return grad, hess


class TopHeavyLambdaRankModel:
    """横截面 LambdaMART 头部排序强化模型 (NDCG@10 导向)"""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        eval_at: List[int] = [5, 10, 20]
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.eval_at = eval_at
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=eval_at,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: List[str], label_col: str):
        """
        以交易日 date 分组，训练横截面 Pairwise Ranker
        """
        clean_df = df.copy().sort_values(by="date").reset_index(drop=True)
        # 将连续收益率转换为整数 relevance 分数 (0~4 档)
        clean_df["relevance"] = clean_df.groupby("date")[label_col].transform(
            lambda s: pd.qcut(s, q=5, labels=False, duplicates="drop")
        ).fillna(0).astype(int)

        group_sizes = clean_df.groupby("date", sort=False).size().values

        X = clean_df[feature_cols].fillna(0.0)
        y = clean_df["relevance"]

        logger.info(f"正在训练 Top-Heavy LambdaRanker (样本数: {len(X)}, 截面日数量: {len(group_sizes)})...")
        self.model.fit(X, y, group=group_sizes)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TopHeavyLambdaRankModel has not been fitted.")
        X = df[feature_cols].fillna(0.0)
        return self.model.predict(X)
