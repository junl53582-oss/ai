"""
多随机种子袋装集成模型 (models/bagging_ensemble.py)
升级版: 支持浅树随机子空间 Bagging + L2 线性正则化底仓 (Hybrid Stacking)
数学原理:
E[y_ensemble] = w_tree * (1/K * sum(y_tree_k)) + w_linear * y_linear
在弱信噪比金融时序中, 线性底仓提供全局单调性锚点, 树模型捕捉高阶非线性 Alpha。
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import RobustScaler

from config.settings import settings
from models.lightgbm_model import LightGBMQuantModel

logger = logging.getLogger(__name__)

class MultiSeedBaggingModel:
    """多随机种子与特征子空间强正则化袋装集成模型 (含线性底仓混合增强)"""

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        task_type: str = settings.TASK_TYPE,
        num_leaves: int = 15,
        min_child_samples: int = 60,
        learning_rate: float = 0.02,
        reg_alpha: float = 1.0,
        reg_lambda: float = 5.0,
        feature_fraction: float = 0.70,
        n_estimators: int = 300,
        strict_mode: bool = False,
        enable_linear_base: bool = True,
        linear_weight: float = 0.20
    ):
        self.seeds = seeds or [42, 100, 2024, 7, 999]
        self.task_type = task_type
        self.strict_mode = strict_mode
        self.enable_linear_base = enable_linear_base
        self.linear_weight = linear_weight
        self.feature_names: List[str] = []

        # 强正则化超参配置
        self.base_params = {
            "objective": "binary" if task_type == "classification" else "regression",
            "metric": "binary_logloss" if task_type == "classification" else "rmse",
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "learning_rate": learning_rate,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "feature_fraction": feature_fraction,
            "n_estimators": n_estimators,
            "n_jobs": 4,
            "verbose": -1
        }
        self.models: List[LightGBMQuantModel] = []
        self.linear_model = None
        self.scaler = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "MultiSeedBaggingModel":
        self.feature_names = [c for c in (feature_names or list(X_train.columns)) if c != 'date']
        self.models = []

        X_tr = X_train[self.feature_names]

        # 1. 训练多种子浅树袋装集成
        for s in self.seeds:
            p = self.base_params.copy()
            p["random_state"] = s
            p["feature_fraction_seed"] = s
            p["bagging_seed"] = s
            p["data_random_seed"] = s

            m = LightGBMQuantModel(
                params=p,
                task_type=self.task_type,
                random_state=s,
                strict_mode=self.strict_mode
            )
            m.fit(X_tr, y_train, X_val=X_val[self.feature_names] if X_val is not None else None, y_val=y_val, sample_weight=sample_weight, feature_names=self.feature_names)
            self.models.append(m)

        # 2. 训练 L2 线性正则化底仓 (RobustScaler + Ridge)
        if self.enable_linear_base:
            try:
                self.scaler = RobustScaler()
                X_clean = X_tr.fillna(0.0)
                X_scaled = self.scaler.fit_transform(X_clean)
                y_clean = np.nan_to_num(y_train.values, nan=0.0)
                if self.task_type == "classification":
                    self.linear_model = RidgeClassifier(alpha=10.0, random_state=42)
                    self.linear_model.fit(X_scaled, y_clean.astype(int), sample_weight=sample_weight)
                else:
                    self.linear_model = Ridge(alpha=10.0, random_state=42)
                    self.linear_model.fit(X_scaled, y_clean, sample_weight=sample_weight)
            except Exception as e:
                logger.warning(f"线性底仓训练异常，退化为纯树集成: {e}")
                self.linear_model = None

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("模型尚未训练完成，无法执行 predict")
        X_eval = X[self.feature_names]
        tree_preds = [m.predict(X_eval) for m in self.models]
        mean_tree = np.mean(tree_preds, axis=0)

        if self.enable_linear_base and self.linear_model is not None:
            try:
                X_scaled = self.scaler.transform(X_eval.fillna(0.0))
                if hasattr(self.linear_model, "decision_function"):
                    lin_raw = self.linear_model.decision_function(X_scaled)
                    # Sigmoid 归一化到 (0, 1)
                    lin_pred = 1.0 / (1.0 + np.exp(-np.clip(lin_raw, -10, 10)))
                else:
                    lin_pred = self.linear_model.predict(X_scaled)
                w_lin = self.linear_weight
                w_tree = 1.0 - w_lin
                return w_tree * mean_tree + w_lin * lin_pred
            except Exception as e:
                logger.warning(f"线性预测计算异常: {e}")
                return mean_tree

        return mean_tree

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """多模型集成特征重要性加权平均"""
        if not self.models:
            return pd.DataFrame()
        dfs = [m.get_feature_importance(top_n=len(self.feature_names)) for m in self.models]
        merged = pd.concat(dfs).groupby("feature")["importance_pct"].mean().reset_index()
        merged.sort_values(by="importance_pct", ascending=False, inplace=True)
        return merged.head(top_n).reset_index(drop=True)
