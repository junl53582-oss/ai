"""
多随机种子袋装集成模型 (models/bagging_ensemble.py)
用于击碎 Seed 方差、提升泛化能力与极端宏观风格周期的穿透力。
数学原理:
E[y_ensemble] = 1/K * sum(E[y_k])
Var(y_ensemble) = 1/K * Var(y_k) (在模型误差弱相关前提下方差衰减)
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from config.settings import settings
from models.lightgbm_model import LightGBMQuantModel

logger = logging.getLogger(__name__)

class MultiSeedBaggingModel:
    """多随机种子与特征子空间强正则化袋装集成模型"""

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
        strict_mode: bool = False
    ):
        self.seeds = seeds or [42, 100, 2024, 7, 999]
        self.task_type = task_type
        self.strict_mode = strict_mode
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

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> "MultiSeedBaggingModel":
        self.feature_names = feature_names or list(X_train.columns)
        self.models = []

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
            m.fit(X_train, y_train, X_val=X_val, y_val=y_val, sample_weight=sample_weight, feature_names=self.feature_names)
            self.models.append(m)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("模型尚未训练完成，无法执行 predict")
        preds = [m.predict(X) for m in self.models]
        # 多种子打分等权均值化 (抹平种子抖动)
        return np.mean(preds, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("模型尚未训练完成，无法执行 predict_proba")
        probas = [m.predict_proba(X) for m in self.models]
        return np.mean(probas, axis=0)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """多模型集成特征重要性加权平均"""
        if not self.models:
            return pd.DataFrame()
        dfs = [m.get_feature_importance(top_n=len(self.feature_names)) for m in self.models]
        merged = pd.concat(dfs).groupby("feature")["importance_pct"].mean().reset_index()
        merged.sort_values(by="importance_pct", ascending=False, inplace=True)
        return merged.head(top_n).reset_index(drop=True)
