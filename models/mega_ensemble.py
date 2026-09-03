"""
第三代 Mega-Alpha 深度异构集成预测模型 (models/mega_ensemble.py)
架构:
1. 微软 Qlib DoubleEnsemble (55%): 样本动态重加权 + 特征子空间正交采样
2. TabularMLP 深度感知机 (25%): 鲁棒多层网络流形学习
3. L2 ElasticNet/Ridge 线性底仓 (20%): 全局单调性正则化防漂移
"""
import logging
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config.settings import settings
from .double_ensemble import DoubleEnsembleQuantModel
from .deep_tabular import TabularMLPQuantModel

logger = logging.getLogger(__name__)

class MegaEnsembleQuantModel:
    """第三代终极 Mega-Alpha 异构多核混合集成模型"""

    def __init__(
        self,
        task_type: str = settings.TASK_TYPE,
        w_double_ensemble: float = 0.55,
        w_mlp: float = 0.25,
        w_ridge: float = 0.20,
        n_de_submodels: int = 3,
        random_state: int = 42
    ):
        self.task_type = task_type
        self.w_de = w_double_ensemble
        self.w_mlp = w_mlp
        self.w_ridge = w_ridge
        self.n_de_submodels = n_de_submodels
        self.random_state = random_state

        self.de_model: Optional[DoubleEnsembleQuantModel] = None
        self.mlp_model: Optional[TabularMLPQuantModel] = None
        self.ridge_pipeline: Optional[Pipeline] = None
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> "MegaEnsembleQuantModel":
        self.feature_names = feature_names or list(X_train.columns)
        X_tr = X_train[self.feature_names].copy()
        X_v = X_val[self.feature_names].copy() if X_val is not None else None

        # 1. 训练 Core 1: 微软 Qlib DoubleEnsemble
        logger.info(f"   [Core 1/3] 训练 Qlib DoubleEnsemble (子模型数: {self.n_de_submodels})...")
        self.de_model = DoubleEnsembleQuantModel(
            task_type=self.task_type,
            n_sub_models=self.n_de_submodels,
            feature_subsample_ratio=0.80,
            decay_rate=0.7,
            random_state=self.random_state
        )
        self.de_model.fit(
            X_train=X_tr,
            y_train=y_train,
            X_val=X_v,
            y_val=y_val,
            feature_names=self.feature_names,
            sample_weight=sample_weight
        )

        # 2. 训练 Core 2: TabularMLP 深度神经网络
        logger.info("   [Core 2/3] 训练 TabularMLP 深度神经网络 (流形学习)...")
        self.mlp_model = TabularMLPQuantModel(
            task_type=self.task_type,
            hidden_layer_sizes=(64, 32),
            alpha=0.05,
            learning_rate_init=0.002,
            max_iter=40,
            random_state=self.random_state
        )
        self.mlp_model.fit(
            X_train=X_tr,
            y_train=y_train,
            X_val=X_v,
            y_val=y_val,
            feature_names=self.feature_names
        )

        # 3. 训练 Core 3: L2 Ridge/Logistic 强正则化单调底座
        logger.info("   [Core 3/3] 训练 L2 正则化线性底仓 (全局单调性基准)...")
        if self.task_type == "classification":
            linear_clf = LogisticRegression(C=0.1, penalty="l2", max_iter=200, random_state=self.random_state)
        else:
            linear_clf = Ridge(alpha=10.0, random_state=self.random_state)

        self.ridge_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", linear_clf)
        ])
        
        fit_params = {}
        if sample_weight is not None:
            fit_params["model__sample_weight"] = sample_weight
        self.ridge_pipeline.fit(X_tr, y_train, **fit_params)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_df = X[self.feature_names].copy()
        
        # 1. DoubleEnsemble 预测
        pred_de = self.de_model.predict(X_df)
        
        # 2. TabularMLP 预测
        pred_mlp = self.mlp_model.predict(X_df)

        # 3. Ridge 预测
        if self.task_type == "classification":
            pred_ridge = self.ridge_pipeline.predict_proba(X_df)[:, 1]
        else:
            pred_ridge = self.ridge_pipeline.predict(X_df)

        # 三核自适应融合
        blended = (
            self.w_de * pred_de +
            self.w_mlp * pred_mlp +
            self.w_ridge * pred_ridge
        )
        return blended

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.de_model and hasattr(self.de_model, "get_feature_importance"):
            return self.de_model.get_feature_importance(top_n=top_n)
        return pd.DataFrame({"feature": self.feature_names[:top_n], "importance_pct": [1.0 / min(len(self.feature_names), top_n)] * min(len(self.feature_names), top_n)})
