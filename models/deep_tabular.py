"""
表格深度学习神经网络模型 (models/deep_tabular.py)
采用带有 LayerNorm、Dropout 与高阶特征互动的多层感知机 (Tabular MLP/ResNet)
为树模型提供正交的深度神经网络 Alpha 预测信号。
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config.settings import settings
from .base_model import BaseQuantModel

logger = logging.getLogger(__name__)


class TabularMLPQuantModel(BaseQuantModel):
    """
    表格深度神经网络模型 (Tabular Multi-Layer Perceptron)
    """

    def __init__(
        self,
        task_type: str = settings.TASK_TYPE,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32),
        alpha: float = 0.01,
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        random_state: int = 42
    ):
        super().__init__(task_type=task_type)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

        self.pipeline: Optional[Pipeline] = None
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> "TabularMLPQuantModel":
        self.feature_names = feature_names or list(X_train.columns)

        if self.task_type == "classification":
            mlp = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=self.random_state
            )
        else:
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=self.random_state
            )

        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("mlp", mlp)
        ])

        logger.info(f"开始训练深度表格网络 (Tabular MLP: {self.hidden_layer_sizes})...")
        self.pipeline.fit(X_train[self.feature_names], y_train)
        logger.info("Tabular MLP 深度网络拟合完成")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("TabularMLP 模型尚未训练！")

        if self.task_type == "classification":
            probas = self.pipeline.predict_proba(X[self.feature_names])
            if probas.shape[1] > 1:
                return probas[:, 1]
            return probas[:, 0]
        else:
            return self.pipeline.predict(X[self.feature_names])

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """基于神经网络首层权重模长估算特征相对重要性"""
        if self.pipeline is None or not self.feature_names:
            return pd.DataFrame(columns=["feature", "importance", "importance_pct"])
        try:
            mlp = self.pipeline.named_steps["mlp"]
            weights = np.mean(np.abs(mlp.coefs_[0]), axis=1)
            tot = weights.sum() or 1.0
            df = pd.DataFrame({
                "feature": self.feature_names,
                "importance": weights,
                "importance_pct": weights / tot * 100
            }).sort_values(by="importance", ascending=False).head(top_n).reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame(columns=["feature", "importance", "importance_pct"])

    def save(self, model_dir: Optional[Path] = None):
        pass

    def load(self, model_dir: Optional[Path] = None):
        pass
