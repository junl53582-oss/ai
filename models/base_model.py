"""
量化机器学习模型统一基类接口 (models/base_model.py)
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path


class BaseQuantModel(ABC):
    """量化预测模型抽象基类"""

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> "BaseQuantModel":
        """模型训练"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """模型预测 (分类模式输出上涨概率 [0,1]，回归模式输出预期超额收益)"""
        pass

    @abstractmethod
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """获取特征重要度表"""
        pass

    @abstractmethod
    def save(self, filepath: Optional[Path] = None) -> Path:
        """保存模型"""
        pass

    @abstractmethod
    def load(self, filepath: Path) -> "BaseQuantModel":
        """加载模型"""
        pass
