"""
统一量化模型适配器基类 (models/adapters/base_adapter.py)
采用 Adapter Pattern 将异构量化模型统一封装为标准截面推理接口。
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Any, Union
import pandas as pd
import numpy as np


class BaseModelAdapter(ABC):
    """
    统一量化模型适配器基类 (Model Adapter Pattern)
    
    统一异构量化模型制品 (LightGBM, Bagging Ensemble, DRL, PyTorch 等) 的前向推理规范:
    1. load(artifact_path) -> self
    2. predict(X: pd.DataFrame) -> np.ndarray (1D 连续或概率预测得分)
    3. feature_names -> List[str]
    """

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.model: Any = None
        self._feature_names: Optional[List[str]] = None

    @abstractmethod
    def load(self, artifact_path: Union[str, Path]) -> "BaseModelAdapter":
        """从制品路径加载模型对象并初始化特征元数据"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """输入特征截面 DataFrame，输出 1D float 预测值数组"""
        pass

    @property
    def feature_names(self) -> List[str]:
        """模型所需的特征名列表"""
        if self._feature_names is not None:
            return self._feature_names
        if hasattr(self.model, "feature_names") and self.model.feature_names:
            return list(self.model.feature_names)
        return []

    def get_feature_names(self) -> List[str]:
        """兼容性方法: 获取模型所需的特征名列表"""
        return self.feature_names

    def __getattr__(self, name: str) -> Any:
        """透明转发底层模型对象的其他属性与方法"""
        if self.model is not None and hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        model_cls = type(self.model).__name__ if self.model is not None else "None"
        return f"<{self.__class__.__name__}(model={model_cls}, task_type='{self.task_type}', features={len(self.feature_names)})>"
