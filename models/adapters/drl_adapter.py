"""
DRL 强化学习模型适配器 (models/adapters/drl_adapter.py)
用于加载并运行 DRL 强化学习自适应量化模型 (DRLStrengthenedQuantModel)
"""
from pathlib import Path
from typing import List, Optional, Any, Union
import joblib
import pandas as pd
import numpy as np

from models.adapters.base_adapter import BaseModelAdapter


class DRLAdapter(BaseModelAdapter):
    """
    强化学习模型适配器 (兼容 DRLStrengthenedQuantModel 等 Gen 4/5 增强模型)
    """

    def load(self, artifact_path: Union[str, Path]) -> "DRLAdapter":
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"未找到 DRL 模型制品: {path}")
        self.model = joblib.load(path)
        if hasattr(self.model, "feature_names") and self.model.feature_names:
            self._feature_names = list(self.model.feature_names)
        elif hasattr(self.model, "feature_model") and hasattr(self.model.feature_model, "feature_names"):
            self._feature_names = list(self.model.feature_model.feature_names or [])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DRL 模型未加载，无法执行 predict")
        if hasattr(self.model, "predict_alpha"):
            preds = self.model.predict_alpha(X)
        elif hasattr(self.model, "predict"):
            preds = self.model.predict(X)
        else:
            raise AttributeError(f"DRL 模型对象缺少 predict / predict_alpha 方法: {type(self.model)}")
        return np.asarray(preds, dtype=float)

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is not None:
            return self._feature_names
        if self.model is not None:
            if hasattr(self.model, "feature_names") and self.model.feature_names:
                return list(self.model.feature_names)
            if hasattr(self.model, "feature_model") and hasattr(self.model.feature_model, "feature_names"):
                return list(self.model.feature_model.feature_names or [])
        return super().feature_names
