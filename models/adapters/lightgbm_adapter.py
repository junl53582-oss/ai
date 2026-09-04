"""
LightGBM 模型适配器 (models/adapters/lightgbm_adapter.py)
"""
from pathlib import Path
from typing import List, Optional, Any, Union
import pandas as pd
import numpy as np

from models.adapters.base_adapter import BaseModelAdapter


class LightGBMAdapter(BaseModelAdapter):
    """LightGBM 系列模型适配器 (封装 LightGBMQuantModel)"""

    def load(self, artifact_path: Union[str, Path]) -> "LightGBMAdapter":
        from models.lightgbm_model import LightGBMQuantModel
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"未找到 LightGBM 模型制品: {path}")
        m = LightGBMQuantModel(task_type=self.task_type)
        self.model = m.load(path)
        if hasattr(self.model, "feature_names") and self.model.feature_names:
            self._feature_names = list(self.model.feature_names)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBM 模型未加载，无法执行 predict")
        preds = self.model.predict(X)
        return np.asarray(preds, dtype=float)

    @property
    def feature_names(self) -> List[str]:
        if self.model is not None and hasattr(self.model, "feature_names") and self.model.feature_names:
            return list(self.model.feature_names)
        return super().feature_names
