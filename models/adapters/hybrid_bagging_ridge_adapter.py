"""
Hybrid Bagging Ridge 模型适配器 (models/adapters/hybrid_bagging_ridge_adapter.py)
用于加载并运行多随机种子袋装浅树 + L2 线性正则化底仓混合模型 (MultiSeedBaggingModel)
"""
from pathlib import Path
from typing import List, Optional, Any, Union
import joblib
import pandas as pd
import numpy as np

from models.adapters.base_adapter import BaseModelAdapter


class HybridBaggingRidgeAdapter(BaseModelAdapter):
    """
    多随机种子袋装浅树 + Ridge 线性底仓混合模型适配器
    (兼容 MultiSeedBaggingModel / Bagging Ensemble 系列模型制品)
    """

    def load(self, artifact_path: Union[str, Path]) -> "HybridBaggingRidgeAdapter":
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"未找到混合袋装模型制品: {path}")
        self.model = joblib.load(path)
        if hasattr(self.model, "feature_names") and self.model.feature_names:
            self._feature_names = list(self.model.feature_names)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("混合袋装模型未加载，无法执行 predict")
        preds = self.model.predict(X)
        return np.asarray(preds, dtype=float)

    @property
    def feature_names(self) -> List[str]:
        if self.model is not None and hasattr(self.model, "feature_names") and self.model.feature_names:
            return list(self.model.feature_names)
        return super().feature_names
