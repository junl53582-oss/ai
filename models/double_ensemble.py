"""
微软 Qlib 级 Double Ensemble 算法 (models/double_ensemble.py)
参考论文: "DoubleEnsemble: A New Ensemble Framework for Financial Time Series Forecasting"
核心机制:
1. 样本空间动态重加权 (Sample Loss-based Reweighting):
   根据前序子模型的预测残差识别噪声样本与强信号样本，动态调整样本权重，降低金融时间序列中的非平稳噪声干扰。
2. 特征子空间正交采样 (Feature Sub-spacing):
   在特征维度进行互补性子空间抽取，训练多个异构的 LightGBM / GBDT 基学习器，使各子模型之间的预测误差正交化，大幅降低集成方差。
3. 动态验证集权重衰减融合 (Validation-weighted Blending):
   基于验证集 OOS 表现分配 Softmax 权重，输出稳健的横截面 Alpha 打分。
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error

from config.settings import settings
from .base_model import BaseQuantModel
from .lightgbm_model import LightGBMQuantModel

logger = logging.getLogger(__name__)


class DoubleEnsembleQuantModel(BaseQuantModel):
    """
    Double Ensemble 深度集成量化预测模型
    """

    def __init__(
        self,
        task_type: str = settings.TASK_TYPE,
        n_sub_models: int = 5,
        feature_subsample_ratio: float = 0.75,
        decay_rate: float = 0.8,
        random_state: int = 42
    ):
        super().__init__(task_type=task_type)
        self.n_sub_models = n_sub_models
        self.feature_subsample_ratio = feature_subsample_ratio
        self.decay_rate = decay_rate
        self.random_state = random_state

        self.sub_models: List[LightGBMQuantModel] = []
        self.sub_features: List[List[str]] = []
        self.sub_weights: List[float] = []
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> "DoubleEnsembleQuantModel":
        """
        双重集成训练流程:
        1. 循环训练 n_sub_models 个子模型
        2. 特征子空间随机采样 (Feature Sub-spacing)
        3. 样本损失动态重加权 (Sample Reweighting)
        """
        self.feature_names = feature_names or list(X_train.columns)
        n_features = len(self.feature_names)
        n_sub_feats = min(n_features, max(int(n_features * self.feature_subsample_ratio), 1))

        rng = np.random.RandomState(self.random_state)
        current_sample_weight = (
            sample_weight.copy() if sample_weight is not None
            else np.ones(len(X_train), dtype=float)
        )

        self.sub_models.clear()
        self.sub_features.clear()
        self.sub_weights.clear()

        val_scores: List[float] = []

        for m_idx in range(self.n_sub_models):
            # 1. 特征子空间采样
            sub_feats = list(rng.choice(self.feature_names, size=n_sub_feats, replace=False))
            self.sub_features.append(sub_feats)

            # 2. 实例化基学习器 (LightGBM)
            sub_model = LightGBMQuantModel(task_type=self.task_type)
            sub_model.fit(
                X_train=X_train[sub_feats],
                y_train=y_train,
                X_val=X_val[sub_feats] if X_val is not None else None,
                y_val=y_val if y_val is not None else None,
                feature_names=sub_feats,
                sample_weight=current_sample_weight
            )
            self.sub_models.append(sub_model)

            # 3. 评估子模型在验证集上的表现
            if X_val is not None and y_val is not None and len(X_val) > 0:
                val_preds = sub_model.predict(X_val[sub_feats])
                if self.task_type == "classification":
                    try:
                        score = roc_auc_score(y_val, val_preds)
                    except Exception:
                        score = 0.5
                else:
                    score = -mean_squared_error(y_val, val_preds)
            else:
                score = 1.0
            val_scores.append(score)

            # 4. 样本重加权 (Sample Reweighting)
            # 计算训练集残差: 残差越大的样本给予衰减或聚焦 (抑制极端离群噪声)
            train_preds = sub_model.predict(X_train[sub_feats])
            if self.task_type == "classification":
                residuals = np.abs(y_train.values.astype(float) - train_preds)
            else:
                residuals = np.abs(y_train.values.astype(float) - train_preds)

            # 样本权重平滑更新 (加权向高残差有效样本倾斜同时避免被离群点主导)
            res_norm = residuals / (np.nanmean(residuals) + 1e-8)
            current_sample_weight = current_sample_weight * np.exp(-self.decay_rate * (res_norm - 1.0))
            current_sample_weight = np.clip(current_sample_weight, 0.1, 5.0)
            current_sample_weight = current_sample_weight / (np.mean(current_sample_weight) + 1e-8)

            logger.info(f"DoubleEnsemble [子模型 {m_idx+1}/{self.n_sub_models}] 训练完成 | 验证得分: {score:.4f}")

        # 5. 计算子模型融合权重 (Softmax)
        scores_arr = np.array(val_scores)
        exp_s = np.exp((scores_arr - np.max(scores_arr)) / 0.1)
        self.sub_weights = list(exp_s / np.sum(exp_s))
        logger.info(f"DoubleEnsemble 最终子模型融合权重: {[round(w, 3) for w in self.sub_weights]}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """多子模型加权预测"""
        if not self.sub_models:
            raise ValueError("DoubleEnsemble 模型尚未训练！")

        ensemble_preds = np.zeros(len(X), dtype=float)
        for sub_model, sub_feats, w in zip(self.sub_models, self.sub_features, self.sub_weights):
            preds = sub_model.predict(X[sub_feats])
            ensemble_preds += w * preds

        return ensemble_preds

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """聚合各子模型的特征重要度"""
        if not self.sub_models:
            return pd.DataFrame(columns=["feature", "importance", "importance_pct"])
        
        imp_dfs = []
        for sub_m, w in zip(self.sub_models, self.sub_weights):
            imp = sub_m.get_feature_importance(top_n=top_n * 2)
            if not imp.empty:
                imp_val = imp["importance"] if "importance" in imp.columns else imp["importance_gain"]
                sub_df = pd.DataFrame({
                    "feature": imp["feature"],
                    "weighted_imp": imp_val * w
                })
                imp_dfs.append(sub_df)

        if not imp_dfs:
            return pd.DataFrame(columns=["feature", "importance", "importance_pct"])

        all_imp = pd.concat(imp_dfs, ignore_index=True)
        agg = all_imp.groupby("feature")["weighted_imp"].sum().reset_index()
        agg.rename(columns={"weighted_imp": "importance"}, inplace=True)
        tot = agg["importance"].sum() or 1.0
        agg["importance_pct"] = agg["importance"] / tot * 100
        return agg.sort_values(by="importance", ascending=False).head(top_n).reset_index(drop=True)

    def save(self, filepath_or_dir: Optional[Path] = None) -> Path:
        """保存 DoubleEnsemble 模型"""
        import joblib
        target = filepath_or_dir or settings.MODELS_DIR
        if target.is_dir() or target.suffix == "":
            target.mkdir(parents=True, exist_ok=True)
            save_path = target / "double_ensemble_latest.pkl"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            save_path = target

        bundle = {
            "sub_models": self.sub_models,
            "sub_features": self.sub_features,
            "sub_weights": self.sub_weights,
            "feature_names": self.feature_names,
            "task_type": self.task_type,
            "n_sub_models": self.n_sub_models,
            "feature_subsample_ratio": self.feature_subsample_ratio,
            "decay_rate": self.decay_rate,
            "random_state": self.random_state
        }
        joblib.dump(bundle, save_path)
        logger.info(f"DoubleEnsemble 模型已保存至: {save_path}")
        return save_path

    def load(self, filepath_or_dir: Optional[Path] = None) -> "DoubleEnsembleQuantModel":
        """加载 DoubleEnsemble 模型"""
        import joblib
        target = filepath_or_dir or settings.MODELS_DIR
        if target.is_dir() or target.suffix == "":
            load_path = target / "double_ensemble_latest.pkl"
        else:
            load_path = target

        if not load_path.exists():
            raise FileNotFoundError(f"未找到 DoubleEnsemble 模型文件: {load_path}")

        bundle = joblib.load(load_path)
        self.sub_models = bundle["sub_models"]
        self.sub_features = bundle["sub_features"]
        self.sub_weights = bundle["sub_weights"]
        self.feature_names = bundle["feature_names"]
        self.task_type = bundle.get("task_type", self.task_type)
        self.n_sub_models = bundle.get("n_sub_models", self.n_sub_models)
        self.feature_subsample_ratio = bundle.get("feature_subsample_ratio", self.feature_subsample_ratio)
        self.decay_rate = bundle.get("decay_rate", self.decay_rate)
        self.random_state = bundle.get("random_state", self.random_state)
        logger.info(f"成功从 {load_path} 加载 DoubleEnsemble 模型 (包含 {len(self.sub_models)} 个子模型)")
        return self
