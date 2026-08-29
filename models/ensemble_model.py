"""
多模型动态集成融合引擎 (models/ensemble_model.py)
基于 Model Zoo 架构，支持 LightGBM、随机森林 (Random Forest)、L2正则化模型 (Ridge/Logistic)
以及可选 XGBoost/CatBoost 的多模型自适应加权融合 (Adaptive Ensemble Blending)。
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from config.settings import settings
from .base_model import BaseQuantModel
from .lightgbm_model import LightGBMQuantModel

logger = logging.getLogger(__name__)


class EnsembleQuantModel(BaseQuantModel):
    """自适应多模型加权集成融合器"""

    def __init__(
        self,
        task_type: str = settings.TASK_TYPE,
        model_types: Optional[List[str]] = None,
        weighting_strategy: str = "dynamic_val_score",  # dynamic_val_score / equal
        model_dir: Optional[Path] = None
    ):
        self.task_type = task_type
        self.model_types = model_types or ["lightgbm", "random_forest", "linear"]
        self.weighting_strategy = weighting_strategy
        self.model_dir = model_dir or settings.MODELS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.feature_names: List[str] = []
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> "EnsembleQuantModel":
        """训练各个子模型并动态计算融合权重"""
        self.feature_names = feature_names or list(X_train.columns)
        X_tr = X_train[self.feature_names].copy()
        X_v = X_val[self.feature_names].copy() if X_val is not None else None

        # 准备标准填充与归一化数据 (供线性与RF使用)
        X_tr_imp = self.imputer.fit_transform(X_tr)
        X_tr_scaled = self.scaler.fit_transform(X_tr_imp)
        
        if X_v is not None:
            X_v_imp = self.imputer.transform(X_v)
            X_v_scaled = self.scaler.transform(X_v_imp)
        else:
            X_v_imp = None
            X_v_scaled = None

        val_scores: Dict[str, float] = {}

        # 1. 训练 LightGBM
        if "lightgbm" in self.model_types:
            logger.info("  -> 训练集成子模型 [1]: LightGBM...")
            lgb_model = LightGBMQuantModel(task_type=self.task_type)
            lgb_model.fit(X_train=X_tr, y_train=y_train, X_val=X_v, y_val=y_val, feature_names=self.feature_names)
            self.models["lightgbm"] = lgb_model

            if X_v is not None and y_val is not None:
                preds = lgb_model.predict(X_v)
                val_scores["lightgbm"] = self._calc_val_metric(y_val, preds)
            else:
                val_scores["lightgbm"] = 1.0

        # 2. 训练 Random Forest
        if "random_forest" in self.model_types:
            logger.info("  -> 训练集成子模型 [2]: Random Forest...")
            if self.task_type == "classification":
                rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=50,
                    n_jobs=-1,
                    random_state=42
                )
                rf.fit(X_tr_imp, y_train)
                if X_v_imp is not None and y_val is not None:
                    preds = rf.predict_proba(X_v_imp)[:, 1]
                    val_scores["random_forest"] = self._calc_val_metric(y_val, preds)
                else:
                    val_scores["random_forest"] = 1.0
            else:
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=50,
                    n_jobs=-1,
                    random_state=42
                )
                rf.fit(X_tr_imp, y_train)
                if X_v_imp is not None and y_val is not None:
                    preds = rf.predict(X_v_imp)
                    val_scores["random_forest"] = self._calc_val_metric(y_val, preds)
                else:
                    val_scores["random_forest"] = 1.0
            self.models["random_forest"] = rf

        # 3. 训练 Regularized Linear (Ridge / Logistic)
        if "linear" in self.model_types:
            logger.info("  -> 训练集成子模型 [3]: L2 Regularized Linear...")
            if self.task_type == "classification":
                lr = LogisticRegression(C=0.1, penalty="l2", max_iter=200, random_state=42)
                lr.fit(X_tr_scaled, y_train, sample_weight=sample_weight)
                if X_v_scaled is not None and y_val is not None:
                    preds = lr.predict_proba(X_v_scaled)[:, 1]
                    val_scores["linear"] = self._calc_val_metric(y_val, preds)
                else:
                    val_scores["linear"] = 1.0
            else:
                lr = Ridge(alpha=10.0, random_state=42)
                lr.fit(X_tr_scaled, y_train, sample_weight=sample_weight)
                if X_v_scaled is not None and y_val is not None:
                    preds = lr.predict(X_v_scaled)
                    val_scores["linear"] = self._calc_val_metric(y_val, preds)
                else:
                    val_scores["linear"] = 1.0
            self.models["linear"] = lr

        # 4. 训练 Tabular MLP 深度网络
        if "mlp" in self.model_types:
            logger.info("  -> 训练集成子模型 [4]: Tabular MLP 深度网络...")
            from .deep_tabular import TabularMLPQuantModel
            mlp_model = TabularMLPQuantModel(task_type=self.task_type)
            mlp_model.fit(X_train=X_tr, y_train=y_train, X_val=X_v, y_val=y_val, feature_names=self.feature_names)
            self.models["mlp"] = mlp_model
            if X_v is not None and y_val is not None:
                preds = mlp_model.predict(X_v)
                val_scores["mlp"] = self._calc_val_metric(y_val, preds)
            else:
                val_scores["mlp"] = 1.0

        # 5. 训练 Double Ensemble (微软 Qlib)
        if "double_ensemble" in self.model_types:
            logger.info("  -> 训练集成子模型 [5]: 微软 Qlib DoubleEnsemble...")
            from .double_ensemble import DoubleEnsembleQuantModel
            de_model = DoubleEnsembleQuantModel(task_type=self.task_type, n_sub_models=3)
            de_model.fit(X_train=X_tr, y_train=y_train, X_val=X_v, y_val=y_val, feature_names=self.feature_names, sample_weight=sample_weight)
            self.models["double_ensemble"] = de_model
            if X_v is not None and y_val is not None:
                preds = de_model.predict(X_v)
                val_scores["double_ensemble"] = self._calc_val_metric(y_val, preds)
            else:
                val_scores["double_ensemble"] = 1.0

        # 6. 计算融合权重
        self._compute_weights(val_scores)
        logger.info(f"多模型融合权重配置完成: {self.model_weights}")
        return self

    def _calc_val_metric(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """计算验证集评估分"""
        try:
            if self.task_type == "classification":
                # AUC 分数
                if len(np.unique(y_true)) > 1:
                    score = roc_auc_score(y_true, y_pred)
                    return max(0.01, float(score))
                return 0.5
            else:
                # RankIC 分数
                from scipy.stats import spearmanr
                corr, _ = spearmanr(y_true, y_pred)
                return max(0.001, float(corr)) if not np.isnan(corr) else 0.001
        except Exception:
            return 0.5

    def _compute_weights(self, val_scores: Dict[str, float]):
        """根据验证集得分动态 Softmax 或归一化计算权重"""
        if self.weighting_strategy == "equal" or not val_scores:
            n = len(self.models)
            self.model_weights = {k: 1.0 / n for k in self.models.keys()}
            return

        # 给予基础主力 LightGBM 较高的基准先验
        scores = np.array(list(val_scores.values()))
        # 温度缩放 Softmax
        exp_s = np.exp(scores * 4.0)
        norm_w = exp_s / np.sum(exp_s)

        self.model_weights = {}
        for (m_name, _), w in zip(val_scores.items(), norm_w):
            self.model_weights[m_name] = round(float(w), 4)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """多模型加权融合预测"""
        X_df = X[self.feature_names].copy()
        X_imp = self.imputer.transform(X_df)
        X_scaled = self.scaler.transform(X_imp)

        blended_preds = np.zeros(len(X_df), dtype=float)

        for m_name, model in self.models.items():
            w = self.model_weights.get(m_name, 0.0)
            if w <= 0.0:
                continue

            if m_name == "lightgbm":
                preds = model.predict(X_df)
            elif m_name == "random_forest":
                if self.task_type == "classification":
                    preds = model.predict_proba(X_imp)[:, 1]
                else:
                    preds = model.predict(X_imp)
            elif m_name == "linear":
                if self.task_type == "classification":
                    preds = model.predict_proba(X_scaled)[:, 1]
                else:
                    preds = model.predict(X_scaled)
            elif m_name in ("mlp", "double_ensemble"):
                preds = model.predict(X_df)
            else:
                preds = np.zeros(len(X_df))

            blended_preds += w * preds

        return blended_preds

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """提取主力树模型的特征重要度"""
        if "lightgbm" in self.models:
            return self.models["lightgbm"].get_feature_importance(top_n=top_n)
        
        # 回退为 RF 特征重要性
        if "random_forest" in self.models:
            rf = self.models["random_forest"]
            imp = rf.feature_importances_
            tot = imp.sum() or 1.0
            df = pd.DataFrame({
                "feature": self.feature_names,
                "importance": imp,
                "importance_pct": imp / tot * 100
            }).sort_values(by="importance", ascending=False).head(top_n).reset_index(drop=True)
            return df

        return pd.DataFrame(columns=["feature", "importance", "importance_pct"])

    def save(self, filepath: Optional[Path] = None) -> Path:
        fp = filepath or self.model_dir / "latest_ensemble.pkl"
        joblib.dump({
            "models": self.models,
            "weights": self.model_weights,
            "features": self.feature_names,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "task_type": self.task_type
        }, fp)
        logger.info(f"集成模型已保存至: {fp}")
        return fp

    def load(self, filepath: Path) -> "EnsembleQuantModel":
        data = joblib.load(filepath)
        self.models = data["models"]
        self.model_weights = data["weights"]
        self.feature_names = data["features"]
        self.imputer = data["imputer"]
        self.scaler = data["scaler"]
        self.task_type = data["task_type"]
        return self
