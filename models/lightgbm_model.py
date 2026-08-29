"""
LightGBM 量化预测模型封装
支持回归与排序目标、早停机制以及特征重要性（Gain/Split）评估与模型持久化
"""
import os
import joblib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from config.settings import settings

logger = logging.getLogger(__name__)


class LightGBMQuantModel:
    """LightGBM 预测引擎封装类 (支持回归与二分类两种模式，支持非对称下行风控惩罚目标)"""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        model_dir: Optional[Path] = None,
        task_type: str = settings.TASK_TYPE,
        use_asymmetric_loss: bool = False
    ):
        self.task_type = task_type
        self.use_asymmetric_loss = use_asymmetric_loss
        # 分类模式使用分类参数，回归模式使用回归参数
        if params is not None:
            self.params = params
        elif task_type == "classification":
            self.params = settings.LGBM_PARAMS_CLF.copy()
        else:
            self.params = settings.LGBM_PARAMS.copy()
        self.model_dir = model_dir or settings.MODELS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_names: List[str] = []
        self.calibrator = None  # Isotonic 概率校准器 (仅分类模式)

        self.custom_obj = None
        if self.use_asymmetric_loss:
            from .asymmetric_loss import AsymmetricLossObjective, AsymmetricRegressionObjective
            if self.task_type == "classification":
                self.custom_obj = AsymmetricLossObjective(false_positive_penalty=2.5)
            else:
                self.custom_obj = AsymmetricRegressionObjective(overpredict_loss=2.5)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> "LightGBMQuantModel":
        """
        训练 LightGBM 模型：
        - 支持 early stopping (基于验证集)
        - 分类模式下自动做 scale_pos_weight 类别平衡
        - 验证集可用时拟合 Isotonic 概率校准器
        """
        self.feature_names = feature_names or list(X_train.columns)
        n_estimators = self.params.pop("n_estimators", 500)
        early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)

        # 分类模式: 自动类别平衡 (负样本数/正样本数)，缓解类别不平衡导致的概率偏斜
        if self.task_type == "classification" and "scale_pos_weight" not in self.params:
            pos = int((y_train == 1).sum())
            neg = int((y_train == 0).sum())
            if pos > 0 and neg > 0:
                self.params["scale_pos_weight"] = neg / pos
        
        fit_params = self.params.copy()

        # 非对称下行风险加权: 对负样本 (下跌样本) 施加 2.5x 惩罚，严惩假阳性/假突破
        if self.use_asymmetric_loss:
            penalty_weights = np.where(y_train == 0, 2.5, 1.0)
            if sample_weight is not None:
                sample_weight = sample_weight * penalty_weights
            else:
                sample_weight = penalty_weights

        if lgb is not None:
            if self.task_type == "classification":
                self.model = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    **fit_params
                )
            else:
                self.model = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    **fit_params
                )

            callbacks = [lgb.log_evaluation(period=0)]
            if X_val is not None and y_val is not None and len(X_val) > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
                eval_set = [(X_val[self.feature_names], y_val)]
            else:
                eval_set = None

            self.model.fit(
                X_train[self.feature_names],
                y_train,
                sample_weight=sample_weight,
                eval_set=eval_set,
                callbacks=callbacks
            )
        else:
            # Fallback 采用 scikit-learn HistGradientBoosting
            if self.task_type == "classification":
                from sklearn.ensemble import HistGradientBoostingClassifier
                self.model = HistGradientBoostingClassifier(
                    max_iter=n_estimators,
                    learning_rate=self.params.get("learning_rate", 0.03),
                    random_state=self.params.get("random_state", 42)
                )
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                self.model = HistGradientBoostingRegressor(
                    max_iter=n_estimators,
                    learning_rate=self.params.get("learning_rate", 0.03),
                    random_state=self.params.get("random_state", 42)
                )
            self.model.fit(X_train[self.feature_names], y_train, sample_weight=sample_weight)

        self.params["early_stopping_rounds"] = early_stopping_rounds
        self.params["n_estimators"] = n_estimators

        # 概率校准 (Isotonic): 用验证集拟合，修正原始概率的可信度 (改善 Brier Score)
        # 概率校准 (Platt Scaling): 使用逻辑斯蒂回归拟合验证集，消除 Isotonic 分段阶梯导致的相同概率坍塌
        self.calibrator = None
        if (
            self.task_type == "classification"
            and settings.PROBABILITY_CALIBRATION
            and X_val is not None and y_val is not None and len(X_val) > 0
        ):
            try:
                from sklearn.linear_model import LogisticRegression
                if hasattr(self.model, "booster_") and self.custom_obj is not None:
                    margin = self.model.booster_.predict(X_val[self.feature_names], raw_score=True)
                    raw_proba = 1.0 / (1.0 + np.exp(-np.clip(margin, -15.0, 15.0)))
                else:
                    raw_proba = self.model.predict_proba(X_val[self.feature_names])[:, 1]
                
                self.calibrator = LogisticRegression(solver="lbfgs", max_iter=200, C=1.0)
                self.calibrator.fit(raw_proba.reshape(-1, 1), y_val.values)
                logger.info("已拟合 Platt Scaling (Logistic) 概率校准器 (严格保持截面单调性与连续区分度)")
            except Exception as e:
                logger.warning(f"概率校准拟合失败，跳过: {e}")
                self.calibrator = None

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测样本得分：
        - 分类模式: 返回上涨概率 P(label=1) ∈ [0, 1] (经 Platt Scaling 连续校准)
        - 回归模式: 返回连续预期超额收益
        """
        if self.model is None:
            raise ValueError("模型尚未训练或加载！")
        if self.task_type == "classification":
            try:
                if hasattr(self.model, "booster_") and self.custom_obj is not None:
                    margin = self.model.booster_.predict(X[self.feature_names], raw_score=True)
                    raw = 1.0 / (1.0 + np.exp(-np.clip(margin, -15.0, 15.0)))
                else:
                    raw = self.model.predict_proba(X[self.feature_names])[:, 1]
            except Exception:
                if hasattr(self.model, "booster_"):
                    margin = self.model.booster_.predict(X[self.feature_names], raw_score=True)
                else:
                    margin = self.model.predict(X[self.feature_names])
                raw = 1.0 / (1.0 + np.exp(-np.clip(margin, -15.0, 15.0)))
            if self.calibrator is not None:
                return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
            return raw
        return self.model.predict(X[self.feature_names])

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        提取特征重要度 (Gain 贡献与 Split 次数)
        """
        if self.model is None:
            raise ValueError("模型尚未训练！")

        if hasattr(self.model, "booster_"):
            booster = self.model.booster_
            gain_imp = booster.feature_importance(importance_type="gain")
            split_imp = booster.feature_importance(importance_type="split")
        else:
            gain_imp = np.zeros(len(self.feature_names))
            split_imp = np.zeros(len(self.feature_names))

        imp_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance_gain": gain_imp,
            "importance_split": split_imp
        })

        total_gain = imp_df["importance_gain"].sum()
        if total_gain > 0:
            imp_df["importance_pct"] = (imp_df["importance_gain"] / total_gain) * 100.0
        else:
            imp_df["importance_pct"] = 0.0

        imp_df.sort_values(by="importance_gain", ascending=False, inplace=True)
        imp_df.reset_index(drop=True, inplace=True)
        return imp_df.head(top_n)

    def save(self, filepath: Optional[Path] = None) -> Path:
        """保存模型及特征元数据"""
        save_path = filepath or (self.model_dir / "latest_lightgbm.pkl")
        joblib.dump({
            "model": self.model,
            "features": self.feature_names,
            "params": self.params,
            "task_type": self.task_type,
            "calibrator": self.calibrator
        }, save_path)
        logger.info(f"模型已保存至: {save_path}")
        return save_path

    def load(self, filepath: Optional[Path] = None) -> "LightGBMQuantModel":
        """加载已保存的模型"""
        load_path = filepath or (self.model_dir / "latest_lightgbm.pkl")
        if not load_path.exists():
            raise FileNotFoundError(f"未找到模型文件: {load_path}")
        data = joblib.load(load_path)
        self.model = data["model"]
        self.feature_names = data["features"]
        self.params = data.get("params", self.params)
        self.task_type = data.get("task_type", self.task_type)
        self.calibrator = data.get("calibrator", None)
        logger.info(f"成功从 {load_path} 加载模型，特征数: {len(self.feature_names)} (task_type={self.task_type})")
        return self
