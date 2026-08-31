"""
LightGBM 量化预测模型封装 (models/lightgbm_model.py)
支持回归与排序目标、早停机制以及特征重要性（Gain/Split）评估与模型持久化
Research Integrity Hardened:
- 严格 Model Identity 门禁，LightGBM 缺失时在 strict_mode 下直接 Fail-Closed，禁止静默回退到 HistGradientBoosting
- 输出详细 Model Identity 元数据 (requested_estimator, actual_estimator, library, library_version, config_hash)
- 严格生产模型目录物理隔离
"""
from __future__ import annotations

import os
import json
import hashlib
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
    """LightGBM 预测引擎封装类 (支持回归与二分类两种模式，支持非对称下行风控惩罚目标与严格模型身份核验)"""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        model_dir: Optional[Path] = None,
        task_type: str = settings.TASK_TYPE,
        use_asymmetric_loss: bool = False,
        random_state: Optional[int] = None,
        strict_mode: bool = True
    ):
        self.task_type = task_type
        self.use_asymmetric_loss = use_asymmetric_loss
        self.strict_mode = bool(strict_mode)

        if params is not None:
            self.params = params.copy()
        elif task_type == "classification":
            self.params = settings.LGBM_PARAMS_CLF.copy()
        else:
            self.params = settings.LGBM_PARAMS.copy()

        self.random_state = int(random_state) if random_state is not None else int(self.params.get("random_state", 42))
        self.params["random_state"] = self.random_state
        self.params["feature_fraction_seed"] = self.random_state
        self.params["bagging_seed"] = self.random_state
        self.params["data_random_seed"] = self.random_state

        if model_dir is not None:
            self.model_dir = Path(model_dir)
            prod_root = Path(settings.MODELS_DIR).resolve()
            resolved_md = self.model_dir.resolve()
            if self.strict_mode and (resolved_md == prod_root or prod_root in resolved_md.parents):
                raise RuntimeError(
                    f"FATAL: Research runner attempted to configure model_dir directly to production directory {settings.MODELS_DIR}!"
                )
            self.model_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.model_dir = None

        self.model = None
        self.feature_names: List[str] = []
        self.calibrator = None
        self.identity_metadata: Dict[str, Any] = {}

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
        - 验证集可用时拟合 Isotonic / Platt Scaling 概率校准器
        - 严格执行 Model Identity 门禁
        """
        self.feature_names = feature_names or list(X_train.columns)
        n_estimators = self.params.pop("n_estimators", 500)
        early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)

        # 严格 Model Identity 门禁 (Model Identity Fail-Closed Gate)
        if lgb is None and self.strict_mode:
            raise RuntimeError(
                "FATAL: LightGBM is requested but not installed/importable in strict certified mode! "
                "Silent fallback to HistGradientBoosting is strictly disallowed."
            )

        if self.task_type == "classification" and "scale_pos_weight" not in self.params:
            pos = int((y_train == 1).sum())
            neg = int((y_train == 0).sum())
            if pos > 0 and neg > 0:
                self.params["scale_pos_weight"] = neg / pos

        fit_params = self.params.copy()

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
                est_name = "LGBMClassifier"
            elif self.task_type == "ranking":
                if "objective" not in fit_params:
                    fit_params["objective"] = "lambdarank"
                self.model = lgb.LGBMRanker(
                    n_estimators=n_estimators,
                    **fit_params
                )
                est_name = "LGBMRanker"
            else:
                self.model = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    **fit_params
                )
                est_name = "LGBMRegressor"

            lib_name = "lightgbm"
            lib_ver = getattr(lgb, "__version__", "unknown")

            callbacks = [lgb.log_evaluation(period=0)]
            eval_set = None
            eval_group = None
            train_group = None

            if self.task_type == "ranking":
                if "date" in X_train.columns:
                    train_group = list(X_train.groupby("date", sort=False).size())
                if X_val is not None and "date" in X_val.columns:
                    eval_group = [list(X_val.groupby("date", sort=False).size())]

            if X_val is not None and y_val is not None and len(X_val) > 0:
                y_tr_classes = set(np.unique(y_train))
                y_val_classes = set(np.unique(y_val))
                if self.task_type == "classification" and (y_val_classes - y_tr_classes):
                    eval_set = None
                else:
                    callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
                    eval_set = [(X_val[self.feature_names], y_val)]
            else:
                eval_set = None

            fit_kwargs = {
                "sample_weight": sample_weight,
                "callbacks": callbacks
            }
            if eval_set is not None and len(eval_set) > 0:
                import inspect
                fit_sig = inspect.signature(self.model.fit)
                if "eval_X" in fit_sig.parameters and "eval_y" in fit_sig.parameters:
                    fit_kwargs["eval_X"] = eval_set[0][0]
                    fit_kwargs["eval_y"] = eval_set[0][1]
                else:
                    fit_kwargs["eval_set"] = eval_set

            if train_group is not None:
                fit_kwargs["group"] = train_group
            if eval_group is not None:
                fit_kwargs["eval_group"] = eval_group

            self.model.fit(
                X_train[self.feature_names],
                y_train,
                **fit_kwargs
            )
        else:
            # Fallback 采用 scikit-learn HistGradientBoosting (仅在 strict_mode=False 时允许)
            import sklearn
            lib_name = "scikit-learn"
            lib_ver = getattr(sklearn, "__version__", "unknown")
            if self.task_type == "classification":
                from sklearn.ensemble import HistGradientBoostingClassifier
                self.model = HistGradientBoostingClassifier(
                    max_iter=n_estimators,
                    learning_rate=self.params.get("learning_rate", 0.03),
                    random_state=self.params.get("random_state", 42)
                )
                est_name = "HistGradientBoostingClassifier"
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                self.model = HistGradientBoostingRegressor(
                    max_iter=n_estimators,
                    learning_rate=self.params.get("learning_rate", 0.03),
                    random_state=self.params.get("random_state", 42)
                )
                est_name = "HistGradientBoostingRegressor"
            self.model.fit(X_train[self.feature_names], y_train, sample_weight=sample_weight)

        self.params["early_stopping_rounds"] = early_stopping_rounds
        self.params["n_estimators"] = n_estimators

        # 记录完整 Model Identity 元数据
        config_str = json.dumps(self.params, sort_keys=True, default=str)
        self.identity_metadata = {
            "requested_estimator": "LightGBM",
            "actual_estimator": est_name,
            "library": lib_name,
            "library_version": lib_ver,
            "model_class": self.model.__class__.__name__,
            "config_hash": hashlib.sha256(config_str.encode("utf-8")).hexdigest(),
            "strict_identity_verified": bool(lib_name == "lightgbm")
        }

        # 概率校准 (Platt Scaling)
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

                lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
                lr.fit(raw_proba.reshape(-1, 1), y_val)
                self.calibrator = lr
                logger.info("已完成 Platt Scaling (Logistic) 概率校准器拟合")
            except Exception as e:
                logger.warning(f"Platt Scaling 概率校准拟合失败: {e}，将使用原始概率输出")
                self.calibrator = None

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        根据任务类型生成预测输出：
        - 分类任务: 输出正类概率 P(y=1)
        - 回归任务: 输出预期连续超额收益
        - 排序任务: 输出相对排序预测值
        """
        if self.model is None:
            raise ValueError("模型尚未训练！请先调用 fit()")

        X_feat = X[self.feature_names]

        if self.task_type == "classification":
            if hasattr(self.model, "booster_") and self.custom_obj is not None:
                margin = self.model.booster_.predict(X_feat, raw_score=True)
                raw_proba = 1.0 / (1.0 + np.exp(-np.clip(margin, -15.0, 15.0)))
            else:
                raw_proba = self.model.predict_proba(X_feat)[:, 1]

            if self.calibrator is not None:
                try:
                    cal_proba = self.calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
                    return cal_proba
                except Exception as e:
                    logger.warning(f"概率校准推理失败: {e}，回退至原始概率")
                    return raw_proba
            return raw_proba
        else:
            return self.model.predict(X_feat)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """获取特征重要性 (Split 次数与 Gain 增益贡献)"""
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
            "importance": gain_imp,
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

    def save(self, filepath: Optional[Path] = None, allow_production_write: bool = False) -> Path:
        """保存模型及特征元数据 (严格隔离保存路径，杜绝非授权写入生产目录)"""
        if filepath is not None:
            save_path = Path(filepath)
        elif self.model_dir is not None:
            save_path = self.model_dir / "latest_lightgbm.pkl"
        else:
            raise RuntimeError("Cannot save model: No model_dir or filepath provided!")

        prod_root = Path(settings.MODELS_DIR).resolve()
        resolved_save = save_path.resolve()
        if (resolved_save == prod_root or prod_root in resolved_save.parents) and not allow_production_write:
            raise RuntimeError(
                f"FATAL: Direct LightGBMQuantModel.save attempted to write into production directory {settings.MODELS_DIR}! "
                f"Production models must be upgraded only through formal promotion workflows."
            )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "features": self.feature_names,
            "params": self.params,
            "task_type": self.task_type,
            "calibrator": self.calibrator,
            "identity_metadata": self.identity_metadata
        }, save_path)
        logger.info(f"模型已保存至: {save_path}")
        return save_path
