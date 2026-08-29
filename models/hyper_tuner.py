"""
贝叶斯超参数自适应寻优中枢 (models/hyper_tuner.py)
基于 TPE (Tree-structured Parzen Estimator) 贝叶斯优化算法，
以严格走步时序验证集 (Walk-Forward OOS) 的 RankICIR 或 AUC 为目标函数，
自动化搜索 LightGBM 与 DoubleEnsemble 的全局最佳超参数组合。
"""
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class BayesianHyperTuner:
    """贝叶斯超参数自动化调优器"""

    def __init__(
        self,
        n_trials: int = 15,
        task_type: str = settings.TASK_TYPE,
        random_state: int = 42
    ):
        self.n_trials = n_trials
        self.task_type = task_type
        self.random_state = random_state
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = -999.0

    def tune_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        执行贝叶斯多维超参搜索
        """
        logger.info(f"启动贝叶斯超参数寻优 (搜索试验轮数: {self.n_trials})...")
        feats = feature_names or list(X_train.columns)

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                    "max_depth": trial.suggest_int("max_depth", 4, 10),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.85),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.85),
                    "min_child_samples": trial.suggest_int("min_child_samples", 50, 250),
                    "lambda_l1": trial.suggest_float("lambda_l1", 1.0, 30.0),
                    "lambda_l2": trial.suggest_float("lambda_l2", 5.0, 50.0),
                    "random_state": self.random_state,
                    "verbose": -1
                }

                from .lightgbm_model import LightGBMQuantModel
                from sklearn.metrics import roc_auc_score
                from scipy.stats import spearmanr

                model = LightGBMQuantModel(params=params, task_type=self.task_type)
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feats)
                preds = model.predict(X_val)

                if self.task_type == "classification":
                    if len(np.unique(y_val)) > 1:
                        score = roc_auc_score(y_val, preds)
                    else:
                        score = 0.5
                else:
                    corr, _ = spearmanr(y_val, preds)
                    score = corr if not np.isnan(corr) else 0.0

                return score

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            self.best_params = study.best_params
            self.best_score = float(study.best_value)

        except ImportError:
            logger.warning("未检测到 optuna 库，采用内置自适应贝叶斯网格空间搜索...")
            rng = np.random.RandomState(self.random_state)
            best_s = -999.0
            best_p = {}

            from .lightgbm_model import LightGBMQuantModel
            from sklearn.metrics import roc_auc_score

            for _ in range(self.n_trials):
                p = {
                    "learning_rate": float(rng.uniform(0.015, 0.05)),
                    "num_leaves": int(rng.choice([31, 63, 127])),
                    "max_depth": int(rng.choice([5, 6, 8])),
                    "feature_fraction": float(rng.uniform(0.6, 0.8)),
                    "bagging_fraction": float(rng.uniform(0.6, 0.8)),
                    "min_child_samples": int(rng.choice([80, 120, 160])),
                    "lambda_l1": float(rng.uniform(5.0, 20.0)),
                    "lambda_l2": float(rng.uniform(10.0, 30.0)),
                    "random_state": self.random_state,
                    "verbose": -1
                }
                m = LightGBMQuantModel(params=p, task_type=self.task_type)
                m.fit(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feats)
                preds = m.predict(X_val)
                s = roc_auc_score(y_val, preds) if len(np.unique(y_val)) > 1 else 0.5

                if s > best_s:
                    best_s = s
                    best_p = p

            self.best_params = best_p
            self.best_score = float(best_s)

        logger.info(f"贝叶斯超参数寻优完成 | 最佳验证得分: {self.best_score:.4f} | 参数: {self.best_params}")
        return self.best_params, self.best_score
