"""
Walk-Forward 严格时序走步滚动训练器 (models/walk_forward.py)
结合 Purged Gap 机制彻底杜绝未来标签泄漏 (No Lookahead / No Leakage)
严格限制训练与验证仅在 in_universe == True 样本上执行，并审计 warmup_rows_excluded
Research Integrity Hardened:
- 严格 Production Model 物理隔离，研究模式默认 save_model=False，禁止直接写入 settings.MODELS_DIR
- 标签丢失 Fail-Closed，禁止静默自动回退
- Purge 窗口不足 Fail-Closed，禁止无 Purge 训练
- 消除 Python assert，使用显式 RuntimeError 门禁
- 记录每 Fold 详细 Purge 时序审计指标
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

from config.settings import settings
from .lightgbm_model import LightGBMQuantModel
from factors.processor import FactorProcessor

logger = logging.getLogger(__name__)


class WalkForwardTrainer:
    """走步滚动时序训练与预测执行器 (支持 Purged Gap 隔离、in_universe 样本约束与生产模型物理隔离)"""

    def __init__(
        self,
        train_years: float = settings.TRAIN_WINDOW_YEARS,
        val_months: int = settings.VAL_WINDOW_MONTHS,
        test_months: int = settings.TEST_WINDOW_MONTHS,
        purge_gap_days: int = settings.PURGE_GAP_DAYS,
        label_col: Optional[str] = None,
        task_type: str = settings.TASK_TYPE,
        model_type: str = "lightgbm",
        feature_selection_method: str = "all",
        top_k_features: int = 20,
        weighting_mode: str = "recency_magnitude",
        random_state: int = 42,
        model_dir: Optional[Path] = None,
        model_params: Optional[Dict[str, Any]] = None,
        save_model: Optional[bool] = None,
        strict_mode: bool = True
    ):
        self.train_years = float(train_years)
        self.val_months = int(val_months)
        self.test_months = int(test_months)
        self.purge_gap_days = max(int(purge_gap_days), settings.LABEL_HORIZON)
        self.task_type = task_type
        self.model_type = model_type
        self.feature_selection_method = feature_selection_method
        self.top_k_features = top_k_features
        self.weighting_mode = weighting_mode
        self.random_state = int(random_state)
        self.save_model = bool(model_dir is not None) if save_model is None else bool(save_model)
        self.strict_mode = bool(strict_mode)

        # 生产模型物理隔离审计门禁 (Production Isolation Guard)
        prod_dir_resolved = Path(settings.MODELS_DIR).resolve()
        if model_dir is not None:
            resolved_model_dir = Path(model_dir).resolve()
            if self.strict_mode and resolved_model_dir == prod_dir_resolved and not getattr(self, "_allow_production_promotion", False):
                raise RuntimeError(
                    f"FATAL: Research runner attempted to configure model_dir directly to production settings.MODELS_DIR ({prod_dir_resolved})! "
                    "Research runners must use run-scoped isolated directories."
                )
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = None

        self.model_params = model_params.copy() if model_params is not None else None
        self._label_col_explicitly_set = bool(label_col is not None)
        self.label_col = label_col or (settings.LABEL_COLUMN_CLF if task_type == "classification" else settings.LABEL_COLUMN)
        self.models: List[Dict[str, Any]] = []
        self.fold_audit_records: List[Dict[str, Any]] = []
        self.warmup_rows_excluded: int = 0
        self.st_unknown_rows: int = 0
        self.st_training_excluded_rows: int = 0
        self.st_trading_excluded_rows: int = 0

    @staticmethod
    def _compute_sample_weights(
        train_df: pd.DataFrame,
        label_col: str,
        half_life_days: int = 120,
        magnitude_boost: float = 1.5,
        mode: str = "recency_magnitude"
    ) -> Optional[np.ndarray]:
        """
        样本加权引擎：
        - "none": 不做样本加权 (等权)
        - "recency": 仅做时间半衰期衰减
        - "recency_magnitude": 复合加权 (时间衰减 + 连续超额收益幅度提升)
        """
        if train_df is None or len(train_df) == 0 or mode == "none":
            return None
        max_date = train_df["date"].max()
        time_delta = (max_date - train_df["date"]).dt.days.values.astype(float)
        weights = np.power(0.5, time_delta / half_life_days)

        if mode == "recency_magnitude":
            mag_col = None
            for cand in (settings.LABEL_COLUMN, label_col):
                if cand in train_df.columns:
                    col_vals = train_df[cand].values.astype(float)
                    uniq = np.unique(col_vals[np.isfinite(col_vals)])
                    if len(uniq) <= 2 and set(np.nan_to_num(uniq)).issubset({0.0, 1.0}):
                        continue
                    mag_col = col_vals
                    break

            if mag_col is not None:
                valid_mask = np.isfinite(mag_col)
                if valid_mask.any():
                    std_val = np.nanstd(mag_col) or 0.05
                    mag_factor = 1.0 + np.clip(np.abs(mag_col) / (std_val + 1e-6), 0.0, magnitude_boost)
                    weights = weights * mag_factor

        weights = weights / (np.nanmean(weights) + 1e-8)
        return weights

    @staticmethod
    def _compute_recency_weights(train_df: pd.DataFrame, half_life_days: int = 120) -> Optional[np.ndarray]:
        """向后兼容旧接口"""
        if train_df is None or len(train_df) == 0:
            return None
        max_date = train_df["date"].max()
        time_delta = (max_date - train_df["date"]).dt.days.values.astype(float)
        return np.power(0.5, time_delta / half_life_days)

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Optional[Any]]:
        """
        执行严格的 Walk-Forward 滚动时序训练：
        1. 划分训练集、验证集与测试集时序区间
        2. 应用 Purged Gap：剔除训练集末尾与验证集末尾各 PURGE_GAP_DAYS 天
        3. 仅使用 in_universe == True 样本参与训练与验证
        4. 统计并审计 warmup_rows_excluded
        """
        logger.info(f"开始执行 Walk-Forward 走步训练 (Purged Gap = {self.purge_gap_days} 个交易日)...")
        raw_feature_cols = feature_cols or FactorProcessor.get_all_factor_cols()
        feature_cols = [c for c in raw_feature_cols if c in df.columns]

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(by=["date", "symbol"], inplace=True)

        # 标签检查与严格 Fail-Closed (Label Fail-Closed Gate)
        if self.label_col not in df.columns:
            available_labels = [c for c in df.columns if "label" in c or "target" in c or "ab_label" in c]
            if self._label_col_explicitly_set and self.strict_mode:
                raise KeyError(
                    f"FATAL: Requested label '{self.label_col}' not found in dataset! "
                    f"Available label candidates: {available_labels}"
                )
            else:
                clf_cands = [c for c in df.columns if c.startswith("label_up_down_") or c.startswith("label_direction_") or c.startswith("ab_label_")]
                reg_cands = [c for c in df.columns if c.startswith("label_excess_") or c.startswith("label_net_alpha_")]
                cands = clf_cands if self.task_type == "classification" else reg_cands
                if not cands:
                    cands = available_labels
                if cands:
                    logger.info(f"指定的标签列 '{self.label_col}' 不在数据集中，自适应选用 '{cands[0]}'")
                    self.label_col = cands[0]
                else:
                    raise KeyError(f"No valid label column found in dataset! Available: {list(df.columns)}")

        all_dates = pd.Series(df["date"].unique()).sort_values().reset_index(drop=True)
        total_days = len(all_dates)

        train_days = int(self.train_years * 242)
        val_days = int(self.val_months * 20)
        test_days = int(self.test_months * 20)
        min_required_days = train_days + val_days + test_days + self.purge_gap_days * 2

        if total_days < min_required_days:
            if self.strict_mode and total_days < (self.purge_gap_days * 2 + 30):
                raise ValueError(
                    f"FATAL: Insufficient total trading days ({total_days}) for walk-forward with purge gap {self.purge_gap_days}!"
                )
            logger.warning(f"数据总交易日数 ({total_days}) 较少，自适应调整走步窗口...")
            train_days = max(int(total_days * 0.5), 60)
            val_days = max(int(total_days * 0.15), 15)
            test_days = max(int(total_days * 0.1), 10)

        out_of_sample_preds = []
        current_test_start_idx = train_days + val_days
        fold = 1
        latest_model = None
        total_warmup_excluded = 0

        not_excluded_col = ~df.get("excluded_from_training", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        self.st_unknown_rows = int(df.get("is_st_unknown", pd.Series(False, index=df.index)).fillna(False).sum())
        self.st_training_excluded_rows = int(df.get("excluded_from_training", pd.Series(False, index=df.index)).fillna(False).sum())
        self.st_trading_excluded_rows = int(df.get("is_nontradable", pd.Series(False, index=df.index)).fillna(False).sum())

        self.fold_audit_records = []

        while current_test_start_idx < total_days:
            train_start_idx = max(0, current_test_start_idx - val_days - train_days)
            val_start_idx = current_test_start_idx - val_days
            test_end_idx = min(total_days, current_test_start_idx + test_days)

            raw_train_dates = all_dates.iloc[train_start_idx:val_start_idx]
            raw_val_dates = all_dates.iloc[val_start_idx:current_test_start_idx]
            test_dates = all_dates.iloc[current_test_start_idx:test_end_idx]

            # ---------------- 严格 Purged 隔离与 Fail-Closed 门禁 ----------------
            if len(raw_train_dates) <= self.purge_gap_days:
                if self.strict_mode:
                    raise RuntimeError(
                        f"FATAL: Fold {fold} raw train dates ({len(raw_train_dates)}) <= purge gap ({self.purge_gap_days})! "
                        "Cannot train without sufficient purge gap."
                    )
                purged_train_dates = raw_train_dates
                purge_train_passed = False
            else:
                purged_train_dates = raw_train_dates.iloc[:-self.purge_gap_days]
                purge_train_passed = True

            if len(raw_val_dates) <= self.purge_gap_days:
                purged_val_dates = pd.Series(dtype=all_dates.dtype)
                purge_val_passed = bool(len(raw_val_dates) == 0)
            else:
                purged_val_dates = raw_val_dates.iloc[:-self.purge_gap_days]
                purge_val_passed = True

            # 筛选数据：严格仅在 in_universe == True 且未被排除的样本上训练
            in_univ_col = df["in_universe"] if "in_universe" in df.columns else pd.Series(True, index=df.index)

            raw_train_subset = df[df["date"].isin(purged_train_dates) & in_univ_col & not_excluded_col]
            valid_label_mask = raw_train_subset[self.label_col].notna()
            total_warmup_excluded += int((~valid_label_mask).sum())

            train_df = raw_train_subset[valid_label_mask]
            val_df = df[df["date"].isin(purged_val_dates) & in_univ_col & not_excluded_col & df[self.label_col].notna()]
            test_df = df[df["date"].isin(test_dates)].copy()

            if len(train_df) < 10 or len(test_df) == 0:
                current_test_start_idx += test_days
                fold += 1
                continue

            # 显式科研安全 Invariant 门禁 (不用 assert，防止 python -O 消除)
            train_max_date = train_df["date"].max()
            val_min_date = val_df["date"].min() if not val_df.empty else None
            val_max_date = val_df["date"].max() if not val_df.empty else None
            test_min_date = test_df["date"].min()

            if val_min_date is not None and train_max_date >= val_min_date:
                raise RuntimeError(
                    f"FATAL: Fold {fold} temporal overlap detected between train max ({train_max_date}) and val min ({val_min_date})!"
                )
            if val_max_date is not None and val_max_date >= test_min_date:
                raise RuntimeError(
                    f"FATAL: Fold {fold} temporal overlap detected between val max ({val_max_date}) and test min ({test_min_date})!"
                )
            if val_min_date is None and train_max_date >= test_min_date:
                raise RuntimeError(
                    f"FATAL: Fold {fold} temporal overlap detected between train max ({train_max_date}) and test min ({test_min_date})!"
                )

            actual_train_test_gap = int((test_min_date - train_max_date).days)
            actual_val_test_gap = int((test_min_date - val_max_date).days) if val_max_date is not None else None

            self.fold_audit_records.append({
                "fold": fold,
                "raw_train_start": str(raw_train_dates.min().date()),
                "raw_train_end": str(raw_train_dates.max().date()),
                "purged_train_start": str(purged_train_dates.min().date()),
                "purged_train_end": str(purged_train_dates.max().date()),
                "raw_val_start": str(raw_val_dates.min().date()) if not raw_val_dates.empty else None,
                "raw_val_end": str(raw_val_dates.max().date()) if not raw_val_dates.empty else None,
                "purged_val_start": str(purged_val_dates.min().date()) if not purged_val_dates.empty else None,
                "purged_val_end": str(purged_val_dates.max().date()) if not purged_val_dates.empty else None,
                "test_start": str(test_dates.min().date()),
                "test_end": str(test_dates.max().date()),
                "actual_train_test_gap_days": actual_train_test_gap,
                "actual_val_test_gap_days": actual_val_test_gap,
                "purge_gate_passed": bool(purge_train_passed and purge_val_passed),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df)
            })

            logger.info(
                f"[Fold {fold}] Purged训练集: {purged_train_dates.min().strftime('%Y%m%d')}~{purged_train_dates.max().strftime('%Y%m%d')} "
                f"({len(train_df)}条) | 样本外测试: {test_dates.min().strftime('%Y%m%d')}~{test_dates.max().strftime('%Y%m%d')}"
            )

            # 严格基于 Train 窗口做特征选择 (Train-Only Feature Selection)
            if self.feature_selection_method != "all":
                from .fold_feature_selector import FoldFeatureSelector
                sel_method = "top_n" if "top" in self.feature_selection_method else "rank_ic_pruned"
                selector = FoldFeatureSelector(top_n=self.top_k_features)
                fold_feats, feat_df = selector.select_features(
                    train_df=train_df,
                    candidate_features=feature_cols,
                    label_col=self.label_col,
                    method=sel_method
                )
            else:
                fold_feats = feature_cols

            train_weights = self._compute_sample_weights(
                train_df,
                label_col=self.label_col,
                mode=self.weighting_mode
            )

            # 实例化预测模型 (严格隔离 model_dir)
            if self.model_type == "ensemble":
                from .ensemble_model import EnsembleQuantModel
                model = EnsembleQuantModel(task_type=self.task_type)
            elif self.model_type == "double_ensemble":
                from .double_ensemble import DoubleEnsembleQuantModel
                model = DoubleEnsembleQuantModel(task_type=self.task_type)
            elif self.model_type == "mlp":
                from .deep_tabular import TabularMLPQuantModel
                model = TabularMLPQuantModel(task_type=self.task_type)
            elif self.model_type in ("lightgbm_ranker", "ranking"):
                model = LightGBMQuantModel(
                    params=self.model_params, task_type="ranking", random_state=self.random_state,
                    model_dir=self.model_dir, strict_mode=self.strict_mode
                )
            elif self.model_type in ("lightgbm_reg", "regression"):
                model = LightGBMQuantModel(
                    params=self.model_params, task_type="regression", random_state=self.random_state,
                    model_dir=self.model_dir, strict_mode=self.strict_mode
                )
            else:
                model = LightGBMQuantModel(
                    params=self.model_params, task_type=self.task_type, random_state=self.random_state,
                    model_dir=self.model_dir, strict_mode=self.strict_mode
                )

            X_tr = train_df[fold_feats].copy()
            if "date" in train_df.columns:
                X_tr["date"] = train_df["date"]

            X_v = val_df[fold_feats].copy() if not val_df.empty else None
            if X_v is not None and "date" in val_df.columns:
                X_v["date"] = val_df["date"]

            if self.model_type in ("lightgbm_ranker", "ranking"):
                col_data = train_df[self.label_col].dropna()
                is_int_grade = pd.api.types.is_integer_dtype(train_df[self.label_col]) or (
                    len(col_data) > 0 and np.all(np.equal(np.mod(col_data, 1), 0)) and col_data.min() >= 0 and col_data.max() <= 31
                )
                if is_int_grade:
                    y_tr = train_df[self.label_col].fillna(0).astype(int)
                    y_v = val_df[self.label_col].fillna(0).astype(int) if not val_df.empty else None
                else:
                    y_tr = (train_df.groupby("date")[self.label_col].rank(pct=True) * 4.999).astype(int)
                    y_v = (val_df.groupby("date")[self.label_col].rank(pct=True) * 4.999).astype(int) if not val_df.empty else None
            else:
                y_tr = train_df[self.label_col]
                y_v = val_df[self.label_col] if not val_df.empty else None

            model.fit(
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_v,
                y_val=y_v,
                feature_names=fold_feats,
                sample_weight=train_weights
            )
            latest_model = model

            test_df["pred_score"] = model.predict(test_df[fold_feats])

            if "in_universe" in test_df.columns:
                in_univ_mask = test_df["in_universe"].fillna(False).astype(bool)
                test_df["pred_rank"] = np.nan
                test_df.loc[in_univ_mask, "pred_rank"] = (
                    test_df[in_univ_mask].groupby("date")["pred_score"].rank(ascending=False, pct=True)
                )
            else:
                test_df["pred_rank"] = test_df.groupby("date")["pred_score"].rank(ascending=False, pct=True)

            out_of_sample_preds.append(test_df)

            self.models.append({
                "fold": fold,
                "train_start": purged_train_dates.min(),
                "train_end": purged_train_dates.max(),
                "val_start": purged_val_dates.min() if not purged_val_dates.empty else None,
                "val_end": purged_val_dates.max() if not purged_val_dates.empty else None,
                "test_start": test_dates.min(),
                "test_end": test_dates.max(),
                "feature_count": len(fold_feats),
                "selected_features": fold_feats,
                "model": model
            })

            current_test_start_idx += test_days
            fold += 1

        self.warmup_rows_excluded = total_warmup_excluded

        if not out_of_sample_preds:
            raise ValueError("走步训练未能生成任何有效样本外预测！请检查样本量。")

        full_oos_df = pd.concat(out_of_sample_preds, ignore_index=True)
        full_oos_df.sort_values(by=["date", "symbol"], inplace=True)
        full_oos_df.reset_index(drop=True, inplace=True)

        # 仅在显式 save_model=True 且配置了合法 isolated model_dir 时才保存 (Production Isolation Guard)
        if self.save_model and latest_model is not None and self.model_dir is not None:
            latest_model.save()

        logger.info(f"Walk-Forward 滚动训练完成，共 {fold - 1} 折，样本外记录数: {len(full_oos_df)}")
        return full_oos_df, latest_model
