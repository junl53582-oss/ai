"""
Walk-Forward 严格时序走步滚动训练器 (models/walk_forward.py)
结合 Purged Gap 机制彻底杜绝未来标签泄漏 (No Lookahead / No Leakage)
严格限制训练与验证仅在 in_universe == True 样本上执行，并审计 warmup_rows_excluded
"""
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
    """走步滚动时序训练与预测执行器 (支持 Purged Gap 隔离与 in_universe 样本约束)"""

    def __init__(
        self,
        train_years: int = settings.TRAIN_WINDOW_YEARS,
        val_months: int = settings.VAL_WINDOW_MONTHS,
        test_months: int = settings.TEST_WINDOW_MONTHS,
        purge_gap_days: int = settings.PURGE_GAP_DAYS,
        label_col: Optional[str] = None,
        task_type: str = settings.TASK_TYPE,
        model_type: str = "lightgbm"
    ):
        self.train_years = train_years
        self.val_months = val_months
        self.test_months = test_months
        self.purge_gap_days = max(purge_gap_days, settings.LABEL_HORIZON)
        self.task_type = task_type
        self.model_type = model_type
        # 默认使用当前任务类型对应的标签列
        self.label_col = label_col or (settings.LABEL_COLUMN_CLF if task_type == "classification" else settings.LABEL_COLUMN)
        self.models: List[Dict[str, Any]] = []
        self.warmup_rows_excluded: int = 0

    @staticmethod
    def _compute_sample_weights(
        train_df: pd.DataFrame,
        label_col: str,
        half_life_days: int = 120,
        magnitude_boost: float = 1.5
    ) -> np.ndarray:
        """
        复合样本加权引擎：
        1. 时间衰减权重 (Recency Weighting): 距离训练集末尾越近权重越高 (半衰期 120 交易日)
        2. 收益率幅度加权 (Magnitude Boost): 捕捉大幅超额收益与暴跌信号，强化强 Alpha 的惩罚梯度

        重要修正: 幅度加权必须基于【连续超额收益】列，绝不能用二分类标签！
        分类标签是 0/1，用它算 |lbl|/std 会退化成"正样本统一放大、负样本不变"，
        等价在 scale_pos_weight 之上再人为注入严重的类别失衡，使模型系统性偏多、
        买入信号失真、回测路径恶化 (实测 Alpha +20.47% → +10.36%)。
        """
        if train_df is None or len(train_df) == 0:
            return None
        max_date = train_df["date"].max()
        time_delta = (max_date - train_df["date"]).dt.days.values.astype(float)
        weights = np.power(0.5, time_delta / half_life_days)

        # 优先使用连续超额收益列衡量"幅度"；仅在确实不存在时才回退到传入标签
        mag_col = None
        for cand in (settings.LABEL_COLUMN, label_col):
            if cand in train_df.columns:
                col_vals = train_df[cand].values.astype(float)
                # 二分类标签(仅 0/1)不适合衡量幅度, 跳过
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
    def _compute_recency_weights(train_df: pd.DataFrame, half_life_days: int = 120) -> np.ndarray:
        """向后兼容旧单测的接口"""
        if train_df is None or len(train_df) == 0:
            return None
        max_date = train_df["date"].max()
        time_delta = (max_date - train_df["date"]).dt.days.values.astype(float)
        return np.power(0.5, time_delta / half_life_days)

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, LightGBMQuantModel]:
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

        all_dates = pd.Series(df["date"].unique()).sort_values().reset_index(drop=True)
        total_days = len(all_dates)

        # 转换为大致的交易日天数
        train_days = int(self.train_years * 242)
        val_days = int(self.val_months * 20)
        test_days = int(self.test_months * 20)
        min_required_days = train_days + val_days + test_days + self.purge_gap_days * 2

        if total_days < min_required_days:
            logger.warning(f"数据总交易日数 ({total_days}) 较少，自动自适应调整走步窗口...")
            train_days = max(int(total_days * 0.5), 60)
            val_days = max(int(total_days * 0.15), 15)
            test_days = max(int(total_days * 0.1), 10)

        out_of_sample_preds = []
        current_test_start_idx = train_days + val_days
        fold = 1
        latest_model = None
        total_warmup_excluded = 0

        # 严格考虑 excluded_from_training 与 ST 状态隔离 (P0-7)
        not_excluded_col = ~df.get("excluded_from_training", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        self.st_unknown_rows = int(df.get("is_st_unknown", pd.Series(False, index=df.index)).fillna(False).sum())
        self.st_training_excluded_rows = int(df.get("excluded_from_training", pd.Series(False, index=df.index)).fillna(False).sum())
        self.st_trading_excluded_rows = int(df.get("is_nontradable", pd.Series(False, index=df.index)).fillna(False).sum())

        while current_test_start_idx < total_days:
            train_start_idx = max(0, current_test_start_idx - val_days - train_days)
            val_start_idx = current_test_start_idx - val_days
            test_end_idx = min(total_days, current_test_start_idx + test_days)

            raw_train_dates = all_dates.iloc[train_start_idx:val_start_idx]
            raw_val_dates = all_dates.iloc[val_start_idx:current_test_start_idx]
            test_dates = all_dates.iloc[current_test_start_idx:test_end_idx]

            # ---------------- 严格 Purged 隔离 ----------------
            if len(raw_train_dates) > self.purge_gap_days:
                purged_train_dates = raw_train_dates.iloc[:-self.purge_gap_days]
            else:
                purged_train_dates = raw_train_dates

            if len(raw_val_dates) > self.purge_gap_days:
                purged_val_dates = raw_val_dates.iloc[:-self.purge_gap_days]
            else:
                purged_val_dates = raw_val_dates

            # 筛选数据：严格仅在 in_universe == True 且未被排除的样本上训练 (P0-2, P0-7)
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

            # 自动检测断言：无时序重叠
            train_max_date = train_df["date"].max()
            val_min_date = val_df["date"].min() if not val_df.empty else None
            val_max_date = val_df["date"].max() if not val_df.empty else None
            test_min_date = test_df["date"].min()

            if val_min_date is not None:
                assert train_max_date < val_min_date, f"Fold {fold}: 训练集与验证集存在时序重叠！"
            if val_max_date is not None:
                assert val_max_date < test_min_date, f"Fold {fold}: 验证集与测试集存在时序重叠！"

            logger.info(
                f"[Fold {fold}] Purged训练集: {purged_train_dates.min().strftime('%Y%m%d')}~{purged_train_dates.max().strftime('%Y%m%d')} "
                f"({len(train_df)}条) | 样本外测试: {test_dates.min().strftime('%Y%m%d')}~{test_dates.max().strftime('%Y%m%d')}"
            )

            # 训练模型
            # 样本复合加权: 距离验证集越近 + 收益率波动越显著的样本权重越高
            train_weights = self._compute_sample_weights(train_df, label_col=self.label_col)

            if self.model_type == "ensemble":
                from .ensemble_model import EnsembleQuantModel
                model = EnsembleQuantModel(task_type=self.task_type)
            elif self.model_type == "double_ensemble":
                from .double_ensemble import DoubleEnsembleQuantModel
                model = DoubleEnsembleQuantModel(task_type=self.task_type)
            elif self.model_type == "mlp":
                from .deep_tabular import TabularMLPQuantModel
                model = TabularMLPQuantModel(task_type=self.task_type)
            else:
                model = LightGBMQuantModel(task_type=self.task_type)

            model.fit(
                X_train=train_df[feature_cols],
                y_train=train_df[self.label_col],
                X_val=val_df[feature_cols] if not val_df.empty else None,
                y_val=val_df[self.label_col] if not val_df.empty else None,
                feature_names=feature_cols,
                sample_weight=train_weights
            )
            latest_model = model

            # 样本外打分与截面排名 (P0-5 仅在 in_universe == True 内排名)
            # 分类模式: pred_score 为上涨概率；回归模式: 为连续超额收益
            test_df["pred_score"] = model.predict(test_df[feature_cols])
            
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
                "train_end": purged_train_dates.max(),
                "test_start": test_dates.min(),
                "test_end": test_dates.max(),
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
        
        if latest_model is not None:
            latest_model.save()

        logger.info(f"Walk-Forward 滚动训练完成，共 {fold - 1} 折，样本外记录数: {len(full_oos_df)}")
        return full_oos_df, latest_model
