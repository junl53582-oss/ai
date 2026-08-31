"""
未来超额收益率标签生成器 (models/labeler.py)
严格基于市场 Canonical Trading Calendar 计算未来 N 个真实交易日的 Alpha 超额收益
杜绝因停牌缺行导致 shift(-5) 跨越数月造成的时间轴弹性失真！
采用完全向量化合并计算，毫秒级高效响应
Research Integrity Hardened:
- 支持严格 Canonical Calendar 认证门禁 (CALENDAR_CERTIFICATION Fail-Closed)
"""
from __future__ import annotations

import logging
from typing import Optional, List
import pandas as pd
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class TargetLabeler:
    """标签计算与样本清洗器 (基于真实交易日历时间轴，向量化运算)"""

    def __init__(
        self,
        horizon: int = settings.LABEL_HORIZON,
        label_col: Optional[str] = None,
        task_type: str = settings.TASK_TYPE,
        label_col_clf: Optional[str] = None,
        threshold: float = settings.LABEL_THRESHOLD,
        threshold_mode: str = settings.LABEL_THRESHOLD_MODE,
        require_canonical_calendar: bool = False
    ):
        self.horizon = int(horizon)
        self.label_col = label_col or f"label_excess_{horizon}d"
        self.task_type = task_type
        self.label_col_clf = label_col_clf or f"label_up_down_{horizon}d"
        self.threshold = float(threshold)
        self.threshold_mode = threshold_mode
        self.require_canonical_calendar = bool(require_canonical_calendar)

    def compute_excess_return_label(
        self,
        df: pd.DataFrame,
        canonical_dates: Optional[List[pd.Timestamp]] = None,
        require_canonical_calendar: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        基于全市场统一交易日历计算未来 N 个交易日超额收益率：
        Label = (Stock_Price(t+N_trd) / Stock_Price(t) - 1) - (Benchmark_Price(t+N_trd) / Benchmark_Price(t) - 1)
        """
        must_require_cal = self.require_canonical_calendar if require_canonical_calendar is None else bool(require_canonical_calendar)
        if must_require_cal and (canonical_dates is None or len(canonical_dates) == 0):
            raise RuntimeError(
                "FATAL: Canonical exchange trading calendar is required in certified mode! "
                "Deducing trading calendar from filtered stock rows is disallowed."
            )

        if "benchmark_close" not in df.columns:
            raise ValueError(
                "Dataframe missing required 'benchmark_close' column for excess return labeling! "
                "股票超额收益计算必须依赖基准收盘价，禁止将未对冲基准的绝对收益伪装成 Alpha 标签 (Fail-Closed)."
            )

        logger.info(f"正在基于市场真实交易日历计算未来 {self.horizon} 交易日超额收益率标签 ({self.label_col})...")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(by=["date", "symbol"], inplace=True)

        must_require_cal = self.require_canonical_calendar if require_canonical_calendar is None else bool(require_canonical_calendar)
        if must_require_cal and (canonical_dates is None or len(canonical_dates) == 0):
            raise RuntimeError("FATAL: require_canonical_calendar=True but canonical_dates was not provided!")

        if canonical_dates is not None and len(canonical_dates) > 0:
            s_ts = df["date"].min()
            e_ts = df["date"].max()
            cal_filtered = [pd.to_datetime(d) for d in canonical_dates if s_ts <= pd.to_datetime(d) <= e_ts]
            if not cal_filtered:
                raise RuntimeError("FATAL: Canonical calendar has zero overlap with research dataset!")
            market_dates = sorted(list(set(cal_filtered)))
        else:
            market_dates = sorted(df["date"].unique())

        if len(market_dates) <= self.horizon:
            df[self.label_col] = np.nan
            if self.task_type == "classification" and self.label_col_clf not in df.columns:
                df[self.label_col_clf] = np.nan
            logger.warning(
                f"可用交易日 {len(market_dates)} 不足持有期 {self.horizon}，"
                f"标签全部置 NaN (已显式创建 {self.label_col} 与分类列)"
            )
            return df

        date_map = pd.DataFrame({
            "date": market_dates[:-self.horizon],
            "future_date": market_dates[self.horizon:]
        })

        price_col = "adj_close" if "adj_close" in df.columns else "close"
        sub_stock = df[["symbol", "date", price_col]].rename(columns={"date": "future_date", price_col: "future_price"})

        merged = pd.merge(df, date_map, on="date", how="left")
        merged = pd.merge(merged, sub_stock, on=["symbol", "future_date"], how="left")

        stock_ret = np.where(
            merged["future_price"].notna() & (merged[price_col] > 0),
            (merged["future_price"] / merged[price_col]) - 1.0,
            np.nan
        )

        bench_daily = df.groupby("date")["benchmark_close"].first().reset_index()
        bench_future = bench_daily.rename(columns={"date": "future_date", "benchmark_close": "future_bench_close"})
        merged = pd.merge(merged, bench_future, on="future_date", how="left")
        bench_ret = np.where(
            merged["future_bench_close"].notna() & (merged["benchmark_close"] > 0),
            (merged["future_bench_close"] / merged["benchmark_close"]) - 1.0,
            np.nan
        )

        merged[self.label_col] = stock_ret - bench_ret

        # 二分类标签计算
        if self.threshold_mode == "cross_sectional_median":
            daily_med = merged.groupby("date")[self.label_col].transform("median")
            merged[self.label_col_clf] = np.where(
                merged[self.label_col].notna() & daily_med.notna(),
                (merged[self.label_col] > daily_med).astype(float),
                np.nan
            )
        elif self.threshold_mode == "cross_sectional_extreme":
            daily_rank_pct = merged.groupby("date")[self.label_col].rank(pct=True)
            extreme_q = float(getattr(settings, "LABEL_EXTREME_QUANTILE", 0.30))
            is_top = daily_rank_pct > (1.0 - extreme_q)
            is_bottom = daily_rank_pct < extreme_q
            merged[self.label_col_clf] = np.nan
            valid_mask = merged[self.label_col].notna()
            merged.loc[valid_mask & is_top, self.label_col_clf] = 1.0
            merged.loc[valid_mask & is_bottom, self.label_col_clf] = 0.0
        else:
            merged[self.label_col_clf] = np.where(
                merged[self.label_col].notna(),
                (merged[self.label_col] > self.threshold).astype(float),
                np.nan
            )

        drop_cols = ["future_date", "future_price", "future_bench_close"]
        merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True)
        return merged
