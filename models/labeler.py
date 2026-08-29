"""
未来超额收益率标签生成器 (models/labeler.py)
严格基于市场 Canonical Trading Calendar 计算未来 N 个真实交易日的 Alpha 超额收益
杜绝因停牌缺行导致 shift(-5) 跨越数月造成的时间轴弹性失真！
采用完全向量化合并计算，毫秒级高效响应
"""
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
        label_col: str = settings.LABEL_COLUMN,
        task_type: str = settings.TASK_TYPE,
        label_col_clf: str = settings.LABEL_COLUMN_CLF,
        threshold: float = settings.LABEL_THRESHOLD,
        threshold_mode: str = settings.LABEL_THRESHOLD_MODE
    ):
        self.horizon = horizon
        self.label_col = label_col  # 连续超额收益列
        self.task_type = task_type
        self.label_col_clf = label_col_clf  # 二分类标签列
        self.threshold = threshold
        self.threshold_mode = threshold_mode  # "fixed" | "cross_sectional_median"

    def compute_excess_return_label(
        self,
        df: pd.DataFrame,
        canonical_dates: Optional[List[pd.Timestamp]] = None
    ) -> pd.DataFrame:
        """
        基于全市场统一交易日历计算未来 N 个交易日超额收益率 (P1-2 支持注入 canonical_dates)：
        Label = (Stock_Price(t+N_trd) / Stock_Price(t) - 1) - (Benchmark_Price(t+N_trd) / Benchmark_Price(t) - 1)
        """
        logger.info(f"正在基于市场真实交易日历计算未来 {self.horizon} 交易日超额收益率标签 ({self.label_col})...")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(by=["date", "symbol"], inplace=True)

        if canonical_dates is not None and len(canonical_dates) > 0:
            s_ts = df["date"].min()
            e_ts = df["date"].max()
            cal_filtered = [pd.to_datetime(d) for d in canonical_dates if s_ts <= pd.to_datetime(d) <= e_ts]
            market_dates = sorted(list(set(cal_filtered))) if cal_filtered else sorted(df["date"].unique())
        else:
            market_dates = sorted(df["date"].unique())

        if len(market_dates) <= self.horizon:
            # 数据天数不足以覆盖持有期: 连续标签置 NaN。
            # 同时必须显式创建分类标签列(全 NaN)，否则下游 walk_forward 会因缺列直接 KeyError 崩溃。
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

        # 股票未来收益
        stock_ret = np.where(
            merged["future_price"].notna() & (merged[price_col] > 0),
            (merged["future_price"] / merged[price_col]) - 1.0,
            np.nan
        )

        # 基准未来收益 (P0-11 基准缺失置 NaN 绝不赋 0.0)
        if "benchmark_close" in df.columns:
            bench_daily = df.groupby("date")["benchmark_close"].first().reset_index()
            bench_future = bench_daily.rename(columns={"date": "future_date", "benchmark_close": "future_bench_close"})
            merged = pd.merge(merged, bench_future, on="future_date", how="left")
            bench_ret = np.where(
                merged["future_bench_close"].notna() & (merged["benchmark_close"] > 0),
                (merged["future_bench_close"] / merged["benchmark_close"]) - 1.0,
                np.nan
            )
        else:
            bench_ret = 0.0

        merged[self.label_col] = stock_ret - bench_ret

        # 分类模式: 生成二分类标签 (1=涨/跑赢基准, 0=跌/跑输基准)
        if self.task_type == "classification":
            if self.threshold_mode == "cross_sectional_median":
                # 市场中性: 每日截面内，超额收益 > 当日中位数 判定为 1 (相对强弱标签)
                # 优点: 正负样本天然各占约 50%，消除大盘整体涨跌导致的标签偏斜
                merged["_daily_median"] = merged.groupby("date")[self.label_col].transform("median")
                merged[self.label_col_clf] = (merged[self.label_col] > merged["_daily_median"]).astype(int)
                merged.loc[merged[self.label_col].isna() | merged["_daily_median"].isna(), self.label_col_clf] = np.nan
                merged.drop(columns=["_daily_median"], inplace=True)
                logger.info(
                    f"已生成市场中性二分类标签 {self.label_col_clf} (横截面中位数阈值): "
                    f"上涨占比={merged[self.label_col_clf].mean()*100:.1f}%"
                )
            elif self.threshold_mode == "cross_sectional_extreme":
                # 极端分组: 每日截面按超额收益排名，top q 分位标 1、bottom q 分位标 0，
                # 中间 1-2q 标 NaN (训练与评估均剔除)。拉大正负样本信号差距，
                # 用于诊断模型对「强跑赢 vs 强跑输」的区分能力 (也即 Top-K 策略的实际可用空间)。
                q = float(getattr(settings, "LABEL_EXTREME_QUANTILE", 0.30))
                rank_pct = merged.groupby("date")[self.label_col].rank(pct=True)
                merged[self.label_col_clf] = np.where(
                    rank_pct > 1.0 - q, 1.0,
                    np.where(rank_pct < q, 0.0, np.nan)
                )
                # 超额收益缺失时标签保持 NaN
                merged.loc[merged[self.label_col].isna(), self.label_col_clf] = np.nan
                logger.info(
                    f"已生成极端分组二分类标签 {self.label_col_clf} (top/bottom {q:.0%} 分位): "
                    f"上涨占比={merged[self.label_col_clf].mean()*100:.1f}%, "
                    f"保留样本占比={merged[self.label_col_clf].notna().mean()*100:.1f}%"
                )
            else:
                merged[self.label_col_clf] = (merged[self.label_col] > self.threshold).astype(int)
                logger.info(
                    f"已生成二分类标签 {self.label_col_clf}: "
                    f"上涨占比={merged[self.label_col_clf].mean()*100:.1f}%, "
                    f"阈值={self.threshold}"
                )

        merged.drop(columns=["future_date", "future_price", "future_bench_close"], errors="ignore", inplace=True)
        merged.sort_values(by=["date", "symbol"], inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged
