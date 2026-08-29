"""
因子时间稳定性与市场状态分化研究引擎 (research/factor_stability.py)
按年度切片、滚动窗口 (60D/120D/252D)、符号一致性检验，以及牛市/熊市/震荡市分市场状态评价。
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from .factor_metrics import FactorMetricsEngine
from .config import ResearchConfig, default_research_config

logger = logging.getLogger(__name__)


@dataclass
class FactorStabilityResult:
    """因子的时间稳定性与市场状态分析结果"""
    factor_name: str
    overall_mean_rank_ic: float
    annual_rank_ic: Dict[str, float] = field(default_factory=dict)       # "2021" -> 0.05
    annual_rank_icir: Dict[str, float] = field(default_factory=dict)
    annual_long_short_return: Dict[str, float] = field(default_factory=dict)
    sign_consistency_ratio: float = 0.0                                 # 与全样本符号同向的年份占比
    
    # 滚动稳定性统计
    rolling_stats: Dict[str, Dict[str, float]] = field(default_factory=dict) # "60D" -> {mean, std, min, max, positive_ratio}
    
    # 市场状态划分结果
    bull_rank_ic: float = 0.0
    bear_rank_ic: float = 0.0
    sideways_rank_ic: float = 0.0
    regime_breakdown: Dict[str, Any] = field(default_factory=dict)


class FactorStabilityEngine:
    """因子时序稳定性分析器"""

    @classmethod
    def evaluate_stability(
        cls,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str,
        benchmark_col: Optional[str] = "benchmark_close",
        config: Optional[ResearchConfig] = None
    ) -> FactorStabilityResult:
        """
        全面评估因子的年度稳定性、滚动稳健性与牛熊分化表现
        """
        cfg = config or default_research_config
        daily_rank_ic = FactorMetricsEngine.compute_daily_ic(df, factor_col, return_col, method="spearman")
        if daily_rank_ic.empty:
            return FactorStabilityResult(factor_name=factor_col, overall_mean_rank_ic=0.0)

        overall_mean = float(daily_rank_ic.mean())
        overall_sign = np.sign(overall_mean) if abs(overall_mean) > 1e-8 else 1.0

        daily_rank_ic.index = pd.to_datetime(daily_rank_ic.index)
        daily_df = pd.DataFrame({"rank_ic": daily_rank_ic})
        daily_df["year"] = daily_df.index.year.astype(str)

        # 1. 按年度统计
        annual_rank_ic = {}
        annual_icir = {}
        annual_ls = {}
        sign_match_count = 0
        total_years = len(daily_df["year"].unique())

        for yr, grp in daily_df.groupby("year"):
            y_mean = float(grp["rank_ic"].mean())
            y_std = float(grp["rank_ic"].std(ddof=1)) if len(grp) > 1 else 0.0
            y_icir = (y_mean / y_std) if y_std > 1e-8 else 0.0

            annual_rank_ic[yr] = round(y_mean, 4)
            annual_icir[yr] = round(y_icir, 4)

            # 该年符号是否与全样本一致
            if np.sign(y_mean) == overall_sign:
                sign_match_count += 1

            # 计算该年度的多空收益
            sub_yr_df = df[pd.to_datetime(df["date"]).dt.year.astype(str) == yr]
            direction = 1 if overall_mean >= 0 else -1
            ls_res = FactorMetricsEngine.compute_turnover_and_long_short(
                sub_yr_df, factor_col, return_col, n_quantiles=5, direction=direction
            )
            annual_ls[yr] = round(ls_res["net_long_short_return"], 6)

        sign_consistency = round(sign_match_count / max(total_years, 1), 4)

        # 2. 滚动窗口稳定性 (60D, 120D, 252D)
        rolling_stats = {}
        for w in cfg.ROLLING_WINDOWS:
            if len(daily_rank_ic) >= w:
                roll_s = daily_rank_ic.rolling(w).mean().dropna()
                rolling_stats[f"{w}D"] = {
                    "mean": round(float(roll_s.mean()), 4),
                    "std": round(float(roll_s.std(ddof=1)), 4) if len(roll_s) > 1 else 0.0,
                    "min": round(float(roll_s.min()), 4),
                    "max": round(float(roll_s.max()), 4),
                    "positive_ratio": round(float((roll_s > 0).mean()), 4)
                }

        # 3. 市场状态划分 (牛市 / 熊市 / 震荡市)
        regime_res = cls._evaluate_regimes(df, daily_rank_ic, benchmark_col, cfg)

        return FactorStabilityResult(
            factor_name=factor_col,
            overall_mean_rank_ic=round(overall_mean, 4),
            annual_rank_ic=annual_rank_ic,
            annual_rank_icir=annual_icir,
            annual_long_short_return=annual_ls,
            sign_consistency_ratio=sign_consistency,
            rolling_stats=rolling_stats,
            bull_rank_ic=regime_res["bull_rank_ic"],
            bear_rank_ic=regime_res["bear_rank_ic"],
            sideways_rank_ic=regime_res["sideways_rank_ic"],
            regime_breakdown=regime_res
        )

    @classmethod
    def _evaluate_regimes(
        cls,
        df: pd.DataFrame,
        daily_rank_ic: pd.Series,
        benchmark_col: Optional[str],
        config: ResearchConfig
    ) -> Dict[str, Any]:
        """按基准指数趋势划分市场状态并评估 RankIC"""
        res = {"bull_rank_ic": 0.0, "bear_rank_ic": 0.0, "sideways_rank_ic": 0.0, "counts": {}}
        if not benchmark_col or benchmark_col not in df.columns:
            return res

        # 提取逐日基准收盘价
        bench_daily = df.groupby("date")[benchmark_col].first().dropna()
        bench_daily.index = pd.to_datetime(bench_daily.index)
        if len(bench_daily) < config.MARKET_REGIME_WINDOW:
            return res

        # 计算 20 日动量
        bench_ret_20d = bench_daily.pct_change(config.MARKET_REGIME_WINDOW)
        
        regimes = {}
        for dt, ret in bench_ret_20d.items():
            if np.isnan(ret):
                continue
            if ret >= config.MARKET_REGIME_THRESHOLD:
                regimes[dt] = "BULL"
            elif ret <= -config.MARKET_REGIME_THRESHOLD:
                regimes[dt] = "BEAR"
            else:
                regimes[dt] = "SIDEWAYS"

        regime_s = pd.Series(regimes)
        aligned = pd.DataFrame({"rank_ic": daily_rank_ic, "regime": regime_s}).dropna()

        if not aligned.empty:
            for reg in ["BULL", "BEAR", "SIDEWAYS"]:
                sub = aligned[aligned["regime"] == reg]
                res["counts"][reg] = len(sub)
                if not sub.empty:
                    val = round(float(sub["rank_ic"].mean()), 4)
                    if reg == "BULL":
                        res["bull_rank_ic"] = val
                    elif reg == "BEAR":
                        res["bear_rank_ic"] = val
                    else:
                        res["sideways_rank_ic"] = val

        return res
