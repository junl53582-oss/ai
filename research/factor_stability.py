"""
因子时序稳定性、牛熊市场状态与多年度稳健性分析引擎 (research/factor_stability.py)
Phase 1.4: 年度有效天数门禁与无有效年份 Fail-Closed (sign_consistency_ratio = None)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from .config import ResearchConfig, default_research_config
from .factor_metrics import FactorMetricsEngine

logger = logging.getLogger(__name__)


@dataclass
class FactorStabilityResult:
    """因子时序稳定性评估结果"""
    factor_name: str
    overall_mean_rank_ic: float = 0.0
    sign_consistency_ratio: Optional[float] = None
    annual_stability_status: str = "INSUFFICIENT_DATA"
    bull_rank_ic: float = 0.0
    bear_rank_ic: float = 0.0
    sideways_rank_ic: float = 0.0
    annual_rank_ic: Dict[str, float] = field(default_factory=dict)
    annual_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rolling_mean_ic_dict: Dict[int, float] = field(default_factory=dict)


class FactorStabilityEngine:
    """因子时间序列与多市场状态稳定性评估器"""

    @classmethod
    def evaluate_stability(
        cls,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str,
        config: Optional[ResearchConfig] = None
    ) -> FactorStabilityResult:
        """评估因子的年度一致性与牛熊震荡市表现"""
        cfg = config or default_research_config
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # 1. 逐日 RankIC 序列
        daily_rank_ic = FactorMetricsEngine.compute_daily_ic(df, factor_col, return_col, method="spearman")
        if daily_rank_ic.empty:
            return FactorStabilityResult(factor_name=factor_col)

        overall_mean = float(daily_rank_ic.mean())
        target_sign = 1 if overall_mean >= 0 else -1

        # 2. 分年度 RankIC 计算与有效天数门禁 (Phase 1.4 Fail-Closed)
        daily_df = daily_rank_ic.to_frame(name="rank_ic")
        daily_df["year"] = daily_df.index.year

        annual_ic = {}
        annual_details = {}
        valid_sign_years = 0
        total_valid_years = 0

        for yr, grp in daily_df.groupby("year"):
            yr_str = str(yr)
            n_days = len(grp)
            m_ic = float(grp["rank_ic"].mean())
            std_ic = float(grp["rank_ic"].std(ddof=1)) if n_days > 1 else 1e-6
            icir = (m_ic / std_ic) if std_ic > 1e-8 else 0.0

            if n_days < cfg.MIN_VALID_DAYS_PER_YEAR:
                status_str = "INSUFFICIENT_YEAR_SAMPLE"
            else:
                status_str = "VALID"
                total_valid_years += 1
                if (m_ic * target_sign) > 0:
                    valid_sign_years += 1

            annual_ic[yr_str] = round(m_ic, 4)
            annual_details[yr_str] = {
                "year": yr,
                "valid_ic_days": n_days,
                "mean_rank_ic": round(m_ic, 4),
                "icir": round(icir, 4),
                "status": status_str
            }

        # 符号一致性: 若无任何达标有效年份，严格 Fail-Closed 为 None
        if total_valid_years > 0:
            sign_consistency = round(float(valid_sign_years / total_valid_years), 4)
            stab_status = "VALID"
        else:
            sign_consistency = None
            stab_status = "INSUFFICIENT_DATA"

        # 3. 牛熊与震荡市场状态划分
        market_bench_col = cfg.BENCHMARK_CLOSE_COL if cfg.BENCHMARK_CLOSE_COL in df.columns else None
        if market_bench_col:
            bench_daily = df.groupby("date")[market_bench_col].first().sort_index()
            bench_20d_ret = bench_daily.pct_change(cfg.MARKET_REGIME_WINDOW)
            
            regime_map = {}
            for dt, ret in bench_20d_ret.items():
                if np.isnan(ret):
                    regime_map[dt] = "SIDEWAYS"
                elif ret >= cfg.MARKET_REGIME_THRESHOLD:
                    regime_map[dt] = "BULL"
                elif ret <= -cfg.MARKET_REGIME_THRESHOLD:
                    regime_map[dt] = "BEAR"
                else:
                    regime_map[dt] = "SIDEWAYS"
                    
            daily_df["regime"] = daily_df.index.map(regime_map).fillna("SIDEWAYS")
            
            bull_ic = float(daily_df[daily_df["regime"] == "BULL"]["rank_ic"].mean()) if (daily_df["regime"] == "BULL").any() else 0.0
            bear_ic = float(daily_df[daily_df["regime"] == "BEAR"]["rank_ic"].mean()) if (daily_df["regime"] == "BEAR").any() else 0.0
            side_ic = float(daily_df[daily_df["regime"] == "SIDEWAYS"]["rank_ic"].mean()) if (daily_df["regime"] == "SIDEWAYS").any() else 0.0
        else:
            bull_ic = bear_ic = side_ic = overall_mean

        # 4. 滚动均值 IC
        rolling_dict = {}
        for w in cfg.ROLLING_WINDOWS:
            roll_mean = daily_rank_ic.rolling(w, min_periods=min(w, len(daily_rank_ic))).mean().dropna()
            rolling_dict[w] = round(float(roll_mean.mean()), 4) if not roll_mean.empty else 0.0

        return FactorStabilityResult(
            factor_name=factor_col,
            overall_mean_rank_ic=round(overall_mean, 4),
            sign_consistency_ratio=sign_consistency,
            annual_stability_status=stab_status,
            bull_rank_ic=round(bull_ic, 4),
            bear_rank_ic=round(bear_ic, 4),
            sideways_rank_ic=round(side_ic, 4),
            annual_rank_ic=annual_ic,
            annual_details=annual_details,
            rolling_mean_ic_dict=rolling_dict
        )
