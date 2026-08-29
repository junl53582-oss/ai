"""
因子时序衰减与最佳预测视界分析引擎 (research/factor_decay.py)
计算 1D, 3D, 5D, 10D, 20D 多周期 IC/RankIC，绘制衰减曲线并确定最优 Alpha 释放周期。
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from .factor_metrics import FactorMetricsEngine
from .config import ResearchConfig, default_research_config

logger = logging.getLogger(__name__)


@dataclass
class FactorDecayResult:
    """单个因子的预测视界衰减分析结果"""
    factor_name: str
    ic_by_horizon: Dict[str, float] = field(default_factory=dict)       # "1D" -> 0.05, "5D" -> 0.08
    rank_ic_by_horizon: Dict[str, float] = field(default_factory=dict)  # "1D" -> 0.06, "5D" -> 0.09
    rank_icir_by_horizon: Dict[str, float] = field(default_factory=dict)
    best_horizon: str = "20D"
    best_rank_ic: float = 0.0
    half_life_days: Optional[float] = None
    decay_curve: List[Dict[str, Any]] = field(default_factory=list)


class FactorDecayEngine:
    """因子时序衰减分析器"""

    @classmethod
    def analyze_decay(
        cls,
        df: pd.DataFrame,
        factor_col: str,
        horizons: Optional[List[int]] = None,
        return_col_prefix: str = "future_excess_return_"
    ) -> FactorDecayResult:
        """
        跨 1D, 3D, 5D, 10D, 20D 评估因子 IC 衰减
        """
        horizons = horizons or [1, 3, 5, 10, 20]
        ic_dict = {}
        rank_ic_dict = {}
        rank_icir_dict = {}
        decay_curve = []

        best_h_str = f"{horizons[0]}D"
        best_abs_rank_ic = -1.0
        best_signed_rank_ic = 0.0

        for h in horizons:
            h_str = f"{h}D"
            # 优先使用超额收益列，兜底使用绝对收益列
            ret_col = f"{return_col_prefix}{h}d"
            if ret_col not in df.columns:
                ret_col = f"future_return_{h}d"
            if ret_col not in df.columns:
                continue

            ic_series = FactorMetricsEngine.compute_daily_ic(df, factor_col, ret_col, method="pearson")
            rank_ic_series = FactorMetricsEngine.compute_daily_ic(df, factor_col, ret_col, method="spearman")

            m_ic = float(ic_series.mean()) if not ic_series.empty else 0.0
            m_rank_ic = float(rank_ic_series.mean()) if not rank_ic_series.empty else 0.0
            std_rank_ic = float(rank_ic_series.std(ddof=1)) if len(rank_ic_series) > 1 else 0.0
            icir = (m_rank_ic / std_rank_ic) if std_rank_ic > 1e-8 else 0.0

            ic_dict[h_str] = round(m_ic, 4)
            rank_ic_dict[h_str] = round(m_rank_ic, 4)
            rank_icir_dict[h_str] = round(icir, 4)

            decay_curve.append({
                "horizon": h_str,
                "horizon_days": h,
                "mean_ic": round(m_ic, 4),
                "mean_rank_ic": round(m_rank_ic, 4),
                "rank_icir": round(icir, 4)
            })

            if abs(m_rank_ic) > best_abs_rank_ic:
                best_abs_rank_ic = abs(m_rank_ic)
                best_signed_rank_ic = m_rank_ic
                best_h_str = h_str

        # 估算半衰期 (指数衰减拟合 |RankIC(t)| = RankIC(0) * e^(-lambda * t))
        half_life = None
        if len(decay_curve) >= 3 and best_abs_rank_ic > 0.01:
            try:
                days = np.array([x["horizon_days"] for x in decay_curve])
                abs_ics = np.array([max(abs(x["mean_rank_ic"]), 1e-5) for x in decay_curve])
                if abs_ics[0] > abs_ics[-1]:
                    # 线性回归 log(IC) = a - lambda * t
                    slope, _ = np.polyfit(days, np.log(abs_ics), 1)
                    if slope < 0:
                        decay_lambda = -slope
                        half_life = round(float(np.log(2.0) / decay_lambda), 1)
            except Exception:
                half_life = None

        return FactorDecayResult(
            factor_name=factor_col,
            ic_by_horizon=ic_dict,
            rank_ic_by_horizon=rank_ic_dict,
            rank_icir_by_horizon=rank_icir_dict,
            best_horizon=best_h_str,
            best_rank_ic=round(best_signed_rank_ic, 4),
            half_life_days=half_life,
            decay_curve=decay_curve
        )
