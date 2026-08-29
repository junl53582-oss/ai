"""
动态多维风控与宏观市场状态识别引擎 (strategy/risk_manager.py)
核心机制:
1. 市场状态判别 (Market Regime Detection): 依据基准均线形态与波动率将行情划分为牛市、震荡、高波剧震与熊市防守
2. 波动率目标控制 (Volatility Targeting): 依据实时已实现波动率动态缩放仓位，锁定组合目标波动率
3. 动态回撤梯度熔断 (Drawdown Circuit Breaker): 依据净值水下深度阶梯式削减仓位，控制极端尾部下行风险
"""
import logging
from enum import Enum
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL_TREND = "牛市多头 (Bull Trend)"
    LOW_VOL_RANGE = "低波震荡 (Low-Vol Range)"
    HIGH_VOL_TURBULENT = "高波剧震 (High-Vol Turbulent)"
    BEAR_CRISIS = "熊市防守 (Bear Crisis)"


class MarketRegimeDetector:
    """宏观市场状态与湍流指数判别器"""

    @classmethod
    def detect_regime(
        cls,
        benchmark_prices: pd.Series,
        lookback_vol: int = 20,
        high_vol_annual_thresh: float = 0.25
    ) -> Dict[str, Any]:
        """
        判断当前市场所处宏观状态与推荐基准仓位上限:
        """
        if len(benchmark_prices) < 60:
            return {
                "regime": MarketRegime.LOW_VOL_RANGE.value,
                "recommended_gross_exposure": 0.80,
                "realized_vol_annual": 0.18,
                "ma20_over_ma60": 1.0,
                "reason": "样本长度不足，默认中性震荡仓位"
            }

        p = benchmark_prices.dropna()
        close = p.values
        ret = np.diff(close) / (close[:-1] + 1e-8)

        # 20日年化波动率
        vol_20 = float(np.std(ret[-lookback_vol:]) * np.sqrt(242.0))
        ma20 = float(np.mean(close[-20:]))
        ma60 = float(np.mean(close[-60:]))
        cur_p = float(close[-1])

        trend_ratio = ma20 / (ma60 + 1e-8)

        # 判别状态
        if cur_p > ma20 and ma20 > ma60 and vol_20 <= high_vol_annual_thresh:
            regime = MarketRegime.BULL_TREND
            exposure = 1.00
            reason = "均线多头排列且波动处于舒适区间，全仓多头运作"
        elif vol_20 > high_vol_annual_thresh:
            regime = MarketRegime.HIGH_VOL_TURBULENT
            exposure = 0.50
            reason = f"市场波动率剧增 ({vol_20*100:.1f}% > {high_vol_annual_thresh*100:.1f}%)，主动降半仓避险"
        elif cur_p < ma20 and ma20 < ma60:
            regime = MarketRegime.BEAR_CRISIS
            exposure = 0.25
            reason = "均线空头破位下行，执行极低仓位防守策略"
        else:
            regime = MarketRegime.LOW_VOL_RANGE
            exposure = 0.80
            reason = "震荡无序阶段，维持 80% 稳健标准仓位"

        return {
            "regime": regime.value,
            "recommended_gross_exposure": exposure,
            "realized_vol_annual": round(vol_20, 4),
            "ma20_over_ma60": round(trend_ratio, 4),
            "reason": reason
        }


class VolatilityTargetingEngine:
    """组合波动率目标自适应缩放器"""

    def __init__(self, target_vol_annual: float = 0.15, max_leverage: float = 1.0):
        self.target_vol_annual = target_vol_annual
        self.max_leverage = max_leverage

    def compute_vol_scaling_factor(self, portfolio_returns: pd.Series, lookback_days: int = 20) -> float:
        """
        计算波动率目标调整系数:
        scaling = min(max_leverage, target_vol / realized_vol)
        """
        if len(portfolio_returns) < lookback_days:
            return 1.0

        r = portfolio_returns.iloc[-lookback_days:].dropna().values
        if len(r) < 5:
            return 1.0

        realized_vol = float(np.std(r) * np.sqrt(242.0))
        if realized_vol <= 1e-4:
            return 1.0

        factor = self.target_vol_annual / realized_vol
        return float(np.clip(factor, 0.2, self.max_leverage))


class DynamicDrawdownController:
    """组合动态回撤阶梯熔断控制器"""

    @classmethod
    def evaluate_drawdown_exposure_limit(
        cls,
        cumulative_equity: pd.Series
    ) -> Tuple[float, str]:
        """
        根据当前回撤深度决定允许的最大多头仓位比例:
        - Drawdown < 6%: 100% 正常
        - 6% <= Drawdown < 10%: 限制最大仓位 75%
        - 10% <= Drawdown < 15%: 限制最大仓位 50%
        - Drawdown >= 15%: 触发紧急熔断，强制空仓 0%
        """
        if len(cumulative_equity) < 2:
            return 1.0, "初始运行阶段，无回撤触发"

        eq = cumulative_equity.values
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / (peak + 1e-8)
        current_dd = float(dd[-1])

        if current_dd < 0.06:
            return 1.0, f"回撤平稳 ({current_dd*100:.1f}%)，允许 100% 仓位"
        elif current_dd < 0.10:
            return 0.75, f"回撤初显 ({current_dd*100:.1f}%)，限制多头上限至 75%"
        elif current_dd < 0.15:
            return 0.50, f"进入深水回撤区 ({current_dd*100:.1f}%)，限制多头上限至 50%"
        else:
            return 0.0, f"⚠️ 触碰极端回撤预警线 ({current_dd*100:.1f}%)，强制清仓熔断避险！"
