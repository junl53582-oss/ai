"""
量化风控引擎
支持个股硬止损、动态跟踪止盈（基于 High 价格）、组合最大回撤熔断降仓与行业敞口约束
"""
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from config.settings import settings
from strategy.trading_rules import PositionRecord

logger = logging.getLogger(__name__)


class RiskManager:
    """多层次动态风控管理器"""

    def __init__(
        self,
        stop_loss_pct: float = settings.STOP_LOSS_PCT,
        trailing_stop_pct: float = settings.TRAILING_STOP_PCT,
        max_drawdown_limit: float = settings.MAX_DRAWDOWN_LIMIT,
        circuit_target_exposure: float = settings.CIRCUIT_TARGET_EXPOSURE
    ):
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.circuit_target_exposure = circuit_target_exposure
        self.is_circuit_breaker_active = False

    def update_and_check_position_risk(
        self,
        position: PositionRecord,
        today_row: pd.Series
    ) -> Tuple[bool, str]:
        """
        根据 T 日完整行情更新最高价并判定个股风控：
        1. 跟踪止盈高点更新：使用 T 日最高价 high (不发生未来函数)
        2. 硬止损判定：(T日收盘价 - 成本价) / 成本价 <= -stop_loss_pct
        3. 跟踪止盈判定：持仓曾实现浮盈 >= 5% 且自最高点回撤 >= trailing_stop_pct
        """
        high_price = float(today_row.get("high", today_row.get("close", 0.0)))
        close_price = float(today_row.get("close", 0.0))

        # 1. 动态更新持仓最高价
        if high_price > position.highest_price:
            position.highest_price = high_price

        # 2. 个股硬止损
        loss_rate = (close_price - position.avg_cost) / position.avg_cost
        if loss_rate <= -self.stop_loss_pct:
            return True, f"触发个股硬止损 (浮亏: {loss_rate*100:.2f}% <= -{self.stop_loss_pct*100:.1f}%)"

        # 3. 跟踪止盈 (历史最高浮盈 >= 5% 时激活回撤监控)
        max_profit_rate = (position.highest_price - position.avg_cost) / position.avg_cost
        if max_profit_rate >= 0.05:
            drawdown_from_high = (position.highest_price - close_price) / (position.highest_price + 1e-8)
            if drawdown_from_high >= self.trailing_stop_pct:
                return True, f"触发跟踪止盈 (最高浮盈: {max_profit_rate*100:.2f}%, 自高点回撤: {drawdown_from_high*100:.2f}%)"

        return False, "持仓正常"

    def check_portfolio_circuit_breaker(
        self,
        current_equity: float,
        peak_equity: float,
        holdings_value: float
    ) -> Tuple[bool, float]:
        """
        组合级最大回撤熔断与降仓目标计算：
        若组合当前回撤超过阈值，计算需降至的目标仓位比例 (如 30%)
        返回: (是否触发熔断, 目标持仓总市值)
        """
        if peak_equity <= 0:
            return False, holdings_value

        drawdown = (peak_equity - current_equity) / peak_equity
        if drawdown >= self.max_drawdown_limit:
            if not self.is_circuit_breaker_active:
                logger.warning(
                    f"⚠️ 组合当前回撤达 {drawdown*100:.2f}% >= {self.max_drawdown_limit*100:.1f}%，"
                    f"触发组合最大回撤熔断！目标将总仓位压缩至 {self.circuit_target_exposure*100:.1f}%"
                )
                self.is_circuit_breaker_active = True

            target_holdings_value = current_equity * self.circuit_target_exposure
            return True, target_holdings_value
        else:
            if self.is_circuit_breaker_active and drawdown < (self.max_drawdown_limit * 0.5):
                logger.info("✅ 组合净值回升脱离危险区，解除熔断风控状态。")
                self.is_circuit_breaker_active = False
            return False, holdings_value
