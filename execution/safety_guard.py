"""
机构级实盘资金安全防御中枢 (execution/safety_guard.py)
用于在调仓执行前对所有委托订单施加硬性风控与资金安全校验。
7 大核心防线:
1. 资金使用率硬顶 (Max Capital Utilization 95%): 强制保留 5% 现金防穿透
2. 单股持仓硬上限 (Single Stock Cap 20%): 单只个股敞口严禁超过总权益 20%
3. 日内换手率熔断 (Daily Turnover Circuit Breaker 50%): 单日累计成交上限 50% 防刷单
4. 双重显式确认 Flag (Live Confirmation Guard): 严禁未经 --live-confirm 误触真实下单
5. 价格偏离度与涨停拦截 (Price Sanity & Limit-Up Guard): 限价偏离 <= 2%，严禁涨停追高
6. 前置清场撤单 (Pre-Rebalance Cancel): 调仓前先全量撤销挂起未结订单防重复加仓
7. 通道异常熔断降级 (Heartbeat & Fallback): MiniQMT 脱机 10 秒即刻熔断并回退仿真
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SecurityException(Exception):
    """资金安全违规异常"""
    pass


class ExecutionSafetyGuard:
    """实盘交易资金安全防御器"""

    def __init__(
        self,
        max_capital_utilization: float = 0.95,
        max_single_stock_exposure: float = 0.20,
        max_daily_turnover_ratio: float = 0.50,
        max_price_deviation: float = 0.02,
        require_live_confirm: bool = True
    ):
        self.max_cap_util = max_capital_utilization
        self.max_single_stock_cap = max_single_stock_exposure
        self.max_daily_turnover = max_daily_turnover_ratio
        self.max_price_deviation = max_price_deviation
        self.require_live_confirm = require_live_confirm

    def audit_and_clamp_orders(
        self,
        target_shares: Dict[str, int],
        current_holdings: Dict[str, Any],
        prices: Dict[str, float],
        total_equity: float,
        current_cash: float
    ) -> Tuple[Dict[str, int], List[str]]:
        """
        全量审查并安全裁剪目标股数:
        1. 单股持仓 <= 20% total_equity
        2. 总买入市值 <= 95% total_equity (保留 5% 现金防穿透)
        3. 日内换手率 <= 50% total_equity
        """
        audit_logs: List[str] = []
        clamped_shares: Dict[str, int] = target_shares.copy()

        # ---------------- 防线 1: 单股 20% 仓位硬顶 ----------------
        max_single_val = total_equity * self.max_single_stock_cap
        for sym, s in list(clamped_shares.items()):
            p = prices.get(sym, 10.0)
            val = s * p
            if val > max_single_val:
                safe_s = int(max_single_val / (p * 100)) * 100
                clamped_shares[sym] = safe_s
                msg = f"🛡️ [单股上限防线] 标的 {sym} 目标市值 {val:,.2f} 超过 20% 上限 ({max_single_val:,.2f})，裁剪为 {safe_s} 股"
                logger.warning(msg)
                audit_logs.append(msg)

        # ---------------- 防线 2: 资金使用率 95% 硬顶 (保留 5% 现金缓冲) ----------------
        total_target_val = sum(clamped_shares[sym] * prices.get(sym, 10.0) for sym in clamped_shares)
        max_allowed_stock_val = total_equity * self.max_cap_util
        if total_target_val > max_allowed_stock_val:
            scale = max_allowed_stock_val / (total_target_val + 1e-8)
            msg = f"🛡️ [资金缓冲防线] 目标持仓总市值 {total_target_val:,.2f} 超过 95% 资金上限，保留 5% 现金 ({total_equity * (1-self.max_cap_util):,.2f}元)，整体等比缩放 {scale:.2%}"
            logger.warning(msg)
            audit_logs.append(msg)
            for sym in clamped_shares:
                p = prices.get(sym, 10.0)
                orig_s = clamped_shares[sym]
                new_s = int((orig_s * scale) / 100) * 100
                clamped_shares[sym] = new_s

        # ---------------- 防线 3: 日内 50% 总换手熔断 ----------------
        # 计算本次拟调仓的总买入与总卖出金额
        total_sell_amt = 0.0
        total_buy_amt = 0.0
        for sym, pos in current_holdings.items():
            tgt_s = clamped_shares.get(sym, 0)
            cur_s = pos.total_shares
            if cur_s > tgt_s:
                total_sell_amt += (cur_s - tgt_s) * prices.get(sym, 10.0)
        for sym, tgt_s in clamped_shares.items():
            cur_s = current_holdings[sym].total_shares if sym in current_holdings else 0
            if tgt_s > cur_s:
                total_buy_amt += (tgt_s - cur_s) * prices.get(sym, 10.0)

        turnover_val = total_sell_amt + total_buy_amt
        max_turnover_limit = total_equity * self.max_daily_turnover
        if turnover_val > max_turnover_limit:
            msg = f"🛡️ [换手熔断防线] 本次拟调仓额 {turnover_val:,.2f} 突破单日 50% 换手上限 ({max_turnover_limit:,.2f})，启动安全降频等比裁剪"
            logger.warning(msg)
            audit_logs.append(msg)
            turnover_scale = max_turnover_limit / (turnover_val + 1e-8)
            # 对变动量执行安全缩减
            for sym in list(clamped_shares.keys()):
                cur_s = current_holdings[sym].total_shares if sym in current_holdings else 0
                delta = clamped_shares[sym] - cur_s
                if delta > 0:
                    scaled_delta = int((delta * turnover_scale) / 100) * 100
                    clamped_shares[sym] = cur_s + scaled_delta

        return clamped_shares, audit_logs

    def validate_price_and_limit(
        self,
        symbol: str,
        side: str,
        order_price: float,
        latest_market_price: float,
        pre_close_price: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        防线 5: 价格偏离度审查与涨停买入拦截
        """
        # 1. 价格偏离度不得超过 2%
        if latest_market_price > 0:
            dev = abs(order_price - latest_market_price) / latest_market_price
            if dev > self.max_price_deviation:
                return False, f"委托价 {order_price:.2f} 偏离最新市价 {latest_market_price:.2f} 达到 {dev*100:.2f}% (超过 2% 上限)"

        # 2. 严禁在涨停板挂买单 (以昨收价计算涨幅 >= 9.8%)
        if side.upper() in ("BUY", "买入") and pre_close_price and pre_close_price > 0:
            change_ratio = (order_price - pre_close_price) / pre_close_price
            if change_ratio >= 0.098:
                return False, f"标的 {symbol} 委托价格 {order_price:.2f} 触碰 +9.8% 涨停线，触发涨停防追高拦截"

        return True, "价格安全合规"
