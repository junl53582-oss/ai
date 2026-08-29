"""
A股实盘交易硬约束、历史费率与持仓批次管理模块 (strategy/trading_rules.py)
包含 T+1 制度、ST 5% 涨跌停、历史分段印花税与过户费计算、
FIFO 批次持仓管理 (支持 accumulated_cash_dividend)、
订单数量守恒 (requested == cumulative_filled + remaining + cancelled) 与状态重算
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List, Union
import pandas as pd
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    """订单生命周期状态"""
    PENDING = "PENDING"                   # 待执行 (T日产生，等待T+1开盘)
    FILLED = "FILLED"                     # 全部成交
    PARTIALLY_FILLED = "PARTIALLY_FILLED" # 部分成交 (受流动性成交量占比约束，余量继续挂单)
    REJECTED = "REJECTED"                 # 拒绝 (如停牌、一字涨停无法买入)
    DEFERRED = "DEFERRED"                 # 延期顺延 (如T+1限制、跌停卖不掉，顺延至下一交易日)
    CANCELLED = "CANCELLED"               # 撤单 (回测结束或显式撤销)


class OrderSide(str, Enum):
    """交易方向"""
    BUY = "BUY"
    SELL = "SELL"


class RejectReason(str, Enum):
    """标准化订单拒绝/延期原因"""
    T_PLUS_1_LOCK = "T+1_LOCK"                 # T+1 可用股数锁定
    SUSPENDED = "SUSPENDED"                     # 股票停牌
    LIMIT_UP = "LIMIT_UP"                       # 一字涨停无法买入
    LIMIT_DOWN = "LIMIT_DOWN"                   # 一字跌停无法卖出
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"     # 现金不足
    MIN_LOT = "MIN_LOT"                         # 不足 100 股最小交易单位
    NO_MARKET_DATA = "NO_MARKET_DATA"           # 当日无行情数据
    LIQUIDITY_LIMIT = "LIQUIDITY_LIMIT"         # 超过单日流动性成交量容量上限
    END_OF_BACKTEST = "END_OF_BACKTEST"         # 回测结束强制撤销挂单


@dataclass
class Order:
    """标准交易订单结构 (支持严格订单数量守恒与父子订单关联)"""
    order_id: str
    symbol: str
    side: OrderSide
    signal_date: str                     # 信号产生日期 (T 日收盘)
    parent_order_id: Optional[str] = None# 父订单 ID (用于追踪部分成交余量顺延)
    execution_date: Optional[str] = None # 实际撮合成交日期 (T+1 日开盘)
    requested_shares: int = 0            # 初始请求总股数
    cumulative_filled_shares: int = 0    # 累计已成交总股数
    remaining_shares: int = 0            # 剩余待成交股数
    cancelled_shares: int = 0            # 撤单股数
    filled_shares: int = 0               # 当次撮合成交股数
    signal_price: float = 0.0            # 信号基准价 (T 日收盘价)
    execution_price: float = 0.0         # 实际成交价格 (已受涨跌停价格截断保护与滑点)
    commission: float = 0.0              # 显式券商佣金
    stamp_tax: float = 0.0               # 显式印花税 (仅卖出)
    transfer_fee: float = 0.0            # 显式过户费 (双边)
    slippage: float = 0.0                # 预估滑点成本金额 (仅作为统计展示，不重复扣减现金)
    status: OrderStatus = OrderStatus.PENDING
    reason: str = "调仓"
    reject_reason: Optional[str] = None

    def __post_init__(self):
        if self.remaining_shares == 0 and self.cumulative_filled_shares == 0 and self.cancelled_shares == 0:
            self.remaining_shares = self.requested_shares

    def verify_quantity_conservation(self) -> bool:
        """验证订单数量严格守恒定律: requested == cumulative_filled + remaining + cancelled"""
        return self.requested_shares == (self.cumulative_filled_shares + self.remaining_shares + self.cancelled_shares)


@dataclass
class PositionLot:
    """单个买入批次记录 (精确包含 buy_commission 与持仓期间累积分红)"""
    shares: int
    buy_execution_price: float
    buy_date: str
    buy_commission: float = 0.0
    accumulated_cash_dividend: float = 0.0 # 该批次在持有期间获得的累积现金分红总额 (元)


@dataclass
class PositionRecord:
    """单只股票持仓记录 (支持批次 Lots)"""
    symbol: str
    shares: int                           # 总持仓股数
    available_shares: int                 # T+1 可用卖出股数
    avg_cost: float                       # 持仓加权均价
    last_price: float                     # 最新估值价格
    buy_date: str                         # 最早 lot 的买入日期
    highest_price: float                  # 持仓期间最高价 (用于移动止盈，除息除权同比例调整)
    last_price_date: Optional[str] = None # 最近一次有效行情更新日期
    lots: List[PositionLot] = field(default_factory=list)


def recalculate_position_from_lots(pos: PositionRecord) -> None:
    """
    统一持仓状态重建函数：
    在发生 BUY、PARTIAL SELL 或 CORPORATE ACTIONS 后，
    严格根据当前剩余 PositionLots 重新计算持仓总股数与加权平均成本，杜绝状态漂移！
    """
    if not pos.lots:
        pos.shares = 0
        pos.available_shares = 0
        pos.avg_cost = 0.0
        return

    total_shares = sum(l.shares for l in pos.lots)
    total_cost = sum(l.shares * l.buy_execution_price for l in pos.lots)

    pos.shares = total_shares
    pos.avg_cost = total_cost / total_shares if total_shares > 0 else 0.0
    pos.buy_date = pos.lots[0].buy_date
    pos.available_shares = min(pos.available_shares, pos.shares)


class TradingFeeSchedule:
    """A股历史交易税费与规费计算器 (支持历史费率切换)"""

    @staticmethod
    def get_stamp_duty_rate(trade_date: Optional[Union[str, pd.Timestamp]] = None) -> float:
        """
        获取历史真实印花税率（仅卖出收取）：
        - 2023-08-28 之前：0.1% (单边 1‰)
        - 2023-08-28 之后：0.05% (减半后单边 0.5‰)
        """
        if trade_date is None:
            return settings.STAMP_DUTY
        dt = pd.to_datetime(trade_date)
        if dt < pd.Timestamp("2023-08-28"):
            return 0.001
        return 0.0005

    @staticmethod
    def get_transfer_fee_rate(trade_date: Optional[Union[str, pd.Timestamp]] = None) -> float:
        """
        获取历史真实过户费率（双向收取）：
        - 2022-04-29 之前：0.002% (十万分之二)
        - 2022-04-29 之后：0.001% (十万分之一)
        """
        if trade_date is None:
            return 0.00001
        dt = pd.to_datetime(trade_date)
        if dt < pd.Timestamp("2022-04-29"):
            return 0.00002
        return 0.00001


@dataclass
class FillRecord:
    """单次撮合成交明细记录 (P1-1 Fill / Execution Record 拆分)"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    execution_date: str
    filled_shares: int
    execution_price: float
    commission: float = 0.0
    stamp_tax: float = 0.0
    transfer_fee: float = 0.0
    slippage: float = 0.0
    reason: str = "调仓"


@dataclass
class PriceLimitSpec:
    """A股价格涨跌幅限制规范详情 (P0-6 Single Source of Truth)"""
    upper_ratio: float
    lower_ratio: float
    has_limit: bool
    rule_id: str
    rule_source: str
    effective_start: Optional[str] = None
    effective_end: Optional[str] = None


class PriceLimitRuleEngine:
    """
    A股全历史板块与时点价格涨跌幅规则引擎 (P0-6 Board × Date × Status)
    作为全系统唯一合规价格限制规则源，覆盖：
    1. 主板 (Main Board): 10% (ST: 5%)
       - 2023-04-10 全面注册制前：新股上市首日涨幅上限 44%，跌幅下限 36% (次日其后 10%)
       - 2023-04-10 全面注册制后：新股上市前 5 个交易日不设涨跌幅限制
    2. 创业板 (ChiNext 300xxx / 301xxx):
       - < 2020-08-24: 10% (ST: 5%)
       - >= 2020-08-24 (注册制改革): 20% (ST 股票同为 20%)，新股上市前 5 个交易日无限制
    3. 科创板 (STAR Market 688xxx / 689xxx): 20% (ST: 20%)，新股上市前 5 个交易日无限制
    4. 北交所 (BSE 83xxxx / 87xxxx / 88xxxx / 43xxxx / 920xxx): 30% (ST: 30%)，新股上市首日无限制
    """

    @classmethod
    def get_price_limit_spec(
        cls,
        symbol: str,
        trade_date: Optional[Union[str, pd.Timestamp]] = None,
        is_st: bool = False,
        listing_days: Optional[int] = None,
        special_status: Optional[str] = None
    ) -> PriceLimitSpec:
        sym = symbol.strip().upper()
        # 提取纯数字代码
        code = sym.split(".")[0] if "." in sym else sym
        dt = pd.to_datetime(trade_date) if trade_date else None

        # 1. 科创板 688 / 689
        if code.startswith("688") or code.startswith("689"):
            if listing_days is not None and listing_days <= 5:
                return PriceLimitSpec(
                    upper_ratio=999.0, lower_ratio=999.0, has_limit=False,
                    rule_id="STAR_MARKET_IPO_5D", rule_source="SSE_STAR_MARKET_RULES"
                )
            # 科创板 ST 股票涨跌幅限制同样为 20%
            rule_id = "STAR_MARKET_ST_20PCT" if is_st else "STAR_MARKET_20PCT"
            return PriceLimitSpec(
                upper_ratio=0.20, lower_ratio=0.20, has_limit=True,
                rule_id=rule_id, rule_source="SSE_STAR_MARKET_RULES"
            )

        # 2. 创业板 300 / 301
        if code.startswith("300") or code.startswith("301"):
            chinext_reform_date = pd.Timestamp("2020-08-24")
            if dt and dt < chinext_reform_date:
                # 注册制前创业板规则同主板：普通10%，ST 5%
                if is_st or "ST" in sym:
                    return PriceLimitSpec(
                        upper_ratio=0.05, lower_ratio=0.05, has_limit=True,
                        rule_id="CHINEXT_PRE_REFORM_ST_5PCT", rule_source="SZSE_CHINEXT_PRE_20200824",
                        effective_end="2020-08-23"
                    )
                if listing_days is not None and listing_days <= 1:
                    return PriceLimitSpec(
                        upper_ratio=0.44, lower_ratio=0.36, has_limit=True,
                        rule_id="CHINEXT_PRE_REFORM_IPO_1D", rule_source="SZSE_CHINEXT_PRE_20200824"
                    )
                return PriceLimitSpec(
                    upper_ratio=0.10, lower_ratio=0.10, has_limit=True,
                    rule_id="CHINEXT_PRE_REFORM_10PCT", rule_source="SZSE_CHINEXT_PRE_20200824",
                    effective_end="2020-08-23"
                )
            else:
                # 2020-08-24 注册制改革后：前5日无限制，其后20%（包括ST也是20%）
                if listing_days is not None and listing_days <= 5:
                    return PriceLimitSpec(
                        upper_ratio=999.0, lower_ratio=999.0, has_limit=False,
                        rule_id="CHINEXT_REFORM_IPO_5D", rule_source="SZSE_CHINEXT_REGISTRATION",
                        effective_start="2020-08-24"
                    )
                rule_id = "CHINEXT_REFORM_ST_20PCT" if is_st else "CHINEXT_REFORM_20PCT"
                return PriceLimitSpec(
                    upper_ratio=0.20, lower_ratio=0.20, has_limit=True,
                    rule_id=rule_id, rule_source="SZSE_CHINEXT_REGISTRATION",
                    effective_start="2020-08-24"
                )

        # 3. 北交所 8 / 4 / 920
        if code.startswith("8") or code.startswith("4") or code.startswith("920"):
            if listing_days is not None and listing_days <= 1:
                return PriceLimitSpec(
                    upper_ratio=999.0, lower_ratio=999.0, has_limit=False,
                    rule_id="BSE_IPO_1D", rule_source="BSE_TRADING_RULES"
                )
            rule_id = "BSE_ST_30PCT" if is_st else "BSE_30PCT"
            return PriceLimitSpec(
                upper_ratio=0.30, lower_ratio=0.30, has_limit=True,
                rule_id=rule_id, rule_source="BSE_TRADING_RULES"
            )

        # 4. 主板 (沪市 600/601/603/605, 深市 000/001/002/003)
        if is_st or "ST" in sym:
            return PriceLimitSpec(
                upper_ratio=0.05, lower_ratio=0.05, has_limit=True,
                rule_id="MAIN_BOARD_ST_5PCT", rule_source="EXCHANGE_MAIN_BOARD_RULES"
            )

        main_reg_date = pd.Timestamp("2023-04-10")
        if listing_days is not None:
            if dt and dt >= main_reg_date:
                if listing_days <= 5:
                    return PriceLimitSpec(
                        upper_ratio=999.0, lower_ratio=999.0, has_limit=False,
                        rule_id="MAIN_BOARD_REGISTRATION_IPO_5D", rule_source="EXCHANGE_MAIN_BOARD_RULES",
                        effective_start="2023-04-10"
                    )
            else:
                if listing_days <= 1:
                    return PriceLimitSpec(
                        upper_ratio=0.44, lower_ratio=0.36, has_limit=True,
                        rule_id="MAIN_BOARD_APPROVAL_IPO_1D", rule_source="EXCHANGE_MAIN_BOARD_RULES",
                        effective_end="2023-04-09"
                    )

        return PriceLimitSpec(
            upper_ratio=0.10, lower_ratio=0.10, has_limit=True,
            rule_id="MAIN_BOARD_10PCT", rule_source="EXCHANGE_MAIN_BOARD_RULES"
        )

    @classmethod
    def get_price_limit_ratio(
        cls,
        symbol: str,
        trade_date: Optional[Union[str, pd.Timestamp]] = None,
        is_st: bool = False,
        listing_days: Optional[int] = None,
        special_status: Optional[str] = None
    ) -> float:
        spec = cls.get_price_limit_spec(
            symbol=symbol,
            trade_date=trade_date,
            is_st=is_st,
            listing_days=listing_days,
            special_status=special_status
        )
        return spec.upper_ratio


class AShareTradingRules:
    """A股交易规则校验与硬约束执行"""

    def __init__(self, lot_size: int = settings.LOT_SIZE):
        self.lot_size = lot_size
        self.fee_schedule = TradingFeeSchedule()
        self.limit_engine = PriceLimitRuleEngine()

    @staticmethod
    def get_limit_ratio(symbol: str, date_str: Optional[str] = None, is_st: bool = False) -> float:
        """获取标的在特定历史时期的涨跌停幅度"""
        return PriceLimitRuleEngine.get_price_limit_ratio(symbol, trade_date=date_str, is_st=is_st)

    @staticmethod
    def calculate_limit_prices(pre_close: float, limit_ratio: float) -> Tuple[float, float]:
        """按 A 股 2 位小数精确四舍五入规则计算涨跌停价格"""
        limit_up = round(pre_close * (1.0 + limit_ratio) + 1e-5, 2)
        limit_down = round(pre_close * (1.0 - limit_ratio) + 1e-5, 2)
        return limit_up, limit_down

    def can_buy(self, row: pd.Series, execution_price: float) -> Tuple[bool, str]:
        """在 T+1 执行日开盘检查买入合法性"""
        if row.get("is_suspended", False) or row.get("volume", 0) <= 0:
            return False, RejectReason.SUSPENDED.value

        is_locked = row.get("is_limit_up_locked", False)
        if is_locked:
            return False, RejectReason.LIMIT_UP.value

        limit_up_p = row.get("limit_up_price", 999999.0)
        if execution_price >= limit_up_p - 0.01 and row.get("open", 0) == row.get("high", 0) == row.get("low", 0):
            return False, RejectReason.LIMIT_UP.value

        return True, "允许买入"

    def can_sell(self, row: pd.Series, position: PositionRecord, execution_price: float) -> Tuple[bool, str]:
        """在 T+1 执行日开盘检查卖出合法性"""
        if position.available_shares <= 0:
            return False, RejectReason.T_PLUS_1_LOCK.value

        if row.get("is_suspended", False) or row.get("volume", 0) <= 0:
            return False, RejectReason.SUSPENDED.value

        is_locked = row.get("is_limit_down_locked", False)
        if is_locked:
            return False, RejectReason.LIMIT_DOWN.value

        limit_down_p = row.get("limit_down_price", 0.0)
        if execution_price <= limit_down_p + 0.01 and row.get("open", 0) == row.get("high", 0) == row.get("low", 0):
            return False, RejectReason.LIMIT_DOWN.value

        return True, "允许卖出"

    def calculate_lot_shares(self, target_amount: float, price: float) -> int:
        """计算满足 100 股整手要求的买入股数 (向下取整)"""
        if price <= 0 or target_amount <= 0:
            return 0
        raw_shares = int(target_amount / price)
        lot_shares = (raw_shares // self.lot_size) * self.lot_size
        return lot_shares

    def compute_transaction_cost(
        self,
        amount: float,
        is_buy: bool,
        trade_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> Tuple[float, float, float, float]:
        """
        计算历史真实交易显式费用与统计滑点：
        返回: (佣金, 印花税, 过户费, 预估滑点成本展示值)
        """
        if amount <= 0:
            return 0.0, 0.0, 0.0, 0.0

        commission = max(amount * settings.COMMISSION_RATE, settings.MIN_COMMISSION)
        stamp_rate = self.fee_schedule.get_stamp_duty_rate(trade_date)
        stamp_tax = (amount * stamp_rate) if not is_buy else 0.0
        transfer_rate = self.fee_schedule.get_transfer_fee_rate(trade_date)
        transfer_fee = amount * transfer_rate
        slippage_estimate = amount * settings.SLIPPAGE_RATE

        return commission, stamp_tax, transfer_fee, slippage_estimate
