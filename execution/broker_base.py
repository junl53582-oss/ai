"""
实盘与模拟券商交易网关基类 (execution/broker_base.py)
定义券商账户、持仓、订单数据结构以及统一的交易执行抽象接口。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"   # 限价单
    MARKET = "MARKET" # 市价单


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Position:
    """账户单只证券持仓"""
    symbol: str
    total_shares: int = 0
    available_shares: int = 0
    avg_cost_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    floating_pnl: float = 0.0
    floating_pnl_pct: float = 0.0


@dataclass
class Account:
    """券商账户资产概况"""
    account_id: str
    total_equity: float = 0.0
    cash: float = 0.0
    frozen_cash: float = 0.0
    market_value: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)


@dataclass
class ExecutionOrder:
    """实盘/模拟委托订单"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    requested_shares: int
    requested_price: float
    filled_shares: int = 0
    avg_filled_price: float = 0.0
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_time: str = ""
    error_msg: str = ""


class BaseBroker(ABC):
    """统一券商交易网关基类"""

    @abstractmethod
    def connect(self) -> bool:
        """连接券商终端或初始化沙盒"""
        pass

    @abstractmethod
    def get_account(self) -> Account:
        """获取最新账户资产概况"""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """获取当前全部持仓"""
        pass

    @abstractmethod
    def send_order(
        self,
        symbol: str,
        side: OrderSide,
        shares: int,
        price: float,
        order_type: OrderType = OrderType.LIMIT
    ) -> ExecutionOrder:
        """提交委托挂单"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤销指定委托"""
        pass

    @abstractmethod
    def cancel_all_pending_orders(self) -> int:
        """撤销所有挂起未成交的委托单"""
        pass

    @abstractmethod
    def get_orders(self, status: Optional[ExecutionStatus] = None) -> List[ExecutionOrder]:
        """查询当日委托订单列表"""
        pass
