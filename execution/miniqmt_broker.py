"""
迅投 MiniQMT (xtquant) 实盘与模拟交易网关 (execution/miniqmt_broker.py)
用于直连国内主流券商 MiniQMT 终端，实现全自动资产查询、持仓同步与程序化批量下单。
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from .broker_base import BaseBroker, Account, Position, ExecutionOrder, OrderSide, OrderType, ExecutionStatus

logger = logging.getLogger(__name__)

try:
    from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
    from xtquant.xttype import StockAccount
    XTQUANT_AVAILABLE = True
except ImportError:
    XTQUANT_AVAILABLE = False


class MiniQMTBroker(BaseBroker):
    """迅投 MiniQMT 交易网关适配器"""

    def __init__(
        self,
        qmt_path: str = r"D:\国金证券QMT交易端\userdata_mini",
        account_id: str = "5500123456",
        account_type: str = "STOCK",
        session_id: int = 123456
    ):
        self.qmt_path = qmt_path
        self.account_id = account_id
        self.account_type = account_type
        self.session_id = session_id

        self.trader: Optional[Any] = None
        self.acc: Optional[Any] = None
        self.is_connected = False
        self.orders: List[ExecutionOrder] = []

    def connect(self) -> bool:
        if not XTQUANT_AVAILABLE:
            logger.warning("未检测到 xtquant 依赖包！如需使用 MiniQMT 实盘，请在 Python 环境中安装 xtquant。")
            return False

        try:
            self.trader = XtQuantTrader(self.qmt_path, self.session_id)
            self.acc = StockAccount(self.account_id, self.account_type)
            self.trader.start()
            connect_res = self.trader.connect()
            if connect_res == 0:
                self.trader.subscribe(self.acc)
                self.is_connected = True
                logger.info(f"成功连接 MiniQMT 券商终端 (账号: {self.account_id})")
                return True
            else:
                logger.error(f"连接 MiniQMT 终端失败，错误码: {connect_res}")
                return False
        except Exception as e:
            logger.error(f"初始化 MiniQMT 异常: {e}")
            return False

    def get_account(self) -> Account:
        if not self.is_connected or self.trader is None:
            return Account(account_id=self.account_id)

        try:
            asset = self.trader.query_stock_asset(self.acc)
            positions = self.get_positions()
            return Account(
                account_id=self.account_id,
                total_equity=asset.total_asset if asset else 0.0,
                cash=asset.cash if asset else 0.0,
                frozen_cash=asset.frozen_cash if asset else 0.0,
                market_value=asset.market_value if asset else 0.0,
                positions=positions
            )
        except Exception as e:
            logger.error(f"查询 MiniQMT 资产异常: {e}")
            return Account(account_id=self.account_id)

    def get_positions(self) -> Dict[str, Position]:
        positions: Dict[str, Position] = {}
        if not self.is_connected or self.trader is None:
            return positions

        try:
            q_pos_list = self.trader.query_stock_positions(self.acc)
            for p in q_pos_list:
                sym = p.stock_code
                positions[sym] = Position(
                    symbol=sym,
                    total_shares=p.volume,
                    available_shares=p.can_use_volume,
                    avg_cost_price=p.open_price,
                    current_price=p.last_price,
                    market_value=p.market_value,
                    floating_pnl=p.floating_pnl,
                    floating_pnl_pct=p.floating_pnl / (p.open_price * p.volume + 1e-6) * 100
                )
        except Exception as e:
            logger.error(f"查询 MiniQMT 持仓异常: {e}")
        return positions

    def send_order(
        self,
        symbol: str,
        side: OrderSide,
        shares: int,
        price: float,
        order_type: OrderType = OrderType.LIMIT
    ) -> ExecutionOrder:
        order_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.is_connected or self.trader is None:
            logger.warning(f"MiniQMT 未连接，仿真模拟委托: {side.value} {symbol} {shares}股 @ {price:.2f}元")
            order = ExecutionOrder(
                order_id=f"SIM_QMT_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                symbol=symbol, side=side, order_type=order_type,
                requested_shares=shares, requested_price=price,
                status=ExecutionStatus.SUBMITTED, created_time=order_time
            )
            self.orders.append(order)
            return order

        try:
            # 转换 QMT 订单参数
            xt_order_type = 23 if side == OrderSide.BUY else 24 # 23: 买入, 24: 卖出
            xt_price_type = 50 if order_type == OrderType.LIMIT else 51 # 50: 单一限价
            seq = self.trader.order_stock(self.acc, symbol, xt_order_type, shares, xt_price_type, price)
            
            order = ExecutionOrder(
                order_id=str(seq), symbol=symbol, side=side, order_type=order_type,
                requested_shares=shares, requested_price=price,
                status=ExecutionStatus.SUBMITTED if seq > 0 else ExecutionStatus.REJECTED,
                created_time=order_time,
                error_msg="" if seq > 0 else "QMT 挂单接口返回失败"
            )
            self.orders.append(order)
            return order
        except Exception as e:
            logger.error(f"MiniQMT 委托异常: {e}")
            order = ExecutionOrder(
                order_id="ERR", symbol=symbol, side=side, order_type=order_type,
                requested_shares=shares, requested_price=price,
                status=ExecutionStatus.REJECTED, created_time=order_time, error_msg=str(e)
            )
            self.orders.append(order)
            return order

    def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected or self.trader is None:
            return False
        try:
            cancel_res = self.trader.cancel_order_stock_sysid(self.acc, 0, order_id)
            return cancel_res == 0
        except Exception as e:
            logger.error(f"MiniQMT 撤单异常: {e}")
            return False

    def cancel_all_pending_orders(self) -> int:
        """全量撤销 QMT 终端挂单"""
        if not self.is_connected or self.trader is None:
            return 0
        try:
            # 查询未结订单并逐一撤单
            cancel_count = 0
            for ord in self.orders:
                if ord.status in (ExecutionStatus.SUBMITTED, ExecutionStatus.PENDING):
                    if self.cancel_order(ord.order_id):
                        ord.status = ExecutionStatus.CANCELLED
                        cancel_count += 1
            logger.info(f"🛡️ [前置撤单防线] MiniQMT 已撤销 {cancel_count} 笔挂起订单")
            return cancel_count
        except Exception as e:
            logger.error(f"MiniQMT 批量撤单异常: {e}")
            return 0

    def get_orders(self, status: Optional[ExecutionStatus] = None) -> List[ExecutionOrder]:
        return self.orders
