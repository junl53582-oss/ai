import uuid
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from config.settings import settings
from .broker_base import BaseBroker, Account, Position, ExecutionOrder, OrderSide, OrderType, ExecutionStatus

logger = logging.getLogger(__name__)


class PaperBroker(BaseBroker):
    """本地仿真交易沙盒网关 (支持跨日/跨调仓真实持久化)"""

    def __init__(self, initial_cash: float = 1_000_000.0, commission_rate: float = 0.00025, stamp_duty: float = 0.0005, persist: bool = False):
        self.account_id = "PAPER_ACCOUNT_01"
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.positions: Dict[str, Position] = {}
        self.orders: List[ExecutionOrder] = []
        self.is_connected = False
        self.persist = persist
        self._last_buy_dates: Dict[str, str] = {}

    def _get_state_file(self) -> Path:
        return settings.DATA_DIR / "paper_broker_state.json"

    def _save_state(self):
        if not self.persist:
            return
        try:
            state = {
                "account_id": self.account_id,
                "cash": self.cash,
                "last_buy_dates": self._last_buy_dates,
                "positions": {
                    sym: {
                        "symbol": pos.symbol,
                        "total_shares": pos.total_shares,
                        "available_shares": pos.available_shares,
                        "avg_cost_price": pos.avg_cost_price,
                        "current_price": pos.current_price,
                        "market_value": pos.market_value
                    } for sym, pos in self.positions.items()
                },
                "updated_at": datetime.now().isoformat()
            }
            p = self._get_state_file()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存 PaperBroker 状态异常: {e}")

    def _load_state(self) -> bool:
        if not self.persist:
            return False
        p = self._get_state_file()
        if not p.exists():
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.cash = float(state.get("cash", self.initial_cash))
            self._last_buy_dates = state.get("last_buy_dates", {})
            self.positions = {}
            for sym, pos_d in state.get("positions", {}).items():
                self.positions[sym] = Position(
                    symbol=pos_d["symbol"],
                    total_shares=int(pos_d["total_shares"]),
                    available_shares=int(pos_d["available_shares"]),
                    avg_cost_price=float(pos_d["avg_cost_price"]),
                    current_price=float(pos_d.get("current_price", pos_d["avg_cost_price"])),
                    market_value=float(pos_d.get("market_value", 0.0))
                )
            logger.info(f"已成功恢复 PaperBroker 历史持久化状态: 持仓 {len(self.positions)} 支标的 | 资金 {self.cash:,.2f} 元")
            return True
        except Exception as e:
            logger.warning(f"加载 PaperBroker 历史状态失败: {e}")
            return False

    def connect(self) -> bool:
        self.is_connected = True
        has_loaded = self._load_state()
        if not has_loaded:
            logger.info(f"PaperBroker 本地仿真账户 [{self.account_id}] 初始化就绪，初始可用资金: {self.cash:,.2f} 元")
        return True

    def get_account(self) -> Account:
        market_val = sum(pos.market_value for pos in self.positions.values())
        return Account(
            account_id=self.account_id,
            total_equity=self.cash + market_val,
            cash=self.cash,
            frozen_cash=0.0,
            market_value=market_val,
            positions=self.positions
        )

    def get_positions(self) -> Dict[str, Position]:
        return self.positions

    def send_order(
        self,
        symbol: str,
        side: OrderSide,
        shares: int,
        price: float,
        order_type: OrderType = OrderType.LIMIT
    ) -> ExecutionOrder:
        order_id = f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:6]}"
        
        # 100 股向下取整
        shares = (shares // 100) * 100
        if shares <= 0:
            order = ExecutionOrder(
                order_id=order_id, symbol=symbol, side=side, order_type=order_type,
                requested_shares=shares, requested_price=price, status=ExecutionStatus.REJECTED,
                error_msg="委托数量必须为 100 股整数倍"
            )
            self.orders.append(order)
            return order

        trade_amount = shares * price
        if side == OrderSide.BUY:
            fee = max(5.0, trade_amount * self.commission_rate)
            total_cost = trade_amount + fee
            if self.cash < total_cost:
                order = ExecutionOrder(
                    order_id=order_id, symbol=symbol, side=side, order_type=order_type,
                    requested_shares=shares, requested_price=price, status=ExecutionStatus.REJECTED,
                    error_msg=f"资金不足 (需要: {total_cost:.2f}, 可用: {self.cash:.2f})"
                )
                self.orders.append(order)
                return order

            # 扣款与更新持仓
            self.cash -= total_cost
            pos = self.positions.get(symbol, Position(symbol=symbol))
            old_shares = pos.total_shares
            new_shares = old_shares + shares
            pos.avg_cost_price = (pos.avg_cost_price * old_shares + trade_amount) / new_shares
            pos.total_shares = new_shares
            # T+1 语义 (已修复): 当日买入计入总持仓但不可用, 跨日由 unlock_t1_shares() 解锁
            pos.available_shares = pos.available_shares
            self._last_buy_dates[symbol] = datetime.now().strftime("%Y-%m-%d")
            pos.current_price = price
            pos.market_value = pos.total_shares * price
            pos.floating_pnl = (pos.current_price - pos.avg_cost_price) * pos.total_shares
            pos.floating_pnl_pct = (pos.current_price / pos.avg_cost_price - 1.0) * 100 if pos.avg_cost_price > 0 else 0.0
            self.positions[symbol] = pos

            order = ExecutionOrder(
                order_id=order_id, symbol=symbol, side=side, order_type=order_type,
                requested_shares=shares, requested_price=price, filled_shares=shares,
                avg_filled_price=price, status=ExecutionStatus.FILLED,
                created_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            self.orders.append(order)
            self._save_state()
            return order

        else: # SELL
            pos = self.positions.get(symbol)
            if pos is None or pos.available_shares < shares:
                order = ExecutionOrder(
                    order_id=order_id, symbol=symbol, side=side, order_type=order_type,
                    requested_shares=shares, requested_price=price, status=ExecutionStatus.REJECTED,
                    error_msg=f"可用股份不足 (可用: {pos.available_shares if pos else 0})"
                )
                self.orders.append(order)
                return order

            fee = max(5.0, trade_amount * self.commission_rate) + trade_amount * self.stamp_duty
            net_income = trade_amount - fee
            self.cash += net_income
            pos.total_shares -= shares
            pos.available_shares -= shares
            pos.market_value = pos.total_shares * price
            if pos.total_shares == 0:
                del self.positions[symbol]
            else:
                self.positions[symbol] = pos

            order = ExecutionOrder(
                order_id=order_id, symbol=symbol, side=side, order_type=order_type,
                requested_shares=shares, requested_price=price, filled_shares=shares,
                avg_filled_price=price, status=ExecutionStatus.FILLED,
                created_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            self.orders.append(order)
            self._save_state()
            return order

    def cancel_order(self, order_id: str) -> bool:
        for ord in self.orders:
            if ord.order_id == order_id and ord.status in (ExecutionStatus.SUBMITTED, ExecutionStatus.PENDING):
                ord.status = ExecutionStatus.CANCELLED
                return True
        return False

    def cancel_all_pending_orders(self) -> int:
        """全量撤销挂起未成交订单"""
        cancelled_count = 0
        for ord in self.orders:
            if ord.status in (ExecutionStatus.SUBMITTED, ExecutionStatus.PENDING):
                ord.status = ExecutionStatus.CANCELLED
                cancelled_count += 1
        if cancelled_count > 0:
            logger.info(f"🛡️ [前置撤单防线] 已自动撤销 {cancelled_count} 笔挂起订单")
        return cancelled_count

    def get_orders(self, status: Optional[ExecutionStatus] = None) -> List[ExecutionOrder]:
        if status is None:
            return self.orders
        return [o for o in self.orders if o.status == status]

    def unlock_t1_shares(self, today: Optional[str] = None):
        """跨日调用：将昨日及更早买入的持仓解锁为可用股数 (T+1)。

        Args:
            today: 当前交易日 (YYYY-MM-DD)。为 None 时无条件解锁全部持仓；
                   给定日期时仅解锁"最后买入日 < today"的标的，当日买入的股份保持锁定。
        """
        unlocked = []
        for sym, pos in self.positions.items():
            if pos.total_shares <= pos.available_shares:
                continue  # 已全部可用, 无需解锁
            if today is not None:
                last_buy = self._last_buy_dates.get(sym)
                if last_buy is not None and last_buy >= today:
                    continue  # 当日买入, 保持 T+1 锁定
            pos.available_shares = pos.total_shares
            unlocked.append(f"{sym}({pos.total_shares}股)")
        if unlocked:
            logger.info(f"已执行 T+1 跨日持仓可用股数解锁: {', '.join(unlocked)}")
        else:
            logger.info("T+1 解锁扫描完成: 无需解锁的持仓")
