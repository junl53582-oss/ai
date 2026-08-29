"""
走步事件驱动量化回测引擎 (backtest/engine.py)
严格遵循 Point-In-Time 原则与状态守恒定律：
1. 订单数量守恒 (requested == cumulative_filled + remaining + cancelled) 与余量顺延执行
2. 标的共享单日流动性容量 (Per-Symbol-Per-Day Shared Volume Capacity)
3. 公司行为除权除息全状态守恒：
   - 送转股同步除权 highest_price (杜绝 trailing stop 虚假回撤) 与 pending 订单股数
   - 现金分红归集至 Lot accumulated_cash_dividend 并计入闭环平仓真实 Net Total Return
4. 停牌 Mark-to-Market 估值日期保持与严格可解释性 Stale Price 审计指标 (distinct events / symbol-days)
5. 回测结束显式撤销余量挂单 (END_OF_BACKTEST)
"""
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np

from config.settings import settings
from strategy.trading_rules import (
    AShareTradingRules,
    PositionRecord,
    PositionLot,
    Order,
    OrderStatus,
    OrderSide,
    RejectReason,
    recalculate_position_from_lots
)
from strategy.portfolio import PortfolioBuilder
from strategy.corporate_actions import CorporateActionProvider, CorporateAction
from data.data_manager import DataManager, count_trading_days
from .risk_control import RiskManager

logger = logging.getLogger(__name__)


class BacktestEngine:
    """A股实盘级量化走步回测引擎 (Hardened Enterprise Release v2.0)"""

    def __init__(
        self,
        initial_cash: float = settings.INITIAL_CASH,
        top_k_buy: int = settings.TOP_K_BUY,
        top_k_hold: int = settings.TOP_K_HOLD,
        rebalance_freq: int = settings.REBALANCE_FREQ,
        rules: Optional[AShareTradingRules] = None,
        portfolio_builder: Optional[PortfolioBuilder] = None,
        risk_manager: Optional[RiskManager] = None,
        corporate_actions: Optional[CorporateActionProvider] = None,
        enable_liquidity_constraint: bool = True,
        max_volume_participation: float = 0.05
    ):
        self.initial_cash = initial_cash
        self.top_k_buy = top_k_buy
        self.top_k_hold = top_k_hold
        self.rebalance_freq = rebalance_freq

        self.rules = rules or AShareTradingRules()
        self.builder = portfolio_builder or PortfolioBuilder(
            top_k_buy=top_k_buy,
            top_k_hold=top_k_hold,
            trading_rules=self.rules
        )
        self.risk_manager = risk_manager or RiskManager()
        self.corporate_actions = corporate_actions or CorporateActionProvider()
        self.enable_liquidity_constraint = enable_liquidity_constraint
        self.max_volume_participation = max_volume_participation

        # 运行时状态
        self.cash = initial_cash
        self.positions: Dict[str, PositionRecord] = {}
        self.pending_orders: List[Order] = []
        self.completed_orders: List[Order] = []
        self.executed_fills: List[Dict[str, Any]] = []
        self.closed_trades: List[Dict[str, Any]] = []
        self.daily_records: List[Dict[str, Any]] = []
        self.peak_equity = initial_cash
        self.trading_calendar: List[pd.Timestamp] = []
        self._order_counter = 0

        # 可解释性审计统计 (P1-1, P0-5, P0-6)
        self.stale_price_warning_events: int = 0      # 首次跨过阈值的独立停牌事件数
        self.stale_price_symbol_days: int = 0         # 处于超期停牌状态的标的-交易日总数
        self.stale_price_affected_symbols: Set[str] = set()
        self.max_stale_price_days: int = 0
        self._stale_active_symbols: Set[str] = set()

        self.partial_fill_count: int = 0
        self.liquidity_rejected_count: int = 0
        self.pending_order_count: int = 0
        self.cancelled_order_count: int = 0
        self.deferred_order_count: int = 0
        self.order_quantity_conservation_passed: bool = True
        
        self.corporate_action_adjustment_available: bool = self.corporate_actions.has_actions_data()
        self.backtest_total_return_reliability: str = "standard" if self.corporate_actions.coverage_complete else "limited"

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"ORD_{self._order_counter:06d}"

    def _apply_corporate_actions(self, date_str: str):
        """开盘前应用当日生效的公司行为 (除息分红与送转股，全状态守恒调整 P0-7)"""
        actions = self.corporate_actions.get_actions_on_date(date_str)
        for act in actions:
            sym = act.symbol
            if sym in self.positions:
                pos = self.positions[sym]
                
                # 1. 现金分红
                if act.cash_dividend_per_share > 0:
                    dividend_amount = pos.shares * act.cash_dividend_per_share
                    self.cash += dividend_amount
                    # 归集到各个 Lot 的累积分红
                    for lot in pos.lots:
                        lot.accumulated_cash_dividend += lot.shares * act.cash_dividend_per_share
                    
                    # 调整估值价格基准与最高价基准，防止机械除息触发虚假移动止损 (P0-15)
                    pos.last_price = max(0.01, pos.last_price - act.cash_dividend_per_share)
                    pos.highest_price = max(pos.last_price, pos.highest_price - act.cash_dividend_per_share)
                    logger.info(f"💰 [{date_str}] {sym} 实施现金分红: 每股派现 {act.cash_dividend_per_share:.2f} 元，获得分红现金 {dividend_amount:,.2f} 元")

                # 2. 送转股 / 拆股
                if act.share_ratio > 0:
                    multiplier = 1.0 + act.share_ratio
                    for lot in pos.lots:
                        lot.shares = int(lot.shares * multiplier)
                        lot.buy_execution_price = lot.buy_execution_price / multiplier
                    recalculate_position_from_lots(pos)
                    pos.last_price = pos.last_price / multiplier
                    
                    # 必须同步除权 highest_price，杜绝 trailing stop 虚假大回撤！ (P0-7, P0-15)
                    pos.highest_price = pos.highest_price / multiplier

                    # 同步等比调整所有未完成的 Pending 订单股数与全状态 (保持数量守恒 P0-14)
                    for order in self.pending_orders:
                        if order.symbol == sym:
                            order.requested_shares = int(order.requested_shares * multiplier)
                            order.cumulative_filled_shares = int(order.cumulative_filled_shares * multiplier)
                            order.cancelled_shares = int(order.cancelled_shares * multiplier)
                            order.remaining_shares = order.requested_shares - order.cumulative_filled_shares - order.cancelled_shares
                            order.signal_price = order.signal_price / multiplier

                    logger.info(f"📈 [{date_str}] {sym} 实施送转股: 每股送转 {act.share_ratio:.2f} 股，持仓调整为 {pos.shares} 股，最高价基准同步调整为 {pos.highest_price:.2f} 元")

    def mark_to_market(self, current_date: pd.Timestamp, daily_map: Dict[str, pd.Series]) -> Tuple[float, float]:
        """
        逐日盯市估值 (Mark-to-Market)：
        1. 当日有成交且未停牌：使用当日收盘价更新 last_price 与 last_price_date
        2. 当日无行情或处于停牌状态：维持原 last_price，绝不刷新 last_price_date
        3. 精确统计 distinct stale events 与 symbol-days (P1-1)
        """
        date_str = current_date.strftime("%Y-%m-%d")
        holdings_market_value = 0.0

        for sym, pos in self.positions.items():
            if sym in daily_map:
                row = daily_map[sym]
                is_suspended = bool(row.get("is_suspended", False) or row.get("volume", 0) <= 0)
                close_p = float(row.get("close", 0.0))
                
                if not is_suspended and close_p > 0:
                    pos.last_price = close_p
                    pos.last_price_date = date_str
                    self._stale_active_symbols.discard(sym)

            if pos.last_price_date:
                stale_days = (current_date - pd.to_datetime(pos.last_price_date)).days
                if stale_days > settings.MAX_STALE_PRICE_DAYS:
                    if sym not in self._stale_active_symbols:
                        self.stale_price_warning_events += 1
                        self._stale_active_symbols.add(sym)
                    self.stale_price_affected_symbols.add(sym)
                    self.stale_price_symbol_days += 1
                    self.max_stale_price_days = max(self.max_stale_price_days, stale_days)

            holdings_market_value += pos.shares * pos.last_price

        total_equity = self.cash + holdings_market_value
        return holdings_market_value, total_equity

    def run(self, oos_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """执行严格走步回测时序循环"""
        logger.info(f"启动 A股实盘级走步回测 (初始资金: {self.initial_cash:,.2f} 元)...")
        df = oos_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(by=["date", "symbol"], inplace=True)

        trading_dates = sorted(df["date"].unique())
        self.trading_calendar = trading_dates
        total_dates = len(trading_dates)

        if total_dates == 0:
            raise ValueError("回测输入数据集为空！")

        first_date_df = df[df["date"] == trading_dates[0]]
        initial_bench_price = first_date_df["benchmark_close"].iloc[0] if "benchmark_close" in first_date_df.columns else 1.0

        # 验证公司行为覆盖范围 (P0/P1-4: 传递真实 evidence_dir 进行物理证据核验)
        all_symbols = df["symbol"].unique().tolist()
        s_date_str = trading_dates[0].strftime("%Y-%m-%d")
        e_date_str = trading_dates[-1].strftime("%Y-%m-%d")
        ev_dir = getattr(self.corporate_actions, "evidence_dir", None)
        self.corporate_actions.validate_coverage(all_symbols, s_date_str, e_date_str, evidence_dir=ev_dir)
        self.corporate_action_adjustment_available = self.corporate_actions.has_actions_data()
        self.backtest_total_return_reliability = "standard" if self.corporate_actions.coverage_complete else "limited"

        for t_idx, current_date in enumerate(trading_dates):
            date_str = current_date.strftime("%Y-%m-%d")
            daily_df = df[df["date"] == current_date].copy()
            daily_map = {row["symbol"]: row for _, row in daily_df.iterrows()}

            # ---------------- 阶段 0: 开盘前公司行为除息调整 ----------------
            self._apply_corporate_actions(date_str)

            # ---------------- 阶段 1: T 日开盘撮合执行 ----------------
            self._execute_morning_auction(daily_map, date_str)

            # ---------------- 阶段 2: T 日收盘盯市结算 ----------------
            holdings_market_value, total_equity = self.mark_to_market(current_date, daily_map)
            self.peak_equity = max(self.peak_equity, total_equity)

            bench_price = float(daily_df["benchmark_close"].iloc[0]) if "benchmark_close" in daily_df.columns else 1.0
            bench_equity = self.initial_cash * (bench_price / initial_bench_price)

            self.daily_records.append({
                "date": current_date,
                "total_equity": total_equity,
                "cash": self.cash,
                "holdings_value": holdings_market_value,
                "num_positions": len(self.positions),
                "benchmark_equity": bench_equity,
                "benchmark_close": bench_price
            })

            # ---------------- 阶段 3: T 日收盘后生成交易决策 ----------------
            is_last_day = (t_idx == total_dates - 1)
            if not is_last_day:
                self._generate_evening_signals(daily_df, daily_map, date_str, total_equity, t_idx)
            else:
                # 回测最后一个交易日结束：清理所有剩余未成交/延期挂单
                self._cancel_remaining_pending_orders(date_str)

        # 整理输出结果
        equity_df = pd.DataFrame(self.daily_records)
        equity_df["strategy_return"] = equity_df["total_equity"].pct_change().fillna(0.0)
        equity_df["benchmark_return"] = equity_df["benchmark_equity"].pct_change().fillna(0.0)
        equity_df["excess_return"] = equity_df["strategy_return"] - equity_df["benchmark_return"]
        equity_df["cum_strategy_return"] = equity_df["total_equity"] / self.initial_cash - 1.0
        equity_df["cum_benchmark_return"] = equity_df["benchmark_equity"] / self.initial_cash - 1.0

        orders_df = pd.DataFrame([
            {
                "order_id": o.order_id,
                "parent_order_id": o.parent_order_id,
                "symbol": o.symbol,
                "side": o.side.value,
                "signal_date": o.signal_date,
                "execution_date": o.execution_date,
                "requested_shares": o.requested_shares,
                "cumulative_filled_shares": o.cumulative_filled_shares,
                "remaining_shares": o.remaining_shares,
                "cancelled_shares": o.cancelled_shares,
                "filled_shares": o.filled_shares,
                "signal_price": o.signal_price,
                "execution_price": o.execution_price,
                "commission": o.commission,
                "stamp_tax": o.stamp_tax,
                "transfer_fee": o.transfer_fee,
                "slippage": o.slippage,
                "total_cost": round(o.commission + o.stamp_tax + o.transfer_fee + o.slippage, 2),
                "status": o.status.value,
                "reason": o.reason,
                "reject_reason": o.reject_reason
            }
            for o in self.completed_orders
        ])

        orders_df.attrs["closed_trades"] = self.closed_trades
        orders_df.attrs["executed_fills"] = self.executed_fills

        # 校验全订单数量守恒
        self.order_quantity_conservation_passed = all(o.verify_quantity_conservation() for o in self.completed_orders)

        # 统计订单终态
        self.pending_order_count = len(self.pending_orders)
        self.cancelled_order_count = sum(1 for o in self.completed_orders if o.status == OrderStatus.CANCELLED)
        self.deferred_order_count = sum(1 for o in self.completed_orders if o.status == OrderStatus.DEFERRED)

        logger.info(
            f"回测完成！总交易日: {len(equity_df)} | 完成订单数: {len(orders_df)} (取消: {self.cancelled_order_count}) | "
            f"平仓批次数: {len(self.closed_trades)} | 部分成交: {self.partial_fill_count} 次 | 流动性拒绝: {self.liquidity_rejected_count} 次 | "
            f"订单数量守恒校验: {'通过' if self.order_quantity_conservation_passed else '失败'}"
        )
        return equity_df, orders_df

    def _cancel_remaining_pending_orders(self, date_str: str):
        """回测结束时强制取消所有仍处于挂单/延期状态的订单 (满足数量守恒)"""
        for order in self.pending_orders:
            order.cancelled_shares = order.remaining_shares
            order.remaining_shares = 0
            order.status = OrderStatus.CANCELLED
            order.reject_reason = RejectReason.END_OF_BACKTEST.value
            order.execution_date = date_str
            self.completed_orders.append(order)
        self.pending_orders.clear()

    def _execute_morning_auction(self, daily_map: Dict[str, pd.Series], date_str: str):
        """
        开盘撮合逻辑：
        1. 建立标的共享单日流动性容量 (Per-Symbol-Per-Day Shared Volume Capacity, P0-5)
        2. 部分成交余量严格保留，继续顺延至下一交易日挂单 (P0-5)
        3. FIFO 批次买卖配对，将持有期间现金分红完整计入真实 Net Total Return (P0-7)
        """
        # 解锁历史持仓 T+1 卖出限制
        for sym, pos in self.positions.items():
            if pos.buy_date < date_str:
                pos.available_shares = pos.shares

        if not self.pending_orders:
            return

        # 计算当日各股票的共享可用流动性容量
        shared_capacity: Dict[str, int] = {}
        for sym, row in daily_map.items():
            day_vol = float(row.get("volume", 0))
            if day_vol > 0:
                shared_capacity[sym] = int((day_vol * self.max_volume_participation) // self.rules.lot_size) * self.rules.lot_size
            else:
                shared_capacity[sym] = 0

        orders_to_process = self.pending_orders
        self.pending_orders = []

        sell_orders = [o for o in orders_to_process if o.side == OrderSide.SELL]
        buy_orders = [o for o in orders_to_process if o.side == OrderSide.BUY]

        # ---------------- 1. 优先撮合卖单 ----------------
        for order in sell_orders:
            sym = order.symbol
            if sym not in daily_map:
                order.status = OrderStatus.DEFERRED
                order.reject_reason = RejectReason.NO_MARKET_DATA.value
                self.pending_orders.append(order)
                continue

            if sym not in self.positions:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "已无持仓标的"
                order.cancelled_shares = order.remaining_shares
                order.remaining_shares = 0
                self.completed_orders.append(order)
                continue

            row = daily_map[sym]
            pos = self.positions[sym]
            open_price = float(row.get("open", row.get("close", 0.0)))
            limit_down_p = float(row.get("limit_down_price", 0.0))

            raw_sell_p = open_price * (1.0 - settings.SLIPPAGE_RATE)
            exec_price = max(round(raw_sell_p, 2), limit_down_p)

            can_s, msg = self.rules.can_sell(row, pos, exec_price)
            if not can_s:
                order.status = OrderStatus.DEFERRED
                order.reject_reason = msg
                self.pending_orders.append(order)
                continue

            shares_requested = order.remaining_shares
            shares_available = pos.available_shares

            if shares_available <= 0:
                order.status = OrderStatus.DEFERRED
                order.reject_reason = RejectReason.T_PLUS_1_LOCK.value
                self.pending_orders.append(order)
                continue

            shares_to_sell = min(shares_requested, shares_available)

            # 流动性共享容量扣减
            if self.enable_liquidity_constraint:
                cap = shared_capacity.get(sym, 0)
                if cap < self.rules.lot_size:
                    order.status = OrderStatus.DEFERRED
                    order.reject_reason = RejectReason.LIQUIDITY_LIMIT.value
                    self.liquidity_rejected_count += 1
                    self.pending_orders.append(order)
                    continue

                if shares_to_sell > cap:
                    shares_to_sell = cap
                    self.partial_fill_count += 1

                shared_capacity[sym] = max(0, cap - shares_to_sell)

            trade_amount = shares_to_sell * exec_price
            comm, stamp, trans_fee, _ = self.rules.compute_transaction_cost(trade_amount, is_buy=False, trade_date=date_str)
            net_cash = trade_amount - (comm + stamp + trans_fee)
            slippage_val = round(abs(open_price - exec_price) * shares_to_sell, 2)

            # FIFO 逐批次平仓记录 (将持仓期间累积分红计入真实 Net Total Return)
            remaining_to_sell = shares_to_sell
            while remaining_to_sell > 0 and pos.lots:
                lot = pos.lots[0]
                if lot.shares <= remaining_to_sell:
                    lot_shares = lot.shares
                    lot_buy_comm = lot.buy_commission
                    lot_div = lot.accumulated_cash_dividend
                    pos.lots.pop(0)
                else:
                    lot_shares = remaining_to_sell
                    lot_buy_comm = lot.buy_commission * (lot_shares / lot.shares)
                    lot_div = lot.accumulated_cash_dividend * (lot_shares / lot.shares)
                    lot.buy_commission -= lot_buy_comm
                    lot.accumulated_cash_dividend -= lot_div
                    lot.shares -= remaining_to_sell

                remaining_to_sell -= lot_shares

                lot_sell_comm = comm * (lot_shares / shares_to_sell)
                lot_stamp_tax = stamp * (lot_shares / shares_to_sell)
                lot_trans_fee = trans_fee * (lot_shares / shares_to_sell)
                lot_sell_fees = lot_sell_comm + lot_stamp_tax + lot_trans_fee

                gross_pnl_amount = (exec_price - lot.buy_execution_price) * lot_shares
                # 真实净收益 = 价差收益 + 持有期现金分红 - 买入费用 - 卖出费用
                net_pnl_amount = gross_pnl_amount + lot_div - lot_buy_comm - lot_sell_fees
                
                gross_pnl_pct = (exec_price - lot.buy_execution_price) / lot.buy_execution_price
                net_pnl_pct = net_pnl_amount / (lot.buy_execution_price * lot_shares)

                lot_buy_d = pd.to_datetime(lot.buy_date)
                lot_sell_d = pd.to_datetime(date_str)
                holding_cal_days = max(1, (lot_sell_d - lot_buy_d).days)
                holding_trd_days = count_trading_days(lot_buy_d, lot_sell_d, self.trading_calendar)

                self.closed_trades.append({
                    "symbol": sym,
                    "buy_date": lot.buy_date,
                    "sell_date": date_str,
                    "shares": lot_shares,
                    "buy_cost": lot.buy_execution_price,
                    "sell_price": exec_price,
                    "gross_realized_pnl_pct": gross_pnl_pct,
                    "net_realized_pnl_pct": net_pnl_pct,
                    "realized_pnl_pct": net_pnl_pct,
                    "gross_pnl_amount": round(gross_pnl_amount, 2),
                    "net_pnl_amount": round(net_pnl_amount, 2),
                    "cash_dividends": round(lot_div, 2),
                    "buy_commission": round(lot_buy_comm, 2),
                    "sell_commission": round(lot_sell_comm, 2),
                    "stamp_tax": round(lot_stamp_tax, 2),
                    "transfer_fee": round(lot_trans_fee, 2),
                    "holding_days": holding_trd_days,
                    "holding_trading_days": holding_trd_days,
                    "holding_calendar_days": holding_cal_days,
                    "reason": order.reason
                })

            self.cash += net_cash
            pos.available_shares -= shares_to_sell
            recalculate_position_from_lots(pos)
            pos.last_price = exec_price
            pos.last_price_date = date_str

            order.execution_date = date_str
            order.filled_shares = shares_to_sell
            order.cumulative_filled_shares += shares_to_sell
            order.remaining_shares = order.requested_shares - order.cumulative_filled_shares
            order.execution_price = exec_price
            order.commission = round(comm, 2)
            order.stamp_tax = round(stamp, 2)
            order.transfer_fee = round(trans_fee, 2)
            order.slippage = slippage_val

            if order.remaining_shares == 0:
                order.status = OrderStatus.FILLED
                self.completed_orders.append(order)
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                # 记录当次部分成交记录
                fill_record = Order(
                    order_id=f"{order.order_id}_P{order.cumulative_filled_shares}",
                    parent_order_id=order.order_id,
                    symbol=sym,
                    side=order.side,
                    signal_date=order.signal_date,
                    execution_date=date_str,
                    requested_shares=order.requested_shares,
                    cumulative_filled_shares=order.cumulative_filled_shares,
                    remaining_shares=order.remaining_shares,
                    filled_shares=shares_to_sell,
                    signal_price=order.signal_price,
                    execution_price=exec_price,
                    commission=order.commission,
                    stamp_tax=order.stamp_tax,
                    transfer_fee=order.transfer_fee,
                    slippage=order.slippage,
                    status=OrderStatus.PARTIALLY_FILLED,
                    reason=order.reason
                )
                self.completed_orders.append(fill_record)
                # 余量继续在下一交易日挂单撮合 (P0-5)
                self.pending_orders.append(order)

            if pos.shares <= 0:
                del self.positions[sym]

        # ---------------- 2. 撮合买单 ----------------
        for order in buy_orders:
            sym = order.symbol
            if sym not in daily_map:
                order.status = OrderStatus.REJECTED
                order.reject_reason = RejectReason.NO_MARKET_DATA.value
                order.cancelled_shares = order.remaining_shares
                order.remaining_shares = 0
                self.completed_orders.append(order)
                continue

            row = daily_map[sym]
            open_price = float(row.get("open", row.get("close", 0.0)))
            limit_up_p = float(row.get("limit_up_price", 999999.0))

            raw_buy_p = open_price * (1.0 + settings.SLIPPAGE_RATE)
            exec_price = min(round(raw_buy_p, 2), limit_up_p)

            can_b, msg = self.rules.can_buy(row, exec_price)
            if not can_b:
                order.status = OrderStatus.REJECTED
                order.reject_reason = msg
                order.cancelled_shares = order.remaining_shares
                order.remaining_shares = 0
                self.completed_orders.append(order)
                continue

            shares_to_buy = order.remaining_shares

            # 流动性共享容量扣减
            if self.enable_liquidity_constraint:
                cap = shared_capacity.get(sym, 0)
                if cap < self.rules.lot_size:
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = RejectReason.LIQUIDITY_LIMIT.value
                    order.cancelled_shares = order.remaining_shares
                    order.remaining_shares = 0
                    self.liquidity_rejected_count += 1
                    self.completed_orders.append(order)
                    continue

                if shares_to_buy > cap:
                    shares_to_buy = cap
                    self.partial_fill_count += 1

                shared_capacity[sym] = max(0, cap - shares_to_buy)

            cost_estimate = shares_to_buy * exec_price * (1.0 + settings.COMMISSION_RATE)
            if cost_estimate > self.cash:
                max_shares = self.rules.calculate_lot_shares(self.cash * 0.99, exec_price)
                if max_shares < self.rules.lot_size:
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = RejectReason.INSUFFICIENT_CASH.value
                    order.cancelled_shares = order.remaining_shares
                    order.remaining_shares = 0
                    self.completed_orders.append(order)
                    continue
                shares_to_buy = min(shares_to_buy, max_shares)

            trade_amount = shares_to_buy * exec_price
            comm, stamp, trans_fee, _ = self.rules.compute_transaction_cost(trade_amount, is_buy=True, trade_date=date_str)
            total_spent = trade_amount + comm + trans_fee
            slippage_val = round(abs(exec_price - open_price) * shares_to_buy, 2)

            if total_spent > self.cash:
                order.status = OrderStatus.REJECTED
                order.reject_reason = RejectReason.INSUFFICIENT_CASH.value
                order.cancelled_shares = order.remaining_shares
                order.remaining_shares = 0
                self.completed_orders.append(order)
                continue

            self.cash -= total_spent

            # 维护 FIFO Lots
            new_lot = PositionLot(
                shares=shares_to_buy,
                buy_execution_price=exec_price,
                buy_date=date_str,
                buy_commission=round(comm + trans_fee, 2),
                accumulated_cash_dividend=0.0
            )
            if sym in self.positions:
                pos = self.positions[sym]
                pos.lots.append(new_lot)
                recalculate_position_from_lots(pos)
                pos.last_price = exec_price
                pos.last_price_date = date_str
                pos.highest_price = max(pos.highest_price, exec_price)
            else:
                self.positions[sym] = PositionRecord(
                    symbol=sym,
                    shares=shares_to_buy,
                    available_shares=0,
                    avg_cost=exec_price,
                    last_price=exec_price,
                    buy_date=date_str,
                    highest_price=exec_price,
                    last_price_date=date_str,
                    lots=[new_lot]
                )

            order.execution_date = date_str
            order.filled_shares = shares_to_buy
            order.cumulative_filled_shares += shares_to_buy
            order.remaining_shares = order.requested_shares - order.cumulative_filled_shares
            order.execution_price = exec_price
            order.commission = round(comm, 2)
            order.stamp_tax = 0.0
            order.transfer_fee = round(trans_fee, 2)
            order.slippage = slippage_val

            if order.remaining_shares == 0:
                order.status = OrderStatus.FILLED
                self.completed_orders.append(order)
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                fill_record = Order(
                    order_id=f"{order.order_id}_P{order.cumulative_filled_shares}",
                    parent_order_id=order.order_id,
                    symbol=sym,
                    side=order.side,
                    signal_date=order.signal_date,
                    execution_date=date_str,
                    requested_shares=order.requested_shares,
                    cumulative_filled_shares=order.cumulative_filled_shares,
                    remaining_shares=order.remaining_shares,
                    filled_shares=shares_to_buy,
                    signal_price=order.signal_price,
                    execution_price=exec_price,
                    commission=order.commission,
                    stamp_tax=0.0,
                    transfer_fee=order.transfer_fee,
                    slippage=order.slippage,
                    status=OrderStatus.PARTIALLY_FILLED,
                    reason=order.reason
                )
                self.completed_orders.append(fill_record)
                self.pending_orders.append(order)

    def _generate_evening_signals(
        self,
        daily_df: pd.DataFrame,
        daily_map: Dict[str, pd.Series],
        date_str: str,
        total_equity: float,
        t_idx: int
    ):
        """T 日收盘后生成交易决策"""
        existing_order_symbols = {o.symbol for o in self.pending_orders}

        # 1. 个股风控检测
        for sym, pos in list(self.positions.items()):
            if sym in daily_map and sym not in existing_order_symbols:
                row = daily_map[sym]
                triggered, reason = self.risk_manager.update_and_check_position_risk(pos, row)
                if triggered:
                    self.pending_orders.append(Order(
                        order_id=self._next_order_id(),
                        symbol=sym,
                        side=OrderSide.SELL,
                        signal_date=date_str,
                        requested_shares=pos.shares,
                        remaining_shares=pos.shares,
                        signal_price=float(row["close"]),
                        status=OrderStatus.PENDING,
                        reason=reason
                    ))
                    existing_order_symbols.add(sym)

        # 2. 组合最大回撤熔断降仓
        holdings_val = sum(pos.shares * pos.last_price for pos in self.positions.values())
        circuit_triggered, target_holdings_val = self.risk_manager.check_portfolio_circuit_breaker(
            total_equity, self.peak_equity, holdings_val
        )
        if circuit_triggered and holdings_val > target_holdings_val:
            reduction_ratio = (holdings_val - target_holdings_val) / holdings_val
            for sym, pos in list(self.positions.items()):
                if sym not in existing_order_symbols:
                    sell_shares = self.rules.calculate_lot_shares(pos.shares * pos.last_price * reduction_ratio, pos.last_price)
                    if sell_shares >= self.rules.lot_size:
                        self.pending_orders.append(Order(
                            order_id=self._next_order_id(),
                            symbol=sym,
                            side=OrderSide.SELL,
                            signal_date=date_str,
                            requested_shares=sell_shares,
                            remaining_shares=sell_shares,
                            signal_price=pos.last_price,
                            status=OrderStatus.PENDING,
                            reason="组合最大回撤熔断降仓"
                        ))
                        existing_order_symbols.add(sym)

        # 3. 定期调仓与双向再平衡
        is_rebalance_day = (t_idx % self.rebalance_freq == 0)
        if is_rebalance_day:
            current_holdings = set(self.positions.keys())
            target_df = self.builder.build_target_portfolio(daily_df, current_holdings, date=date_str)

            if not target_df.empty:
                target_symbols = set(target_df["symbol"].tolist())

                # A. 卖出落选标的
                for sym in current_holdings:
                    if sym not in target_symbols and sym not in existing_order_symbols:
                        pos = self.positions[sym]
                        close_p = float(daily_map[sym]["close"]) if sym in daily_map else pos.last_price
                        self.pending_orders.append(Order(
                            order_id=self._next_order_id(),
                            symbol=sym,
                            side=OrderSide.SELL,
                            signal_date=date_str,
                            requested_shares=pos.shares,
                            remaining_shares=pos.shares,
                            signal_price=close_p,
                            status=OrderStatus.PENDING,
                            reason="调仓剔除(掉出Top-K)"
                        ))
                        existing_order_symbols.add(sym)

                # B. 目标池标的双向再平衡
                avail_cash_ratio = settings.CIRCUIT_TARGET_EXPOSURE if self.risk_manager.is_circuit_breaker_active else 0.95
                for _, trow in target_df.iterrows():
                    sym = trow["symbol"]
                    if sym in existing_order_symbols or sym not in daily_map:
                        continue

                    row = daily_map[sym]
                    close_p = float(row["close"])
                    target_w = float(trow["target_weight"])
                    target_val = total_equity * avail_cash_ratio * target_w

                    current_shares = self.positions[sym].shares if sym in self.positions else 0
                    target_shares = self.rules.calculate_lot_shares(target_val, close_p)

                    if target_shares > current_shares:
                        buy_delta = target_shares - current_shares
                        buy_shares = (buy_delta // self.rules.lot_size) * self.rules.lot_size
                        if buy_shares >= self.rules.lot_size:
                            self.pending_orders.append(Order(
                                order_id=self._next_order_id(),
                                symbol=sym,
                                side=OrderSide.BUY,
                                signal_date=date_str,
                                requested_shares=buy_shares,
                                remaining_shares=buy_shares,
                                signal_price=close_p,
                                status=OrderStatus.PENDING,
                                reason="Top-K目标建仓/加仓"
                            ))
                            existing_order_symbols.add(sym)

                    elif target_shares < current_shares and sym in self.positions:
                        sell_delta = current_shares - target_shares
                        sell_shares = (sell_delta // self.rules.lot_size) * self.rules.lot_size
                        if sell_shares >= self.rules.lot_size:
                            self.pending_orders.append(Order(
                                order_id=self._next_order_id(),
                                symbol=sym,
                                side=OrderSide.SELL,
                                signal_date=date_str,
                                requested_shares=sell_shares,
                                remaining_shares=sell_shares,
                                signal_price=close_p,
                                status=OrderStatus.PENDING,
                                reason="Top-K目标再平衡减仓"
                            ))
                            existing_order_symbols.add(sym)

    @property
    def corporate_action_source(self) -> str:
        return getattr(self.corporate_actions, "source", "unknown")

    @property
    def corporate_action_coverage_ratio(self) -> float:
        return getattr(self.corporate_actions, "coverage_ratio", 0.0)

    @property
    def corporate_action_coverage_complete(self) -> bool:
        return getattr(self.corporate_actions, "coverage_complete", False)

    @property
    def corporate_action_bias_risk(self) -> bool:
        return not self.corporate_action_coverage_complete

    @property
    def corporate_action_zero_event_proof_verified(self) -> bool:
        return getattr(self.corporate_actions, "zero_event_proof_verified", False)

    @property
    def corporate_action_provenance_verified(self) -> bool:
        return getattr(self.corporate_actions, "corporate_action_provenance_verified", False)

    @property
    def corporate_action_dataset_hash_verified(self) -> bool:
        return getattr(self.corporate_actions, "corporate_action_dataset_hash_verified", False)

    @property
    def corporate_action_manifest_hash(self) -> Optional[str]:
        return getattr(self.corporate_actions, "corporate_action_manifest_hash", None)

    @property
    def corporate_action_manifest_hash_verified(self) -> bool:
        return getattr(self.corporate_actions, "corporate_action_manifest_hash_verified", False)

    @property
    def corporate_action_manifest_result(self) -> Optional[Any]:
        return getattr(self.corporate_actions, "manifest_verification_result", None)
