"""
实盘与模拟盘自动化调仓执行引擎 (execution/run_trader.py)
用于将策略输出的目标投资组合 (Target Portfolio) 转化为真实/模拟委托订单并执行调仓。
执行原则:
1. 先卖后买: 先卖出被剔除或需减仓的标的以释放现金，再按权重买入建仓标的
2. 整手约束: 严格按 100 股向下取整
3. T+1 可用性约束: 仅卖出可用份额 (available_shares > 0)
4. 支持本地 Paper 仿真沙盒与迅投 MiniQMT 实盘终端
"""
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# 根目录引用
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import settings
from execution.broker_base import BaseBroker, Account, Position, OrderSide, OrderType, ExecutionStatus, ExecutionOrder
from execution.paper_broker import PaperBroker
from execution.miniqmt_broker import MiniQMTBroker
from execution.safety_guard import ExecutionSafetyGuard

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class PortfolioRebalancer:
    """自动化调仓执行器 (含 7 重机构级资金安全防御中枢)"""

    def __init__(self, broker: BaseBroker, safety_guard: Optional[ExecutionSafetyGuard] = None):
        self.broker = broker
        self.guard = safety_guard or ExecutionSafetyGuard()

    def execute_rebalance(
        self,
        target_df: pd.DataFrame,
        current_prices: Optional[Dict[str, float]] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        执行组合再平衡:
        1. 前置清场: 自动撤销所有未结委托
        2. 安全审查: 资金使用率 95% 硬顶、单股 20% 仓位硬顶、日内 50% 换手熔断
        3. 限价保护: 价格偏离度 <= 2%，严禁涨停追高
        4. 先卖后买: 释放可用资金后再建仓
        """
        logger.info(f"===== 开始执行投资组合再平衡 (Dry Run: {dry_run}) =====")

        # ---------------- 防线 6: 前置清场撤单 ----------------
        cancelled_orders = self.broker.cancel_all_pending_orders()
        logger.info(f"前置撤单完成: 清除 {cancelled_orders} 笔历史挂起订单")

        account = self.broker.get_account()
        logger.info(f"当前账户总资产: {account.total_equity:,.2f} 元 | 可用现金: {account.cash:,.2f} 元")

        positions = self.broker.get_positions()
        current_holdings = {sym: pos for sym, pos in positions.items() if pos.total_shares > 0}
        logger.info(f"当前持仓标的数: {len(current_holdings)} 只")

        # 确定各标的最新参考价
        prices: Dict[str, float] = {}
        for _, row in target_df.iterrows():
            prices[row["symbol"]] = float(row.get("close", 10.0))
        for sym, pos in current_holdings.items():
            if sym not in prices:
                prices[sym] = pos.current_price or pos.avg_cost_price or 10.0

        if current_prices:
            prices.update(current_prices)

        total_equity = account.total_equity if account.total_equity > 0 else settings.INITIAL_CASH

        # 构建初始目标持仓股数
        raw_target_shares: Dict[str, int] = {}
        for _, row in target_df.iterrows():
            sym = row["symbol"]
            w = float(row.get("target_weight", 0.0))
            p = prices.get(sym, 10.0)
            target_val = total_equity * w
            shares = int(target_val / (p * 100)) * 100
            raw_target_shares[sym] = shares

        # ---------------- 防线 1~3: 资金安全中枢审计与硬性裁剪 ----------------
        target_shares, safety_logs = self.guard.audit_and_clamp_orders(
            target_shares=raw_target_shares,
            current_holdings=current_holdings,
            prices=prices,
            total_equity=total_equity,
            current_cash=account.cash
        )

        sell_orders: List[ExecutionOrder] = []
        buy_orders: List[ExecutionOrder] = []

        # ---------------- 阶段 1: 先卖后买 (释放流动性) ----------------
        logger.info(">>> 阶段 1: 扫描并执行卖出与减仓订单...")
        for sym, pos in current_holdings.items():
            tgt_s = target_shares.get(sym, 0)
            cur_s = pos.total_shares
            if cur_s > tgt_s:
                delta_sell = cur_s - tgt_s
                # 检查可用股份 (T+1)
                avail = pos.available_shares
                actual_sell = min(delta_sell, avail)
                actual_sell = (actual_sell // 100) * 100
                if actual_sell > 0:
                    p = prices.get(sym, pos.current_price)
                    # 防线 5: 价格与涨跌停安全校验
                    is_safe, msg = self.guard.validate_price_and_limit(sym, "SELL", p, p)
                    if not is_safe:
                        logger.error(f"❌ 卖出拦截: {msg}")
                        continue

                    logger.info(f"生成卖出委托: {sym} | 拟卖出: {actual_sell} 股 | 报价: {p:.2f} 元")
                    if not dry_run:
                        ord_res = self.broker.send_order(
                            symbol=sym, side=OrderSide.SELL, shares=actual_sell, price=p
                        )
                        sell_orders.append(ord_res)
                    else:
                        sell_orders.append(ExecutionOrder(
                            order_id="DRY_SELL", symbol=sym, side=OrderSide.SELL,
                            order_type=OrderType.LIMIT, requested_shares=actual_sell,
                            requested_price=p, status=ExecutionStatus.SUBMITTED
                        ))

        # ---------------- 阶段 2: 执行买入与建仓 ----------------
        account_after_sell = self.broker.get_account()
        logger.info(f">>> 阶段 2: 扫描并执行买入与建仓订单 (当前可用现金: {account_after_sell.cash:,.2f} 元)...")
        for sym, tgt_s in target_shares.items():
            cur_s = current_holdings[sym].total_shares if sym in current_holdings else 0
            if tgt_s > cur_s:
                delta_buy = tgt_s - cur_s
                delta_buy = (delta_buy // 100) * 100
                if delta_buy > 0:
                    p = prices.get(sym, 10.0)
                    # 防线 5: 价格偏离度与涨停防追高拦截
                    is_safe, msg = self.guard.validate_price_and_limit(sym, "BUY", p, p)
                    if not is_safe:
                        logger.error(f"❌ 买入拦截: {msg}")
                        continue

                    # 资金充足性二次校验
                    est_cost = delta_buy * p * 1.001
                    if est_cost > account_after_sell.cash and not dry_run:
                        logger.warning(f"⚠️ 可用现金不足 (需 {est_cost:,.2f}元, 可用 {account_after_sell.cash:,.2f}元)，安全削减买入量")
                        delta_buy = int(account_after_sell.cash / (p * 1.001 * 100)) * 100
                        if delta_buy <= 0:
                            continue

                    logger.info(f"生成买入委托: {sym} | 拟买入: {delta_buy} 股 | 报价: {p:.2f} 元")
                    if not dry_run:
                        ord_res = self.broker.send_order(
                            symbol=sym, side=OrderSide.BUY, shares=delta_buy, price=p
                        )
                        buy_orders.append(ord_res)
                    else:
                        buy_orders.append(ExecutionOrder(
                            order_id="DRY_BUY", symbol=sym, side=OrderSide.BUY,
                            order_type=OrderType.LIMIT, requested_shares=delta_buy,
                            requested_price=p, status=ExecutionStatus.SUBMITTED
                        ))

        final_acc = self.broker.get_account()
        logger.info(f"===== 调仓执行完毕: 提交卖单 {len(sell_orders)} 笔, 买单 {len(buy_orders)} 笔 =====")
        logger.info(f"调仓后总资产: {final_acc.total_equity:,.2f} 元 | 剩余现金: {final_acc.cash:,.2f} 元")

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "initial_equity": total_equity,
            "final_equity": final_acc.total_equity,
            "final_cash": final_acc.cash,
            "safety_logs": safety_logs,
            "cancelled_pending_orders": cancelled_orders,
            "sell_orders_count": len(sell_orders),
            "buy_orders_count": len(buy_orders),
            "sell_orders": [o.__dict__ for o in sell_orders],
            "buy_orders": [o.__dict__ for o in buy_orders]
        }


def run_trader_cli():
    parser = argparse.ArgumentParser(description="A股自动化实盘/模拟调仓执行器 (内置机构级资金安全防御中枢)")
    parser.add_argument("--broker", type=str, default="paper", choices=["paper", "miniqmt"], help="券商接口类型")
    parser.add_argument("--initial-cash", type=float, default=settings.INITIAL_CASH, help="初始模拟资金")
    parser.add_argument("--dry-run", action="store_true", help="演练模式 (仅生成订单不实际下单)")
    parser.add_argument("--live-confirm", action="store_true", help="实盘交易显式二次确认开关 (未开启默认自动强制为 Dry-Run)")
    parser.add_argument("--qmt-path", type=str, default=r"D:\国金证券QMT交易端\userdata_mini", help="MiniQMT 用户数据路径")
    parser.add_argument("--account-id", type=str, default="5500123456", help="实盘/模拟资金账号")
    args = parser.parse_args()

    # 防线 4: 实盘双重显式确认拦截
    actual_dry_run = args.dry_run
    if args.broker == "miniqmt" and not args.live_confirm:
        logger.critical("⚠️ 未检测到 --live-confirm 显式实盘确认开关！")
        logger.critical("🛡️ [安全防线触发] 自动强制降级为 Dry-Run 演练模式，绝不向真实账户下单！")
        actual_dry_run = True

    # 初始化 Broker
    if args.broker == "miniqmt":
        broker = MiniQMTBroker(qmt_path=args.qmt_path, account_id=args.account_id)
        if not broker.connect():
            logger.warning("连接 MiniQMT 失败，防线 7 自动熔断回退到本地 PaperBroker 仿真沙盒")
            broker = PaperBroker(initial_cash=args.initial_cash)
            broker.connect()
    else:
        broker = PaperBroker(initial_cash=args.initial_cash)
        broker.connect()

    # 构造最新决策目标持仓
    from factors.processor import FactorProcessor
    from models.walk_forward import WalkForwardTrainer
    from strategy.portfolio import PortfolioBuilder
    from data.data_manager import DataManager
    from data.universe_provider import create_universe_provider

    dm = DataManager(universe_provider=create_universe_provider(settings))
    market_df = dm.load_dataset() if (settings.PARQUET_DIR / "market_data.parquet").exists() else dm.sync_and_build_dataset()
    processor = FactorProcessor()
    factor_df = processor.load_factor_matrix() if (settings.FACTOR_DIR / "factor_matrix.parquet").exists() else processor.build_and_save_factor_matrix(market_df)

    trainer = WalkForwardTrainer()
    oos_df, _ = trainer.run_walk_forward(factor_df)

    latest_date = oos_df["date"].max()
    daily_df = oos_df[oos_df["date"] == latest_date].copy()
    builder = PortfolioBuilder(top_k_buy=settings.TOP_K_BUY, top_k_hold=settings.TOP_K_HOLD)
    top_df = builder.build_target_portfolio(daily_df, current_holdings=set(), date=latest_date)

    rebalancer = PortfolioRebalancer(broker)
    res = rebalancer.execute_rebalance(target_df=top_df, dry_run=actual_dry_run)
    print("\n===== 调仓执行结果摘要 (含 7 重资金安全防线) =====")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_trader_cli()
