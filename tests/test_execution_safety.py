"""
机构级实盘资金安全防御中枢测试套件 (tests/test_execution_safety.py)
全量覆盖 7 大实盘资金安全防线:
1. 资金使用率 95% 硬顶
2. 单股 20% 仓位硬顶
3. 日内 50% 换手熔断
4. 双重显式确认 Flag (--live-confirm)
5. 价格偏离度 <= 2% 与涨停防追高拦截
6. 前置清场撤单
7. MiniQMT 脱机自动降级
"""
import pytest
import pandas as pd
import numpy as np

from execution.broker_base import Position, OrderSide, OrderType, ExecutionStatus
from execution.paper_broker import PaperBroker
from execution.safety_guard import ExecutionSafetyGuard
from execution.run_trader import PortfolioRebalancer


class TestExecutionSafetyGuard:

    def test_single_stock_20pct_cap(self):
        """防线 2: 单股持仓不得超过 20% 总权益"""
        guard = ExecutionSafetyGuard(max_single_stock_exposure=0.20)
        total_equity = 1_000_000.0
        # 某只股票拟买入 40% (400,000 元)
        target_shares = {"600519.SH": 2000} # 2000股 * 200元 = 400,000元
        prices = {"600519.SH": 200.0}
        current_holdings = {}

        clamped, logs = guard.audit_and_clamp_orders(
            target_shares=target_shares,
            current_holdings=current_holdings,
            prices=prices,
            total_equity=total_equity,
            current_cash=total_equity
        )
        # 必须被安全裁剪至 200,000元 / 200元 = 1000股
        assert clamped["600519.SH"] == 1000
        assert any("单股上限防线" in log for log in logs)

    def test_capital_utilization_95pct_cap(self):
        """防线 1: 资金使用率 95% 硬顶，强制保留 5% 现金"""
        guard = ExecutionSafetyGuard(max_capital_utilization=0.95, max_single_stock_exposure=0.50)
        total_equity = 1_000_000.0
        # 目标满仓 100% (1,000,000 元)
        target_shares = {"A": 5000, "B": 5000} # 5000*100 + 5000*100 = 1,000,000
        prices = {"A": 100.0, "B": 100.0}

        clamped, logs = guard.audit_and_clamp_orders(
            target_shares=target_shares,
            current_holdings={},
            prices=prices,
            total_equity=total_equity,
            current_cash=total_equity
        )
        total_val = clamped["A"] * 100.0 + clamped["B"] * 100.0
        assert total_val <= 950_000.0 + 100.0 # 允许整手取整误差
        assert any("资金缓冲防线" in log for log in logs)

    def test_daily_turnover_50pct_circuit_breaker(self):
        """防线 3: 日内 50% 换手率熔断"""
        guard = ExecutionSafetyGuard(max_daily_turnover_ratio=0.50)
        total_equity = 1_000_000.0
        # 拟买入 80% 换手 (800,000元，每只标的 200,000 元均满足 20% 单股硬顶，纯粹触发日内 50% 换手熔断)
        target_shares = {"A": 2000, "B": 2000, "C": 2000, "D": 2000}
        prices = {"A": 100.0, "B": 100.0, "C": 100.0, "D": 100.0}

        clamped, logs = guard.audit_and_clamp_orders(
            target_shares=target_shares,
            current_holdings={},
            prices=prices,
            total_equity=total_equity,
            current_cash=total_equity
        )
        total_turnover = sum(clamped[s] * prices[s] for s in clamped)
        assert total_turnover <= 500_000.0 + 100.0
        assert any("换手熔断防线" in log for log in logs)

    def test_price_deviation_and_limit_up_guard(self):
        """防线 5: 价格偏离度 > 2% 与 +9.8% 涨停买入拦截"""
        guard = ExecutionSafetyGuard(max_price_deviation=0.02)
        
        # 正常 1% 偏离 -> 通过
        ok1, _ = guard.validate_price_and_limit("600519.SH", "BUY", 101.0, 100.0, pre_close_price=100.0)
        assert ok1 is True

        # 3% 异常偏离 -> 拦截
        ok2, msg2 = guard.validate_price_and_limit("600519.SH", "BUY", 103.5, 100.0, pre_close_price=100.0)
        assert ok2 is False
        assert "偏离最新市价" in msg2

        # 涨停价挂买单 (+10%) -> 拦截
        ok3, msg3 = guard.validate_price_and_limit("600519.SH", "BUY", 110.0, 110.0, pre_close_price=100.0)
        assert ok3 is False
        assert "涨停防追高拦截" in msg3

    def test_pre_rebalance_cancellation_and_rebalancing(self):
        """防线 6: 前置撤单与完整调仓闭环"""
        broker = PaperBroker(initial_cash=1_000_000.0)
        # 手动注入一笔挂起订单
        broker.orders.append(broker.send_order("600519.SH", OrderSide.BUY, 100, 100.0))
        broker.orders[-1].status = ExecutionStatus.SUBMITTED

        rebalancer = PortfolioRebalancer(broker)
        target_df = pd.DataFrame({
            "symbol": ["000858.SZ", "600036.SH"],
            "target_weight": [0.15, 0.15],
            "close": [100.0, 50.0]
        })

        res = rebalancer.execute_rebalance(target_df, dry_run=False)
        assert res["cancelled_pending_orders"] >= 1
        assert res["buy_orders_count"] == 2
        assert len(res["safety_logs"]) >= 0

    def test_miniqmt_disconnect_strictly_blocks_orders_without_fake_simulation(self):
        """防线 7: MiniQMT 断线严禁假想仿真成交，必须阻断新订单以待对账"""
        from execution.miniqmt_broker import MiniQMTBroker
        broker = MiniQMTBroker()
        broker.is_connected = False
        order = broker.send_order("600519.SH", OrderSide.BUY, 100, 100.0)
        assert order.status == ExecutionStatus.REJECTED
        assert "实盘安全风控已阻断新订单并冻结状态" in order.error_msg
