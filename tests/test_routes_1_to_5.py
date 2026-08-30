"""
路线一至路线五端到端集成测试套件 (tests/test_routes_1_to_5.py)
"""
import pytest
import pandas as pd
import numpy as np

from config.universe_profiles import UniverseProfileManager
from execution.broker_base import OrderSide, OrderType, ExecutionStatus
from execution.paper_broker import PaperBroker
from execution.run_trader import PortfolioRebalancer
from scheduler.notifier import MessageNotifier
from strategy.optimizer import RiskParityOptimizer, ConstrainedQPOptimizer


class TestRoutes1To5Integration:

    def test_route_5_universe_profiles(self):
        """路线五: 股票池 Profile 管理测试"""
        profiles = UniverseProfileManager.list_profiles()
        assert "HS300_CORE" in profiles
        assert "ZZ500_GROWTH" in profiles
        assert "TECH_INNOVATION" in profiles
        assert "HIGH_DIVIDEND" in profiles

        symbols = UniverseProfileManager.get_symbols("TECH_INNOVATION")
        assert len(symbols) > 5
        assert "688981.SH" in symbols or "300750.SZ" in symbols

    def test_route_4_rebalancer_end_to_end(self):
        """路线四: 自动化调仓先卖后买与整手交易测试"""
        broker = PaperBroker(initial_cash=1_000_000.0)
        broker.connect()

        # 初始买入 1000 股 600519.SH 并跨日解锁
        broker.send_order("600519.SH", side=OrderSide.BUY, shares=1000, price=100.0)
        broker.unlock_t1_shares()

        # 新目标: 调出 600519.SH (0权重), 调入 300750.SZ (50%权重)
        target_df = pd.DataFrame({
            "symbol": ["300750.SZ"],
            "target_weight": [0.50],
            "close": [200.0]
        })

        rebalancer = PortfolioRebalancer(broker)
        res = rebalancer.execute_rebalance(target_df=target_df, dry_run=False)

        assert res["sell_orders_count"] == 1
        assert res["buy_orders_count"] == 1
        assert "600519.SH" not in broker.get_positions()
        assert "300750.SZ" in broker.get_positions()
        # 验证 300750.SZ 买入股数是 100 股整数倍
        assert broker.get_positions()["300750.SZ"].total_shares % 100 == 0

    def test_route_3_notification_formatting(self):
        """路线三: 自动化调度决策推送卡片测试"""
        top_df = pd.DataFrame({
            "symbol": ["600519.SH", "300750.SZ"],
            "name": ["贵州茅台", "宁德时代"],
            "industry": ["食品饮料", "电力设备"],
            "pred_score": [0.82, 0.76],
            "target_weight": [0.20, 0.15],
            "close": [1600.0, 190.0]
        })
        md = MessageNotifier.format_daily_report_markdown("2026-08-28", "2026-08-31", top_df)
        assert "【A股量化系统 · 每日交易决策报告】" in md
        assert "2026-08-28" in md
        assert "贵州茅台" in md

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_verification_e2e(self):
        """路线一至五全链路端到端综合测试"""
        from scripts.verify_full_pipeline_e2e import verify_all_routes
        verify_all_routes()
