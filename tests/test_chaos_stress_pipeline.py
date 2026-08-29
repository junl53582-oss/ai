"""
全链路混沌压力测试与极限真实性压测套件 (tests/test_chaos_stress_pipeline.py)
"""
import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import settings
from data.security_master import SecurityMaster
from factors.alpha158 import Alpha158Subset
from factors.processor import FactorProcessor
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from backtest.engine import BacktestEngine
from strategy.trading_rules import AShareTradingRules, PositionRecord, RejectReason
from execution.broker_base import Position, OrderSide, OrderType, ExecutionStatus
from execution.paper_broker import PaperBroker
from execution.safety_guard import ExecutionSafetyGuard
from execution.run_trader import PortfolioRebalancer


class TestChaosStressPipeline:

    def test_chaos_1_consecutive_limit_down_liquidity_freeze(self):
        """混沌测试 1: 连续跌停锁死导致无法卖出，严禁虚假平仓"""
        rules = AShareTradingRules()
        pos = PositionRecord(
            symbol="600519.SH",
            shares=100,
            available_shares=100,
            avg_cost=100.0,
            last_price=100.0,
            buy_date="2023-01-01",
            highest_price=100.0
        )
        # 标的当日跌停 -10.0% 且一字锁死
        row_locked = pd.Series({
            "is_suspended": False,
            "is_limit_down_locked": True,
            "volume": 1000.0,
            "limit_down_price": 90.0,
            "open": 90.0,
            "high": 90.0,
            "low": 90.0,
            "close": 90.0
        })
        can_sell, reason = rules.can_sell(row_locked, pos, execution_price=90.0)
        assert can_sell is False
        assert reason == RejectReason.LIMIT_DOWN.value

        # 跌停但未锁死（有成交开板）-> 允许卖出
        row_open = pd.Series({
            "is_suspended": False,
            "is_limit_down_locked": False,
            "volume": 1000.0,
            "limit_down_price": 90.0,
            "open": 92.0,
            "high": 95.0,
            "low": 90.0,
            "close": 91.0
        })
        can_sell_open, _ = rules.can_sell(row_open, pos, execution_price=92.0)
        assert can_sell_open is True

    def test_chaos_2_consecutive_limit_up_anti_chasing(self):
        """混沌测试 2: 开盘一字涨停无法买进，杜绝纸面虚假成交"""
        rules = AShareTradingRules()
        # 开盘一字涨停
        row_locked = pd.Series({
            "is_suspended": False,
            "is_limit_up_locked": True,
            "volume": 1000.0,
            "limit_up_price": 110.0,
            "open": 110.0,
            "high": 110.0,
            "low": 110.0,
            "close": 110.0
        })
        can_buy, reason = rules.can_buy(row_locked, execution_price=110.0)
        assert can_buy is False
        assert reason == RejectReason.LIMIT_UP.value

    def test_chaos_3_dirty_data_resilience(self):
        """混沌测试 3: 因子计算抗脏数据攻击 (NaN, Inf, 极端噪点)"""
        dates = pd.date_range("2023-01-01", periods=60, freq="B")
        dirty_df = pd.DataFrame({
            "date": dates,
            "symbol": "600519.SH",
            "open": [100.0] * 58 + [np.nan, np.inf],
            "high": [105.0] * 58 + [np.nan, 1e9],
            "low": [95.0] * 58 + [-999.0, np.nan],
            "close": [100.0] * 58 + [0.0, np.nan],
            "adj_open": [100.0] * 58 + [np.nan, np.inf],
            "adj_high": [105.0] * 58 + [np.nan, 1e9],
            "adj_low": [95.0] * 58 + [-999.0, np.nan],
            "adj_close": [100.0] * 58 + [0.0, np.nan],
            "pct_change": [0.0] * 60,
            "adj_pct_change": [0.0] * 60,
            "volume": [10000.0] * 58 + [np.nan, -500],
            "amount": [1000000.0] * 58 + [np.nan, np.nan],
            "turnover": [0.02] * 58 + [np.nan, 999.0]
        })
        calc = Alpha158Subset()
        feat_df = calc.compute_all(dirty_df)
        assert len(feat_df) == 60
        factor_cols = [c for c in Alpha158Subset.get_factor_names() if c in feat_df.columns]
        assert not np.isinf(feat_df[factor_cols].values).any()

    def test_chaos_4_capital_conservation_law(self):
        """混沌测试 4: 资金守恒定律严格验证 (现金 + 市值 == 总资产，绝无凭空增减)"""
        broker = PaperBroker(initial_cash=100_000.0)
        # 买入 200 股 @ 100 元
        order1 = broker.send_order("600519.SH", OrderSide.BUY, 200, 100.0)
        assert order1.status == ExecutionStatus.FILLED
        acc1 = broker.get_account()
        # 考虑手续费
        total1 = acc1.cash + acc1.market_value
        assert np.isclose(total1 + acc1.positions["600519.SH"].avg_cost_price * 200 - 20000.0, 100_000.0, atol=10.0)
        assert acc1.cash >= 0.0

    def test_chaos_5_small_capital_boundary_10k(self):
        """混沌测试 5: 1 万元极小本金整手 100 股边界与现金不足防御"""
        guard = ExecutionSafetyGuard(max_capital_utilization=0.95, max_single_stock_exposure=0.50)
        total_equity = 10_000.0
        target_shares = {"600519.SH": 100}
        prices = {"600519.SH": 1800.0}
        
        clamped, logs = guard.audit_and_clamp_orders(
            target_shares=target_shares,
            current_holdings={},
            prices=prices,
            total_equity=total_equity,
            current_cash=total_equity
        )
        assert clamped["600519.SH"] == 0

    def test_chaos_6_temporal_permutation_invariance(self):
        """混沌测试 6: 未来时间序列随机置乱，历史 T 日因子输出 100% 确定性不变"""
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        np.random.seed(42)
        rets = np.random.normal(0.0005, 0.02, 100)
        prices = 100.0 * np.exp(np.cumsum(rets))

        df_base = pd.DataFrame({
            "date": dates,
            "symbol": "600519.SH",
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "adj_open": prices * 0.99,
            "adj_high": prices * 1.01,
            "adj_low": prices * 0.98,
            "adj_close": prices,
            "pct_change": pd.Series(prices).pct_change().fillna(0.0),
            "adj_pct_change": pd.Series(prices).pct_change().fillna(0.0),
            "volume": np.random.lognormal(14, 0.5, 100),
            "amount": np.random.lognormal(18, 0.5, 100),
            "turnover": np.random.uniform(0.01, 0.05, 100)
        })

        calc = Alpha158Subset()
        f_base = calc.compute_all(df_base)

        df_tampered = df_base.copy()
        df_tampered.loc[70:, ["open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close"]] *= 10.0
        df_tampered.loc[70:, "volume"] *= 5.0
        f_tampered = calc.compute_all(df_tampered)

        feat_cols = Alpha158Subset.get_factor_names()
        diff = np.abs(np.nan_to_num(f_base.iloc[:69][feat_cols].values) - np.nan_to_num(f_tampered.iloc[:69][feat_cols].values))
        assert np.max(diff) < 1e-6, "❌ 发现严重未来函数漏洞: 未来数据篡改导致了历史特征变化！"
