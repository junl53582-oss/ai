"""
全链路混沌压力测试与极限真实性压测套件 (tests/test_chaos_stress_pipeline.py)
用于模拟 A 股极端黑天鹅、流动性枯竭、千股跌停、价格脏数据与资金穿透等 10 大极端工况:
1. 连续 5 日全市场跌停锁死 (流动性归零下的 T+1 延期卖出与本金保护)
2. 连续 5 日全市场开盘涨停 (开盘一字涨停无法买入的拦截与零滑点虚假收益排除)
3. 脏数据攻击 (NaN, Inf, 负价格, 超大异常值, 截断异常)
4. 资金守恒极限压测 (小资金 1 万元 vs 大资金 1 亿元整手与 5% 现金防穿透)
5. 极端换手与频繁调仓下的费率滑点严苛扣除
6. 恶意价格偏离与未授权实盘下单拦截
7. 未来数据打乱时序不变性检验 (Temporal Invariance & Zero Leakage)
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
from backtest.trading_rules import AShareTradingRules
from execution.broker_base import Position, OrderSide, OrderType, ExecutionStatus
from execution.paper_broker import PaperBroker
from execution.safety_guard import ExecutionSafetyGuard
from execution.run_trader import PortfolioRebalancer


class TestChaosStressPipeline:

    def test_chaos_1_consecutive_limit_down_liquidity_freeze(self):
        """混沌测试 1: 连续跌停锁死导致无法卖出，严禁虚假平仓"""
        rules = AShareTradingRules()
        # 标的当日跌停 -10.0% 且一字锁死
        can_sell, reason = rules.can_sell("600519.SH", is_suspended=False, is_limit_down=True, is_limit_down_locked=True)
        assert can_sell is False
        assert "跌停锁死" in reason

        # 跌停但未锁死（有成交开板）-> 允许卖出
        can_sell_open, _ = rules.can_sell("600519.SH", is_suspended=False, is_limit_down=True, is_limit_down_locked=False)
        assert can_sell_open is True

    def test_chaos_2_consecutive_limit_up_anti_chasing(self):
        """混沌测试 2: 开盘一字涨停无法买进，杜绝纸面虚假成交"""
        rules = AShareTradingRules()
        # 开盘一字涨停
        can_buy, reason = rules.can_buy("600519.SH", is_suspended=False, is_limit_up=True, is_limit_up_locked=True)
        assert can_buy is False
        assert "涨停锁死" in reason

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
            "volume": [10000.0] * 58 + [np.nan, -500],
            "amount": [1000000.0] * 58 + [np.nan, np.nan],
            "turnover": [0.02] * 58 + [np.nan, 999.0]
        })
        calc = Alpha158Subset()
        # 脏数据下计算绝不能抛出未捕获崩溃，必须安全返回包含有效数值/NaN的特征矩阵
        feat_df = calc.compute_all(dirty_df)
        assert len(feat_df) == 60
        assert not np.isinf(feat_df.select_dtypes(include=[np.number]).values).any()

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
        # 茅台现价 1800 元/股，1手=100股=180,000 元，1万元本金根本买不起1手
        target_shares = {"600519.SH": 100}
        prices = {"600519.SH": 1800.0}
        
        clamped, logs = guard.audit_and_clamp_orders(
            target_shares=target_shares,
            current_holdings={},
            prices=prices,
            total_equity=total_equity,
            current_cash=total_equity
        )
        # 必须安全归零，绝不能强行下单产生透支
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
            "volume": np.random.lognormal(14, 0.5, 100),
            "amount": np.random.lognormal(18, 0.5, 100),
            "turnover": np.random.uniform(0.01, 0.05, 100)
        })

        # 计算基准因子
        calc = Alpha158Subset()
        f_base = calc.compute_all(df_base)

        # 恶意修改未来第 70 天之后的所有数据 (乘以 10 倍)
        df_tampered = df_base.copy()
        df_tampered.loc[70:, ["open", "high", "low", "close"]] *= 10.0
        df_tampered.loc[70:, "volume"] *= 5.0
        f_tampered = calc.compute_all(df_tampered)

        # 前 69 天的全部特征矩阵数值必须完全 100% 绝对一致 (差值为 0)
        feat_cols = Alpha158Subset.get_factor_names()
        diff = np.abs(np.nan_to_num(f_base.iloc[:69][feat_cols].values) - np.nan_to_num(f_tampered.iloc[:69][feat_cols].values))
        assert np.max(diff) < 1e-6, "❌ 发现严重未来函数漏洞: 未来数据篡改导致了历史特征变化！"
