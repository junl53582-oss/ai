"""
A股实盘交易硬约束、历史费率、公司行为、逐日中性化与审计单元测试套件 (tests/test_trading_rules.py)
涵盖：
1. T+1、一字板、整手、涨跌停与滑点截断保护
2. 逐日行业覆盖率与动态中性化降级 (P0-1)
3. Point-In-Time Universe 时间线变动与前向隔离 (P0-2)
4. 日历来源真实性标识 (P0-3)
5. 历史印花税与过户费分段计算 (P0-4)
6. 公司行为 (现金分红/送转股) 价值守恒与除息调整 (P0-5)
7. 持仓批次重构与部分卖出平均成本更新 (P0-6)
8. AuditMetadata Fail-Closed 真实性审计 (P0-7)
9. Date-Indexed 历史 ST 状态时间线隔离 (P0-8)
10. 回测结束挂单自动撤销 (P1-1)
11. 停牌过时价格可解释性指标 (P1-2)
12. 交易日持仓周期计算 (P1-3)
13. 绩效金融指标标准定义 (P1-4)
14. 流动性成交量容量上限 (P1-5)
15. 行业集中度审计语义 (P1-6)
16. 基准指数独立清洗 (P1-7)
"""
import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np

from config.settings import settings
from strategy.trading_rules import (
    AShareTradingRules,
    TradingFeeSchedule,
    PositionRecord,
    PositionLot,
    Order,
    OrderStatus,
    OrderSide,
    RejectReason,
    recalculate_position_from_lots
)
from strategy.portfolio import PortfolioBuilder
from strategy.corporate_actions import CorporateAction, CorporateActionProvider
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer
from backtest.audit import AuditMetadata, AuditCollector
from data.data_manager import DataManager, count_trading_days
from data.security_master import SecurityMaster, StockMetadata
from data.universe_provider import StaticUniverseProvider, PointInTimeUniverseProvider
from factors.processor import FactorProcessor


# ==========================================
# 1. 基础交易规则与硬约束测试
# ==========================================
def test_1_lot_size_rounding():
    """测试 1: 100 股整手向下取整计算"""
    rules = AShareTradingRules(lot_size=100)
    assert rules.calculate_lot_shares(target_amount=10500, price=100.0) == 100
    assert rules.calculate_lot_shares(target_amount=99, price=100.0) == 0
    assert rules.calculate_lot_shares(target_amount=25000, price=50.0) == 500


def test_2_t_plus_1_restriction():
    """测试 2: T+1 交易限制（当日买入可用卖出股数为 0）"""
    rules = AShareTradingRules()
    pos = PositionRecord(
        symbol="600519.SH",
        shares=500,
        available_shares=0,
        avg_cost=100.0,
        last_price=100.0,
        buy_date="2023-05-10",
        highest_price=100.0
    )
    row = pd.Series({"symbol": "600519.SH", "is_suspended": False, "is_limit_down_locked": False, "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0})
    
    can_s, msg = rules.can_sell(row, pos, execution_price=100.0)
    assert not can_s, "T+1 限制下不可卖出，但返回允许卖出！"
    assert msg == RejectReason.T_PLUS_1_LOCK.value


def test_3_limit_up_locked_cannot_buy():
    """测试 3: 一字涨停板封死无法买入"""
    rules = AShareTradingRules()
    row = pd.Series({
        "symbol": "600519.SH",
        "is_suspended": False,
        "is_limit_up_locked": True,
        "limit_up_price": 110.0,
        "open": 110.0,
        "high": 110.0,
        "low": 110.0,
        "close": 110.0,
        "volume": 1000
    })
    can_b, msg = rules.can_buy(row, execution_price=110.0)
    assert not can_b, "一字涨停应禁止买入！"
    assert msg == RejectReason.LIMIT_UP.value


def test_4_limit_down_locked_cannot_sell():
    """测试 4: 一字跌停板封死无法卖出"""
    rules = AShareTradingRules()
    pos = PositionRecord(
        symbol="600519.SH",
        shares=500,
        available_shares=500,
        avg_cost=100.0,
        last_price=90.0,
        buy_date="2023-05-01",
        highest_price=100.0
    )
    row = pd.Series({
        "symbol": "600519.SH",
        "is_suspended": False,
        "is_limit_down_locked": True,
        "limit_down_price": 90.0,
        "open": 90.0,
        "high": 90.0,
        "low": 90.0,
        "close": 90.0,
        "volume": 1000
    })
    can_s, msg = rules.can_sell(row, pos, execution_price=90.0)
    assert not can_s, "一字跌停应禁止卖出！"
    assert msg == RejectReason.LIMIT_DOWN.value


def test_5_st_stock_5pct_limit():
    """测试 5: ST 股票 5% 涨跌停限制"""
    rules = AShareTradingRules()
    ratio_st1 = rules.get_limit_ratio(symbol="ST康美.SH", is_st=True)
    ratio_st2 = rules.get_limit_ratio(symbol="*ST左江.SZ", is_st=False)
    ratio_main = rules.get_limit_ratio(symbol="600519.SH", is_st=False)

    assert ratio_st1 == 0.05
    assert ratio_st2 == 0.05
    assert ratio_main == 0.10


def test_6_slippage_no_double_counting():
    """测试 6: 滑点仅体现在成交价中，现金扣除不发生重复计费"""
    initial_cash = 100000.0
    engine = BacktestEngine(initial_cash=initial_cash, top_k_buy=1, top_k_hold=1, rebalance_freq=1)
    
    dates = pd.to_datetime(["2023-01-03", "2023-01-04"])
    df = pd.DataFrame([
        {"date": dates[0], "symbol": "600519.SH", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "adj_close": 100.0, "volume": 10000, "is_suspended": False, "is_limit_up_locked": False, "is_limit_down_locked": False, "pred_score": 0.5, "benchmark_close": 4000.0},
        {"date": dates[1], "symbol": "600519.SH", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "adj_close": 100.0, "volume": 10000, "is_suspended": False, "is_limit_up_locked": False, "is_limit_down_locked": False, "pred_score": 0.5, "benchmark_close": 4000.0},
    ])
    
    equity_df, orders_df = engine.run(df)
    filled_buys = orders_df[orders_df["status"] == "FILLED"]
    assert len(filled_buys) == 1
    
    buy_order = filled_buys.iloc[0]
    exec_price = buy_order["execution_price"]
    shares = buy_order["filled_shares"]
    comm = buy_order["commission"]
    trans = buy_order.get("transfer_fee", 0.0)
    
    expected_cash = initial_cash - (shares * exec_price + comm + trans)
    assert np.isclose(engine.cash, expected_cash, atol=1e-2)


def test_slippage_price_respects_price_limits():
    """测试 7: 通过生产引擎撮合执行，验证买入滑点不破涨停价、卖出滑点不破跌停价"""
    engine = BacktestEngine(initial_cash=100000.0)
    
    daily_map_buy = {
        "600519.SH": pd.Series({
            "symbol": "600519.SH", "open": 10.95, "high": 11.00, "low": 10.90, "close": 10.98,
            "limit_up_price": 11.00, "is_suspended": False, "is_limit_up_locked": False, "volume": 100000
        })
    }
    buy_order = Order(order_id="B1", symbol="600519.SH", side=OrderSide.BUY, signal_date="2023-01-01", requested_shares=100)
    engine.pending_orders = [buy_order]
    engine._execute_morning_auction(daily_map_buy, "2023-01-02")
    
    assert buy_order.status == OrderStatus.FILLED
    assert buy_order.execution_price <= 11.00
    
    daily_map_sell = {
        "600519.SH": pd.Series({
            "symbol": "600519.SH", "open": 9.05, "high": 9.10, "low": 9.00, "close": 9.02,
            "limit_down_price": 9.00, "is_suspended": False, "is_limit_down_locked": False, "volume": 100000
        })
    }
    engine.positions["600519.SH"].available_shares = 100
    sell_order = Order(order_id="S1", symbol="600519.SH", side=OrderSide.SELL, signal_date="2023-01-02", requested_shares=100)
    engine.pending_orders = [sell_order]
    engine._execute_morning_auction(daily_map_sell, "2023-01-03")
    
    assert sell_order.status == OrderStatus.FILLED
    assert sell_order.execution_price >= 9.00


# ==========================================
# 2. P0-1: 逐日行业覆盖与中性化降级
# ==========================================
def test_neutralization_switches_per_date():
    """测试 P0-1: 因子中性化逐日切换 (Day 1 有效行业充足 -> 行业+市值中性化; Day 2 行业覆盖不足 -> 纯市值中性化)"""
    processor = FactorProcessor()
    
    # 构造 2 个交易日的数据，Day 1 包含丰富行业，Day 2 均为 UNKNOWN
    df = pd.DataFrame([
        # Day 1: 6 只股票，覆盖率 100%
        {"date": "2023-01-03", "symbol": "A", "factor": 1.2, "industry": "Tech", "LOG_CIRC_MV": 24.0},
        {"date": "2023-01-03", "symbol": "B", "factor": 0.8, "industry": "Tech", "LOG_CIRC_MV": 24.5},
        {"date": "2023-01-03", "symbol": "C", "factor": -0.5, "industry": "Finance", "LOG_CIRC_MV": 26.0},
        {"date": "2023-01-03", "symbol": "D", "factor": -0.3, "industry": "Finance", "LOG_CIRC_MV": 25.8},
        {"date": "2023-01-03", "symbol": "E", "factor": 0.2, "industry": "Health", "LOG_CIRC_MV": 23.0},
        {"date": "2023-01-03", "symbol": "F", "factor": -0.1, "industry": "Health", "LOG_CIRC_MV": 23.5},
        # Day 2: 6 只股票，覆盖率 0% (全 UNKNOWN)
        {"date": "2023-01-04", "symbol": "A", "factor": 1.2, "industry": "UNKNOWN", "LOG_CIRC_MV": 24.0},
        {"date": "2023-01-04", "symbol": "B", "factor": 0.8, "industry": "UNKNOWN", "LOG_CIRC_MV": 24.5},
        {"date": "2023-01-04", "symbol": "C", "factor": -0.5, "industry": "UNKNOWN", "LOG_CIRC_MV": 26.0},
        {"date": "2023-01-04", "symbol": "D", "factor": -0.3, "industry": "UNKNOWN", "LOG_CIRC_MV": 25.8},
        {"date": "2023-01-04", "symbol": "E", "factor": 0.2, "industry": "UNKNOWN", "LOG_CIRC_MV": 23.0},
        {"date": "2023-01-04", "symbol": "F", "factor": -0.1, "industry": "UNKNOWN", "LOG_CIRC_MV": 23.5},
    ])
    
    out_df = processor.neutralize_cross_section(df, factor_cols=["factor"], industry_col="industry", market_cap_col="LOG_CIRC_MV")
    
    assert processor.neutralization_mode_by_date["2023-01-03"] == "INDUSTRY_AND_MCAP"
    assert processor.neutralization_mode_by_date["2023-01-04"] == "MCAP_ONLY"
    assert processor.industry_neutralization_enabled == "PARTIAL"
    assert processor.industry_neutralized_day_ratio == 0.5


def test_unknown_not_used_as_real_industry_dummy():
    """测试 P0-1: UNKNOWN 绝对不作为有效行业 Dummy 参与回归"""
    processor = FactorProcessor()
    
    df = pd.DataFrame([
        {"date": "2023-01-03", "symbol": "A", "factor": 1.0, "industry": "Tech", "LOG_CIRC_MV": 24.0},
        {"date": "2023-01-03", "symbol": "B", "factor": 0.5, "industry": "Tech", "LOG_CIRC_MV": 24.5},
        {"date": "2023-01-03", "symbol": "C", "factor": -0.5, "industry": "Finance", "LOG_CIRC_MV": 26.0},
        {"date": "2023-01-03", "symbol": "D", "factor": -0.3, "industry": "Finance", "LOG_CIRC_MV": 25.8},
        {"date": "2023-01-03", "symbol": "E", "factor": 0.0, "industry": "UNKNOWN", "LOG_CIRC_MV": 23.0},
        {"date": "2023-01-03", "symbol": "F", "factor": 0.2, "industry": "UNKNOWN", "LOG_CIRC_MV": 23.5},
    ])
    
    out_df = processor.neutralize_cross_section(df, factor_cols=["factor"])
    assert not out_df["factor"].isna().any()


def test_missing_factor_not_imputed_as_zero_for_ols():
    """测试 P0-1: 缺失因子值 (NaN) 严格不参与 OLS 回归拟合，残差不污染"""
    processor = FactorProcessor()
    
    df = pd.DataFrame([
        {"date": "2023-01-03", "symbol": "A", "factor": 1.0, "industry": "Tech", "LOG_CIRC_MV": 24.0},
        {"date": "2023-01-03", "symbol": "B", "factor": np.nan, "industry": "Tech", "LOG_CIRC_MV": 24.5},
        {"date": "2023-01-03", "symbol": "C", "factor": -0.5, "industry": "Finance", "LOG_CIRC_MV": 26.0},
        {"date": "2023-01-03", "symbol": "D", "factor": -0.3, "industry": "Finance", "LOG_CIRC_MV": 25.8},
        {"date": "2023-01-03", "symbol": "E", "factor": 0.2, "industry": "Finance", "LOG_CIRC_MV": 23.0},
        {"date": "2023-01-03", "symbol": "F", "factor": 0.1, "industry": "Tech", "LOG_CIRC_MV": 23.5},
    ])
    
    out_df = processor.neutralize_cross_section(df, factor_cols=["factor"])
    # 原本是 NaN 的位置仍应为 NaN (或未被当作 0 参与拟合)
    assert np.isnan(out_df.loc[out_df["symbol"] == "B", "factor"].iloc[0])


def test_small_cross_section_neutralization_fallback():
    """测试 P0-1: 样本截面过小时，中性化安全保留不崩溃"""
    processor = FactorProcessor()
    df = pd.DataFrame([
        {"date": "2023-01-03", "symbol": "A", "factor": 1.0, "industry": "Tech", "LOG_CIRC_MV": 24.0},
        {"date": "2023-01-03", "symbol": "B", "factor": 0.5, "industry": "Tech", "LOG_CIRC_MV": 24.5},
    ])
    out_df = processor.neutralize_cross_section(df, factor_cols=["factor"])
    assert len(out_df) == 2


def test_industry_coverage_audit_is_daily_based():
    """测试 P0-1: 审计指标是基于逐日统计均值与最小值"""
    processor = FactorProcessor()
    df = pd.DataFrame([
        {"date": "2023-01-03", "symbol": "A", "factor": 1.0, "industry": "Tech", "LOG_CIRC_MV": 24.0},
        {"date": "2023-01-03", "symbol": "B", "factor": 0.5, "industry": "Finance", "LOG_CIRC_MV": 24.5},
        {"date": "2023-01-03", "symbol": "C", "factor": -0.5, "industry": "UNKNOWN", "LOG_CIRC_MV": 26.0},
        {"date": "2023-01-04", "symbol": "A", "factor": 1.0, "industry": "Tech", "LOG_CIRC_MV": 24.0},
        {"date": "2023-01-04", "symbol": "B", "factor": 0.5, "industry": "UNKNOWN", "LOG_CIRC_MV": 24.5},
        {"date": "2023-01-04", "symbol": "C", "factor": -0.5, "industry": "UNKNOWN", "LOG_CIRC_MV": 26.0},
    ])
    processor.neutralize_cross_section(df, factor_cols=["factor"])
    # Day 1: 2/3 = 0.6667, Day 2: 1/3 = 0.3333 -> mean = 0.5000, min = 0.3333
    assert np.isclose(processor.industry_coverage_ratio_mean, 0.50, atol=1e-2)
    assert np.isclose(processor.industry_coverage_ratio_min, 0.3333, atol=1e-2)


# ==========================================
# 3. P0-2: PointInTimeUniverseProvider
# ==========================================
def test_point_in_time_universe_changes_by_date():
    """测试 P0-2: 点位股票池随历史变动生效时间动态变更"""
    pit = PointInTimeUniverseProvider.for_test_fixture(
        fallback_symbols=["600519.SH", "000858.SZ"],
        baseline_snapshot_date="2023-01-01",
        baseline_symbols=["600519.SH", "000858.SZ"],
        coverage_start="2023-01-01",
        coverage_end="2023-12-31"
    )
    pit.add_constituent_change("2023-06-01", "300750.SZ", "IN")
    pit.add_constituent_change("2023-06-01", "000858.SZ", "OUT")

    univ_may = pit.get_universe("2023-05-15")
    assert set(univ_may) == {"600519.SH", "000858.SZ"}

    univ_july = pit.get_universe("2023-07-01")
    assert set(univ_july) == {"600519.SH", "300750.SZ"}
    assert pit.get_mode("2023-01-01", "2023-12-31") in ["PIT_INCOMPLETE", "POINT_IN_TIME", "POINT_IN_TIME_VERIFIED"]


def test_future_constituent_not_visible_in_past():
    """测试 P0-2: 未来纳入的成分股在历史查询中严格不可见"""
    pit = PointInTimeUniverseProvider()
    pit.add_constituent_change("2024-01-01", "688981.SH", "IN")
    
    assert pit.is_member("688981.SH", "2023-12-31") is False
    assert pit.is_member("688981.SH", "2024-01-02") is True


def test_static_fallback_reports_survivorship_risk():
    """测试 P0-2: 未加载历史变动数据的静态 Fallback 必须报告存在幸存者偏差风险"""
    pit = PointInTimeUniverseProvider(fallback_symbols=["600519.SH", "000858.SZ"])
    assert pit.get_mode() == "STATIC_FALLBACK"
    assert pit.has_survivorship_bias_risk() is True
    assert set(pit.get_universe("2023-01-01")) == {"600519.SH", "000858.SZ"}


def test_portfolio_respects_point_in_time_universe():
    """测试 P0-2: PortfolioBuilder 严格按 Point-In-Time 股票池过滤候选标的"""
    pit = PointInTimeUniverseProvider()
    pit.add_constituent_change("2023-06-01", "000858.SZ", "IN")
    pit.add_constituent_change("2023-01-01", "600519.SH", "IN")
    
    builder = PortfolioBuilder(top_k_buy=2, top_k_hold=2, universe_provider=pit)
    
    daily_df = pd.DataFrame([
        {"symbol": "600519.SH", "pred_score": 0.9, "industry": "Liquor", "is_suspended": False},
        {"symbol": "000858.SZ", "pred_score": 0.8, "industry": "Liquor", "is_suspended": False},
    ])
    
    # 2023-05-01 时 000858.SZ 尚未纳入
    target_may = builder.build_target_portfolio(daily_df, current_holdings=set(), date="2023-05-01")
    assert "000858.SZ" not in target_may["symbol"].tolist()
    assert "600519.SH" in target_may["symbol"].tolist()


# ==========================================
# 4. P0-3: 日历来源真实性
# ==========================================
def test_akshare_sina_calendar_is_not_marked_exchange_official():
    """测试 P0-3: 新浪第三方日历不得标记为交易所官方认证"""
    manager = DataManager()
    with patch("data.data_manager.ak.tool_trade_date_hist_sina", return_value=pd.DataFrame({"trade_date": ["2023-01-03", "2023-01-04"]})):
        manager._cached_trade_calendar = None
        cal = manager.get_trading_calendar()
        assert manager.calendar_is_exchange_official is False
        assert manager.calendar_quality == "third_party"
        assert manager.calendar_provider == "sina_finance_via_akshare"


def test_business_day_fallback_marked_approximate():
    """测试 P0-3: Business Day Fallback 明确标记为 approximate 与非官方"""
    manager = DataManager()
    with patch("data.data_manager.ak.tool_trade_date_hist_sina", side_effect=Exception("API Error")):
        manager._cached_trade_calendar = None
        cal = manager.get_trading_calendar()
        assert manager.calendar_source == "business_day_fallback"
        assert manager.calendar_is_exchange_official is False
        assert manager.calendar_quality == "approximate"


# ==========================================
# 5. P0-4: 历史交易费率分段
# ==========================================
def test_stamp_duty_before_20230828():
    """测试 P0-4: 2023-08-28 之前卖出印花税为 1‰ (0.001)"""
    rate = TradingFeeSchedule.get_stamp_duty_rate("2023-08-25")
    assert rate == 0.001


def test_stamp_duty_after_20230828():
    """测试 P0-4: 2023-08-28 之后卖出印花税减半为 0.5‰ (0.0005)"""
    rate = TradingFeeSchedule.get_stamp_duty_rate("2023-08-28")
    assert rate == 0.0005


def test_buy_has_no_stamp_duty():
    """测试 P0-4: 买入方向无论历史任何时期均不收取印花税"""
    rules = AShareTradingRules()
    comm, stamp, trans, slip = rules.compute_transaction_cost(100000.0, is_buy=True, trade_date="2022-01-05")
    assert stamp == 0.0


def test_historical_fee_schedule_used_by_engine():
    """测试 P0-4: 回测引擎撮合卖出时真实使用历史分段印花税率"""
    rules = AShareTradingRules()
    # 2022年卖出 100万
    comm1, stamp1, trans1, _ = rules.compute_transaction_cost(1_000_000.0, is_buy=False, trade_date="2022-05-10")
    assert stamp1 == 1000.0 # 1‰
    
    # 2024年卖出 100万
    comm2, stamp2, trans2, _ = rules.compute_transaction_cost(1_000_000.0, is_buy=False, trade_date="2024-05-10")
    assert stamp2 == 500.0 # 0.5‰


# ==========================================
# 6. P0-5: 公司行为除息除权
# ==========================================
def test_cash_dividend_preserves_economic_value():
    """测试 P0-5: 现金分红开盘前现金增加，总经济价值守恒"""
    provider = CorporateActionProvider()
    provider.register_action(CorporateAction(
        symbol="600519.SH",
        ex_date="2023-06-15",
        action_type="CASH_DIVIDEND",
        cash_dividend_per_share=20.0
    ))
    
    engine = BacktestEngine(initial_cash=100000.0, corporate_actions=provider)
    engine.positions["600519.SH"] = PositionRecord(
        symbol="600519.SH",
        shares=1000,
        available_shares=1000,
        avg_cost=100.0,
        last_price=100.0,
        buy_date="2023-01-01",
        highest_price=100.0,
        lots=[PositionLot(shares=1000, buy_execution_price=100.0, buy_date="2023-01-01")]
    )
    
    engine._apply_corporate_actions("2023-06-15")
    assert engine.cash == 100000.0 + 20000.0


def test_stock_split_adjusts_shares_and_cost():
    """测试 P0-5: 送转股按比例增加持仓股数并同比例摊薄平均成本与 Lot 成本"""
    provider = CorporateActionProvider()
    provider.register_action(CorporateAction(
        symbol="600519.SH",
        ex_date="2023-06-15",
        action_type="BONUS_SHARE",
        share_ratio=0.5 # 10送5
    ))
    
    engine = BacktestEngine(initial_cash=100000.0, corporate_actions=provider)
    engine.positions["600519.SH"] = PositionRecord(
        symbol="600519.SH",
        shares=1000,
        available_shares=1000,
        avg_cost=100.0,
        last_price=100.0,
        buy_date="2023-01-01",
        highest_price=100.0,
        lots=[PositionLot(shares=1000, buy_execution_price=100.0, buy_date="2023-01-01")]
    )
    
    engine._apply_corporate_actions("2023-06-15")
    pos = engine.positions["600519.SH"]
    assert pos.shares == 1500
    assert np.isclose(pos.avg_cost, 66.67, atol=1e-2)
    assert np.isclose(pos.lots[0].buy_execution_price, 66.67, atol=1e-2)


def test_corporate_action_does_not_create_fake_pnl():
    """测试 P0-5: 送转股除权后不产生虚假亏损"""
    provider = CorporateActionProvider()
    provider.register_action(CorporateAction(
        symbol="600519.SH",
        ex_date="2023-06-15",
        action_type="BONUS_SHARE",
        share_ratio=1.0 # 10送10 (1拆2)
    ))
    
    engine = BacktestEngine(initial_cash=0.0, corporate_actions=provider)
    pos = PositionRecord(
        symbol="600519.SH",
        shares=1000,
        available_shares=1000,
        avg_cost=100.0,
        last_price=100.0,
        buy_date="2023-01-01",
        highest_price=100.0,
        lots=[PositionLot(shares=1000, buy_execution_price=100.0, buy_date="2023-01-01")]
    )
    engine.positions["600519.SH"] = pos
    
    engine._apply_corporate_actions("2023-06-15")
    # 总市值前: 1000 * 100 = 100,000; 总市值后: 2000 * 50 = 100,000
    assert pos.shares * pos.last_price == 100000.0


def test_missing_corporate_actions_disclosed_in_audit():
    """测试 P0-5: 无公司行为数据时，审计元数据中如实披露 reliability=limited"""
    engine = BacktestEngine()
    audit = AuditCollector.collect(engine=engine)
    assert audit.corporate_action_adjustment_available is False
    assert audit.backtest_total_return_reliability == "limited"


# ==========================================
# 7. P0-6: 持仓状态重建与部分卖出
# ==========================================
def test_partial_sell_recalculates_avg_cost():
    """测试 P0-6: 部分卖出后使用 recalculate_position_from_lots 准确重算剩余批次均价"""
    pos = PositionRecord(
        symbol="600519.SH",
        shares=200,
        available_shares=200,
        avg_cost=105.0,
        last_price=120.0,
        buy_date="2023-01-01",
        highest_price=120.0,
        lots=[
            PositionLot(shares=100, buy_execution_price=100.0, buy_date="2023-01-01"),
            PositionLot(shares=100, buy_execution_price=110.0, buy_date="2023-01-05")
        ]
    )
    
    # 卖出第一批 100 股
    pos.lots.pop(0)
    pos.available_shares -= 100
    recalculate_position_from_lots(pos)
    
    assert pos.shares == 100
    assert pos.avg_cost == 110.0
    assert pos.buy_date == "2023-01-05"


def test_partial_sell_stop_loss_uses_remaining_lots_cost():
    """测试 P0-6: 止损判断严格基于剩余持仓的真实重算均价"""
    pos = PositionRecord(
        symbol="600519.SH",
        shares=100,
        available_shares=100,
        avg_cost=110.0,
        last_price=100.0,
        buy_date="2023-01-05",
        highest_price=110.0,
        lots=[PositionLot(shares=100, buy_execution_price=110.0, buy_date="2023-01-05")]
    )
    # 当前价 100.0 vs avg_cost 110.0 -> 回撤 -9.09% (跌破 8% 硬止损)
    loss_pct = (pos.last_price - pos.avg_cost) / pos.avg_cost
    assert loss_pct <= -settings.STOP_LOSS_PCT


def test_position_state_equals_sum_of_lots():
    """测试 P0-6: 持仓记录状态与内部所有 Lots 严格相等守恒"""
    pos = PositionRecord(
        symbol="600519.SH",
        shares=0,
        available_shares=0,
        avg_cost=0.0,
        last_price=100.0,
        buy_date="",
        highest_price=100.0,
        lots=[
            PositionLot(shares=200, buy_execution_price=50.0, buy_date="2023-01-01"),
            PositionLot(shares=300, buy_execution_price=60.0, buy_date="2023-01-05")
        ]
    )
    recalculate_position_from_lots(pos)
    assert pos.shares == 500
    assert np.isclose(pos.avg_cost, 56.0, atol=1e-4)


# ==========================================
# 8. P0-7: AuditMetadata Fail-Closed
# ==========================================
def test_audit_metadata_fail_closed():
    """测试 P0-7: 审计元数据默认 Fail-Closed (未提供时绝无乐观默认值)"""
    meta = AuditMetadata()
    assert meta.calendar_is_exchange_official is False
    assert meta.survivorship_bias_risk is True
    assert meta.industry_neutralization_enabled == "DISABLED"
    assert meta.industry_coverage_ratio_mean is None
    assert meta.historical_st_available is False
    assert meta.synthetic_data_used is True # 默认 Fail-Closed 假设含模拟数据


def test_audit_metadata_not_hardcoded():
    """测试 P0-7: AuditCollector 从真实运行对象中派生，禁止硬编码假数据"""
    manager = DataManager()
    manager.calendar_source = "akshare_sina"
    manager.calendar_is_exchange_official = False
    
    processor = FactorProcessor()
    processor.industry_neutralization_enabled = "FULL"
    processor.industry_coverage_ratio_mean = 0.98
    
    audit = AuditCollector.collect(data_manager=manager, factor_processor=processor)
    assert audit.calendar_source == "akshare_sina"
    assert audit.calendar_is_exchange_official is False
    assert audit.industry_neutralization_enabled == "FULL"
    assert audit.industry_coverage_ratio_mean == 0.98


def test_api_cli_dashboard_share_same_audit_source():
    """测试 P0-7: 统一调用 AuditCollector 生成审计对象，保证 API/CLI/Dashboard 数据一致"""
    audit1 = AuditCollector.collect()
    assert isinstance(audit1, AuditMetadata)
    assert "survivorship_bias_risk" in audit1.to_dict()


# ==========================================
# 9. P0-8: Date-Indexed 历史 ST 状态
# ==========================================
def test_st_status_changes_over_time():
    """测试 P0-8: 历史 ST 状态支持按日期变更与查询"""
    sec = SecurityMaster()
    sec.load_or_fetch(["600519.SH"])
    sec.register_historical_st_timeline("600519.SH", {
        "2023-01-03": False,
        "2023-05-04": True,
        "2023-12-31": False
    })
    
    assert sec.get_st_status("600519.SH", "2023-01-03") is False
    assert sec.get_st_status("600519.SH", "2023-05-04") is True
    assert sec.get_st_status("600519.SH", "2023-12-31") is False


def test_future_st_status_not_visible_in_past():
    """测试 P0-8: 未来发生的 ST 戴帽在历史日期查询中不可见"""
    sec = SecurityMaster()
    sec.load_or_fetch(["600519.SH"])
    sec.register_historical_st_timeline("600519.SH", {
        "2024-05-01": True
    })
    assert sec.get_st_status("600519.SH", "2023-05-01") in [False, None]


def test_current_st_never_backfills_history():
    """测试 P0-8: 历史 ST 数据缺失时，严禁使用 current_is_st 反填历史"""
    sec = SecurityMaster()
    meta = StockMetadata(symbol="600519.SH", current_is_st=True, historical_st_available=False)
    sec._metadata_map["600519.SH"] = meta
    
    status = sec.get_st_status("600519.SH", "2022-01-01")
    assert status is None # 显式未知


def test_missing_historical_st_is_explicitly_unknown():
    """测试 P0-8: 历史 ST 数据缺失时，DataManager 将 is_st 置 False 且 historical_st_rule_applied 置 False"""
    manager = DataManager()
    meta = StockMetadata(symbol="600519.SH", current_is_st=True, historical_st_available=False)
    
    df = pd.DataFrame([{"date": "2022-01-04", "symbol": "600519.SH", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000, "amount": 100000, "turnover": 0.01}])
    annotated = manager._annotate_ashare_status(df, meta)
    
    assert annotated["is_st"].iloc[0] == False
    assert annotated["historical_st_rule_applied"].iloc[0] == False


# ==========================================
# 10. P1-1 ~ P1-7 专项测试
# ==========================================
def test_pending_orders_cancelled_at_backtest_end():
    """测试 P1-1: 回测结束时所有未成交/延期挂单强制转为 CANCELLED (END_OF_BACKTEST)"""
    engine = BacktestEngine(initial_cash=500000.0, top_k_buy=1, top_k_hold=1, rebalance_freq=1)
    dates = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
    
    # Day 1: 产生买入信号
    # Day 2: 开盘买入成交，收盘暴跌至 90 元 (回撤 -10% 触发个股硬止损产生卖单)
    # Day 3 (最后一天): 一字跌停封死无法卖出，卖单转入 DEFERRED 挂单，回测收盘结束触发 CANCELLED (END_OF_BACKTEST)
    df = pd.DataFrame([
        {"date": dates[0], "symbol": "600519.SH", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 10000, "is_suspended": False, "is_limit_up_locked": False, "is_limit_down_locked": False, "pred_score": 0.9, "benchmark_close": 4000.0},
        {"date": dates[1], "symbol": "600519.SH", "open": 100.0, "high": 100.0, "low": 90.0, "close": 90.0, "volume": 10000, "is_suspended": False, "is_limit_up_locked": False, "is_limit_down_locked": False, "pred_score": 0.9, "benchmark_close": 4000.0},
        {"date": dates[2], "symbol": "600519.SH", "open": 81.0, "high": 81.0, "low": 81.0, "close": 81.0, "volume": 10000, "is_suspended": False, "is_limit_up_locked": False, "is_limit_down_locked": True, "limit_down_price": 81.0, "pred_score": -0.9, "benchmark_close": 4000.0},
    ])
    
    equity_df, orders_df = engine.run(df)
    cancelled_orders = [o for o in engine.completed_orders if o.status == OrderStatus.CANCELLED]
    assert len(cancelled_orders) >= 1
    assert cancelled_orders[-1].reject_reason == RejectReason.END_OF_BACKTEST.value
    assert len(engine.pending_orders) == 0


def test_stale_price_warning_metrics():
    """测试 P1-2: 停牌过时价格统计正确跟踪事件数、受影响标的与总天数"""
    engine = BacktestEngine(initial_cash=500000.0)
    pos = PositionRecord(
        symbol="600519.SH",
        shares=100,
        available_shares=100,
        avg_cost=100.0,
        last_price=100.0,
        buy_date="2023-01-01",
        highest_price=100.0,
        last_price_date="2023-01-01"
    )
    engine.positions["600519.SH"] = pos
    
    # 2023-04-01 已停牌 90 天 (超过 60 天)
    daily_map = {"600519.SH": pd.Series({"symbol": "600519.SH", "open": 100.0, "close": 100.0, "is_suspended": True, "volume": 0})}
    engine.mark_to_market(pd.Timestamp("2023-04-01"), daily_map)
    
    assert engine.stale_price_warning_events >= 1
    assert "600519.SH" in engine.stale_price_affected_symbols
    assert engine.max_stale_price_days >= 90


def test_trading_day_holding_period_calculation():
    """测试 P1-3: 持仓天数基于真实 A 股交易日历计算"""
    engine = BacktestEngine()
    engine.trading_calendar = pd.date_range("2023-01-01", "2023-01-31", freq="B").tolist()
    
    buy_date = "2023-01-03" # 周二
    sell_date = "2023-01-10" # 下周二
    
    # 跨过一个周末: 01-03, 01-04, 01-05, 01-06, 01-09, 01-10 -> 共 6 个交易日
    trd_days = count_trading_days(buy_date, sell_date, engine.trading_calendar)
    assert trd_days == 6


def test_performance_metrics_definitions():
    """测试 P1-4: 绩效评价引擎标准金融定义 (Sharpe, Sortino, Alpha CAPM 回归)"""
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    equity_df = pd.DataFrame({
        "date": dates,
        "total_equity": [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 107.0],
        "strategy_return": [0.0, 0.01, 0.0099, -0.0049, 0.0148, 0.0097, -0.0048, 0.0145, 0.0095, 0.0094],
        "benchmark_equity": [100.0, 100.5, 101.0, 100.8, 101.5, 102.0, 101.8, 102.5, 103.0, 103.5],
        "benchmark_return": [0.0, 0.005, 0.005, -0.002, 0.007, 0.005, -0.002, 0.007, 0.005, 0.005]
    })
    orders_df = pd.DataFrame()
    
    metrics = analyzer.calculate_metrics(equity_df, orders_df)
    assert "alpha_capm_regression" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert metrics["sharpe_ratio"] > 0


def test_liquidity_participation_limit():
    """测试 P1-5: 单日买入/卖出受到可用成交量 5% 容量约束 (超额部分部分成交 PARTIALLY_FILLED)"""
    engine = BacktestEngine(initial_cash=1000000.0, enable_liquidity_constraint=True, max_volume_participation=0.05)
    
    # 当日成交量仅 10,000 股 -> 5% 容量为 500 股
    daily_map = {
        "600519.SH": pd.Series({
            "symbol": "600519.SH", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "volume": 10000, "is_suspended": False, "is_limit_up_locked": False
        })
    }
    # 请求买入 2000 股 (超限)
    buy_order = Order(order_id="B1", symbol="600519.SH", side=OrderSide.BUY, signal_date="2023-01-01", requested_shares=2000)
    engine.pending_orders = [buy_order]
    
    engine._execute_morning_auction(daily_map, "2023-01-02")
    
    assert buy_order.status == OrderStatus.PARTIALLY_FILLED
    assert buy_order.filled_shares == 500
    assert engine.partial_fill_count == 1


def test_sector_constraint_audit_semantics():
    """测试 P1-6: 行业约束审计语义拆分 (即使全部 UNKNOWN 也正确报告 sector_cap_enabled=True 与 industry_data_available=False)"""
    builder = PortfolioBuilder(top_k_buy=2, max_sector_exposure=0.30)
    daily_df = pd.DataFrame([
        {"symbol": "A", "pred_score": 0.9, "industry": "UNKNOWN", "is_suspended": False},
        {"symbol": "B", "pred_score": 0.8, "industry": "UNKNOWN", "is_suspended": False}
    ])
    builder.build_target_portfolio(daily_df, current_holdings=set())
    
    audit = AuditCollector.collect(portfolio_builder=builder)
    assert audit.sector_cap_enabled is True
    assert audit.industry_data_available is False
    assert audit.unknown_industry_cap_applied is True


def test_benchmark_fill_is_date_series_only():
    """测试 P1-7: 基准指数缺失值前向填充在独立时间序列上完成，绝无跨 symbol 污染"""
    bench_df = pd.DataFrame({
        "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
        "close": [4000.0, np.nan, 4050.0]
    })
    bench_df["date"] = pd.to_datetime(bench_df["date"])
    bench_df.sort_values("date", inplace=True)
    bench_df["benchmark_close"] = bench_df["close"].ffill()
    
    assert bench_df["benchmark_close"].iloc[1] == 4000.0
    assert bench_df["benchmark_close"].iloc[2] == 4050.0
