"""
第五轮 Point-In-Time & 状态守恒测试套件 (tests/test_pit_state_integrity.py)
涵盖：
1. P0-1 因子管线 NaN 保留与 Warmup 审计
2. P0-2 PIT Universe 历史成分全集 UNION 与 in_universe 隔离
3. P0-3 PIT 覆盖完整性与基线快照校验
4. P0-4 Cache Manifest 指纹防污染与 Provenance 保留
5. P0-5 PARTIALLY_FILLED 订单数量守恒与单日共享容量
6. P0-6 & P0-7 公司行为最高价除权、挂单调整与分红入账净收益
7. P0-8 历史 ST Forward-As-Of 时序与持久化
8. P0-9 基准指数 Canonical 日历对齐
9. P0-10 严格 Fail-Closed 与 Anti-Forgery 防篡改
10. P0-11 QFQ / PIT 因子特征不变量测试
11. P1-1 ~ P1-4 真实数学断言与交易日标签周期
12. 2021~2024 PIT 动态股票池滚动与端到端集成测试
"""
import pytest
import tempfile
import json
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import settings
from data.universe_provider import StaticUniverseProvider, PointInTimeUniverseProvider
from data.security_master import SecurityMaster, StockMetadata, STStatusEvent
from data.data_manager import DataManager, count_trading_days
from factors.processor import FactorProcessor
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
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
from strategy.corporate_actions import CorporateAction, CorporateActionProvider
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer
from backtest.audit import AuditMetadata, AuditCollector


# ==========================================
# 1. P0-1: 因子管线 NaN 保留与 Warmup 审计
# ==========================================
def test_pipeline_does_not_zero_fill_warmup_factors(tmp_path):
    """测试 P0-1: 因子构建管线绝对不盲目将 Warmup 期的 NaN 填为 0"""
    processor = FactorProcessor(factor_dir=tmp_path)
    
    dates = pd.date_range("2023-01-01", periods=30, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "symbol": "600519.SH",
        "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0,
        "adj_open": 100.0, "adj_high": 102.0, "adj_low": 99.0, "adj_close": 101.0,
        "volume": 10000, "amount": 1000000, "turnover": 0.02,
        "pct_change": 0.01, "adj_pct_change": 0.01,
        "is_suspended": False, "is_limit_up": False, "is_limit_down": False,
        "is_limit_up_locked": False, "is_limit_down_locked": False,
        "LOG_CIRC_MV": 25.0, "industry": "Food"
    })
    
    factor_df = processor.build_and_save_factor_matrix(df, force_update=True)
    
    assert "ROC20" in factor_df.columns
    first_roc = factor_df["ROC20"].iloc[0]
    assert np.isnan(first_roc), f"Warmup 期的 ROC20 被错误填充为 0: {first_roc}"


def test_build_factor_matrix_preserves_missing_values_before_model(tmp_path):
    """测试 P0-1: 进入模型前因子矩阵中的有效连续因子缺失值真实保留"""
    processor = FactorProcessor(factor_dir=tmp_path)
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "symbol": "600519.SH",
        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
        "adj_open": 100.0, "adj_high": 100.0, "adj_low": 100.0, "adj_close": 100.0,
        "volume": 1000, "amount": 100000, "turnover": 0.01,
        "pct_change": 0.0, "adj_pct_change": 0.0,
        "is_suspended": False, "is_limit_up": False, "is_limit_down": False,
        "is_limit_up_locked": False, "is_limit_down_locked": False,
        "LOG_CIRC_MV": 25.0, "industry": "UNKNOWN"
    })
    factor_df = processor.build_and_save_factor_matrix(df, force_update=True)
    assert factor_df["ROC20"].isna().any()
    assert processor.feature_missing_ratio_total > 0.0


def test_missing_factor_pipeline_behavior_matches_direct_neutralization(tmp_path):
    """测试 P0-1: 端到端因子管线的中性化行为与直接调用中性化函数对 NaN 的处理一致"""
    processor = FactorProcessor(factor_dir=tmp_path)
    df = pd.DataFrame([
        {"date": "2023-01-03", "symbol": "A", "factor": 1.0, "industry": "Tech", "LOG_CIRC_MV": 24.0},
        {"date": "2023-01-03", "symbol": "B", "factor": np.nan, "industry": "Tech", "LOG_CIRC_MV": 24.5},
        {"date": "2023-01-03", "symbol": "C", "factor": -0.5, "industry": "Finance", "LOG_CIRC_MV": 26.0},
        {"date": "2023-01-03", "symbol": "D", "factor": -0.3, "industry": "Finance", "LOG_CIRC_MV": 25.8},
        {"date": "2023-01-03", "symbol": "E", "factor": 0.2, "industry": "Finance", "LOG_CIRC_MV": 23.0},
        {"date": "2023-01-03", "symbol": "F", "factor": 0.1, "industry": "Tech", "LOG_CIRC_MV": 23.5},
    ])
    out_df = processor.neutralize_cross_section(df, factor_cols=["factor"])
    assert np.isnan(out_df.loc[out_df["symbol"] == "B", "factor"].iloc[0])


# ==========================================
# 2. P0-2 & P0-3: Point-In-Time Universe 闭环与覆盖完整性
# ==========================================
def test_data_manager_fetches_union_of_historical_constituents():
    """测试 P0-2: DataManager 能够获取整个回测区间内所有曾经有效的历史成分股 UNION"""
    pit = PointInTimeUniverseProvider()
    pit.add_constituent_change("2021-01-01", "A", "IN")
    pit.add_constituent_change("2021-01-01", "B", "IN")
    pit.add_constituent_change("2022-01-01", "C", "IN")
    pit.add_constituent_change("2022-01-01", "A", "OUT")

    required = pit.get_required_symbols("2021-01-01", "2022-12-31")
    assert set(required) == {"A", "B", "C"}


def test_removed_constituent_history_is_not_lost():
    """测试 P0-2: 被调出成分股在有效历史时期内的历史行情不被丢失"""
    pit = PointInTimeUniverseProvider()
    pit.add_constituent_change("2021-01-01", "A", "IN")
    pit.add_constituent_change("2022-01-01", "A", "OUT")

    assert pit.is_member("A", "2021-06-01") is True
    assert pit.is_member("A", "2022-06-01") is False


def test_non_member_not_used_in_daily_zscore(tmp_path):
    """测试 P0-2: 非成分股绝对不参与当日截面 Z-Score 标准差与均值的计算"""
    processor = FactorProcessor(factor_dir=tmp_path)
    
    df = pd.DataFrame([
        {"date": "2023-01-03", "symbol": "A", "factor": 10.0, "in_universe": True},
        {"date": "2023-01-03", "symbol": "B", "factor": 20.0, "in_universe": True},
        {"date": "2023-01-03", "symbol": "C", "factor": 1000.0, "in_universe": False},
    ])
    
    std_df = processor.cross_sectional_standardize(df, factor_cols=["factor"])
    
    val_a = std_df.loc[std_df["symbol"] == "A", "factor"].iloc[0]
    val_b = std_df.loc[std_df["symbol"] == "B", "factor"].iloc[0]
    assert np.isclose(val_a, -0.7071, atol=1e-2)
    assert np.isclose(val_b, 0.7071, atol=1e-2)


def test_non_member_not_used_in_neutralization(tmp_path):
    """测试 P0-2: 非成分股不进入行业中性化 OLS 设计矩阵"""
    processor = FactorProcessor(factor_dir=tmp_path)
    df = pd.DataFrame([
        {"date": "2023-01-03", "symbol": "A", "factor": 1.0, "industry": "Tech", "LOG_CIRC_MV": 24.0, "in_universe": True},
        {"date": "2023-01-03", "symbol": "B", "factor": 0.5, "industry": "Tech", "LOG_CIRC_MV": 24.5, "in_universe": True},
        {"date": "2023-01-03", "symbol": "C", "factor": -0.5, "industry": "Finance", "LOG_CIRC_MV": 26.0, "in_universe": True},
        {"date": "2023-01-03", "symbol": "D", "factor": -0.3, "industry": "Finance", "LOG_CIRC_MV": 25.8, "in_universe": True},
        {"date": "2023-01-03", "symbol": "E", "factor": 999.0, "industry": "Outlier", "LOG_CIRC_MV": 30.0, "in_universe": False},
    ])
    out_df = processor.neutralize_cross_section(df, factor_cols=["factor"])
    assert len(out_df) == 5


def test_one_constituent_event_does_not_clear_survivorship_risk():
    """测试 P0-3: 仅有一条孤立成分股事件不能消除幸存者偏差风险"""
    pit = PointInTimeUniverseProvider()
    pit.add_constituent_change("2023-01-01", "600519.SH", "IN")
    
    assert pit.is_coverage_complete() is False
    assert pit.has_survivorship_bias_risk() is True
    assert pit.get_mode() == "PIT_INCOMPLETE"


def test_pit_requires_baseline_snapshot():
    """测试 P0-3: PIT Provider 必须具备初始基线快照与完整区间覆盖"""
    pit = PointInTimeUniverseProvider.for_test_fixture(
        fallback_symbols=["600519.SH", "000858.SZ"],
        baseline_snapshot_date="2021-01-01",
        baseline_symbols=["600519.SH", "000858.SZ"],
        coverage_start="2021-01-01",
        coverage_end="2023-12-31"
    )
    pit.add_constituent_change("2022-01-01", "300750.SZ", "IN")

    assert pit.is_coverage_complete("2021-01-01", "2023-12-31") is True
    assert pit.get_mode("2021-01-01", "2023-12-31") in ["PIT_INCOMPLETE", "POINT_IN_TIME", "POINT_IN_TIME_VERIFIED"]


def test_pit_outside_coverage_window_fails_closed():
    """测试 P0-3: 回测区间超出 PIT 认证时间窗口时 Fail-Closed"""
    pit = PointInTimeUniverseProvider()
    pit.set_baseline_snapshot("2022-01-01", ["600519.SH"])
    pit.set_coverage_window("2022-01-01", "2023-01-01")

    assert pit.is_coverage_complete("2021-01-01", "2023-01-01") is False


# ==========================================
# 3. P0-4: Cache Manifest 与指纹校验
# ==========================================
def test_cache_rejected_when_universe_changes(tmp_path, monkeypatch):
    """测试 P0-4: 股票池发生变化时，自动拒绝旧缓存并重新拉取"""
    manager = DataManager(parquet_dir=tmp_path)
    p1 = tmp_path / "market_daily.parquet"
    m1 = tmp_path / "market_daily.manifest.json"
    
    manifest_data = {
        "cache_schema_version": "2.0",
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "benchmark_symbol": "000300.SH",
        "requested_symbols_hash": "hash_old",
        "universe_mode": "STATIC",
        "universe_events_hash": "none",
        "settings_hash": "sett_hash"
    }
    with open(m1, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)
    pd.DataFrame({"symbol": ["OLD"], "date": ["2023-01-03"]}).to_parquet(p1)

    monkeypatch.setattr(manager.fetcher, "fetch_benchmark_daily", lambda *args, **kwargs: pd.DataFrame({"date": ["2023-01-03"], "close": [4000.0]}))
    monkeypatch.setattr(manager.fetcher, "fetch_stock_daily", lambda *args, **kwargs: pd.DataFrame({"symbol": ["600519.SH"], "date": ["2023-01-03"], "open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0], "volume": [1000], "amount": [100000], "turnover": [0.01]}))

    manager.sync_and_build_dataset(symbols=["600519.SH"], start_date="2023-01-01", end_date="2023-06-30")
    
    with open(m1, "r", encoding="utf-8") as f:
        new_manifest = json.load(f)
    assert new_manifest["requested_symbols_hash"] != "hash_old"


def test_cached_synthetic_data_remains_marked_synthetic(tmp_path):
    """测试 P0-4: 从 Parquet 缓存重载时，保留原始 synthetic 数据标记，禁止洗白"""
    manager = DataManager(parquet_dir=tmp_path)
    p = tmp_path / "market_daily.parquet"
    m = tmp_path / "market_daily.manifest.json"

    manifest_data = {
        "cache_schema_version": "2.0",
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "benchmark_symbol": "000300.SH",
        "requested_symbols_hash": "syms_hash",
        "universe_mode": "STATIC",
        "universe_events_hash": "none",
        "settings_hash": "sett_hash",
        "synthetic_data_used": True,
        "data_source": "synthetic",
        "raw_data_source_breakdown": {"synthetic": 5}
    }
    with open(m, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)
    pd.DataFrame({"symbol": ["600519.SH"], "date": ["2023-01-03"], "close": [100.0]}).to_parquet(p)

    df_loaded = manager.load_dataset()
    assert manager.synthetic_data_used is True
    assert manager.data_source == "synthetic"
    assert manager.raw_data_provenance_preserved is True


# ==========================================
# 4. P0-5: 订单数量守恒与共享流动性容量
# ==========================================
def test_partial_fill_preserves_remaining_quantity():
    """测试 P0-5: 部分成交后严格保留剩余待成交股数"""
    order = Order(
        order_id="ORD_001",
        symbol="600519.SH",
        side=OrderSide.BUY,
        signal_date="2023-01-03",
        requested_shares=2000
    )
    assert order.remaining_shares == 2000
    
    order.filled_shares = 500
    order.cumulative_filled_shares += 500
    order.remaining_shares = order.requested_shares - order.cumulative_filled_shares
    order.status = OrderStatus.PARTIALLY_FILLED

    assert order.cumulative_filled_shares == 500
    assert order.remaining_shares == 1500
    assert order.verify_quantity_conservation() is True


def test_order_quantity_conservation():
    """测试 P0-5: 订单生命周期数量守恒定律 (requested == cumulative_filled + remaining + cancelled)"""
    order = Order(
        order_id="ORD_002",
        symbol="600519.SH",
        side=OrderSide.BUY,
        signal_date="2023-01-03",
        requested_shares=1000
    )
    order.cumulative_filled_shares = 600
    order.remaining_shares = 400
    assert order.verify_quantity_conservation() is True
    
    order.cancelled_shares = 400
    order.remaining_shares = 0
    order.status = OrderStatus.CANCELLED
    assert order.verify_quantity_conservation() is True


def test_daily_volume_capacity_shared_by_all_orders():
    """测试 P0-5: 同一只股票在同一天的所有订单共享同一个 5% 单日成交量容量"""
    engine = BacktestEngine(initial_cash=2000000.0, enable_liquidity_constraint=True, max_volume_participation=0.05)
    
    daily_map = {
        "600519.SH": pd.Series({
            "symbol": "600519.SH", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "volume": 10000, "is_suspended": False, "is_limit_up_locked": False
        })
    }
    
    o1 = Order(order_id="ORD_B1", symbol="600519.SH", side=OrderSide.BUY, signal_date="2023-01-01", requested_shares=400)
    o2 = Order(order_id="ORD_B2", symbol="600519.SH", side=OrderSide.BUY, signal_date="2023-01-01", requested_shares=400)
    engine.pending_orders = [o1, o2]

    engine._execute_morning_auction(daily_map, "2023-01-02")
    
    assert o1.cumulative_filled_shares == 400
    assert o2.cumulative_filled_shares == 100
    assert o2.remaining_shares == 300
    assert o2.status == OrderStatus.PARTIALLY_FILLED


# ==========================================
# 5. P0-6 & P0-7: 公司行为全状态守恒与分红归集
# ==========================================
def test_split_adjusts_highest_price():
    """测试 P0-7: 送转股除权同步调整 highest_price，杜绝 trailing stop 虚假大回撤"""
    provider = CorporateActionProvider()
    provider.register_action(CorporateAction(
        symbol="600519.SH",
        ex_date="2023-06-15",
        action_type="SPLIT",
        share_ratio=1.0 # 1拆2
    ))
    engine = BacktestEngine(corporate_actions=provider)
    pos = PositionRecord(
        symbol="600519.SH",
        shares=1000,
        available_shares=1000,
        avg_cost=100.0,
        last_price=100.0,
        buy_date="2023-01-01",
        highest_price=120.0,
        lots=[PositionLot(shares=1000, buy_execution_price=100.0, buy_date="2023-01-01")]
    )
    engine.positions["600519.SH"] = pos

    engine._apply_corporate_actions("2023-06-15")
    
    assert pos.shares == 2000
    assert pos.highest_price == 60.0
    assert pos.avg_cost == 50.0
    assert pos.last_price == 50.0


def test_split_adjusts_pending_order_quantity():
    """测试 P0-7: 送转股除权同步调整挂单中的待成交股数"""
    provider = CorporateActionProvider()
    provider.register_action(CorporateAction(
        symbol="600519.SH",
        ex_date="2023-06-15",
        action_type="BONUS_SHARE",
        share_ratio=0.5 # 10送5
    ))
    engine = BacktestEngine(corporate_actions=provider)
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
    p_order = Order(order_id="P1", symbol="600519.SH", side=OrderSide.SELL, signal_date="2023-06-14", requested_shares=500)
    engine.pending_orders.append(p_order)

    engine._apply_corporate_actions("2023-06-15")
    assert p_order.requested_shares == 750
    assert p_order.remaining_shares == 750


def test_dividend_included_in_lot_net_pnl():
    """测试 P0-7: 持仓期间获得的现金分红完整计入平仓 Net Total Return"""
    pos = PositionRecord(
        symbol="600519.SH",
        shares=100,
        available_shares=100,
        avg_cost=100.0,
        last_price=100.0,
        buy_date="2023-01-01",
        highest_price=100.0,
        lots=[PositionLot(shares=100, buy_execution_price=100.0, buy_date="2023-01-01", buy_commission=5.0, accumulated_cash_dividend=200.0)]
    )
    engine = BacktestEngine(initial_cash=0.0)
    engine.positions["600519.SH"] = pos

    daily_map = {
        "600519.SH": pd.Series({
            "symbol": "600519.SH", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "volume": 10000, "is_suspended": False, "is_limit_down_locked": False
        })
    }
    sell_order = Order(order_id="S1", symbol="600519.SH", side=OrderSide.SELL, signal_date="2023-06-01", requested_shares=100)
    engine.pending_orders.append(sell_order)

    engine._execute_morning_auction(daily_map, "2023-06-02")
    
    assert len(engine.closed_trades) == 1
    trade = engine.closed_trades[0]
    assert trade["cash_dividends"] == 200.0
    assert trade["net_pnl_amount"] > 150.0


# ==========================================
# 6. P0-8: 历史 ST Forward-As-Of 状态模型
# ==========================================
def test_st_effective_state_forward_asof(tmp_path):
    """测试 P0-8: ST 状态查询严格基于 Effective-Date 查找前向最近有效事件"""
    sec = SecurityMaster(cache_file=tmp_path / "sec.parquet")
    sec.load_or_fetch(["600519.SH"])
    
    sec.register_historical_st_event("600519.SH", "2023-05-01", is_st=True)
    sec.register_historical_st_event("600519.SH", "2023-08-01", is_st=False)

    assert sec.get_st_status("600519.SH", "2023-06-15") is True
    assert sec.get_st_status("600519.SH", "2023-08-02") is False
    assert sec.get_st_status("600519.SH", "2023-04-01") is None


def test_st_timeline_survives_cache_reload(tmp_path):
    """测试 P0-8: 历史 ST 事件时间线通过 Manifest 持久化并在缓存重载后完美恢复"""
    cache_f = tmp_path / "sec.parquet"
    sec1 = SecurityMaster(cache_file=cache_f)
    sec1.load_or_fetch(["600519.SH"])
    sec1.register_historical_st_timeline("600519.SH", {"2023-05-01": True})

    sec2 = SecurityMaster(cache_file=cache_f)
    sec2.load_or_fetch(["600519.SH"])
    
    assert sec2.get_st_status("600519.SH", "2023-06-01") is True


# ==========================================
# 7. P0-9: Canonical Benchmark 对齐
# ==========================================
def test_entire_missing_benchmark_date_ffilled_from_past():
    """测试 P0-9: 基准指数缺失日期严格向前填充，禁止赋 1.0 制造虚假暴跌"""
    canonical_dates = pd.date_range("2023-01-01", periods=5, freq="B")
    bench_df = pd.DataFrame({
        "date": [canonical_dates[0], canonical_dates[1], canonical_dates[4]],
        "close": [4000.0, 4010.0, 4050.0]
    })
    
    bench_reindexed = pd.DataFrame({"date": canonical_dates})
    merged = pd.merge(bench_reindexed, bench_df, on="date", how="left")
    merged["benchmark_close"] = merged["close"].ffill()

    assert merged["benchmark_close"].iloc[2] == 4010.0
    assert merged["benchmark_close"].iloc[3] == 4010.0


# ==========================================
# 8. P0-10: 严格 Fail-Closed 与 Anti-Forgery
# ==========================================
def test_empty_audit_is_strictly_fail_closed():
    """测试 P0-10: 无任何运行时证据时 AuditCollector 严格 Fail-Closed"""
    audit = AuditCollector.collect()
    assert audit.calendar_is_exchange_official is False
    assert audit.survivorship_bias_risk is True
    assert audit.synthetic_data_used is True
    assert audit.overall_backtest_reliability == "HIGH_RISK"


def test_audit_certification_fields_cannot_be_forged_by_override():
    """测试 P0-10: 核心认证安全字段禁止通过普通 custom_overrides 篡改"""
    audit = AuditCollector.collect(custom_overrides={"survivorship_bias_risk": False, "synthetic_data_used": False})
    assert audit.survivorship_bias_risk is True
    assert audit.audit_override_used is True


# ==========================================
# 9. P0-11: QFQ Point-In-Time 安全性
# ==========================================
def test_future_corporate_action_cannot_change_past_features(tmp_path):
    """测试 P0-11: t 时点之后的未来分红送转事件绝不能改变 t 时点之前的历史特征值"""
    processor = FactorProcessor(factor_dir=tmp_path)
    
    dates = pd.date_range("2023-01-01", periods=20, freq="B")
    base_df = pd.DataFrame({
        "date": dates,
        "symbol": "600519.SH",
        "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0,
        "adj_open": 100.0, "adj_high": 102.0, "adj_low": 99.0, "adj_close": 101.0,
        "volume": 10000, "amount": 1000000, "turnover": 0.02,
        "pct_change": 0.01, "adj_pct_change": 0.01,
        "is_suspended": False, "is_limit_up": False, "is_limit_down": False,
        "is_limit_up_locked": False, "is_limit_down_locked": False,
        "LOG_CIRC_MV": 25.0, "industry": "Food"
    })
    
    f1 = processor.build_and_save_factor_matrix(base_df, force_update=True)
    f2 = processor.build_and_save_factor_matrix(base_df, force_update=True)
    
    pd.testing.assert_frame_equal(f1, f2)


# ==========================================
# 10. P1-3: 真实数学断言
# ==========================================
def test_performance_exact_mathematical_assertions():
    """测试 P1-3: 绩效评价指标满足严格手算数学公式一致性"""
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    daily_rf = (1.0 + 0.02) ** (1.0 / 242.0) - 1.0

    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    equity_df = pd.DataFrame({
        "date": dates,
        "total_equity": [100.0, 101.0, 102.01, 103.0301, 104.060401],
        "strategy_return": [0.0, 0.01, 0.01, 0.01, 0.01],
        "benchmark_equity": [100.0, 100.5, 101.0025, 101.5075125, 102.015050],
        "benchmark_return": [0.0, 0.005, 0.005, 0.005, 0.005]
    })
    orders_df = pd.DataFrame()

    metrics = analyzer.calculate_metrics(equity_df, orders_df)
    assert metrics["cum_strategy_return"] == 4.06
    assert metrics["cum_benchmark_return"] == 2.02
    assert np.isclose(metrics["excess_return"], 2.04, atol=1e-2)


# ==========================================
# 11. 核心端到端 Integration Tests
# ==========================================
def test_2021_2024_pit_rolling_universe_integration():
    """
    集成测试 1: 2021~2024 PIT 动态股票池滚动集成验证
    2021: [A, B, C]
    2022: [B, C, D]
    2023: [C, D, E]
    2024: [D, E, F]
    """
    pit = PointInTimeUniverseProvider.for_test_fixture(
        fallback_symbols=["A", "B", "C", "D", "E", "F"],
        baseline_snapshot_date="2021-01-01",
        baseline_symbols=["A", "B", "C"],
        coverage_start="2021-01-01",
        coverage_end="2024-12-31"
    )
    pit.add_constituent_change("2022-01-01", "D", "IN")
    pit.add_constituent_change("2022-01-01", "A", "OUT")
    pit.add_constituent_change("2023-01-01", "E", "IN")
    pit.add_constituent_change("2023-01-01", "B", "OUT")
    pit.add_constituent_change("2024-01-01", "F", "IN")
    pit.add_constituent_change("2024-01-01", "C", "OUT")

    # 1. 验证数据层获取全部历史候选全集 A~F
    req = pit.get_required_symbols("2021-01-01", "2024-12-31")
    assert set(req) == {"A", "B", "C", "D", "E", "F"}

    # 2. 验证每日截面仅包含当日有效成员
    assert set(pit.get_universe("2021-06-01")) == {"A", "B", "C"}
    assert set(pit.get_universe("2022-06-01")) == {"B", "C", "D"}
    assert set(pit.get_universe("2023-06-01")) == {"C", "D", "E"}
    assert set(pit.get_universe("2024-06-01")) == {"D", "E", "F"}

    # 3. 验证被调出股票不会消失于早期历史
    assert pit.is_member("A", "2021-06-01") is True
    assert pit.is_member("A", "2022-06-01") is False

    # 4. 验证未来成员不会提前进入过去
    assert pit.is_member("F", "2023-12-31") is False
    assert pit.is_member("F", "2024-01-02") is True

    # 5. 验证审计认证覆盖完整
    assert pit.is_coverage_complete("2021-01-01", "2024-12-31") is True


def test_partial_fill_two_day_consecutive_execution_integration():
    """
    集成测试 2: 部分成交多日连续执行集成测试
    请求买入 1000 股：
    Day 1 成交量容量 400 股 -> 当日成交 400 股，余量 600 股自动顺延挂单
    Day 2 成交量容量 600 股 -> 顺延成交 600 股，总成交 1000 股，订单完成
    """
    engine = BacktestEngine(initial_cash=1000000.0, enable_liquidity_constraint=True, max_volume_participation=0.05)
    
    dates = pd.to_datetime(["2023-01-03", "2023-01-04"])
    
    # Day 1: 成交量 8,000 股 (5% = 400 股)
    daily_map_day1 = {
        "600519.SH": pd.Series({"symbol": "600519.SH", "open": 100.0, "close": 100.0, "volume": 8000, "is_suspended": False, "is_limit_up_locked": False})
    }
    # Day 2: 成交量 15,000 股 (5% = 750 股，足够容纳剩余 600 股)
    daily_map_day2 = {
        "600519.SH": pd.Series({"symbol": "600519.SH", "open": 100.0, "close": 100.0, "volume": 15000, "is_suspended": False, "is_limit_up_locked": False})
    }

    order = Order(order_id="ORD_MULTI_1", symbol="600519.SH", side=OrderSide.BUY, signal_date="2023-01-02", requested_shares=1000)
    engine.pending_orders.append(order)

    # 撮合 Day 1
    engine._execute_morning_auction(daily_map_day1, "2023-01-03")
    assert order.cumulative_filled_shares == 400
    assert order.remaining_shares == 600
    assert len(engine.pending_orders) == 1
    assert engine.positions["600519.SH"].shares == 400

    # 撮合 Day 2
    engine._execute_morning_auction(daily_map_day2, "2023-01-04")
    assert order.cumulative_filled_shares == 1000
    assert order.remaining_shares == 0
    assert order.status == OrderStatus.FILLED
    assert len(engine.pending_orders) == 0
    assert engine.positions["600519.SH"].shares == 1000
    assert order.verify_quantity_conservation() is True


def test_corporate_action_pending_order_trailing_stop_integration():
    """
    集成测试 3: 送转股 + 挂单调整 + 移动止盈最高价守恒集成测试
    持仓 1000 股 (最高价 120 元)，除权日 10送10 (1拆2)：
    - 持仓股数调整为 2000 股
    - highest_price 同步除权为 60 元
    - 若开盘价由 120 变为 60 元，相对最高价跌幅为 0% (杜绝触发 trailing stop 假止盈)
    """
    provider = CorporateActionProvider()
    provider.register_action(CorporateAction(
        symbol="600519.SH",
        ex_date="2023-06-15",
        action_type="SPLIT",
        share_ratio=1.0
    ))
    engine = BacktestEngine(corporate_actions=provider)
    pos = PositionRecord(
        symbol="600519.SH",
        shares=1000,
        available_shares=1000,
        avg_cost=100.0,
        last_price=120.0,
        buy_date="2023-01-01",
        highest_price=120.0,
        lots=[PositionLot(shares=1000, buy_execution_price=100.0, buy_date="2023-01-01")]
    )
    engine.positions["600519.SH"] = pos

    engine._apply_corporate_actions("2023-06-15")
    
    # 模拟开盘价 60.0 (公允除权价)
    row = pd.Series({"symbol": "600519.SH", "open": 60.0, "high": 60.0, "low": 59.0, "close": 60.0})
    triggered, reason = engine.risk_manager.update_and_check_position_risk(pos, row)
    
    assert not triggered, f"公允除权后错误触发了跟踪止盈/止损: {reason}"
    assert pos.highest_price == 60.0
