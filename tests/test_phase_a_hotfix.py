"""
Phase A 热修复回归测试 (tests/test_phase_a_hotfix.py)

锁定 2026-09-01 Phase A 修复的三类 P0/P1 缺陷:
1. PaperBroker T+1 锁死 bug: 买入后 available_shares 永不增加、unlock_t1_shares 无调用点
2. 防线 5 失效 bug: run_trader 以 (p, p) 调用且不传 pre_close → 涨停拦截恒不触发;
   且 +9.8% 硬编码阈值对 20%/30% 涨跌幅板块错误
3. daily_runner 健壮性: 分阶段重试 (_execute_stage) 与失败告警 (_push_failure_alert)
"""
import pytest
from datetime import datetime

from execution.paper_broker import PaperBroker
from execution.broker_base import OrderSide, OrderType, ExecutionStatus
from execution.safety_guard import ExecutionSafetyGuard


# ============================================================
# 1. PaperBroker T+1 修复
# ============================================================

class TestPaperBrokerT1:
    def _buy(self, broker: PaperBroker, symbol: str = "600519.SH", shares: int = 1000, price: float = 10.0):
        return broker.send_order(symbol=symbol, side=OrderSide.BUY, shares=shares, price=price)

    def test_buy_creates_locked_position(self):
        """买入后: 总持仓增加, 可用股份保持为 0 (T+1 锁定)"""
        b = PaperBroker(initial_cash=100_000.0)
        b.connect()
        ord_res = self._buy(b)
        assert ord_res.status == ExecutionStatus.FILLED
        pos = b.get_positions()["600519.SH"]
        assert pos.total_shares == 1000
        assert pos.available_shares == 0  # 修复核心: 不再是永久锁死, 而是等待跨日解锁

    def test_sell_before_unlock_rejected(self):
        """跨日解锁前卖出必须被拒绝 (T+1 语义保持)"""
        b = PaperBroker(initial_cash=100_000.0)
        b.connect()
        self._buy(b)
        sell = b.send_order(symbol="600519.SH", side=OrderSide.SELL, shares=1000, price=10.5)
        assert sell.status == ExecutionStatus.REJECTED
        assert "可用股份不足" in sell.error_msg

    def test_unlock_then_sell_succeeds(self):
        """调用 unlock_t1_shares() 后可正常卖出 (修复: 此前该函数无任何调用点)"""
        b = PaperBroker(initial_cash=100_000.0)
        b.connect()
        self._buy(b)
        b.unlock_t1_shares()  # run_trader 现在在启动时调用
        sell = b.send_order(symbol="600519.SH", side=OrderSide.SELL, shares=1000, price=10.5)
        assert sell.status == ExecutionStatus.FILLED
        assert "600519.SH" not in b.get_positions()

    def test_unlock_does_not_free_same_day_buys(self):
        """带日期的解锁: 当日买入的股份必须保持锁定 (防止长驻进程误解锁)"""
        b = PaperBroker(initial_cash=100_000.0)
        b.connect()
        today = datetime.now().strftime("%Y-%m-%d")
        self._buy(b)
        b.unlock_t1_shares(today=today)  # 当日买入 → 不解锁
        assert b.get_positions()["600519.SH"].available_shares == 0
        b.unlock_t1_shares(today="2999-12-31")  # 次日 → 解锁
        assert b.get_positions()["600519.SH"].available_shares == 1000

    def test_unlock_frees_only_stale_buys_in_mixed_position(self):
        """混合持仓: 昨日买入的标的解锁, 当日买入的另一标的保持锁定"""
        b = PaperBroker(initial_cash=1_000_000.0)
        b.connect()
        today = datetime.now().strftime("%Y-%m-%d")
        self._buy(b, symbol="600519.SH")
        self._buy(b, symbol="000858.SZ")
        # 模拟 600519 是昨日买的
        b._last_buy_dates["600519.SH"] = "2000-01-01"
        b.unlock_t1_shares(today=today)
        assert b.get_positions()["600519.SH"].available_shares == 1000   # 昨日买入 → 解锁
        assert b.get_positions()["000858.SZ"].available_shares == 0      # 当日买入 → 锁定

    def test_full_rebalance_cycle_two_days(self):
        """端到端: 第 1 日买入 → 跨日解锁 → 第 2 日可调仓卖出 (修复前该周期必然卡死)"""
        b = PaperBroker(initial_cash=200_000.0)
        b.connect()
        # Day 1: 买入
        self._buy(b, shares=1000)
        assert b.send_order(symbol="600519.SH", side=OrderSide.SELL, shares=500, price=10.0).status == ExecutionStatus.REJECTED
        # 跨日
        b.unlock_t1_shares(today="2999-12-31")
        # Day 2: 减仓卖出成功, 剩余持仓仍可查询
        sell = b.send_order(symbol="600519.SH", side=OrderSide.SELL, shares=500, price=10.2)
        assert sell.status == ExecutionStatus.FILLED
        pos = b.get_positions()["600519.SH"]
        assert pos.total_shares == 500
        assert pos.available_shares == 500


# ============================================================
# 2. 防线 5 修复: 板块感知涨停阈值 + 偏离校验
# ============================================================

class TestSafetyGuardPriceAndLimit:
    def setup_method(self):
        self.guard = ExecutionSafetyGuard()

    # --- 涨停拦截: 主板 10% ---
    def test_main_board_buy_at_limit_blocked(self):
        ok, msg = self.guard.validate_price_and_limit("600519.SH", "BUY", 11.0, 11.0, pre_close_price=10.0)
        assert not ok and "涨停" in msg

    def test_main_board_buy_below_limit_allowed(self):
        ok, msg = self.guard.validate_price_and_limit("600519.SH", "BUY", 10.9, 10.9, pre_close_price=10.0)
        assert ok

    # --- 涨停拦截: 创业板/科创板 20% ---
    def test_chinext_board_threshold(self):
        ok, _ = self.guard.validate_price_and_limit("300750.SZ", "BUY", 11.9, 11.9, pre_close_price=10.0)
        assert ok, "19% 涨幅在 20% 板不应拦截"
        ok, msg = self.guard.validate_price_and_limit("688001.SH", "BUY", 12.0, 12.0, pre_close_price=10.0)
        assert not ok and "涨停" in msg, "20% 板 +20% 必须拦截"

    # --- 涨停拦截: 北交所 30% ---
    def test_bse_board_threshold(self):
        ok, _ = self.guard.validate_price_and_limit("832566.BJ", "BUY", 12.9, 12.9, pre_close_price=10.0)
        assert ok, "29% 涨幅在 30% 板不应拦截"
        ok, msg = self.guard.validate_price_and_limit("430047.BJ", "BUY", 13.0, 13.0, pre_close_price=10.0)
        assert not ok and "涨停" in msg, "30% 板 +30% 必须拦截"

    # --- 偏离度校验: 委托价与市价独立时真实生效 ---
    def test_price_deviation_blocked(self):
        ok, msg = self.guard.validate_price_and_limit("600519.SH", "BUY", 10.0, 10.5)
        assert not ok and "偏离" in msg

    def test_price_deviation_allowed_within_threshold(self):
        ok, _ = self.guard.validate_price_and_limit("600519.SH", "BUY", 10.0, 10.15)
        assert ok

    # --- 防线降级路径: 缺 pre_close 时放行 (不误杀) ---
    def test_buy_without_pre_close_allowed(self):
        ok, _ = self.guard.validate_price_and_limit("600519.SH", "BUY", 10.0, 10.0, pre_close_price=None)
        assert ok

    def test_sell_ignores_limit_check(self):
        ok, _ = self.guard.validate_price_and_limit("600519.SH", "SELL", 11.0, 11.0, pre_close_price=10.0)
        assert ok, "涨停拦截只约束买入侧, 卖出不受限"

    def test_invalid_price_rejected(self):
        ok, msg = self.guard.validate_price_and_limit("600519.SH", "BUY", 0.0, 10.0, pre_close_price=10.0)
        assert not ok


# ============================================================
# 3. daily_runner 分阶段重试
# ============================================================

class TestExecuteStageRetry:
    def test_success_first_try(self):
        from scheduler.daily_runner import _execute_stage
        calls = {"n": 0}
        def fn():
            calls["n"] += 1
            return "ok"
        assert _execute_stage("T", fn, retries=3, retry_delay=0.01) == "ok"
        assert calls["n"] == 1

    def test_retry_then_success(self):
        from scheduler.daily_runner import _execute_stage
        calls = {"n": 0}
        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise IOError("transient network error")
            return "recovered"
        assert _execute_stage("T", fn, retries=3, retry_delay=0.01) == "recovered"
        assert calls["n"] == 3

    def test_final_failure_raises(self):
        from scheduler.daily_runner import _execute_stage
        calls = {"n": 0}
        def fn():
            calls["n"] += 1
            raise ValueError("permanent failure")
        with pytest.raises(RuntimeError, match="重试"):
            _execute_stage("T", fn, retries=2, retry_delay=0.01)
        assert calls["n"] == 3  # 1 次原始 + 2 次重试

    def test_retries_zero_means_single_attempt(self):
        from scheduler.daily_runner import _execute_stage
        calls = {"n": 0}
        def fn():
            calls["n"] += 1
            raise ValueError("boom")
        with pytest.raises(RuntimeError):
            _execute_stage("T", fn, retries=0, retry_delay=0.01)
        assert calls["n"] == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
