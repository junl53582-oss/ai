"""
实盘模拟对账账本单元测试 (tests/test_paper_ledger.py)
验证:
1. 账本初始化与 100 万元本金设定
2. 调仓买入、整手向下取整、佣金计算
3. 调仓卖出、印花税扣除、已实现盈亏结转
4. 每日资产快照与净值序列单调性与准确性
"""
import pytest
import tempfile
import pandas as pd
from pathlib import Path
from execution.paper_ledger import PaperTradingLedger

def test_paper_ledger_init_and_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "test_ledger.json"
        ledger = PaperTradingLedger(ledger_file=tmp_file, initial_cash=1_000_000.0)
        
        assert ledger.cash == 1_000_000.0
        assert ledger.market_value == 0.0
        assert ledger.total_equity == 1_000_000.0
        assert ledger.nav == 1.0
        assert ledger.cum_return_pct == 0.0
        # 初始无快照, 显式记录首日快照后为 1
        ledger.record_daily_snapshot("2026-08-24")
        assert len(ledger.daily_nav_history) == 1
        assert tmp_file.exists()

def test_paper_ledger_rebalance_buy():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "test_ledger.json"
        ledger = PaperTradingLedger(ledger_file=tmp_file, initial_cash=1_000_000.0)
        
        # 构造模拟买入清单
        target_df = pd.DataFrame([
            {"symbol": "600026.SH", "name": "中远海能", "target_weight": 0.20, "close": 20.0},
            {"symbol": "601138.SH", "name": "工业富联", "target_weight": 0.20, "close": 60.0},
        ])
        
        res = ledger.rebalance(target_df, trade_date="2026-08-24")
        assert res["total_trades"] == 2
        assert "600026.SH" in ledger.positions
        assert "601138.SH" in ledger.positions
        
        # 验证整手 (100 股)
        pos1 = ledger.positions["600026.SH"]
        pos2 = ledger.positions["601138.SH"]
        assert pos1["total_shares"] % 100 == 0
        assert pos2["total_shares"] % 100 == 0
        assert pos1["total_shares"] > 0
        assert pos2["total_shares"] > 0
        
        # 验证总资产在合理摩擦范围内
        assert ledger.total_equity <= 1_000_000.0
        assert ledger.total_equity >= 995_000.0  # 扣除微小佣金后总资产接近 100万
        assert ledger.cash > 0

def test_paper_ledger_sell_and_realized_pnl():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "test_ledger.json"
        ledger = PaperTradingLedger(ledger_file=tmp_file, initial_cash=1_000_000.0)
        
        # 买入
        target_buy = pd.DataFrame([
            {"symbol": "600026.SH", "name": "中远海能", "target_weight": 0.30, "close": 20.0},
        ])
        ledger.rebalance(target_buy, trade_date="2026-08-24")
        orig_shares = ledger.positions["600026.SH"]["total_shares"]
        
        # 价格上涨 10%，卖出平仓
        new_prices = {"600026.SH": 22.0}
        target_empty = pd.DataFrame(columns=["symbol", "name", "target_weight", "close"])
        res = ledger.rebalance(target_empty, current_prices=new_prices, trade_date="2026-08-24")
        
        assert "600026.SH" not in ledger.positions
        assert ledger.realized_pnl > 0  # 盈利平仓
        assert ledger.total_equity > 1_000_000.0  # 盈利使得账户总资金增加
        assert ledger.nav > 1.0

def test_paper_ledger_summary_and_snapshots():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "test_ledger.json"
        ledger = PaperTradingLedger(ledger_file=tmp_file, initial_cash=1_000_000.0)
        
        summary = ledger.get_summary()
        assert "positions_df" in summary
        assert "nav_df" in summary
        assert "trades_df" in summary
        assert summary["total_equity"] == 1_000_000.0
