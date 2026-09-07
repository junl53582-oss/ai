"""
Stage C 定向测试: 真实交易日历、模拟盘记账与前瞻结算严肃化
(tests/test_stage_c_trading_calendar_and_settlement.py)

验证范围:
1. 权威日历仅承认物理 canonical_calendar_v1.parquet 内部交易日，超出覆盖范围或周末严格拦截
2. 交易日记账正常，快照真实累加
3. 结算跨度穿透计算 (trading_days_between 真实穿透，禁止虚标与任意整数覆盖)
4. 历史账本被污染记录已正确物理隔离至 quarantine 且 active observed_trading_days 为 0
5. 空白账本初始化不自动打周末快照
"""
import json
import pytest
from pathlib import Path
import pandas as pd

from config.settings import settings
from data.trading_calendar import CanonicalTradingCalendar, TradeDateError
from execution.paper_ledger import PaperTradingLedger, ShadowTradingLedger
from scripts.phase21h_prospective_runner import (
    validate_trading_calendar,
    settle_after_holding_period,
)


class TestStageCTradingCalendarAndSettlement:

    def test_canonical_trading_calendar_verifies_real_exchange_days(self):
        """1. 权威日历正确识别物理交易日与超出物理覆盖范围/周末的日期 (去推断化)"""
        cal = CanonicalTradingCalendar.get_instance()
        assert cal.is_trading_day("2026-08-21") is True   # 周五 物理交易日
        assert cal.is_trading_day("2026-08-24") is True   # 周一 物理最后交易日
        assert cal.is_trading_day("2026-08-25") is False  # 超出物理覆盖 (截至 2026-08-24)，fail-closed
        assert cal.is_trading_day("2026-09-04") is False  # 超出物理覆盖，fail-closed
        assert cal.is_trading_day("2026-09-05") is False  # 周六 休市且超出覆盖
        assert cal.is_trading_day("2026-09-06") is False  # 周日 休市且超出覆盖
        assert cal.is_trading_day("2026-09-25") is False  # 中秋节且超出覆盖
        assert cal.is_trading_day("2026-10-01") is False  # 国庆节且超出覆盖

        # 验证物理覆盖状态
        assert cal.check_coverage("2026-08-24") == "CALENDAR_COVERAGE_VALID"
        assert cal.check_coverage("2026-08-25") == "CALENDAR_COVERAGE_BLOCKED"

    def test_fresh_ledger_initialization_does_not_auto_snapshot(self, tmp_path):
        """2. 空白账本初始化绝不自动记录快照 (杜绝非交易日初始化污染账本)"""
        p_file = tmp_path / "fresh_paper.json"
        ledger = PaperTradingLedger(ledger_file=p_file)
        assert len(ledger.daily_nav_history) == 0
        assert ledger.observed_trading_days == 0
        assert ledger.get_summary()["status"] == "NOT_STARTED"

    def test_weekend_snapshot_intercepted_fail_closed(self, tmp_path):
        """3. 周末尝试打快照被硬拦截 (raise TradeDateError)"""
        p_file = tmp_path / "paper.json"
        ledger = PaperTradingLedger(ledger_file=p_file)

        with pytest.raises(TradeDateError, match="非交易所交易日严禁记录资产净值快照"):
            ledger.record_daily_snapshot("2026-09-05")

        with pytest.raises(TradeDateError, match="非交易所交易日严禁记录资产净值快照"):
            ledger.record_daily_snapshot("2026-09-06")

        with pytest.raises(TradeDateError, match="必须显式指定 trade_date"):
            ledger.record_daily_snapshot("")

        # 账本仍应保持为空
        assert len(ledger.daily_nav_history) == 0

    def test_trading_day_snapshot_recorded_normally(self, tmp_path):
        """4. 正常物理交易日快照记录正常，observed_trading_days 正确累加"""
        p_file = tmp_path / "paper.json"
        ledger = PaperTradingLedger(ledger_file=p_file)

        ledger.record_daily_snapshot("2026-08-21")  # 周五 (物理有效交易日)
        assert len(ledger.daily_nav_history) == 1
        assert ledger.observed_trading_days == 1

        ledger.record_daily_snapshot("2026-08-24")  # 下周一 (物理有效交易日)
        assert len(ledger.daily_nav_history) == 2
        assert ledger.observed_trading_days == 2

    def test_rebalance_enforces_trading_day(self, tmp_path):
        """5. 调仓撮合强制检查交易日"""
        p_file = tmp_path / "paper.json"
        ledger = PaperTradingLedger(ledger_file=p_file)
        target_df = pd.DataFrame([
            {"symbol": "600000.SH", "name": "浦发银行", "target_weight": 0.5, "close": 10.0}
        ])

        with pytest.raises(TradeDateError, match="非交易所交易日禁止执行调仓仿真"):
            ledger.rebalance(target_df, trade_date="2026-09-05")

        # 正常物理交易日成功调仓
        res = ledger.rebalance(target_df, trade_date="2026-08-24")
        assert res["total_trades"] == 1
        assert len(ledger.daily_nav_history) == 1
        assert ledger.daily_nav_history[0]["date"] == "2026-08-24"

    def test_settle_after_holding_period_trading_days_span_calculation(self, tmp_path):
        """6. 前瞻结算穿透计算实际跨越交易日数，拒绝整数虚标与天数不足"""
        ledger_file = tmp_path / "PROSPECTIVE_LEDGER.jsonl"

        # 2026-08-17 (周一) 到 2026-08-24 (周一):
        # 实际物理交易日序列 (排除建仓日): 08-18(二), 08-19(三), 08-20(四), 08-21(五), 08-24(一) -> 共 5 个真实持有交易日
        prices_df = pd.DataFrame([{"close": 10.5}])

        # 要求 5 天持有期，实际刚好 5 天 -> 结算成功
        rec = settle_after_holding_period(
            trade_date="2026-08-17",
            settle_date="2026-08-24",
            realized_prices_df=prices_df,
            min_holding_days=5,
            ledger_file=ledger_file,
            allow_historical=True
        )
        assert rec["holding_days_observed"] == 5
        assert rec["trade_date"] == "2026-08-17"
        assert rec["settle_date"] == "2026-08-24"

        # 若要求 6 天持有期，但实际仅 5 天 -> fail-closed 抛错
        with pytest.raises(ValueError, match="实际观察交易日天数不足: 实际 5 天 < 要求的持有期 6 天"):
            settle_after_holding_period(
                trade_date="2026-08-17",
                settle_date="2026-08-24",
                realized_prices_df=prices_df,
                min_holding_days=6,
                ledger_file=ledger_file,
                allow_historical=True
            )

    def test_settle_rejects_weekend_or_holiday(self, tmp_path):
        """7. 结算日期为周末或未覆盖日时 fail-closed"""
        ledger_file = tmp_path / "PROSPECTIVE_LEDGER.jsonl"
        prices_df = pd.DataFrame([{"close": 10.5}])

        # trade_date 为周六 (非合法交易日)
        with pytest.raises(ValueError, match="不是交易所合法交易日"):
            settle_after_holding_period(
                trade_date="2026-09-05",
                settle_date="2026-09-10",
                realized_prices_df=prices_df,
                ledger_file=ledger_file
            )

        # settle_date 为周日
        with pytest.raises(ValueError, match="不是交易所合法交易日"):
            settle_after_holding_period(
                trade_date="2026-08-24",
                settle_date="2026-09-06",
                realized_prices_df=prices_df,
                ledger_file=ledger_file,
                allow_historical=True
            )

        # settle_date 超出物理覆盖 (截至 2026-08-24)
        with pytest.raises(ValueError, match="不是交易所合法交易日"):
            settle_after_holding_period(
                trade_date="2026-08-24",
                settle_date="2026-09-25",
                realized_prices_df=prices_df,
                ledger_file=ledger_file,
                allow_historical=True
            )

    def test_shadow_ledger_rejects_weekend_observation(self, tmp_path):
        """8. 影子账本严禁在周末记录观察"""
        shadow_file = tmp_path / "shadow.json"
        shadow = ShadowTradingLedger(ledger_file=shadow_file)
        target_df = pd.DataFrame([
            {"symbol": "000001.SZ", "target_weight": 1.0, "close": 10.0}
        ])

        with pytest.raises(TradeDateError, match="非交易所交易日禁止记录影子观察"):
            shadow.record_shadow_observation(
                target_df=target_df,
                model_id="TEST_MODEL",
                data_date="2026-09-05"  # 周六
            )

        obs = shadow.record_shadow_observation(
            target_df=target_df,
            model_id="TEST_MODEL",
            data_date="2026-08-24"  # 周一 物理有效交易日
        )
        assert obs["date"] == "2026-08-24"
        assert shadow.observed_trading_days == 1

    def test_historical_contaminated_records_quarantined_and_excluded(self):
        """9. 验证生产账本已干净重置，且 2026-09-05/06 历史错误记录已隔离至 quarantine 目录"""
        real_ledger_file = Path(settings.DATA_DIR) / "paper_trading_ledger.json"
        assert real_ledger_file.exists()

        # 检查隔离产物
        quarantine_file = Path(settings.BASE_DIR) / "reports" / "accounting_quarantine" / "paper_ledger_invalid_weekend_20260905.json"
        assert quarantine_file.exists(), "隔离账本文件必须存在"
        q_data = json.loads(quarantine_file.read_text(encoding="utf-8"))
        quarantined = [s for s in q_data.get("daily_nav_history", []) if s.get("status") == "INVALID_NON_TRADING_DAY"]
        assert len(quarantined) == 2
        for s in quarantined:
            assert s["excluded_from_evidence"] is True
            assert s["date"] in ("2026-09-05", "2026-09-06")

        # 载入 PaperTradingLedger 实例，验证 active observed_trading_days 绝对为 0
        ledger = PaperTradingLedger(ledger_file=real_ledger_file)
        assert ledger.observed_trading_days == 0
        assert ledger.get_summary()["observed_trading_days"] == 0
        assert ledger.get_summary()["evidence_maturity"] == "IMMATURE"
