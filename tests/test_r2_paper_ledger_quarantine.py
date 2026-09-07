import json
import hashlib
import pytest
from pathlib import Path

from config.settings import settings
from execution.paper_ledger import PaperTradingLedger, TradeDateError


def test_active_paper_ledger_is_clean_and_uncontaminated():
    """验证当前在役的 active 模拟盘账本已彻底归零重置为干净状态"""
    ledger_path = settings.DATA_DIR / "paper_trading_ledger.json"
    assert ledger_path.exists(), "Active paper ledger must exist"

    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert data["initial_cash"] == 1_000_000.0
    assert data["cash"] == 1_000_000.0
    assert data["positions"] == {}
    assert data["trade_history"] == []
    assert data["daily_nav_history"] == []
    assert data["observed_trading_days"] == 0
    assert data["evidence_maturity"] == "IMMATURE"
    assert data["status"] == "NOT_STARTED"

    # 载入该账本不应触发任何异常
    ledger = PaperTradingLedger(ledger_file=ledger_path)
    assert ledger.observed_trading_days == 0
    assert ledger.market_value == 0.0
    assert ledger.total_equity == 1_000_000.0


def test_quarantined_ledger_and_manifest_integrity():
    """验证被隔离的周末污染账本与审计清单的完整性与哈希一致性"""
    quarantine_dir = settings.BASE_DIR / "reports" / "accounting_quarantine"
    quarantined_json = quarantine_dir / "paper_ledger_invalid_weekend_20260905.json"
    quarantined_manifest = quarantine_dir / "paper_ledger_invalid_weekend_20260905.manifest.json"

    assert quarantined_json.exists(), "Quarantined ledger json must exist"
    assert quarantined_manifest.exists(), "Quarantined manifest must exist"

    manifest = json.loads(quarantined_manifest.read_text(encoding="utf-8"))
    assert manifest["excluded_from_all_evidence"] is True
    assert "2026-09-05" in manifest["invalid_trade_dates"]
    assert len(manifest["invalid_trade_ids"]) == 8

    # 校验隔离文件的实际 sha256 与清单记录一致
    actual_hash = hashlib.sha256(quarantined_json.read_bytes()).hexdigest()
    assert actual_hash == manifest["original_sha256"]


def test_loading_contaminated_ledger_fails_closed(tmp_path):
    """验证尝试加载包含非交易日成交的脏账本时强制 Fail-Closed 抛出 TradeDateError"""
    dirty_ledger_file = tmp_path / "dirty_ledger.json"
    dirty_data = {
        "account_id": "PAPER_SIM_DIRTY",
        "initial_cash": 1000000.0,
        "cash": 800000.0,
        "positions": {},
        "trade_history": [
            {
                "trade_id": "bad_trade_01",
                "timestamp": "2026-09-05 10:00:00",  # 2026-09-05 为周六
                "action": "BUY",
                "symbol": "600026.SH",
                "shares": 1000,
                "price": 20.0
            }
        ],
        "daily_nav_history": []
    }
    dirty_ledger_file.write_text(json.dumps(dirty_data), encoding="utf-8")

    # 尝试加载脏账本必须 Fail-Closed
    with pytest.raises(TradeDateError, match="non-trading day"):
        PaperTradingLedger(ledger_file=dirty_ledger_file)


def test_quarantine_helper_produces_valid_manifest(tmp_path):
    """验证隔离工具方法能够正确迁移脏账本并输出合法清单"""
    dirty_file = tmp_path / "test_dirty.json"
    dirty_file.write_text(json.dumps({
        "trade_history": [
            {"trade_id": "t1", "timestamp": "2026-09-05 09:30:00"}
        ]
    }), encoding="utf-8")

    out_dir = tmp_path / "quarantine_out"
    manifest = PaperTradingLedger.quarantine_contaminated_ledger(
        src_ledger_file=dirty_file,
        quarantine_dir=out_dir,
        reason="Test quarantine"
    )
    assert manifest["excluded_from_all_evidence"] is True
    assert "t1" in manifest["invalid_trade_ids"]
    assert "2026-09-05" in manifest["invalid_trade_dates"]

    mf_file = out_dir / "paper_ledger_invalid_weekend_20260905.manifest.json"
    assert mf_file.exists()
