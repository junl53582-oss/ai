"""
Unit and integration tests for Stage 4:
- Deterministic Manifest reconciliation against physical parquet
- check_committed_dataset_schema two-way integrity gate
- validate_research_artifacts and validate_certification_artifacts passes
- Paper Trading vs Shadow Trading segregated accounting and evidence hash
- Phase 2.1-H runner fail-closed on historical backfills and calendar absence
"""
import sys
import json
import hashlib
import pytest
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tools.rebuild_manifest_from_physical import compute_file_sha256, rebuild_manifest_for_parquet
from tools.check_committed_dataset_schema import verify_manifest_consistency
from execution.paper_ledger import PaperTradingLedger, ShadowTradingLedger
from scripts.phase21h_prospective_runner import (
    validate_trading_calendar,
    seal_inputs,
    predict_ex_ante,
    settle_after_holding_period,
)


def test_manifest_matches_physical_parquet():
    """Verify factor_matrix_300.manifest.json is 100% truthful to physical file."""
    parquet_path = root_dir / "data_storage" / "research" / "factor_matrix_300.parquet"
    manifest_path = root_dir / "data_storage" / "research" / "factor_matrix_300.manifest.json"

    assert parquet_path.exists(), "factor_matrix_300.parquet must exist"
    assert manifest_path.exists(), "factor_matrix_300.manifest.json must exist"

    df = pd.read_parquet(parquet_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    physical_sha = compute_file_sha256(parquet_path)

    # 1. SHA256 matches
    assert manifest["file_sha256"].lower() == physical_sha.lower()

    # 2. Row count matches exactly
    assert manifest["row_count"] == len(df)
    assert manifest["row_count"] == 349379

    # 3. Symbol count matches
    assert manifest["symbol_count"] == df["symbol"].nunique()
    assert manifest["symbol_count"] == 300

    # 4. Date range matches
    dt_series = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    assert manifest["date_min"] == dt_series.min()
    assert manifest["date_max"] == dt_series.max()
    assert manifest["date_min"] == "2021-09-29"
    assert manifest["date_max"] == "2026-08-24"

    # 5. Feature count matches
    assert manifest["feature_count"] == len(df.columns)
    assert manifest["feature_count"] == 126


def test_verify_manifest_consistency_detects_tamper(tmp_path):
    """Verify verify_manifest_consistency fails-closed if row_count or hash is tampered."""
    df = pd.DataFrame({
        "date": ["2026-09-01", "2026-09-02"],
        "symbol": ["000001.SZ", "000002.SZ"],
        "close": [10.0, 11.0]
    })
    p_path = tmp_path / "test_data.parquet"
    m_path = tmp_path / "test_data.manifest.json"
    df.to_parquet(p_path, index=False)
    actual_sha = hashlib.sha256(p_path.read_bytes()).hexdigest().lower()

    # Valid manifest
    valid_manifest = {
        "file_sha256": actual_sha,
        "row_count": 2,
        "symbol_count": 2,
        "date_range": ["2026-09-01", "2026-09-02"],
        "feature_count": 3
    }
    m_path.write_text(json.dumps(valid_manifest), encoding="utf-8")
    assert verify_manifest_consistency(p_path, df) is True

    # Tampered hash
    tampered_manifest = valid_manifest.copy()
    tampered_manifest["file_sha256"] = "0" * 64
    m_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    assert verify_manifest_consistency(p_path, df) is False

    # Tampered row count
    tampered_rows = valid_manifest.copy()
    tampered_rows["row_count"] = 999
    m_path.write_text(json.dumps(tampered_rows), encoding="utf-8")
    assert verify_manifest_consistency(p_path, df) is False


def test_validate_research_artifacts_script():
    """Verify validate_research_artifacts.py passes 100%."""
    from tools.validate_research_artifacts import validate_artifacts
    report_dir = root_dir / "reports" / "production_research"
    assert validate_artifacts(report_dir, mode="production") is True


def test_validate_certification_artifacts_script():
    """Verify validate_certification_artifacts.py passes 100%."""
    from tools.validate_certification_artifacts import validate_artifacts
    assert validate_artifacts() is True


def test_paper_and_shadow_ledgers_segregated(tmp_path):
    """Verify Paper and Shadow ledgers maintain independent accounting and schema."""
    paper_file = tmp_path / "paper_ledger.json"
    shadow_file = tmp_path / "shadow_ledger.json"

    paper = PaperTradingLedger(ledger_file=paper_file, initial_cash=1000000.0)
    shadow = ShadowTradingLedger(ledger_file=shadow_file, initial_cash=1000000.0)

    # Verify separate accounts
    assert paper_file.exists()
    assert shadow_file.exists()
    assert paper.ledger_file != shadow.ledger_file

    target_df = pd.DataFrame([
        {"symbol": "000001.SZ", "name": "平安银行", "target_weight": 0.5, "close": 12.0},
        {"symbol": "600000.SH", "name": "浦发银行", "target_weight": 0.5, "close": 8.0},
    ])

    market_df = pd.DataFrame([
        {"symbol": "000001.SZ", "is_suspended": False, "is_limit_up_locked": False, "amount": 5e7},
        {"symbol": "600000.SH", "is_suspended": True, "is_limit_up_locked": False, "amount": 5e7},
    ])

    obs = shadow.record_shadow_observation(
        target_df=target_df,
        model_id="TEST_PRODUCTION_MODEL",
        data_date="2026-08-24",
        market_df=market_df
    )

    assert obs["model_id"] == "TEST_PRODUCTION_MODEL"
    assert obs["date"] == "2026-08-24"
    assert len(obs["shadow_fills"]) == 2

    # Verify tradability detection: 600000.SH is suspended, so shadow fill is 0
    f1 = next(f for f in obs["shadow_fills"] if f["symbol"] == "000001.SZ")
    f2 = next(f for f in obs["shadow_fills"] if f["symbol"] == "600000.SH")
    assert f1["tradable"] is True
    assert f1["shadow_fill_shares"] > 0
    assert f2["tradable"] is False
    assert f2["shadow_fill_shares"] == 0
    assert f2["untradable_reason"] == "SUSPENDED"

    # Verify cryptographic hash presence
    assert len(obs["evidence_sha256"]) == 64

    # Verify maturity is IMMATURE when observed days < 20
    summary = shadow.get_summary()
    assert summary["evidence_maturity"] == "IMMATURE"
    assert summary["observed_trading_days"] == 1


def test_phase21h_prospective_runner_fail_closed_on_historical_backfill():
    """Verify prospective runner rejects backfilling historical dates."""
    with pytest.raises(ValueError, match="禁止回填历史日期"):
        validate_trading_calendar("2026-08-20")

    with pytest.raises(ValueError, match="禁止回填历史日期"):
        seal_inputs("2026-08-24")

    # Settlement before or on trade date is rejected
    with pytest.raises(ValueError, match="结算日期 .* 必须晚于前瞻预测日期"):
        settle_after_holding_period(
            trade_date="2026-09-01",
            settle_date="2026-09-01",
            realized_prices_df=pd.DataFrame([{"close": 10.0}])
        )
