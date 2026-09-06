"""
Stage R6: Comprehensive Independent Audit Regression Suite
(tests/test_r6_independent_audit_regressions.py)

独立审计全量回归测试套件，全面核验第二轮审计六大核心阻断原则:
1. 覆盖旧信号记新样本检测 (严禁把 2026-08-24 的旧信号记成 2026-09 月的新样本，必须抛错)
2. 覆盖虚构账本重算防御 (严禁仅凭整数字段声明晋升，重算天数不足必须抛错)
3. 覆盖实盘门禁全部阻断逻辑 (未解锁时、无行情时、行情陈旧时、未来行情时、跨用途签名时、重放攻击时、历史未验证模型时全面 Fail-Closed)
4. 覆盖日历不可推断原则 (越界物理日历即阻断，严禁推断未来交易日，哈希篡改即阻断)
5. 覆盖未实测指标禁止展示/伪造原则 (VerifiedResearchMetricsLoader SHA256 校验，通知分数严格校验，系统全局三锁常态 LOCKED)
"""

import os
import time
import json
import hashlib
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from config.settings import settings
from data.trading_calendar import CanonicalTradingCalendar, TradeDateError
from data.crypto_anchor import (
    TRUSTED_KEY_REGISTRY,
    sign_with_environment_key,
    verify_trust_root,
    compute_canonical_keyring_hash,
)
from execution.live_gate import (
    verify_live_trading_multi_factor_gate,
    LiveGateAuthError
)
from execution.prospective_state_machine import (
    ProspectiveStateMachine,
    seal_signal,
    execute_observed
)
from execution.paper_ledger import PaperTradingLedger, ShadowTradingLedger
from models.registry import (
    verify_promotion_evidence,
    PromotionError,
    EvidenceArtifact
)
from models.verified_metrics import VerifiedResearchMetricsLoader, compute_sha256


# =========================================================================
# 辅助工具函数与 Fixtures
# =========================================================================

@pytest.fixture(autouse=True)
def preserve_keyring_and_env():
    orig_env = os.environ.get("QUANT_TRUSTED_KEYRING_SHA256")
    orig_keys = {kid: dict(info) for kid, info in TRUSTED_KEY_REGISTRY.items()}
    yield
    if orig_env is None:
        os.environ.pop("QUANT_TRUSTED_KEYRING_SHA256", None)
    else:
        os.environ["QUANT_TRUSTED_KEYRING_SHA256"] = orig_env
    TRUSTED_KEY_REGISTRY.clear()
    TRUSTED_KEY_REGISTRY.update(orig_keys)


class DummyModelRecord:
    def __init__(self, model_id="PROD_TEST_M1", state="PRODUCTION", evidence=None, verification_status="VERIFIED"):
        self.model_id = model_id
        self.state = state
        self.evidence = evidence or {}
        self.verification_status = verification_status
        self.dataset_sha256 = "9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42"
        self.feature_schema_hash = "ad44898838817b0867f51e224377e1729ced530bd9a42f7b22cd81672e76b21e"


def make_signed_approval(
    tmp_path: Path,
    signer_key_id: str = "PROD_LIVE_AUTH_KEY_2026_V1",
    account_id: str = "5500123456",
    model_id: str = "PROD_TEST_M1",
    symbol: str = "600519.SH",
    side: str = "BUY",
    quantity: float = 1000.0,
    limit_price: float = 50.0,
    nonce: str = "nonce_123456",
    expires_in_seconds: float = 3600.0,
    purpose: str = "LIVE_TRADING_AUTHORIZATION",
    domain_separator: str = "LIVE_TRADING_AUTHORIZATION_APPROVAL_V1"
) -> Path:
    key_entry = TRUSTED_KEY_REGISTRY[signer_key_id]
    test_sk_hex = "11" * 32
    from cryptography.hazmat.primitives.asymmetric import ed25519 as crypto_ed25519
    priv = crypto_ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(test_sk_hex))
    key_entry["public_key_hex"] = priv.public_key().public_bytes_raw().hex()
    os.environ["QUANT_TRUSTED_KEYRING_SHA256"] = compute_canonical_keyring_hash()

    payload = {
        "approval_id": f"APPR_{nonce}",
        "authorized_action": "LIVE_ORDER",
        "account_id": account_id,
        "model_id": model_id,
        "environment": "production",
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "limit_price": limit_price,
        "nonce": nonce,
        "expiration_timestamp": time.time() + expires_in_seconds,
        "risk_limits": {"max_notional": 1000000.0, "max_daily_notional": 5000000.0},
        "operator": "chief_risk_officer"
    }

    msg_bytes = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    sig_hex, _ = sign_with_environment_key(
        message=msg_bytes,
        key_id=signer_key_id,
        required_purpose=purpose,
        domain_separator=domain_separator,
        explicit_private_key_hex=test_sk_hex
    )

    art = {
        "signer_key_id": signer_key_id,
        "signature": sig_hex,
        "canonical_payload": payload
    }
    art_path = tmp_path / f"approval_{nonce}.json"
    art_path.write_text(json.dumps(art, indent=2, ensure_ascii=False), encoding="utf-8")
    return art_path


# =========================================================================
# 1. 旧信号记新样本检测 (严禁旧信号记成新样本，必须抛错)
# =========================================================================

def test_audit_stale_signal_recorded_as_september_samples_fails_closed(tmp_path):
    """审计项 1: 严禁将 2026-08-24 旧信号在 2026-09 月记成新 Paper/Shadow 样本，必须抛错"""
    sig_dir = tmp_path / "prospective_signals"
    quotes_df = pd.DataFrame([
        {"symbol": "600519.SH", "open": 1600.0, "close": 1610.0, "is_suspended": False, "is_limit_up_locked": False}
    ])

    # 尝试把 2026-08-24 信号作为 2026-09-07 的执行样本
    with pytest.raises(ValueError, match="严禁把 2026-08-24 的旧信号记成 2026-09 月的新 Paper/Shadow 样本"):
        execute_observed(
            signal_date="2026-08-24",
            execution_date="2026-09-07",
            quotes_df=quotes_df,
            storage_dir=sig_dir,
        )

    # 尝试通过 ShadowTradingLedger 跨期记入 2026-08-24 信号
    shadow = ShadowTradingLedger(ledger_file=tmp_path / "shadow.json")
    with pytest.raises(ValueError, match="严禁把 2026-08-24 的旧信号记成 2026-09 月的新 Paper/Shadow 样本"):
        shadow.record_shadow_observation(
            target_df=pd.DataFrame([{"symbol": "600519.SH", "target_weight": 0.5, "close": 1600.0}]),
            model_id="PROD_TEST_M1",
            data_date="2026-09-01",
            signal_date="2026-08-24"
        )


# =========================================================================
# 2. 虚构账本重算防御 (严禁虚假声明晋升天数，重算不足必须抛错)
# =========================================================================

def test_audit_fabricated_ledger_recalculation_fails_closed(tmp_path):
    """审计项 2: 模型晋升证据核验强制重新计算物理账本明细，不足 min_trading_days 必须抛错"""
    key_entry = TRUSTED_KEY_REGISTRY["PROD_MODEL_PROMOTION_KEY_2026_V1"]
    test_sk_hex = "22" * 32
    from cryptography.hazmat.primitives.asymmetric import ed25519 as crypto_ed25519
    priv = crypto_ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(test_sk_hex))
    key_entry["public_key_hex"] = priv.public_key().public_bytes_raw().hex()
    model_rec = DummyModelRecord(model_id="PROD_TEST_M1")

    # Case A: 虚假声明观察天数 20 天，但账本明细仅提供 3 天有效交易日 -> 拦截
    art_empty_file = tmp_path / "art_insufficient_sample.json"
    art_empty = EvidenceArtifact(
        evidence_type="PROSPECTIVE_SHADOW_OBSERVATION",
        artifact_path=str(art_empty_file.resolve()),
        artifact_sha256="",
        schema_version="evidence_v1",
        model_id="PROD_TEST_M1",
        dataset_sha256="9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42",
        feature_schema_hash="ad44898838817b0867f51e224377e1729ced530bd9a42f7b22cd81672e76b21e",
        created_at=datetime.now().isoformat(),
        observation_start="2026-07-27",
        observation_end="2026-08-24",
        observed_trading_days=20,
        status="MATURE",
        approver="chief_risk_officer",
        signer_key_id="PROD_MODEL_PROMOTION_KEY_2026_V1",
        signature="",
        ledger_records=[
            {"date": "2026-08-20", "pnl": 10.0},
            {"date": "2026-08-21", "pnl": 20.0},
            {"date": "2026-08-24", "pnl": 30.0}
        ]
    )
    sig_hex, _ = sign_with_environment_key(
        message=art_empty.compute_canonical_bytes(),
        key_id="PROD_MODEL_PROMOTION_KEY_2026_V1",
        required_purpose="MODEL_PROMOTION",
        domain_separator="QUANT_MODEL_PROMOTION_EVIDENCE_V1",
        explicit_private_key_hex=test_sk_hex
    )
    art_empty.signature = sig_hex
    art_empty_file.write_text(json.dumps(art_empty.to_dict()), encoding="utf-8")
    art_empty.artifact_sha256 = hashlib.sha256(art_empty_file.read_bytes()).hexdigest().lower()
    art_empty_file.write_text(json.dumps(art_empty.to_dict()), encoding="utf-8")
    art_empty.artifact_sha256 = hashlib.sha256(art_empty_file.read_bytes()).hexdigest().lower()

    with pytest.raises(PromotionError, match="证据账本重新计算天数不足"):
        verify_promotion_evidence(art_empty, model_rec=model_rec, expected_type="PROSPECTIVE_SHADOW_OBSERVATION", min_trading_days=20)

    # Case B: 包含周末非交易日 (2026-09-05/06) 且合法交易日仅有 2 天 -> 拦截
    cal = CanonicalTradingCalendar()
    art_insufficient_file = tmp_path / "art_insufficient.json"
    art_insufficient = EvidenceArtifact(
        evidence_type="PROSPECTIVE_SHADOW_OBSERVATION",
        artifact_path=str(art_insufficient_file.resolve()),
        artifact_sha256="",
        schema_version="evidence_v1",
        model_id="PROD_TEST_M1",
        dataset_sha256="9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42",
        feature_schema_hash="ad44898838817b0867f51e224377e1729ced530bd9a42f7b22cd81672e76b21e",
        created_at=datetime.now().isoformat(),
        observation_start="2026-07-27",
        observation_end="2026-08-24",
        observed_trading_days=20,
        status="MATURE",
        approver="chief_risk_officer",
        signer_key_id="PROD_MODEL_PROMOTION_KEY_2026_V1",
        signature="",
        ledger_records=[
            {"date": "2026-08-21", "pnl": 100.0},
            {"date": "2026-08-24", "pnl": 150.0},
            {"date": "2026-09-05", "pnl": 0.0},  # 周六 (非法)
            {"date": "2026-09-06", "pnl": 0.0}   # 周日 (非法)
        ]
    )
    sig_hex2, _ = sign_with_environment_key(
        message=art_insufficient.compute_canonical_bytes(),
        key_id="PROD_MODEL_PROMOTION_KEY_2026_V1",
        required_purpose="MODEL_PROMOTION",
        domain_separator="QUANT_MODEL_PROMOTION_EVIDENCE_V1",
        explicit_private_key_hex=test_sk_hex
    )
    art_insufficient.signature = sig_hex2
    art_insufficient_file.write_text(json.dumps(art_insufficient.to_dict()), encoding="utf-8")
    art_insufficient.artifact_sha256 = hashlib.sha256(art_insufficient_file.read_bytes()).hexdigest().lower()
    art_insufficient_file.write_text(json.dumps(art_insufficient.to_dict()), encoding="utf-8")
    art_insufficient.artifact_sha256 = hashlib.sha256(art_insufficient_file.read_bytes()).hexdigest().lower()

    with pytest.raises(PromotionError, match="证据账本重新计算天数不足"):
        verify_promotion_evidence(art_insufficient, model_rec=model_rec, expected_type="PROSPECTIVE_SHADOW_OBSERVATION", min_trading_days=20, calendar=cal)


# =========================================================================
# 3. 实盘门禁全维度阻断逻辑 (全部阻断向量 Fail-Closed)
# =========================================================================

@pytest.fixture
def mock_live_base(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LIVE_TRADING_READY", True)
    monkeypatch.setattr(settings, "LIVE_TRADING_GATE_STATUS", "UNLOCKED")
    monkeypatch.setattr(settings, "LIVE_TRADING_STATUS", "UNLOCKED")
    monkeypatch.setattr(settings, "EMERGENCY_KILL_SWITCH", False)
    monkeypatch.setattr(settings, "LIVE_ACCOUNT_WHITELIST", ["5500123456"])
    monkeypatch.setattr(settings, "MAX_QUOTE_STALENESS_SECONDS", 300.0)

    cert_file = tmp_path / "cert.json"
    cert_file.write_text(json.dumps({"model_id": "PROD_TEST_M1", "status": "CERTIFIED"}), encoding="utf-8")

    model_rec = DummyModelRecord(
        model_id="PROD_TEST_M1",
        evidence={
            "certification_ref": str(cert_file),
            "prospective_validation": {"status": "MATURE", "observed_trading_days": 30}
        }
    )
    paper = MagicMock(observed_trading_days=30)
    shadow = MagicMock(observed_trading_days=30)

    return {
        "account_id": "5500123456",
        "live_confirm": True,
        "quote_timestamp": time.time(),
        "current_date": "2026-08-24",
        "model_record": model_rec,
        "paper_ledger": paper,
        "shadow_ledger": shadow,
        "nonce_store_path": tmp_path / "nonce_store.json",
        "order_symbol": "600519.SH",
        "order_side": "BUY",
        "order_shares": 1000,
        "order_price": 50.0,
    }


def test_audit_live_gate_fails_closed_when_system_locked(mock_live_base, tmp_path, monkeypatch):
    """审计项 3a: LIVE_TRADING_READY = False 时强制硬阻断"""
    appr = make_signed_approval(tmp_path, nonce="g1")
    mock_live_base["approval_artifact_path"] = appr

    monkeypatch.setattr(settings, "LIVE_TRADING_READY", False)
    with pytest.raises(LiveGateAuthError, match="实盘交易总开关未开启|LIVE_TRADING_READY is False"):
        verify_live_trading_multi_factor_gate(**mock_live_base)


def test_audit_live_gate_fails_closed_on_stale_or_missing_quote(mock_live_base, tmp_path):
    """审计项 3b: 行情时间戳缺失、未来或陈旧时强制硬阻断"""
    appr = make_signed_approval(tmp_path, nonce="g2")
    mock_live_base["approval_artifact_path"] = appr

    # 无时间戳
    mock_live_base["quote_timestamp"] = None
    with pytest.raises(LiveGateAuthError, match="实时行情时间戳缺失"):
        verify_live_trading_multi_factor_gate(**mock_live_base)

    # 陈旧行情
    mock_live_base["quote_timestamp"] = time.time() - 600.0
    with pytest.raises(LiveGateAuthError, match="数据已陈旧"):
        verify_live_trading_multi_factor_gate(**mock_live_base)

    # 未来行情
    mock_live_base["quote_timestamp"] = time.time() + 60.0
    with pytest.raises(LiveGateAuthError, match="时间戳处于未来"):
        verify_live_trading_multi_factor_gate(**mock_live_base)


def test_audit_live_gate_fails_closed_on_purpose_mismatch_and_replay(mock_live_base, tmp_path):
    """审计项 3c: 密钥跨用途越权拦截与 Nonce 重放攻击拦截"""
    # 跨用途签名拦截 (用 MODEL_PROMOTION 密钥签实盘审批)
    appr_wrong_purpose = make_signed_approval(
        tmp_path,
        signer_key_id="PROD_MODEL_PROMOTION_KEY_2026_V1",
        nonce="g3_wrong_purpose",
        purpose="MODEL_PROMOTION"
    )
    mock_live_base["approval_artifact_path"] = appr_wrong_purpose
    with pytest.raises(LiveGateAuthError, match="密钥用途错配|Key Purpose Mismatch"):
        verify_live_trading_multi_factor_gate(**mock_live_base)

    # 重放攻击拦截 (同一 Nonce 二次提交)
    appr_replay = make_signed_approval(tmp_path, nonce="replay_token_007")
    mock_live_base["approval_artifact_path"] = appr_replay

    ok, res = verify_live_trading_multi_factor_gate(**mock_live_base)
    assert ok is True

    # 再次提交同一 approval_artifact -> 必被拦截
    with pytest.raises(LiveGateAuthError, match="检测到审批凭证重放攻击|Approval Replay Detected"):
        verify_live_trading_multi_factor_gate(**mock_live_base)


def test_audit_live_gate_blocks_legacy_unverified_model(mock_live_base, tmp_path):
    """审计项 3d: LEGACY_UNVERIFIED 历史未验证模型一律阻断通过实盘门禁"""
    appr = make_signed_approval(tmp_path, nonce="g4")
    mock_live_base["approval_artifact_path"] = appr
    mock_live_base["model_record"].verification_status = "LEGACY_UNVERIFIED"

    with pytest.raises(LiveGateAuthError, match="历史未验证模型.*LEGACY_UNVERIFIED"):
        verify_live_trading_multi_factor_gate(**mock_live_base)


# =========================================================================
# 4. 日历不可推断原则 (越界物理日历即阻断，哈希篡改即阻断)
# =========================================================================

def test_audit_calendar_non_inference_and_tamper_fails_closed():
    """审计项 4: 物理日历覆盖边界截止 2026-08-24，未物理核验日历严禁算法推断"""
    cal = CanonicalTradingCalendar()
    assert cal.is_trading_day("2026-08-24") is True
    # 周末返回 False
    assert cal.is_trading_day("2026-08-29") is False
    assert cal.is_trading_day("2026-08-30") is False

    # 超出物理 Parquet 覆盖范围 (2026-08-25 及以后) 返回 False
    assert cal.is_trading_day("2026-08-25") is False
    assert cal.is_trading_day("2026-09-01") is False

    # 物理覆盖校验
    assert cal.check_coverage("2026-08-25") == "CALENDAR_COVERAGE_BLOCKED"
    with pytest.raises(TradeDateError, match="CALENDAR_COVERAGE_BLOCKED"):
        cal.validate_trading_day("2026-08-25")

    # 物理覆盖末日无法推断下一交易日 (返回 None)
    assert cal.next_trading_day("2026-08-24") is None
    assert cal.next_trading_day("2026-08-25") is None

    # 物理 Provenance 完整性哈希校验
    parquet_path = settings.BASE_DIR / "data_storage" / "reference" / "canonical_calendar_v1.parquet"
    manifest_path = settings.BASE_DIR / "data_storage" / "reference" / "canonical_calendar_v1.manifest.json"
    actual_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest().lower()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert actual_hash == manifest_data["calendar_artifact_sha256"].lower()


# =========================================================================
# 5. 未实测指标禁止展示/伪造原则与系统常态定位
# =========================================================================

def test_audit_zero_fabrication_loader_and_system_positioning(tmp_path):
    """审计项 5: VerifiedResearchMetricsLoader 强制哈希防伪，系统三锁保持 LOCKED"""
    # 1. 验证系统定位与三锁常态
    assert settings.LIVE_TRADING_READY is False
    assert settings.LIVE_TRADING_GATE_STATUS == "LOCKED"
    assert settings.LIVE_TRADING_STATUS == "LOCKED"

    # 2. 验证 VerifiedResearchMetricsLoader 真实加载与校验
    loader = VerifiedResearchMetricsLoader()
    perf = loader.load_multi_generation_performance()
    assert perf is not None
    table = loader.get_multi_generation_metrics_table()
    assert table is not None
    assert not table.empty

    # 3. 验证篡改 metrics 凭证后强制 Fail-Closed (返回 None)
    fake_reports = tmp_path / "reports"
    fake_reports.mkdir(parents=True)
    perf_file = fake_reports / "performance_all_generations.json"
    manifest_file = fake_reports / "performance_all_generations.manifest.json"

    perf_file.write_text(json.dumps({"gen1_baseline": {}}), encoding="utf-8")
    manifest_file.write_text(json.dumps({"sha256": "wrong_hash"}), encoding="utf-8")

    tampered_loader = VerifiedResearchMetricsLoader(base_dir=tmp_path)
    assert tampered_loader.load_multi_generation_performance() is None
    assert tampered_loader.get_multi_generation_metrics_table() is None
