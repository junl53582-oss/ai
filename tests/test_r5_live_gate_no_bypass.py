"""
Stage R5: Comprehensive Live Gate Bypass Defense & Hardening Verification
(tests/test_r5_live_gate_no_bypass.py)

验证:
1. 行情新鲜度与委托校验硬化:
   - 拒绝 None、无效/无法解析的时间戳
   - 拒绝未来时间戳 (> 5秒)
   - 拒绝过时时间戳 (> 300秒)
   - 严禁任何绕过机制
2. 密钥用途与信任根隔离:
   - 严禁跨用途签名 (MODEL_PROMOTION 与 LIVE_TRADING_AUTHORIZATION 严格隔离)
   - 严禁 RUNTIME_ATTESTATION 密钥签发实盘审批
   - 信任根校验 (verify_trust_root) 被篡改时 fail-closed
3. 历史未验证模型硬阻断:
   - LEGACY_UNVERIFIED 模型永远无法通过实盘门禁
4. 实盘授权 Payload 深度绑定与重放防护:
   - 账户绑定不符时 fail-closed
   - 模型绑定不符时 fail-closed
   - 过期时间戳 fail-closed
   - 重放攻击 (同一 Nonce 二次提交) 必须拦截并抛错
"""

import os
import time
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from config.settings import settings
from data.crypto_anchor import (
    TRUSTED_KEY_REGISTRY,
    sign_with_environment_key,
    verify_trust_root,
    compute_canonical_keyring_hash
)
from execution.live_gate import verify_live_trading_multi_factor_gate, LiveGateAuthError
from models.registry import verify_promotion_evidence, PromotionError, EvidenceArtifact


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


@pytest.fixture
def base_live_env(tmp_path, monkeypatch):
    """构建基础的通过环境配置"""
    monkeypatch.setattr(settings, "LIVE_TRADING_READY", True)
    monkeypatch.setattr(settings, "LIVE_TRADING_GATE_STATUS", "UNLOCKED")
    monkeypatch.setattr(settings, "LIVE_TRADING_STATUS", "UNLOCKED")
    monkeypatch.setattr(settings, "EMERGENCY_KILL_SWITCH", False)
    monkeypatch.setattr(settings, "LIVE_ACCOUNT_WHITELIST", ["5500123456"])
    monkeypatch.setattr(settings, "MAX_QUOTE_STALENESS_SECONDS", 300.0)

    canonical_hash = compute_canonical_keyring_hash()
    monkeypatch.setenv("QUANT_TRUSTED_KEYRING_SHA256", canonical_hash)

    cert_file = tmp_path / "cert.json"
    cert_file.write_text(json.dumps({"model_id": "PROD_TEST_M1", "status": "CERTIFIED"}), encoding="utf-8")

    model_rec = DummyModelRecord(
        model_id="PROD_TEST_M1",
        evidence={
            "certification_ref": str(cert_file),
            "prospective_validation": {"status": "MATURE", "observed_trading_days": 30}
        }
    )

    paper = MagicMock()
    paper.observed_trading_days = 30
    shadow = MagicMock()
    shadow.observed_trading_days = 30

    nonce_file = tmp_path / "test_nonces.json"

    return {
        "account_id": "5500123456",
        "live_confirm": True,
        "quote_timestamp": time.time(),
        "current_date": "2026-08-24",
        "model_record": model_rec,
        "paper_ledger": paper,
        "shadow_ledger": shadow,
        "nonce_store_path": nonce_file,
        "order_symbol": "600519.SH",
        "order_side": "BUY",
        "order_shares": 1000,
        "order_price": 50.0,
    }


def test_r5_quote_timestamp_none_fails_closed(base_live_env, tmp_path):
    """1. quote_timestamp 为 None 时强制 Fail-Closed"""
    appr = make_signed_approval(tmp_path, nonce="n1")
    base_live_env["approval_artifact_path"] = appr
    base_live_env["quote_timestamp"] = None

    with pytest.raises(LiveGateAuthError, match="实时行情时间戳缺失|quote_timestamp is None"):
        verify_live_trading_multi_factor_gate(**base_live_env)


def test_r5_quote_timestamp_future_fails_closed(base_live_env, tmp_path):
    """1b. quote_timestamp 为未来时间戳 (>5s) 时强制 Fail-Closed"""
    appr = make_signed_approval(tmp_path, nonce="n2")
    base_live_env["approval_artifact_path"] = appr
    base_live_env["quote_timestamp"] = time.time() + 60.0

    with pytest.raises(LiveGateAuthError, match="时间戳处于未来"):
        verify_live_trading_multi_factor_gate(**base_live_env)


def test_r5_quote_timestamp_stale_fails_closed(base_live_env, tmp_path):
    """1c. quote_timestamp 过期 (>300s) 时强制 Fail-Closed"""
    appr = make_signed_approval(tmp_path, nonce="n3")
    base_live_env["approval_artifact_path"] = appr
    base_live_env["quote_timestamp"] = time.time() - 350.0

    with pytest.raises(LiveGateAuthError, match="数据已陈旧.*允许上限"):
        verify_live_trading_multi_factor_gate(**base_live_env)


def test_r5_key_purpose_isolation_fails_closed(base_live_env, tmp_path):
    """2. 密钥用途严格隔离: PROD_RUNTIME_KEY 无法签发实盘授权 (缺少 LIVE_TRADING_AUTHORIZATION)"""
    # 构造用 PROD_RUNTIME_KEY_2026_V1 签名的凭据 (该密钥仅有 RUNTIME_ATTESTATION)
    appr = make_signed_approval(tmp_path, signer_key_id="PROD_RUNTIME_KEY_2026_V1", nonce="n4", purpose="RUNTIME_ATTESTATION")
    base_live_env["approval_artifact_path"] = appr

    with pytest.raises(LiveGateAuthError, match="密钥用途错配|Key Purpose Mismatch"):
        verify_live_trading_multi_factor_gate(**base_live_env)


def test_r5_trust_root_tampering_fails_closed(base_live_env, tmp_path, monkeypatch):
    """2b. 信任根外部 Pin 不匹配时强制 Fail-Closed"""
    appr = make_signed_approval(tmp_path, nonce="n5")
    base_live_env["approval_artifact_path"] = appr
    # 故意设置被篡改的外部信任根
    monkeypatch.setenv("QUANT_TRUSTED_KEYRING_SHA256", "0" * 64)

    with pytest.raises(LiveGateAuthError, match="密码学信任根校验失败"):
        verify_live_trading_multi_factor_gate(**base_live_env)


def test_r5_legacy_unverified_model_permanently_blocked(base_live_env, tmp_path):
    """3. LEGACY_UNVERIFIED 历史未验证模型永远被实盘门禁硬阻断"""
    appr = make_signed_approval(tmp_path, nonce="n6")
    base_live_env["approval_artifact_path"] = appr
    base_live_env["model_record"].verification_status = "LEGACY_UNVERIFIED"

    with pytest.raises(LiveGateAuthError, match="历史未验证模型.*LEGACY_UNVERIFIED"):
        verify_live_trading_multi_factor_gate(**base_live_env)


def test_r5_bound_payload_account_mismatch_fails_closed(base_live_env, tmp_path):
    """4. 凭证绑定账户与当前发起账户不符时 Fail-Closed"""
    appr = make_signed_approval(tmp_path, account_id="8888888888", nonce="n7")
    base_live_env["approval_artifact_path"] = appr

    with pytest.raises(LiveGateAuthError, match="审批凭证账户绑定错配"):
        verify_live_trading_multi_factor_gate(**base_live_env)


def test_r5_bound_payload_model_mismatch_fails_closed(base_live_env, tmp_path):
    """4b. 凭证绑定模型与生产模型不符时 Fail-Closed"""
    appr = make_signed_approval(tmp_path, model_id="OTHER_MODEL_V99", nonce="n8")
    base_live_env["approval_artifact_path"] = appr

    with pytest.raises(LiveGateAuthError, match="审批凭证模型绑定错配"):
        verify_live_trading_multi_factor_gate(**base_live_env)


def test_r5_bound_payload_expired_timestamp_fails_closed(base_live_env, tmp_path):
    """4c. 凭证已过期时 Fail-Closed"""
    appr = make_signed_approval(tmp_path, expires_in_seconds=-10.0, nonce="n9")
    base_live_env["approval_artifact_path"] = appr

    with pytest.raises(LiveGateAuthError, match="审批凭证已过期"):
        verify_live_trading_multi_factor_gate(**base_live_env)


def test_r5_replay_protection_blocks_duplicate_nonce(base_live_env, tmp_path):
    """4d. 重放防护: 同一 Nonce 再次提交时强制硬阻断"""
    appr = make_signed_approval(tmp_path, nonce="unique_nonce_001")
    base_live_env["approval_artifact_path"] = appr

    # 第一次使用合法凭证 -> 成功
    ok, res = verify_live_trading_multi_factor_gate(**base_live_env)
    assert ok is True

    # 第二次使用相同凭据发起实盘 -> 触发重放防御阻断
    with pytest.raises(LiveGateAuthError, match="检测到审批凭证重放攻击|Approval Replay Detected"):
        verify_live_trading_multi_factor_gate(**base_live_env)
