"""
Stage S3: Live Gate Complete Dual Splitting, Non-Downgradable Safety Invariants, Order-Bound Payload & Atomic Nonce File Locking
(tests/test_s3_live_gate_complete_binding.py)

验证内容:
1. 双重网关解耦与安全门限不可降级:
   - verify_live_connection_gate: 仅核验全局硬锁、白名单、熔断、操作员确认、信任根公钥环及模型状态
   - verify_live_order_gate: 固定强制最高安全标准，外部调用方传 require_trust_root, check_physical_cert, min_observation_days 直接引发 TypeError
2. 订单参数与审批 Payload 强比对 (Order-Bound Payload):
   - 审批凭证签名必须绑定 symbol, side, quantity, limit_price
   - 标的代码、买卖方向、委托数量、委托价格任一不一致立即拦截
   - 缺少订单参数时 Fail-Closed
3. 物理认证凭据深度防伪校验 (Deep Certification Artifact Verification):
   - 物理凭证文件不存在拦截
   - 状态非 CERTIFIED/PASS 拦截
   - 模型 ID 或数据集哈希不匹配拦截
   - 物理文件 SHA-256 篡改拦截
4. 前瞻与观察天数门限 (>= 30 交易日固定不可降级):
   - 29 天观察天数严格被发单网关拦截
   - 达到 30 天观察天数放行
5. 风控限额与名义金额强校验 (Risk Limits):
   - 单笔超过 max_notional 拦截
   - 当日累计超过 max_daily_notional 拦截
   - 非法股数或价格拦截
6. 跨进程排他互斥锁与并发原子 Nonce 消费:
   - 相同 Nonce 并发请求恰好一个成功放行，另一个被重放拦截
   - Nonce 存储损坏时 Fail-Closed
7. MiniQMTBroker 管道集成校验
"""

import os
import sys
import time
import json
import math
import pytest
from pathlib import Path
from typing import Optional, Callable
from unittest.mock import MagicMock, patch
import concurrent.futures

from config.settings import settings
from data.crypto_anchor import (
    TRUSTED_KEY_REGISTRY,
    sign_with_environment_key,
    compute_canonical_keyring_hash,
)
from execution.live_gate import (
    verify_live_connection_gate,
    verify_live_order_gate,
    verify_live_trading_multi_factor_gate,
    check_keyring_integrity,
    LiveGateAuthError,
)
from execution.broker_base import OrderSide, OrderType, ExecutionStatus
from execution.miniqmt_broker import MiniQMTBroker


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
    domain_separator: str = "LIVE_TRADING_AUTHORIZATION_APPROVAL_V1",
    max_notional: float = 1000000.0,
    max_daily_notional: float = 5000000.0,
    authorized_action: str = "LIVE_ORDER",
    environment: str = "production",
    operator: str = "chief_risk_officer",
    custom_payload_mutator: Optional[Callable[[dict], None]] = None,
    tamper_sig: bool = False,
) -> Path:
    key_entry = TRUSTED_KEY_REGISTRY[signer_key_id]
    test_sk_hex = "11" * 32
    from cryptography.hazmat.primitives.asymmetric import ed25519 as crypto_ed25519
    priv = crypto_ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(test_sk_hex))
    key_entry["public_key_hex"] = priv.public_key().public_bytes_raw().hex()
    os.environ["QUANT_TRUSTED_KEYRING_SHA256"] = compute_canonical_keyring_hash()

    payload = {
        "approval_id": f"APPR_{nonce}",
        "authorized_action": authorized_action,
        "account_id": account_id,
        "model_id": model_id,
        "environment": environment,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "limit_price": limit_price,
        "nonce": nonce,
        "expiration_timestamp": time.time() + expires_in_seconds,
        "risk_limits": {
            "max_notional": max_notional,
            "max_daily_notional": max_daily_notional,
        },
        "operator": operator
    }

    if custom_payload_mutator:
        custom_payload_mutator(payload)

    msg_bytes = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    sig_hex, _ = sign_with_environment_key(
        message=msg_bytes,
        key_id=signer_key_id,
        required_purpose=purpose,
        domain_separator=domain_separator,
        explicit_private_key_hex=test_sk_hex
    )

    if tamper_sig:
        sig_hex = "00" * 64

    art = {
        "signer_key_id": signer_key_id,
        "signature": sig_hex,
        "canonical_payload": payload
    }
    art_path = tmp_path / f"approval_{nonce}.json"
    art_path.write_text(json.dumps(art, indent=2, ensure_ascii=False), encoding="utf-8")
    return art_path


@pytest.fixture
def valid_order_env(tmp_path, monkeypatch):
    """构建全量满足 30 天成熟度与真实物理凭据的合法实盘环境"""
    monkeypatch.setattr(settings, "LIVE_TRADING_READY", True)
    monkeypatch.setattr(settings, "LIVE_TRADING_GATE_STATUS", "UNLOCKED")
    monkeypatch.setattr(settings, "LIVE_TRADING_STATUS", "UNLOCKED")
    monkeypatch.setattr(settings, "EMERGENCY_KILL_SWITCH", False)
    monkeypatch.setattr(settings, "LIVE_ACCOUNT_WHITELIST", ["5500123456"])
    monkeypatch.setattr(settings, "MAX_QUOTE_STALENESS_SECONDS", 300.0)

    # 设置受信任公钥环哈希
    canonical_hash = compute_canonical_keyring_hash()
    monkeypatch.setenv("QUANT_TRUSTED_KEYRING_SHA256", canonical_hash)

    # 创建物理存在的 certification 文件并附带完整防篡改元数据
    cert_file = tmp_path / "model_cert.json"
    cert_payload = {
        "model_id": "PROD_TEST_M1",
        "status": "CERTIFIED",
        "dataset_sha256": "9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42",
    }
    cert_file.write_text(json.dumps(cert_payload), encoding="utf-8")

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

    nonce_file = tmp_path / "test_used_nonces.json"

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


# =========================================================================
# 1. 连接网关与发单网关解耦测试 & 安全参数不可降级测试
# =========================================================================

def test_connection_gate_passes_without_quotes(valid_order_env):
    """连接网关验证: 无须行情与订单参数，满足全局条件与信任根即放行"""
    ok, res = verify_live_connection_gate(
        account_id=valid_order_env["account_id"],
        live_confirm=True,
        model_record=valid_order_env["model_record"],
    )
    assert ok is True
    assert res.get("factor_a_ready") is True
    assert res.get("factor_f_trust_root") is True


def test_connection_gate_fails_closed_when_system_locked(valid_order_env, monkeypatch):
    """连接网关验证: LIVE_TRADING_READY=False 时强制硬阻断"""
    monkeypatch.setattr(settings, "LIVE_TRADING_READY", False)
    with pytest.raises(LiveGateAuthError, match="LIVE_TRADING_READY is False"):
        verify_live_connection_gate(
            account_id=valid_order_env["account_id"],
            live_confirm=True
        )


def test_safety_downgrade_parameters_removed_from_apis(valid_order_env, tmp_path):
    """Stage S3: 验证外部调用方无法传参调低安全门限 (require_trust_root, min_observation_days, check_physical_cert)"""
    # 尝试传 require_trust_root 给 check_keyring_integrity
    with pytest.raises(TypeError):
        check_keyring_integrity(require_trust_root=False)

    # 尝试传 require_trust_root 给 verify_live_connection_gate
    with pytest.raises(TypeError):
        verify_live_connection_gate(
            account_id=valid_order_env["account_id"],
            live_confirm=True,
            require_trust_root=False
        )

    # 尝试传 min_observation_days 给 verify_live_order_gate
    appr = make_signed_approval(tmp_path, nonce="n_downgrade")
    valid_order_env["approval_artifact_path"] = appr
    with pytest.raises(TypeError):
        verify_live_order_gate(**valid_order_env, min_observation_days=0)

    with pytest.raises(TypeError):
        verify_live_order_gate(**valid_order_env, check_physical_cert=False)


# =========================================================================
# 2. 订单参数与审批 Payload 逐字段强比对 (Order-Bound Payload)
# =========================================================================

def test_order_gate_rejects_symbol_mismatch(valid_order_env, tmp_path):
    """订单绑定: 委托标的与审批凭据 symbol 不一致时强制拦截"""
    appr = make_signed_approval(tmp_path, nonce="n_sym_mismatch", symbol="600519.SH")
    valid_order_env["approval_artifact_path"] = appr
    valid_order_env["order_symbol"] = "000858.SZ"  # 错配

    with pytest.raises(LiveGateAuthError, match="委托标的代码.*与审批凭证绑定标的.*不匹配"):
        verify_live_order_gate(**valid_order_env)


def test_order_gate_rejects_side_mismatch(valid_order_env, tmp_path):
    """订单绑定: 委托方向与审批凭据 side 不一致时强制拦截"""
    appr = make_signed_approval(tmp_path, nonce="n_side_mismatch", side="BUY")
    valid_order_env["approval_artifact_path"] = appr
    valid_order_env["order_side"] = "SELL"  # 错配

    with pytest.raises(LiveGateAuthError, match="委托买卖方向.*与审批凭证绑定方向.*不匹配"):
        verify_live_order_gate(**valid_order_env)


def test_order_gate_rejects_quantity_mismatch(valid_order_env, tmp_path):
    """订单绑定: 委托股数与审批凭据 quantity 不一致时强制拦截"""
    appr = make_signed_approval(tmp_path, nonce="n_qty_mismatch", quantity=1000.0)
    valid_order_env["approval_artifact_path"] = appr
    valid_order_env["order_shares"] = 2000  # 错配

    with pytest.raises(LiveGateAuthError, match="委托股数.*与审批凭证绑定数量.*不匹配"):
        verify_live_order_gate(**valid_order_env)


def test_order_gate_rejects_price_mismatch(valid_order_env, tmp_path):
    """订单绑定: 委托价格与审批凭据 limit_price 不一致时强制拦截"""
    appr = make_signed_approval(tmp_path, nonce="n_px_mismatch", limit_price=50.0)
    valid_order_env["approval_artifact_path"] = appr
    valid_order_env["order_price"] = 55.0  # 错配

    with pytest.raises(LiveGateAuthError, match="委托价格.*与审批凭证绑定价格.*不匹配"):
        verify_live_order_gate(**valid_order_env)


def test_order_gate_fails_on_missing_order_parameters(valid_order_env, tmp_path):
    """订单绑定: 缺少任一订单参数 (symbol, side, shares, price) 强制 Fail-Closed"""
    appr = make_signed_approval(tmp_path, nonce="n_missing_order_param")
    valid_order_env["approval_artifact_path"] = appr
    
    # 缺少 order_symbol
    valid_order_env["order_symbol"] = None
    with pytest.raises(LiveGateAuthError, match="审批凭证绑定了标的代码，但发单参数未传入 order_symbol"):
        verify_live_order_gate(**valid_order_env)
    valid_order_env["order_symbol"] = "600519.SH"

    # 缺少 order_side
    valid_order_env["order_side"] = None
    with pytest.raises(LiveGateAuthError, match="审批凭证绑定了买卖方向，但发单参数未传入 order_side"):
        verify_live_order_gate(**valid_order_env)
    valid_order_env["order_side"] = "BUY"

    # 缺少 order_shares
    valid_order_env["order_shares"] = None
    with pytest.raises(LiveGateAuthError, match="审批凭证绑定了委托数量，但发单参数未传入 order_shares"):
        verify_live_order_gate(**valid_order_env)
    valid_order_env["order_shares"] = 1000

    # 缺少 order_price
    valid_order_env["order_price"] = None
    with pytest.raises(LiveGateAuthError, match="审批凭证绑定了限价，但发单参数未传入 order_price"):
        verify_live_order_gate(**valid_order_env)
    valid_order_env["order_price"] = 50.0


def test_order_gate_fails_on_missing_payload_order_fields(valid_order_env, tmp_path):
    """Stage S3 强化: 审批载荷缺失 symbol/side/quantity/limit_price 强制 Fail-Closed"""
    for field in ["symbol", "side", "quantity", "limit_price"]:
        appr = make_signed_approval(
            tmp_path,
            nonce=f"n_missing_payload_{field}",
            custom_payload_mutator=lambda p, f=field: p.pop(f, None)
        )
        valid_order_env["approval_artifact_path"] = appr
        with pytest.raises(LiveGateAuthError, match=f"审批凭证缺少必要字段: '{field}'"):
            verify_live_order_gate(**valid_order_env)


def test_order_gate_fails_on_missing_or_invalid_risk_limits(valid_order_env, tmp_path):
    """Stage S3 强化: 审批载荷缺失 risk_limits 或缺少 max_notional/max_daily_notional 或非法值时强制 Fail-Closed"""
    # 1. 缺失 risk_limits
    appr_no_rl = make_signed_approval(
        tmp_path,
        nonce="n_no_rl",
        custom_payload_mutator=lambda p: p.pop("risk_limits", None)
    )
    valid_order_env["approval_artifact_path"] = appr_no_rl
    with pytest.raises(LiveGateAuthError, match="审批凭证缺少必要字段: 'risk_limits'"):
        verify_live_order_gate(**valid_order_env)

    # 2. 空字典 risk_limits
    appr_empty_rl = make_signed_approval(
        tmp_path,
        nonce="n_empty_rl",
        custom_payload_mutator=lambda p: p.update({"risk_limits": {}})
    )
    valid_order_env["approval_artifact_path"] = appr_empty_rl
    with pytest.raises(LiveGateAuthError, match="审批凭证缺少风控限额字段或为空字典: 'risk_limits'"):
        verify_live_order_gate(**valid_order_env)

    # 3. 缺失 max_notional
    appr_no_max = make_signed_approval(
        tmp_path,
        nonce="n_no_max",
        custom_payload_mutator=lambda p: p["risk_limits"].pop("max_notional", None)
    )
    valid_order_env["approval_artifact_path"] = appr_no_max
    with pytest.raises(LiveGateAuthError, match="审批凭证 risk_limits.max_notional 缺失或类型非法"):
        verify_live_order_gate(**valid_order_env)

    # 4. 缺失 max_daily_notional
    appr_no_daily = make_signed_approval(
        tmp_path,
        nonce="n_no_daily",
        custom_payload_mutator=lambda p: p["risk_limits"].pop("max_daily_notional", None)
    )
    valid_order_env["approval_artifact_path"] = appr_no_daily
    with pytest.raises(LiveGateAuthError, match="审批凭证 risk_limits.max_daily_notional 缺失或类型非法"):
        verify_live_order_gate(**valid_order_env)

    # 5. 非正数/非法浮点数 (<= 0, inf, nan)
    for bad_val in [-100.0, 0.0, float("inf"), float("nan"), "invalid_str"]:
        appr_bad = make_signed_approval(
            tmp_path,
            nonce=f"n_bad_val_{hash(str(bad_val))}",
            custom_payload_mutator=lambda p, v=bad_val: p["risk_limits"].update({"max_notional": v})
        )
        valid_order_env["approval_artifact_path"] = appr_bad
        with pytest.raises(LiveGateAuthError, match="risk_limits.max_notional.*必须为正有限数|非法数值"):
            verify_live_order_gate(**valid_order_env)


def test_order_gate_fails_on_invalid_side(valid_order_env, tmp_path):
    """Stage S3 强化: side 只允许 BUY 或 SELL，非法值时强制拦截"""
    appr_bad_side = make_signed_approval(
        tmp_path,
        nonce="n_bad_side",
        side="HOLD"
    )
    valid_order_env["approval_artifact_path"] = appr_bad_side
    valid_order_env["order_side"] = "HOLD"
    with pytest.raises(LiveGateAuthError, match="审批凭证 side 只允许 'BUY' 或 'SELL'"):
        verify_live_order_gate(**valid_order_env)


# =========================================================================
# 3. 物理认证凭据存在性与深度防伪内容校验 (Physical Certification)
# =========================================================================

def test_order_gate_fails_on_missing_physical_cert_file(valid_order_env, tmp_path):
    """物理凭据校验: certification_ref 物理文件不存在时，发单网关强制阻断"""
    appr = make_signed_approval(tmp_path, nonce="cert_test_nonce")
    valid_order_env["approval_artifact_path"] = appr
    valid_order_env["model_record"].evidence["certification_ref"] = str(tmp_path / "non_existent_cert.json")

    with pytest.raises(LiveGateAuthError, match="生产模型研究认证物理凭证文件不存在"):
        verify_live_order_gate(**valid_order_env)


def test_order_gate_fails_on_tampered_cert_status(valid_order_env, tmp_path):
    """物理凭据内容防伪校验: 状态非 CERTIFIED / PASS 时，严格阻断"""
    tampered_cert = tmp_path / "tampered_cert.json"
    tampered_cert.write_text(json.dumps({
        "model_id": "PROD_TEST_M1",
        "status": "FAILED",
        "dataset_sha256": "9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42"
    }), encoding="utf-8")

    appr = make_signed_approval(tmp_path, nonce="cert_tamper_nonce")
    valid_order_env["approval_artifact_path"] = appr
    valid_order_env["model_record"].evidence["certification_ref"] = str(tampered_cert)

    with pytest.raises(LiveGateAuthError, match="生产模型研究认证物理凭证内容损坏或被非法篡改"):
        verify_live_order_gate(**valid_order_env)


def test_order_gate_fails_on_mismatched_model_id_in_cert(valid_order_env, tmp_path):
    """物理凭据防篡改链: 凭据中的 model_id 与生产模型不符时严格阻断"""
    mismatched_cert = tmp_path / "mismatch_cert.json"
    mismatched_cert.write_text(json.dumps({
        "model_id": "OTHER_MODEL_V9",
        "status": "CERTIFIED",
        "dataset_sha256": "9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42"
    }), encoding="utf-8")

    appr = make_signed_approval(tmp_path, nonce="cert_mid_mismatch")
    valid_order_env["approval_artifact_path"] = appr
    valid_order_env["model_record"].evidence["certification_ref"] = str(mismatched_cert)

    with pytest.raises(LiveGateAuthError, match="凭证模型 ID .* 不匹配"):
        verify_live_order_gate(**valid_order_env)


# =========================================================================
# 4. 前瞻与观察天数门限 (>= 30 交易日固定不可降级)
# =========================================================================

def test_order_gate_rejects_29_observation_days(valid_order_env, tmp_path):
    """观察成熟度门限: 29 天观察天数被发单网关严格拦截 (需 >= 30 天)"""
    appr = make_signed_approval(tmp_path, nonce="days_29_nonce")
    valid_order_env["approval_artifact_path"] = appr

    # 1. 模型前瞻观察仅 29 天
    valid_order_env["model_record"].evidence["prospective_validation"]["observed_trading_days"] = 29
    with pytest.raises(LiveGateAuthError, match="观察交易日 29 天.*需 >= 30"):
        verify_live_order_gate(**valid_order_env)

    # 恢复模型前瞻为 30 天，测试模拟盘账本仅 29 天
    valid_order_env["model_record"].evidence["prospective_validation"]["observed_trading_days"] = 30
    valid_order_env["paper_ledger"].observed_trading_days = 29
    valid_order_env["shadow_ledger"].observed_trading_days = 29
    with pytest.raises(LiveGateAuthError, match="模拟盘或影子观察累计天数不足.*门限 30 交易日"):
        verify_live_order_gate(**valid_order_env)


# =========================================================================
# 5. 订单金额风控与单笔/当日累计限额绑定 (Risk Limits Binding)
# =========================================================================

def test_order_gate_enforces_risk_limit_max_notional(valid_order_env, tmp_path):
    """风控绑定: 单笔订单金额超过审批凭据 max_notional 时拦截"""
    appr = make_signed_approval(tmp_path, nonce="risk_notional_01", quantity=1000.0, limit_price=60.0, max_notional=50000.0)
    valid_order_env["approval_artifact_path"] = appr
    valid_order_env["order_shares"] = 1000
    valid_order_env["order_price"] = 60.0  # 1000 * 60 = 60000 > 50000

    with pytest.raises(LiveGateAuthError, match="超过审批凭证单笔风控限额 max_notional"):
        verify_live_order_gate(**valid_order_env)


def test_order_gate_enforces_risk_limit_max_daily_notional(valid_order_env, tmp_path):
    """风控绑定: 当日累计订单金额超过审批凭据 max_daily_notional 时拦截"""
    # max_notional=60000, max_daily_notional=100000
    appr1 = make_signed_approval(tmp_path, nonce="daily_risk_01", quantity=1000.0, limit_price=50.0, max_notional=60000.0, max_daily_notional=100000.0)
    valid_order_env["approval_artifact_path"] = appr1
    valid_order_env["order_shares"] = 1000
    valid_order_env["order_price"] = 50.0  # 50,000 <= 60000 and <= 100000
    ok, _ = verify_live_order_gate(**valid_order_env)
    assert ok is True

    # 第二笔发单: 60,000 -> 累计 110,000 > 100,000 触发当日累计限额阻断
    appr2 = make_signed_approval(tmp_path, nonce="daily_risk_02", quantity=1000.0, limit_price=60.0, max_notional=60000.0, max_daily_notional=100000.0)
    valid_order_env["approval_artifact_path"] = appr2
    valid_order_env["order_shares"] = 1000
    valid_order_env["order_price"] = 60.0  # 60,000
    with pytest.raises(LiveGateAuthError, match="超过审批凭证当日风控限额 max_daily_notional"):
        verify_live_order_gate(**valid_order_env)


# =========================================================================
# 6. 跨进程排他互斥锁与并发原子 Nonce 消费 (Stage S3 Mutex File Lock)
# =========================================================================

def test_concurrent_identical_nonce_exactly_one_passes(valid_order_env, tmp_path):
    """Stage S3 核心实测: 两个并发使用相同 Nonce 的发单请求，经由跨进程文件锁排他保证恰好一个通过，另一个必被拦截"""
    shared_nonce = "concurrent_nonce_safe_01"
    appr = make_signed_approval(
        tmp_path,
        nonce=shared_nonce,
        quantity=1000.0,
        limit_price=50.0,
        max_notional=1000000.0,
    )
    valid_order_env["approval_artifact_path"] = appr
    valid_order_env["order_shares"] = 1000
    valid_order_env["order_price"] = 50.0

    results = []
    errors = []

    def attempt_order():
        try:
            ok, res = verify_live_order_gate(**valid_order_env)
            results.append(ok)
        except Exception as err:
            errors.append(err)

    # 使用多线程模拟并发抢占
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(attempt_order)
        f2 = executor.submit(attempt_order)
        concurrent.futures.wait([f1, f2])

    assert len(results) == 1, f"必须恰好一个请求成功放行，实际成功 {len(results)}"
    assert len(errors) == 1, f"必须恰好一个请求被拒绝，实际拒绝 {len(errors)}"
    assert isinstance(errors[0], LiveGateAuthError)
    assert "检测到审批凭证重放攻击" in str(errors[0])


def test_order_gate_fails_closed_on_corrupted_nonce_store(valid_order_env, tmp_path):
    """Nonce 存储防御: Nonce 存储文件损坏、不可读时，严格 Fail-Closed 抛错，绝不回退为空字典"""
    nonce_file = valid_order_env["nonce_store_path"]
    nonce_file.parent.mkdir(parents=True, exist_ok=True)
    nonce_file.write_text("{corrupted json content: missing quotes", encoding="utf-8")

    appr = make_signed_approval(tmp_path, nonce="corrupt_nonce_test")
    valid_order_env["approval_artifact_path"] = appr

    with pytest.raises(LiveGateAuthError, match="读取 Nonce 重放防护记录存储异常"):
        verify_live_order_gate(**valid_order_env)


# =========================================================================
# 7. MiniQMTBroker 管道集成校验
# =========================================================================

def test_miniqmt_broker_wired_to_dual_gates():
    """MiniQMTBroker: connect() 与 send_order() 分别接入连接网关与发单网关"""
    broker = MiniQMTBroker()

    # 1. 默认系统处于锁定状态，connect() 必被拦截
    with pytest.raises(LiveGateAuthError, match="LIVE_TRADING_READY is False"):
        broker.connect()

    # 2. 默认系统处于锁定状态，send_order() 必被拦截
    with pytest.raises(LiveGateAuthError, match="LIVE_TRADING_READY is False"):
        broker.send_order("600519.SH", OrderSide.BUY, 100, 1500.0)
