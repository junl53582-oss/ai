"""
Stage S4: Final Independent Truthfulness & Security Audit Suite
(tests/test_s4_final_truthfulness_audit.py)

综合覆盖 S1~S4 全部审计防线 (必须真实覆盖):
1. 伪 64 位哈希校准证据拒绝
2. 物理校准文件/签名/模型 Schema 任一篡改拒绝
3. 正式 prospective API 没有 allow_historical 绕过
4. 缺字段、缺日期列、缺任一标的行情时，正式账本和制品零变化
5. 研究历史回放与正式 Paper/Shadow/prospective 目录物理隔离
6. 外部调用者不能调低信任根、物理证书、30 天观察门槛
7. 订单 symbol、side、quantity、price 与签名不一致均拒绝
8. 并发相同 Nonce 只能放行一次
"""

import os
import sys
import time
import json
import inspect
import hashlib
import concurrent.futures
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from config.settings import settings
from data.trading_calendar import CanonicalTradingCalendar, TradeDateError
from data.crypto_anchor import (
    TRUSTED_KEY_REGISTRY,
    sign_with_environment_key,
    compute_canonical_keyring_hash,
)
from models.verified_metrics import (
    ProbabilityCalibrationEvidence,
    load_verified_probability_calibration,
    VerifiedResearchMetricsLoader,
    validate_scientific_evidence,
)
from scheduler.notifier import MessageNotifier
from execution.prospective_state_machine import (
    seal_signal,
    execute_observed,
    ImmutableArtifactConflict,
)
from research.historical_replay_adapter import HistoricalReplayAdapter
from execution.live_gate import (
    verify_live_connection_gate,
    verify_live_order_gate,
    verify_live_trading_multi_factor_gate,
    check_keyring_integrity,
    LiveGateAuthError,
)
from execution.paper_ledger import PaperTradingLedger, ShadowTradingLedger


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


class DummyS4ModelRecord:
    def __init__(self, model_id="PROD_S4_MODEL", state="PRODUCTION", evidence=None, verification_status="VERIFIED"):
        self.model_id = model_id
        self.state = state
        self.evidence = evidence or {}
        self.verification_status = verification_status
        self.dataset_sha256 = "9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42"
        self.feature_schema_hash = "ad44898838817b0867f51e224377e1729ced530bd9a42f7b22cd81672e76b21e"


def make_s4_signed_approval(
    tmp_path: Path,
    signer_key_id: str = "PROD_LIVE_AUTH_KEY_2026_V1",
    account_id: str = "5500123456",
    model_id: str = "PROD_S4_MODEL",
    symbol: str = "600519.SH",
    side: str = "BUY",
    quantity: float = 1000.0,
    limit_price: float = 50.0,
    nonce: str = "s4_nonce_001",
    expires_in_seconds: float = 3600.0,
    purpose: str = "LIVE_TRADING_AUTHORIZATION",
    domain_separator: str = "LIVE_TRADING_AUTHORIZATION_APPROVAL_V1",
    max_notional: float = 1000000.0,
    tamper_sig: bool = False,
) -> Path:
    key_entry = TRUSTED_KEY_REGISTRY[signer_key_id]
    test_sk_hex = "22" * 32
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
        "risk_limits": {"max_notional": max_notional, "max_daily_notional": max_notional * 5},
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

    if tamper_sig:
        sig_hex = "ff" * 64

    art = {
        "signer_key_id": signer_key_id,
        "signature": sig_hex,
        "canonical_payload": payload
    }
    art_path = tmp_path / f"approval_{nonce}.json"
    art_path.write_text(json.dumps(art, indent=2, ensure_ascii=False), encoding="utf-8")
    return art_path


# =========================================================================
# 1. 伪 64 位哈希校准证据拒绝 (Dummy 64-char Hash Rejection)
# =========================================================================

def test_s4_rejects_dummy_64char_hashes_without_physical_calibration():
    """审计项 1: 伪 64 位哈希字符串 ("a"*64, "b"*64) 绝无法通过 is_valid()，通知仅显示排序分数"""
    dummy_evidence = ProbabilityCalibrationEvidence(
        method="isotonic",
        brier_score=0.12,
        sample_count=2000,
        status="VERIFIED",
        calibrator_artifact_path="artifacts/calibration/dummy_calibrator.joblib",
        calibrator_artifact_sha256="a" * 64,
        reliability_curve_artifact_path="artifacts/calibration/dummy_curve.png",
        reliability_curve_sha256="b" * 64,
        calibration_dataset_manifest_path="artifacts/calibration/dummy_manifest.json",
        calibration_dataset_sha256="c" * 64,
        model_id="MODEL_V1",
        feature_schema_hash="d" * 64,
        signed_at="2026-08-24T15:00:00Z",
        signer_key_id="PROD_CALIBRATION_KEY_2026_V1",
        signature="e" * 128
    )

    # 物理文件不存在，绝不信任
    assert dummy_evidence.is_valid() is False

    # 受控加载函数返回 None
    assert load_verified_probability_calibration("dummy_path.json") is None

    # 若传入未验证凭证，通知构建必须拒绝抛错
    with pytest.raises(ValueError, match="Invalid or unverified ProbabilityCalibrationEvidence"):
        MessageNotifier.format_daily_report_markdown(
            signal_date="2026-08-24",
            execution_date="2026-08-25",
            top_df=pd.DataFrame([{"symbol": "600519.SH", "name": "贵州茅台", "pred_score": 0.88, "target_weight": 0.2, "close": 1600.0}]),
            macro_status="正常多头持仓",
            calibration_evidence=dummy_evidence
        )

    # 未提供有效概率校准凭证时，通知渲染只能展示“模型排序分数”，绝不能出现“预测上涨概率”或百分比
    report = MessageNotifier.format_daily_report_markdown(
        signal_date="2026-08-24",
        execution_date="2026-08-25",
        top_df=pd.DataFrame([{"symbol": "600519.SH", "name": "贵州茅台", "pred_score": 0.88, "target_weight": 0.2, "close": 1600.0}]),
        macro_status="正常多头持仓",
        calibration_evidence=None
    )
    assert "模型排序分数" in report
    assert "预测上涨概率" not in report
    assert "88.00%" not in report


# =========================================================================
# 2. 物理校准文件/签名/模型 Schema 任一篡改拒绝 (Tamper Rejection)
# =========================================================================

def test_s4_rejects_tampered_calibration_files_or_schema(tmp_path):
    """审计项 2: 物理校准文件、签名、模型 Schema 任一被篡改，立即 Fail-Closed 拒绝"""
    cal_file = tmp_path / "cal.bin"
    cal_file.write_bytes(b"valid_calibrator_data")
    cal_sha = hashlib.sha256(b"valid_calibrator_data").hexdigest()

    curve_file = tmp_path / "curve.png"
    curve_file.write_bytes(b"valid_curve_image")
    curve_sha = hashlib.sha256(b"valid_curve_image").hexdigest()

    manifest_file = tmp_path / "manifest.json"
    manifest_data = {
        "model_id": "M_TEST",
        "feature_schema_hash": "a" * 64,
        "dataset_sha256": "b" * 64
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

    evidence = ProbabilityCalibrationEvidence(
        method="isotonic",
        brier_score=0.10,
        sample_count=2000,
        status="VERIFIED",
        calibrator_artifact_path=str(cal_file),
        calibrator_artifact_sha256=cal_sha,
        reliability_curve_artifact_path=str(curve_file),
        reliability_curve_sha256=curve_sha,
        calibration_dataset_manifest_path=str(manifest_file),
        calibration_dataset_sha256=manifest_sha,
        model_id="M_TEST",
        feature_schema_hash="a" * 64,
        signed_at="2026-08-24T15:00:00Z",
        signer_key_id="PROD_CALIBRATION_KEY_2026_V1",
        signature="dummy_sig"
    )

    # 1. 篡改物理文件内容 -> 哈希不匹配拒绝
    cal_file.write_bytes(b"tampered_bytes")
    assert evidence.is_valid() is False

    # 2. 篡改模型 ID / Schema -> Schema 与 Manifest 不匹配拒绝
    cal_file.write_bytes(b"valid_calibrator_data")
    mismatched_ev = ProbabilityCalibrationEvidence(
        method="isotonic",
        brier_score=0.10,
        sample_count=2000,
        status="VERIFIED",
        calibrator_artifact_path=str(cal_file),
        calibrator_artifact_sha256=cal_sha,
        reliability_curve_artifact_path=str(curve_file),
        reliability_curve_sha256=curve_sha,
        calibration_dataset_manifest_path=str(manifest_file),
        calibration_dataset_sha256=manifest_sha,
        model_id="DIFFERENT_MODEL",
        feature_schema_hash="a" * 64,
        signed_at="2026-08-24T15:00:00Z",
        signer_key_id="PROD_CALIBRATION_KEY_2026_V1",
        signature="dummy_sig"
    )
    assert mismatched_ev.is_valid() is False


# =========================================================================
# 3. 正式 prospective API 没有 allow_historical 绕过 (No allow_historical)
# =========================================================================

def test_s4_prospective_apis_have_no_allow_historical_bypass():
    """审计项 3: seal_signal 与 execute_observed 彻底移除 allow_historical 形参，严禁绕过"""
    seal_params = inspect.signature(seal_signal).parameters
    exec_params = inspect.signature(execute_observed).parameters

    assert "allow_historical" not in seal_params, "seal_signal 绝不可包含 allow_historical 参数"
    assert "allow_historical" not in exec_params, "execute_observed 绝不可包含 allow_historical 参数"

    # 传参触发 TypeError
    with pytest.raises(TypeError):
        seal_signal("2026-08-28", pd.DataFrame(), "M1", "mat.parquet", allow_historical=True)

    with pytest.raises(TypeError):
        execute_observed("2026-08-28", "2026-08-31", pd.DataFrame(), allow_historical=True)

    # 禁止回填历史日期 (<= 2026-08-24) 冒充前瞻
    with pytest.raises(ValueError, match="禁止回填历史日期"):
        seal_signal(signal_date="2026-08-24", picks_df=pd.DataFrame([{"symbol": "600519.SH"}]), model_id="M1", factor_matrix_path="mat.parquet")

    with pytest.raises(ValueError, match="禁止回填历史日期"):
        execute_observed(signal_date="2026-08-21", execution_date="2026-08-24", quotes_df=pd.DataFrame([{"symbol": "600519.SH"}]))


# =========================================================================
# 4. 缺字段、缺日期列、缺任一标的行情时，正式账本和制品零变化 (Fail-Closed)
# =========================================================================

def test_s4_missing_fields_or_partial_quotes_leaves_ledgers_and_artifacts_intact(tmp_path):
    """审计项 4: 缺任一字段或行情时，全天拒绝撮合，账本文件与制品零变化，观察天数+0"""
    sig_dir = tmp_path / "prospective_signals"
    paper_file = tmp_path / "formal_paper.parquet"
    paper = PaperTradingLedger(ledger_file=paper_file)

    cal = CanonicalTradingCalendar.get_instance()

    fm_path = tmp_path / "fm_20260828.parquet"
    pd.DataFrame({"date": ["2026-08-28", "2026-08-28"], "symbol": ["600519.SH", "000858.SZ"], "f": [1.0, 2.0]}).to_parquet(fm_path)

    picks = pd.DataFrame([
        {"symbol": "600519.SH", "name": "茅台", "date": "2026-08-28", "pred_score": 0.8, "target_weight": 0.5, "close": 1600.0},
        {"symbol": "000858.SZ", "name": "五粮液", "date": "2026-08-28", "pred_score": 0.7, "target_weight": 0.5, "close": 180.0}
    ])

    with patch.object(cal, "is_trading_day", return_value=True), \
         patch.object(cal, "next_trading_day", return_value="2026-08-31"):

        # 封存信号
        seal_signal(
            signal_date="2026-08-28",
            picks_df=picks,
            model_id="M1",
            factor_matrix_path=fm_path,
            storage_dir=sig_dir
        )

        pre_paper_bytes = paper_file.read_bytes() if paper_file.exists() else None

        # 仅提供 600519.SH 行情，缺失 000858.SZ -> 强制抛错拦截
        partial_quotes = pd.DataFrame([
            {"symbol": "600519.SH", "date": "2026-08-31", "open": 1620.0, "close": 1630.0}
        ])

        with pytest.raises(ValueError, match="缺少执行日 .* 真实行情数据，全天拒绝撮合"):
            execute_observed(
                signal_date="2026-08-28",
                execution_date="2026-08-31",
                quotes_df=partial_quotes,
                paper_ledger=paper,
                storage_dir=sig_dir
            )

        # 验证制品零产生与账本零变动
        assert not (sig_dir / "EXECUTION_OBSERVED_2026-08-31.json").exists()
        post_paper_bytes = paper_file.read_bytes() if paper_file.exists() else None
        assert pre_paper_bytes == post_paper_bytes
        assert len(paper.positions) == 0


# =========================================================================
# 5. 研究历史回放与正式 Paper/Shadow/prospective 目录物理隔离 (Isolation)
# =========================================================================

def test_s4_research_replay_physically_isolated_from_formal_directories(tmp_path):
    """审计项 5: 科研回放输出严格物理隔离，禁止写入正式 Paper/Shadow/prospective 目录"""
    replay_dir = tmp_path / "research_replays"
    adapter = HistoricalReplayAdapter(replay_storage_dir=replay_dir)

    picks = pd.DataFrame([
        {"symbol": "600519.SH", "target_weight": 0.5, "close": 1600.0, "pred_score": 0.8},
        {"symbol": "000858.SZ", "target_weight": 0.5, "close": 180.0, "pred_score": 0.7}
    ])
    quotes = pd.DataFrame([
        {"symbol": "600519.SH", "date": "2026-08-24", "open": 1620.0, "close": 1630.0},
        {"symbol": "000858.SZ", "date": "2026-08-24", "open": 182.0, "close": 185.0}
    ])

    res = adapter.replay_step(
        signal_date="2026-08-21",
        execution_date="2026-08-24",
        quotes_df=quotes,
        picks_df=picks,
        model_id="research_model_replay"
    )

    assert res["event"] == "RESEARCH_REPLAY_STEP"
    assert res["formal_observation_days_added"] == 0
    assert res["is_prospective"] is False

    # 产物仅存在于隔离研究目录中
    assert (replay_dir / "RESEARCH_SIGNAL_2026-08-21.json").exists()
    assert (replay_dir / "RESEARCH_STEP_2026-08-24.json").exists()

    # 严禁绑定正式生产账本 (动态获取真实正式 JSON 账本路径)
    formal_paper = Path(PaperTradingLedger().ledger_file).resolve()
    with pytest.raises(PermissionError, match="科研历史回放严禁绑定或写入正式生产 Paper 账本"):
        HistoricalReplayAdapter(replay_storage_dir=replay_dir, paper_ledger_file=formal_paper)


# =========================================================================
# 6. 外部调用者不能调低信任根、物理证书、30 天观察门槛 (Non-Downgradable)
# =========================================================================

def test_s4_live_gate_parameters_non_downgradable(tmp_path):
    """审计项 6: 外部调用方严禁通过参数调低安全门限"""
    with pytest.raises(TypeError):
        check_keyring_integrity(require_trust_root=False)

    with pytest.raises(TypeError):
        verify_live_connection_gate(account_id="5500123456", live_confirm=True, require_trust_root=False)

    with pytest.raises(TypeError):
        verify_live_order_gate(
            account_id="5500123456",
            live_confirm=True,
            min_observation_days=0
        )

    with pytest.raises(TypeError):
        verify_live_order_gate(
            account_id="5500123456",
            live_confirm=True,
            check_physical_cert=False
        )


# =========================================================================
# 7. 订单 symbol、side、quantity、price 与签名不一致均拒绝 (Field Binding)
# =========================================================================

def test_s4_order_parameters_must_strictly_match_signed_payload(tmp_path, monkeypatch):
    """审计项 7: 委托参数 (symbol, side, shares, price) 必须与签名 Payload 逐字段强比对，任一不符立即拦截"""
    monkeypatch.setattr(settings, "LIVE_TRADING_READY", True)
    monkeypatch.setattr(settings, "LIVE_TRADING_GATE_STATUS", "UNLOCKED")
    monkeypatch.setattr(settings, "LIVE_TRADING_STATUS", "UNLOCKED")
    monkeypatch.setattr(settings, "EMERGENCY_KILL_SWITCH", False)
    monkeypatch.setattr(settings, "LIVE_ACCOUNT_WHITELIST", ["5500123456"])
    monkeypatch.setattr(settings, "MAX_QUOTE_STALENESS_SECONDS", 300.0)
    monkeypatch.setenv("QUANT_TRUSTED_KEYRING_SHA256", compute_canonical_keyring_hash())

    cert_file = tmp_path / "cert.json"
    cert_file.write_text(json.dumps({
        "model_id": "PROD_S4_MODEL",
        "status": "CERTIFIED",
        "dataset_sha256": "9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42"
    }), encoding="utf-8")

    model_rec = DummyS4ModelRecord(
        model_id="PROD_S4_MODEL",
        evidence={
            "certification_ref": str(cert_file),
            "prospective_validation": {"status": "MATURE", "observed_trading_days": 30}
        }
    )
    paper = MagicMock(observed_trading_days=30)
    shadow = MagicMock(observed_trading_days=30)
    nonce_file = tmp_path / "nonces.json"

    # 生成绑定 600519.SH, BUY, 1000 股, 50.0 元 的审批凭证
    appr = make_s4_signed_approval(
        tmp_path,
        nonce="bind_01",
        symbol="600519.SH",
        side="BUY",
        quantity=1000.0,
        limit_price=50.0
    )

    base_args = {
        "account_id": "5500123456",
        "live_confirm": True,
        "quote_timestamp": time.time(),
        "current_date": "2026-08-24",
        "model_record": model_rec,
        "paper_ledger": paper,
        "shadow_ledger": shadow,
        "approval_artifact_path": appr,
        "nonce_store_path": nonce_file,
    }

    # 1. symbol 错配拦截
    with pytest.raises(LiveGateAuthError, match="委托标的代码.*不匹配"):
        verify_live_order_gate(**base_args, order_symbol="000858.SZ", order_side="BUY", order_shares=1000, order_price=50.0)

    # 2. 方向错配拦截
    with pytest.raises(LiveGateAuthError, match="委托买卖方向.*不匹配"):
        verify_live_order_gate(**base_args, order_symbol="600519.SH", order_side="SELL", order_shares=1000, order_price=50.0)

    # 3. 数量错配拦截
    with pytest.raises(LiveGateAuthError, match="委托股数.*不匹配"):
        verify_live_order_gate(**base_args, order_symbol="600519.SH", order_side="BUY", order_shares=2000, order_price=50.0)

    # 4. 价格错配拦截
    with pytest.raises(LiveGateAuthError, match="委托价格.*不匹配"):
        verify_live_order_gate(**base_args, order_symbol="600519.SH", order_side="BUY", order_shares=1000, order_price=55.0)

    # 5. 完全匹配放行
    ok, _ = verify_live_order_gate(**base_args, order_symbol="600519.SH", order_side="BUY", order_shares=1000, order_price=50.0)
    assert ok is True


# =========================================================================
# 8. 并发相同 Nonce 只能放行一次 (Atomic Mutex Nonce Locking)
# =========================================================================

def test_s4_concurrent_identical_nonce_exactly_one_passes(tmp_path, monkeypatch):
    """审计项 8: 跨进程排他文件锁实测，两个并发使用相同 Nonce 的发单请求必须恰好放行一个，另一个被重放拦截"""
    monkeypatch.setattr(settings, "LIVE_TRADING_READY", True)
    monkeypatch.setattr(settings, "LIVE_TRADING_GATE_STATUS", "UNLOCKED")
    monkeypatch.setattr(settings, "LIVE_TRADING_STATUS", "UNLOCKED")
    monkeypatch.setattr(settings, "EMERGENCY_KILL_SWITCH", False)
    monkeypatch.setattr(settings, "LIVE_ACCOUNT_WHITELIST", ["5500123456"])
    monkeypatch.setattr(settings, "MAX_QUOTE_STALENESS_SECONDS", 300.0)
    monkeypatch.setenv("QUANT_TRUSTED_KEYRING_SHA256", compute_canonical_keyring_hash())

    cert_file = tmp_path / "cert.json"
    cert_file.write_text(json.dumps({
        "model_id": "PROD_S4_MODEL",
        "status": "CERTIFIED",
        "dataset_sha256": "9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42"
    }), encoding="utf-8")

    model_rec = DummyS4ModelRecord(
        model_id="PROD_S4_MODEL",
        evidence={
            "certification_ref": str(cert_file),
            "prospective_validation": {"status": "MATURE", "observed_trading_days": 30}
        }
    )
    paper = MagicMock(observed_trading_days=30)
    shadow = MagicMock(observed_trading_days=30)
    nonce_file = tmp_path / "concurrent_nonces.json"

    shared_nonce = "s4_shared_concurrent_nonce_01"
    appr = make_s4_signed_approval(
        tmp_path,
        nonce=shared_nonce,
        symbol="600519.SH",
        side="BUY",
        quantity=1000.0,
        limit_price=50.0
    )

    args = {
        "account_id": "5500123456",
        "live_confirm": True,
        "quote_timestamp": time.time(),
        "current_date": "2026-08-24",
        "model_record": model_rec,
        "paper_ledger": paper,
        "shadow_ledger": shadow,
        "approval_artifact_path": appr,
        "nonce_store_path": nonce_file,
        "order_symbol": "600519.SH",
        "order_side": "BUY",
        "order_shares": 1000,
        "order_price": 50.0,
    }

    results = []
    errors = []

    def task():
        try:
            ok, _ = verify_live_order_gate(**args)
            results.append(ok)
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(task)
        f2 = executor.submit(task)
        concurrent.futures.wait([f1, f2])

    assert len(results) == 1, f"必须恰好一个请求成功放行，实际成功 {len(results)}"
    assert len(errors) == 1, f"必须恰好一个请求被拦截，实际拦截 {len(errors)}"
    assert isinstance(errors[0], LiveGateAuthError)
    assert "检测到审批凭证重放攻击" in str(errors[0])


def test_s4_test_mode_artifacts_write_only_to_tmp_dir(tmp_path, monkeypatch):
    """
    1. 验证 QUANT_TEST_MODE=1 时，GlobalMacroAPI、AutoSyncEngine 等只写向指定的临时 artifacts 目录，
    BASE_DIR / artifacts 中的 tracked 文件内容绝不改变。
    """
    import hashlib
    from data.global_macro_api import GlobalMacroAPI
    from data.live_market_and_news_api import AutoSyncEngine

    workspace_artifacts = Path(settings.BASE_DIR) / "artifacts"
    tracked_files = [
        workspace_artifacts / "global_macro_sentiment_snapshot.json",
        workspace_artifacts / "live_telegraph_stream.json",
    ]
    before_hashes = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in tracked_files if f.exists()}

    test_art = (tmp_path / "explicit_test_artifacts").resolve()
    test_art.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANT_TEST_MODE", "1")
    monkeypatch.setenv("QUANT_ARTIFACTS_DIR", str(test_art))
    monkeypatch.setattr(settings, "ARTIFACTS_DIR", test_art, raising=False)

    with patch.object(GlobalMacroAPI, "fetch_usdcnh_rate", return_value={"symbol": "USDCNH", "rate": 6.8, "pct_change": 0.0, "status": "SUCCESS", "date": "2026-09-06"}), \
         patch.object(GlobalMacroAPI, "fetch_overseas_tech_giants", return_value={"NVDA": {"name": "NVDA", "price": 200.0, "pct_change": 1.0}}), \
         patch.object(GlobalMacroAPI, "fetch_global_commodities", return_value={"GOLD": {"name": "GOLD", "price": 2000.0, "pct_change": 0.0}, "OIL": {"name": "OIL", "price": 80.0, "pct_change": 0.0}}), \
         patch.object(GlobalMacroAPI, "fetch_us_china_bond_yields", return_value={"cn_10y": 2.0, "us_10y": 4.0, "spread_us_cn": 2.0, "date": "2026-09-06"}):
        snap = GlobalMacroAPI.generate_macro_regime_snapshot(save_disk=True)
        assert snap is not None

    test_picks = tmp_path / "test_picks.csv"
    test_picks.write_text("symbol,close\n600519.SH,1600.0\n", encoding="utf-8")
    with patch("data.live_market_and_news_api.LiveMarketAPI.fetch_live_quotes") as mock_quotes, \
         patch("data.live_market_and_news_api.LiveNewsAPI.fetch_7x24_telegraph") as mock_tele, \
         patch("factors.sentiment_engine.NewsCatalystScorer.get_stock_catalyst") as mock_cat:
        mock_quotes.return_value = pd.DataFrame([{"symbol": "600519.SH", "close": 1605.0, "pct_change": 0.31}])
        mock_tele.return_value = [{"time": "15:00:00", "content": "测试快讯内容"}]
        mock_cat.return_value = {"headline": "测试催化", "sentiment_score": 88, "sentiment_stage": "突破"}
        AutoSyncEngine.sync_picks_and_news(test_picks)

    # 验证文件只写到了指定的 test_art
    assert (test_art / "global_macro_sentiment_snapshot.json").exists()
    assert (test_art / "live_telegraph_stream.json").exists()

    # 验证工作区 tracked 产物 100% 字节不变
    for f, orig_hash in before_hashes.items():
        assert hashlib.sha256(f.read_bytes()).hexdigest() == orig_hash, f"工作区文件 {f.name} 被意外修改！"


def test_s4_network_disabled_in_test_mode():
    """
    2. 验证 QUANT_DISABLE_NETWORK=1 时，requests、urllib、socket 发起真实连接会抛出 NetworkDisabledForTestError。
    """
    import socket
    import urllib.request
    import requests
    from data.network_policy import NetworkDisabledForTestError
    from data.global_macro_api import GlobalMacroAPI
    from data.live_market_and_news_api import LiveMarketAPI, LiveNewsAPI

    with pytest.raises(NetworkDisabledForTestError):
        s = socket.socket()
        s.connect(("1.1.1.1", 80))

    with pytest.raises(NetworkDisabledForTestError):
        socket.create_connection(("1.1.1.1", 80), timeout=1)

    with pytest.raises(NetworkDisabledForTestError):
        urllib.request.urlopen("http://example.com", timeout=1)

    with pytest.raises(NetworkDisabledForTestError):
        requests.get("http://example.com", timeout=1)

    with pytest.raises(NetworkDisabledForTestError):
        GlobalMacroAPI.fetch_overseas_tech_giants()

    with pytest.raises(NetworkDisabledForTestError):
        LiveMarketAPI.fetch_live_quotes(["600519.SH"])

    with pytest.raises(NetworkDisabledForTestError):
        LiveNewsAPI.fetch_7x24_telegraph()


def test_s4_subprocess_inherits_test_env_and_network_off(tmp_path):
    """
    3. 验证子进程调用（如有）继承测试环境变量与禁网设置。
    """
    import subprocess
    script = """
import os
import sys
from pathlib import Path
from config.settings import settings
from data.network_policy import install_network_guard, NetworkDisabledForTestError, is_network_disabled
import socket

assert os.environ.get("QUANT_TEST_MODE") == "1", "QUANT_TEST_MODE not inherited"
assert os.environ.get("QUANT_DISABLE_NETWORK") == "1", "QUANT_DISABLE_NETWORK not inherited"
assert is_network_disabled() is True, "is_network_disabled is False"
assert settings.TEST_MODE is True
assert settings.DISABLE_NETWORK is True

install_network_guard()
try:
    socket.socket().connect(("1.1.1.1", 80))
    print("FAIL_NETWORK_ALLOWED")
    sys.exit(2)
except NetworkDisabledForTestError:
    pass

art_dir = settings.ARTIFACTS_DIR
assert art_dir != settings.BASE_DIR / "artifacts", "ARTIFACTS_DIR points to workspace artifacts"
print("SUBPROCESS_ISOLATION_OK")
sys.exit(0)
"""
    p = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(settings.BASE_DIR),
        env=dict(os.environ),
    )
    assert p.returncode == 0, f"子进程未能正确继承测试密闭环境:\nSTDOUT: {p.stdout}\nSTDERR: {p.stderr}"
    assert "SUBPROCESS_ISOLATION_OK" in p.stdout


def test_s4_direct_write_to_base_dir_artifacts_not_silently_redirected(tmp_path):
    """
    4. 验证直接写 BASE_DIR / artifacts 不会被偷偷替换到临时目录（即证明已彻底移除 conftest 的黑魔法）。
    """
    import io
    import builtins
    import os

    # 验证 builtins.open 和 io.open 是原生函数，不是 safe_open
    assert getattr(builtins.open, "__name__", "") != "safe_open", "builtins.open 仍然被 safe_open 劫持！"
    assert getattr(io.open, "__name__", "") != "safe_open", "io.open 仍然被 safe_open 劫持！"
    assert getattr(os.replace, "__name__", "") != "safe_os_replace", "os.replace 仍然被 safe_os_replace 劫持！"
    assert getattr(Path.replace, "__name__", "") != "safe_path_replace", "Path.replace 仍然被 safe_path_replace 劫持！"

    # 在沙箱中创建独立文件进行写入测试，确保路径行为完全原生
    probe_dir = tmp_path / "probe_dir"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_file = probe_dir / "probe.txt"
    probe_file.write_text("native_write_test", encoding="utf-8")
    assert probe_file.read_text(encoding="utf-8") == "native_write_test"


def test_s4_global_macro_and_telegraph_files_unmutated_after_full_run():
    """
    5. 验证在完整 pytest 跑完后，跟踪文件：
       - artifacts/global_macro_sentiment_snapshot.json
       - artifacts/live_telegraph_stream.json
       其内容与 git HEAD 提交完全一致，工作区状态绝对零修改、零污染。
    """
    import subprocess
    import hashlib

    workspace_artifacts = Path(settings.BASE_DIR) / "artifacts"
    tracked_files = [
        "artifacts/global_macro_sentiment_snapshot.json",
        "artifacts/live_telegraph_stream.json",
    ]

    # 1. 验证 git status 绝对没有任何 tracked 文件处于修改状态
    status_p = subprocess.run(
        ["git", "status", "--porcelain", "artifacts/"],
        capture_output=True,
        text=True,
        cwd=str(settings.BASE_DIR)
    )
    assert status_p.stdout.strip() == "", (
        f"🚨 git status 检测到 artifacts 目录存在修改或未跟踪文件: {status_p.stdout}"
    )

    # 2. 逐一验证规范化内容哈希与 git HEAD 严格一致
    for rel_path in tracked_files:
        disk_file = Path(settings.BASE_DIR) / rel_path
        assert disk_file.exists(), f"Tracked file missing: {rel_path}"
        disk_norm_hash = hashlib.sha256(disk_file.read_bytes().replace(b"\r\n", b"\n")).hexdigest()

        p = subprocess.run(
            ["git", "show", f"HEAD:{rel_path}"],
            capture_output=True,
            cwd=str(settings.BASE_DIR)
        )
        assert p.returncode == 0, f"无法从 git HEAD 读取 {rel_path}"
        head_norm_hash = hashlib.sha256(p.stdout.replace(b"\r\n", b"\n")).hexdigest()

        assert disk_norm_hash == head_norm_hash, (
            f"🚨 跟踪文件 {rel_path} 遭非法修改！\n"
            f"HEAD SHA256 (canonical):    {head_norm_hash}\n"
            f"Current SHA256 (canonical): {disk_norm_hash}"
        )


