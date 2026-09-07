"""
Stage B: Physical Signed Promotion Evidence Test Suite
(tests/test_stage_b_physical_signed_evidence.py)

验证模型注册表晋升物理签署证据全要素防伪门禁 (Fail-Closed):
1. 文件不存在拒绝
2. 哈希不匹配拒绝
3. 签名不匹配拒绝
4. 未知 signer 拒绝
5. 模型 ID 不一致拒绝
6. Schema hash 不一致拒绝
7. IMMATURE 拒绝
8. 观察天数不足拒绝
9. 非交易日拒绝
10. 合法测试证据通过
11. 修改证据文件后再次验证失败 (Tamper Detection)
12. CLI 彻底移除 manual_claim 并强制物理文件
"""
import json
import hashlib
from pathlib import Path
import pytest
from datetime import datetime

from models.registry import (
    ModelRegistry,
    ModelState,
    PromotionError,
    EvidenceArtifact,
    verify_promotion_evidence,
)
from data.crypto_anchor import (
    TRUSTED_KEY_REGISTRY,
    DOMAIN_SEPARATOR_PROMOTION,
    generate_keypair,
)
from data.trading_calendar import CanonicalTradingCalendar


@pytest.fixture
def stage_b_crypto_env(monkeypatch):
    """为 Stage B 测试生成并注册专属合法测试公私钥"""
    sk_bytes, pk_bytes = generate_keypair()
    key_id = "STAGE_B_TEST_SIGNER_KEY_2026"
    test_entry = {
        "algorithm": "ED25519",
        "key_id": key_id,
        "public_key_hex": pk_bytes.hex(),
        "allowed_purposes": ["MODEL_PROMOTION", "RUNTIME_ATTESTATION"],
        "issuer_type": "PROJECT_TEST",
        "institution": "STAGE_B_TEST_AUTHORITY",
        "status": "ACTIVE",
        "not_before": "2025-01-01T00:00:00Z",
        "not_after": "2030-01-01T00:00:00Z",
        "is_production": False,
    }
    # 注册到受信任公钥环
    monkeypatch.setitem(TRUSTED_KEY_REGISTRY, key_id, test_entry)
    return {
        "key_id": key_id,
        "private_key_hex": sk_bytes.hex(),
        "public_key_hex": pk_bytes.hex(),
    }


def make_valid_evidence_artifact(
    tmp_path: Path,
    model_id: str,
    crypto_info: dict,
    evidence_type: str = "PROSPECTIVE_VALIDATION",
    dataset_sha256: str = "9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42",
    feature_schema_hash: str = "ad44898838817b0867f51e224377e1729ced530bd9a42f7b22cd81672e76b21e",
    observed_trading_days: int = 21,
    observation_start: str = "2026-07-27",
    observation_end: str = "2026-08-24",
    status: str = "MATURE",
    filename: str = "evidence.json",
) -> EvidenceArtifact:
    """构造合法的物理签署证据对象并落盘"""
    art_path = tmp_path / filename
    cal = CanonicalTradingCalendar.get_instance()
    trading_dates = cal.trading_days_between(observation_start, observation_end)
    ledger = [{"trade_date": d, "pnl": 0.001} for d in trading_dates[:observed_trading_days]]

    art = EvidenceArtifact(
        evidence_type=evidence_type,
        artifact_path=str(art_path.resolve()),
        artifact_sha256="",
        schema_version="evidence_v1",
        model_id=model_id,
        dataset_sha256=dataset_sha256,
        feature_schema_hash=feature_schema_hash,
        created_at=datetime.now().isoformat(),
        observation_start=observation_start,
        observation_end=observation_end,
        observed_trading_days=observed_trading_days,
        status=status,
        approver="test_auditor",
        signer_key_id=crypto_info["key_id"],
        signature="",
        metrics={"rank_ic": 0.035, "sharpe": 1.95},
        ledger_records=ledger,
    )
    # 签署并保存
    art.sign(private_key_hex=crypto_info["private_key_hex"])
    art.save(art_path)
    return art


@pytest.fixture
def test_model_setup(tmp_path):
    """注册一个具备完整血缘的研究制品并晋升为 APPROVED 状态"""
    reg = ModelRegistry(root=tmp_path / "registry")
    dummy_model_file = tmp_path / "dummy_model.pkl"
    dummy_model_file.write_bytes(b"dummy_model_bytes")

    mid = reg.register_research_artifact(
        artifact_path=dummy_model_file,
        model_type="lightgbm",
        metrics={"auc": 0.72},
        dataset_sha256="9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42",
        feature_schema_hash="ad44898838817b0867f51e224377e1729ced530bd9a42f7b22cd81672e76b21e",
    )
    reg.promote(mid, ModelState.CANDIDATE, approver="researcher")
    reg.promote(mid, ModelState.APPROVED, approver="quant_chair", evidence={"certification_ref": "reports/cert.json"})
    rec = reg.get(mid)
    return reg, rec


class TestStageBPhysicalSignedEvidence:
    def test_reject_nonexistent_evidence_file(self, tmp_path, test_model_setup, stage_b_crypto_env):
        """1. 文件不存在拒绝"""
        reg, rec = test_model_setup
        non_existent = tmp_path / "does_not_exist_evidence.json"

        with pytest.raises(PromotionError, match="物理文件不存在|不存在"):
            verify_promotion_evidence(
                raw_ev=str(non_existent),
                model_rec=rec,
                expected_type="prospective_validation",
            )

    def test_reject_hash_mismatch(self, tmp_path, test_model_setup, stage_b_crypto_env):
        """2. 哈希不匹配拒绝"""
        reg, rec = test_model_setup
        art = make_valid_evidence_artifact(tmp_path, rec.model_id, stage_b_crypto_env, filename="hash_test.json")
        art.artifact_sha256 = "0" * 64  # 故意伪造错误哈希

        with pytest.raises(PromotionError, match="SHA-256 不匹配"):
            verify_promotion_evidence(
                raw_ev=art,
                model_rec=rec,
                expected_type="prospective_validation",
            )

    def test_reject_signature_mismatch(self, tmp_path, test_model_setup, stage_b_crypto_env):
        """3. 签名不匹配拒绝"""
        reg, rec = test_model_setup
        art = make_valid_evidence_artifact(tmp_path, rec.model_id, stage_b_crypto_env, filename="sig_test.json")
        # 篡改签名
        fake_sig = "a" * 128
        art.signature = fake_sig
        art.save()

        with pytest.raises(PromotionError, match="数字签名验证失败"):
            verify_promotion_evidence(
                raw_ev=art,
                model_rec=rec,
                expected_type="prospective_validation",
            )

    def test_reject_unknown_signer(self, tmp_path, test_model_setup, stage_b_crypto_env):
        """4. 未知 signer 拒绝"""
        reg, rec = test_model_setup
        art = make_valid_evidence_artifact(tmp_path, rec.model_id, stage_b_crypto_env, filename="unknown_signer.json")
        art.signer_key_id = "UNKNOWN_ATTACKER_KEY_666"
        art.save()

        with pytest.raises(PromotionError, match="未知或未注册的 signer_key_id"):
            verify_promotion_evidence(
                raw_ev=art,
                model_rec=rec,
                expected_type="prospective_validation",
            )

    def test_reject_model_id_mismatch(self, tmp_path, test_model_setup, stage_b_crypto_env):
        """5. 模型 ID 不一致拒绝"""
        reg, rec = test_model_setup
        art = make_valid_evidence_artifact(
            tmp_path,
            model_id="m_another_unrelated_model",
            crypto_info=stage_b_crypto_env,
            filename="model_mismatch.json",
        )

        with pytest.raises(PromotionError, match="model_id 不一致"):
            verify_promotion_evidence(
                raw_ev=art,
                model_rec=rec,
                expected_type="prospective_validation",
            )

    def test_reject_schema_hash_mismatch(self, tmp_path, test_model_setup, stage_b_crypto_env):
        """6. Schema hash 不一致拒绝"""
        reg, rec = test_model_setup
        art = make_valid_evidence_artifact(
            tmp_path,
            model_id=rec.model_id,
            crypto_info=stage_b_crypto_env,
            feature_schema_hash="0" * 64,
            filename="schema_mismatch.json",
        )

        with pytest.raises(PromotionError, match="feature_schema_hash 不一致"):
            verify_promotion_evidence(
                raw_ev=art,
                model_rec=rec,
                expected_type="prospective_validation",
            )

    def test_reject_immature_status(self, tmp_path, test_model_setup, stage_b_crypto_env):
        """7. IMMATURE 拒绝"""
        reg, rec = test_model_setup
        art = make_valid_evidence_artifact(
            tmp_path,
            model_id=rec.model_id,
            crypto_info=stage_b_crypto_env,
            status="IMMATURE",
            filename="immature.json",
        )

        with pytest.raises(PromotionError, match="状态不达标"):
            verify_promotion_evidence(
                raw_ev=art,
                model_rec=rec,
                expected_type="prospective_validation",
            )

    def test_reject_insufficient_observation_days(self, tmp_path, test_model_setup, stage_b_crypto_env):
        """8. 观察天数不足拒绝"""
        reg, rec = test_model_setup
        art = make_valid_evidence_artifact(
            tmp_path,
            model_id=rec.model_id,
            crypto_info=stage_b_crypto_env,
            observed_trading_days=10,  # 仅 10 天 < 20 天门限
            filename="insufficient_days.json",
        )

        with pytest.raises(PromotionError, match="观察天数不足"):
            verify_promotion_evidence(
                raw_ev=art,
                model_rec=rec,
                expected_type="prospective_validation",
                min_trading_days=20,
            )

    def test_reject_non_trading_day_dates(self, tmp_path, test_model_setup, stage_b_crypto_env):
        """9. 非合法交易日拒绝 (如周末 2026-09-05/06)"""
        reg, rec = test_model_setup
        art = make_valid_evidence_artifact(
            tmp_path,
            model_id=rec.model_id,
            crypto_info=stage_b_crypto_env,
            observation_start="2026-09-05",  # 周六
            observation_end="2026-09-06",    # 周日
            filename="weekend.json",
        )

        with pytest.raises(PromotionError, match="不是合法交易所交易日"):
            verify_promotion_evidence(
                raw_ev=art,
                model_rec=rec,
                expected_type="prospective_validation",
            )

    def test_accept_valid_signed_evidence(self, tmp_path, test_model_setup, stage_b_crypto_env):
        """10. 合法签署测试证据全要素通过"""
        reg, rec = test_model_setup
        pv_art = make_valid_evidence_artifact(
            tmp_path,
            model_id=rec.model_id,
            crypto_info=stage_b_crypto_env,
            evidence_type="PROSPECTIVE_VALIDATION",
            observed_trading_days=21,
            filename="valid_pv.json",
        )
        pt_art = make_valid_evidence_artifact(
            tmp_path,
            model_id=rec.model_id,
            crypto_info=stage_b_crypto_env,
            evidence_type="PAPER_TRADING",
            observed_trading_days=21,
            filename="valid_pt.json",
        )

        promoted_rec = reg.promote(
            model_id=rec.model_id,
            to_state=ModelState.PRODUCTION,
            approver="chief_risk_officer",
            evidence={
                "prospective_validation": pv_art,
                "paper_trading": pt_art,
            },
        )
        assert promoted_rec.state == ModelState.PRODUCTION
        last_promo = promoted_rec.promotion_history[-1]
        assert last_promo["evidence"]["prospective_validation"]["status"] == "MATURE"
        assert last_promo["evidence"]["paper_trading"]["status"] == "MATURE"

    def test_reject_tampered_evidence_file_after_signing(self, tmp_path, test_model_setup, stage_b_crypto_env):
        """11. 修改证据文件后再次验证失败 (Tamper Detection)"""
        reg, rec = test_model_setup
        pv_art = make_valid_evidence_artifact(
            tmp_path,
            model_id=rec.model_id,
            crypto_info=stage_b_crypto_env,
            evidence_type="PROSPECTIVE_VALIDATION",
            observed_trading_days=21,
            filename="tamper_target.json",
        )
        file_path = Path(pv_art.artifact_path)

        # 攻击方式 A: 恶意修改物理文件内容 (篡改观察天数) 但保持原声明哈希
        raw_text = file_path.read_text(encoding="utf-8")
        tampered_text = raw_text.replace('"observed_trading_days": 21', '"observed_trading_days": 99')
        file_path.write_text(tampered_text, encoding="utf-8")

        # 校验必定失败 (物理哈希不匹配)
        with pytest.raises(PromotionError, match="SHA-256 不匹配"):
            verify_promotion_evidence(
                raw_ev=pv_art,
                model_rec=rec,
                expected_type="prospective_validation",
            )

        # 攻击方式 B: 重新计算并更新声明哈希以企图蒙混过关
        tampered_bytes = file_path.read_bytes()
        new_sha = hashlib.sha256(tampered_bytes).hexdigest()
        pv_art.artifact_sha256 = new_sha
        pv_art.observed_trading_days = 99

        # 校验必定失败 (数字签名校验失败，待签文字节流已被破坏)
        with pytest.raises(PromotionError, match="数字签名验证失败"):
            verify_promotion_evidence(
                raw_ev=pv_art,
                model_rec=rec,
                expected_type="prospective_validation",
            )

    def test_prohibit_manual_claim_and_fake_inputs(self, test_model_setup):
        """12. 彻底拒绝 manual_claim、True/False、空字典等无物理凭证输入"""
        reg, rec = test_model_setup

        for bad_input in [True, False, {}, "", "manual_claim", {"ref": "manual_claim"}]:
            with pytest.raises(PromotionError):
                verify_promotion_evidence(
                    raw_ev=bad_input,
                    model_rec=rec,
                    expected_type="prospective_validation",
                )
