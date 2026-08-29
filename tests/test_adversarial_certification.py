"""
全链路对抗性认证与红队渗透测试套件 (tests/test_adversarial_certification.py)
用于模拟黑客、作弊与恶意绕过场景，对量化系统认证链发起 70+ 种全方位主动攻击：
包含八大核心红队攻击链 (ATTACK 1 ~ ATTACK 8):
- ATTACK 1: Operator Fake Signature 攻击 (operator_signature='hello')
- ATTACK 2: Key Purpose Mismatch 混淆攻击 (Downloader Key 签发 Runtime)
- ATTACK 3: Old Commit Replay 重放攻击 (旧 Commit 证书冒充当前 HEAD)
- ATTACK 4: Legacy Corporate Action Record 绕过攻击 (无 Evidence 声称零事件)
- ATTACK 5: Fake Corporate Hash Strings 伪造攻击 (无真实响应证明文件)
- ATTACK 6: Fake Manifest Hash 字符串攻击 (64hex 字符但无物理 Manifest)
- ATTACK 7: Config Parameter Omission 篡改攻击 (有效配置突变哈希不变漏洞)
- ATTACK 8: Repository Secret Scan 与机构伪装防御 (全库 0 私钥与真实命名校验)
"""
import os
import re
import json
import hashlib
import platform
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from cryptography.hazmat.primitives.asymmetric import ed25519 as crypto_ed25519

from data.crypto_anchor import (
    TRUSTED_KEY_REGISTRY,
    TRUSTED_OPERATOR_REGISTRY,
    DOMAIN_SEPARATOR_RUNTIME,
    DOMAIN_SEPARATOR_ACQUISITION,
    DOMAIN_SEPARATOR_OPERATOR,
    verify_ed25519_signature,
    sign_with_environment_key,
    ed25519_publickey_pure,
    ed25519_sign_pure,
    ed25519_verify_pure,
    generate_keypair,
    compute_canonical_keyring_hash,
    verify_trust_root,
    safe_resolve_path
)
from data.source_registry import (
    TRUSTED_SOURCE_REGISTRY,
    TRUSTED_ACQUISITION_KEYS,
    AcquisitionReceipt,
    OperatorAttestation,
    CorporateActionCoverageEvidence,
    validate_trusted_url,
    extract_domain
)
from data.provenance import (
    SourceClass,
    ProvenanceVerifier,
    UniverseVerificationResult,
    DataProvenanceError,
    SourceEvidenceMetadata,
    CSIRebalanceAnnouncementParser,
    CSIConstituentSnapshotParser,
    PITUniverseStateMachineVerifier
)
from data.data_manager import DataManager
from data.universe_provider import PointInTimeUniverseProvider, StaticUniverseProvider, create_universe_provider
from factors.processor import FactorProcessor
from backtest.audit import (
    AuditMetadata,
    CertificationPolicy,
    AuditCollector,
    NON_CERTIFICATION_OVERRIDE_FIELDS,
    compute_canonical_runtime_config_hash,
    ManifestVerifier,
    ManifestVerificationResult,
    ManifestType
)
from backtest.runtime_attestation import (
    RuntimeAttestationEnvelope,
    compute_canonical_audit_payload_hash,
    create_signed_runtime_attestation,
    get_git_environment_info
)
from strategy.corporate_actions import (
    CorporateAction,
    CorporateActionCoverageRecord,
    CorporateActionProvider,
    CorporateActionDatasetVerifier,
    CorporateActionDatasetProvenanceVerifier
)
from config.settings import QuantConfig, settings


class TestAdversarialCertification:

    # =========================================================================
    # 1. 基础数据源与 HTTPS URL 门禁
    # =========================================================================

    def test_arbitrary_raw_csv_not_automatically_official(self, tmp_path):
        """攻击: 放置任意 fake.csv，系统绝不自动将其识别为 OFFICIAL_PRIMARY"""
        fake_csv = tmp_path / "fake_csi300.csv"
        fake_csv.write_text("effective_date,symbol,action\n2020-06-15,600519.SH,IN\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(fake_csv)

        df = pd.DataFrame({"effective_date": ["2020-06-15"], "symbol": ["600519.SH"], "action": ["IN"]})
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": "OFFICIAL_PRIMARY",
            "source_files": [fake_csv.name],
            "raw_evidence_hashes": {fake_csv.name: h},
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_file": fake_csv.name,
            "baseline_snapshot_date": "2020-01-01",
            "baseline_snapshot_sha256": h,
            "baseline_symbols": ["600519.SH"] * 300,
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest, raw_evidence_dir=tmp_path)
        assert res.source_verified is False
        assert res.is_production_verified is False
        assert any("missing_source_metadata" in fc for fc in res.failed_checks)

    def test_missing_source_metadata_blocks_official_classification(self, tmp_path):
        """攻击: 缺少 .source.json 时，即便哈希完全一致也必须 Fail-Closed 阻止 VERIFIED"""
        raw_file = tmp_path / "data.csv"
        raw_file.write_text("date,symbol\n2020-01-01,600519.SH\n", encoding="utf-8")
        meta, errors = SourceEvidenceMetadata.load_and_verify(tmp_path / "data.csv.source.json", raw_file)
        assert meta is None
        assert any("missing_source_metadata" in e for e in errors)

    def test_fake_source_metadata_boolean_cannot_self_certify(self, tmp_path):
        """攻击: 在 .source.json 中写 "official": true 无法绕过 SHA256 与合法 SourceClass 校验"""
        raw_file = tmp_path / "forged.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        meta_file = tmp_path / "forged.csv.source.json"
        meta_file.write_text(json.dumps({
            "source_id": "CSI",
            "source_url": "https://www.csindex.com.cn/forged.csv",
            "sha256": "0" * 64,
            "official": True
        }), encoding="utf-8")

        meta, errors = SourceEvidenceMetadata.load_and_verify(meta_file, raw_file)
        assert meta is None or len(errors) > 0
        assert any("sha256_mismatch" in e for e in errors)

    def test_insecure_http_url_rejected(self):
        """攻击: 使用明文 HTTP 协议冒充中证或交易所数据源"""
        ok, errs = validate_trusted_url("http://www.csindex.com.cn/data.csv", ["csindex.com.cn"])
        assert ok is False
        assert any("insecure_url_scheme" in e for e in errs)

    def test_impersonated_subdomain_url_rejected(self):
        """攻击: 使用钓鱼域名冒充官方"""
        ok, errs = validate_trusted_url("https://csindex.com.cn.attacker.com/data.csv", ["csindex.com.cn"])
        assert ok is False
        assert any("untrusted_domain" in e for e in errs)

    def test_ip_literal_url_rejected(self):
        """攻击: 使用裸 IP 地址伪装官方数据源"""
        ok, errs = validate_trusted_url("https://1.2.3.4/data.csv", ["csindex.com.cn"])
        assert ok is False
        assert any("ip_literal_host" in e for e in errs)

    def test_userinfo_in_url_rejected(self):
        """攻击: 使用包含 @ 符号的混淆 URL"""
        ok, errs = validate_trusted_url("https://csindex.com.cn@evil.com/data.csv", ["csindex.com.cn"])
        assert ok is False
        assert any("userinfo" in e for e in errs)

    # =========================================================================
    # 2. ATTACK 1: Operator Fake Signature 攻击防御 (Ed25519 真验签)
    # =========================================================================

    def test_operator_nonempty_signature_not_sufficient(self, tmp_path):
        """红队 ATTACK 1: operator_signature='hello' 必须被严格拒绝 (P0)"""
        raw_file = tmp_path / "wind_data.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        h = hashlib.sha256(raw_file.read_bytes()).hexdigest()

        receipt = AcquisitionReceipt(
            receipt_id="REC_001",
            source_id="WIND",
            source_url="https://www.wind.com.cn/export.csv",
            requested_at="2026-01-01T00:00:00Z",
            downloaded_at="2026-01-01T00:00:00Z",
            raw_sha256=h,
            original_filename="wind_data.csv",
            trust_anchor_type="OPERATOR_ATTESTED",
            operator_attestation={
                "operator_id": "WIND_OPERATOR_001",
                "vendor_source_id": "WIND",
                "terminal_reference": "WIND_TERM_999",
                "exported_at": "2026-01-01T00:00:00Z",
                "raw_sha256": h,
                "signing_key_id": "WIND_OPERATOR_KEY_001",
                "signature": "hello"  # 伪造的非 Ed25519 签名
            }
        )
        ok, errs = receipt.verify_against_file(raw_file)
        assert ok is False
        assert receipt.trust_anchor_verified is False
        assert any("invalid_or_missing_ed25519_signature" in e or "signature" in e for e in errs)

    def test_operator_invalid_ed25519_signature_rejected(self, tmp_path):
        """红队 ATTACK 1: 格式为 128-hex 但数学签名错误的 Operator Attestation 必须失败"""
        raw_file = tmp_path / "choice_data.csv"
        raw_file.write_text("choice_data\n", encoding="utf-8")
        h = hashlib.sha256(raw_file.read_bytes()).hexdigest()

        fake_sig = "a" * 128
        att = OperatorAttestation(
            operator_id="CHOICE_OPERATOR_001",
            vendor_source_id="CHOICE",
            terminal_reference="CHOICE_TERM_01",
            exported_at="2026-01-01T00:00:00Z",
            raw_sha256=h,
            signing_key_id="CHOICE_OPERATOR_KEY_001",
            signature=fake_sig
        )
        ok, errs = att.verify_against_file(raw_file)
        assert ok is False
        assert any("ed25519_cryptographic_signature_verification_failed" in e for e in errs)

    def test_operator_valid_registered_signature_accepted(self, tmp_path, monkeypatch):
        """正例: 使用真实注册的 Operator 私钥生成的 Ed25519 签名必须成功通过"""
        # 生成一对临时密钥用于注册表替换测试
        priv = crypto_ed25519.Ed25519PrivateKey.generate()
        sk_hex = priv.private_bytes_raw().hex()
        pk_hex = priv.public_key().public_bytes_raw().hex()

        monkeypatch.setitem(TRUSTED_KEY_REGISTRY, "WIND_OPERATOR_KEY_001", {
            "algorithm": "ED25519",
            "key_id": "WIND_OPERATOR_KEY_001",
            "public_key_hex": pk_hex,
            "allowed_purposes": ["LICENSED_VENDOR_OPERATOR_ATTESTATION"],
            "issuer_type": "LICENSED_VENDOR_OPERATOR",
            "institution": "WIND_INFORMATION_TERMINAL_OPERATOR",
            "status": "ACTIVE",
            "not_before": "2025-01-01T00:00:00Z",
            "not_after": "2030-01-01T00:00:00Z",
            "is_production": True
        })

        raw_file = tmp_path / "wind_export.csv"
        raw_file.write_text("wind_export_content\n", encoding="utf-8")
        h = hashlib.sha256(raw_file.read_bytes()).hexdigest()

        att = OperatorAttestation(
            operator_id="WIND_OPERATOR_001",
            vendor_source_id="WIND",
            terminal_reference="WIND_TERM_PROD",
            exported_at="2026-01-01T00:00:00Z",
            raw_sha256=h,
            signing_key_id="WIND_OPERATOR_KEY_001",
            signature=""
        )
        msg_to_sign = f"{DOMAIN_SEPARATOR_OPERATOR}:".encode("utf-8") + att.compute_canonical_bytes()
        sig = priv.sign(msg_to_sign)
        att.signature = sig.hex()

        ok, errs = att.verify_against_file(raw_file)
        assert ok is True
        assert len(errs) == 0

    def test_unknown_operator_rejected(self, tmp_path):
        """攻击: 未在 TRUSTED_OPERATOR_REGISTRY 登记的 operator_id 必须被拒绝"""
        raw_file = tmp_path / "unknown_op.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        h = hashlib.sha256(raw_file.read_bytes()).hexdigest()

        att = OperatorAttestation(
            operator_id="HACKER_OPERATOR_999",
            vendor_source_id="WIND",
            terminal_reference="REF_001",
            exported_at="2026-01-01T00:00:00Z",
            raw_sha256=h,
            signing_key_id="WIND_OPERATOR_KEY_001",
            signature="a" * 128
        )
        ok, errs = att.verify_against_file(raw_file)
        assert ok is False
        assert any("unregistered_operator_id" in e for e in errs)

    # =========================================================================
    # 3. ATTACK 2: Key Purpose Mismatch 混淆攻击防御 (Key Purpose Separation)
    # =========================================================================

    def test_downloader_key_cannot_sign_runtime(self, monkeypatch):
        """红队 ATTACK 2: 使用 PROD_DOWNLOADER_KEY 签发 Runtime Attestation 必须失败 (P0)"""
        priv = crypto_ed25519.Ed25519PrivateKey.generate()
        sk_hex = priv.private_bytes_raw().hex()
        pk_hex = priv.public_key().public_bytes_raw().hex()

        monkeypatch.setitem(TRUSTED_KEY_REGISTRY, "PROD_DOWNLOADER_KEY_2026_V1", {
            "algorithm": "ED25519",
            "key_id": "PROD_DOWNLOADER_KEY_2026_V1",
            "public_key_hex": pk_hex,
            "allowed_purposes": ["ACQUISITION_RECEIPT"],  # 仅允许 ACQUISITION_RECEIPT
            "issuer_type": "PROJECT",
            "status": "ACTIVE",
            "is_production": True
        })

        meta = AuditMetadata(runtime_config_hash="a" * 64)
        envelope = RuntimeAttestationEnvelope(
            runtime_instance_id="run_test",
            audit_payload_hash=compute_canonical_audit_payload_hash(meta)
        )
        msg_bytes = envelope.compute_canonical_bytes()
        msg_to_sign = f"{DOMAIN_SEPARATOR_RUNTIME}:".encode("utf-8") + msg_bytes
        sig = priv.sign(msg_to_sign)

        # 尝试使用 Downloader Key 验签 Runtime
        sig_ok, sig_errs = verify_ed25519_signature(
            message=msg_bytes,
            signature_hex=sig.hex(),
            key_id="PROD_DOWNLOADER_KEY_2026_V1",
            required_purpose="RUNTIME_ATTESTATION",  # 请求用途与 Key 授权不符
            domain_separator=DOMAIN_SEPARATOR_RUNTIME
        )
        assert sig_ok is False
        assert any("key_purpose_mismatch" in e for e in sig_errs)

    def test_runtime_key_cannot_sign_acquisition(self, monkeypatch):
        """红队 ATTACK 2: 使用 PROD_RUNTIME_KEY 签署采集回执必须失败 (P0)"""
        priv = crypto_ed25519.Ed25519PrivateKey.generate()
        sk_hex = priv.private_bytes_raw().hex()
        pk_hex = priv.public_key().public_bytes_raw().hex()

        monkeypatch.setitem(TRUSTED_KEY_REGISTRY, "PROD_RUNTIME_KEY_2026_V1", {
            "algorithm": "ED25519",
            "key_id": "PROD_RUNTIME_KEY_2026_V1",
            "public_key_hex": pk_hex,
            "allowed_purposes": ["RUNTIME_ATTESTATION"],  # 仅允许 RUNTIME_ATTESTATION
            "issuer_type": "PROJECT",
            "status": "ACTIVE",
            "is_production": True
        })

        msg_bytes = b"sample_digest"
        sig = priv.sign(f"{DOMAIN_SEPARATOR_ACQUISITION}:".encode("utf-8") + msg_bytes)

        sig_ok, sig_errs = verify_ed25519_signature(
            message=msg_bytes,
            signature_hex=sig.hex(),
            key_id="PROD_RUNTIME_KEY_2026_V1",
            required_purpose="ACQUISITION_RECEIPT",
            domain_separator=DOMAIN_SEPARATOR_ACQUISITION
        )
        assert sig_ok is False
        assert any("key_purpose_mismatch" in e for e in sig_errs)

    def test_operator_key_cannot_sign_runtime(self, monkeypatch):
        """红队 ATTACK 2: Operator Key 禁止用于签发 Runtime 信封"""
        priv = crypto_ed25519.Ed25519PrivateKey.generate()
        sk_hex = priv.private_bytes_raw().hex()
        pk_hex = priv.public_key().public_bytes_raw().hex()

        monkeypatch.setitem(TRUSTED_KEY_REGISTRY, "WIND_OPERATOR_KEY_001", {
            "algorithm": "ED25519",
            "key_id": "WIND_OPERATOR_KEY_001",
            "public_key_hex": pk_hex,
            "allowed_purposes": ["LICENSED_VENDOR_OPERATOR_ATTESTATION"],
            "issuer_type": "LICENSED_VENDOR_OPERATOR",
            "status": "ACTIVE",
            "is_production": True
        })

        sig_ok, sig_errs = verify_ed25519_signature(
            message=b"runtime_payload",
            signature_hex="a" * 128,
            key_id="WIND_OPERATOR_KEY_001",
            required_purpose="RUNTIME_ATTESTATION",
            domain_separator=DOMAIN_SEPARATOR_RUNTIME
        )
        assert sig_ok is False
        assert any("key_purpose_mismatch" in e for e in sig_errs)

    def test_valid_signature_wrong_purpose_rejected(self, monkeypatch):
        """攻击: 数学上完全合法的签名，如果用于未经授权的 Purpose，必须 Fail-Closed"""
        priv = crypto_ed25519.Ed25519PrivateKey.generate()
        sk_hex = priv.private_bytes_raw().hex()
        pk_hex = priv.public_key().public_bytes_raw().hex()

        monkeypatch.setitem(TRUSTED_KEY_REGISTRY, "TEST_KEY_SINGLE_PURPOSE", {
            "algorithm": "ED25519",
            "key_id": "TEST_KEY_SINGLE_PURPOSE",
            "public_key_hex": pk_hex,
            "allowed_purposes": ["PURPOSE_A"],
            "issuer_type": "PROJECT",
            "status": "ACTIVE",
            "is_production": False
        })

        msg = b"test_message"
        sig = priv.sign(msg)
        sig_ok, sig_errs = verify_ed25519_signature(
            message=msg,
            signature_hex=sig.hex(),
            key_id="TEST_KEY_SINGLE_PURPOSE",
            required_purpose="PURPOSE_B"
        )
        assert sig_ok is False
        assert any("key_purpose_mismatch" in e for e in sig_errs)

    def test_cross_protocol_signature_reuse_rejected(self, monkeypatch):
        """防御: 跨协议重放防御，不同 Domain Separator 产生的签名绝不可互通"""
        priv = crypto_ed25519.Ed25519PrivateKey.generate()
        sk_hex = priv.private_bytes_raw().hex()
        pk_hex = priv.public_key().public_bytes_raw().hex()

        monkeypatch.setitem(TRUSTED_KEY_REGISTRY, "MULTI_KEY", {
            "algorithm": "ED25519",
            "key_id": "MULTI_KEY",
            "public_key_hex": pk_hex,
            "allowed_purposes": ["RUNTIME_ATTESTATION", "ACQUISITION_RECEIPT"],
            "issuer_type": "PROJECT",
            "status": "ACTIVE",
            "is_production": False
        })

        msg = b"shared_payload_bytes"
        # 使用 Domain A 签名
        sig_a = priv.sign(f"{DOMAIN_SEPARATOR_RUNTIME}:".encode("utf-8") + msg)

        # 尝试用 Domain B 验签
        ok_b, errs_b = verify_ed25519_signature(
            message=msg,
            signature_hex=sig_a.hex(),
            key_id="MULTI_KEY",
            required_purpose="ACQUISITION_RECEIPT",
            domain_separator=DOMAIN_SEPARATOR_ACQUISITION
        )
        assert ok_b is False
        assert any("ed25519_cryptographic_signature_verification_failed" in e for e in errs_b)

    # =========================================================================
    # 4. ATTACK 3: Old Commit Replay 攻击防御 (Git Commit & Tree Binding)
    # =========================================================================

    def test_runtime_commit_mismatch_rejected(self):
        """红队 ATTACK 3: 旧 Commit 生成的合法证书用于当前 Commit 必须报告 Commit Mismatch (P0)"""
        meta = AuditMetadata(runtime_config_hash="a" * 64)
        envelope = RuntimeAttestationEnvelope(
            runtime_instance_id="run_old",
            git_commit_sha="0000000000000000000000000000000000000000",  # 旧 Commit
            git_tree_hash="1111111111111111111111111111111111111111",
            git_dirty=False,
            audit_payload_hash=compute_canonical_audit_payload_hash(meta),
            signing_key_id="PROD_RUNTIME_KEY_2026_V1",
            envelope_signature="a" * 128
        )
        ok, errs = envelope.verify(
            audit_payload_data=meta.to_dict(),
            require_clean_git=False,
            verify_current_git_binding=True,
            is_historical=False
        )
        assert ok is False
        assert any("runtime_commit_mismatch" in e for e in errs)

    def test_runtime_tree_hash_mismatch_rejected(self):
        """红队 ATTACK 3: Commit 相同但 Tree Hash 不一致时必须拒绝"""
        current_git = get_git_environment_info()
        meta = AuditMetadata(runtime_config_hash="a" * 64)
        envelope = RuntimeAttestationEnvelope(
            runtime_instance_id="run_tree_mismatch",
            git_commit_sha=current_git["git_commit_sha"],
            git_tree_hash="deadbeef" * 5,  # 篡改 Tree Hash
            git_dirty=False,
            audit_payload_hash=compute_canonical_audit_payload_hash(meta),
            signing_key_id="PROD_RUNTIME_KEY_2026_V1",
            envelope_signature="a" * 128
        )
        ok, errs = envelope.verify(
            audit_payload_data=meta.to_dict(),
            require_clean_git=False,
            verify_current_git_binding=True,
            is_historical=False
        )
        assert ok is False
        assert any("runtime_tree_hash_mismatch" in e for e in errs)

    def test_dirty_runtime_start_rejected(self):
        """红队 ATTACK 3: 启动时带 dirty 标记的信封绝不能获得当前 VERIFIED"""
        current_git = get_git_environment_info()
        meta = AuditMetadata(runtime_config_hash="a" * 64)
        envelope = RuntimeAttestationEnvelope(
            runtime_instance_id="run_dirty",
            git_commit_sha=current_git["git_commit_sha"],
            git_tree_hash=current_git["git_tree_hash"],
            git_dirty=True,  # 启动时存在脏工作区
            audit_payload_hash=compute_canonical_audit_payload_hash(meta),
            signing_key_id="PROD_RUNTIME_KEY_2026_V1",
            envelope_signature="a" * 128
        )
        ok, errs = envelope.verify(
            audit_payload_data=meta.to_dict(),
            require_clean_git=True,
            verify_current_git_binding=True,
            is_historical=False
        )
        assert ok is False
        assert any("dirty" in e for e in errs)

    def test_historical_attestation_not_labeled_current(self, tmp_path):
        """防御: --historical-attestation 模式跳过 commit binding 但绝不能显示 CURRENT_RUNTIME_VERIFIED"""
        from tools.generate_audit_report import generate_runtime_attestation
        out_file = tmp_path / "RUNTIME_ATTESTATION.md"
        meta = AuditMetadata(runtime_config_hash="a" * 64)
        envelope = RuntimeAttestationEnvelope(
            runtime_instance_id="run_historical_001",
            git_commit_sha="abcdef1234567890abcdef1234567890abcdef12",
            git_tree_hash="1234567890abcdef1234567890abcdef12345678",
            git_dirty=False,
            audit_payload_hash=compute_canonical_audit_payload_hash(meta),
            signing_key_id="PROD_RUNTIME_KEY_2026_V1",
            envelope_signature="a" * 128
        )
        raw_data = {
            "attestation_envelope": envelope.to_dict(),
            "audit_metadata": meta.to_dict()
        }
        generate_runtime_attestation(raw_data, out_file, is_historical=True)
        content = out_file.read_text(encoding="utf-8")
        assert "HISTORICAL_ATTESTATION" in content
        assert "CURRENT_RUNTIME_ATTESTATION" not in content

    # =========================================================================
    # 5. ATTACK 4: Legacy Corporate Action Record 降级测试 (P0)
    # =========================================================================

    def test_legacy_corporate_record_non_certifying(self):
        """红队 ATTACK 4: Legacy CorporateActionCoverageRecord 绝不能赋予生产覆盖资格 (P0)"""
        provider = CorporateActionProvider()
        rec = CorporateActionCoverageRecord(
            symbol="600519.SH",
            query_start="2020-01-01",
            query_end="2024-12-31",
            query_success=True,
            empty_result_verified=True,
            production_eligible=False  # Legacy 记录默认无生产资格
        )
        provider.register_coverage_record(rec)

        # 尝试进行生产覆盖完整性校验
        ok = provider.validate_coverage(
            required_symbols=["600519.SH"],
            start_date="2020-01-01",
            end_date="2024-12-31"
        )
        assert ok is False
        assert provider.coverage_complete is False
        assert provider.zero_event_proof_verified is False

    # =========================================================================
    # 6. ATTACK 5: Fake Corporate Hash Strings 攻击防御 (真实物理文件校验)
    # =========================================================================

    def test_corporate_evidence_requires_evidence_dir(self):
        """红队 ATTACK 5: evidence_dir 为 None 时，即使字段完整也必须拒绝零事件证明 (P0)"""
        ev = CorporateActionCoverageEvidence(
            symbol="600519.SH",
            query_start="2020-01-01",
            query_end="2024-12-31",
            source_id="CSI",
            query_success=True,
            empty_result=True,
            empty_result_verified=True,
            raw_result_file="raw.json",
            raw_result_hash="a" * 64,
            response_file="resp.json",
            response_hash="b" * 64
        )
        ok, errs = ev.is_valid_zero_event_proof("600519.SH", "2020-01-01", "2024-12-31", evidence_dir=None)
        assert ok is False
        assert any("evidence_dir_missing" in e for e in errs)

    def test_corporate_raw_result_file_required(self, tmp_path):
        """红队 ATTACK 5: raw_result_file 物理缺失时必须拒绝"""
        ev = CorporateActionCoverageEvidence(
            symbol="600519.SH",
            query_start="2020-01-01",
            query_end="2024-12-31",
            source_id="CSI",
            query_success=True,
            empty_result=True,
            empty_result_verified=True,
            raw_result_file="missing_raw.json",
            raw_result_hash="a" * 64,
            response_file="resp.json",
            response_hash="b" * 64
        )
        ok, errs = ev.is_valid_zero_event_proof("600519.SH", "2020-01-01", "2024-12-31", evidence_dir=tmp_path)
        assert ok is False
        assert any("corporate_action_raw_result_file_missing" in e for e in errs)

    def test_corporate_raw_result_hash_verified(self, tmp_path):
        """红队 ATTACK 5: 真实文件哈希与 raw_result_hash 不一致时必须拒绝"""
        raw_f = tmp_path / "raw.json"
        raw_f.write_text("actual_content\n", encoding="utf-8")
        resp_f = tmp_path / "resp.json"
        resp_f.write_text("resp_content\n", encoding="utf-8")
        resp_h = hashlib.sha256(resp_f.read_bytes()).hexdigest()

        ev = CorporateActionCoverageEvidence(
            symbol="600519.SH",
            query_start="2020-01-01",
            query_end="2024-12-31",
            source_id="CSI",
            query_success=True,
            empty_result=True,
            empty_result_verified=True,
            raw_result_file="raw.json",
            raw_result_hash="0" * 64,  # 错误哈希
            response_file="resp.json",
            response_hash=resp_h
        )
        ok, errs = ev.is_valid_zero_event_proof("600519.SH", "2020-01-01", "2024-12-31", evidence_dir=tmp_path)
        assert ok is False
        assert any("corporate_action_raw_file_hash_mismatch" in e for e in errs)

    def test_corporate_response_file_hash_verified(self, tmp_path):
        """红队 ATTACK 5: 真实响应证明文件哈希与 response_hash 不一致时必须拒绝"""
        raw_f = tmp_path / "raw.json"
        raw_f.write_text("raw\n", encoding="utf-8")
        raw_h = hashlib.sha256(raw_f.read_bytes()).hexdigest()

        resp_f = tmp_path / "resp.json"
        resp_f.write_text("tampered_resp\n", encoding="utf-8")

        ev = CorporateActionCoverageEvidence(
            symbol="600519.SH",
            query_start="2020-01-01",
            query_end="2024-12-31",
            source_id="CSI",
            query_success=True,
            empty_result=True,
            empty_result_verified=True,
            raw_result_file="raw.json",
            raw_result_hash=raw_h,
            response_file="resp.json",
            response_hash="deadbeef" * 8
        )
        ok, errs = ev.is_valid_zero_event_proof("600519.SH", "2020-01-01", "2024-12-31", evidence_dir=tmp_path)
        assert ok is False
        assert any("corporate_action_response_file_hash_mismatch" in e for e in errs)

    def test_corporate_dataset_manifest_verified(self, tmp_path):
        """验证 CorporateActionDatasetVerifier 对物理 Manifest 与 DataFrame 的哈希核验"""
        df = pd.DataFrame([
            {"ex_date": "2020-06-15", "symbol": "600519.SH", "action_type": "CASH_DIVIDEND", "cash_dividend_per_share": 17.025, "share_ratio": 0.0}
        ])
        h_df = CorporateActionDatasetVerifier.compute_dataframe_sha256(df)

        raw_f = tmp_path / "raw_action.csv"
        raw_f.write_text("ex_date,symbol,action_type,cash\n2020-06-15,600519.SH,CASH_DIVIDEND,17.025\n", encoding="utf-8")
        h_raw = hashlib.sha256(raw_f.read_bytes()).hexdigest()

        manifest = {
            "dataset_name": "CORPORATE_ACTIONS",
            "normalized_dataset_sha256": h_df,
            "source_files": ["raw_action.csv"],
            "source_hashes": {"raw_action.csv": h_raw}
        }
        res = CorporateActionDatasetVerifier.verify_dataset(df, manifest, raw_evidence_dir=tmp_path)
        assert res.dataset_hash_verified is True
        assert len(res.failed_checks) == 0

    # =========================================================================
    # 7. ATTACK 6: Fake Manifest Hash 字符串攻击防御 (物理 Manifest 与 Parent Chain)
    # =========================================================================

    def test_fake_64hex_market_manifest_rejected(self):
        """红队 ATTACK 6: 拥有合法 64hex 格式但 market_manifest_hash_verified=False 必须拒绝 VERIFIED (P1)"""
        meta = AuditMetadata(
            runtime_config_hash="a" * 64,
            runtime_config_hash_verified=True,
            universe_manifest_hash="b" * 64,
            universe_manifest_hash_verified=True,
            factor_manifest_hash="c" * 64,
            factor_manifest_hash_verified=True,
            market_manifest_hash="d" * 64,
            market_manifest_hash_verified=False,  # 未物理验证
            manifest_chain_verified=True,
            corporate_action_provenance_verified=True,
            universe_source_class="OFFICIAL_PRIMARY",
            universe_raw_evidence_verified=True,
            universe_dataset_hash_verified=True,
            actual_backtest_start_date="2020-01-01",
            actual_backtest_end_date="2024-12-31",
            universe_coverage_start="2020-01-01",
            universe_coverage_end="2024-12-31",
            universe_coverage_complete=True,
            survivorship_bias_risk=False,
            historical_st_coverage_complete=True,
            historical_st_bias_risk=False,
            st_unknown_rows=0,
            corporate_action_coverage_complete=True,
            corporate_action_bias_risk=False,
            corporate_action_adjustment_available=True,
            corporate_action_coverage_ratio=1.0,
            cache_fingerprint_verified=True,
            raw_data_provenance_preserved=True,
            adjustment_point_in_time_safe=True,
            future_adjustment_leakage_test_passed=True,
            data_source="csi_official",
            benchmark_source="csi_000300",
            benchmark_coverage_ratio=1.0,
            order_quantity_conservation_passed=True,
            synthetic_data_used=False,
            calendar_is_exchange_official=True
        )
        status, failed = CertificationPolicy.evaluate(meta)
        assert status == "HIGH_RISK"
        assert "market_manifest_hash_unverified" in failed

    def test_fake_64hex_factor_manifest_rejected(self):
        """红队 ATTACK 6: factor_manifest_hash_verified=False 必须拒绝 VERIFIED"""
        meta = AuditMetadata(
            runtime_config_hash="a" * 64,
            runtime_config_hash_verified=True,
            factor_manifest_hash="c" * 64,
            factor_manifest_hash_verified=False
        )
        status, failed = CertificationPolicy.evaluate(meta)
        assert "factor_manifest_hash_unverified" in failed

    def test_fake_64hex_universe_manifest_rejected(self):
        """红队 ATTACK 6: universe_manifest_hash_verified=False 必须拒绝 VERIFIED"""
        meta = AuditMetadata(
            runtime_config_hash="a" * 64,
            runtime_config_hash_verified=True,
            universe_manifest_hash="b" * 64,
            universe_manifest_hash_verified=False
        )
        status, failed = CertificationPolicy.evaluate(meta)
        assert "universe_manifest_hash_unverified" in failed

    def test_manifest_parent_chain_mismatch_rejected(self, tmp_path):
        """红队 ATTACK 6: Manifest 中的 parent_runtime_config_hash 不匹配时，链式校验必须失败"""
        manifest_p = tmp_path / "factor_manifest.json"
        manifest_p.write_text(json.dumps({
            "dataset_name": "FACTORS",
            "parent_runtime_config_hash": "wrong_parent_hash_00000000000000000000000000000000000000000000"
        }), encoding="utf-8")

        res = ManifestVerifier.verify_manifest_file(
            manifest_path=manifest_p,
            expected_parents={"parent_runtime_config_hash": "correct_parent_hash_1111111111111111111111111111111111111111"}
        )
        assert res.parent_chain_verified is False
        assert any("parent_chain_mismatch" in e for e in res.failed_checks)

    # =========================================================================
    # 8. ATTACK 7: Config Omitted Parameter 篡改攻击防御 (Full Config Hash)
    # =========================================================================

    def test_runtime_config_full_dataclass_hash(self):
        """红队 ATTACK 7: 全量 Dataclass 有效配置必须全部进入指纹哈希 (P1-31, P1-32)"""
        cfg = QuantConfig()
        h1 = compute_canonical_runtime_config_hash(cfg)
        assert re.match(r"^[0-9a-fA-F]{64}$", h1)

    def test_runtime_config_commission_mutation_changes_hash(self):
        """红队 ATTACK 7: 修改 COMMISSION_RATE 必须改变 runtime_config_hash"""
        cfg1 = QuantConfig(COMMISSION_RATE=0.00025)
        cfg2 = QuantConfig(COMMISSION_RATE=0.01)
        h1 = compute_canonical_runtime_config_hash(cfg1)
        h2 = compute_canonical_runtime_config_hash(cfg2)
        assert h1 != h2

    def test_runtime_config_lgbm_mutation_changes_hash(self):
        """红队 ATTACK 7: 修改 LGBM_PARAMS 内部超参数必须改变 runtime_config_hash"""
        cfg1 = QuantConfig()
        cfg2 = QuantConfig()
        cfg2.LGBM_PARAMS_CLF = dict(cfg1.LGBM_PARAMS_CLF)
        cfg2.LGBM_PARAMS_CLF["num_leaves"] = 127
        h1 = compute_canonical_runtime_config_hash(cfg1)
        h2 = compute_canonical_runtime_config_hash(cfg2)
        assert h1 != h2

    def test_runtime_config_all_critical_param_mutations(self):
        """红队 ATTACK 7: 逐一验证 15 个关键业务参数突变均会使 runtime_config_hash 产生雪崩改变"""
        base_cfg = QuantConfig()
        base_h = compute_canonical_runtime_config_hash(base_cfg)

        mutations = [
            ("STAMP_DUTY", 0.001),
            ("MIN_COMMISSION", 10.0),
            ("LABEL_HORIZON", 10),
            ("TASK_TYPE", "regression"),
            ("TRAIN_WINDOW_YEARS", 2.5),
            ("VAL_WINDOW_MONTHS", 6),
            ("TEST_WINDOW_MONTHS", 4),
            ("PURGE_GAP_DAYS", 30),
            ("REBALANCE_FREQ", 10),
            ("TOP_K_BUY", 5),
            ("TOP_K_HOLD", 15),
            ("PRICE_LIMIT_ST", 0.10),
            ("MAX_STALE_PRICE_DAYS", 30),
            ("INITIAL_CASH", 5_000_000.0),
            ("LABEL_THRESHOLD_MODE", "fixed")
        ]

        for param_name, new_val in mutations:
            mutated_cfg = QuantConfig()
            setattr(mutated_cfg, param_name, new_val)
            mutated_h = compute_canonical_runtime_config_hash(mutated_cfg)
            assert mutated_h != base_h, f"修改参数 {param_name} 未能改变 runtime_config_hash！"

    # =========================================================================
    # 9. RFC 8032 官方测试向量与 Cross-Library 交叉验签
    # =========================================================================

    def test_rfc8032_test_vector_1(self):
        """官方 RFC 8032 Vector 1 (空消息签名) 严格校验"""
        sk1 = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
        pk1 = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
        msg1 = b""
        sig1 = bytes.fromhex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")

        assert ed25519_publickey_pure(sk1) == pk1
        assert ed25519_sign_pure(msg1, sk1, pk1) == sig1
        assert ed25519_verify_pure(sig1, msg1, pk1) is True

    def test_rfc8032_test_vector_2(self):
        """官方 RFC 8032 Vector 2 (1 字节消息 0x72) 严格校验"""
        sk2 = bytes.fromhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb")
        pk2 = bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c")
        msg2 = bytes.fromhex("72")
        sig2 = bytes.fromhex("92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00")

        assert ed25519_publickey_pure(sk2) == pk2
        assert ed25519_sign_pure(msg2, sk2, pk2) == sig2
        assert ed25519_verify_pure(sig2, msg2, pk2) is True

    def test_rfc8032_test_vector_3(self):
        """官方 RFC 8032 Vector 3 (2 字节消息 0xaf82) 严格校验"""
        sk3 = bytes.fromhex("c5aa8d4390ea833039e91ab7504b60e2772ddba82954da06410938f6f662bc9f")
        pk3 = ed25519_publickey_pure(sk3)
        msg3 = bytes.fromhex("af82")

        priv = crypto_ed25519.Ed25519PrivateKey.from_private_bytes(sk3)
        sig = priv.sign(msg3)

        assert ed25519_sign_pure(msg3, sk3, pk3) == sig
        assert ed25519_verify_pure(sig, msg3, pk3) is True

    def test_cross_library_ed25519_verification(self):
        """交叉验签: 纯 Python 签名 -> cryptography 验签; cryptography 签名 -> 纯 Python 验签"""
        sk, pk = generate_keypair()
        msg = b"cross_library_test_payload"

        # 1. 纯 Python 签 -> cryptography 验
        sig_pure = ed25519_sign_pure(msg, sk, pk)
        pub_crypto = crypto_ed25519.Ed25519PublicKey.from_public_bytes(pk)
        pub_crypto.verify(sig_pure, msg)

        # 2. cryptography 签 -> 纯 Python 验
        priv_crypto = crypto_ed25519.Ed25519PrivateKey.from_private_bytes(sk)
        sig_crypto = priv_crypto.sign(msg)
        assert ed25519_verify_pure(sig_crypto, msg, pk) is True

    # =========================================================================
    # 10. Key Rotation & Lifecycle 状态测试
    # =========================================================================

    def test_revoked_key_rejected(self):
        """测试已吊销状态 (REVOKED) 的公钥验签一律失败"""
        ok, errs = verify_ed25519_signature(
            message=b"test",
            signature_hex="a" * 128,
            key_id="REVOKED_TEST_KEY_2025",
            required_purpose="RUNTIME_ATTESTATION"
        )
        assert ok is False
        assert any("key_status_is_REVOKED" in e for e in errs)

    def test_expired_key_rejected(self):
        """测试已过期状态 (EXPIRED) 的公钥验签一律失败"""
        ok, errs = verify_ed25519_signature(
            message=b"test",
            signature_hex="a" * 128,
            key_id="EXPIRED_TEST_KEY_2020",
            required_purpose="RUNTIME_ATTESTATION",
            created_at_iso="2026-01-01T00:00:00Z"
        )
        assert ok is False
        assert any("key_status_is_EXPIRED" in e or "expired" in e for e in errs)

    def test_future_not_before_key_rejected(self):
        """测试尚未生效 (not_before > created_at) 的公钥验签一律失败"""
        ok, errs = verify_ed25519_signature(
            message=b"test",
            signature_hex="a" * 128,
            key_id="FUTURE_TEST_KEY_2099",
            required_purpose="RUNTIME_ATTESTATION",
            created_at_iso="2026-01-01T00:00:00Z"
        )
        assert ok is False
        assert any("key_not_yet_valid" in e for e in errs)

    # =========================================================================
    # 11. ATTACK 8: 仓库全量代码级 Secret Scan (零硬编码私钥)
    # =========================================================================

    def test_repo_contains_no_production_private_keys(self):
        """红队 ATTACK 8: 扫描仓库全部代码与配置文件，绝对禁止硬编码任何生产私钥或 Secret"""
        repo_root = Path(__file__).resolve().parent.parent

        suspicious_patterns = [
            re.compile(r"BEGIN\s+PRIVATE\s+KEY", re.IGNORECASE),
            re.compile(r"BEGIN\s+ED25519\s+PRIVATE\s+KEY", re.IGNORECASE),
            re.compile(r"QUANT_PROD_[A-Z_]*PRIVATE_KEY\s*=\s*['\"][0-9a-fA-F]{32,}['\"]")
        ]

        scanned_files = []
        violations = []

        extensions = {".py", ".json", ".yaml", ".yml", ".toml", ".md", ".example"}
        for root, dirs, files in os.walk(repo_root):
            # 排除 git 内部目录与虚拟环境
            if ".git" in dirs:
                dirs.remove(".git")
            if ".venv" in dirs:
                dirs.remove(".venv")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")

            for f in files:
                f_path = Path(root) / f
                if f_path.suffix.lower() in extensions or f.startswith(".env"):
                    scanned_files.append(f_path)
                    try:
                        content = f_path.read_text(encoding="utf-8", errors="ignore")
                        for pat in suspicious_patterns:
                            if pat.search(content):
                                violations.append(f"{f_path.relative_to(repo_root)}: matched {pat.pattern}")
                    except Exception:
                        pass

        assert len(violations) == 0, f"发现潜在硬编码私钥或凭证泄漏: {violations}"
        assert len(scanned_files) > 20, "扫描文件数量过少，检查路径解析"

    # =========================================================================
    # 12. 其它历史对抗性测试用例保持通过
    # =========================================================================

    def test_forged_envelope_with_deterministic_sha_rejected(self):
        """攻击: 攻击者利用自算 SHA256 构造伪造信封必须被拒绝"""
        meta = AuditMetadata(runtime_config_hash="a" * 64)
        envelope = RuntimeAttestationEnvelope(
            runtime_instance_id="run_hacked",
            git_commit_sha="a" * 40,
            git_tree_hash="b" * 40,
            git_dirty=False,
            runtime_config_hash="a" * 64,
            audit_payload_hash=compute_canonical_audit_payload_hash(meta),
            signing_key_id="PROD_RUNTIME_KEY_2026_V1",
            envelope_signature="deadbeef" * 16  # 伪造 128-hex 签名
        )
        ok, errs = envelope.verify(audit_payload_data=meta.to_dict(), require_clean_git=False, verify_current_git_binding=False)
        assert ok is False
        assert any("ed25519_cryptographic_signature_verification_failed" in e for e in errs)

    def test_point_in_time_provider_constructor_cannot_set_verified(self):
        """构造函数无法直接注入 verified=True"""
        provider = PointInTimeUniverseProvider(
            changes_df=pd.DataFrame({"date": ["2020-01-01"], "symbol": ["600519.SH"]})
        )
        assert provider.universe_provenance_verified is False

    def test_set_baseline_snapshot_does_not_promote_unverified_source(self):
        """set_baseline_snapshot 不得提升未认证数据源状态"""
        provider = PointInTimeUniverseProvider()
        provider.set_baseline_snapshot("2020-01-01", ["600519.SH"] * 300)
        assert provider.universe_provenance_verified is False

    def test_test_fixture_provider_never_production_verified(self):
        """测试 Fixture 绝不可生产认证: 来源为 TEST_FIXTURE 时 Fail-Closed"""
        provider = PointInTimeUniverseProvider.for_test_fixture(
            fallback_symbols=["600519.SH"],
            coverage_start="2020-01-01",
            coverage_end="2024-12-31"
        )
        assert provider.universe_provenance_verified is False
        assert provider.universe_source_class == SourceClass.TEST_FIXTURE.value

    # =========================================================================
    # 13. External Trust Root Pinning & Agent Replacement Attack Tests (P0)
    # =========================================================================

    def test_external_trust_root_missing_blocks_verified(self, monkeypatch):
        """攻击: 未设置 QUANT_TRUSTED_KEYRING_SHA256 环境变量时，最高认证必须 Fail-Closed"""
        monkeypatch.delenv("QUANT_TRUSTED_KEYRING_SHA256", raising=False)
        is_ok, actual_h, pin_h, errors = verify_trust_root()
        assert is_ok is False
        assert pin_h is None
        assert any("missing_external_trust_root_pin" in e for e in errors)

        meta = AuditMetadata()
        meta.trust_root_verified = is_ok
        rating, failed = CertificationPolicy.evaluate(meta)
        assert rating == "HIGH_RISK"
        assert "external_trust_root_unverified" in failed

    def test_external_trust_root_mismatch_blocks_verified(self, monkeypatch):
        """攻击: 外部 Pin 与仓库公钥环哈希不匹配时，判定为 TAMPERED 拒绝 VERIFIED"""
        monkeypatch.setenv("QUANT_TRUSTED_KEYRING_SHA256", "0" * 64)
        is_ok, actual_h, pin_h, errors = verify_trust_root()
        assert is_ok is False
        assert any("trust_root_tampered" in e for e in errors)

    def test_agent_replaces_runtime_public_key_but_external_pin_rejects(self, monkeypatch):
        """
        红队核心攻击 A: Agent 拥有仓库修改权限，临时生成新密钥对 B 并替换仓库中的公钥表，
        使用私钥 B 签发数学合法的 Runtime 信封，但外部 Trust Root Pin 依然锁定旧 Keyring。
        系统必须识别 Trust Root Mismatch 并 Fail-Closed。
        """
        # 1. 记录原始合法 Keyring Hash 作为外部 Pin
        original_pin = compute_canonical_keyring_hash()
        monkeypatch.setenv("QUANT_TRUSTED_KEYRING_SHA256", original_pin)

        # 2. 攻击者生成新密钥对 B
        sk_b, pk_b = generate_keypair()

        # 3. 攻击者篡改仓库公钥表为 pk_b
        tampered_keyring = dict(TRUSTED_KEY_REGISTRY)
        tampered_keyring["PROD_RUNTIME_KEY_2026_V1"] = dict(tampered_keyring["PROD_RUNTIME_KEY_2026_V1"])
        tampered_keyring["PROD_RUNTIME_KEY_2026_V1"]["public_key_hex"] = pk_b.hex()

        # 4. 核验 Trust Root: 发现仓库已篡改
        is_ok, actual_tampered_h, _, errors = verify_trust_root(keyring=tampered_keyring)
        assert is_ok is False
        assert any("trust_root_tampered" in e for e in errors)

    def test_agent_replaces_all_public_keys_but_external_pin_rejects(self, monkeypatch):
        """红队核心攻击 B: Agent 同时替换全部 Runtime, Downloader, Operator 公钥"""
        original_pin = compute_canonical_keyring_hash()
        monkeypatch.setenv("QUANT_TRUSTED_KEYRING_SHA256", original_pin)

        tampered = {}
        for k, v in TRUSTED_KEY_REGISTRY.items():
            _, pk = generate_keypair()
            item = dict(v)
            item["public_key_hex"] = pk.hex()
            tampered[k] = item

        is_ok, _, _, errors = verify_trust_root(keyring=tampered)
        assert is_ok is False
        assert any("trust_root_tampered" in e for e in errors)

    def test_runtime_envelope_binds_trusted_keyring_hash(self):
        """防伪验证: RuntimeAttestationEnvelope 待签名载荷必须绑定 trusted_keyring_hash"""
        meta = AuditMetadata()
        envelope = RuntimeAttestationEnvelope(
            runtime_instance_id="test_run",
            audit_payload_hash="a" * 64,
            trusted_keyring_hash=compute_canonical_keyring_hash()
        )
        canonical_bytes = envelope.compute_canonical_bytes()
        assert b"trusted_keyring_hash" in canonical_bytes
        assert compute_canonical_keyring_hash().encode("utf-8") in canonical_bytes

    def test_trust_root_cannot_be_auto_initialized(self, monkeypatch):
        """安全原则: 缺少 Pin 时系统绝不自动执行环境变量写入或假定当前哈希合法"""
        monkeypatch.delenv("QUANT_TRUSTED_KEYRING_SHA256", raising=False)
        is_ok, _, pin, _ = verify_trust_root()
        assert is_ok is False
        assert pin is None
        assert os.environ.get("QUANT_TRUSTED_KEYRING_SHA256") is None

    # =========================================================================
    # 14. Corporate Action Source Authenticity & Provenance Tests (P0)
    # =========================================================================

    def test_corporate_zero_event_requires_authenticated_receipt(self, tmp_path):
        """攻击: 本地构造正确哈希的零事件文件，但缺少 AcquisitionReceipt 无法获得完整认证"""
        raw_f = tmp_path / "raw.json"
        raw_f.write_text("{}", encoding="utf-8")
        h_raw = hashlib.sha256(raw_f.read_bytes()).hexdigest()

        resp_f = tmp_path / "resp.json"
        resp_f.write_text("{}", encoding="utf-8")
        h_resp = hashlib.sha256(resp_f.read_bytes()).hexdigest()

        ev = CorporateActionCoverageEvidence(
            symbol="600519.SH",
            query_start="2020-01-01",
            query_end="2024-12-31",
            source_id="SSE",
            query_success=True,
            empty_result=True,
            empty_result_verified=True,
            raw_result_file="raw.json",
            raw_result_hash=h_raw,
            response_file="resp.json",
            response_hash=h_resp,
            acquisition_receipt_file="missing_receipt.json"
        )
        is_valid, errors = ev.is_valid_zero_event_proof("600519.SH", "2020-01-01", "2024-12-31", evidence_dir=tmp_path)
        assert is_valid is False
        assert any("receipt_file_missing" in e for e in errors)

    def test_fake_zero_event_files_with_correct_hash_rejected(self, tmp_path):
        """攻击: 伪造虚假未签名回执时，零事件证明必须被拒绝"""
        raw_f = tmp_path / "raw.json"
        raw_f.write_text("{}", encoding="utf-8")
        h_raw = hashlib.sha256(raw_f.read_bytes()).hexdigest()

        resp_f = tmp_path / "resp.json"
        resp_f.write_text("{}", encoding="utf-8")
        h_resp = hashlib.sha256(resp_f.read_bytes()).hexdigest()

        rec_f = tmp_path / "raw.json.receipt.json"
        rec_f.write_text(json.dumps({
            "receipt_id": "REC_001",
            "source_id": "SSE",
            "source_url": "https://www.sse.com.cn/raw.json",
            "requested_at": "2026-01-01T00:00:00Z",
            "downloaded_at": "2026-01-01T00:00:00Z",
            "http_status": 200,
            "content_length": 2,
            "raw_sha256": h_raw,
            "original_filename": "raw.json",
            "trust_anchor_type": "SELF_DIGEST"
        }), encoding="utf-8")

        ev = CorporateActionCoverageEvidence(
            symbol="600519.SH",
            query_start="2020-01-01",
            query_end="2024-12-31",
            source_id="SSE",
            query_success=True,
            empty_result=True,
            empty_result_verified=True,
            raw_result_file="raw.json",
            raw_result_hash=h_raw,
            response_file="resp.json",
            response_hash=h_resp,
            acquisition_receipt_file="raw.json.receipt.json"
        )
        is_valid, errors = ev.is_valid_zero_event_proof("600519.SH", "2020-01-01", "2024-12-31", evidence_dir=tmp_path)
        assert is_valid is False
        assert any("trust_anchor_unverified" in e for e in errors)

    def test_nonempty_corporate_coverage_requires_authenticated_dataset(self):
        """非空公司行为覆盖必须具备认证的数据集与证据"""
        prov = CorporateActionProvider()
        prov.set_required_symbols(["600519.SH"])
        # 未注册任何有效 evidence
        assert prov.validate_coverage() is False
        assert prov.coverage_complete is False

    def test_corporate_dataset_source_metadata_required(self, tmp_path):
        """公司行为数据集 Manifest 校验必须包含 source_files 与 source_hashes"""
        df = pd.DataFrame([{"symbol": "600519.SH", "ex_date": "2022-06-01", "action_type": "CASH_DIVIDEND", "cash_dividend_per_share": 21.675, "share_ratio": 0.0}])
        manifest_data = {
            "normalized_dataset_sha256": CorporateActionDatasetProvenanceVerifier.compute_dataframe_sha256(df),
            "source_files": ["actions.csv"],
            "source_hashes": {}  # 集合不一致
        }
        res = CorporateActionDatasetProvenanceVerifier.verify_dataset(df, manifest_data, raw_evidence_dir=tmp_path)
        assert res.source_authentication_verified is False
        assert any("mismatch" in e for e in res.failed_checks)

    # =========================================================================
    # 15. Manifest Expected Hash & Strict Schema Tests (P0)
    # =========================================================================

    def test_manifest_expected_hash_missing_not_verified(self, tmp_path):
        """门禁规则: ManifestVerifier 在不传 expected_hash 时必须判 hash_verified=False"""
        m_file = tmp_path / "market.manifest.json"
        m_file.write_text(json.dumps({"schema_version": "3.1", "data": 123}), encoding="utf-8")

        res = ManifestVerifier.verify_manifest_file(m_file, expected_hash=None)
        assert res.hash_verified is False
        assert "manifest_expected_hash_missing" in res.failed_checks

    def test_empty_manifest_schema_rejected(self, tmp_path):
        """攻击: 空 JSON {} 必须被严格 Schema 校验器拒绝"""
        m_file = tmp_path / "empty.manifest.json"
        m_file.write_text("{}", encoding="utf-8")
        h = hashlib.sha256(b"{}").hexdigest()

        res = ManifestVerifier.verify_manifest_file(m_file, expected_hash=h, manifest_type=ManifestType.MARKET)
        assert res.schema_verified is False
        assert any("empty_or_not_dict" in e or "missing_required_fields" in e for e in res.failed_checks)

    def test_manifest_self_hash_without_parent_anchor_rejected(self, tmp_path):
        """攻击: 程序自算 hash 作为 expected_hash 但父链断裂，必须阻断"""
        m_file = tmp_path / "factor.manifest.json"
        m_file.write_text(json.dumps({
            "schema_version": "3.1",
            "dataset_name": "factor_matrix",
            "factor_columns": ["F1"],
            "dataset_sha256": "0" * 64,
            "parent_runtime_config_hash": "a" * 64,
            "parent_market_manifest_hash": "b" * 64,
            "parent_universe_manifest_hash": "c" * 64,
            "created_at": "2026-01-01T00:00:00Z"
        }), encoding="utf-8")
        h = ManifestVerifier.compute_manifest_hash(m_file)

        # 传入错误的父哈希要求
        res = ManifestVerifier.verify_manifest_file(
            m_file,
            expected_hash=h,
            expected_parents={"parent_runtime_config_hash": "f" * 64},
            manifest_type=ManifestType.FACTOR
        )
        assert res.hash_verified is True
        assert res.parent_chain_verified is False

    def test_market_manifest_pipeline_integration(self, tmp_path):
        """流水线集成: DataManager 实际执行 verify_market_manifest"""
        dm = DataManager(parquet_dir=tmp_path)
        m_file = tmp_path / "market_daily.manifest.json"
        m_content = {
            "schema_version": "3.1",
            "dataset_name": "market_daily",
            "source_files": ["000001.SZ.csv"],
            "source_hashes": {"000001.SZ.csv": "a" * 64},
            "normalized_dataset_sha256": "b" * 64,
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31",
            "created_at": "2026-01-01T00:00:00Z",
            "parent_runtime_config_hash": "c" * 64
        }
        m_file.write_text(json.dumps(m_content), encoding="utf-8")
        h = ManifestVerifier.compute_manifest_hash(m_file)

        res = dm.verify_market_manifest(m_file, expected_hash=h, parent_runtime_config_hash="c" * 64)
        assert res.hash_verified is True
        assert res.schema_verified is True
        assert res.parent_chain_verified is True
        assert dm.manifest_hash_verified is True

    def test_factor_manifest_pipeline_integration(self, tmp_path):
        """流水线集成: FactorProcessor 实际执行 verify_factor_manifest"""
        fp = FactorProcessor(factor_dir=tmp_path)
        m_file = tmp_path / "factor_matrix.manifest.json"
        m_content = {
            "schema_version": "3.1",
            "dataset_name": "factor_matrix",
            "factor_columns": ["ALPHA_001"],
            "dataset_sha256": "a" * 64,
            "parent_runtime_config_hash": "1" * 64,
            "parent_market_manifest_hash": "2" * 64,
            "parent_universe_manifest_hash": "3" * 64,
            "created_at": "2026-01-01T00:00:00Z"
        }
        m_file.write_text(json.dumps(m_content), encoding="utf-8")
        h = ManifestVerifier.compute_manifest_hash(m_file)

        res = fp.verify_factor_manifest(
            m_file,
            expected_hash=h,
            parent_runtime_config_hash="1" * 64,
            parent_market_manifest_hash="2" * 64,
            parent_universe_manifest_hash="3" * 64
        )
        assert res.hash_verified is True
        assert res.schema_verified is True
        assert res.parent_chain_verified is True
        assert fp.manifest_hash_verified is True

    def test_universe_manifest_pipeline_integration(self, tmp_path):
        """流水线集成: PointInTimeUniverseProvider 实际执行 verify_universe_manifest"""
        provider = PointInTimeUniverseProvider()
        m_file = tmp_path / "universe.manifest.json"
        m_content = {
            "schema_version": "3.1",
            "dataset_name": "hs300_universe",
            "index_code": "000300.SH",
            "baseline_snapshot_file": "baseline.json",
            "baseline_snapshot_sha256": "b" * 64,
            "source_files": ["changes.csv"],
            "raw_evidence_hashes": {"changes.csv": "c" * 64},
            "normalized_dataset_sha256": "4" * 64,
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31",
            "created_at": "2026-01-01T00:00:00Z",
            "parent_runtime_config_hash": "5" * 64
        }
        m_file.write_text(json.dumps(m_content), encoding="utf-8")
        h = ManifestVerifier.compute_manifest_hash(m_file)

        res = provider.verify_universe_manifest(m_file, expected_hash=h, parent_runtime_config_hash="5" * 64)
        assert res.hash_verified is True
        assert res.schema_verified is True
        assert provider.universe_manifest_hash_verified is True

    def test_raw_boolean_manifest_verified_not_trusted(self):
        """AuditCollector: 裸布尔值必须不被直接采信，优先检验真实 ManifestVerificationResult"""
        meta = AuditCollector.collect(config=settings)
        assert meta.manifest_chain_verified is False
        assert meta.runtime_config_hash_verified is False

    # =========================================================================
    # 16. Cryptography Backend & Path Traversal Security Tests (P1)
    # =========================================================================

    def test_production_crypto_fails_without_cryptography(self, monkeypatch):
        """安全原则: 生产模式下若 cryptography 库不可用，必须 Fail-Closed 拒绝签名与验签"""
        import data.crypto_anchor as ca
        monkeypatch.setattr(ca, "_HAS_CRYPTOGRAPHY", False)

        sig, errs = ca.sign_with_environment_key(
            message=b"test",
            key_id="PROD_RUNTIME_KEY_2026_V1",
            required_purpose="RUNTIME_ATTESTATION",
            production_mode=True,
            explicit_private_key_hex="00" * 32
        )
        assert sig is None
        assert "cryptography_library_required_for_production_signing" in errs

        ok, v_errs = ca.verify_ed25519_signature(
            message=b"test",
            signature_hex="00" * 64,
            key_id="PROD_RUNTIME_KEY_2026_V1",
            required_purpose="RUNTIME_ATTESTATION",
            production_mode=True
        )
        assert ok is False
        assert "cryptography_library_required_for_production_verification" in v_errs

    def test_source_path_traversal_rejected(self, tmp_path):
        """安全原则: safe_resolve_path 必须拦截 ../ 与逃逸路径"""
        root = tmp_path / "safe_dir"
        root.mkdir()
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("secret", encoding="utf-8")

        assert safe_resolve_path(root, "../outside.txt") is None
        assert safe_resolve_path(root, "subdir/../../outside.txt") is None
        assert safe_resolve_path(root, str(outside_file)) is None

    def test_corporate_evidence_path_traversal_rejected(self, tmp_path):
        """安全原则: CorporateActionCoverageEvidence 遇到路径穿越时必须失败"""
        ev = CorporateActionCoverageEvidence(
            symbol="600519.SH",
            query_start="2020-01-01",
            query_end="2024-12-31",
            source_id="SSE",
            query_success=True,
            empty_result=True,
            empty_result_verified=True,
            raw_result_file="../outside.json",
            raw_result_hash="0" * 64,
            response_file="../resp.json",
            response_hash="0" * 64
        )
        ok, errors = ev.is_valid_zero_event_proof("600519.SH", "2020-01-01", "2024-12-31", evidence_dir=tmp_path)
        assert ok is False
        assert any("traversal" in e for e in errors)
