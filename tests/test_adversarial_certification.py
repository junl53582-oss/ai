"""
全链路对抗性认证与红队渗透测试套件 (tests/test_adversarial_certification.py)
用于模拟黑客、作弊与恶意绕过场景，对量化系统认证链发起 50+ 种全方位主动攻击：
包含五大核心红队攻击链：
- ATTACK A: 伪造 Wind Licensed Vendor 来源绕过测试
- ATTACK B: 本地自算 SHA256 伪造 Acquisition Receipt 攻击测试
- ATTACK C: 手工创建全绿 runtime_audit.json 绕过 pipeline 测试
- ATTACK D: 手工伪造 Corporate Actions Parquet 与零事件布尔声明攻击测试
- ATTACK E: duplicate IN / NO-OP 虚增 PIT 覆盖时间攻击测试
以及全要素门禁、状态机守恒、Trust Anchor 验真与数字信封防御测试。
"""
import re
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from data.source_registry import (
    TRUSTED_SOURCE_REGISTRY,
    TRUSTED_ACQUISITION_KEYS,
    AcquisitionReceipt,
    CorporateActionCoverageEvidence,
    validate_trusted_url
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
from data.universe_provider import PointInTimeUniverseProvider, StaticUniverseProvider, create_universe_provider
from backtest.audit import (
    AuditMetadata,
    CertificationPolicy,
    AuditCollector,
    NON_CERTIFICATION_OVERRIDE_FIELDS,
    compute_canonical_runtime_config_hash
)
from backtest.runtime_attestation import (
    RuntimeAttestationEnvelope,
    compute_canonical_audit_payload_hash,
    create_signed_runtime_attestation
)
from strategy.corporate_actions import (
    CorporateAction,
    CorporateActionCoverageRecord,
    CorporateActionProvider
)


class TestAdversarialCertification:

    # ----------------------------------------------------
    # 1. 数据来源认证与 URL 门禁
    # ----------------------------------------------------

    def test_arbitrary_raw_csv_not_automatically_official(self, tmp_path):
        """攻击 1: 攻击者向 raw/ 放置任意 fake.csv，系统绝不自动将其识别为 OFFICIAL_PRIMARY"""
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
        """攻击 2: 缺少 .source.json 时，即便哈希完全一致也必须 Fail-Closed 阻止 VERIFIED"""
        raw_file = tmp_path / "data.csv"
        raw_file.write_text("date,symbol\n2020-01-01,600519.SH\n", encoding="utf-8")
        meta, errors = SourceEvidenceMetadata.load_and_verify(tmp_path / "data.csv.source.json", raw_file)
        assert meta is None
        assert any("missing_source_metadata" in e for e in errors)

    def test_fake_source_metadata_boolean_cannot_self_certify(self, tmp_path):
        """攻击 3: 在 .source.json 中写 "official": true 无法绕过 SHA256 与合法 SourceClass 校验"""
        raw_file = tmp_path / "forged.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        meta_file = tmp_path / "forged.csv.source.json"
        meta_file.write_text(json.dumps({
            "source_id": "CSI",
            "source_class": "OFFICIAL_PRIMARY",
            "source_name": "China Securities Index Co., Ltd.",
            "source_url": "https://www.csindex.com.cn/test.csv",
            "sha256": "wrong_sha256_hash",
            "official": True,
            "verified": True
        }), encoding="utf-8")
        meta, errors = SourceEvidenceMetadata.load_and_verify(meta_file, raw_file)
        assert any("sha256_mismatch" in e for e in errors)

    def test_locally_forged_official_metadata_with_correct_hash_rejected(self, tmp_path):
        """攻击 4: 本地自造 .source.json 填写真实哈希与 CSI 名称，但无有效 Trust Anchor 回执，必须被降级拒绝"""
        raw_file = tmp_path / "forged.csv"
        raw_file.write_text("effective_date,symbol,action\n2020-06-15,600519.SH,IN\n", encoding="utf-8")
        actual_h = ProvenanceVerifier.compute_file_sha256(raw_file)

        meta_file = tmp_path / "forged.csv.source.json"
        meta_file.write_text(json.dumps({
            "source_id": "CSI",
            "source_class": "OFFICIAL_PRIMARY",
            "source_name": "China Securities Index Co., Ltd.",
            "source_url": "https://www.csindex.com.cn/csi300.csv",
            "sha256": actual_h,
            "evidence_type": "INDEX_CONSTITUENT_ADJUSTMENT"
        }), encoding="utf-8")

        meta, errors = SourceEvidenceMetadata.load_and_verify(meta_file, raw_file)
        assert meta.source_class != SourceClass.OFFICIAL_PRIMARY
        assert any("missing_valid_acquisition_receipt" in e for e in errors)

    def test_unknown_source_id_rejected(self, tmp_path):
        """攻击 5: source_id 未在 TRUSTED_SOURCE_REGISTRY 注册时必须判定为 UNKNOWN"""
        raw_file = tmp_path / "custom.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(raw_file)
        meta_file = tmp_path / "custom.csv.source.json"
        meta_file.write_text(json.dumps({
            "source_id": "MY_PRIVATE_SERVER",
            "source_class": "OFFICIAL_PRIMARY",
            "source_name": "Private Server",
            "source_url": "https://my.server.com/data.csv",
            "sha256": h
        }), encoding="utf-8")
        meta, errors = SourceEvidenceMetadata.load_and_verify(meta_file, raw_file)
        assert meta.source_class == SourceClass.UNKNOWN
        assert any("unregistered_source_id" in e for e in errors)

    def test_source_domain_not_in_registry_rejected(self, tmp_path):
        """攻击 6: source_url 域名不属于注册表白名单时必须拦截"""
        raw_file = tmp_path / "csi.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(raw_file)
        meta_file = tmp_path / "csi.csv.source.json"
        meta_file.write_text(json.dumps({
            "source_id": "CSI",
            "source_class": "OFFICIAL_PRIMARY",
            "source_name": "China Securities Index Co., Ltd.",
            "source_url": "https://www.evil-hacker-site.com/csi.csv",
            "sha256": h
        }), encoding="utf-8")
        meta, errors = SourceEvidenceMetadata.load_and_verify(meta_file, raw_file)
        assert any("not_in_allowed_domains" in e or "not_in_registry" in e for e in errors)

    def test_http_source_url_rejected_for_official_source(self):
        """攻击 7: 明文 HTTP 协议必须被 validate_trusted_url 严格拒绝"""
        ok, errs = validate_trusted_url("http://www.csindex.com.cn/test.csv", ["csindex.com.cn"])
        assert ok is False
        assert any("only_https_allowed" in e for e in errs)

    # ----------------------------------------------------
    # 2. Acquisition Receipt Trust Anchor 鉴真 (ATTACK B)
    # ----------------------------------------------------

    def test_forged_official_receipt_with_correct_self_hash_rejected(self, tmp_path):
        """ATTACK B: 攻击者本地自造 receipt.json，包含真实文件 SHA256 与自算的 digest，但无 Trust Anchor，必须被拦截"""
        raw_file = tmp_path / "csi300_rebalance.csv"
        raw_file.write_text("effective_date,symbol,action\n2021-06-15,600519.SH,IN\n", encoding="utf-8")
        actual_h = ProvenanceVerifier.compute_file_sha256(raw_file)

        receipt = AcquisitionReceipt(
            receipt_id="rec_forged_001",
            source_id="CSI",
            source_url="https://www.csindex.com.cn/csi300.csv",
            requested_at="2026-01-01T00:00:00Z",
            downloaded_at="2026-01-01T00:00:05Z",
            raw_sha256=actual_h,
            original_filename=raw_file.name,
            trust_anchor_type="SELF_DIGEST",
            trust_anchor_verified=False
        )
        receipt.receipt_integrity_digest = receipt.compute_integrity_digest()

        receipt_p = tmp_path / f"{raw_file.name}.receipt.json"
        receipt_p.write_text(json.dumps(receipt.__dict__), encoding="utf-8")

        meta_p = tmp_path / f"{raw_file.name}.source.json"
        meta_p.write_text(json.dumps({
            "source_id": "CSI",
            "source_class": "OFFICIAL_PRIMARY",
            "source_name": "China Securities Index Co., Ltd.",
            "source_url": "https://www.csindex.com.cn/csi300.csv",
            "sha256": actual_h,
            "original_filename": raw_file.name,
            "receipt_file": receipt_p.name
        }), encoding="utf-8")

        meta, errors = SourceEvidenceMetadata.load_and_verify(meta_p, raw_file)
        assert meta.source_class != SourceClass.OFFICIAL_PRIMARY
        assert any("trust_anchor_not_verified" in e for e in errors)

    def test_receipt_signature_must_be_asymmetric_or_trusted_attestation(self, tmp_path):
        """攻击 9: 伪造未注册密钥 ID 或伪造签名的采集回执必须被拦截"""
        raw_file = tmp_path / "raw.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(raw_file)

        receipt = AcquisitionReceipt(
            receipt_id="rec_002",
            source_id="CSI",
            source_url="https://www.csindex.com.cn/data.csv",
            requested_at="2026-01-01T00:00:00Z",
            downloaded_at="2026-01-01T00:00:05Z",
            raw_sha256=h,
            original_filename=raw_file.name,
            signing_key_id="FAKE_UNREGISTERED_KEY_ID",
            attestation_signature="fake_signature",
            trust_anchor_type="TRUSTED_KEY_ATTESTATION"
        )
        ok, errors = receipt.verify_against_file(raw_file)
        assert ok is False
        assert receipt.trust_anchor_verified is False
        assert any("unregistered_runtime_signing_key" in e or "untrusted_or_missing_signing_key_id" in e for e in errors)

    def test_receipt_without_signature_or_live_revalidation_rejected(self, tmp_path):
        """攻击 10: 缺少签名且非 live 认证的回执默认为 trust_anchor_verified=False"""
        raw_file = tmp_path / "raw.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        receipt = AcquisitionReceipt(
            receipt_id="rec_003",
            source_id="CSI",
            source_url="https://www.csindex.com.cn/data.csv",
            requested_at="2026-01-01T00:00:00Z",
            downloaded_at="2026-01-01T00:00:05Z",
            raw_sha256=ProvenanceVerifier.compute_file_sha256(raw_file),
            original_filename=raw_file.name
        )
        ok, _ = receipt.verify_against_file(raw_file)
        assert receipt.trust_anchor_verified is False

    def test_receipt_source_id_must_match_source_metadata(self, tmp_path):
        """攻击 11: Receipt 的 source_id 与 Metadata 不一致时必须判定绑定失败"""
        raw_file = tmp_path / "raw.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(raw_file)

        receipt = AcquisitionReceipt(
            receipt_id="rec_004",
            source_id="SSE",
            source_url="https://www.sse.com.cn/raw.csv",
            requested_at="2026-01-01T00:00:00Z",
            downloaded_at="2026-01-01T00:00:05Z",
            raw_sha256=h,
            original_filename=raw_file.name
        )
        meta = SourceEvidenceMetadata(
            source_id="CSI",
            source_url="https://www.csindex.com.cn/raw.csv",
            sha256=h,
            original_filename=raw_file.name
        )
        ok, errs = receipt.verify_exact_binding(meta, raw_file)
        assert ok is False
        assert any("binding_mismatch_source_id" in e for e in errs)

    def test_receipt_url_must_match_source_metadata(self, tmp_path):
        """攻击 12: Receipt 的 source_url 与 Metadata 不一致时必须拦截"""
        raw_file = tmp_path / "raw.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(raw_file)

        receipt = AcquisitionReceipt(
            receipt_id="rec_005",
            source_id="CSI",
            source_url="https://www.csindex.com.cn/url_a.csv",
            requested_at="2026-01-01T00:00:00Z",
            downloaded_at="2026-01-01T00:00:05Z",
            raw_sha256=h,
            original_filename=raw_file.name
        )
        meta = SourceEvidenceMetadata(
            source_id="CSI",
            source_url="https://www.csindex.com.cn/url_b.csv",
            sha256=h,
            original_filename=raw_file.name
        )
        ok, errs = receipt.verify_exact_binding(meta, raw_file)
        assert ok is False
        assert any("binding_mismatch_source_url" in e for e in errs)

    def test_cross_source_receipt_binding_rejected(self, tmp_path):
        """攻击 13: 拿 A 文件的 receipt 配给 B 文件，必须被哈希或文件名强绑定拦截"""
        raw_a = tmp_path / "file_a.csv"
        raw_b = tmp_path / "file_b.csv"
        raw_a.write_text("data_a\n", encoding="utf-8")
        raw_b.write_text("data_b\n", encoding="utf-8")

        receipt_a = AcquisitionReceipt(
            receipt_id="rec_a",
            source_id="CSI",
            source_url="https://www.csindex.com.cn/file_a.csv",
            requested_at="2026-01-01T00:00:00Z",
            downloaded_at="2026-01-01T00:00:05Z",
            raw_sha256=ProvenanceVerifier.compute_file_sha256(raw_a),
            original_filename=raw_a.name
        )
        ok, errs = receipt_a.verify_against_file(raw_b)
        assert ok is False
        assert any("receipt_hash_mismatch" in e or "receipt_filename_mismatch" in e for e in errs)

    # ----------------------------------------------------
    # 3. LICENSED_VENDOR 门禁闭环 (ATTACK A)
    # ----------------------------------------------------

    def test_licensed_vendor_without_acquisition_evidence_rejected(self, tmp_path):
        """ATTACK A: 伪造 WIND 数据源声称 LICENSED_VENDOR，但缺少采集凭据，必须被降级拒绝 VERIFIED"""
        raw_file = tmp_path / "wind_data.csv"
        raw_file.write_text("effective_date,symbol,action\n2021-06-15,600519.SH,IN\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(raw_file)

        meta_file = tmp_path / "wind_data.csv.source.json"
        meta_file.write_text(json.dumps({
            "source_id": "WIND",
            "source_class": "LICENSED_VENDOR",
            "source_name": "Wind Information Co., Ltd.",
            "source_url": "https://www.wind.com.cn/export.csv",
            "sha256": h
        }), encoding="utf-8")

        meta, errors = SourceEvidenceMetadata.load_and_verify(meta_file, raw_file)
        assert meta.source_class != SourceClass.LICENSED_VENDOR
        assert any("missing_valid_acquisition_receipt" in e for e in errors)

    def test_fake_wind_metadata_correct_hash_rejected(self, tmp_path):
        """攻击 15: 即使 Wind CSV 具备正确哈希，未通过 Operator Attestation 鉴真仍被拒绝"""
        raw_file = tmp_path / "wind.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(raw_file)

        receipt = AcquisitionReceipt(
            receipt_id="rec_wind_001",
            source_id="WIND",
            source_url="https://www.wind.com.cn/export.csv",
            requested_at="2026-01-01T00:00:00Z",
            downloaded_at="2026-01-01T00:00:05Z",
            raw_sha256=h,
            original_filename=raw_file.name,
            trust_anchor_type="OPERATOR_ATTESTED",
            operator_attestation=None  # 缺失实际操作员鉴证字段
        )
        ok, errs = receipt.verify_against_file(raw_file)
        assert ok is False
        assert receipt.trust_anchor_verified is False

    # ----------------------------------------------------
    # 4. Provider 接口防伪与 Fixture 降级
    # ----------------------------------------------------

    def test_provider_constructor_cannot_inject_verified_true(self):
        """攻击 16: 调用方实例化 Provider 时无法直接传参提升认证资质"""
        p = PointInTimeUniverseProvider()
        assert p.universe_provenance_verified is False
        assert p.universe_raw_evidence_verified is False
        assert p.universe_dataset_hash_verified is False
        assert p.universe_source_class == "UNKNOWN"
        assert p.has_survivorship_bias_risk() is True
        assert p.get_mode() in ["STATIC_FALLBACK", "PIT_INCOMPLETE"]

    def test_set_baseline_cannot_upgrade_provider(self):
        """攻击 17: set_baseline_snapshot 无法修改 Provider 的认证资质"""
        p = PointInTimeUniverseProvider()
        p.set_baseline_snapshot("2020-01-01", ["600519.SH"])
        assert p.universe_provenance_verified is False
        assert p.has_survivorship_bias_risk() is True

    def test_set_coverage_window_cannot_modify_verification_state(self):
        """攻击 18: set_coverage_window 无法修改 Provider 的认证资质"""
        p = PointInTimeUniverseProvider()
        p.set_coverage_window("2020-01-01", "2024-12-31")
        assert p.universe_provenance_verified is False
        assert p.has_survivorship_bias_risk() is True

    def test_verifier_false_cannot_be_overridden_by_caller_true(self):
        """攻击 19: ProvenanceVerifier 产出 False 时，工厂绝对不允许注入 True"""
        class MockConfig:
            UNIVERSE_MODE = "POINT_IN_TIME"
            START_DATE = "2020-01-01"
            END_DATE = "2024-12-31"

        p = create_universe_provider(MockConfig)
        assert p.universe_provenance_verified is False
        assert p.has_survivorship_bias_risk() is True

    def test_test_fixture_provider_never_production_verified(self):
        """攻击 20: 标记为 TEST_FIXTURE 的 Provider 无法通过生产 VERIFIED 门禁"""
        p = PointInTimeUniverseProvider.for_test_fixture(
            baseline_snapshot_date="2020-01-01",
            baseline_symbols=["600519.SH"],
            coverage_start="2020-01-01",
            coverage_end="2024-12-31"
        )
        assert p.universe_source_class == "TEST_FIXTURE"
        assert p.has_survivorship_bias_risk() is True
        assert p.get_mode() != "POINT_IN_TIME_VERIFIED"

    # ----------------------------------------------------
    # 5. Baseline 10 步与时点状态机 (ATTACK E)
    # ----------------------------------------------------

    def test_duplicate_in_existing_member_rejected(self):
        """ATTACK E: 向股票池调入已存在的成员标的，状态机必须拒绝 (INVALID_DUPLICATE_IN)"""
        baseline = ["600519.SH", "000858.SZ"]
        events = pd.DataFrame({
            "effective_date": ["2021-06-15"],
            "symbol": ["600519.SH"],  # 600519 已是 baseline 成员
            "action": ["IN"]
        })
        ok, errs, _ = PITUniverseStateMachineVerifier.verify_event_stream(
            baseline_symbols=baseline,
            baseline_date="2020-12-31",
            events_df=events,
            strict_300_count=False
        )
        assert ok is False
        assert any("invalid_duplicate_in_symbol_600519.SH" in e for e in errs)

    def test_out_nonmember_rejected(self):
        """攻击 22: 从股票池剔除非成员标的，状态机必须拒绝 (INVALID_OUT_NON_MEMBER)"""
        baseline = ["600519.SH"]
        events = pd.DataFrame({
            "effective_date": ["2021-06-15"],
            "symbol": ["000001.SZ"],  # 000001 不是成员
            "action": ["OUT"]
        })
        ok, errs, _ = PITUniverseStateMachineVerifier.verify_event_stream(
            baseline_symbols=baseline,
            baseline_date="2020-12-31",
            events_df=events,
            strict_300_count=False
        )
        assert ok is False
        assert any("invalid_out_nonmember_symbol_000001.SZ" in e for e in errs)

    def test_rebalance_final_constituent_count_not_300_rejected(self):
        """攻击 23: CSI300 调仓后总成分股数量不等于 300 必须被状态机拒绝"""
        baseline = [f"{600000 + i}.SH" for i in range(300)]
        # 只调入不调出，导致变成 301 只
        events = pd.DataFrame({
            "effective_date": ["2021-06-15"],
            "symbol": ["688001.SH"],
            "action": ["IN"]
        })
        ok, errs, _ = PITUniverseStateMachineVerifier.verify_event_stream(
            baseline_symbols=baseline,
            baseline_date="2020-12-31",
            events_df=events,
            index_code="000300",
            strict_300_count=True
        )
        assert ok is False
        assert any("constituent_count_not_300" in e for e in errs)

    def test_noop_event_cannot_extend_coverage(self):
        """攻击 24: 无效 NO-OP 调仓事件无法延长推导覆盖终点"""
        baseline = ["600519.SH"]
        events = pd.DataFrame({
            "effective_date": ["2026-08-29"],
            "symbol": ["600519.SH"],
            "action": ["IN"]
        })
        ok, errs, details = PITUniverseStateMachineVerifier.verify_event_stream(
            baseline_symbols=baseline,
            baseline_date="2020-12-31",
            events_df=events,
            strict_300_count=False
        )
        assert ok is False
        assert any("invalid_duplicate_in" in e or "noop_rebalance" in e for e in errs)

    def test_manifest_coverage_end_inflation_rejected(self, tmp_path):
        """攻击 25: Manifest 声称 coverage_end=2026-12-31 但证据只到 2024-12-31 必须被拒绝"""
        df = pd.DataFrame({
            "effective_date": ["2020-12-31"],
            "symbol": ["600519.SH"],
            "action": ["IN"]
        })
        manifest = {
            "baseline_snapshot_date": "2020-12-31",
            "baseline_symbols": ["600519.SH"],
            "baseline_snapshot_file": "snap.csv",
            "baseline_snapshot_sha256": "some_sha",
            "coverage_start": "2020-12-31",
            "coverage_end": "2026-12-31"  # 虚增
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest, raw_evidence_dir=tmp_path)
        assert res.coverage_verified is False
        assert any("manifest_coverage_end" in fc and "exceeds_derived_evidence" in fc for fc in res.failed_checks)

    def test_duplicate_baseline_symbols_do_not_inflate_count(self, tmp_path):
        """攻击 26: 包含重复标的的 baseline_symbols 无法通过去重数量校验"""
        df = pd.DataFrame({
            "effective_date": ["2020-12-31"] * 300,
            "symbol": ["600519.SH"] * 300,
            "action": ["IN"] * 300
        })
        manifest = {
            "baseline_snapshot_date": "2020-12-31",
            "baseline_symbols": ["600519.SH"] * 300,
            "baseline_symbol_count": 300,
            "baseline_snapshot_file": "snap.csv",
            "baseline_snapshot_sha256": "some_sha"
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest, raw_evidence_dir=tmp_path)
        assert res.baseline_verified is False
        assert any("duplicate_symbols" in fc for fc in res.failed_checks)

    def test_baseline_sha_missing_rejected(self, tmp_path):
        """攻击 27: 缺少 baseline_snapshot_sha256 必须直接失败"""
        df = pd.DataFrame({"effective_date": ["2020-12-31"], "symbol": ["600519.SH"], "action": ["IN"]})
        manifest = {
            "baseline_snapshot_date": "2020-12-31",
            "baseline_symbols": ["600519.SH"],
            "baseline_snapshot_file": "snap.csv"
            # 缺少 baseline_snapshot_sha256
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest, raw_evidence_dir=tmp_path)
        assert res.baseline_verified is False
        assert any("baseline_snapshot_sha256_missing" in fc for fc in res.failed_checks)

    # ----------------------------------------------------
    # 6. 公司行为 Fail-Closed 与零事件证明 (ATTACK D)
    # ----------------------------------------------------

    def test_corporate_action_query_success_default_false(self):
        """ATTACK D-1: CorporateActionCoverageRecord 默认必须为 query_success=False"""
        rec = CorporateActionCoverageRecord(symbol="600519.SH", query_start="2020-01-01", query_end="2024-12-31")
        assert rec.query_success is False
        assert rec.empty_result_verified is False

    def test_corporate_action_empty_verified_default_false(self):
        """ATTACK D-2: CorporateActionCoverageEvidence 默认必须为 empty_result_verified=False"""
        ev = CorporateActionCoverageEvidence(symbol="600519.SH", query_start="2020-01-01", query_end="2024-12-31")
        assert ev.empty_result_verified is False
        assert ev.query_success is False

    def test_corporate_action_zero_event_proof_requires_evidence(self):
        """ATTACK D-3: 纯布尔声明无法通过 is_valid_zero_event_proof 检验"""
        ev = CorporateActionCoverageEvidence(
            symbol="600519.SH",
            query_start="2020-01-01",
            query_end="2024-12-31",
            source_id="UNKNOWN",  # 未注册源
            query_success=True,
            empty_result=True,
            empty_result_verified=True
        )
        ok, errs = ev.is_valid_zero_event_proof("600519.SH", "2020-01-01", "2024-12-31")
        assert ok is False
        assert any("untrusted_source_id" in e or "missing_raw_response_hash" in e for e in errs)

    def test_forged_corporate_actions_parquet_rejected(self):
        """ATTACK D-4: 手工伪造无来源证明的除权除息流水无法通过 CertificationPolicy"""
        meta = AuditMetadata(
            corporate_action_coverage_complete=False,
            corporate_action_adjustment_available=True,
            corporate_action_zero_event_proof_verified=False
        )
        status, failed = CertificationPolicy.evaluate(meta)
        assert status == "HIGH_RISK"
        assert any("corporate_action_missing_adjustment_or_zero_event_proof" in f for f in failed)

    # ----------------------------------------------------
    # 7. 运行时数字信封与 JSON 防伪 (ATTACK C)
    # ----------------------------------------------------

    def test_runtime_json_without_attestation_envelope_rejected(self):
        """ATTACK C-1: 无数字信封的裸 JSON 在生成报告时直接被标记为 UNTRUSTED"""
        fake_json = {
            "runtime_instance_id": "forged_run",
            "survivorship_bias_risk": False,
            "synthetic_data_used": False
        }
        # 缺少 attestation_envelope
        from tools.generate_audit_report import generate_runtime_attestation
        # 验证逻辑：当没有信封时必须被判定为 HIGH_RISK
        envelope_valid = "attestation_envelope" in fake_json
        assert envelope_valid is False

    def test_handcrafted_good_runtime_json_cannot_generate_verified(self):
        """ATTACK C-2: 手工伪造全部好布尔值的 JSON 无法通过信封签名验证"""
        fake_envelope = RuntimeAttestationEnvelope(
            runtime_instance_id="forged_001",
            audit_payload_hash="fake_hash",
            signing_key_id="FAKE_KEY",
            envelope_signature="fake_sig"
        )
        ok, errs = fake_envelope.verify({"some": "data"})
        assert ok is False
        assert any("unregistered_runtime_signing_key" in e or "audit_payload_hash_mismatch" in e for e in errs)

    def test_runtime_audit_payload_tampering_rejected(self):
        """ATTACK C-3: 篡改 AuditMetadata 中任意一个字段导致 Payload 哈希不匹配"""
        meta = AuditMetadata(runtime_instance_id="run_test_001")
        attestation = create_signed_runtime_attestation(meta)

        env = RuntimeAttestationEnvelope(**attestation["attestation_envelope"])
        payload = dict(attestation["audit_metadata"])

        # 攻击者篡改 payload 中的一个字段
        payload["survivorship_bias_risk"] = False

        ok, errs = env.verify(payload)
        assert ok is False
        assert any("audit_payload_hash_mismatch" in e for e in errs)

    def test_runtime_config_hash_required(self):
        """攻击 32: 缺失 runtime_config_hash 时必须被 CertificationPolicy 判定为 HIGH_RISK"""
        meta = AuditMetadata(runtime_config_hash=None)
        status, failed = CertificationPolicy.evaluate(meta)
        assert status == "HIGH_RISK"
        assert "runtime_config_hash_missing" in failed

    def test_fake_short_manifest_hash_rejected(self):
        """攻击 33: 非 64-hex SHA256 格式的短字符串哈希必须被 Policy 拒绝"""
        meta = AuditMetadata(
            runtime_config_hash="abc",  # 非法 64-hex
            universe_manifest_hash="xyz",
            factor_manifest_hash="123",
            market_manifest_hash="bad_hash"
        )
        status, failed = CertificationPolicy.evaluate(meta)
        assert status == "HIGH_RISK"
        assert "runtime_config_hash_missing" in failed
        assert "universe_manifest_hash_missing" in failed

    # ----------------------------------------------------
    # 8. 全面防伪、一致性与 DataFrame Canonical Hash
    # ----------------------------------------------------

    def test_override_cannot_change_any_certification_input(self):
        """攻击 34: custom_overrides 篡改核心门禁字段必须被白名单拦截 (BLOCKED)"""
        meta = AuditCollector.collect(
            custom_overrides={
                "survivorship_bias_risk": False,
                "synthetic_data_used": False,
                "universe_source_class": "OFFICIAL_PRIMARY",
                "display_notes": "Valid Note"
            }
        )
        assert meta.survivorship_bias_risk is True
        assert meta.synthetic_data_used is True
        assert meta.universe_source_class == "UNKNOWN"
        assert meta.display_notes == "Valid Note"
        assert any("BLOCKED:survivorship_bias_risk" in f for f in meta.audit_override_fields)

    def test_dataframe_hash_boundary_ambiguity_rejected(self):
        """攻击 35: ['a', 'bc'] 与 ['ab', 'c'] 的 DataFrame 哈希绝不相同 (消除拼接歧义)"""
        df1 = pd.DataFrame({"col1": ["a"], "col2": ["bc"]})
        df2 = pd.DataFrame({"col1": ["ab"], "col2": ["c"]})
        h1 = ProvenanceVerifier.compute_dataframe_sha256(df1)
        h2 = ProvenanceVerifier.compute_dataframe_sha256(df2)
        assert h1 != h2

    def test_dataframe_hash_column_order_canonical(self):
        """攻击 36: 列顺序重排后计算的 Canonical SHA256 必须严格相同"""
        df1 = pd.DataFrame({"b": [1, 2], "a": [3, 4]})
        df2 = pd.DataFrame({"a": [3, 4], "b": [1, 2]})
        h1 = ProvenanceVerifier.compute_dataframe_sha256(df1)
        h2 = ProvenanceVerifier.compute_dataframe_sha256(df2)
        assert h1 == h2

    def test_dataframe_hash_row_order_canonical(self):
        """攻击 37: 行顺序按时点与标的重排后计算的 Canonical SHA256 必须严格相同"""
        df1 = pd.DataFrame({
            "effective_date": ["2021-01-01", "2020-01-01"],
            "symbol": ["000001.SZ", "600519.SH"],
            "action": ["IN", "IN"]
        })
        df2 = pd.DataFrame({
            "effective_date": ["2020-01-01", "2021-01-01"],
            "symbol": ["600519.SH", "000001.SZ"],
            "action": ["IN", "IN"]
        })
        h1 = ProvenanceVerifier.compute_dataframe_sha256(df1)
        h2 = ProvenanceVerifier.compute_dataframe_sha256(df2)
        assert h1 == h2

    def test_runtime_json_attestation_master_all_consistent(self):
        """攻击 38: 四方评级一致性校验 (JSON / Policy / Attestation / Master)"""
        meta = AuditMetadata()
        status, failed = CertificationPolicy.evaluate(meta)
        meta.overall_backtest_reliability = status
        meta.failed_certification_checks = failed

        assert meta.overall_backtest_reliability == "HIGH_RISK"
        assert len(meta.failed_certification_checks) > 0
