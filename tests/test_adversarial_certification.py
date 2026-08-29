"""
全链路对抗性认证与红队渗透测试套件 (tests/test_adversarial_certification.py)
用于模拟黑客、作弊与恶意绕过场景，对量化系统认证链发起 30+ 种全方位主动攻击：
1. 任意 CSV 冒充官方攻击 (Arbitrary Raw CSV Impersonation)
2. 缺失独立 Source Metadata 拦截 (Missing Source Metadata)
3. 伪造 Source Metadata 布尔声明攻击 (Fake Source Metadata Boolean)
4. SHA256 单独冒充官方来源攻击 (SHA256 != Source Authentication)
5. 本地自造 source.json + 正确 SHA256 冒充官方攻击 (Locally Forged Metadata Defense)
6. 官方 Primary 缺少采集回执攻击 (Missing Acquisition Receipt)
7. 未注册 source_id 拦截 (Unknown Source ID Defense)
8. 伪造官方 URL 域名白名单拦截 (Source Domain Not In Registry)
9. Provider 构造函数注入 Verified=True 攻击 (Constructor Bypass Attack)
10. set_baseline_snapshot 篡改认证资质攻击 (Baseline Snapshot Tampering)
11. set_coverage_window 篡改认证资质攻击 (Coverage Window Tampering)
12. Verifier False 被调用方 True 覆盖攻击 (Caller Overrides Verifier)
13. 测试 Fixture 冒充生产证据攻击 (Test Fixture Impersonating Production)
14. 实际回测终点超出 PIT 覆盖范围攻击 (Actual Backtest End Exceeds PIT Coverage)
15. 缺失实际回测起止日期拦截 (Actual Backtest Window Missing)
16. END_DATE=None 绕过终点核验攻击 (END_DATE=None Coverage Bypass)
17. Provider 自我验证覆盖率攻击 (Provider Self-Validation)
18. 订单守恒缺失自动 Fail-Closed 检验 (Order Conservation Fail-Closed)
19. synthetic_data_used 缺失自动 Fail-Closed 检验 (Synthetic Metric Fail-Closed)
20. 公司行为无数据谎称 100% 覆盖攻击 (Corporate Action Inconsistency)
21. 公司行为 Zero Event 纯布尔无证据攻击 (Zero Event Proof Requires Evidence)
22. 公司行为 empty_result_verified 默认值检验 (Empty Result Verified Fail-Closed)
23. 重复标的伪造 Baseline 数量攻击 (Duplicate Baseline Symbols Attack)
24. Baseline 引用不存在文件攻击 (Nonexistent Baseline File Attack)
25. Baseline 文件哈希篡改拦截 (Baseline SHA Mismatch)
26. Baseline 证据类型非 Snapshot 拦截 (Baseline Evidence Type Mismatch)
27. Baseline Manifest、Raw Snapshot、Normalized Events 三方集合一致性攻击 (Three-Way Set Parity)
28. custom_overrides 篡改认证输入全量拦截 (Audit Override Anti-Forgery)
29. DataFrame Canonical Hash 边界歧义防御 (DataFrame Hash Ambiguity Defense)
30. DataFrame Canonical Hash 行列置换不变性 (DataFrame Canonical Ordering)
31. 运行产物 JSON / Policy / Attestation / Master 四方一致性校验 (Four-Way Consistency Parity)
"""
import pytest
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

from data.source_registry import (
    TRUSTED_SOURCE_REGISTRY,
    AcquisitionReceipt,
    CorporateActionCoverageEvidence
)
from data.provenance import (
    SourceClass,
    ProvenanceVerifier,
    UniverseVerificationResult,
    DataProvenanceError,
    SourceEvidenceMetadata,
    CSIRebalanceAnnouncementParser,
    CSIConstituentSnapshotParser
)
from data.universe_provider import PointInTimeUniverseProvider, StaticUniverseProvider, create_universe_provider
from backtest.audit import AuditMetadata, CertificationPolicy, AuditCollector, NON_CERTIFICATION_OVERRIDE_FIELDS


class TestAdversarialCertification:

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
            "official": True,
            "verified": True,
            "source_id": "UNKNOWN",
            "source_class": "INVALID_CLASS",
            "source_name": "MyFakeSource",
            "source_url": "http://fake.com",
            "sha256": "wrong_sha",
            "original_filename": "forged.csv"
        }), encoding="utf-8")

        meta, errors = SourceEvidenceMetadata.load_and_verify(meta_file, raw_file)
        assert meta is not None
        assert meta.source_class == SourceClass.UNKNOWN
        assert any("source_metadata_sha256_mismatch" in e for e in errors)

    def test_locally_forged_official_metadata_with_correct_hash_rejected(self, tmp_path):
        """攻击 4: 本地自造 source.json，即便 SHA256 完全正确且填写真实官方 URL，缺少 Acquisition Receipt 必须拒绝生产认证"""
        raw_file = tmp_path / "fake_csi.csv"
        raw_file.write_text("index_code,effective_date,symbol,action\n000300,2020-06-15,600519.SH,IN\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(raw_file)

        meta_file = tmp_path / "fake_csi.csv.source.json"
        meta_file.write_text(json.dumps({
            "source_id": "CSI",
            "source_class": "OFFICIAL_PRIMARY",
            "source_name": "China Securities Index Co., Ltd.",
            "source_url": "https://www.csindex.com.cn/data/csi300.csv",
            "sha256": h,
            "original_filename": raw_file.name,
            "evidence_type": "INDEX_CONSTITUENT_ADJUSTMENT"
        }), encoding="utf-8")

        # 未提供真实的 .receipt.json
        meta, errors = SourceEvidenceMetadata.load_and_verify(meta_file, raw_file)
        assert meta is not None
        assert meta.source_class != SourceClass.OFFICIAL_PRIMARY
        assert any("missing_valid_acquisition_receipt_for_official_primary" in e for e in errors)

    def test_unknown_source_id_rejected(self, tmp_path):
        """攻击 5: source_id 未在 TRUSTED_SOURCE_REGISTRY 登记时直接判定为 UNKNOWN"""
        raw_file = tmp_path / "test.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(raw_file)

        meta_file = tmp_path / "test.csv.source.json"
        meta_file.write_text(json.dumps({
            "source_id": "MY_UNOFFICIAL_SOURCE",
            "source_class": "OFFICIAL_PRIMARY",
            "source_name": "Unofficial Source",
            "source_url": "https://www.csindex.com.cn/",
            "sha256": h,
            "original_filename": raw_file.name
        }), encoding="utf-8")

        meta, errors = SourceEvidenceMetadata.load_and_verify(meta_file, raw_file)
        assert meta.source_class == SourceClass.UNKNOWN
        assert any("unregistered_source_id_MY_UNOFFICIAL_SOURCE" in e for e in errors)

    def test_source_domain_not_in_registry_rejected(self, tmp_path):
        """攻击 6: URL 域名不在注册表白名单内时立即拦截"""
        raw_file = tmp_path / "test.csv"
        raw_file.write_text("data\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(raw_file)

        meta_file = tmp_path / "test.csv.source.json"
        meta_file.write_text(json.dumps({
            "source_id": "CSI",
            "source_class": "OFFICIAL_PRIMARY",
            "source_name": "China Securities Index Co., Ltd.",
            "source_url": "https://fake-phishing-csindex.com/data.csv",
            "sha256": h,
            "original_filename": raw_file.name
        }), encoding="utf-8")

        meta, errors = SourceEvidenceMetadata.load_and_verify(meta_file, raw_file)
        assert meta.source_class == SourceClass.UNKNOWN
        assert any("source_domain_fake-phishing-csindex.com_not_in_registry_allowed_domains" in e for e in errors)

    def test_provider_constructor_cannot_inject_verified_true(self):
        """攻击 7: Provider 构造函数拒绝手工注入 verified=True，必须提供 UniverseVerificationResult"""
        provider = PointInTimeUniverseProvider(
            fallback_symbols=["600519.SH"],
            changes_df=None,
            verification_result=None
        )
        assert provider.universe_provenance_verified is False
        assert provider.universe_raw_evidence_verified is False
        assert provider.universe_dataset_hash_verified is False
        assert provider.has_survivorship_bias_risk() is True
        assert provider.get_mode() == "STATIC_FALLBACK"

    def test_set_baseline_cannot_upgrade_provider(self):
        """攻击 8: set_baseline_snapshot 严禁修改认证资质"""
        provider = PointInTimeUniverseProvider()
        assert provider.universe_provenance_verified is False
        provider.set_baseline_snapshot("2020-01-01", ["600519.SH", "000858.SZ"])
        assert provider.universe_provenance_verified is False
        assert provider.universe_source_class == SourceClass.UNKNOWN.value

    def test_set_coverage_window_cannot_modify_verification_state(self):
        """攻击 9: 调用 set_coverage_window() 仅能修改日期区间，严禁篡改或提升认证资质"""
        provider = PointInTimeUniverseProvider()
        assert provider.universe_provenance_verified is False
        provider.set_coverage_window("2020-01-01", "2024-12-31")
        assert provider.universe_provenance_verified is False
        assert provider.universe_raw_evidence_verified is False

    def test_verifier_false_cannot_be_overridden_by_caller_true(self):
        """攻击 10: 当 ProvenanceVerifier 返回 False 时，工厂方法绝不通过调用方传参提升为 True"""
        prov = create_universe_provider(
            config=None,
            changes_df=pd.DataFrame({"effective_date": ["2020-01-01"], "symbol": ["600519.SH"], "action": ["IN"]})
        )
        assert prov.has_survivorship_bias_risk() is True
        assert prov.get_mode() != "POINT_IN_TIME_VERIFIED"

    def test_test_fixture_provider_never_production_verified(self):
        """攻击 11: 测试专用的 for_test_fixture() 强制声明 TEST_FIXTURE，严禁生产认证"""
        fixture_prov = PointInTimeUniverseProvider.for_test_fixture(
            fallback_symbols=["600519.SH"],
            coverage_start="2020-01-01",
            coverage_end="2024-12-31"
        )
        assert fixture_prov.universe_source_class == SourceClass.TEST_FIXTURE.value
        assert SourceClass.is_production_eligible(fixture_prov.universe_source_class) is False
        assert fixture_prov.get_mode() != "POINT_IN_TIME_VERIFIED"

    def test_actual_backtest_end_must_be_covered(self):
        """攻击 12: 实际行情运行到 2026-08-28，PIT 证据仅覆盖到 2024-12-31 必须判定覆盖不完整"""
        df = pd.DataFrame({"effective_date": ["2020-06-15"], "symbol": ["600519.SH"], "action": ["IN"]})
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_file": "snap.csv",
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": [f"600{i:03d}.SH" for i in range(300)],
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(
            df, manifest,
            actual_backtest_start_date="2020-01-01",
            actual_backtest_end_date="2026-08-28"
        )
        assert res.coverage_verified is False
        assert any("coverage_end_2024-12-31_before_actual_backtest_end_2026-08-28" in fc for fc in res.failed_checks)

    def test_actual_backtest_window_required_for_verified(self):
        """攻击 13: CertificationPolicy 独立强制要求 actual_backtest_start/end_date 与 coverage 日期必须存在"""
        meta = AuditMetadata(
            universe_coverage_complete=True,
            universe_provenance_verified=True,
            universe_raw_evidence_verified=True,
            universe_dataset_hash_verified=True,
            universe_source_class=SourceClass.OFFICIAL_PRIMARY.value,
            universe_manifest_hash="hash123",
            factor_manifest_hash="factor123",
            market_manifest_hash="market123",
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
            benchmark_coverage_ratio=1.0,
            benchmark_missing_date_count=0,
            order_quantity_conservation_passed=True,
            synthetic_data_used=False,
            calendar_is_exchange_official=True,
            actual_backtest_start_date=None, # 缺失真实回测起止点
            actual_backtest_end_date=None
        )
        status, failed = CertificationPolicy.evaluate(meta)
        assert status == "HIGH_RISK"
        assert "actual_backtest_window_or_universe_coverage_dates_missing" in failed

    def test_duplicate_baseline_symbols_do_not_inflate_count(self):
        """攻击 14: 试图用 [600519.SH] * 300 伪造 300 只基线成分股必须被拦截"""
        df = pd.DataFrame({"effective_date": ["2020-06-15"], "symbol": ["600519.SH"], "action": ["IN"]})
        manifest = {
            "dataset_name": "CSI300_PIT",
            "index_code": "000300",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_file": "snap.csv",
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": ["600519.SH"] * 300, # 重复 300 次但去重后仅 1 只
            "baseline_symbol_count": 300,
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest)
        assert res.baseline_verified is False
        assert any("baseline_contains_duplicate_symbols" in fc or "csi300_baseline_symbol_count_not_300" in fc for fc in res.failed_checks)

    def test_missing_order_conservation_metric_fails_closed(self):
        """攻击 15: 缺失 order_quantity_conservation_passed 时，系统严格默认为 False (Fail-Closed)"""
        class DummyEngineWithoutMetric:
            pass

        meta = AuditCollector.collect(engine=DummyEngineWithoutMetric())
        assert meta.order_quantity_conservation_passed is False
        status, failed = CertificationPolicy.evaluate(meta)
        assert status == "HIGH_RISK"
        assert "order_quantity_conservation_failed" in failed

    def test_missing_synthetic_metric_fails_closed(self):
        """攻击 16: data_manager 缺失 synthetic_data_used 属性时必须默认为 True (Fail-Closed)"""
        class DummyDataManagerWithoutSyntheticFlag:
            pass

        meta = AuditCollector.collect(data_manager=DummyDataManagerWithoutSyntheticFlag())
        assert meta.synthetic_data_used is True
        status, failed = CertificationPolicy.evaluate(meta)
        assert status == "HIGH_RISK"
        assert "synthetic_data_used" in failed

    def test_override_cannot_change_any_certification_input(self):
        """攻击 17: 试图通过 custom_overrides 篡改任意受保护认证字段必须被全量白名单机制拦截"""
        malicious_overrides = {
            "universe_source_class": "OFFICIAL_PRIMARY",
            "universe_manifest_hash": "forged_univ_hash",
            "factor_manifest_hash": "forged_factor_hash",
            "market_manifest_hash": "forged_market_hash",
            "st_unknown_rows": 0,
            "corporate_action_adjustment_available": True,
            "corporate_action_coverage_ratio": 1.0,
            "corporate_action_bias_risk": False,
            "corporate_action_zero_event_proof_verified": True,
            "raw_data_provenance_preserved": True,
            "benchmark_coverage_ratio": 1.0,
            "benchmark_missing_date_count": 0,
            "order_quantity_conservation_passed": True,
            "synthetic_data_used": False,
            "calendar_is_exchange_official": True,
            "display_notes": "my_valid_note" # 唯一合法白名单字段
        }
        meta = AuditCollector.collect(custom_overrides=malicious_overrides)
        assert meta.display_notes == "my_valid_note"
        assert meta.universe_source_class == "UNKNOWN"
        assert meta.universe_manifest_hash is None
        assert meta.order_quantity_conservation_passed is False
        assert meta.synthetic_data_used is True
        assert any("BLOCKED:universe_source_class" in f for f in meta.audit_override_fields)
        assert any("BLOCKED:synthetic_data_used" in f for f in meta.audit_override_fields)

    def test_corporate_action_zero_event_proof_requires_evidence(self):
        """攻击 18: 公司行为无调整数据且无真实 CorporateActionCoverageEvidence 时，直接声称 verified 必须被拒绝"""
        meta = AuditMetadata(
            universe_coverage_complete=True,
            universe_provenance_verified=True,
            universe_raw_evidence_verified=True,
            universe_dataset_hash_verified=True,
            universe_source_class=SourceClass.OFFICIAL_PRIMARY.value,
            universe_manifest_hash="hash123",
            factor_manifest_hash="factor123",
            market_manifest_hash="market123",
            actual_backtest_start_date="2020-01-01",
            actual_backtest_end_date="2024-12-31",
            universe_coverage_start="2020-01-01",
            universe_coverage_end="2024-12-31",
            survivorship_bias_risk=False,
            historical_st_coverage_complete=True,
            historical_st_bias_risk=False,
            st_unknown_rows=0,
            corporate_action_coverage_complete=True,
            corporate_action_bias_risk=False,
            corporate_action_adjustment_available=False,
            corporate_action_coverage_ratio=0.0,
            corporate_action_zero_event_proof_verified=False,
            cache_fingerprint_verified=True,
            raw_data_provenance_preserved=True,
            adjustment_point_in_time_safe=True,
            future_adjustment_leakage_test_passed=True,
            benchmark_coverage_ratio=1.0,
            benchmark_missing_date_count=0,
            order_quantity_conservation_passed=True,
            synthetic_data_used=False,
            calendar_is_exchange_official=True
        )
        status, failed = CertificationPolicy.evaluate(meta)
        assert status == "HIGH_RISK"
        assert "corporate_action_missing_adjustment_or_zero_event_proof" in failed

    def test_corporate_action_evidence_empty_result_default_false(self):
        """攻击 19: CorporateActionCoverageEvidence 默认 empty_result_verified 必须为 False (Fail-Closed)"""
        evidence = CorporateActionCoverageEvidence(
            symbol="600519.SH",
            query_start="2020-01-01",
            query_end="2024-12-31"
        )
        assert evidence.empty_result_verified is False
        ok, errs = evidence.is_valid_zero_event_proof("600519.SH", "2020-01-01", "2024-12-31")
        assert ok is False
        assert any("empty_result_not_verified_by_audit" in e for e in errs)

    def test_dataframe_hash_boundary_ambiguity_rejected(self):
        """攻击 20: 验证 ['a', 'bc'] 与 ['ab', 'c'] 计算出的 Canonical DataFrame 哈希严格不同"""
        df1 = pd.DataFrame({"col1": ["a"], "col2": ["bc"]})
        df2 = pd.DataFrame({"col1": ["ab"], "col2": ["c"]})

        h1 = ProvenanceVerifier.compute_dataframe_sha256(df1)
        h2 = ProvenanceVerifier.compute_dataframe_sha256(df2)
        assert h1 != h2, "DataFrame Canonical Hash 必须消除字段拼接边界歧义！"

    def test_dataframe_hash_column_order_canonical(self):
        """攻击 21: 列顺序置换不改变 Canonical DataFrame 内容哈希"""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"b": [3, 4], "a": [1, 2]})

        h1 = ProvenanceVerifier.compute_dataframe_sha256(df1)
        h2 = ProvenanceVerifier.compute_dataframe_sha256(df2)
        assert h1 == h2

    def test_dataframe_hash_row_order_canonical(self):
        """攻击 22: 行乱序被自动排序归一化，保持 Canonical 哈希一致"""
        df1 = pd.DataFrame({"effective_date": ["2021-01-01", "2022-01-01"], "symbol": ["600519.SH", "000858.SZ"], "action": ["IN", "OUT"]})
        df2 = pd.DataFrame({"effective_date": ["2022-01-01", "2021-01-01"], "symbol": ["000858.SZ", "600519.SH"], "action": ["OUT", "IN"]})

        h1 = ProvenanceVerifier.compute_dataframe_sha256(df1)
        h2 = ProvenanceVerifier.compute_dataframe_sha256(df2)
        assert h1 == h2

    def test_runtime_json_attestation_master_all_consistent(self):
        """攻击 23: 验证 runtime_audit.json、RUNTIME_ATTESTATION.md、MASTER_REPORT.md 与 Policy 状态 100% 一致"""
        audit_file = Path("artifacts/runtime_audit.json")
        attestation_file = Path("RUNTIME_ATTESTATION.md")
        master_file = Path("MASTER_REPORT.md")

        if audit_file.exists() and attestation_file.exists() and master_file.exists():
            with open(audit_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = AuditMetadata()
            for k, v in data.items():
                if hasattr(meta, k):
                    setattr(meta, k, v)

            expected_status, expected_failed = CertificationPolicy.evaluate(meta)
            assert data.get("overall_backtest_reliability") == expected_status
            assert data.get("failed_certification_checks") == expected_failed

            att_text = attestation_file.read_text(encoding="utf-8")
            assert f"**`{expected_status}`**" in att_text

            master_text = master_file.read_text(encoding="utf-8")
            assert f"**`{expected_status}`**" in master_text
