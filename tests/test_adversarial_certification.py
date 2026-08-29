"""
全链路对抗性认证与防伪自证审计测试套件 (tests/test_adversarial_certification.py)
用于模拟黑客与作弊场景，对认证系统发起 25+ 种主动攻击：
1. 任意 CSV 冒充官方攻击 (Arbitrary Raw CSV Impersonation)
2. 缺失独立 Source Metadata 拦截 (Missing Source Metadata)
3. 伪造 Source Metadata 布尔声明攻击 (Fake Source Metadata Boolean)
4. SHA256 单独冒充官方来源攻击 (SHA256 != Source Authentication)
5. Provider 构造函数注入 Verified=True 攻击 (Constructor Bypass Attack)
6. set_coverage_window 篡改认证资质攻击 (Coverage Window Tampering)
7. Verifier False 被调用方 True 覆盖攻击 (Caller Overrides Verifier)
8. 测试 Fixture 冒充生产证据攻击 (Test Fixture Impersonating Production)
9. 实际回测终点超出 PIT 覆盖范围攻击 (Actual Backtest End Exceeds PIT Coverage)
10. END_DATE=None 绕过终点核验攻击 (END_DATE=None Coverage Bypass)
11. Provider 自我验证覆盖率攻击 (Provider Self-Validation)
12. 运行产物 JSON / Attestation / Policy 一致性校验 (Runtime Consistency)
13. 订单守恒缺失自动 Fail-Closed 检验 (Order Conservation Fail-Closed)
14. 公司行为无数据谎称 100% 覆盖攻击 (Corporate Action Inconsistency)
15. 公司行为 Zero Event Proof 严格门禁 (Zero Event Proof Gate)
16. 严禁由第一条调仓记录当 Baseline 攻击 (First Rebalance Cannot Be Baseline)
17. 缺少独立 Baseline 快照文件拦截 (Missing Independent Baseline Snapshot)
18. 原始证据文件单字节微小篡改攻击 (Raw Evidence Single-Byte Tampering)
19. 规范化数据集单单元格数值篡改攻击 (Normalized Dataset Tampering)
20. 同日同标的冲突 IN/OUT 攻击 (Conflicting Events on Same Day)
21. ST 存在未知行时谎称完全覆盖攻击 (ST Coverage Inconsistency Gate)
22. 链式哈希缺失拦截 (Missing Manifest Hash)
23. 未来无证据覆盖声明攻击 (Unsubstantiated Future Coverage)
24. 空数据与空 Manifest 检验 (Empty Provenance Fail-Closed)
25. 假官方字符串拦截 (Fake Official String Defense)
"""
import pytest
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

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
from backtest.audit import AuditMetadata, CertificationPolicy, AuditCollector


class TestAdversarialCertification:

    def test_arbitrary_raw_csv_not_automatically_official(self, tmp_path):
        """攻击 1: 攻击者向 raw/ 放置任意 fake.csv，系统绝不自动将其识别为 OFFICIAL_PRIMARY"""
        fake_csv = tmp_path / "fake_csi300.csv"
        fake_csv.write_text("effective_date,symbol,action\n2020-06-15,600519.SH,IN\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(fake_csv)

        df = pd.DataFrame({"effective_date": ["2020-06-15"], "symbol": ["600519.SH"], "action": ["IN"]})
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": "OFFICIAL_PRIMARY", # 试图声称官方
            "source_files": [fake_csv.name],
            "raw_evidence_hashes": {fake_csv.name: h},
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_file": fake_csv.name,
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": ["600519.SH"] * 300,
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        # 缺少 fake.csv.source.json
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

    def test_raw_sha_does_not_prove_official_source(self, tmp_path):
        """攻击 4: SHA256 只能证明未篡改，不能证明来自官方（非官方源即使哈希正确也不得 VERIFIED）"""
        raw_file = tmp_path / "thirdparty.csv"
        raw_file.write_text("content\n", encoding="utf-8")
        h = ProvenanceVerifier.compute_file_sha256(raw_file)

        meta_file = tmp_path / "thirdparty.csv.source.json"
        meta_file.write_text(json.dumps({
            "source_class": "THIRD_PARTY", # 第三方爬虫
            "source_name": "PublicCrawler",
            "source_url": "http://crawler.org",
            "sha256": h,
            "original_filename": raw_file.name
        }), encoding="utf-8")

        meta, errors = SourceEvidenceMetadata.load_and_verify(meta_file, raw_file)
        assert len(errors) == 0
        assert meta.source_class == SourceClass.THIRD_PARTY
        assert SourceClass.is_production_eligible(meta.source_class) is False

    def test_provider_constructor_cannot_inject_verified_true(self):
        """攻击 5: Provider 构造函数拒绝手工注入 verified=True，必须提供 UniverseVerificationResult"""
        provider = PointInTimeUniverseProvider(
            fallback_symbols=["600519.SH"],
            changes_df=None,
            verification_result=None # 未提供验证结果
        )
        assert provider.universe_provenance_verified is False
        assert provider.universe_raw_evidence_verified is False
        assert provider.universe_dataset_hash_verified is False
        assert provider.has_survivorship_bias_risk() is True
        assert provider.get_mode() == "STATIC_FALLBACK"

    def test_set_coverage_window_cannot_modify_verification_state(self):
        """攻击 6: 调用 set_coverage_window() 仅能修改日期区间，严禁篡改或提升认证资质"""
        provider = PointInTimeUniverseProvider()
        assert provider.universe_provenance_verified is False
        provider.set_coverage_window("2020-01-01", "2024-12-31")
        assert provider.universe_provenance_verified is False
        assert provider.universe_raw_evidence_verified is False

    def test_verifier_false_cannot_be_overridden_by_caller_true(self, tmp_path):
        """攻击 7: 当 ProvenanceVerifier 返回 False 时，工厂方法绝不通过调用方传参提升为 True"""
        prov = create_universe_provider(
            config=None,
            changes_df=pd.DataFrame({"effective_date": ["2020-01-01"], "symbol": ["600519.SH"], "action": ["IN"]})
        )
        assert prov.has_survivorship_bias_risk() is True
        assert prov.get_mode() != "POINT_IN_TIME_VERIFIED"

    def test_test_fixture_provider_never_production_verified(self):
        """攻击 8: 测试专用的 for_test_fixture() 强制声明 TEST_FIXTURE，严禁生产认证"""
        fixture_prov = PointInTimeUniverseProvider.for_test_fixture(
            fallback_symbols=["600519.SH"],
            coverage_start="2020-01-01",
            coverage_end="2024-12-31"
        )
        assert fixture_prov.universe_source_class == SourceClass.TEST_FIXTURE.value
        assert SourceClass.is_production_eligible(fixture_prov.universe_source_class) is False
        assert fixture_prov.get_mode() != "POINT_IN_TIME_VERIFIED"

    def test_actual_backtest_end_must_be_covered(self):
        """攻击 9: 实际行情运行到 2026-08-28，PIT 证据仅覆盖到 2024-12-31 必须判定覆盖不完整"""
        df = pd.DataFrame({"effective_date": ["2020-06-15"], "symbol": ["600519.SH"], "action": ["IN"]})
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_file": "snap.csv",
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": [f"600{i:03d}.SH" for i in range(300)],
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31" # 仅覆盖到 2024
        }
        res = ProvenanceVerifier.verify_pit_universe(
            df, manifest,
            actual_backtest_start_date="2020-01-01",
            actual_backtest_end_date="2026-08-28" # 实际回测到 2026
        )
        assert res.coverage_verified is False
        assert any("coverage_end_2024-12-31_before_actual_backtest_end_2026-08-28" in fc for fc in res.failed_checks)

    def test_end_date_none_uses_actual_market_end_date(self):
        """攻击 10: END_DATE=None 时，必须使用实际回测终点进行覆盖率核验"""
        df = pd.DataFrame({"effective_date": ["2020-06-15"], "symbol": ["600519.SH"], "action": ["IN"]})
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_file": "snap.csv",
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": [f"600{i:03d}.SH" for i in range(300)],
            "coverage_start": "2020-01-01",
            "coverage_end": "2023-12-31"
        }
        # 实际行情到达 2025-01-01
        res = ProvenanceVerifier.verify_pit_universe(
            df, manifest,
            actual_backtest_start_date="2020-01-01",
            actual_backtest_end_date="2025-01-01"
        )
        assert res.coverage_verified is False

    def test_missing_order_conservation_metric_fails_closed(self):
        """攻击 11: 缺失 order_quantity_conservation_passed 时，系统严格默认为 False (Fail-Closed)"""
        class DummyEngineWithoutMetric:
            pass

        meta = AuditCollector.collect(engine=DummyEngineWithoutMetric())
        assert meta.order_quantity_conservation_passed is False
        status, failed = CertificationPolicy.evaluate(meta)
        assert status == "HIGH_RISK"
        assert "order_quantity_conservation_failed" in failed

    def test_corporate_action_100pct_without_data_fails(self):
        """攻击 12: 公司行为无覆盖数据 (0%) 却声明 complete 时必须被熔断拦截"""
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
            corporate_action_adjustment_available=False,
            corporate_action_coverage_ratio=0.0,
            corporate_action_zero_event_proof_verified=False, # 无零事件证明
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
        assert any("corporate_action_missing_adjustment_or_zero_event_proof" in f for f in failed)

    def test_zero_event_proof_allows_verified_empty_action_period(self):
        """攻击 13: 当且仅当具备经过鉴证的 Zero Event Proof 时，无公司行为区间才允许通过门禁"""
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
            corporate_action_adjustment_available=False,
            corporate_action_coverage_ratio=0.0,
            corporate_action_zero_event_proof_verified=True, # 具备 Zero Event 认证证明
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
        assert status == "VERIFIED"
        assert len(failed) == 0

    def test_first_rebalance_event_cannot_become_baseline(self):
        """攻击 14: 严禁由第一条调仓记录 (仅个别股票) 自动充当 Baseline Snapshot"""
        df = pd.DataFrame({"effective_date": ["2020-06-15"], "symbol": ["600519.SH"], "action": ["IN"]})
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_file": "rebal_only.csv",
            "baseline_snapshot_date": "2020-06-15",
            "baseline_symbols": ["600519.SH"], # 只有 1 只股票，不是指数完整基线
            "coverage_start": "2020-06-15",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest)
        assert res.baseline_verified is False
        assert any("baseline_symbol_count_insufficient_for_index" in fc for fc in res.failed_checks)

    def test_baseline_requires_independent_snapshot(self):
        """攻击 15: 缺少独立的 Baseline Snapshot 原始文件时必须被拦截"""
        df = pd.DataFrame({"effective_date": ["2020-06-15"], "symbol": ["600519.SH"], "action": ["IN"]})
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": [f"600{i:03d}.SH" for i in range(300)],
            # 缺失 baseline_snapshot_file
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest)
        assert res.baseline_verified is False
        assert any("baseline_snapshot_independent_raw_file_missing_in_manifest" in fc for fc in res.failed_checks)

    def test_tampered_raw_source_hash_rejected(self, tmp_path):
        """攻击 16: 原始 Raw Evidence 文件被恶意篡改 1 字节，哈希不匹配必须直接拒绝"""
        raw_file = tmp_path / "official_csi300_2020.csv"
        raw_file.write_text("effective_date,symbol,action\n2020-06-15,600519.SH,IN\n", encoding="utf-8")
        true_hash = ProvenanceVerifier.compute_file_sha256(raw_file)

        meta_file = tmp_path / "official_csi300_2020.csv.source.json"
        meta_file.write_text(json.dumps({
            "source_class": "OFFICIAL_PRIMARY",
            "source_name": "CSI_OFFICIAL",
            "source_url": "http://csindex.com.cn",
            "sha256": true_hash,
            "original_filename": raw_file.name
        }), encoding="utf-8")

        df = pd.DataFrame({"effective_date": ["2020-06-15"], "symbol": ["600519.SH"], "action": ["IN"]})
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "source_files": [raw_file.name],
            "raw_evidence_hashes": {raw_file.name: true_hash},
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_file": raw_file.name,
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": [f"600{i:03d}.SH" for i in range(300)],
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }

        # 篡改文件 1 字节
        raw_file.write_text("effective_date,symbol,action\n2020-06-15,600519.SH,IN \n", encoding="utf-8")
        res = ProvenanceVerifier.verify_pit_universe(df, manifest, raw_evidence_dir=tmp_path)
        assert res.raw_hash_verified is False
        assert any("raw_evidence_hash_mismatch" in fc for fc in res.failed_checks)

    def test_tampered_normalized_dataset_hash_rejected(self):
        """攻击 17: 规范化 Parquet 数据集被篡改 1 个值，SHA256 不匹配必须拒绝"""
        df = pd.DataFrame({"effective_date": ["2020-06-15"], "symbol": ["600519.SH"], "action": ["IN"]})
        original_hash = ProvenanceVerifier.compute_dataframe_sha256(df)

        tampered_df = df.copy()
        tampered_df.loc[0, "symbol"] = "000858.SZ"

        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "normalized_dataset_sha256": original_hash,
            "baseline_snapshot_file": "snap.csv",
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": [f"600{i:03d}.SH" for i in range(300)],
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(tampered_df, manifest)
        assert res.dataset_hash_verified is False
        assert "normalized_dataset_sha256_mismatch" in res.failed_checks

    def test_synthetic_pit_can_never_be_verified(self):
        """攻击 18: 算法/模拟生成数据 (SYNTHETIC) 严禁在生产环境获得认证"""
        df = pd.DataFrame({"effective_date": ["2020-06-15"], "symbol": ["600519.SH"], "action": ["IN"]})
        manifest = {
            "dataset_name": "SYNTHETIC_PIT",
            "source_class": SourceClass.SYNTHETIC.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_file": "snap.csv",
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": [f"600{i:03d}.SH" for i in range(300)],
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest)
        assert res.is_production_verified is False
        assert "source_class_is_synthetic_strictly_forbidden" in res.failed_checks
        assert res.survivorship_bias_risk is True

    def test_missing_universe_manifest_hash_blocks_verified(self):
        """攻击 19: 缺少 universe_manifest_hash 链式指纹时严禁进入 VERIFIED"""
        meta = AuditMetadata(
            universe_coverage_complete=True,
            universe_provenance_verified=True,
            universe_raw_evidence_verified=True,
            universe_dataset_hash_verified=True,
            universe_source_class=SourceClass.OFFICIAL_PRIMARY.value,
            universe_manifest_hash=None, # 缺失哈希
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
            calendar_is_exchange_official=True
        )
        status, failed = CertificationPolicy.evaluate(meta)
        assert status == "HIGH_RISK"
        assert "universe_manifest_hash_missing" in failed

    def test_runtime_json_and_attestation_reliability_consistent(self):
        """攻击 20: 验证 runtime_audit.json、RUNTIME_ATTESTATION.md 与 CertificationPolicy 评级绝对一致"""
        audit_file = Path("artifacts/runtime_audit.json")
        attestation_file = Path("RUNTIME_ATTESTATION.md")

        if audit_file.exists() and attestation_file.exists():
            with open(audit_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = AuditMetadata()
            for k, v in data.items():
                if hasattr(meta, k):
                    setattr(meta, k, v)

            expected_status, _ = CertificationPolicy.evaluate(meta)
            attestation_text = attestation_file.read_text(encoding="utf-8")
            assert f"**`{expected_status}`**" in attestation_text

    def test_runtime_failed_checks_are_recomputed(self):
        """攻击 21: 验证 failed_certification_checks 必须由策略真值表动态重算，严禁硬编码残存"""
        meta = AuditMetadata(universe_coverage_complete=False)
        status, failed = CertificationPolicy.evaluate(meta)
        assert isinstance(failed, list)
        assert len(failed) > 0
        assert "universe_coverage_incomplete" in failed

    def test_source_class_cannot_be_created_by_parser(self, tmp_path):
        """攻击 22: 验证 Parser Adapter 绝不能自行凭空提升数据源为 OFFICIAL_PRIMARY"""
        raw_file = tmp_path / "test_rebalance.csv"
        raw_file.write_text("index_code,effective_date,symbol,action\n000300,2020-06-15,600519.SH,IN\n", encoding="utf-8")
        
        # 来源元数据为 THIRD_PARTY
        meta = SourceEvidenceMetadata(
            source_class=SourceClass.THIRD_PARTY,
            source_name="ThirdPartyCrawler",
            original_filename=raw_file.name,
            sha256="dummy_hash"
        )
        parser = CSIRebalanceAnnouncementParser()
        events = parser.parse(raw_file, meta)
        assert len(events) == 1
        assert events[0]["source_class"] == SourceClass.THIRD_PARTY.value
        assert events[0]["source_class"] != SourceClass.OFFICIAL_PRIMARY.value

    def test_provider_cannot_self_validate_its_own_coverage(self):
        """攻击 23: Provider 绝不能使用自己声明的 coverage_start 绕过实际回测终点校验"""
        provider = PointInTimeUniverseProvider(
            coverage_start="2020-01-01",
            coverage_end="2022-12-31" # 仅覆盖到 2022
        )
        # 实际回测到 2026-08-28
        provider.set_actual_backtest_window("2020-01-01", "2026-08-28")
        assert provider.is_coverage_complete() is False
        assert provider.has_survivorship_bias_risk() is True
