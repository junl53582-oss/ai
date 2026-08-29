"""
全链路对抗性认证与防伪自证审计测试套件 (tests/test_adversarial_certification.py)
用于模拟黑客与作弊场景，对认证系统发起 17 种主动攻击：
1. 伪造官方声明与假字符串攻击 (Fake Official Strings)
2. Manifest 布尔值伪造与自我认证攻击 (Manifest Boolean Forgery)
3. 原始证据文件单字节微小篡改攻击 (Raw Evidence Single-Byte Tampering)
4. 规范化数据集单单元格数值篡改攻击 (Normalized Dataset Tampering)
5. 缺失 Raw 证据文件攻击 (Missing Raw Evidence)
6. 测试 Fixture 冒充生产证据攻击 (Test Fixture Impersonating Production)
7. 算法/模拟生成数据冒充官方攻击 (Synthetic Data Attack)
8. 未来无证据覆盖声明攻击 (Unsubstantiated Future Coverage)
9. 非法股票代码与格式攻击 (Invalid Symbol Format)
10. 同日同标的冲突 IN/OUT 攻击 (Conflicting Events on Same Day)
11. 基线时点晚于回测起始点攻击 (Baseline Timeliness Violation)
12. ST 存在未知行时谎称完全覆盖攻击 (ST Coverage Inconsistency Gate)
13. 公司行为缺失数据时谎称完全覆盖攻击 (Corporate Action Inconsistency Gate)
14. 链式哈希缺失拦截 (Missing Manifest Hash)
"""
import pytest
import tempfile
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

from data.provenance import SourceClass, ProvenanceVerifier, UniverseVerificationResult, DataProvenanceError
from data.universe_provider import PointInTimeUniverseProvider, StaticUniverseProvider
from backtest.audit import AuditMetadata, CertificationPolicy, AuditCollector


class TestAdversarialCertification:

    def test_synthetic_pit_can_never_be_verified(self):
        """攻击 1: 算法/模拟生成数据 (SYNTHETIC) 严禁在生产环境获得认证"""
        df = pd.DataFrame({
            "effective_date": ["2020-06-15"],
            "symbol": ["600519.SH"],
            "action": ["IN"]
        })
        manifest = {
            "dataset_name": "SYNTHETIC_PIT",
            "source_class": SourceClass.SYNTHETIC.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": ["600519.SH"],
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest)
        assert res.is_production_verified is False
        assert "source_class_is_synthetic_strictly_forbidden" in res.failed_checks
        assert res.survivorship_bias_risk is True

    def test_test_fixture_cannot_become_production_verified(self):
        """攻击 2: 单测 Fixture (TEST_FIXTURE) 仅能证明代码能力，绝不能升级为生产认证"""
        df = pd.DataFrame({
            "effective_date": ["2020-06-15"],
            "symbol": ["600519.SH"],
            "action": ["IN"]
        })
        manifest = {
            "dataset_name": "TEST_FIXTURE_PIT",
            "source_class": SourceClass.TEST_FIXTURE.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": ["600519.SH"],
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest)
        assert res.is_production_verified is False
        assert "source_class_is_test_fixture_not_production_eligible" in res.failed_checks

    def test_manifest_claim_true_cannot_self_certify(self):
        """攻击 3: Manifest 自身声明 verified=True / provenance_verified=True 绝不能绕过检查"""
        df = pd.DataFrame({
            "effective_date": ["2020-06-15"],
            "symbol": ["600519.SH"],
            "action": ["IN"]
        })
        forged_manifest = {
            "dataset_name": "FORGED_OFFICIAL",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "provenance_verified": True,
            "constituent_event_source_verified": True,
            "survivorship_bias_risk": False,
            "universe_coverage_complete": True,
            "verification_method": "EXCHANGE_OFFICIAL_HISTORICAL_REBALANCE",
            # 故意不提供真实的 raw evidence 文件哈希
            "source_files": ["non_existent_raw.csv"],
            "raw_evidence_hashes": {"non_existent_raw.csv": "abc123fakehash"}
        }
        res = ProvenanceVerifier.verify_pit_universe(df, forged_manifest)
        # 必须失败，绝不轻信 Manifest 内的布尔声明
        assert res.is_production_verified is False
        assert res.survivorship_bias_risk is True
        assert any("raw_evidence" in fc for fc in res.failed_checks)

    def test_tampered_raw_source_hash_rejected(self, tmp_path):
        """攻击 4: 原始 Raw Evidence 文件被恶意篡改 1 字节，哈希不匹配必须直接拒绝"""
        raw_file = tmp_path / "official_csi300_2020.csv"
        raw_file.write_text("effective_date,symbol,action\n2020-06-15,600519.SH,IN\n", encoding="utf-8")
        true_hash = ProvenanceVerifier.compute_file_sha256(raw_file)

        df = pd.DataFrame({
            "effective_date": ["2020-06-15"],
            "symbol": ["600519.SH"],
            "action": ["IN"]
        })
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "source_files": [raw_file.name],
            "raw_evidence_hashes": {raw_file.name: true_hash},
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": ["600519.SH"],
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }

        # 篡改文件 1 字节
        raw_file.write_text("effective_date,symbol,action\n2020-06-15,600519.SH,IN \n", encoding="utf-8")
        res = ProvenanceVerifier.verify_pit_universe(df, manifest, raw_evidence_dir=tmp_path)
        assert res.raw_hash_verified is False
        assert any("raw_evidence_hash_mismatch" in fc for fc in res.failed_checks)

    def test_tampered_normalized_dataset_hash_rejected(self):
        """攻击 5: 规范化 Parquet 数据集被篡改 1 个值，SHA256 不匹配必须拒绝"""
        df = pd.DataFrame({
            "effective_date": ["2020-06-15"],
            "symbol": ["600519.SH"],
            "action": ["IN"]
        })
        original_hash = ProvenanceVerifier.compute_dataframe_sha256(df)

        tampered_df = df.copy()
        tampered_df.loc[0, "symbol"] = "000858.SZ" # 篡改标的代码

        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "normalized_dataset_sha256": original_hash,
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": ["600519.SH"],
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(tampered_df, manifest)
        assert res.dataset_hash_verified is False
        assert "normalized_dataset_sha256_mismatch" in res.failed_checks

    def test_invalid_symbol_rejected(self):
        """攻击 6: 非法股票代码 (如美股 AAPL 或伪造格式) 必须被识别并拦截"""
        df = pd.DataFrame({
            "effective_date": ["2020-06-15"],
            "symbol": ["AAPL.US"],
            "action": ["IN"]
        })
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": ["AAPL.US"],
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest)
        assert res.event_integrity_verified is False
        assert any("invalid_a_share_symbol_format" in fc for fc in res.failed_checks)

    def test_conflicting_in_out_same_day_rejected(self):
        """攻击 7: 同一天对同一股票同时出现 IN 和 OUT 事件必须被拦截"""
        df = pd.DataFrame({
            "effective_date": ["2020-06-15", "2020-06-15"],
            "symbol": ["600519.SH", "600519.SH"],
            "action": ["IN", "OUT"]
        })
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": ["600519.SH"],
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest)
        assert res.event_integrity_verified is False
        assert any("conflicting_in_out_events_on_same_day" in fc for fc in res.failed_checks)

    def test_baseline_after_backtest_start_rejected(self):
        """攻击 8: 基线时点 (2021-01-01) 晚于回测开始时点 (2020-01-01) 必须拒绝"""
        df = pd.DataFrame({
            "effective_date": ["2021-06-15"],
            "symbol": ["600519.SH"],
            "action": ["IN"]
        })
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_date": "2021-01-01",
            "baseline_symbols": ["600519.SH"],
            "coverage_start": "2021-01-01",
            "coverage_end": "2024-12-31"
        }
        res = ProvenanceVerifier.verify_pit_universe(
            df, manifest,
            backtest_start_date="2020-01-01",
            backtest_end_date="2024-12-31"
        )
        assert res.baseline_verified is False
        assert any("baseline_date" in fc and "after_backtest_start" in fc for fc in res.failed_checks)

    def test_unknown_st_rows_consistency_gate(self):
        """攻击 9: 存在 5000 行未知 ST 状态时，声明 complete coverage 必须被 CertificationPolicy 熔断拦截"""
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
            # ST 存在矛盾
            historical_st_coverage_complete=True,
            historical_st_bias_risk=False,
            st_unknown_rows=5000,
            # 其余全满足
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
        assert status != "VERIFIED"
        assert any("st_unknown_rows" in f for f in failed)

    def test_corporate_action_missing_data_consistency_gate(self):
        """攻击 10: 公司行为无覆盖数据 (0%) 却声明 complete 时必须被熔断拦截"""
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
            # 公司行为矛盾
            corporate_action_coverage_complete=True,
            corporate_action_bias_risk=False,
            corporate_action_adjustment_available=False,
            corporate_action_coverage_ratio=0.0,
            # 其余全满足
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
        assert status != "VERIFIED"
        assert any("corporate_action_missing_data" in f for f in failed)

    def test_missing_universe_manifest_hash_blocks_verified(self):
        """攻击 11: 缺少 universe_manifest_hash 链式指纹时严禁进入 VERIFIED"""
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

    def test_future_coverage_cannot_be_claimed_without_evidence(self):
        """攻击 12: 声明覆盖至 2099 年未来日期却无真实证据时必须被拦截"""
        df = pd.DataFrame({
            "effective_date": ["2020-06-15"],
            "symbol": ["600519.SH"],
            "action": ["IN"]
        })
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": SourceClass.OFFICIAL_PRIMARY.value,
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": ["600519.SH"],
            "coverage_start": "2020-01-01",
            "coverage_end": "2099-12-31" # 夸大未来覆盖
        }
        res = ProvenanceVerifier.verify_pit_universe(df, manifest)
        assert res.coverage_verified is False
        assert any("unsubstantiated_future_coverage_claimed" in fc for fc in res.failed_checks)

    def test_empty_provenance_never_verified(self):
        """攻击 13: 空数据与空 Manifest 必须 Fail-Closed 绝不认证"""
        res = ProvenanceVerifier.verify_pit_universe(None, None)
        assert res.is_production_verified is False
        assert res.survivorship_bias_risk is True
        assert res.mode == "STATIC_FALLBACK"

    def test_fake_official_string_does_not_bypass_verifier(self):
        """攻击 14: 伪造官方字符串 source_name='OFFICIAL' 绝不能绕过真实哈希核验"""
        df = pd.DataFrame({
            "effective_date": ["2020-06-15"],
            "symbol": ["600519.SH"],
            "action": ["IN"]
        })
        manifest = {
            "dataset_name": "CSI300_PIT",
            "source_class": "OFFICIAL_PRIMARY",
            "source_name": "SSE_OFFICIAL_WEBSITE_EXTRACT",
            "source_files": ["official_rebalance.csv"],
            "raw_evidence_hashes": {"official_rebalance.csv": "abc123hash"},
            "normalized_dataset_sha256": ProvenanceVerifier.compute_dataframe_sha256(df),
            "baseline_snapshot_date": "2020-01-01",
            "baseline_symbols": ["600519.SH"],
            "coverage_start": "2020-01-01",
            "coverage_end": "2024-12-31"
        }
        # 磁盘上根本没有 official_rebalance.csv
        res = ProvenanceVerifier.verify_pit_universe(df, manifest, raw_evidence_dir=Path("/non/existent/dir"))
        assert res.raw_hash_verified is False
        assert res.is_production_verified is False
