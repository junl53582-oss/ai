"""
Stage R1 独立审计回归测试：零伪造与硬编码彻底清除验真
(tests/test_r1_zero_fabrication_completion.py)

验证点：
1. dashboard/app.py 中彻底根除残余硬编码指标 (56.75, 56.30, 56.69, 76.80, 39.4, 50.28, 54.35, 55.80, 199.45, 26.12)。
2. data_as_of 严格动态读取物理科研产物，无凭证时 Fail-Closed 显示 "暂无可验证数据"。
3. VerifiedResearchMetricsLoader 哈希与 Manifest 防伪校验生效，凭证篡改或缺失时 Fail-Closed。
4. scheduler/notifier.py 严格校验 pred_score，禁止 0.0% 默认兜底，缺省标识为 "模型排序分数"。
"""
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import settings
from models.verified_metrics import VerifiedResearchMetricsLoader, compute_sha256, ProbabilityCalibrationEvidence
from scheduler.notifier import MessageNotifier


def test_dashboard_code_has_no_hardcoded_fake_metrics():
    """验证 dashboard/app.py 源码中不存在任何残余硬编码指标"""
    app_file = settings.BASE_DIR / "dashboard" / "app.py"
    assert app_file.exists(), "dashboard/app.py 文件不存在"
    code = app_file.read_text(encoding="utf-8")

    forbidden_patterns = [
        "56.75",
        "56.30",
        "56.69",
        "76.80",
        "39.4",
        "50.28",
        "54.35",
        "55.80",
        "199.45",
        "26.12",
        "2026-08-24 (已收盘)",
        "cur_perf.get('net_win_rate', 39.4)",
    ]
    for pat in forbidden_patterns:
        assert pat not in code, f"dashboard/app.py 中发现残留伪造/硬编码模式: '{pat}'"


def test_verified_metrics_loader_integrity_and_fail_closed(tmp_path):
    """验证 VerifiedResearchMetricsLoader 能够正确核验哈希，篡改或缺失时 Fail-Closed"""
    # 真实环境加载应通过
    loader = VerifiedResearchMetricsLoader()
    data_as_of = loader.get_data_as_of()
    assert data_as_of != "暂无可验证数据", "真实环境下应能读取有效基准日"

    table = loader.get_multi_generation_metrics_table()
    assert table is not None
    assert not table.empty
    assert "核心量化指标" in table.columns
    assert "第四代 (全景旗舰)" in table.columns

    # 模拟凭证篡改环境 (Fail-Closed)
    fake_reports = tmp_path / "reports"
    fake_reports.mkdir(parents=True)
    fake_perf = fake_reports / "performance_all_generations.json"
    fake_manifest = fake_reports / "performance_all_generations.manifest.json"

    fake_perf.write_text(json.dumps({"gen1_baseline": {}}), encoding="utf-8")
    fake_manifest.write_text(json.dumps({
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "data_as_of": "2026-09-01"
    }), encoding="utf-8")

    tampered_loader = VerifiedResearchMetricsLoader(base_dir=tmp_path)
    # 哈希不匹配时必须 Fail-Closed 返回 None
    assert tampered_loader.load_multi_generation_performance() is None
    assert tampered_loader.get_multi_generation_metrics_table() is None

    # 缺少 manifest 时返回 None
    fake_manifest.unlink()
    assert tampered_loader.load_multi_generation_performance() is None


def test_notifier_score_strict_validation_and_fail_closed(tmp_path, monkeypatch):
    """验证通知中心对 pred_score 的严格校验与 Fail-Closed"""
    from data.crypto_anchor import TRUSTED_KEY_REGISTRY, generate_keypair, sign_with_environment_key
    from models.verified_metrics import compute_sha256

    valid_df = pd.DataFrame([
        {"symbol": "300308.SZ", "name": "中际旭创", "industry": "通信设备", "pred_score": 0.8845, "target_weight": 0.15, "close": 870.22},
        {"symbol": "688256.SH", "name": "寒武纪", "industry": "半导体", "pred_score": 0.7612, "target_weight": 0.10, "close": 969.05}
    ])

    # 1. 默认状态：必须标识为 "模型排序分数"，禁止伪造为 "预测上涨概率"
    report_default = MessageNotifier.format_daily_report_markdown(
        signal_date="2026-08-24",
        execution_date="2026-08-25",
        top_df=valid_df
    )
    assert "模型排序分数" in report_default
    assert "预测上涨概率" not in report_default
    assert "0.8845" in report_default
    assert "**0.0%**" not in report_default

    # 2. 构造合法物理校准凭证与校准概率状态：标识为 "预测上涨概率"
    sk_bytes, pk_bytes = generate_keypair()
    key_id = "R1_TEST_CALIB_KEY_2026"
    test_entry = {
        "algorithm": "ED25519",
        "key_id": key_id,
        "public_key_hex": pk_bytes.hex(),
        "allowed_purposes": ["PROBABILITY_CALIBRATION_AUTHORIZATION", "MODEL_PROMOTION"],
        "issuer_type": "PROJECT_TEST",
        "institution": "TEST_AUTHORITY",
        "status": "ACTIVE",
        "not_before": "2025-01-01T00:00:00Z",
        "not_after": "2030-01-01T00:00:00Z",
        "is_production": False,
    }
    monkeypatch.setitem(TRUSTED_KEY_REGISTRY, key_id, test_entry)

    cal_file = tmp_path / "calibrator.bin"
    cal_file.write_bytes(b"calibrator_bin_data_test_123")
    cal_sha = compute_sha256(cal_file)

    rel_file = tmp_path / "reliability_curve.json"
    rel_file.write_text(json.dumps({"curve": [0.1, 0.2, 0.5]}), encoding="utf-8")
    rel_sha = compute_sha256(rel_file)

    schema_hash = "d" * 64
    manifest_file = tmp_path / "calibration_manifest.json"
    manifest_file.write_text(json.dumps({
        "model_id": "gen4_flagship",
        "feature_schema_hash": schema_hash,
        "dataset_sha256": "9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42"
    }), encoding="utf-8")
    manifest_sha = compute_sha256(manifest_file)

    signed_at = "2026-08-24T12:00:00Z"
    payload = {
        "method": "isotonic",
        "brier_score": 0.125,
        "sample_count": 2000,
        "calibrator_artifact_sha256": cal_sha,
        "reliability_curve_sha256": rel_sha,
        "calibration_dataset_sha256": manifest_sha,
        "model_id": "gen4_flagship",
        "feature_schema_hash": schema_hash,
        "signed_at": signed_at,
        "status": "VERIFIED",
    }
    msg_bytes = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    sig, _ = sign_with_environment_key(
        message=msg_bytes,
        key_id=key_id,
        required_purpose="PROBABILITY_CALIBRATION_AUTHORIZATION",
        domain_separator="QUANT_PROBABILITY_CALIBRATION_V1",
        explicit_private_key_hex=sk_bytes.hex()
    )
    valid_cal_ev = ProbabilityCalibrationEvidence(
        method="isotonic",
        brier_score=0.125,
        sample_count=2000,
        calibrator_artifact_path=str(cal_file),
        calibrator_artifact_sha256=cal_sha,
        reliability_curve_artifact_path=str(rel_file),
        reliability_curve_sha256=rel_sha,
        calibration_dataset_manifest_path=str(manifest_file),
        calibration_dataset_sha256=manifest_sha,
        model_id="gen4_flagship",
        feature_schema_hash=schema_hash,
        signed_at=signed_at,
        signer_key_id=key_id,
        signature=sig,
        status="VERIFIED"
    )
    assert valid_cal_ev.is_valid()
    report_prob = MessageNotifier.format_daily_report_markdown(
        signal_date="2026-08-24",
        execution_date="2026-08-25",
        top_df=valid_df,
        calibration_evidence=valid_cal_ev
    )
    assert "预测上涨概率" in report_prob
    assert "88.45%" in report_prob

    # 3. 声明已校准但无证据或证据无效时强制 Fail-Closed
    with pytest.raises(ValueError, match="Cannot claim calibrated probability without verified"):
        MessageNotifier.format_daily_report_markdown(
            signal_date="2026-08-24",
            execution_date="2026-08-25",
            top_df=valid_df,
            is_calibrated_probability=True
        )

    # 4. 缺失 pred_score 时强制 Fail-Closed 抛出 ValueError
    missing_df = pd.DataFrame([
        {"symbol": "300308.SZ", "name": "中际旭创", "industry": "通信设备", "target_weight": 0.15, "close": 870.22}
    ])
    with pytest.raises(ValueError, match="Missing pred_score"):
        MessageNotifier.format_daily_report_markdown(
            signal_date="2026-08-24",
            execution_date="2026-08-25",
            top_df=missing_df
        )

    # 5. NaN / Inf 异常值时强制 Fail-Closed
    nan_df = pd.DataFrame([
        {"symbol": "300308.SZ", "name": "中际旭创", "industry": "通信设备", "pred_score": np.nan, "target_weight": 0.15, "close": 870.22}
    ])
    with pytest.raises(ValueError, match="Invalid pred_score"):
        MessageNotifier.format_daily_report_markdown(
            signal_date="2026-08-24",
            execution_date="2026-08-25",
            top_df=nan_df
        )

    inf_df = pd.DataFrame([
        {"symbol": "300308.SZ", "name": "中际旭创", "industry": "通信设备", "pred_score": np.inf, "target_weight": 0.15, "close": 870.22}
    ])
    with pytest.raises(ValueError, match="Invalid pred_score"):
        MessageNotifier.format_daily_report_markdown(
            signal_date="2026-08-24",
            execution_date="2026-08-25",
            top_df=inf_df
        )

    # 6. 校准概率越界 (如 >1.0 或 <0.0) 强制 Fail-Closed
    oob_df = pd.DataFrame([
        {"symbol": "300308.SZ", "name": "中际旭创", "industry": "通信设备", "pred_score": 1.5, "target_weight": 0.15, "close": 870.22}
    ])
    with pytest.raises(ValueError, match="Out-of-bounds calibrated probability"):
        MessageNotifier.format_daily_report_markdown(
            signal_date="2026-08-24",
            execution_date="2026-08-25",
            top_df=oob_df,
            calibration_evidence=valid_cal_ev
        )
