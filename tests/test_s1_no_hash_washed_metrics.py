"""
Stage S1 测试套件: 杜绝哈希洗白指标、隔离遗留未实测产物与清洗残余伪造展示
tests/test_s1_no_hash_washed_metrics.py
"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from models.verified_metrics import (
    REQUIRED_METRICS_EVIDENCE_FIELDS,
    VerifiedResearchMetricsLoader,
    compute_sha256,
    load_verified_research_metrics,
    validate_scientific_evidence,
)
from notifications.webhook_notifier import QuantWebhookNotifier
from scheduler.notifier import MessageNotifier


def test_hash_washing_rejected_without_physical_provenance(tmp_path):
    """验证仅有 SHA-256 哈希清单但无物理实测对账凭证的伪造产物会被严格拒绝"""
    fake_metrics_file = tmp_path / "fake_metrics.json"
    fake_payload = {
        "model_id": "gen4_flagship",
        "sharpe_ratio": 2.89,
        "annualized_return": 26.12,
        "max_drawdown": -5.56,
    }
    fake_metrics_file.write_text(json.dumps(fake_payload), encoding="utf-8")

    # 1. 缺少 19 项全血缘字段时校验失败
    is_valid, reason = validate_scientific_evidence(fake_payload, base_dir=tmp_path)
    assert not is_valid
    assert "Missing required evidence field" in reason

    # 2. 补齐字段但缺少物理文件时依然拒绝
    full_payload = {
        "model_id": "gen4_flagship",
        "run_id": "run_test_001",
        "git_commit": "abcdef123456",
        "data_snapshot_hash": "hash_data",
        "factor_matrix_hash": "hash_factor",
        "train_start_date": "2021-09-29",
        "train_end_date": "2025-12-31",
        "test_start_date": "2026-01-01",
        "test_end_date": "2026-08-24",
        "nav_file_path": "non_existent_nav.csv",
        "nav_file_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "trade_ledger_path": "non_existent_trades.csv",
        "trade_ledger_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "orders_file_path": "non_existent_orders.csv",
        "orders_file_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "annualized_return": 0.2612,
        "sharpe_ratio": 2.89,
        "max_drawdown": 0.0556,
        "information_ratio": 2.10,
    }
    is_valid, reason = validate_scientific_evidence(full_payload, base_dir=tmp_path)
    assert not is_valid
    assert "does not exist" in reason

    # 3. 创建物理 NAV、Trade、Orders 文件并哈希匹配，但指标数学重算不符时拒绝
    nav_file = tmp_path / "nav.csv"
    trade_file = tmp_path / "trades.csv"
    orders_file = tmp_path / "orders.csv"

    # 生成一条总收益只有 5% 的净值序列
    nav_df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=100), "nav": np.linspace(1.0, 1.05, 100)})
    nav_df.to_csv(nav_file, index=False)
    trade_file.write_text("trade_id,date,symbol,price,shares\n", encoding="utf-8")
    orders_file.write_text("order_id,date,symbol,shares\n", encoding="utf-8")

    full_payload["nav_file_path"] = str(nav_file)
    full_payload["nav_file_sha256"] = compute_sha256(nav_file)
    full_payload["trade_ledger_path"] = str(trade_file)
    full_payload["trade_ledger_sha256"] = compute_sha256(trade_file)
    full_payload["orders_file_path"] = str(orders_file)
    full_payload["orders_file_sha256"] = compute_sha256(orders_file)
    # 声明的指标是 26% 年化，而实际净值仅有 ~12%
    full_payload["annualized_return"] = 0.2612

    is_valid, reason = validate_scientific_evidence(full_payload, base_dir=tmp_path)
    assert not is_valid
    assert "does not match declared" in reason

    # 4. load_verified_research_metrics 拦截
    valid_json = tmp_path / "verified_metrics.json"
    valid_json.write_text(json.dumps(full_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Scientific evidence validation failed"):
        load_verified_research_metrics(valid_json, base_dir=tmp_path)


def test_legacy_unverified_quarantine_and_exclusion():
    """验证性能指标文件被标记为 LEGACY_UNVERIFIED 并从产品界面物理排除"""
    rep_dir = settings.BASE_DIR / "reports"
    perf_path = rep_dir / "performance_all_generations.json"
    manifest_path = rep_dir / "performance_all_generations.manifest.json"
    quarantine_dir = rep_dir / "legacy_unverified"

    # 1. 确认物理隔离目录与清单
    assert quarantine_dir.exists()
    assert (quarantine_dir / "quarantine_manifest.json").exists()
    assert (quarantine_dir / "performance_all_generations.json").exists()
    assert (quarantine_dir / "performance_all_generations.manifest.json").exists()

    # 2. 确认文件元数据标记为 LEGACY_UNVERIFIED
    perf_data = json.loads(perf_path.read_text(encoding="utf-8"))
    assert perf_data.get("_metadata", {}).get("verification_status") == "LEGACY_UNVERIFIED"
    assert perf_data.get("_metadata", {}).get("excluded_from_product_display") is True
    assert perf_data.get("_metadata", {}).get("excluded_from_scientific_evidence") is True

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data.get("verification_status") == "LEGACY_UNVERIFIED"
    assert manifest_data.get("excluded_from_product_display") is True

    # 3. 前端展示加载时必须 Fail-Closed 返回 None
    loader = VerifiedResearchMetricsLoader()
    assert loader.load_multi_generation_performance(for_product_display=True) is None
    assert loader.get_multi_generation_metrics_table(for_product_display=True) is None

    # 4. load_verified_research_metrics 必须直接抛出 ValueError
    with pytest.raises(ValueError, match="LEGACY_UNVERIFIED"):
        load_verified_research_metrics(perf_path)


def test_verified_metrics_no_zero_fallbacks():
    """验证指标对比表在数据字段缺失时展示 '-'，绝不使用 0.0 或 0.00 默认兜底"""
    loader = VerifiedResearchMetricsLoader()
    # 非前端展示模式下加载表结构进行字段缺失格式验证
    table = loader.get_multi_generation_metrics_table(for_product_display=False)
    assert table is not None
    assert not table.empty

    # 模拟缺失数据的代际
    incomplete_perf = {
        "_metadata": {},
        "gen1_baseline": {},
        "gen2_plans_1_to_4": {},
        "gen3_plans_5_7_9": {},
        "gen4_flagship": {},
    }
    # 临时覆盖加载逻辑验证格式化行为
    import unittest.mock as mock
    with mock.patch.object(loader, "load_multi_generation_performance", return_value=incomplete_perf):
        empty_table = loader.get_multi_generation_metrics_table(for_product_display=False)
        assert empty_table is not None
        for col in ["第一代 (客观基准)", "第二代 (系统增强)", "第三代 (风险自适应)", "第四代 (全景旗舰)"]:
            for val in empty_table[col]:
                assert val == "-", f"Missing field formatted as {val} instead of '-'"
                assert "0.00" not in str(val)


def test_webhook_notifier_strict_validation_and_no_percentage():
    """验证全渠道通知严格校验字段且禁止预测超额百分比伪造"""
    notifier = QuantWebhookNotifier(webhook_url=None)

    # 1. 缺失 pred_score 时抛出 ValueError
    with pytest.raises(ValueError, match="missing required 'pred_score'"):
        notifier.format_markdown_report(
            "2026-08-24",
            [{"symbol": "600519.SH", "close": 1600.0, "target_weight": 0.2}]
        )

    # 2. NaN / Inf 分数时抛出 ValueError
    with pytest.raises(ValueError, match="invalid pred_score"):
        notifier.format_markdown_report(
            "2026-08-24",
            [{"symbol": "600519.SH", "close": 1600.0, "target_weight": 0.2, "pred_score": float("nan")}]
        )

    # 3. 缺失 close 时抛出 ValueError
    with pytest.raises(ValueError, match="missing required 'close'"):
        notifier.format_markdown_report(
            "2026-08-24",
            [{"symbol": "600519.SH", "target_weight": 0.2, "pred_score": 0.85}]
        )

    # 4. 缺失 target_weight 时抛出 ValueError
    with pytest.raises(ValueError, match="missing required 'target_weight'"):
        notifier.format_markdown_report(
            "2026-08-24",
            [{"symbol": "600519.SH", "close": 1600.0, "pred_score": 0.85}]
        )

    # 5. 合法输入下展示纯排序分且绝无“预测超额”
    valid_stocks = [
        {"symbol": "600519.SH", "name": "贵州茅台", "industry": "白酒", "close": 1600.0, "pred_score": 0.8523, "target_weight": 0.20}
    ]
    report = notifier.format_markdown_report("2026-08-24", valid_stocks)
    assert "模型排序分数: 0.8523" in report
    assert "预测超额" not in report
    assert "85.2%" not in report


def test_scheduler_notifier_uncalibrated_no_bracket_percentage():
    """验证 scheduler notifier 对未校准排序分展示纯 0.8500，无 (85.0%) 括号伪造"""
    top_df = pd.DataFrame([
        {"symbol": "600519.SH", "name": "贵州茅台", "industry": "白酒", "close": 1600.0, "pred_score": 0.8500, "target_weight": 0.20}
    ])
    md = MessageNotifier.format_daily_report_markdown(
        signal_date="2026-08-24",
        execution_date="2026-08-25",
        top_df=top_df,
        macro_status="正常持仓",
        is_calibrated_probability=False
    )
    assert "**0.8500**" in md
    assert "(85.0%)" not in md
    assert "85.0%" not in md
    assert "模型排序分数" in md
    assert "预测上涨概率" not in md


def test_dashboard_and_pipeline_no_bypass():
    """验证 dashboard 和 run_pipeline 无直接绕过或伪造概率文案"""
    # 1. dashboard/app.py 校验
    app_code = (settings.BASE_DIR / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert 'with open(perf_all_path, "r", encoding="utf-8") as f:' not in app_code
    assert "metrics_loader.load_multi_generation_performance(for_product_display=True)" in app_code

    # 2. run_pipeline.py 校验
    pipeline_code = (settings.BASE_DIR / "run_pipeline.py").read_text(encoding="utf-8")
    assert "日上涨概率" not in pipeline_code
    assert "日预期超额" not in pipeline_code
    assert "模型排序分数" in pipeline_code


def test_scheduler_notifier_macro_status_does_not_trigger_pseudo_probability():
    """验证宏观状态 (正常多头持仓/正常多头运行) 绝对无法触发伪概率展示"""
    top_df = pd.DataFrame([
        {"symbol": "600519.SH", "name": "贵州茅台", "industry": "白酒", "close": 1600.0, "pred_score": 0.8500, "target_weight": 0.20}
    ])
    for status_str in ["正常多头持仓", "正常多头运行", "防御低仓位", "全面空仓防守"]:
        md = MessageNotifier.format_daily_report_markdown(
            signal_date="2026-08-24",
            execution_date="2026-08-25",
            top_df=top_df,
            macro_status=status_str
        )
        assert "模型排序分数" in md
        assert "预测上涨概率" not in md
        assert "**0.8500**" in md
        assert "85.0%" not in md
        assert "85.00%" not in md


def test_scheduler_notifier_calibration_requires_typed_evidence(tmp_path, monkeypatch):
    """验证必须提供强类型且通过物理检验的 ProbabilityCalibrationEvidence，伪造或篡改一律 Fail-Closed"""
    from data.crypto_anchor import TRUSTED_KEY_REGISTRY, generate_keypair, sign_with_environment_key
    from models.verified_metrics import (
        ProbabilityCalibrationEvidence,
        compute_sha256,
        load_verified_probability_calibration,
    )

    top_df = pd.DataFrame([
        {"symbol": "600519.SH", "name": "贵州茅台", "industry": "白酒", "close": 1600.0, "pred_score": 0.8500, "target_weight": 0.20}
    ])

    # 1. 纯布尔声明没有物理证据 -> 强制报错
    with pytest.raises(ValueError, match="Cannot claim calibrated probability without verified"):
        MessageNotifier.format_daily_report_markdown(
            signal_date="2026-08-24",
            execution_date="2026-08-25",
            top_df=top_df,
            is_calibrated_probability=True
        )

    # 2. 伪造/任意 dict 作为凭证 -> 强制报错
    with pytest.raises(ValueError, match="Invalid or unverified ProbabilityCalibrationEvidence"):
        MessageNotifier.format_daily_report_markdown(
            signal_date="2026-08-24",
            execution_date="2026-08-25",
            top_df=top_df,
            calibration_evidence={"method": "fake", "status": "VERIFIED"}
        )

    # 3. 构造全 "a"*64, "b"*64, "c"*64 假哈希伪凭证 (文件不存在) -> 严格拒绝
    dummy_ev = ProbabilityCalibrationEvidence(
        method="isotonic",
        brier_score=0.12,
        sample_count=2000,
        calibrator_artifact_path=str(tmp_path / "non_existent_cal.bin"),
        calibrator_artifact_sha256="b" * 64,
        reliability_curve_artifact_path=str(tmp_path / "non_existent_rel.json"),
        reliability_curve_sha256="a" * 64,
        calibration_dataset_manifest_path=str(tmp_path / "non_existent_man.json"),
        calibration_dataset_sha256="c" * 64,
        model_id="gen4_flagship",
        feature_schema_hash="d" * 64,
        signed_at="2026-08-24T12:00:00Z",
        signer_key_id="PROD_MODEL_PROMOTION_KEY_2026_V1",
        signature="00" * 64,
        status="VERIFIED"
    )
    assert not dummy_ev.is_valid()
    with pytest.raises(ValueError, match="Invalid or unverified ProbabilityCalibrationEvidence"):
        MessageNotifier.format_daily_report_markdown(
            signal_date="2026-08-24",
            execution_date="2026-08-25",
            top_df=top_df,
            calibration_evidence=dummy_ev
        )

    # 准备真实测试密钥与物理文件
    sk_bytes, pk_bytes = generate_keypair()
    key_id = "S1_TEST_CALIB_KEY_2026"
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
    cal_file.write_bytes(b"calibrator_model_binary_bytes")
    cal_sha = compute_sha256(cal_file)

    rel_file = tmp_path / "reliability_curve.json"
    rel_file.write_text(json.dumps({"bins": [0.1, 0.2, 0.5]}), encoding="utf-8")
    rel_sha = compute_sha256(rel_file)

    schema_hash = "f" * 64
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
        "brier_score": 0.12,
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

    # 4. 物理文件被篡改 (修改 calibrator.bin 内容) -> 校验失败
    tampered_cal_file = tmp_path / "tampered_cal.bin"
    tampered_cal_file.write_bytes(b"corrupted_or_modified_calibrator_data")
    tampered_ev = ProbabilityCalibrationEvidence(
        method="isotonic",
        brier_score=0.12,
        sample_count=2000,
        calibrator_artifact_path=str(tampered_cal_file),
        calibrator_artifact_sha256=cal_sha,  # SHA 仍为原值，实际已被篡改
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
    assert not tampered_ev.is_valid()

    # 5. 模型 ID 或 feature_schema_hash 不一致 -> 校验失败
    mismatched_ev = ProbabilityCalibrationEvidence(
        method="isotonic",
        brier_score=0.12,
        sample_count=2000,
        calibrator_artifact_path=str(cal_file),
        calibrator_artifact_sha256=cal_sha,
        reliability_curve_artifact_path=str(rel_file),
        reliability_curve_sha256=rel_sha,
        calibration_dataset_manifest_path=str(manifest_file),
        calibration_dataset_sha256=manifest_sha,
        model_id="gen3_mismatched_model",  # 模型 ID 与 manifest 冲突
        feature_schema_hash=schema_hash,
        signed_at=signed_at,
        signer_key_id=key_id,
        signature=sig,
        status="VERIFIED"
    )
    assert not mismatched_ev.is_valid()

    # 6. load_verified_probability_calibration 受控加载器测试
    cert_doc = {
        "method": "isotonic",
        "brier_score": 0.12,
        "sample_count": 2000,
        "calibrator_artifact_path": str(cal_file),
        "calibrator_artifact_sha256": cal_sha,
        "reliability_curve_artifact_path": str(rel_file),
        "reliability_curve_sha256": rel_sha,
        "calibration_dataset_manifest_path": str(manifest_file),
        "calibration_dataset_sha256": manifest_sha,
        "model_id": "gen4_flagship",
        "feature_schema_hash": schema_hash,
        "signed_at": signed_at,
        "signer_key_id": key_id,
        "signature": sig,
        "status": "VERIFIED"
    }
    cert_path = tmp_path / "valid_calibration_artifact.json"
    cert_path.write_text(json.dumps(cert_doc, indent=2), encoding="utf-8")

    loaded_ev = load_verified_probability_calibration(cert_path)
    assert loaded_ev is not None
    assert loaded_ev.is_valid()

    # 7. 合法物理凭证展示“预测上涨概率 85.00%”
    md = MessageNotifier.format_daily_report_markdown(
        signal_date="2026-08-24",
        execution_date="2026-08-25",
        top_df=top_df,
        calibration_evidence=loaded_ev
    )
    assert "预测上涨概率" in md
    assert "85.00%" in md

    # 8. 无真实校准凭证时，受控加载器返回 None，严格展示“模型排序分数”
    non_existent_cert = tmp_path / "no_such_cert.json"
    assert load_verified_probability_calibration(non_existent_cert) is None
    md_default = MessageNotifier.format_daily_report_markdown(
        signal_date="2026-08-24",
        execution_date="2026-08-25",
        top_df=top_df,
        macro_status="正常多头持仓"
    )
    assert "模型排序分数" in md_default
    assert "预测上涨概率" not in md_default
    assert "85.00%" not in md_default
    assert "**0.8500**" in md_default


def test_calibration_evidence_dedicated_purpose_and_manifest_negative_cases(tmp_path, monkeypatch):
    """
    S1 强制负向审计:
    1. 使用 MODEL_PROMOTION 密钥签名的校准凭证必须拒绝;
    2. Manifest 缺 model_id 必须拒绝;
    3. Manifest 缺 feature_schema_hash 必须拒绝;
    4. Manifest 数据集哈希不匹配必须拒绝.
    """
    from data.crypto_anchor import TRUSTED_KEY_REGISTRY, generate_keypair, sign_with_environment_key
    from models.verified_metrics import ProbabilityCalibrationEvidence, compute_sha256

    cal_file = tmp_path / "neg_calibrator.bin"
    cal_file.write_bytes(b"calibrator_data_neg")
    cal_sha = compute_sha256(cal_file)

    rel_file = tmp_path / "neg_curve.json"
    rel_file.write_text(json.dumps({"curve": [0.1, 0.2]}), encoding="utf-8")
    rel_sha = compute_sha256(rel_file)

    valid_schema_hash = "a" * 64
    valid_dataset_sha = "b" * 64
    signed_at = "2026-08-24T15:00:00Z"

    # 生成专用校准密钥 (PROBABILITY_CALIBRATION_AUTHORIZATION)
    sk_cal_bytes, pk_cal_bytes = generate_keypair()
    cal_key_id = "CAL_AUTH_DEDICATED_KEY"
    monkeypatch.setitem(TRUSTED_KEY_REGISTRY, cal_key_id, {
        "algorithm": "ED25519",
        "key_id": cal_key_id,
        "public_key_hex": pk_cal_bytes.hex(),
        "allowed_purposes": ["PROBABILITY_CALIBRATION_AUTHORIZATION"],
        "status": "ACTIVE",
        "not_before": "2025-01-01T00:00:00Z",
        "not_after": "2030-01-01T00:00:00Z",
        "is_production": False,
    })

    # 生成仅有 MODEL_PROMOTION 权限的替代密钥
    sk_promo_bytes, pk_promo_bytes = generate_keypair()
    promo_key_id = "MODEL_PROMOTION_KEY_REJECTED"
    monkeypatch.setitem(TRUSTED_KEY_REGISTRY, promo_key_id, {
        "algorithm": "ED25519",
        "key_id": promo_key_id,
        "public_key_hex": pk_promo_bytes.hex(),
        "allowed_purposes": ["MODEL_PROMOTION"],
        "status": "ACTIVE",
        "not_before": "2025-01-01T00:00:00Z",
        "not_after": "2030-01-01T00:00:00Z",
        "is_production": False,
    })

    # 合法 Manifest
    manifest_file = tmp_path / "neg_manifest.json"
    manifest_data = {
        "model_id": "gen4_flagship",
        "feature_schema_hash": valid_schema_hash,
        "dataset_sha256": valid_dataset_sha
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")
    manifest_sha = compute_sha256(manifest_file)

    payload = {
        "method": "isotonic",
        "brier_score": 0.12,
        "sample_count": 2000,
        "calibrator_artifact_sha256": cal_sha,
        "reliability_curve_sha256": rel_sha,
        "calibration_dataset_sha256": manifest_sha,
        "model_id": "gen4_flagship",
        "feature_schema_hash": valid_schema_hash,
        "signed_at": signed_at,
        "status": "VERIFIED",
    }
    msg_bytes = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")

    # 1. 使用 MODEL_PROMOTION 密钥签名的校准凭证 -> 严正拒绝
    sig_promo, _ = sign_with_environment_key(
        message=msg_bytes,
        key_id=promo_key_id,
        required_purpose="MODEL_PROMOTION",
        domain_separator="QUANT_PROBABILITY_CALIBRATION_V1",
        explicit_private_key_hex=sk_promo_bytes.hex()
    )
    ev_promo = ProbabilityCalibrationEvidence(
        method="isotonic",
        brier_score=0.12,
        sample_count=2000,
        calibrator_artifact_path=str(cal_file),
        calibrator_artifact_sha256=cal_sha,
        reliability_curve_artifact_path=str(rel_file),
        reliability_curve_sha256=rel_sha,
        calibration_dataset_manifest_path=str(manifest_file),
        calibration_dataset_sha256=manifest_sha,
        model_id="gen4_flagship",
        feature_schema_hash=valid_schema_hash,
        signed_at=signed_at,
        signer_key_id=promo_key_id,
        signature=sig_promo,
        status="VERIFIED"
    )
    assert not ev_promo.is_valid()

    # 正常签名用于后续 Manifest 缺失负向测试
    sig_cal, _ = sign_with_environment_key(
        message=msg_bytes,
        key_id=cal_key_id,
        required_purpose="PROBABILITY_CALIBRATION_AUTHORIZATION",
        domain_separator="QUANT_PROBABILITY_CALIBRATION_V1",
        explicit_private_key_hex=sk_cal_bytes.hex()
    )

    # 2. Manifest 缺 model_id 必须拒绝
    man_no_mid = tmp_path / "man_no_mid.json"
    man_no_mid.write_text(json.dumps({
        "feature_schema_hash": valid_schema_hash,
        "dataset_sha256": valid_dataset_sha
    }), encoding="utf-8")
    ev_no_mid = ProbabilityCalibrationEvidence(
        method="isotonic",
        brier_score=0.12,
        sample_count=2000,
        calibrator_artifact_path=str(cal_file),
        calibrator_artifact_sha256=cal_sha,
        reliability_curve_artifact_path=str(rel_file),
        reliability_curve_sha256=rel_sha,
        calibration_dataset_manifest_path=str(man_no_mid),
        calibration_dataset_sha256=compute_sha256(man_no_mid),
        model_id="gen4_flagship",
        feature_schema_hash=valid_schema_hash,
        signed_at=signed_at,
        signer_key_id=cal_key_id,
        signature=sig_cal,
        status="VERIFIED"
    )
    assert not ev_no_mid.is_valid()

    # 3. Manifest 缺 feature_schema_hash 必须拒绝
    man_no_fhash = tmp_path / "man_no_fhash.json"
    man_no_fhash.write_text(json.dumps({
        "model_id": "gen4_flagship",
        "dataset_sha256": valid_dataset_sha
    }), encoding="utf-8")
    ev_no_fhash = ProbabilityCalibrationEvidence(
        method="isotonic",
        brier_score=0.12,
        sample_count=2000,
        calibrator_artifact_path=str(cal_file),
        calibrator_artifact_sha256=cal_sha,
        reliability_curve_artifact_path=str(rel_file),
        reliability_curve_sha256=rel_sha,
        calibration_dataset_manifest_path=str(man_no_fhash),
        calibration_dataset_sha256=compute_sha256(man_no_fhash),
        model_id="gen4_flagship",
        feature_schema_hash=valid_schema_hash,
        signed_at=signed_at,
        signer_key_id=cal_key_id,
        signature=sig_cal,
        status="VERIFIED"
    )
    assert not ev_no_fhash.is_valid()

    # 4. Manifest 数据集哈希不匹配必须拒绝 (物理 dataset_path 不匹配 或 expected 不匹配)
    man_wrong_ds = tmp_path / "man_wrong_ds.json"
    raw_dataset_file = tmp_path / "actual_dataset.parquet"
    raw_dataset_file.write_bytes(b"actual_dataset_bytes_for_test")
    man_wrong_ds.write_text(json.dumps({
        "model_id": "gen4_flagship",
        "feature_schema_hash": valid_schema_hash,
        "dataset_sha256": "c" * 64,
        "dataset_path": str(raw_dataset_file)
    }), encoding="utf-8")
    ev_wrong_ds = ProbabilityCalibrationEvidence(
        method="isotonic",
        brier_score=0.12,
        sample_count=2000,
        calibrator_artifact_path=str(cal_file),
        calibrator_artifact_sha256=cal_sha,
        reliability_curve_artifact_path=str(rel_file),
        reliability_curve_sha256=rel_sha,
        calibration_dataset_manifest_path=str(man_wrong_ds),
        calibration_dataset_sha256=compute_sha256(man_wrong_ds),
        model_id="gen4_flagship",
        feature_schema_hash=valid_schema_hash,
        signed_at=signed_at,
        signer_key_id=cal_key_id,
        signature=sig_cal,
        status="VERIFIED"
    )
    assert not ev_wrong_ds.is_valid()
    assert not ev_promo.is_valid(expected_dataset_sha256="wrong_expected" * 4)

