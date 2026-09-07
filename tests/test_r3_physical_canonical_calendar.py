import json
import hashlib
import pytest
from pathlib import Path
import pandas as pd

from config.settings import settings
from data.trading_calendar import CanonicalTradingCalendar, TradeDateError


def test_physical_calendar_provenance_and_manifest_sha256():
    """验证物理日历及其 Manifest 的 SHA-256 签名一致性与严格物理覆盖边界"""
    parquet_path = settings.BASE_DIR / "data_storage" / "reference" / "canonical_calendar_v1.parquet"
    manifest_path = settings.BASE_DIR / "data_storage" / "reference" / "canonical_calendar_v1.manifest.json"

    assert parquet_path.exists(), "canonical_calendar_v1.parquet 必须存在"
    assert manifest_path.exists(), "canonical_calendar_v1.manifest.json 必须存在"

    # 1. 物理哈希验证
    actual_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest().lower()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert actual_hash == manifest_data["calendar_artifact_sha256"].lower()

    # 2. 覆盖范围必须严格对应物理文件 (2021-09-29 至 2026-08-24，共 1187 个交易日)
    cal = CanonicalTradingCalendar.get_instance()
    assert cal.date_min == "2021-09-29"
    assert cal.date_max == "2026-08-24"
    assert len(cal.trading_days_list) == 1187
    assert len(cal.trading_days_set) == 1187


def test_calendar_de_inference_and_coverage_blocked():
    """验证彻底移除推断化后，超出物理覆盖 (截至 2026-08-24) 的日期全部阻断并 fail-closed"""
    cal = CanonicalTradingCalendar.get_instance()

    # 物理覆盖内的交易日与休市日
    assert cal.is_trading_day("2026-08-21") is True   # 周五
    assert cal.is_trading_day("2026-08-22") is False  # 周六
    assert cal.is_trading_day("2026-08-23") is False  # 周日
    assert cal.is_trading_day("2026-08-24") is True   # 周一 (物理覆盖上限)

    # 物理覆盖之外的所有未来日期：严禁任何 weekday() < 5 推断，一律返回 False
    future_dates = [
        "2026-08-25",  # 周二
        "2026-08-26",  # 周三
        "2026-09-01",  # 9月周二
        "2026-09-04",  # 9月周五
        "2026-09-07",  # 9月下周一
        "2026-10-15",  # 10月
    ]
    for d in future_dates:
        assert cal.is_trading_day(d) is False, f"未覆盖日期 {d} 绝不能被判定为有效交易日"
        assert cal.check_coverage(d) == "CALENDAR_COVERAGE_BLOCKED"
        with pytest.raises(TradeDateError, match="CALENDAR_COVERAGE_BLOCKED"):
            cal.validate_trading_day(d)

    # 物理覆盖末日无法推断下一个交易日 (返回 None)
    assert cal.next_trading_day("2026-08-24") is None
    assert cal.next_trading_day("2026-08-25") is None


def test_calendar_tamper_fails_closed(tmp_path):
    """验证日历文件或 Manifest 遭篡改时 Fail-Closed 拒绝启动"""
    fake_parquet = tmp_path / "fake_cal.parquet"
    fake_manifest = tmp_path / "fake_cal.manifest.json"

    # 生成一个微型 parquet
    df = pd.DataFrame([{"date": "2026-08-24", "is_trading_day": True}])
    df.to_parquet(fake_parquet)

    # Manifest 哈希不匹配
    fake_manifest.write_text(json.dumps({
        "calendar_artifact_sha256": "bad_hash_00000000000000000000000000000000000000000000000000000000"
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 校验失败"):
        CanonicalTradingCalendar(parquet_path=fake_parquet, manifest_path=fake_manifest)
