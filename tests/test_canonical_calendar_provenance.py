import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
REF_DIR = ROOT_DIR / "data_storage" / "reference"
CALENDAR_PARQUET = REF_DIR / "canonical_calendar_v1.parquet"
CALENDAR_MANIFEST = REF_DIR / "canonical_calendar_v1.manifest.json"
MARKET_V2 = ROOT_DIR / "data_storage" / "research" / "market_daily_300_v2.parquet"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def test_1_canonical_calendar_files_exist():
    """验证规范日历物理文件与 manifest 存在"""
    assert CALENDAR_PARQUET.exists(), "canonical_calendar_v1.parquet missing"
    assert CALENDAR_MANIFEST.exists(), "canonical_calendar_v1.manifest.json missing"


def test_2_calendar_physical_hash_matches_manifest():
    """验证物理 Parquet 哈希与 manifest 中的 calendar_artifact_sha256 一致"""
    manifest = json.loads(CALENDAR_MANIFEST.read_text(encoding="utf-8"))
    actual_sha = sha256_file(CALENDAR_PARQUET)
    assert actual_sha == manifest["calendar_artifact_sha256"], (
        f"Artifact SHA mismatch: actual={actual_sha} != manifest={manifest['calendar_artifact_sha256']}"
    )


def test_3_calendar_invariants_and_dates_integrity():
    """验证 1187 个交易日严格递增、无重复、哈希严格一致"""
    manifest = json.loads(CALENDAR_MANIFEST.read_text(encoding="utf-8"))
    dates = manifest.get("dates", [])
    assert len(dates) == 1187, f"Expected 1187 dates, got {len(dates)}"
    assert dates == sorted(dates), "Dates must be strictly ascending"
    assert len(dates) == len(set(dates)), "Dates must contain no duplicates"
    
    cal_hash = hashlib.sha256("\n".join(dates).encode("utf-8")).hexdigest()
    assert manifest["calendar_sha256"] == cal_hash, "Calendar SHA fingerprint mismatch"


def test_4_certification_gate_compliance():
    """严格按照 certification.py CANONICAL_CALENDAR_PROVENANCE 门禁规则检验字段"""
    manifest = json.loads(CALENDAR_MANIFEST.read_text(encoding="utf-8"))
    dates = manifest.get("dates", [])
    cal_hash = hashlib.sha256("\n".join(dates).encode()).hexdigest() if dates else ""
    
    assert manifest.get("run_mode") == "certified", "run_mode must be 'certified'"
    assert manifest.get("calendar_source") == "SSE_SZSE_CANONICAL_CALENDAR"
    assert manifest.get("calendar_source") not in {"DATASET_DERIVED", "SYNTHETIC_TEST_CALENDAR"}
    assert bool(manifest.get("calendar_artifact_sha256"))
    assert bool(dates) and dates == sorted(dates) and len(dates) == len(set(dates))
    assert manifest.get("dataset_overlap_count", 0) > 0
    assert manifest.get("calendar_sha256") == cal_hash
    assert bool(manifest.get("source_code_sha"))


def test_5_market_v2_calendar_overlap():
    """验证市场 V2 数据集的日期完全落在规范日历中"""
    if not MARKET_V2.exists():
        pytest.skip("market_daily_300_v2.parquet missing")
    m_df = pd.read_parquet(MARKET_V2, columns=["date"])
    m_dates = set(m_df["date"].astype(str).unique())
    manifest = json.loads(CALENDAR_MANIFEST.read_text(encoding="utf-8"))
    cal_dates = set(manifest.get("dates", []))
    
    assert m_dates.issubset(cal_dates), "Market dates contain dates outside canonical trading calendar"
