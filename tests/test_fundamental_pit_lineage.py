import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
FUND_DIR = ROOT_DIR / "data_storage" / "fundamentals"
PIT_PARQUET = FUND_DIR / "fundamental_announcements_pit.parquet"
PIT_MANIFEST = FUND_DIR / "fundamental_pit_manifest.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def test_1_fundamental_pit_files_exist():
    """验证基本面 PIT 时间轴物理文件与 manifest 存在"""
    assert PIT_PARQUET.exists(), "fundamental_announcements_pit.parquet missing"
    assert PIT_MANIFEST.exists(), "fundamental_pit_manifest.json missing"


def test_2_fundamental_pit_hash_matches_manifest():
    """验证物理 Parquet 哈希与 manifest 中的 file_sha256 一致"""
    manifest = json.loads(PIT_MANIFEST.read_text(encoding="utf-8"))
    actual_sha = sha256_file(PIT_PARQUET)
    assert actual_sha == manifest["file_sha256"], f"SHA mismatch: actual={actual_sha} != manifest={manifest['file_sha256']}"


def test_3_pit_chronology_no_lookahead():
    """验证基本面公告时间轴无前视偏差（公告日必须大于等于报告期截止日）"""
    df = pd.read_parquet(PIT_PARQUET)
    assert "delay_days" in df.columns
    # 验证没有任何前视未来数据（delay_days < 0 为绝对禁止）
    lookahead_count = (df["delay_days"] < 0).sum()
    assert lookahead_count == 0, f"Found {lookahead_count} lookahead violations where announcement < report_date"


def test_4_manifest_certification_gate_compliance():
    """验证 manifest 严格满足 certification.py 中 STRICT_FUNDAMENTAL_PIT 门禁"""
    manifest = json.loads(PIT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest.get("synthetic_delay_certified_count") == 0, "synthetic_delay_certified_count must be 0"
    assert manifest.get("invalid_chronology_count") == 0, "invalid_chronology_count must be 0"
    assert manifest.get("official_announcement_rows", 0) > 0, "official_announcement_rows must be > 0"
    assert bool(manifest.get("source_code_sha")), "source_code_sha must be non-empty"


def test_5_fundamental_indicators_present():
    """验证核心财务指标字段存在且具备有效数据"""
    df = pd.read_parquet(PIT_PARQUET)
    core_cols = ["eps", "revenue", "net_profit", "roe", "bps"]
    for col in core_cols:
        assert col in df.columns, f"Core fundamental col {col} missing"
        assert df[col].notna().sum() > 0, f"Core col {col} is entirely null"
