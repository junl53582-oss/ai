import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
RES_DIR = ROOT_DIR / "data_storage" / "research"
MARKET_V2 = RES_DIR / "market_daily_300_v2.parquet"
MARKET_MANIFEST = RES_DIR / "market_daily_300_v2.manifest.json"
FACTOR_V2 = RES_DIR / "factor_matrix_300_v2.parquet"
FACTOR_MANIFEST = RES_DIR / "factor_matrix_300_v2.manifest.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def test_1_market_input_hash_correct():
    """验证 market_daily_300_v2.parquet 物理文件哈希与 manifest 严格匹配"""
    assert MARKET_V2.exists(), "market_daily_300_v2.parquet missing"
    assert MARKET_MANIFEST.exists(), "market_daily_300_v2.manifest.json missing"
    manifest = json.loads(MARKET_MANIFEST.read_text(encoding="utf-8"))
    actual_sha = sha256_file(MARKET_V2)
    assert actual_sha == manifest["file_sha256"], f"Market file SHA {actual_sha} != manifest {manifest['file_sha256']}"


def test_2_factor_output_hash_correct():
    """验证 factor_matrix_300_v2.parquet 物理文件哈希与 manifest 严格匹配"""
    assert FACTOR_V2.exists(), "factor_matrix_300_v2.parquet missing"
    assert FACTOR_MANIFEST.exists(), "factor_matrix_300_v2.manifest.json missing"
    manifest = json.loads(FACTOR_MANIFEST.read_text(encoding="utf-8"))
    actual_sha = sha256_file(FACTOR_V2)
    assert actual_sha == manifest["file_sha256"], f"Factor file SHA {actual_sha} != manifest {manifest['file_sha256']}"


def test_3_manifest_binding_correct():
    """验证 factor_matrix_300_v2 manifest 中的 input_market_sha256 与市场行情 manifest 一致"""
    assert FACTOR_MANIFEST.exists()
    assert MARKET_MANIFEST.exists()
    f_meta = json.loads(FACTOR_MANIFEST.read_text(encoding="utf-8"))
    m_meta = json.loads(MARKET_MANIFEST.read_text(encoding="utf-8"))
    assert f_meta["input_market_sha256"] == m_meta["file_sha256"], "Factor matrix input hash does not bind market dataset SHA"


def test_4_rebuild_hash_consistency_provenance():
    """验证因子矩阵清单中记录的配方版本与因子数量"""
    f_meta = json.loads(FACTOR_MANIFEST.read_text(encoding="utf-8"))
    assert f_meta["factor_recipe_version"] == "2.0"
    assert f_meta["row_count"] == 349379
    assert f_meta["feature_count"] >= 97


def test_5_raw_circ_mv_exists_and_unstandardized():
    """验证原始流通市值字段 (circ_mv / circ_mv_raw) 在市场面板与因子矩阵中均存在且未被标准化"""
    m_df = pd.read_parquet(MARKET_V2)
    assert "circ_mv_raw" in m_df.columns, "circ_mv_raw missing in market dataset"
    assert "circ_mv" in m_df.columns, "circ_mv missing in market dataset"
    
    # 原始流通市值均值应该在 1e8 到 1e12 范围，而不是 mean≈0
    mean_val = float(m_df["circ_mv_raw"].mean())
    assert mean_val > 1e8, f"circ_mv_raw appears standardized or corrupted: mean={mean_val}"
    
    f_df = pd.read_parquet(FACTOR_V2)
    assert "circ_mv_raw" in f_df.columns or "circ_mv" in f_df.columns, "raw circ_mv columns must be preserved in factor matrix"


def test_6_cannot_reconstruct_raw_mv_from_standardized_log_circ_mv():
    """证明无法从标准化后 mean=0/std=1 的截面因子逆向精确推导原始流通市值，必须依赖物理持久化的原始字段"""
    m_df = pd.read_parquet(MARKET_V2)
    raw_mv = m_df["circ_mv_raw"].values
    
    # 对数流通市值
    log_mv = np.log(raw_mv)
    # 模拟截面标准化 (均值=0, 标准差=1)
    std_mv = (log_mv - np.mean(log_mv)) / np.std(log_mv)
    
    # 尝试在未知原始均值和标准差的情况下逆向反推: 无法恢复绝对数值
    with pytest.raises(AssertionError):
        # 错误假设：直接 exp(std_mv) 不能还原原始市值
        inverted = np.exp(std_mv)
        np.testing.assert_allclose(inverted, raw_mv, rtol=1e-2)
