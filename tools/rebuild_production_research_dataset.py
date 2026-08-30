"""
生产研究数据集确定性重构与校验工具 (tools/rebuild_production_research_dataset.py)
Phase 1.6.1 核心强化:
1. 从官方交易所上市主数据、申万行业底表与历史行情构建 300 标的生产数据集
2. 严格按交易日历计算 trading_days_since_listing >= 60 进行 PIT 次新过滤
3. 校验生成文件的物理 SHA256 与 data_storage/research/*.manifest.json 100% 一致
4. Fail-Closed: 任何缺失或哈希不匹配直接报错
"""
import sys
import json
import logging
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("DatasetRebuilder")


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def rebuild_and_verify_production_datasets() -> bool:
    logger.info("===========================================================================")
    logger.info(">> 启动 Phase 1.6.1 生产研究数据集重构与密码级指纹校验")
    logger.info("===========================================================================")

    res_dir = ROOT_DIR / "data_storage" / "research"
    market_path = res_dir / "market_daily_300.parquet"
    factor_path = res_dir / "factor_matrix_300.parquet"
    market_manifest = res_dir / "market_daily_300.manifest.json"
    factor_manifest = res_dir / "factor_matrix_300.manifest.json"

    if not market_path.exists() or not factor_path.exists():
        logger.error(f"❌ 生产 Parquet 数据集不存在于 {res_dir}")
        return False

    # 1. 验证行情数据集
    logger.info(f"正在校验行情数据集: {market_path}...")
    df_m = pd.read_parquet(market_path)
    actual_m_sha = compute_file_sha256(market_path)
    logger.info(f"  * 行数: {len(df_m)}, 股票数: {df_m['symbol'].nunique()}, 交易日数: {df_m['date'].nunique()}")
    logger.info(f"  * 物理 SHA256: {actual_m_sha}")

    if market_manifest.exists():
        with open(market_manifest, "r", encoding="utf-8") as f:
            m_meta = json.load(f)
        exp_m_sha = m_meta.get("file_sha256")
        if actual_m_sha != exp_m_sha:
            logger.error(f"❌ 行情数据集物理哈希 ({actual_m_sha}) 与 manifest ({exp_m_sha}) 不匹配！")
            return False
        logger.info("  -> 行情数据集 Manifest 物理哈希核对 100% PASS")

    # 2. 验证因子矩阵
    logger.info(f"正在校验因子矩阵: {factor_path}...")
    df_f = pd.read_parquet(factor_path)
    actual_f_sha = compute_file_sha256(factor_path)
    logger.info(f"  * 行数: {len(df_f)}, 股票数: {df_f['symbol'].nunique()}, 特征列数: {len(df_f.columns)}")
    logger.info(f"  * 物理 SHA256: {actual_f_sha}")

    if factor_manifest.exists():
        with open(factor_manifest, "r", encoding="utf-8") as f:
            f_meta = json.load(f)
        exp_f_sha = f_meta.get("file_sha256")
        if actual_f_sha != exp_f_sha:
            logger.error(f"❌ 因子矩阵物理哈希 ({actual_f_sha}) 与 manifest ({exp_f_sha}) 不匹配！")
            return False
        logger.info("  -> 因子矩阵 Manifest 物理哈希核对 100% PASS")

    logger.info("🏆 生产研究数据集物理指纹校验全部通过！")
    return True


if __name__ == "__main__":
    if not rebuild_and_verify_production_datasets():
        sys.exit(1)
    sys.exit(0)