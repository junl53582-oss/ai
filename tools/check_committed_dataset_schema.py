"""
数据集与因子矩阵 Schema 门禁检查器 (tools/check_committed_dataset_schema.py)
用于在 CI / 预运行阶段严格校验本地与已提交数据集的字段完整性。
若缺少必要价格、基准或可交易性字段，直接 fail-closed 退出 (exit code 1)。
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("SchemaGate")

REQUIRED_MARKET_COLS = [
    "date", "symbol", "open", "high", "low", "close",
    "adj_open", "adj_high", "adj_low", "adj_close",
    "volume", "amount",
    "benchmark_open", "benchmark_close",
    "in_universe", "is_suspended", "is_st",
    "is_limit_up_locked", "is_limit_down_locked",
    "limit_up_price", "limit_down_price"
]

REQUIRED_FACTOR_CORE_COLS = [
    "date", "symbol",
    "adj_open", "adj_close",
    "benchmark_open", "benchmark_close",
    "in_universe"
]


def check_market_dataset(path: Path) -> bool:
    logger.info(f"🔍 检查行情数据集 Schema: {path}")
    if not path.exists():
        logger.error(f"❌ 行情数据集文件不存在: {path}")
        return False
    
    try:
        df = pd.read_parquet(path)
        if df.empty:
            logger.error("❌ 行情数据集为空 (0 rows)！")
            return False

        missing = [c for c in REQUIRED_MARKET_COLS if c not in df.columns]
        if missing:
            logger.error(f"❌ 行情数据集缺少必要字段 ({len(missing)} 个): {missing}")
            return False

        # 检查 benchmark_open 与 benchmark_close 覆盖率
        b_open_cov = float((df["benchmark_open"] > 0).mean())
        b_close_cov = float((df["benchmark_close"] > 0).mean())
        logger.info(f"  * 行数: {len(df)}, 标的数: {df['symbol'].nunique()}")
        logger.info(f"  * benchmark_open 覆盖率: {b_open_cov*100:.1f}%")
        logger.info(f"  * benchmark_close 覆盖率: {b_close_cov*100:.1f}%")

        if b_open_cov < 0.80 or b_close_cov < 0.80:
            logger.error(f"❌ 基准数据覆盖率不达标 (open: {b_open_cov:.2f}, close: {b_close_cov:.2f})！")
            return False

        # ---------- 列值分布断言 (Phase A / 2026-09-01 新增, 堵 governance 盲区) ----------
        # 背景: market_daily_300.parquet 的 LOG_CIRC_MV 曾被因子管线的标准化值污染
        # (逐日 mean=0/std=1) 且所有既有 gate 未拦截。原始对数流通市值 (yuan) 应在
        # ~[18, 32] 区间; 若被标准化, 全局均值≈0。此断言按 fail-closed 处理。
        if "LOG_CIRC_MV" in df.columns:
            mv = df["LOG_CIRC_MV"].dropna()
            if len(mv) > 0:
                g_mean, g_std = float(mv.mean()), float(mv.std())
                logger.info(f"  * LOG_CIRC_MV 全局 mean={g_mean:.3f} std={g_std:.3f} (原始值期望 ~23±3)")
                if not (15.0 <= g_mean <= 35.0):
                    logger.error(
                        f"❌ LOG_CIRC_MV 值域异常 (全局均值 {g_mean:.3f})! 疑似标准化值污染原始行情列。"
                        "Fail-Closed: 请重建数据集 (tools/build_dataset_300_v2.py)。"
                    )
                    return False

        logger.info("  -> 行情数据集 Schema 校验通过！")
        return True
    except Exception as e:
        logger.error(f"❌ 读取行情数据集异常: {e}")
        return False


def check_factor_matrix(path: Path) -> bool:
    logger.info(f"🔍 检查因子矩阵 Schema: {path}")
    if not path.exists():
        logger.error(f"❌ 因子矩阵文件不存在: {path}")
        return False

    try:
        df = pd.read_parquet(path)
        if df.empty:
            logger.error("❌ 因子矩阵为空 (0 rows)！")
            return False

        missing = [c for c in REQUIRED_FACTOR_CORE_COLS if c not in df.columns]
        if missing:
            logger.error(f"❌ 因子矩阵缺少核心基准/行情字段 ({len(missing)} 个): {missing}")
            return False

        b_open_cov = float((df["benchmark_open"] > 0).mean())
        b_close_cov = float((df["benchmark_close"] > 0).mean())
        logger.info(f"  * 行数: {len(df)}, 标的数: {df['symbol'].nunique()}, 总列数: {len(df.columns)}")
        logger.info(f"  * benchmark_open 覆盖率: {b_open_cov*100:.1f}%")

        if b_open_cov < 0.80 or b_close_cov < 0.80:
            logger.error(f"❌ 因子矩阵基准数据覆盖率不达标 (open: {b_open_cov:.2f}, close: {b_close_cov:.2f})！")
            return False

        logger.info("  -> 因子矩阵 Schema 校验通过！")
        return True
    except Exception as e:
        logger.error(f"❌ 读取因子矩阵异常: {e}")
        return False


def main():
    root = Path(__file__).resolve().parent.parent
    market_path = root / "data_storage" / "parquet" / "market_daily.parquet"
    factor_path = root / "data_storage" / "factors" / "factor_matrix.parquet"

    m_ok = check_market_dataset(market_path)
    f_ok = check_factor_matrix(factor_path)

    # 检查 Phase 1.6 生产级 300 标的数据集 (若存在)
    prod_m_path = root / "data_storage" / "research" / "market_daily_300.parquet"
    prod_f_path = root / "data_storage" / "research" / "factor_matrix_300.parquet"
    prod_ok = True
    if prod_m_path.exists():
        logger.info("📦 发现 Phase 1.6 生产级 300 标的数据集，执行生产级 Schema 门禁检查...")
        prod_m_ok = check_market_dataset(prod_m_path)
        prod_f_ok = check_factor_matrix(prod_f_path) if prod_f_path.exists() else True
        prod_ok = prod_m_ok and prod_f_ok

    if m_ok and f_ok and prod_ok:
        logger.info("🏆 全量数据集 Schema 门禁校验 100% 通过！(DATASET_SCHEMA_VALID = TRUE)")
        sys.exit(0)
    else:
        logger.error("❌ 数据集 Schema 门禁校验失败！(DATASET_SCHEMA_VALID = FALSE)")
        sys.exit(1)


if __name__ == "__main__":
    main()
