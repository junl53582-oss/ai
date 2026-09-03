"""
300 标的因子矩阵重建与一致性验证工具 (tools/build_factor_matrix_300.py)

背景 (Phase A / 2026-09-01 架构审计):
    data_storage/research/factor_matrix_300.parquet (311MB) 超出 GitHub 单文件 100MB 硬限,
    不适合直接入库。原计划从已入库的 market_daily_300.parquet 确定性重建, 但根因排查
    (2026-09-01) 证明原认证件【不可位级复现】, 证据链如下:

    [根因 1] 血缘污染: market_daily_300.parquet 的 LOG_CIRC_MV 列逐日 mean=0/std=1,
             是因子管线标准化后的值, 不是原始对数市值 (~22-27)。
             原始 circ_mv 只能由 amount/turnover 近似重建, 无法位级还原
             (DataManager 原始 log_circ_mv 来自真实流通股本数据, 未入库)。
    [根因 2] 构建配方未锁定: 原矩阵构建于 2026-08-30 09:26 的一次性会话,
             ENABLE_REGISTRY_FACTORS / ENABLE_FUNDAMENTALS 的构建期取值
             (True/False) 与当前配置 (False/True) 漂移, 且未记录在任何 manifest 中。
    [后果]   LOG_CIRC_MV 是中性化回归的市值控制变量, 其值漂移导致全部 97 个因子
             值连锁偏差 (重建值与认证件相关性高但数值不同)。

职责 (判决后的重新定位):
    1. 本工具保留为【一致性探测器】: 对比重建件与历史认证件的差异, 输出诊断
    2. 它同时是【数据集 v2 重建器】的基底: v2 必须先修好 market 数据集保存路径
       (存原始 circ_mv 而非标准化值), 再以本工具锁定的配方重建并重新认证

用法:
    python tools/build_factor_matrix_300.py            # 只读验证 (当前预期 FAIL, 属诚实行为)
    python tools/build_factor_matrix_300.py --output X # 额外把重建结果写到 X
"""
import sys
import hashlib
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from factors.processor import FactorProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("FactorMatrix300Rebuilder")

MARKET_300 = ROOT_DIR / "data_storage" / "research" / "market_daily_300.parquet"
FACTOR_300 = ROOT_DIR / "data_storage" / "research" / "factor_matrix_300.parquet"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
    return h.hexdigest().lower()


def rebuild_factor_matrix_300(output_path: Path | None = None) -> bool:
    if not MARKET_300.exists():
        logger.error(f"行情数据集不存在: {MARKET_300} (应先入库提交)")
        return False
    if not FACTOR_300.exists():
        logger.error(f"原因子矩阵不存在: {FACTOR_300} (本地重建基准缺失)")
        return False

    # =====================================================================
    # 构建配方 (部分已锁定, 2026-09-01):
    #   1. ENABLE_REGISTRY_FACTORS=True  ✓ 已证实 (97 因子含 19 注册表因子, 含 3 GP_ALPHA)
    #   2. ENABLE_FUNDAMENTALS=False     ✓ 已证实 (原矩阵无 F_* 列)
    #   3. 原始 log_circ_mv 透传         ✗ 不可实现 —— 入库 parquet 的 LOG_CIRC_MV
    #      已被标准化污染 (逐日 mean=0/std=1), 原始 circ_mv 未入库, 无法位级还原。
    #      => 历史认证件不可复现, 本工具降级为一致性探测器; 修复路径 = 数据集 v2。
    # =====================================================================
    from config.settings import settings as _s
    _saved = (_s.ENABLE_REGISTRY_FACTORS, _s.ENABLE_FUNDAMENTALS)
    _s.ENABLE_REGISTRY_FACTORS = True
    _s.ENABLE_FUNDAMENTALS = False

    try:
        logger.info("加载行情数据集...")
        market_df = pd.read_parquet(MARKET_300)

        logger.info("经 FactorProcessor 重建因子矩阵 (临时目录, 不触碰原文件)...")
        with tempfile.TemporaryDirectory() as td:
            processor = FactorProcessor(factor_dir=Path(td))
            rebuilt = processor.build_and_save_factor_matrix(market_df, force_update=True)
    finally:
        _s.ENABLE_REGISTRY_FACTORS, _s.ENABLE_FUNDAMENTALS = _saved

    original = pd.read_parquet(FACTOR_300, columns=None)

    # ---------- 结构对比 ----------
    cols_ok = list(rebuilt.columns) == list(original.columns)
    if not cols_ok:
        only_orig = [c for c in original.columns if c not in rebuilt.columns]
        only_new = [c for c in rebuilt.columns if c not in original.columns]
        logger.error(f"列结构不一致! 仅原有: {only_orig[:10]} | 仅重建: {only_new[:10]}")
    rows_ok = len(rebuilt) == len(original)
    syms_ok = rebuilt["symbol"].nunique() == original["symbol"].nunique()

    # ---------- 值对比 (全列数值全量比对, NaN 位置必须一致) ----------
    common = [c for c in original.columns if c in rebuilt.columns]
    val_mismatches = []
    for c in common:
        a, b = original[c], rebuilt[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            na_a, na_b = a.isna(), b.isna()
            if not (na_a == na_b).all():
                val_mismatches.append(c)
                continue
            mask = ~na_a
            if mask.any():
                if not np.allclose(a[mask].astype(float), b[mask].astype(float), rtol=1e-9, atol=1e-12, equal_nan=True):
                    val_mismatches.append(c)
        else:
            if not (a.fillna("~") == b.fillna("~")).all():
                val_mismatches.append(c)

    logger.info("=" * 60)
    logger.info(f"列结构一致: {cols_ok} | 行数一致: {rows_ok} ({len(rebuilt)} vs {len(original)}) | 标的数一致: {syms_ok}")
    logger.info(f"值全等列: {len(common) - len(val_mismatches)}/{len(common)} | 不一致列: {val_mismatches[:10]}")
    logger.info("=" * 60)

    all_ok = cols_ok and rows_ok and syms_ok and not val_mismatches

    if all_ok and output_path is not None:
        rebuilt.to_parquet(output_path, index=False)
        logger.info(f"重建结果已写出: {output_path} (SHA256: {compute_file_sha256(output_path)})")
    elif not all_ok:
        logger.error("重建与原文件不一致——CI 不得用重建件替换原认证件! 请排查 build 路径差异。")

    return all_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="300 标的因子矩阵确定性重建与一致性验证")
    parser.add_argument("--output", type=str, default=None, help="验证通过时写出重建文件的目标路径 (缺省仅验证)")
    args = parser.parse_args()
    out = Path(args.output) if args.output else None
    ok = rebuild_factor_matrix_300(out)
    sys.exit(0 if ok else 1)
