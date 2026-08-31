"""
数据集 v2 构建器: 300 标的生产行情面板 + 因子矩阵 (tools/build_dataset_300_v2.py)

背景 (Phase A / 2026-09-01 根因排查判决):
    历史认证件 factor_matrix_300.parquet 不可位级复现, 且 market_daily_300.parquet
    的 LOG_CIRC_MV 列被因子管线标准化值污染 (逐日 mean=0/std=1, 原始对数市值应 ~20-28)。
    原始 circ_mv 可由原始 amount/turnover 列经 DataManager 同款公式精确重建
    (data_manager.py:693-696), 本工具据此生成可复现的数据集 v2。

v2 与 legacy 的差异 (全部有意为之):
    1. LOG_CIRC_MV 存原始对数市值 (非标准化值) —— 修复血缘污染
    2. 构建配方锁定: ENABLE_REGISTRY_FACTORS=True / ENABLE_FUNDAMENTALS=False
    3. manifest 记录完整配方 (代码 commit / settings 快照 / 公式版本 / 输入哈希)
    4. 内置列值分布断言 (governance 盲区修复, fail-closed)

产出:
    data_storage/research/market_daily_300_v2.parquet       (入库, ~20MB)
    data_storage/research/market_daily_300_v2.manifest.json (入库)
    data_storage/research/factor_matrix_300_v2.parquet      (本地, ~311MB, 不入库)
    data_storage/research/factor_matrix_300_v2.manifest.json(入库)

用法:
    python tools/build_dataset_300_v2.py
"""
import sys
import json
import hashlib
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from factors.processor import FactorProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("Dataset300V2Builder")

RES_DIR = ROOT_DIR / "data_storage" / "research"
LEGACY_MARKET = RES_DIR / "market_daily_300.parquet"
V2_MARKET = RES_DIR / "market_daily_300_v2.parquet"
V2_FACTOR = RES_DIR / "factor_matrix_300_v2.parquet"

RECIPE = {
    "recipe_version": "2.0",
    "raw_log_circ_mv_formula": "np.log(np.maximum(amount / turnover.replace(0,nan).fillna(0.01), 1e8))  # == data_manager.py:693-696",
    "settings_snapshot": {
        "ENABLE_REGISTRY_FACTORS": True,
        "ENABLE_FUNDAMENTALS": False,
    },
    "universe_state": "inherited from legacy market_daily_300.parquet in_universe column (PIT incomplete, see RUNTIME_ATTESTATION)",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def build_dataset_v2() -> bool:
    if not LEGACY_MARKET.exists():
        logger.error(f"legacy 行情数据集不存在: {LEGACY_MARKET}")
        return False

    # ---------- 1. 重建原始 log_circ_mv (修复血缘污染) ----------
    logger.info("[1/5] 加载 legacy 行情面板并重建原始 log_circ_mv ...")
    market_df = pd.read_parquet(LEGACY_MARKET)
    if "turnover" not in market_df.columns or "amount" not in market_df.columns:
        logger.error("❌ 缺少 turnover/amount 原始列, 无法重建原始流通市值")
        return False
    valid_turnover = market_df["turnover"].replace(0, np.nan).fillna(0.01)
    raw_mv = np.log(np.maximum(market_df["amount"] / valid_turnover, 1e8))
    polluted = market_df["LOG_CIRC_MV"].copy()

    # 列值分布断言 (fail-closed)
    g_mean, g_std = float(raw_mv.mean()), float(raw_mv.std())
    logger.info(f"  * 重建原始 log_circ_mv: 全局 mean={g_mean:.3f} std={g_std:.3f} 范围 [{raw_mv.min():.2f}, {raw_mv.max():.2f}]")
    if not (15.0 <= g_mean <= 35.0):
        logger.error("❌ 重建值域异常, 拒绝继续 (疑似输入列本身被污染)")
        return False
    # 验证 legacy 列确实是污染的 (逐日 mean≈0), 记录证据
    per_day_mean = polluted.groupby(market_df["date"]).mean()
    logger.info(f"  * legacy LOG_CIRC_MV 逐日 mean 范围: [{per_day_mean.min():.4f}, {per_day_mean.max():.4f}] (≈0 即证实污染)")

    market_df["LOG_CIRC_MV"] = raw_mv

    # ---------- 2. 行情 v2 落盘 + manifest ----------
    logger.info("[2/5] 写出 market_daily_300_v2.parquet ...")
    market_df.to_parquet(V2_MARKET, index=False, engine="pyarrow", compression="snappy")
    market_manifest = {
        "dataset_name": "A_SHARE_PIT_RESEARCH_MARKET_DAILY_300_V2",
        "dataset_version": "2.0",
        "is_ci_fixture": False,
        "production_ready": True,
        "recipe": RECIPE,
        "code_commit": git_commit(),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "parent_dataset": {
            "path": "data_storage/research/market_daily_300.parquet",
            "file_sha256": file_sha256(LEGACY_MARKET),
            "note": "LOG_CIRC_MV column in parent is lineage-polluted (standardized); v2 restores raw values",
        },
        "symbol_count": int(market_df["symbol"].nunique()),
        "row_count": int(len(market_df)),
        "date_range": [str(market_df["date"].min()), str(market_df["date"].max())],
        "file_sha256": file_sha256(V2_MARKET),
    }
    (RES_DIR / "market_daily_300_v2.manifest.json").write_text(
        json.dumps(market_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- 3. 锁定配方构建因子矩阵 ----------
    logger.info("[3/5] 以锁定配方构建因子矩阵 (REGISTRY=True / FUNDAMENTALS=False) ...")
    _saved = (settings.ENABLE_REGISTRY_FACTORS, settings.ENABLE_FUNDAMENTALS)
    settings.ENABLE_REGISTRY_FACTORS = True
    settings.ENABLE_FUNDAMENTALS = False
    try:
        with tempfile.TemporaryDirectory() as td:
            processor = FactorProcessor(factor_dir=Path(td))
            factor_df = processor.build_and_save_factor_matrix(market_df, force_update=True)
    finally:
        settings.ENABLE_REGISTRY_FACTORS, settings.ENABLE_FUNDAMENTALS = _saved

    # ---------- 4. 产出断言 (fail-closed) ----------
    logger.info("[4/5] 产出断言 ...")
    assert len(factor_df) == len(market_df), "因子矩阵行数与行情面板不一致"
    assert "LOG_CIRC_MV" in factor_df.columns
    # 标准化后逐日 mean≈0/std≈1 是【预期行为】(矩阵内列), 与行情集的原始值语义区分
    z = factor_df["LOG_CIRC_MV"].dropna()
    logger.info(f"  * 因子矩阵 LOG_CIRC_MV (标准化后): 全局 mean={z.mean():.4f} std={z.std():.4f}")
    factor_count = len([c for c in factor_df.columns if c not in market_df.columns or c == "LOG_CIRC_MV"])

    # ---------- 5. 因子矩阵 v2 落盘 + manifest ----------
    logger.info("[5/5] 写出 factor_matrix_300_v2.parquet ...")
    factor_df.to_parquet(V2_FACTOR, index=False, engine="pyarrow", compression="snappy")
    factor_manifest = {
        "dataset_name": "A_SHARE_PIT_RESEARCH_FACTOR_MATRIX_300_V2",
        "dataset_version": "2.0",
        "is_ci_fixture": False,
        "production_ready": True,
        "recipe": RECIPE,
        "code_commit": git_commit(),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "parent_market_manifest_hash": hashlib.sha256(
            (RES_DIR / "market_daily_300_v2.manifest.json").read_bytes()
        ).hexdigest(),
        "symbol_count": int(factor_df["symbol"].nunique()),
        "row_count": int(len(factor_df)),
        "column_count": int(len(factor_df.columns)),
        "factor_count": int(factor_count),
        "supersedes": "factor_matrix_300.parquet (legacy, unreproducible: lineage-polluted LOG_CIRC_MV + unlocked recipe)",
        "file_sha256": file_sha256(V2_FACTOR),
    }
    (RES_DIR / "factor_matrix_300_v2.manifest.json").write_text(
        json.dumps(factor_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info("🏆 数据集 v2 构建完成:")
    logger.info(f"  * market_daily_300_v2.parquet  sha256={market_manifest['file_sha256'][:16]}...")
    logger.info(f"  * factor_matrix_300_v2.parquet sha256={factor_manifest['file_sha256'][:16]}...")
    logger.info("  * 下一步: 以 v2 数据集重跑 factor research + model research 重新认证")
    return True


if __name__ == "__main__":
    ok = build_dataset_v2()
    sys.exit(0 if ok else 1)
