"""
数据集 v3 重建驱动 (workspace 侧): 300 标的研究面板 + 因子矩阵, 从 2026-09-07 全量新数据重建

背景:
    legacy factor_matrix_300.parquet 基于 pct_change 全 0 的坏数据构建 (7 大 headline
    Alpha 中 4 个失效, train/serve skew 实锤)。2026-09-07 联网重建的
    data_storage/parquet/market_daily.parquet (321 股, 499,080 行, 2020-01-02 ->
    2026-09-04) 已修复 pct_change (非零率 0.9749) 与 turnover (非零率 1.0)。
    本脚本从该干净数据重建研究数据集, 覆写 factor_matrix_300.parquet (下游
    predict_stocks.py / run_model_research.py 等十余处自动升级)。

血缘与配方 (继承 v2 判决并升级):
    1. universe: 继承 legacy market_daily_300.parquet 的 300 symbol 集合
       (PIT 不完整为已知既定状态, 与 v2 RECIPE 一致)
    2. LOG_CIRC_MV: 使用新面板原始 log_circ_mv (真实流通市值对数, 值域 ~19.9-28.8)
       —— 优于 v2 的 amount/turnover 近似公式, 直接修复 legacy 标准化污染
    3. 配方锁定: ENABLE_REGISTRY_FACTORS=True / ENABLE_FUNDAMENTALS=False
    4. fail-closed 断言: pct_change 非零率 / log_circ_mv 值域 / 4 个修复目标
       Alpha 的截面 IQR > 0 (train/serve skew 修复终审证据)

备份:
    legacy market_daily_300.parquet -> market_daily_300.parquet.bak_pre_v3 (首次运行时)
    (factor_matrix_300.parquet 已在此前隔离为 .bak_pre_rebuild, 本脚本无需再备份)
"""
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\股票预测")
sys.path.insert(0, str(ROOT))

PARQUET_NEW = ROOT / "data_storage" / "parquet" / "market_daily.parquet"
RES = ROOT / "data_storage" / "research"
LEGACY_MARKET = RES / "market_daily_300.parquet"
BACKUP_MARKET = RES / "market_daily_300.parquet.bak_pre_v3"
MARKET_300 = RES / "market_daily_300.parquet"
FACTOR_300 = RES / "factor_matrix_300.parquet"

# train/serve skew 的 4 个死亡 Alpha (模型 metadata IQR: 0.1229/0.3118/0.0207/0.0147)
DEAD_ALPHA_TARGETS = [
    "ALPHA_RESIDUAL_MOMENTUM_20",
    "ALPHA_QUALITY_X_MOMENTUM",
    "ALPHA_SHORT_REVERSAL_5",
    "ALPHA_IDIO_VOL_PENALTY",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("Dataset300V3Builder")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def main() -> bool:
    t0 = time.time()

    # ---------- 1. universe 提取 + 过滤 ----------
    logger.info("[1/6] 提取 legacy universe 并过滤新面板 ...")
    legacy_syms = set(pd.read_parquet(LEGACY_MARKET, columns=["symbol"])["symbol"].unique())
    logger.info("  legacy universe symbols: %d", len(legacy_syms))

    df = pd.read_parquet(PARQUET_NEW)
    m300 = df[df["symbol"].isin(legacy_syms)].copy().reset_index(drop=True)
    n_sym = m300["symbol"].nunique()
    missing = sorted(legacy_syms - set(m300["symbol"].unique()))
    logger.info("  过滤后: %d 行 | %d 股 | %s -> %s",
                len(m300), n_sym, m300["date"].min(), m300["date"].max())
    if missing:
        logger.warning("  universe 中缺失: %s", missing)
    assert n_sym >= 295, f"universe 覆盖率过低: {n_sym}/300"

    # ---------- 2. fail-closed 质量断言 ----------
    logger.info("[2/6] 质量断言 ...")
    pc = pd.to_numeric(m300["pct_change"], errors="coerce")
    pc_nz = float((pc != 0).mean())
    to_nz = float((pd.to_numeric(m300["turnover"], errors="coerce") != 0).mean())
    logger.info("  pct_change 非零率: %.4f | turnover 非零率: %.4f", pc_nz, to_nz)
    assert pc_nz > 0.5, f"pct_change 仍疑似坏数据 (非零率 {pc_nz:.4f})"

    lmv = pd.to_numeric(m300["log_circ_mv"], errors="coerce")
    g_mean, g_std = float(lmv.mean()), float(lmv.std())
    logger.info("  log_circ_mv: mean=%.3f std=%.3f range=[%.2f, %.2f]",
                g_mean, g_std, float(lmv.min()), float(lmv.max()))
    assert 15.0 <= g_mean <= 35.0, "log_circ_mv 值域异常 (疑似标准化污染)"

    m300["LOG_CIRC_MV"] = lmv
    m300["circ_mv"] = pd.to_numeric(m300["circ_mv"], errors="coerce")
    m300["circ_mv_raw"] = pd.to_numeric(m300["circ_mv_raw"], errors="coerce")

    # ---------- 3. 备份 legacy 并写出 market_daily_300.parquet ----------
    logger.info("[3/6] 写出 market_daily_300.parquet ...")
    if LEGACY_MARKET.exists() and not BACKUP_MARKET.exists():
        shutil.copy2(LEGACY_MARKET, BACKUP_MARKET)
        logger.info("  legacy market 已备份 -> %s", BACKUP_MARKET.name)
    m300.to_parquet(MARKET_300, index=False, engine="pyarrow", compression="snappy")
    market_sha = sha256_file(MARKET_300)
    logger.info("  written: %d 行, sha256=%s...", len(m300), market_sha[:16])

    market_manifest = {
        "dataset_name": "A_SHARE_PIT_RESEARCH_MARKET_DAILY_300",
        "dataset_version": "3.0",
        "is_ci_fixture": False,
        "production_ready": True,
        "recipe": {
            "recipe_version": "3.0",
            "universe": "inherited from legacy market_daily_300.parquet symbol set (PIT incomplete, known state)",
            "log_circ_mv": "raw from new market_daily.parquet log_circ_mv (real circulating MV, fixes legacy standardized pollution)",
            "settings_snapshot": {"ENABLE_REGISTRY_FACTORS": True, "ENABLE_FUNDAMENTALS": False},
        },
        "code_commit": git_commit(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent_dataset": {
            "path": "data_storage/parquet/market_daily.parquet",
            "file_sha256": sha256_file(PARQUET_NEW),
            "note": "full-market panel rebuilt 2026-09-07 via DataManager.sync_and_build_dataset (pct_change/turnover fixed)",
        },
        "symbol_count": int(n_sym),
        "row_count": int(len(m300)),
        "date_range": [str(m300["date"].min()), str(m300["date"].max())],
        "pct_change_nonzero_ratio": pc_nz,
        "turnover_nonzero_ratio": to_nz,
        "file_sha256": market_sha,
    }
    (RES / "market_daily_300.manifest.json").write_text(
        json.dumps(market_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- 4. 锁定配方构建因子矩阵 ----------
    logger.info("[4/6] 以锁定配方构建因子矩阵 (REGISTRY=True / FUNDAMENTALS=False) ...")
    from config.settings import settings
    from factors.processor import FactorProcessor

    _saved = (settings.ENABLE_REGISTRY_FACTORS, settings.ENABLE_FUNDAMENTALS)
    settings.ENABLE_REGISTRY_FACTORS = True
    settings.ENABLE_FUNDAMENTALS = False
    try:
        with tempfile.TemporaryDirectory() as td:
            processor = FactorProcessor(factor_dir=Path(td))
            factor_df = processor.build_and_save_factor_matrix(m300, force_update=True)
    finally:
        settings.ENABLE_REGISTRY_FACTORS, settings.ENABLE_FUNDAMENTALS = _saved

    # ---------- 5. 终审断言: 4 个死亡 Alpha 的截面 IQR ----------
    logger.info("[5/6] train/serve skew 修复终审 (4 个死亡 Alpha IQR) ...")
    expected_iqr = {
        "ALPHA_RESIDUAL_MOMENTUM_20": 0.1229,
        "ALPHA_QUALITY_X_MOMENTUM": 0.3118,
        "ALPHA_SHORT_REVERSAL_5": 0.0207,
        "ALPHA_IDIO_VOL_PENALTY": 0.0147,
    }
    alpha_report = {}
    for col in DEAD_ALPHA_TARGETS:
        if col not in factor_df.columns:
            logger.error("  ❌ %s 不在因子矩阵中!", col)
            alpha_report[col] = None
            continue
        s = pd.to_numeric(factor_df[col], errors="coerce").dropna()
        iqr = float(s.quantile(0.75) - s.quantile(0.25))
        alpha_report[col] = {"iqr": iqr, "expected_train_iqr": expected_iqr[col],
                             "nonconstant": iqr > 0}
        status = "✓ 有方差" if iqr > 0 else "❌ 恒为常量"
        logger.info("  %s: IQR=%.6f (训练时 %.4f) %s", col, iqr, expected_iqr[col], status)
    dead_now = [c for c, r in alpha_report.items() if r is None or not r["nonconstant"]]
    assert not dead_now, f"仍有恒常量 Alpha: {dead_now} —— skew 未修复, 拒绝写盘"

    # ---------- 6. 写出 factor_matrix_300.parquet + manifest ----------
    logger.info("[6/6] 写出 factor_matrix_300.parquet ...")
    assert len(factor_df) == len(m300), f"行数不一致: {len(factor_df)} vs {len(m300)}"
    factor_df.to_parquet(FACTOR_300, index=False, engine="pyarrow", compression="snappy")
    factor_sha = sha256_file(FACTOR_300)

    cs = factor_df.groupby("date")["symbol"].count() if "date" in factor_df.columns else pd.Series(dtype=int)
    factor_manifest = {
        "dataset_name": "A_SHARE_PIT_RESEARCH_FACTOR_MATRIX_300",
        "dataset_version": "3.0",
        "is_ci_fixture": False,
        "production_ready": True,
        "symbol_count": int(factor_df["symbol"].nunique()),
        "row_count": int(len(factor_df)),
        "date_min": str(factor_df["date"].min()),
        "date_max": str(factor_df["date"].max()),
        "date_range": [str(factor_df["date"].min()), str(factor_df["date"].max())],
        "daily_cross_section_median": float(cs.median()) if not cs.empty else 0.0,
        "feature_count": int(len(factor_df.columns)),
        "factor_count": int(len([c for c in factor_df.columns if c not in m300.columns or c == "LOG_CIRC_MV"])),
        "parent_market_manifest_hash": hashlib.sha256(
            (RES / "market_daily_300.manifest.json").read_bytes()
        ).hexdigest(),
        "builder_version": "workspace/build_dataset_300_v3.py",
        "code_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "alpha_revival_audit": alpha_report,
        "supersedes": "factor_matrix_300.parquet legacy (built from pct_change=0 poisoned data, 4 dead Alphas)",
        "file_sha256": factor_sha,
    }
    (RES / "factor_matrix_300.manifest.json").write_text(
        json.dumps(factor_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info("🏆 数据集 v3 构建完成 (%.1fs):", time.time() - t0)
    logger.info("  market_daily_300.parquet  %d 行 sha=%s...", len(m300), market_sha[:16])
    logger.info("  factor_matrix_300.parquet %d 行 %d 列 sha=%s...", len(factor_df), len(factor_df.columns), factor_sha[:16])
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
