"""
数据集 v3 构建器: PIT 复权链重建 (tools/build_dataset_300_v3.py)

背景 (Phase A / 2026-09-01):
    v2 因子矩阵仍使用 qfq 派生的 adj 价格 (血缘污染的 legacy 链路遗留)。
    v3 用 PIT 安全复权链 (data/adjustment_factors.py 的 hfq-factor 事件表) 重建:
        raw 价格 × f_t / f_base  (f_base = START_DATE 处因子, 固定基准)
    产出:
        data_storage/research/market_daily_300_v3.parquet   (行情 v3, PIT 复权)
        data_storage/research/factor_matrix_300_v3.parquet  (因子矩阵 v3, 本地)
        data_storage/research/*_v3.manifest.json            (配方锁定 manifest)

预期影响:
    复权基准从"最新日期"改为"固定基准日"——RETURNS 不变 (比率尺度无关),
    仅绝对价格水平整体缩放, 标准化后因子值应近似不变 (v2→v3 对比相关性应 ≈ 0.99+)。
    若对比显著漂移, 则需以 v3 重跑因子研究。

用法:
    python tools/build_dataset_300_v3.py
"""
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from data.adjustment_factors import AdjustmentFactorProvider  # noqa: E402
from factors.processor import FactorProcessor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("Dataset300V3Builder")

RES_DIR = ROOT / "data_storage" / "research"
V2_MARKET = RES_DIR / "market_daily_300_v2.parquet"
V3_MARKET = RES_DIR / "market_daily_300_v3.parquet"
V3_FACTOR = RES_DIR / "factor_matrix_300_v3.parquet"

RECIPE = {
    "recipe_version": "3.0",
    "adjustment": "PIT hfq-factor (adj_price_t = raw_t * f_t / f_base, f_base=START_DATE fixed)",
    "settings_snapshot": {"ENABLE_REGISTRY_FACTORS": True, "ENABLE_FUNDAMENTALS": False},
}


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    logger.info("[1/4] 加载 v2 行情面板 (含原始 turnower/amount 与 raw 价格)...")
    market = pd.read_parquet(V2_MARKET)
    logger.info(f"      {len(market)} 行 / {market['symbol'].nunique()} 标的")

    logger.info("[2/4] 拉取 hfq-factor 事件表并按 PIT 复权重算 adj 价格...")
    provider = AdjustmentFactorProvider(cache_dir=ROOT / "data_storage" / "adjust_factors")
    out = market.copy()
    for sym, grp in out.groupby("symbol", sort=False):
        f = provider.get_daily_factor_series(str(sym), grp["date"])
        f_base = float(f.iloc[0]) if len(f) and f.iloc[0] > 0 else 1.0
        ratio = f / f_base
        for raw_c, adj_c in [("open", "adj_open"), ("high", "adj_high"), ("low", "adj_low"), ("close", "adj_close")]:
            if raw_c in grp.columns:
                out.loc[grp.index, adj_c] = (pd.to_numeric(grp[raw_c], errors="coerce") * ratio.values)
    out["adjustment_mode"] = "hfq_factor_pit"
    out.to_parquet(V3_MARKET, index=False, engine="pyarrow", compression="snappy")
    logger.info(f"      行情 v3 落盘: {V3_MARKET} ({_sha256(V3_MARKET)[:12]}...)")

    logger.info("[3/4] 锁定配方构建因子矩阵 (REGISTRY=True / FUNDAMENTALS=False)...")
    _saved = (settings.ENABLE_REGISTRY_FACTORS, settings.ENABLE_FUNDAMENTALS)
    settings.ENABLE_REGISTRY_FACTORS = True
    settings.ENABLE_FUNDAMENTALS = False
    try:
        with tempfile.TemporaryDirectory() as td:
            factor_df = FactorProcessor(factor_dir=Path(td)).build_and_save_factor_matrix(out, force_update=True)
    finally:
        settings.ENABLE_REGISTRY_FACTORS, settings.ENABLE_FUNDAMENTALS = _saved
    factor_df.to_parquet(V3_FACTOR, index=False, engine="pyarrow", compression="snappy")

    logger.info("[4/4] 写 manifest...")
    manifest = {
        "dataset_name": "A_SHARE_PIT_RESEARCH_FACTOR_MATRIX_300_V3",
        "dataset_version": "3.0",
        "recipe": RECIPE,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "parent_market_v2_hash": _sha256(V2_MARKET),
        "market_v3_sha256": _sha256(V3_MARKET),
        "factor_v3_sha256": _sha256(V3_FACTOR),
        "row_count": int(len(factor_df)),
        "symbol_count": int(factor_df["symbol"].nunique()),
        "note": "PIT adj prices (hfq-factor); compare v2-vs-v3 factor correlation before deciding research rerun",
    }
    (RES_DIR / "factor_matrix_300_v3.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"🏆 v3 完成: factor_matrix_300_v3.parquet ({_sha256(V3_FACTOR)[:12]}...)")


if __name__ == "__main__":
    main()
