"""
Deterministic Factor Matrix V2 Builder (tools/build_factor_matrix_300_v2.py)
"""
import sys
import json
import hashlib
import tempfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from factors.processor import FactorProcessor

RES_DIR = ROOT_DIR / "data_storage" / "research"
V2_MARKET = RES_DIR / "market_daily_300_v2.parquet"
V2_FACTOR = RES_DIR / "factor_matrix_300_v2.parquet"
V2_FACTOR_MANIFEST = RES_DIR / "factor_matrix_300_v2.manifest.json"

FACTOR_RECIPE = {
    "recipe_version": "2.0",
    "ENABLE_REGISTRY_FACTORS": True,
    "ENABLE_FUNDAMENTALS": False,
    "neutralization": {
        "enabled": True,
        "industry_col": "industry",
        "market_cap_col": "LOG_CIRC_MV"
    },
    "standardization": "cross_sectional_zscore"
}


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    except Exception:
        return "UNKNOWN"


def build_factor_matrix_v2_df(market_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure market_df has required raw market cap
    if "circ_mv_raw" not in market_df.columns or "circ_mv" not in market_df.columns:
        valid_turnover = market_df["turnover"].replace(0, np.nan).fillna(0.01)
        raw_circ_mv = np.maximum(market_df["amount"] / valid_turnover, 1e8).astype(np.float64)
        market_df = market_df.copy()
        market_df["circ_mv_raw"] = raw_circ_mv
        market_df["circ_mv"] = raw_circ_mv
        market_df["LOG_CIRC_MV"] = np.log(raw_circ_mv).astype(np.float64)

    _saved = (settings.ENABLE_REGISTRY_FACTORS, settings.ENABLE_FUNDAMENTALS)
    settings.ENABLE_REGISTRY_FACTORS = FACTOR_RECIPE["ENABLE_REGISTRY_FACTORS"]
    settings.ENABLE_FUNDAMENTALS = FACTOR_RECIPE["ENABLE_FUNDAMENTALS"]
    try:
        with tempfile.TemporaryDirectory() as td:
            processor = FactorProcessor(factor_dir=Path(td))
            factor_df = processor.build_and_save_factor_matrix(market_df, force_update=True)
    finally:
        settings.ENABLE_REGISTRY_FACTORS, settings.ENABLE_FUNDAMENTALS = _saved

    # Guarantee canonical sorting for bitwise reproducibility
    factor_df = factor_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return factor_df


def write_factor_matrix_v2_and_manifest(
    target_parquet: Path,
    target_manifest: Path,
    market_parquet: Path = V2_MARKET,
    created_at: str = None
) -> tuple[str, dict]:
    if not market_parquet.exists():
        raise FileNotFoundError(f"V2 market dataset not found: {market_parquet}")

    market_df = pd.read_parquet(market_parquet)
    input_market_sha = compute_file_sha256(market_parquet)
    factor_config_sha = hashlib.sha256(json.dumps(FACTOR_RECIPE, sort_keys=True).encode("utf-8")).hexdigest()

    factor_df = build_factor_matrix_v2_df(market_df)

    # Save to parquet
    factor_df.to_parquet(target_parquet, index=False, engine="pyarrow", compression="snappy")
    file_sha = compute_file_sha256(target_parquet)

    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    factor_cols = [c for c in factor_df.columns if c not in market_df.columns or c in ["LOG_CIRC_MV"]]

    manifest = {
        "dataset_name": "A_SHARE_PIT_RESEARCH_FACTOR_MATRIX_300_V2",
        "dataset_version": "2.0",
        "factor_recipe_version": FACTOR_RECIPE["recipe_version"],
        "recipe": FACTOR_RECIPE,
        "factor_config_sha256": factor_config_sha,
        "input_market_sha256": input_market_sha,
        "source_commit": get_git_commit(),
        "created_at": created_at,
        "builder_script": "tools/build_factor_matrix_300_v2.py",
        "row_count": int(len(factor_df)),
        "feature_count": int(len(factor_df.columns)),
        "factor_count": int(len(factor_cols)),
        "symbol_count": int(factor_df["symbol"].nunique()),
        "file_sha256": file_sha
    }
    target_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_sha, manifest


if __name__ == "__main__":
    sha, m = write_factor_matrix_v2_and_manifest(V2_FACTOR, V2_FACTOR_MANIFEST)
    print(f"Built factor_matrix_300_v2.parquet: SHA256={sha}, rows={m['row_count']}, features={m['feature_count']}")
