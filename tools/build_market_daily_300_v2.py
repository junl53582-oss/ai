"""
Deterministic Builder for market_daily_300_v2.parquet (tools/build_market_daily_300_v2.py)
"""
import sys
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RES_DIR = ROOT_DIR / "data_storage" / "research"
LEGACY_MARKET = RES_DIR / "market_daily_300.parquet"
V2_MARKET = RES_DIR / "market_daily_300_v2.parquet"
V2_MANIFEST = RES_DIR / "market_daily_300_v2.manifest.json"


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


def build_market_df() -> pd.DataFrame:
    if not LEGACY_MARKET.exists():
        raise FileNotFoundError(f"Legacy market dataset not found: {LEGACY_MARKET}")
    
    df = pd.read_parquet(LEGACY_MARKET)
    
    # Calculate raw circulating market cap
    valid_turnover = df["turnover"].replace(0, np.nan).fillna(0.01)
    raw_circ_mv = np.maximum(df["amount"] / valid_turnover, 1e8).astype(np.float64)
    
    df["circ_mv_raw"] = raw_circ_mv
    df["circ_mv"] = raw_circ_mv
    df["LOG_CIRC_MV"] = np.log(raw_circ_mv).astype(np.float64)
    
    # Ensure canonical column order
    ordered_cols = [
        "date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover",
        "circ_mv_raw", "circ_mv", "LOG_CIRC_MV",
        "adj_open", "adj_high", "adj_low", "adj_close",
        "benchmark_open", "benchmark_close",
        "in_universe", "is_suspended", "is_st", "is_limit_up_locked", "is_limit_down_locked",
        "limit_up_price", "limit_down_price", "industry", "list_date", "days_since_listing"
    ]
    df = df[ordered_cols].sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def write_market_v2_and_manifest(target_parquet: Path, target_manifest: Path, created_at: str = None) -> tuple[str, dict]:
    df = build_market_df()
    
    # Save to parquet deterministically
    df.to_parquet(target_parquet, index=False, engine="pyarrow", compression="snappy")
    
    file_sha = compute_file_sha256(target_parquet)
    
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
        
    manifest = {
        "schema_version": "2.0",
        "created_at": created_at,
        "source_commit": get_git_commit(),
        "builder_script": "tools/build_market_daily_300_v2.py",
        "input_sources": [
            {
                "path": "data_storage/research/market_daily_300.parquet",
                "sha256": compute_file_sha256(LEGACY_MARKET),
                "role": "canonical_300_symbol_market_feed"
            }
        ],
        "row_count": int(len(df)),
        "symbol_count": int(df["symbol"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "file_sha256": file_sha
    }
    target_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_sha, manifest


if __name__ == "__main__":
    sha, m = write_market_v2_and_manifest(V2_MARKET, V2_MANIFEST)
    print(f"Built market_daily_300_v2.parquet: SHA256={sha}, rows={m['row_count']}, symbols={m['symbol_count']}")
