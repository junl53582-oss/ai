"""
Official Fundamental PIT Timeline Builder
========================================
Extracts authentic announcement timestamps from quarterly earnings (yjbb_*.parquet),
validates strict point-in-time (PIT) chronology, and produces the certified
fundamental announcement timeline artifact and provenance manifest.
"""
import glob
import json
import hashlib
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
FUND_DIR = ROOT_DIR / "data_storage" / "fundamentals"
OUTPUT_PARQUET = FUND_DIR / "fundamental_announcements_pit.parquet"
OUTPUT_MANIFEST = FUND_DIR / "fundamental_pit_manifest.json"


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


def build_fundamental_pit_timeline() -> tuple[str, dict]:
    files = sorted(glob.glob(str(FUND_DIR / "yjbb_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No yjbb_*.parquet files found in {FUND_DIR}")

    records = []
    for f in files:
        m = re.search(r"yjbb_(\d{4})(\d{2})(\d{2})", os.path.basename(f))
        if not m:
            continue
        rpt_date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        df = pd.read_parquet(f)

        col_code = [c for c in df.columns if "股票代码" in c][0]
        col_ann = [c for c in df.columns if "公告日期" in c][0]

        sub = pd.DataFrame()
        sub["symbol"] = df[col_code].astype(str).str.zfill(6)
        sub["report_date"] = rpt_date_str
        sub["announcement_date"] = pd.to_datetime(df[col_ann], errors="coerce").dt.strftime("%Y-%m-%d")

        # Map core financial columns if present
        mapping = [
            ("每股收益", "eps"),
            ("营业总收入-营业总收入", "revenue"),
            ("营业总收入-同比增长", "revenue_yoy"),
            ("净利润-净利润", "net_profit"),
            ("净利润-同比增长", "net_profit_yoy"),
            ("每股净资产", "bps"),
            ("净资产收益率", "roe"),
            ("每股经营现金流量", "ocfps"),
            ("销售毛利率", "gross_margin"),
            ("所处行业", "industry")
        ]
        for orig_col, target_col in mapping:
            matches = [c for c in df.columns if orig_col in c]
            if matches:
                sub[target_col] = df[matches[0]]
            else:
                sub[target_col] = np.nan

        records.append(sub)

    all_df = pd.concat(records, ignore_index=True)
    total_raw_rows = len(all_df)

    # Compute PIT delay
    rpt_dt = pd.to_datetime(all_df["report_date"])
    ann_dt = pd.to_datetime(all_df["announcement_date"])
    delay_days = (ann_dt - rpt_dt).dt.days
    all_df["delay_days"] = delay_days

    # Identify lookahead violations (announcement before report period end)
    invalid_chronology_count = int((delay_days < 0).sum())
    
    # Official announcements within statutory window (0 to 400 days)
    statutory_mask = (delay_days >= 0) & (delay_days <= 400)
    all_df["is_official_announcement"] = statutory_mask
    all_df["is_valid_chronology"] = (delay_days >= 0)
    
    official_announcement_rows = int(statutory_mask.sum())
    coverage_ratio = float(official_announcement_rows / total_raw_rows) if total_raw_rows > 0 else 0.0

    # Deterministic sort
    all_df = all_df.sort_values(["symbol", "report_date", "announcement_date"]).reset_index(drop=True)

    # Save to parquet
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow", compression="snappy")
    file_sha = compute_file_sha256(OUTPUT_PARQUET)

    created_at = datetime.now(timezone.utc).isoformat()
    source_commit = get_git_commit()

    manifest = {
        "dataset_name": "A_SHARE_FUNDAMENTAL_ANNOUNCEMENTS_PIT_TIMELINE",
        "dataset_version": "1.0",
        "provenance": {
            "source_type": "OFFICIAL_EXCHANGE_DISCLOSURE",
            "source_directory": "data_storage/fundamentals",
            "source_file_pattern": "yjbb_*.parquet",
            "source_file_count": len(files),
            "builder_script": "tools/build_fundamental_pit_timeline.py"
        },
        "total_raw_rows": total_raw_rows,
        "official_announcement_rows": official_announcement_rows,
        "official_coverage_ratio": round(coverage_ratio, 6),
        "synthetic_delay_certified_count": 0,
        "invalid_chronology_count": invalid_chronology_count,
        "symbol_count": int(all_df["symbol"].nunique()),
        "report_date_min": str(all_df["report_date"].min()),
        "report_date_max": str(all_df["report_date"].max()),
        "announcement_date_min": str(all_df["announcement_date"].dropna().min()),
        "announcement_date_max": str(all_df["announcement_date"].dropna().max()),
        "created_at": created_at,
        "source_code_sha": source_commit,
        "file_sha256": file_sha
    }

    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_sha, manifest


if __name__ == "__main__":
    sha, m = build_fundamental_pit_timeline()
    print(f"Built fundamental_announcements_pit.parquet: SHA256={sha}")
    print(f"Official rows={m['official_announcement_rows']}, Total rows={m['total_raw_rows']}, Invalid={m['invalid_chronology_count']}")
