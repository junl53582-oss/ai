"""
基本面财务因子数据提供器 (data/fundamentals.py)

基于 AKShare `stock_yjbb_em` (东财「业绩报表」接口，按报告期批量返回全市场数据) 构建
A 股横截面最有效的异源信号——质量(quality)因子与成长(growth)因子：
  - 质量: 净资产收益率(ROE)、销售毛利率、每股收益(EPS)、净利润规模
  - 成长: 主营业务收入同比增长、净利润同比增长

Research Integrity Hardened (Point-In-Time Fail-Closed):
  - 严格区分 OFFICIAL_ANNOUNCEMENT_DATE 与 SYNTHETIC_DELAY_ESTIMATE
  - 在 strict_pit (认证模式) 下，仅 OFFICIAL_ANNOUNCEMENT_DATE 允许作为严格 PIT 财务因子进入模型
  - 严禁通过 +110 天自动将未知公告日期变为 VERIFIED PIT
  - 完整记录数据血缘 (source, report_date, announcement_date, effective_date, effective_date_source, source_file_hash, pit_certified)
"""
from __future__ import annotations

import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

try:
    import akshare as ak
except ImportError:
    ak = None

import socket
socket.setdefaulttimeout(30)

from config.settings import settings

logger = logging.getLogger(__name__)

_QUARTER_ENDS = [(3, 31), (6, 30), (9, 30), (12, 31)]


def _code_to_symbol(code: str) -> str:
    code = str(code).strip()
    if code.startswith("6"):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    if code.startswith(("8", "4")):
        return f"{code}.BJ"
    return f"{code}.SH"


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


RAW_COL_TO_FACTOR: Dict[str, str] = {
    "净资产收益率": "F_ROE",
    "销售毛利率": "F_GROSS_MARGIN",
    "营业总收入-同比增长": "F_REV_GROWTH",
    "净利润-同比增长": "F_PROFIT_GROWTH",
    "每股收益": "F_EPS",
    "净利润-净利润": "F_NET_PROFIT",
    "每股净资产": "F_BPS",
    "每股经营现金流量": "F_OCF_PS",
}

FUNDAMENTAL_FACTOR_NAMES: List[str] = [
    "F_ROE", "F_GROSS_MARGIN", "F_REV_GROWTH",
    "F_PROFIT_GROWTH", "F_EPS", "F_NET_PROFIT",
    "F_BPS", "F_OCF_PS", "F_PB",
]


class FundamentalsProvider:
    """个股基本面财务因子提供器 (拉取 + 缓存 + 季度->日频严格 PIT 对齐与数据血缘审计)"""

    def __init__(self, cache_dir: Optional[Path] = None, delay_days: int = 110):
        self.cache_dir = cache_dir or (settings.DATA_DIR / "fundamentals")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay_days = int(delay_days)
        self.source_counts: Dict[str, int] = {"akshare": 0, "cache": 0, "failed": 0}
        self.coverage_ratio: Optional[float] = None

    def _cache_path(self, date_str: str) -> Path:
        return self.cache_dir / f"yjbb_{date_str}.parquet"

    def _fetch_report(self, date_str: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        cp = self._cache_path(date_str)
        if cp.exists():
            try:
                df = pd.read_parquet(cp)
                self.source_counts["cache"] += 1
                return df
            except Exception as e:
                logger.warning(f"读取缓存 {cp.name} 损坏: {e}, 重新拉取")

        if ak is None:
            logger.warning("未安装 akshare，无法拉取基本面数据")
            self.source_counts["failed"] += 1
            return None

        for attempt in range(1, max_retries + 1):
            try:
                time.sleep(0.3)
                df = ak.stock_yjbb_em(date=date_str)
                if df is not None and not df.empty:
                    df.to_parquet(cp, index=False)
                    self.source_counts["akshare"] += 1
                    logger.info(f"成功拉取并缓存业绩报表 {date_str} ({len(df)} 条)")
                    return df
            except Exception as e:
                logger.warning(f"拉取业绩报表 {date_str} 第 {attempt}/{max_retries} 次失败: {e}")
                time.sleep(1.0)

        self.source_counts["failed"] += 1
        return None

    def fetch_all_reports(self, start_year: str = "2018") -> int:
        today = pd.Timestamp.now()
        end_year = today.year
        recs = []
        total = 0
        for y in range(int(start_year), int(end_year) + 1):
            for (m, d) in _QUARTER_ENDS:
                report_date = pd.Timestamp(y, m, d)
                if report_date > today:
                    continue
                total += 1
                date_str = f"{y}{m:02d}{d:02d}"
                df = self._fetch_report(date_str)
                if df is not None:
                    recs.append((date_str, (y, m, d), df))
        logger.info(
            f"业绩报表拉取完成: 报告期 {total} | 网络新增 {self.source_counts['akshare']} | "
            f"缓存复用 {self.source_counts['cache']} | 失败 {self.source_counts['failed']}"
        )
        return len(recs)

    def build_daily_fundamental_matrix(
        self, market_df: pd.DataFrame, start_year: str = "2018", strict_pit: bool = False, fetch_if_empty: bool = True
    ) -> pd.DataFrame:
        cached_files = list(sorted(self.cache_dir.glob("yjbb_*.parquet")))
        if not cached_files and fetch_if_empty:
            self.fetch_all_reports(start_year)
            cached_files = list(sorted(self.cache_dir.glob("yjbb_*.parquet")))

        if not cached_files:
            logger.warning("未获取到任何基本面数据，返回空基本面因子表")
            return pd.DataFrame(columns=["symbol", "date"] + FUNDAMENTAL_FACTOR_NAMES)

        recs = []
        for cp in cached_files:
            try:
                fdf = pd.read_parquet(cp)
            except Exception:
                continue
            if fdf is None or fdf.empty or "股票代码" not in fdf.columns:
                continue

            ds = cp.stem.replace("yjbb_", "")
            try:
                report_date = pd.Timestamp(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
            except Exception:
                continue

            file_sha = _sha256_file(cp)
            fdf = fdf.copy()
            fdf["symbol"] = fdf["股票代码"].map(_code_to_symbol)
            fdf = fdf.dropna(subset=["symbol"])

            row_map = {}
            for raw, fac in RAW_COL_TO_FACTOR.items():
                if raw in fdf.columns:
                    row_map[fac] = pd.to_numeric(fdf[raw], errors="coerce")
            if not row_map:
                continue

            tmp = pd.DataFrame({"symbol": fdf["symbol"].values})
            tmp["source"] = "akshare.stock_yjbb_em"
            tmp["source_file_hash"] = file_sha
            tmp["report_date"] = report_date

            if "最新公告日期" in fdf.columns:
                ann = pd.to_datetime(fdf["最新公告日期"], errors="coerce")
            else:
                ann = pd.Series(pd.NaT, index=fdf.index)

            tmp["announcement_date"] = ann.values
            upper = report_date + pd.Timedelta(days=400)
            is_official = ann.notna() & (ann >= report_date) & (ann <= upper)

            eff_date = ann.where(is_official, report_date + pd.Timedelta(days=self.delay_days))
            eff_source = np.where(is_official, "OFFICIAL_ANNOUNCEMENT_DATE", "SYNTHETIC_DELAY_ESTIMATE")
            pit_cert = is_official.values.astype(bool)

            tmp["effective_date"] = eff_date.values
            tmp["effective_date_source"] = eff_source
            tmp["pit_certified"] = pit_cert

            for fac, ser in row_map.items():
                tmp[fac] = ser.values

            if strict_pit:
                tmp = tmp[tmp["pit_certified"]].copy()
                if tmp.empty:
                    continue

            recs.append(tmp)

        if not recs:
            logger.warning("解析后未得到任何基本面记录，返回空表")
            return pd.DataFrame(columns=["symbol", "date"] + FUNDAMENTAL_FACTOR_NAMES)

        eff = pd.concat(recs, ignore_index=True)
        for c in FUNDAMENTAL_FACTOR_NAMES:
            if c in eff.columns:
                eff[c] = pd.to_numeric(eff[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

        eff = eff.sort_values(["symbol", "effective_date"]).drop_duplicates(
            subset=["symbol", "effective_date"], keep="last"
        )

        price_col = "adj_close" if "adj_close" in market_df.columns else "close"
        daily = (
            market_df[["symbol", "date", price_col]].drop_duplicates(subset=["symbol", "date"])
            .sort_values("date")
            .reset_index(drop=True)
            .copy()
        )
        keep = ["symbol", "effective_date"] + [c for c in FUNDAMENTAL_FACTOR_NAMES if c in eff.columns]
        eff_keep = (
            eff[keep]
            .dropna(subset=["effective_date"])
            .sort_values("effective_date")
            .reset_index(drop=True)
            .copy()
        )

        merged = pd.merge_asof(
            daily,
            eff_keep,
            left_on="date",
            right_on="effective_date",
            by="symbol",
            direction="backward",
        )
        merged = merged.drop(columns=["effective_date"])

        if "F_BPS" in merged.columns:
            px = pd.to_numeric(merged[price_col], errors="coerce")
            bps = pd.to_numeric(merged["F_BPS"], errors="coerce")
            merged["F_PB"] = np.where(
                (px > 0) & (bps > 0),
                px / bps,
                np.nan
            )
            logger.info(f"已派生价值因子 F_PB (覆盖率 {merged['F_PB'].notna().mean()*100:.1f}%)")

        for c in FUNDAMENTAL_FACTOR_NAMES:
            if c not in merged.columns:
                merged[c] = np.nan

        present_factors = [c for c in FUNDAMENTAL_FACTOR_NAMES if c in merged.columns]
        cov = merged[present_factors].notna().mean().mean() * 100 if present_factors else 0.0
        logger.info(
            f"日频基本面因子表构建完成 (strict_pit={strict_pit}): {len(merged)} 行, 因子 {FUNDAMENTAL_FACTOR_NAMES}, "
            f"非空覆盖率均值 {cov:.1f}%"
        )
        return merged
