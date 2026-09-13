"""业绩预告适配器与 PIT 日频因子构建 (data/fundamentals_forecast.py)

为什么重要: 业绩预告的公告日 (NOTICE_DATE) 比正式财报早 1-2 个月,
用它做 PIT 锚可以把基本面信息的"时效损耗"大幅减少 (正式财报需 110 天延迟兜底,
预告可直接用真实公告日生效)。

因子:
  FF_TYPE_SCORE    预告类型方向分 (+1 预增/扭亏/略增/续盈; -1 预减/首亏/续亏/略减; 0 其他)
  FF_AMP_MID       预告净利变动幅度中值 % (clip ±500)
  FF_DAYS_SINCE    距公告日天数 (时效衰减; 超过 max_recency 置空)
"""
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_POSITIVE = {"预增", "扭亏", "略增", "续盈"}
_NEGATIVE = {"预减", "首亏", "续亏", "略减"}


def forecast_cache_dir() -> Path:
    from config.settings import settings
    p = Path(settings.DATA_DIR) / "fundamentals_forecast"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_symbol(code: str) -> Optional[str]:
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "9")):
        return f"{code}.SH"
    if code.startswith(("00", "30", "20")):
        return f"{code}.SZ"
    if code.startswith(("8", "4")):
        return f"{code}.BJ"
    return None


def fetch_forecast_report(date_str: str, page_delay: float = 2.5,
                          max_pages: int = 60) -> Optional[pd.DataFrame]:
    """拉取某报告期的业绩预告 (分页, 含公告日)。失败返回 None (Fail-Closed)。"""
    rows, page = [], 1
    while page <= max_pages:
        try:
            r = requests.get(_URL, params={
                "reportName": "RPT_PUBLIC_OP_PREDICT", "columns": "ALL",
                "filter": f"(REPORTDATE='{date_str}')",
                "pageSize": 500, "pageNumber": page,
                "sortColumns": "NOTICE_DATE", "sortTypes": "-1",
                "source": "WEB", "client": "WEB",
            }, timeout=15, headers=_HEADERS)
            res = r.json()
            data = (res.get("result") or {}).get("data") or []
            if not data:
                if page == 1:
                    logger.warning(f"[预告] {date_str} 无数据 (success={res.get('success')}, "
                                   f"msg={res.get('message')})")
                break
            rows.extend(data)
            total = int((res.get("result") or {}).get("count") or 0)
            if page * 500 >= total:
                break
            page += 1
            time.sleep(page_delay)
        except Exception as e:
            logger.warning(f"[预告] {date_str} 第 {page} 页失败: {e}")
            return None if not rows else _normalize(rows)
    if not rows:
        return None
    return _normalize(rows)


def _normalize(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "symbol": df["SECURITY_CODE"].map(_to_symbol),
        "name": df.get("SECURITY_NAME_ABBR"),
        "notice_date": pd.to_datetime(df["NOTICE_DATE"], errors="coerce").dt.normalize(),
        "report_date": pd.to_datetime(df["REPORTDATE"], errors="coerce").dt.normalize(),
        "forecast_type": df["FORECASTTYPE"].astype(str),
        "increase_l": pd.to_numeric(df.get("INCREASEL"), errors="coerce"),
        "increase_t": pd.to_numeric(df.get("INCREASET"), errors="coerce"),
        "forecast_l": pd.to_numeric(df.get("FORECASTL"), errors="coerce"),
        "forecast_t": pd.to_numeric(df.get("FORECASTT"), errors="coerce"),
        "year_earlier": pd.to_numeric(df.get("YEAREARLIER"), errors="coerce"),
    }).dropna(subset=["symbol", "notice_date"])
    return out.reset_index(drop=True)


def fetch_all_forecast_reports(start_year: str = "2018") -> int:
    """全历史业绩预告拉取 (按期缓存, 可断点续跑)"""
    y0 = int(start_year)
    d = forecast_cache_dir()
    now = pd.Timestamp.now()
    periods = []
    for y in range(y0, now.year + 1):
        for md in ("0331", "0630", "0930", "1231"):
            p = pd.Timestamp(f"{y}-{md[:2]}-{md[2:]}")
            if p <= now:
                periods.append(f"{y}{md}")
    n_new = 0
    for per in periods:
        cp = d / f"forecast_{per}.parquet"
        if cp.exists():
            continue
        start = f"{per[:4]}-{per[4:6]}-{per[6:]}"
        df = fetch_forecast_report(start)
        if df is not None and not df.empty:
            df.to_parquet(cp, index=False)
            n_new += 1
            logger.info(f"[预告] {start} 缓存 {len(df)} 条")
    return n_new


def build_forecast_daily_matrix(market_df: pd.DataFrame, max_recency_days: int = 150,
                                start_year: str = "2018") -> pd.DataFrame:
    """把各期预告快照展开为日频 PIT 因子矩阵 (key: symbol, date)。

    PIT 纪律: 因子在 NOTICE_DATE 当日盘后才可见 → 生效日 = NOTICE_DATE + 1 天。
    时效: 超过 max_recency_days 未更新则置空 (陈旧预告不再影响信号)。
    """
    d = forecast_cache_dir()
    files = sorted(d.glob("forecast_*.parquet"))
    if not files:
        return pd.DataFrame(columns=["symbol", "date", "FF_TYPE_SCORE", "FF_AMP_MID", "FF_DAYS_SINCE"])
    snaps = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    snaps = snaps.dropna(subset=["notice_date"])
    snaps["type_score"] = snaps["forecast_type"].map(
        lambda t: 1.0 if t in _POSITIVE else (-1.0 if t in _NEGATIVE else 0.0))
    snaps["amp_mid"] = ((snaps["increase_l"] + snaps["increase_t"]) / 2.0).clip(-500, 500)
    snaps["effective_date"] = snaps["notice_date"] + pd.Timedelta(days=1)
    # 同一标的同一公告日可能多条 (不同报告期): 保留公告日最新、报告期最新的
    snaps = snaps.sort_values(["symbol", "effective_date", "report_date"])
    snaps = snaps.drop_duplicates(subset=["symbol", "effective_date"], keep="last")

    cal = market_df[["symbol", "date"]].dropna().copy()
    cal["date"] = pd.to_datetime(cal["date"]).astype("datetime64[ns]")
    cal["symbol"] = cal["symbol"].astype(str)
    cal = cal[cal["date"] >= pd.Timestamp(f"{start_year}-01-01")]
    right = (snaps[["symbol", "effective_date", "notice_date", "type_score", "amp_mid"]]
             .rename(columns={"effective_date": "date"})
             .assign(date=lambda x: pd.to_datetime(x["date"]).astype("datetime64[ns]"),
                     notice_date=lambda x: pd.to_datetime(x["notice_date"]).astype("datetime64[ns]"),
                     symbol=lambda x: x["symbol"].astype(str)))

    # 逐标的 asof 合并 (规避 merge_asof by= 的全局排序校验兼容问题)
    parts = []
    right_by_sym = {s: g.sort_values("date").reset_index(drop=True)
                    for s, g in right.groupby("symbol", sort=False)}
    for sym, g in cal.groupby("symbol", sort=False):
        r = right_by_sym.get(sym)
        if r is None or r.empty:
            continue
        r = r.drop(columns=["symbol"])  # 避免合并后产生 symbol_x/_y 后缀
        parts.append(pd.merge_asof(g.sort_values("date").reset_index(drop=True), r,
                                   on="date", direction="backward"))
    if not parts:
        return pd.DataFrame(columns=["symbol", "date", "FF_TYPE_SCORE", "FF_AMP_MID", "FF_DAYS_SINCE"])
    merged = pd.concat(parts, ignore_index=True)
    merged["FF_DAYS_SINCE"] = (merged["date"] - merged["notice_date"]).dt.days
    merged = merged.rename(columns={"type_score": "FF_TYPE_SCORE", "amp_mid": "FF_AMP_MID"})
    stale = merged["FF_DAYS_SINCE"] > max_recency_days
    for c in ("FF_TYPE_SCORE", "FF_AMP_MID", "FF_DAYS_SINCE"):
        merged.loc[stale, c] = np.nan
    return merged[["symbol", "date", "FF_TYPE_SCORE", "FF_AMP_MID", "FF_DAYS_SINCE"]]
