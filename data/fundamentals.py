"""
基本面财务因子数据提供器 (data/fundamentals.py)

基于 AKShare `stock_yjbb_em` (东财「业绩报表」接口，按报告期批量返回全市场数据) 构建
A 股横截面最有效的异源信号——**质量(quality)因子**与**成长(growth)因子**：
  - 质量: 净资产收益率(ROE)、销售毛利率、每股收益(EPS)、净利润规模
  - 成长: 主营业务收入同比增长、净利润同比增长

为什么用 stock_yjbb_em 而非逐股 stock_financial_analysis_indicator：
  - 前者每次调用返回「全市场」某报告期业绩 (万级行)，一次覆盖 300 只股票池只需 ~36 次调用；
  - 后者需逐股调用 (300 次)，在东财源限流/复核下批量成功率低、耗时长；
  - 且前者自带 `最新公告日期` 字段，可直接用于 Point-In-Time 披露时点对齐，杜绝未来函数。

设计要点：
  1. 季度报告期经 PIT(时点)延迟对齐 (默认延迟 110 天 ≈ 实际披露窗口)，杜绝未来函数；
     若 `最新公告日期` 合法 (落在 [报告期, 报告期+400天]) 则以其为准，更贴近真实披露时点；
  2. 按报告期本地 parquet 缓存 + 失败跳过 + 可断点续拉，规避网络不稳与重复请求；
  3. 通过 merge_asof 将低频财报向后对齐到日频行情，自然实现"财报披露后持续有效"的语义。

这些是纯量价系统 (Alpha158 46 + A股特色 13) 完全缺失的另类信息源，
学术界与业界一致认为质量/价值/成长是 A 股最 robust 的横截面溢价来源。
"""
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

try:
    import akshare as ak
except ImportError:
    ak = None

# 东财接口偶发出现不响应导致整个进程永久挂起 (本次已实测: 37 分钟无输出)。
# 设置全局 socket 默认超时, 让所有 requests 调用在 30 秒内失败, 由上层重试/跳过。
import socket
socket.setdefaulttimeout(30)

from config.settings import settings

logger = logging.getLogger(__name__)


# 报告期季度末 (月, 日)
_QUARTER_ENDS = [(3, 31), (6, 30), (9, 30), (12, 31)]


def _code_to_symbol(code: str) -> str:
    """6 位股票代码 -> 带交易所后缀的标准 symbol (与行情池一致)。"""
    code = str(code).strip()
    if code.startswith("6"):          # 沪市主板 / 科创板(688)
        return f"{code}.SH"
    if code.startswith(("0", "3")):   # 深市主板 / 创业板
        return f"{code}.SZ"
    if code.startswith(("8", "4")):   # 北交所
        return f"{code}.BJ"
    return f"{code}.SH"


# 业绩报表原始列名 -> 因子名映射 (精选 A股最 robust 的质量/成长/价值信号)
RAW_COL_TO_FACTOR: Dict[str, str] = {
    "净资产收益率": "F_ROE",
    "销售毛利率": "F_GROSS_MARGIN",
    "营业总收入-同比增长": "F_REV_GROWTH",
    "净利润-同比增长": "F_PROFIT_GROWTH",
    "每股收益": "F_EPS",
    "净利润-净利润": "F_NET_PROFIT",
    "每股净资产": "F_BPS",              # 账面价值/股 → 用于推导 F_PB 市净率 (价值因子)
    "每股经营现金流量": "F_OCF_PS",      # 经营现金流/股 (盈利质量信号)
}

# 最终进入模型的因子列 (F_PB 为派生价值因子, 在 build_daily_fundamental_matrix 中由 adj_close/F_BPS 计算)
FUNDAMENTAL_FACTOR_NAMES: List[str] = [
    "F_ROE", "F_GROSS_MARGIN", "F_REV_GROWTH",
    "F_PROFIT_GROWTH", "F_EPS", "F_NET_PROFIT",
    "F_BPS", "F_OCF_PS", "F_PB",
]


class FundamentalsProvider:
    """个股基本面财务因子提供器 (拉取 + 缓存 + 季度->日频 PIT 对齐)"""

    def __init__(self, cache_dir: Optional[Path] = None, delay_days: int = 110):
        self.cache_dir = cache_dir or (settings.DATA_DIR / "fundamentals")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay_days = delay_days
        self.source_counts: Dict[str, int] = {"akshare": 0, "cache": 0, "failed": 0}
        self.coverage_ratio: Optional[float] = None

    # ---------------- 单报告期拉取 (带缓存) ----------------
    def _cache_path(self, date_str: str) -> Path:
        return self.cache_dir / f"yjbb_{date_str}.parquet"

    def _fetch_report(self, date_str: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """拉取单报告期全市场业绩报表，优先本地缓存；失败返回 None (绝不阻塞)。"""
        cp = self._cache_path(date_str)
        if cp.exists():
            try:
                df = pd.read_parquet(cp)
                if df is not None and not df.empty:
                    self.source_counts["cache"] += 1
                    return df
            except Exception:
                pass
        if ak is None:
            self.source_counts["failed"] += 1
            return None
        for attempt in range(1, max_retries + 1):
            try:
                df = ak.stock_yjbb_em(date=date_str)
                if df is not None and not df.empty and "股票代码" in df.columns:
                    df.to_parquet(cp, index=False)
                    self.source_counts["akshare"] += 1
                    return df
                break
            except Exception as e:
                logger.warning(f"业绩报表拉取失败 {date_str} (第{attempt}次): {type(e).__name__}: {e}")
                time.sleep(1.0 * attempt)
        self.source_counts["failed"] += 1
        return None

    def fetch_all_reports(self, start_year: str = "2018", end_year: Optional[str] = None) -> int:
        """按季度末批量拉取全市场业绩报表，返回成功报告期数。自动跳过未来日期。"""
        end_year = end_year or str(pd.Timestamp.now().year)
        today = pd.Timestamp.now().normalize()
        recs = []
        total = 0
        for y in range(int(start_year), int(end_year) + 1):
            for (m, d) in _QUARTER_ENDS:
                report_date = pd.Timestamp(y, m, d)
                if report_date > today:
                    # 跳过未来季度 (报告期尚未到来)
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

    # ---------------- 季度 -> 日频 PIT 对齐 ----------------
    def build_daily_fundamental_matrix(
        self, market_df: pd.DataFrame, start_year: str = "2018"
    ) -> pd.DataFrame:
        """
        构建日频基本面因子表 (symbol, date + F_*)：
        1. 批量拉取/复用缓存的季度业绩报表；
        2. 每只股票将各报告期映射为 effective_date (PIT 延迟披露)；
        3. 用 merge_asof 向后对齐到日频行情，自然实现"财报披露后持续有效"。
        """
        n_reports = self.fetch_all_reports(start_year)
        if n_reports == 0:
            logger.warning("未获取到任何基本面数据，返回空基本面因子表")
            return pd.DataFrame(columns=["symbol", "date"] + FUNDAMENTAL_FACTOR_NAMES)

        recs = []
        for cp in sorted(self.cache_dir.glob("yjbb_*.parquet")):
            try:
                fdf = pd.read_parquet(cp)
            except Exception:
                continue
            if fdf is None or fdf.empty or "股票代码" not in fdf.columns:
                continue
            # 解析报告期: 文件名 yjbb_YYYYMMDD.parquet
            ds = cp.stem.replace("yjbb_", "")
            try:
                report_date = pd.Timestamp(
                    int(ds[:4]), int(ds[4:6]), int(ds[6:8])
                )
            except Exception:
                continue

            fdf = fdf.copy()
            fdf["symbol"] = fdf["股票代码"].map(_code_to_symbol)
            fdf = fdf.dropna(subset=["symbol"])

            # 因子映射
            row_map = {}
            for raw, fac in RAW_COL_TO_FACTOR.items():
                if raw in fdf.columns:
                    row_map[fac] = pd.to_numeric(fdf[raw], errors="coerce")
            if not row_map:
                continue

            tmp = pd.DataFrame({"symbol": fdf["symbol"].values})
            tmp["report_date"] = report_date
            for fac, ser in row_map.items():
                tmp[fac] = ser.values

            # PIT 披露时点: 优先用最新公告日期 (需合法落在 [报告期, 报告期+400天])
            if "最新公告日期" in fdf.columns:
                ann = pd.to_datetime(fdf["最新公告日期"], errors="coerce")
            else:
                ann = pd.Series(pd.NaT, index=fdf.index)
            upper = report_date + pd.Timedelta(days=400)
            eff = ann.where((ann >= report_date) & (ann <= upper), report_date + pd.Timedelta(days=self.delay_days))
            tmp["effective_date"] = eff.values

            recs.append(tmp[["symbol", "report_date", "effective_date"] + list(row_map.keys())])

        if not recs:
            logger.warning("解析后未得到任何基本面记录，返回空表")
            return pd.DataFrame(columns=["symbol", "date"] + FUNDAMENTAL_FACTOR_NAMES)

        eff = pd.concat(recs, ignore_index=True)
        # 清洗: 无穷值 -> NaN
        for c in FUNDAMENTAL_FACTOR_NAMES:
            if c in eff.columns:
                eff[c] = pd.to_numeric(eff[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

        # 同一股票同日可能多份报告 (应不会), 取最近 effective
        eff = eff.sort_values(["symbol", "effective_date"]).drop_duplicates(
            subset=["symbol", "effective_date"], keep="last"
        )

        # 日频对齐: merge_asof 按 symbol 分组, 向后取最近生效财报
        # 注意 (pandas 2.x 实测): 使用 by= 时 on 列必须【全局单调递增】,
        # 不能按 (symbol, on) 分组排序 —— 否则抛 "keys must be sorted"。
        # 经逐 symbol 手动 merge_asof 交叉验证, 全局 on 排序 + by 结果完全一致。
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

        # 派生价值因子 F_PB = 当日价格 / 最近披露每股净资产 (PIT: 价格取当日, 净资产取最新已披露)
        if "F_BPS" in merged.columns:
            px = pd.to_numeric(merged[price_col], errors="coerce")
            bps = pd.to_numeric(merged["F_BPS"], errors="coerce")
            merged["F_PB"] = np.where(
                (px > 0) & (bps > 0),
                px / bps,
                np.nan
            )
            logger.info(f"已派生价值因子 F_PB (覆盖率 {merged['F_PB'].notna().mean()*100:.1f}%)")

        # 丢弃引入的价格列 (避免回并 market_df 时与行情列产生 _x/_y 冲突)
        if price_col in merged.columns:
            merged = merged.drop(columns=[price_col])

        cov = merged[FUNDAMENTAL_FACTOR_NAMES].notna().mean().mean() * 100 if FUNDAMENTAL_FACTOR_NAMES else 0.0
        logger.info(
            f"日频基本面因子表构建完成: {len(merged)} 行, 因子 {FUNDAMENTAL_FACTOR_NAMES}, "
            f"非空覆盖率均值 {cov:.1f}%"
        )
        return merged
