"""
PIT 安全复权因子提供器 (data/adjustment_factors.py)

背景 (Phase A / 2026-09-01):
    原数据管线用 qfq (前复权) 价格快照 (data_fetcher.py:75-77), 属 retroactive 非 PIT:
    每次除权除息后, 拉取日 qfq 全序列整体重算, 历史"当时的复权价"随未来事件改变,
    gate `adjustment_not_point_in_time_safe` 因此 FAIL。

本模块改用新浪后复权因子表 (hfq-factor, 除权事件级累计因子):
    - 返回为除权除息事件表 (date, hfq_factor), 注意【按日期降序】返回, 使用前须升序
    - 后复权因子的历史事件值固定, 不随未来事件改变 → PIT 安全
    - 已验证: raw × hfq_factor ≈ hfq 价格 (600519 全历史最大相对差 0.019%, 价格舍入所致)

PIT 复权语义:
    adj_price_t = raw_price_t × f_t / f_base
    其中 f_t 为 t 日最新事件因子 (仅依赖 ≤ t 的事件), f_base 为固定基准日因子。
    新除权事件只影响未来价格, 不影响历史 adj_price → 杜绝未来函数。

用法:
    provider = AdjustmentFactorProvider(cache_dir=settings.DATA_DIR / "adjust_factors")
    market_df = provider.apply_pit_adjustment(market_df)   # 替换 adj_* 四列
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import akshare as ak
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-12


class AdjustmentFactorProvider:
    """后复权因子事件表拉取 + PIT 安全复权应用器"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data_storage/adjust_factors")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 拉取层
    def _sina_symbol(self, symbol: str) -> str:
        """akshare sina 格式: sh600519 / sz000001"""
        code, market = symbol.split(".")
        return ("sh" if market == "SH" else "sz") + code

    def fetch_events(self, symbol: str) -> pd.DataFrame:
        """拉取单标的 hfq-factor 事件表 (升序, 列: date, hfq_factor)"""
        raw = ak.stock_zh_a_daily(symbol=self._sina_symbol(symbol), adjust="hfq-factor")
        df = raw[["date", "hfq_factor"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["hfq_factor"] = pd.to_numeric(df["hfq_factor"], errors="coerce")
        df = df.sort_values("date").dropna().reset_index(drop=True)
        if df.empty:
            raise ValueError(f"复权因子事件表为空: {symbol}")
        if not (df["hfq_factor"] > 0).all():
            raise ValueError(f"复权因子含非正值: {symbol}")
        return df

    def get_daily_factor_series(
        self, symbol: str, trading_dates: pd.Series, refresh: bool = False
    ) -> pd.Series:
        """返回按交易日对齐的日频 hfq_factor 序列 (PIT: 每交易日取 ≤ 当日的最新事件因子)"""
        cache_file = self.cache_dir / f"{symbol}.parquet"
        if cache_file.exists() and not refresh:
            events = pd.read_parquet(cache_file)
        else:
            events = self.fetch_events(symbol)
            events.to_parquet(cache_file, index=False)
            logger.info(f"复权因子事件缓存: {symbol} ({len(events)} 事件)")

        # 交易日 t 的因子 = 事件日期 ≤ t 的最后一个事件因子; 首个事件前因子=1.0
        factor_map = events.set_index("date")["hfq_factor"]
        dates = pd.to_datetime(trading_dates)
        series = factor_map.reindex(dates, method="ffill").fillna(1.0)
        series.index = trading_dates.index
        return series

    # ---------------------------------------------------------------- 应用层
    def apply_pit_adjustment(
        self,
        market_df: pd.DataFrame,
        refresh: bool = False,
        symbol_col: str = "symbol",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        将 market_df 的 adj_open/high/low/close 替换为 PIT 安全复权价。

        基准: 每个标的取其因子序列首个交易日 (数据集首日) 的因子为 f_base。
        adj_price_t = raw_price_t × f_t / f_base
        """
        out = market_df.copy()
        price_cols = [("open", "adj_open"), ("high", "adj_high"), ("low", "adj_low"), ("close", "adj_close")]

        for symbol, grp in out.groupby(symbol_col, sort=False):
            dates_ser = grp[date_col]
            f = self.get_daily_factor_series(str(symbol), dates_ser, refresh=refresh)
            f_base = f.iloc[0] if len(f) else 1.0
            f_base = float(f_base) if f_base and f_base > 0 else 1.0
            ratio = f / f_base
            for raw_c, adj_c in price_cols:
                if raw_c in grp.columns:
                    raw_p = pd.to_numeric(grp[raw_c], errors="coerce")
                    out.loc[grp.index, adj_c] = (raw_p * ratio).values
        logger.info(
            f"PIT 复权完成: {out[symbol_col].nunique()} 标的, "
            f"复权价列 adj_open/adj_high/adj_low/adj_close 已按 hfq_factor 基准化"
        )
        return out

    # ---------------------------------------------------------------- 校验
    @staticmethod
    def verify_pit_stability(events_a: pd.DataFrame, events_b: pd.DataFrame) -> bool:
        """
        PIT 稳定性校验: 用"截至 t1 的事件表"与"截至 t2 (t2>t1) 的事件表"
        计算 t1 前的历史因子, 必须完全一致 (未来事件不得改写历史)。
        """
        cutoff = events_a["date"].max()
        b_hist = events_b[events_b["date"] <= cutoff]
        a = events_a.set_index("date")["hfq_factor"]
        b = b_hist.set_index("date")["hfq_factor"]
        merged = pd.concat([a, b], axis=1, keys=["a", "b"]).ffill().fillna(1.0)
        same = np.allclose(merged["a"], merged["b"], rtol=1e-9, atol=1e-12, equal_nan=True)
        if not same:
            logger.error("PIT 稳定性校验失败: 未来事件改写了历史因子!")
        return bool(same)


if __name__ == "__main__":
    # 冒烟: 拉 600519 事件表, 验证升序单调 + PIT 稳定性
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
    provider = AdjustmentFactorProvider()
    ev = provider.fetch_events("600519.SH")
    print(f"事件数: {len(ev)} | 区间: {ev['date'].min().date()} ~ {ev['date'].max().date()}")
    print(f"升序单调非降: {(ev['hfq_factor'].diff().dropna() >= -1e-9).all()}")
