"""
A股行情数据获取模块
支持 AKShare 线上拉取（分别获取未复权原始价格与前复权价格）、本地 CSV 导入，
并在显式开启 ALLOW_SYNTHETIC_DATA 时支持数据仿真回退，准确统计实际数据来源分布
"""
import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

try:
    import akshare as ak
except ImportError:
    ak = None

from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """行情数据拉取异常类"""
    pass


class DataFetcher:
    """数据拉取与格式统一器"""

    def __init__(self, raw_dir: Optional[Path] = None, allow_synthetic: bool = settings.ALLOW_SYNTHETIC_DATA):
        self.raw_dir = raw_dir or settings.RAW_DATA_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.allow_synthetic = allow_synthetic
        self.source_counts: Dict[str, int] = {"akshare": 0, "local_csv": 0, "synthetic": 0}
        self.last_benchmark_source: str = "unknown"
        self.last_stock_source: str = "unknown"

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        """统一代码格式，如 '600519.SH' -> '600519'"""
        return symbol.split(".")[0].strip()

    @staticmethod
    def _to_sina_symbol(symbol: str) -> str:
        """'600519.SH' -> 'sh600519', '000001.SZ' -> 'sz000001'"""
        code = symbol.split(".")[0].strip()
        if symbol.endswith(".SH"):
            return f"sh{code}"
        return f"sz{code}"

    def fetch_stock_daily(
        self,
        symbol: str,
        start_date: str = settings.START_DATE,
        end_date: Optional[str] = settings.END_DATE,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        拉取单只 A 股日频行情：
        同时获取未复权原始价格 (open, high, low, close) 与前复权价格 (adj_open, adj_high, adj_low, adj_close)
        数据源优先级: 新浪 (直连稳定) -> 东方财富 (原主源) -> 腾讯 (备选) -> 本地CSV/仿真
        """
        code = self._clean_symbol(symbol)
        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "") if end_date else time.strftime("%Y%m%d")

        df = None
        if ak is not None:
            # 1. 优先新浪 (东财历史行情接口部分网络环境被拦，新浪接口更稳定)
            try:
                sina_code = self._to_sina_symbol(symbol)
                raw_df = ak.stock_zh_a_daily(symbol=sina_code, start_date=start_str, end_date=end_str, adjust="")
                time.sleep(0.15)  # 防新浪限流
                adj_df = ak.stock_zh_a_daily(symbol=sina_code, start_date=start_str, end_date=end_str, adjust="qfq")
                if raw_df is not None and not raw_df.empty:
                    self.source_counts["akshare"] = self.source_counts.get("akshare", 0) + 1
                    self.last_stock_source = "sina"
                    df = self._standardize_sina_stock(raw_df, adj_df, symbol)
                    logger.info(f"从新浪获取股票 {symbol} 行情成功 ({len(df)} 条)")
            except Exception as e:
                logger.warning(f"从新浪获取股票 {symbol} 行情失败: {e}")

            # 2. 回退东方财富 (原主数据源)
            if df is None or df.empty:
                for attempt in range(1, max_retries + 1):
                    try:
                        logger.info(f"新浪未成功，从东方财富获取股票 {symbol} 行情 (第 {attempt} 次尝试)...")
                        raw_df = ak.stock_zh_a_hist(
                            symbol=code,
                            period="daily",
                            start_date=start_str,
                            end_date=end_str,
                            adjust=""
                        )
                        adj_df = ak.stock_zh_a_hist(
                            symbol=code,
                            period="daily",
                            start_date=start_str,
                            end_date=end_str,
                            adjust="qfq"
                        )
                        if raw_df is not None and not raw_df.empty:
                            self.source_counts["akshare"] = self.source_counts.get("akshare", 0) + 1
                            self.last_stock_source = "eastmoney"
                            df = self._standardize_and_merge_stock(raw_df, adj_df, symbol)
                            break
                    except Exception as e:
                        logger.warning(f"从东方财富获取股票 {symbol} 行情失败 (尝试 {attempt}/{max_retries}): {e}")
                        time.sleep(0.5)

            # 3. 回退腾讯
            if df is None or df.empty:
                try:
                    tx_code = self._to_sina_symbol(symbol)
                    tx_df = ak.stock_zh_a_hist_tx(symbol=tx_code, start_date=start_str, end_date=end_str, adjust="qfq")
                    if tx_df is not None and not tx_df.empty:
                        self.source_counts["akshare"] = self.source_counts.get("akshare", 0) + 1
                        self.last_stock_source = "tencent"
                        df = self._standardize_sina_stock(tx_df, None, symbol)
                        logger.info(f"从腾讯获取股票 {symbol} 行情成功 ({len(df)} 条)")
                except Exception as e:
                    logger.warning(f"从腾讯获取股票 {symbol} 行情失败: {e}")

        # 若拉取失败或无网络
        if df is None or df.empty:
            logger.info(f"尝试从本地缓存或备用数据源获取股票 {symbol} 行情...")
            df = self._load_or_generate_fallback(symbol, start_date, end_date or time.strftime("%Y-%m-%d"))

        return df

    def fetch_benchmark_daily(
        self,
        symbol: str = settings.BENCHMARK_SYMBOL,
        start_date: str = settings.START_DATE,
        end_date: Optional[str] = settings.END_DATE,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        拉取基准指数 (如沪深300 '000300.SH' -> 'sh000300') 日频行情
        """
        code = self._clean_symbol(symbol)
        sina_code = f"sh{code}" if symbol.endswith(".SH") else f"sz{code}"
        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "") if end_date else time.strftime("%Y%m%d")

        df = None
        if ak is not None:
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"正在从 AKShare 获取基准指数 {symbol} 行情 (第 {attempt} 次尝试)...")
                    raw_df = ak.stock_zh_index_daily(symbol=sina_code)
                    if raw_df is not None and not raw_df.empty:
                        self.source_counts["akshare"] = self.source_counts.get("akshare", 0) + 1
                        self.last_benchmark_source = "akshare"
                        df = self._standardize_index(raw_df, symbol)
                        
                        # 过滤时间区间
                        df = df[(df["date"] >= pd.to_datetime(start_date))].copy()
                        if end_date:
                            df = df[(df["date"] <= pd.to_datetime(end_date))].copy()
                        break
                except Exception as e:
                    logger.warning(f"从 AKShare 获取基准指数 {symbol} 失败 (尝试 {attempt}/{max_retries}): {e}")
                    time.sleep(0.5)

        if df is None or df.empty:
            logger.info(f"尝试从本地缓存或备用数据源获取基准指数 {symbol} 行情...")
            df = self._load_or_generate_benchmark_fallback(symbol, start_date, end_date or time.strftime("%Y-%m-%d"))

        return df

    def _standardize_and_merge_stock(self, raw_df: pd.DataFrame, adj_df: Optional[pd.DataFrame], symbol: str) -> pd.DataFrame:
        """标准化并合并原始未复权价格与前复权价格"""
        raw = raw_df.copy()
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount", "振幅": "amplitude",
            "涨跌幅": "pct_change", "涨跌额": "change", "换手率": "turnover"
        }
        raw.rename(columns=col_map, inplace=True)
        raw["date"] = pd.to_datetime(raw["date"])
        raw["symbol"] = symbol

        num_cols = ["open", "high", "low", "close", "volume", "amount"]
        for c in num_cols:
            if c in raw.columns:
                raw[c] = pd.to_numeric(raw[c], errors="coerce").astype(float)

        raw["pct_change"] = pd.to_numeric(raw.get("pct_change", 0), errors="coerce") / 100.0
        raw["turnover"] = pd.to_numeric(raw.get("turnover", 0), errors="coerce") / 100.0

        if adj_df is not None and not adj_df.empty:
            adj = adj_df.copy()
            adj.rename(columns=col_map, inplace=True)
            adj["date"] = pd.to_datetime(adj["date"])
            adj_cols = {
                "open": "adj_open",
                "high": "adj_high",
                "low": "adj_low",
                "close": "adj_close",
                "pct_change": "adj_pct_change"
            }
            adj_sub = adj[["date", "open", "high", "low", "close", "pct_change"]].rename(columns=adj_cols)
            adj_sub["adj_pct_change"] = pd.to_numeric(adj_sub["adj_pct_change"], errors="coerce") / 100.0
            
            merged = pd.merge(raw, adj_sub, on="date", how="left")
            merged["adj_open"] = merged["adj_open"].fillna(merged["open"])
            merged["adj_high"] = merged["adj_high"].fillna(merged["high"])
            merged["adj_low"] = merged["adj_low"].fillna(merged["low"])
            merged["adj_close"] = merged["adj_close"].fillna(merged["close"])
            merged["adj_pct_change"] = merged["adj_pct_change"].fillna(merged["pct_change"])
        else:
            merged = raw.copy()
            merged["adj_open"] = merged["open"]
            merged["adj_high"] = merged["high"]
            merged["adj_low"] = merged["low"]
            merged["adj_close"] = merged["close"]
            merged["adj_pct_change"] = merged["pct_change"]

        merged["data_source"] = "akshare"
        merged.sort_values("date", inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged

    def _standardize_index(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """标准化指数行情"""
        df = df.copy()
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_change"
        }
        df.rename(columns=col_map, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

        df["pct_change"] = pd.to_numeric(df.get("pct_change", 0), errors="coerce") / 100.0
        df["data_source"] = "akshare"
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _standardize_sina_stock(self, raw_df: pd.DataFrame, adj_df: Optional[pd.DataFrame], symbol: str) -> pd.DataFrame:
        """标准化新浪 stock_zh_a_daily 数据 (英文列名, volume 单位为股, 无 pct_change 列)"""
        def _proc(df: pd.DataFrame) -> pd.DataFrame:
            d = df.copy()
            d["date"] = pd.to_datetime(d["date"])
            for c in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
            # 新浪成交量为"股"，统一转为"手" (100股=1手)，与东方财富口径一致
            if "volume" in d.columns:
                d["volume"] = d["volume"] / 100.0
            # 新浪无涨跌幅列，用收盘价计算
            d["pct_change"] = d["close"].pct_change()
            # 新浪换手率已是小数形式 (0.002678 = 0.2678%)，保持原样
            if "turnover" not in d.columns:
                d["turnover"] = 0.0
            return d

        raw = _proc(raw_df.copy())
        raw["symbol"] = symbol

        if adj_df is not None and not adj_df.empty:
            adj = _proc(adj_df.copy())
            adj_sub = adj[["date", "open", "high", "low", "close"]].rename(columns={
                "open": "adj_open", "high": "adj_high", "low": "adj_low", "close": "adj_close"
            })
            merged = pd.merge(raw, adj_sub, on="date", how="left")
            merged["adj_open"] = merged["adj_open"].fillna(merged["open"])
            merged["adj_high"] = merged["adj_high"].fillna(merged["high"])
            merged["adj_low"] = merged["adj_low"].fillna(merged["low"])
            merged["adj_close"] = merged["adj_close"].fillna(merged["close"])
            merged["adj_pct_change"] = merged["adj_close"].pct_change().fillna(merged["pct_change"])
        else:
            merged = raw.copy()
            merged["adj_open"] = merged["open"]
            merged["adj_high"] = merged["high"]
            merged["adj_low"] = merged["low"]
            merged["adj_close"] = merged["close"]
            merged["adj_pct_change"] = merged["pct_change"]

        merged["data_source"] = "akshare"
        merged.sort_values("date", inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged

    def _load_or_generate_fallback(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从本地 CSV 加载或在允许时生成仿真数据"""
        local_csv = self.raw_dir / f"{symbol.replace('.', '_')}.csv"
        if local_csv.exists():
            try:
                df = pd.read_csv(local_csv)
                raw_df = df.copy()
                self.source_counts["local_csv"] = self.source_counts.get("local_csv", 0) + 1
                res = self._standardize_and_merge_stock(raw_df, None, symbol)
                res["data_source"] = "local_csv"
                return res
            except Exception as e:
                logger.warning(f"读取本地 CSV {local_csv} 失败: {e}")

        # 检查是否允许使用模拟仿真数据
        if not self.allow_synthetic:
            raise DataFetchError(
                f"无法获取股票 {symbol} 行情数据，且系统已禁用模拟仿真数据 (settings.ALLOW_SYNTHETIC_DATA=False)。"
                f"请检查网络或在 {self.raw_dir} 提供对应 CSV 文件。"
            )

        logger.warning(f"⚠️ WARNING: USING SYNTHETIC DATA for stock {symbol} (仅供离线测试使用)！")
        self.source_counts["synthetic"] = self.source_counts.get("synthetic", 0) + 1
        return self._generate_synthetic_stock(symbol, start_date, end_date)

    def _load_or_generate_benchmark_fallback(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从本地 CSV 加载或在允许时生成仿真基准数据"""
        local_csv = self.raw_dir / f"{symbol.replace('.', '_')}.csv"
        if local_csv.exists():
            try:
                df = pd.read_csv(local_csv)
                self.source_counts["local_csv"] = self.source_counts.get("local_csv", 0) + 1
                res = self._standardize_index(df, symbol)
                res["data_source"] = "local_csv"
                return res
            except Exception:
                pass

        if not self.allow_synthetic:
            raise DataFetchError(
                f"无法获取基准指数 {symbol} 行情数据，且系统已禁用模拟仿真数据 (settings.ALLOW_SYNTHETIC_DATA=False)。"
            )

        logger.warning(f"⚠️ WARNING: USING SYNTHETIC DATA for benchmark {symbol} (仅供离线测试使用)！")
        self.source_counts["synthetic"] = self.source_counts.get("synthetic", 0) + 1
        return self._generate_synthetic_benchmark(symbol, start_date, end_date)

    def _generate_synthetic_stock(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成高仿真 A 股真实量价序列"""
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        n = len(dates)
        
        np.random.seed(abs(hash(symbol)) % (2**32))
        mu = 0.0003
        sigma = 0.02
        returns = np.random.normal(mu, sigma, n)
        limit = settings.PRICE_LIMIT_CHINEXT if symbol.startswith("300") or symbol.startswith("688") else settings.PRICE_LIMIT_MAIN
        returns = np.clip(returns, -limit, limit)
        
        price_base = 50.0 + (abs(hash(symbol)) % 150)
        close_prices = price_base * np.exp(np.cumsum(returns))
        high_prices = np.maximum(close_prices * (1 + np.abs(np.random.normal(0, 0.01, n))), close_prices)
        low_prices = np.minimum(close_prices * (1 - np.abs(np.random.normal(0, 0.01, n))), close_prices)
        open_prices = close_prices * (1 + np.random.normal(0, 0.005, n))
        open_prices = np.clip(open_prices, low_prices, high_prices)
        
        volumes = np.random.lognormal(mean=14, sigma=0.8, size=n)
        amounts = volumes * close_prices
        turnovers = np.random.uniform(0.008, 0.06, size=n)
        
        df = pd.DataFrame({
            "date": dates,
            "symbol": symbol,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "adj_open": open_prices,
            "adj_high": high_prices,
            "adj_low": low_prices,
            "adj_close": close_prices,
            "volume": volumes,
            "amount": amounts,
            "pct_change": returns,
            "adj_pct_change": returns,
            "turnover": turnovers,
            "data_source": "synthetic"
        })
        return df

    def _generate_synthetic_benchmark(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成基准指数行情"""
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        n = len(dates)
        np.random.seed(42)
        returns = np.random.normal(0.0002, 0.012, n)
        returns = np.clip(returns, -0.08, 0.08)
        close_prices = 4000.0 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            "date": dates,
            "symbol": symbol,
            "open": close_prices * (1 + np.random.normal(0, 0.002, n)),
            "high": close_prices * (1 + np.abs(np.random.normal(0, 0.006, n))),
            "low": close_prices * (1 - np.abs(np.random.normal(0, 0.006, n))),
            "close": close_prices,
            "volume": np.random.lognormal(mean=18, sigma=0.4, size=n),
            "amount": np.random.lognormal(mean=22, sigma=0.4, size=n),
            "pct_change": returns,
            "turnover": 0.015,
            "data_source": "synthetic"
        })
        return df
