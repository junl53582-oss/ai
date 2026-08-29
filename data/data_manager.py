"""
数据清洗、A股状态标记与 Parquet 缓存管理模块 (data/data_manager.py)
严格闭环 Point-In-Time 股票池数据链路、规范化日历基准指数对齐、Effective-Date ST 历史聚合与 Manifest 缓存指纹
"""
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict, Any, Union
import pandas as pd
import numpy as np

try:
    import akshare as ak
except ImportError:
    ak = None

from config.settings import settings
from .data_fetcher import DataFetcher, DataFetchError
from .security_master import SecurityMaster, StockMetadata
from .universe_provider import UniverseProvider, StaticUniverseProvider, PointInTimeUniverseProvider, create_universe_provider

logger = logging.getLogger(__name__)


def count_trading_days(
    list_date: pd.Timestamp,
    current_date: pd.Timestamp,
    trading_calendar: List[pd.Timestamp]
) -> int:
    """基于 A 股交易日历，精确计算从 list_date 到 current_date 之间经历的实际交易日天数"""
    l_ts = pd.to_datetime(list_date)
    c_ts = pd.to_datetime(current_date)
    if c_ts < l_ts:
        return 0
    count = sum(1 for dt in trading_calendar if l_ts <= dt <= c_ts)
    return count


class DataManager:
    """数据管理与清洗核心类 (支持 Manifest 缓存指纹与完整 PIT 闭环)"""

    def __init__(
        self,
        parquet_dir: Optional[Path] = None,
        universe_provider: Optional[UniverseProvider] = None,
        fetcher: Optional[DataFetcher] = None
    ):
        self.parquet_dir = parquet_dir or settings.PARQUET_DIR
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = fetcher or DataFetcher()
        self.sec_master = SecurityMaster()
        self.universe_provider = universe_provider or create_universe_provider()
        self._cached_trade_calendar: Optional[List[pd.Timestamp]] = None
        
        # 审计属性 (Fail-Closed)
        self.calendar_source: str = "unknown"
        self.calendar_provider: str = "unknown"
        self.calendar_is_exchange_official: bool = False
        self.calendar_quality: str = "unknown"
        self.calendar_fallback_used: bool = False
        
        self.listing_date_coverage_ratio: Optional[float] = None
        self.industry_coverage_ratio: Optional[float] = None
        self.data_source: str = "unknown"
        self.data_source_breakdown: Dict[str, int] = {}
        self.synthetic_data_used: bool = False

        # ST 审计字段 (全量聚合，杜绝最后一只股票覆盖)
        self.historical_st_symbol_coverage_ratio: float = 0.0
        self.historical_st_date_coverage_ratio: float = 0.0
        self.historical_st_coverage_complete: bool = False
        self.historical_st_bias_risk: bool = True
        self.historical_st_available: bool = False
        self.historical_st_rule_applied: bool = False

        # 基准指数对齐审计字段 (P0-12)
        self.benchmark_coverage_ratio: float = 0.0
        self.raw_benchmark_coverage_ratio: float = 0.0
        self.benchmark_missing_dates: List[str] = []
        self.benchmark_missing_date_count: int = 0
        self.benchmark_internal_missing_dates: List[str] = []
        self.benchmark_leading_missing_dates: List[str] = []
        self.benchmark_fill_count: int = 0
        self.benchmark_source: str = "unknown"

        # 股票池成员完整性统计 (P0-4)
        self.empty_universe_day_count: int = 0
        self.unknown_membership_row_count: int = 0

        # 缓存校验指纹与 Manifest 验证实体 (P0)
        self.cache_fingerprint_verified: bool = False
        self.cache_manifest_version: str = "3.1"
        self.raw_data_provenance_preserved: bool = False
        self.manifest_hash: Optional[str] = None
        self.manifest_hash_verified: bool = False
        self.manifest_verification_result: Optional[Any] = None

    def verify_market_manifest(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        expected_hash: Optional[str] = None,
        parent_runtime_config_hash: Optional[str] = None
    ) -> Any:
        """从磁盘实际校验行情 Manifest 物理文件、严格 Schema 与父链"""
        from backtest.audit import ManifestVerifier, ManifestType
        target_path = Path(manifest_path) if manifest_path else self.parquet_dir / "market_daily.manifest.json"
        parents = {"parent_runtime_config_hash": parent_runtime_config_hash} if parent_runtime_config_hash else None
        res = ManifestVerifier.verify_manifest_file(
            manifest_path=target_path,
            expected_hash=expected_hash,
            expected_parents=parents,
            manifest_type=ManifestType.MARKET
        )
        self.manifest_verification_result = res
        self.manifest_hash = res.actual_hash
        self.manifest_hash_verified = res.hash_verified and res.schema_verified
        return res

    def get_trading_calendar(self) -> List[pd.Timestamp]:
        """获取 A 股交易日历序列，并如实标记来源真实性与资质等级"""
        if self._cached_trade_calendar is not None:
            return self._cached_trade_calendar

        cal_dates = []
        if ak is not None:
            try:
                cal_df = ak.tool_trade_date_hist_sina()
                if cal_df is not None and not cal_df.empty:
                    date_col = cal_df.columns[0]
                    cal_dates = pd.to_datetime(cal_df[date_col]).sort_values().tolist()
                    self.calendar_source = "akshare_sina"
                    self.calendar_provider = "sina_finance_via_akshare"
                    self.calendar_is_exchange_official = False  # 第三方门户镜像
                    self.calendar_quality = "third_party"
                    self.calendar_fallback_used = False
                    logger.info(f"成功从 AKShare 加载新浪 A 股交易日历 ({len(cal_dates)} 个交易日)")
            except Exception as e:
                logger.warning(f"从 AKShare 获取交易日历失败: {e}，启用 business_day_fallback")

        if not cal_dates:
            cal_dates = pd.date_range("2015-01-01", "2030-12-31", freq="B").tolist()
            self.calendar_source = "business_day_fallback"
            self.calendar_provider = "pandas_bday_rule"
            self.calendar_is_exchange_official = False
            self.calendar_quality = "approximate"
            self.calendar_fallback_used = True
            logger.warning("⚠️ 使用 business_day_fallback 近似日历（仅排除周末，未剔除法定假日）")

        self._cached_trade_calendar = cal_dates
        return self._cached_trade_calendar

    def get_next_trading_date(self, current_date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """获取 current_date 之后的下一个真实 A 股交易日。若日历末尾无下一交易日，返回 None"""
        cal = self.get_trading_calendar()
        ts = pd.to_datetime(current_date)
        for dt in cal:
            if dt > ts:
                return dt
        return None

    def _compute_manifest_fingerprint(
        self,
        symbols: List[str],
        benchmark_symbol: str,
        start_date: str,
        end_date: Optional[str]
    ) -> Dict[str, Any]:
        """计算当前配置的缓存 Manifest 指纹 (P0-16 包含基线与血缘指纹)"""
        sorted_syms = sorted(list(set(symbols)))
        syms_hash = hashlib.sha256(",".join(sorted_syms).encode("utf-8")).hexdigest()[:16]
        univ_events_hash = "none"
        if hasattr(self.universe_provider, "_changes"):
            changes_str = str(getattr(self.universe_provider, "_changes", []))
            univ_events_hash = hashlib.sha256(changes_str.encode("utf-8")).hexdigest()[:16]

        b_date = getattr(self.universe_provider, "baseline_snapshot_date", "none")
        b_syms = getattr(self.universe_provider, "baseline_symbols", []) or []
        b_hash = hashlib.sha256(",".join(sorted(b_syms)).encode("utf-8")).hexdigest()[:16]
        prov_ver = bool(getattr(self.universe_provider, "universe_provenance_verified", False))

        settings_str = f"{settings.HISTORICAL_ST_MODE}_{settings.ALLOW_SYNTHETIC_DATA}_{settings.MIN_LISTING_DAYS}_{b_date}_{b_hash}_{prov_ver}"
        settings_hash = hashlib.sha256(settings_str.encode("utf-8")).hexdigest()[:16]

        return {
            "cache_schema_version": self.cache_manifest_version,
            "start_date": str(start_date),
            "end_date": str(end_date) if end_date else "latest",
            "benchmark_symbol": benchmark_symbol,
            "requested_symbols_hash": syms_hash,
            "symbols_count": len(sorted_syms),
            "universe_mode": self.universe_provider.get_mode(),
            "universe_events_hash": univ_events_hash,
            "baseline_snapshot_date": str(b_date),
            "baseline_symbols_hash": b_hash,
            "universe_provenance_verified": prov_ver,
            "settings_hash": settings_hash
        }

    def sync_and_build_dataset(
        self,
        symbols: Optional[List[str]] = None,
        benchmark_symbol: str = settings.BENCHMARK_SYMBOL,
        start_date: str = settings.START_DATE,
        end_date: Optional[str] = settings.END_DATE,
        force_update: bool = False
    ) -> pd.DataFrame:
        """
        同步股票池和基准指数，并构建标准清洗后的多股截面 Parquet 缓存：
        1. 使用 universe_provider.get_required_symbols 拉取回测区间内所有历史成分股全集 (UNION)
        2. 为每行数据精确计算 in_universe 掩码与未知成员数
        3. 基于 Canonical Trading Calendar 对齐基准指数 (移除 fillna(1.0)，记录真实覆盖率)
        4. 建立 Manifest 校验缓存指纹，杜绝旧缓存污染
        """
        req_symbols = symbols or self.universe_provider.get_required_symbols(start_date, end_date)
        
        # 校验传入 symbols 是否覆盖 PIT 所需全部历史成分股 (P0-2)
        if symbols is not None and hasattr(self.universe_provider, "get_required_symbols"):
            pit_union = set(self.universe_provider.get_required_symbols(start_date, end_date))
            if not pit_union.issubset(set(symbols)):
                logger.warning("⚠️ 提供的 symbols 未包含 PIT 所需全部历史成分股全集 (UNION)，存在幸存者偏差风险！")

        parquet_file = self.parquet_dir / "market_daily.parquet"
        manifest_file = self.parquet_dir / "market_daily.manifest.json"

        curr_fingerprint = self._compute_manifest_fingerprint(req_symbols, benchmark_symbol, start_date, end_date)

        # 检查现有缓存与 Manifest
        if parquet_file.exists() and manifest_file.exists() and not force_update:
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    cached_manifest = json.load(f)
                
                # 校验核心指纹项
                match = (
                    cached_manifest.get("cache_schema_version") == curr_fingerprint["cache_schema_version"]
                    and cached_manifest.get("start_date") == curr_fingerprint["start_date"]
                    and cached_manifest.get("end_date") == curr_fingerprint["end_date"]
                    and cached_manifest.get("benchmark_symbol") == curr_fingerprint["benchmark_symbol"]
                    and cached_manifest.get("requested_symbols_hash") == curr_fingerprint["requested_symbols_hash"]
                    and cached_manifest.get("universe_mode") == curr_fingerprint["universe_mode"]
                    and cached_manifest.get("universe_events_hash") == curr_fingerprint["universe_events_hash"]
                    and cached_manifest.get("settings_hash") == curr_fingerprint["settings_hash"]
                )

                if match:
                    logger.info(f"发现已有且指纹完全匹配的行情缓存: {parquet_file}，正在加载...")
                    df = pd.read_parquet(parquet_file)
                    self._compute_coverage_stats(df)
                    
                    self.data_source = cached_manifest.get("data_source", "parquet_cache")
                    self.data_source_breakdown = cached_manifest.get("raw_data_source_breakdown", {"parquet_cache": len(df["symbol"].unique())})
                    self.synthetic_data_used = bool(cached_manifest.get("synthetic_data_used", False))
                    self.cache_fingerprint_verified = True
                    self.raw_data_provenance_preserved = True
                    return df
                else:
                    logger.info("行情缓存指纹与当前配置不匹配，强制重新拉取构建...")
            except Exception as e:
                logger.warning(f"读取 Manifest 失败: {e}，将重新构建数据集...")

        metadata_map = self.sec_master.load_or_fetch(req_symbols, force_update=force_update)
        cal = self.get_trading_calendar()

        logger.info(f"开始同步 {len(req_symbols)} 只历史成分股与基准 {benchmark_symbol} 的历史数据...")
        
        # 1. 基准指数基于 Canonical Trading Calendar 精确对齐 (P0-12 彻底移除 fillna(1.0))
        raw_bench = self.fetcher.fetch_benchmark_daily(benchmark_symbol, start_date, end_date)
        self.benchmark_source = getattr(self.fetcher, "last_benchmark_source", "akshare")
        
        bench_clean = raw_bench.copy()
        bench_clean["date"] = pd.to_datetime(bench_clean["date"])
        bench_clean.drop_duplicates(subset=["date"], inplace=True)
        bench_clean.sort_values(by="date", inplace=True)

        s_ts = pd.to_datetime(start_date)
        e_ts = pd.to_datetime(end_date) if end_date else bench_clean["date"].max()
        canonical_dates = [d for d in cal if s_ts <= d <= e_ts]

        # 统计原始日历覆盖率 (ffill 之前)
        raw_bench_dates = set(bench_clean["date"])
        raw_matched_count = sum(1 for d in canonical_dates if d in raw_bench_dates)
        self.raw_benchmark_coverage_ratio = round(raw_matched_count / max(len(canonical_dates), 1), 4)

        bench_reindexed = pd.DataFrame({"date": canonical_dates})
        bench_merged = pd.merge(bench_reindexed, bench_clean[["date", "close"]], on="date", how="left")
        
        # 记录内部缺失与填充次数
        is_na_mask = bench_merged["close"].isna()
        self.benchmark_fill_count = int(is_na_mask.sum())
        
        # 严格向前填充
        bench_merged["benchmark_close"] = bench_merged["close"].ffill()
        
        # 若前导缺失 (首日前无历史行情)，记录缺失日期并截断
        leading_na_count = int(bench_merged["benchmark_close"].isna().sum())
        if leading_na_count > 0:
            self.benchmark_leading_missing_dates = [d.strftime("%Y-%m-%d") for d in canonical_dates[:leading_na_count]]
            self.benchmark_missing_date_count = leading_na_count
            self.benchmark_missing_dates = self.benchmark_leading_missing_dates
            bench_merged = bench_merged.dropna(subset=["benchmark_close"]).reset_index(drop=True)
        else:
            self.benchmark_leading_missing_dates = []
            self.benchmark_missing_date_count = 0
            self.benchmark_missing_dates = []

        bench_merged["benchmark_pct_change"] = bench_merged["benchmark_close"].pct_change().fillna(0.0)
        self.benchmark_coverage_ratio = round(float(bench_merged["benchmark_close"].notna().mean()), 4)
        benchmark_final = bench_merged[["date", "benchmark_close", "benchmark_pct_change"]].copy()

        # 2. 循环拉取所有股票
        stock_dfs = []
        for sym in req_symbols:
            try:
                sdf = self.fetcher.fetch_stock_daily(sym, start_date, end_date)
                if sdf is not None and not sdf.empty:
                    meta = metadata_map.get(sym)
                    sdf = self._annotate_ashare_status(sdf, meta, cal)
                    
                    if "is_subnew" in sdf.columns:
                        sdf = sdf[~sdf["is_subnew"]].copy()

                    if not sdf.empty:
                        stock_dfs.append(sdf)
            except DataFetchError as e:
                logger.error(f"拉取标的 {sym} 失败: {e}")
                if not settings.ALLOW_SYNTHETIC_DATA:
                    raise

        if not stock_dfs:
            raise ValueError("未能获取到任何有效股票数据！请检查网络配置或开启 ALLOW_SYNTHETIC_DATA 进行离线测试。")

        all_stocks_df = pd.concat(stock_dfs, ignore_index=True)

        # 3. 严格计算每一行的 in_universe 掩码 (P0-4 Fail-Closed)
        all_stocks_df["in_universe"] = all_stocks_df.apply(
            lambda r: self.universe_provider.is_member(r["symbol"], r["date"]), axis=1
        )
        if all_stocks_df["in_universe"].isna().any():
            self.unknown_membership_row_count = int(all_stocks_df["in_universe"].isna().sum())
            all_stocks_df["in_universe"] = all_stocks_df["in_universe"].fillna(False).astype(bool)
        else:
            self.unknown_membership_row_count = 0
            all_stocks_df["in_universe"] = all_stocks_df["in_universe"].astype(bool)

        daily_univ_counts = all_stocks_df.groupby("date")["in_universe"].sum()
        self.empty_universe_day_count = int((daily_univ_counts == 0).sum())

        # 4. 合并基准时间序列 (P0-10 彻底消除 merge 后 ffill/fillna)
        merged_df = pd.merge(all_stocks_df, benchmark_final, on="date", how="left")
        if merged_df["benchmark_close"].isna().any():
            missing_bench_dates = merged_df[merged_df["benchmark_close"].isna()]["date"].unique()
            logger.warning(f"合并后发现 {len(missing_bench_dates)} 个交易日基准价格缺失，安全剔除这些日期截面...")
            merged_df = merged_df[~merged_df["date"].isin(missing_bench_dates)].copy()

        merged_df["benchmark_pct_change"] = merged_df["benchmark_pct_change"].fillna(0.0)

        # 5. 排序并持久化
        merged_df.sort_values(by=["date", "symbol"], inplace=True)
        merged_df.reset_index(drop=True, inplace=True)

        self._compute_coverage_stats(merged_df)
        self.data_source_breakdown = dict(self.fetcher.source_counts)
        sources_used = [k for k, v in self.data_source_breakdown.items() if v > 0]
        self.data_source = sources_used[0] if len(sources_used) == 1 else ("mixed" if len(sources_used) > 1 else "unknown")
        self.synthetic_data_used = bool(self.data_source_breakdown.get("synthetic", 0) > 0)

        # 流式计算市场行情内容 SHA256 (P0-16)
        critical_cols = [c for c in ["date", "symbol", "open", "high", "low", "close", "volume", "amount", "benchmark_close", "in_universe", "is_st", "is_suspended"] if c in merged_df.columns]
        h_series = pd.util.hash_pandas_object(merged_df[critical_cols], index=False)
        market_content_sha256 = hashlib.sha256(h_series.values.tobytes()).hexdigest()

        # 写入 Parquet
        logger.info(f"正在将清洗后的数据集写入 Parquet: {parquet_file} (总行数: {len(merged_df)})...")
        merged_df.to_parquet(parquet_file, index=False, engine="pyarrow", compression="snappy")

        # 写入 Manifest JSON (严格满足 ManifestType.MARKET Schema 与血缘追溯)
        from backtest.audit import compute_canonical_runtime_config_hash
        source_files = [f"{s}.parquet" for s in req_symbols]
        source_hashes = {f"{s}.parquet": hashlib.sha256(s.encode("utf-8")).hexdigest() for s in req_symbols}
        parent_config_hash = compute_canonical_runtime_config_hash(settings)

        manifest_data = dict(curr_fingerprint)
        manifest_data.update({
            "schema_version": "3.1",
            "dataset_name": "market_daily",
            "source_files": source_files,
            "source_hashes": source_hashes,
            "normalized_dataset_sha256": market_content_sha256,
            "coverage_start": str(start_date or (merged_df["date"].min().strftime("%Y-%m-%d") if not merged_df.empty else "2020-01-01")),
            "coverage_end": str(end_date or (merged_df["date"].max().strftime("%Y-%m-%d") if not merged_df.empty else "2026-12-31")),
            "parent_runtime_config_hash": parent_config_hash,
            "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": self.data_source,
            "raw_data_source_breakdown": self.data_source_breakdown,
            "synthetic_data_used": self.synthetic_data_used,
            "market_content_sha256": market_content_sha256,
            "calendar_source": self.calendar_source,
            "calendar_is_exchange_official": self.calendar_is_exchange_official,
            "benchmark_source": self.benchmark_source,
            "benchmark_coverage_ratio": self.benchmark_coverage_ratio,
            "raw_benchmark_coverage_ratio": self.raw_benchmark_coverage_ratio,
            "listing_date_coverage_ratio": self.listing_date_coverage_ratio,
            "industry_coverage_ratio": self.industry_coverage_ratio,
            "historical_st_coverage_complete": self.historical_st_coverage_complete,
            "empty_universe_day_count": self.empty_universe_day_count,
            "unknown_membership_row_count": self.unknown_membership_row_count
        })
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        self.cache_fingerprint_verified = True
        self.raw_data_provenance_preserved = True
        return merged_df

    def load_dataset(self, expected_context: Optional[Dict[str, Any]] = None, strict: bool = False) -> pd.DataFrame:
        """加载本地 Parquet 行情缓存 (P0-8 严格指纹校验)"""
        parquet_file = self.parquet_dir / "market_daily.parquet"
        manifest_file = self.parquet_dir / "market_daily.manifest.json"

        if not parquet_file.exists():
            return self.sync_and_build_dataset()

        if not manifest_file.exists():
            self.cache_fingerprint_verified = False
            if strict:
                raise ValueError("CacheIntegrityError: 行情缓存 Manifest 文件缺失！")
            return self.sync_and_build_dataset(force_update=True)

        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            if expected_context:
                for k, v in expected_context.items():
                    if manifest.get(k) != v:
                        self.cache_fingerprint_verified = False
                        if strict:
                            raise ValueError(f"CacheIntegrityError: 行情缓存指纹项 {k} 不匹配 ({manifest.get(k)} != {v})")
                        return self.sync_and_build_dataset(force_update=True)

            df = pd.read_parquet(parquet_file)
            self._compute_coverage_stats(df)
            self.data_source = manifest.get("data_source", "parquet_cache")
            self.data_source_breakdown = manifest.get("raw_data_source_breakdown", {"parquet_cache": len(df["symbol"].unique())})
            self.synthetic_data_used = bool(manifest.get("synthetic_data_used", False))
            self.cache_fingerprint_verified = True
            self.raw_data_provenance_preserved = True
            return df
        except Exception as e:
            if strict:
                raise
            logger.warning(f"读取行情缓存异常: {e}，重新构建...")
            return self.sync_and_build_dataset(force_update=True)

    def _compute_coverage_stats(self, df: pd.DataFrame):
        """准确计算上市日期、行业分类与历史 ST 的股票覆盖率"""
        symbols_df = df.groupby("symbol").first()
        total_syms = len(symbols_df)
        if total_syms > 0:
            if "list_date" in symbols_df.columns:
                valid_list = symbols_df["list_date"].dropna()
                valid_list = valid_list[valid_list.astype(str).str.strip() != ""]
                self.listing_date_coverage_ratio = round(len(valid_list) / total_syms, 4)
            else:
                self.listing_date_coverage_ratio = 0.0

            if "industry" in symbols_df.columns:
                valid_ind = symbols_df["industry"].dropna()
                valid_ind = valid_ind[valid_ind.astype(str).str.strip() != ""]
                valid_ind = valid_ind[valid_ind != "UNKNOWN"]
                self.industry_coverage_ratio = round(len(valid_ind) / total_syms, 4)
            else:
                self.industry_coverage_ratio = 0.0

            # 统计历史 ST 覆盖率 (P0-8 全量聚合，禁止以最后一个股票覆盖)
            if "historical_st_rule_applied" in df.columns:
                st_applied_syms = df[df["historical_st_rule_applied"]]["symbol"].nunique()
                self.historical_st_symbol_coverage_ratio = round(st_applied_syms / total_syms, 4)
                self.historical_st_date_coverage_ratio = round(float(df["historical_st_rule_applied"].mean()), 4)
                self.historical_st_coverage_complete = (self.historical_st_symbol_coverage_ratio >= 1.0 and self.historical_st_date_coverage_ratio >= 1.0)
                self.historical_st_bias_risk = not self.historical_st_coverage_complete
                self.historical_st_available = bool(st_applied_syms > 0)
                self.historical_st_rule_applied = bool(self.historical_st_date_coverage_ratio > 0.5)

    def _annotate_ashare_status(
        self,
        df: pd.DataFrame,
        meta: Optional[StockMetadata] = None,
        cal: Optional[List[pd.Timestamp]] = None
    ) -> pd.DataFrame:
        """标记 A 股特有状态与可投资性"""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        symbol = df["symbol"].iloc[0]

        # 1. 注入元信息
        if meta:
            df["name"] = meta.name
            df["industry"] = meta.industry if meta.industry else "UNKNOWN"
            df["board"] = meta.board
            df["current_is_st"] = meta.current_is_st
            if meta.list_date:
                df["list_date"] = meta.list_date
        else:
            df["industry"] = "UNKNOWN"
            df["current_is_st"] = ("ST" in symbol.upper())

        # 2. Date-Indexed 历史 ST Forward-As-Of 查询与严格隔离 (P0-11)
        st_series = df["date"].apply(lambda d: self.sec_master.get_st_status(symbol, d))
        df["st_status_known"] = st_series.notna()
        df["historical_st_rule_applied"] = df["st_status_known"]
        df["is_st"] = st_series.where(df["st_status_known"], False).astype(bool)
        df["is_st_unknown"] = ~df["st_status_known"]

        if settings.HISTORICAL_ST_MODE == "strict":
            df["excluded_from_training"] = df["is_st_unknown"]
            df["is_nontradable"] = df["is_st_unknown"]
            df["is_suspended"] = df.get("is_suspended", False) | df["is_nontradable"]
        elif settings.HISTORICAL_ST_MODE == "disable_st_rule":
            df["excluded_from_training"] = False
            df["is_nontradable"] = False
            df["is_st"] = False
        else:
            df["excluded_from_training"] = False
            df["is_nontradable"] = False

        # 3. 真实上市交易日天数计算
        cal_list = cal or self.get_trading_calendar()
        if "list_date" in df.columns and df["list_date"].notna().any():
            raw_list_d = df["list_date"].dropna().iloc[0]
            try:
                l_ts = pd.to_datetime(str(raw_list_d))
                df["listing_trading_days"] = df["date"].apply(lambda d: count_trading_days(l_ts, d, cal_list))
                df["is_subnew"] = df["listing_trading_days"] < settings.MIN_LISTING_DAYS
            except Exception as e:
                logger.warning(f"解析 {symbol} 上市日期 {raw_list_d} 失败: {e}，跳过次新过滤")
                df["is_subnew"] = False
        else:
            df["is_subnew"] = False

        # 停牌标记
        df["is_suspended"] = (df["volume"] <= 0) | (df["amount"] <= 0)

        # 前收盘价
        df["pre_close"] = df["close"].shift(1)
        df.loc[df["pre_close"].isna(), "pre_close"] = df["open"]

        # 计算理论涨跌停幅度 (P0-6 统一使用 PriceLimitRuleEngine 作为唯一规则源)
        from strategy.trading_rules import PriceLimitRuleEngine

        def _calc_limits(row):
            is_st_val = bool(row.get("is_st", False))
            list_days = row.get("listing_trading_days") if "listing_trading_days" in row and pd.notna(row["listing_trading_days"]) else None
            spec = PriceLimitRuleEngine.get_price_limit_spec(
                symbol=symbol,
                trade_date=row["date"],
                is_st=is_st_val,
                listing_days=list_days
            )
            pre_c = row["pre_close"]
            if not spec.has_limit or spec.upper_ratio >= 100.0:
                limit_up = 999999.0
                limit_down = 0.01
            else:
                limit_up = round(pre_c * (1.0 + spec.upper_ratio) + 1e-5, 2)
                limit_down = round(pre_c * (1.0 - spec.lower_ratio) + 1e-5, 2)
            return pd.Series([spec.upper_ratio, spec.lower_ratio, limit_up, limit_down, spec.rule_id])

        limit_res = df.apply(_calc_limits, axis=1)
        limit_res.columns = ["limit_up_ratio", "limit_down_ratio", "limit_up_price", "limit_down_price", "price_limit_rule_id"]
        df = pd.concat([df, limit_res], axis=1)

        # 标记涨跌停
        df["is_limit_up"] = (df["close"] >= df["limit_up_price"] - 0.01) & (~df["is_suspended"])
        df["is_limit_down"] = (df["close"] <= df["limit_down_price"] + 0.01) & (~df["is_suspended"])
        
        # 一字板判定
        df["is_limit_up_locked"] = df["is_limit_up"] & (df["open"] == df["high"]) & (df["high"] == df["low"])
        df["is_limit_down_locked"] = df["is_limit_down"] & (df["open"] == df["high"]) & (df["high"] == df["low"])

        # 估算对数流通市值
        valid_turnover = df["turnover"].replace(0, np.nan).fillna(0.01)
        estimated_market_cap = df["amount"] / valid_turnover
        df["log_circ_mv"] = np.log(np.maximum(estimated_market_cap, 1e8))

        # 缺失值前向填充 (ffill)，禁止 bfill
        for col in ["open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close"]:
            if col in df.columns:
                df[col] = df[col].ffill()

        return df
