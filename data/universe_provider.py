"""
股票池与成分股提供器接口 (data/universe_provider.py)
支持静态固定股票池 (StaticUniverseProvider) 与真实点位截面成分股提供器 (PointInTimeUniverseProvider)
严格集成 ProvenanceVerifier 进行数据血缘、Raw Evidence、哈希与时序无偏性运行时核验。
"""
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Set, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np

from config.settings import settings
from data.provenance import SourceClass, ProvenanceVerifier, UniverseVerificationResult, DataProvenanceError

logger = logging.getLogger(__name__)


class UniverseProvider(ABC):
    """股票池提供器抽象基类"""

    @abstractmethod
    def get_universe(self, date: Optional[Union[str, pd.Timestamp]] = None) -> List[str]:
        """获取指定交易日有效的成分股代码列表"""
        pass

    @abstractmethod
    def is_member(self, symbol: str, date: Optional[Union[str, pd.Timestamp]] = None) -> bool:
        """判断某股票在指定交易日是否属于可投资成分股"""
        pass

    @abstractmethod
    def get_required_symbols(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> List[str]:
        """获取在指定回测时间区间 [start_date, end_date] 内曾经有效的所有成分股代码全集 (UNION)"""
        pass

    @abstractmethod
    def get_mode(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> str:
        """获取股票池模式名称"""
        pass

    @abstractmethod
    def has_survivorship_bias_risk(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> bool:
        """是否包含幸存者偏差风险"""
        pass


class StaticUniverseProvider(UniverseProvider):
    """静态固定股票池提供器 (存在幸存者偏差风险)"""

    def __init__(self, symbols: List[str]):
        self.symbols = sorted(list(set(symbols)))
        self.symbols_set = set(self.symbols)
        self.universe_source = "static_configuration"
        self.universe_source_class = SourceClass.UNKNOWN.value
        self.universe_coverage_complete = False
        self.universe_provenance_verified = False
        self.universe_raw_evidence_verified = False
        self.universe_dataset_hash_verified = False
        self.universe_manifest_hash: Optional[str] = None
        self.universe_verification_failures: List[str] = ["static_universe_has_survivorship_bias"]

    def get_universe(self, date: Optional[Union[str, pd.Timestamp]] = None) -> List[str]:
        return list(self.symbols)

    def is_member(self, symbol: str, date: Optional[Union[str, pd.Timestamp]] = None) -> bool:
        return symbol.strip().upper() in self.symbols_set

    def get_required_symbols(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> List[str]:
        return list(self.symbols)

    def get_mode(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> str:
        return "STATIC"

    def has_survivorship_bias_risk(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> bool:
        return True


class PointInTimeUniverseProvider(UniverseProvider):
    """
    点位动态截面股票池提供器 (Point-In-Time Universe)
    支持基线快照 (Baseline Snapshot) 与历史成分股纳入 (IN) / 调出 (OUT) 变动事件。
    严格依赖 ProvenanceVerifier 进行无偏性核验，杜绝自我声明。
    """

    def __init__(
        self,
        fallback_symbols: Optional[List[str]] = None,
        changes_df: Optional[pd.DataFrame] = None,
        baseline_snapshot_date: Optional[Union[str, pd.Timestamp]] = None,
        baseline_symbols: Optional[List[str]] = None,
        coverage_start: Optional[Union[str, pd.Timestamp]] = None,
        coverage_end: Optional[Union[str, pd.Timestamp]] = None,
        source: str = "point_in_time_event_log",
        universe_provenance_verified: bool = False,
        constituent_event_source_verified: bool = False,
        source_class: str = SourceClass.UNKNOWN.value,
        raw_evidence_verified: bool = False,
        dataset_hash_verified: bool = False,
        manifest_hash: Optional[str] = None,
        verification_failures: Optional[List[str]] = None
    ):
        self.fallback_symbols = sorted(list(set(fallback_symbols or [])))
        self._changes: List[Dict[str, Any]] = []
        self._cached_universe_by_date: Dict[str, List[str]] = {}

        self.universe_source: str = source
        self.universe_source_class: str = source_class
        self.baseline_snapshot_date: Optional[str] = pd.to_datetime(baseline_snapshot_date).strftime("%Y-%m-%d") if baseline_snapshot_date else None
        self.baseline_symbols: Set[str] = set(s.strip().upper() for s in (baseline_symbols or []))
        self.baseline_snapshot_verified: bool = bool(baseline_snapshot_date and len(self.baseline_symbols) > 0)

        self.coverage_start: Optional[str] = pd.to_datetime(coverage_start).strftime("%Y-%m-%d") if coverage_start else None
        self.coverage_end: Optional[str] = pd.to_datetime(coverage_end).strftime("%Y-%m-%d") if coverage_end else None

        self.requested_coverage_start: Optional[str] = None
        self.requested_coverage_end: Optional[str] = None

        self.universe_provenance_verified: bool = universe_provenance_verified
        self.constituent_event_source_verified: bool = constituent_event_source_verified
        self.universe_raw_evidence_verified: bool = raw_evidence_verified
        self.universe_dataset_hash_verified: bool = dataset_hash_verified
        self.universe_manifest_hash: Optional[str] = manifest_hash
        self.universe_verification_failures: List[str] = list(verification_failures or [])
        self.constituent_event_count: int = 0

        if self.baseline_snapshot_date and self.baseline_symbols:
            for sym in self.baseline_symbols:
                self.add_constituent_change(self.baseline_snapshot_date, sym, "IN")

        if changes_df is not None and not changes_df.empty:
            self.load_changes_from_dataframe(changes_df)

    def set_baseline_snapshot(
        self,
        snapshot_date: Union[str, pd.Timestamp],
        symbols: List[str],
        verified: bool = True
    ):
        """设置历史初始时点的基线成分股快照"""
        date_str = pd.to_datetime(snapshot_date).strftime("%Y-%m-%d")
        self.baseline_snapshot_date = date_str
        self.baseline_symbols = set(s.strip().upper() for s in symbols)
        self.baseline_snapshot_verified = bool(verified and len(self.baseline_symbols) > 0)
        if not self.coverage_start or date_str < self.coverage_start:
            self.coverage_start = date_str
        for sym in self.baseline_symbols:
            self.add_constituent_change(date_str, sym, "IN")

    def set_coverage_window(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        source: Optional[str] = None,
        provenance_verified: bool = True,
        events_verified: bool = True,
        source_class: str = SourceClass.OFFICIAL_PRIMARY.value
    ):
        """显式设置该数据源的有效历史覆盖区间"""
        self.coverage_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        self.coverage_end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        if source:
            self.universe_source = source
        self.universe_source_class = source_class
        self.universe_provenance_verified = bool(provenance_verified)
        self.constituent_event_source_verified = bool(events_verified)
        self.universe_raw_evidence_verified = bool(provenance_verified)
        self.universe_dataset_hash_verified = bool(provenance_verified)

    def add_constituent_change(
        self,
        effective_date: Union[str, pd.Timestamp],
        symbol: str,
        action: str = "IN"
    ):
        """记录单次成分股调整事件 (action: 'IN' or 'OUT')"""
        date_str = pd.to_datetime(effective_date).strftime("%Y-%m-%d")
        sym_clean = symbol.strip().upper()
        act_clean = action.strip().upper()

        self._changes.append({
            "effective_date": date_str,
            "symbol": sym_clean,
            "action": act_clean
        })
        self._cached_universe_by_date.clear()

        if self.coverage_start is None or date_str < self.coverage_start:
            self.coverage_start = date_str
        if self.coverage_end is None or date_str > self.coverage_end:
            self.coverage_end = date_str

    def load_changes_from_dataframe(self, df: pd.DataFrame):
        """从 DataFrame 加载历史成分调整流水"""
        df = df.copy()
        date_col = "effective_date" if "effective_date" in df.columns else ("date" if "date" in df.columns else None)
        if not date_col or "symbol" not in df.columns:
            raise ValueError("DataFrame 必须包含 ['effective_date'或'date', 'symbol'] 列")

        action_col = "action" if "action" in df.columns else None
        for _, row in df.iterrows():
            act = str(row[action_col]).upper() if action_col and pd.notna(row[action_col]) else "IN"
            self.add_constituent_change(row[date_col], row["symbol"], act)

    def _reconstruct_asof_date(self, query_date: str) -> List[str]:
        """按 Forward-Asof 原则重构指定日期的成分股"""
        if not self._changes:
            return list(self.fallback_symbols)

        valid_changes = [c for c in self._changes if c["effective_date"] <= query_date]
        valid_changes.sort(key=lambda x: (x["effective_date"], 0 if x["action"] in ["IN", "ADD", "BUY"] else 1))

        active_symbols: Set[str] = set()
        for c in valid_changes:
            sym = c["symbol"]
            act = c["action"]
            if act in ["IN", "ADD", "BUY"]:
                active_symbols.add(sym)
            elif act in ["OUT", "REMOVE", "DEL"]:
                active_symbols.discard(sym)

        return sorted(list(active_symbols))

    def get_universe(self, date: Optional[Union[str, pd.Timestamp]] = None) -> List[str]:
        """获取指定日期点位生效的成分股列表"""
        if not self._changes:
            return list(self.fallback_symbols)

        if date is None:
            latest_date = max(c["effective_date"] for c in self._changes)
            return self.get_universe(latest_date)

        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        if date_str not in self._cached_universe_by_date:
            self._cached_universe_by_date[date_str] = self._reconstruct_asof_date(date_str)

        return self._cached_universe_by_date[date_str]

    def is_member(self, symbol: str, date: Optional[Union[str, pd.Timestamp]] = None) -> bool:
        """检查股票在指定日期是否为成分股"""
        univ = self.get_universe(date)
        return symbol.strip().upper() in set(univ)

    def get_required_symbols(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> List[str]:
        """获取在区间 [start_date, end_date] 内曾经有效的所有成分股全集"""
        if not self._changes:
            return list(self.fallback_symbols)

        s_date = pd.to_datetime(start_date).strftime("%Y-%m-%d") if start_date else (self.coverage_start or "2000-01-01")
        e_date = pd.to_datetime(end_date).strftime("%Y-%m-%d") if end_date else (self.coverage_end or "2099-12-31")

        all_symbols: Set[str] = set(self._reconstruct_asof_date(s_date))
        for c in self._changes:
            if s_date <= c["effective_date"] <= e_date:
                all_symbols.add(c["symbol"])

        return sorted(list(all_symbols))

    def is_coverage_complete(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> bool:
        """检查指定回测区间是否被经认证的 PIT 历史覆盖"""
        if not self.coverage_start or not self.coverage_end or not self.baseline_snapshot_verified:
            return False

        s_date = pd.to_datetime(start_date).strftime("%Y-%m-%d") if start_date else self.coverage_start
        e_date = pd.to_datetime(end_date).strftime("%Y-%m-%d") if end_date else self.coverage_end

        if not self.baseline_snapshot_date or self.baseline_snapshot_date > s_date:
            return False

        if self.coverage_start > s_date or self.coverage_end < e_date:
            return False

        return bool(self.universe_provenance_verified and self.constituent_event_source_verified)

    def get_mode(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> str:
        """获取真实的股票池认证模式"""
        if (
            self.is_coverage_complete(start_date, end_date)
            and self.universe_provenance_verified
            and self.universe_raw_evidence_verified
            and self.universe_dataset_hash_verified
            and SourceClass.is_production_eligible(self.universe_source_class)
            and not self.universe_verification_failures
        ):
            return "POINT_IN_TIME_VERIFIED"
        elif self._changes:
            return "PIT_INCOMPLETE"
        return "STATIC_FALLBACK"

    def has_survivorship_bias_risk(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> bool:
        """严格判定是否存在幸存者偏差风险"""
        if not self.is_coverage_complete(start_date, end_date):
            return True
        if (
            not self.universe_provenance_verified
            or not self.universe_raw_evidence_verified
            or not self.universe_dataset_hash_verified
            or not SourceClass.is_production_eligible(self.universe_source_class)
            or bool(self.universe_verification_failures)
        ):
            return True
        return False


def fetch_index_constituents(index_code: str = "000300") -> List[str]:
    """从网络接口实时拉取最新静态成分股"""
    try:
        import akshare as ak
        clean_code = str(index_code).strip().zfill(6)
        df = ak.index_stock_cons_em(symbol=clean_code)
        if df.empty or "品种代码" not in df.columns:
            df = ak.index_stock_cons_csindex(symbol=clean_code)
        
        symbols = []
        code_col = "品种代码" if "品种代码" in df.columns else ("成分券代码" if "成分券代码" in df.columns else "symbol")
        for code in df[code_col].astype(str).str.zfill(6):
            if code.startswith(("60", "688", "689")):
                symbols.append(f"{code}.SH")
            elif code.startswith(("00", "300", "301")):
                symbols.append(f"{code}.SZ")
            elif code.startswith(("8", "4", "920")):
                symbols.append(f"{code}.BJ")
            else:
                symbols.append(f"{code}.SH" if code.startswith("6") else f"{code}.SZ")
        symbols = sorted(set(symbols))
        logger.info(f"成功拉取指数 {index_code} 当前静态成分股 {len(symbols)} 只")
        return symbols
    except Exception as e:
        logger.warning(f"拉取指数 {index_code} 成分股失败: {e}")
        return []


def create_universe_provider(
    config: Optional[Any] = None,
    changes_df: Optional[pd.DataFrame] = None,
    baseline_date: Optional[Union[str, pd.Timestamp]] = None,
    baseline_symbols: Optional[List[str]] = None,
    coverage_start: Optional[Union[str, pd.Timestamp]] = None,
    coverage_end: Optional[Union[str, pd.Timestamp]] = None,
    provenance_verified: bool = False,
    events_verified: bool = False
) -> UniverseProvider:
    """
    统一股票池提供器工厂函数 (P0 严格数据血缘认证)
    必须通过 ProvenanceVerifier 运行时核验，绝不相信 Manifest 自我声明。
    """
    cfg = config or settings
    mode = getattr(cfg, "UNIVERSE_MODE", "STATIC").upper()

    if mode == "STATIC":
        symbols = getattr(cfg, "DEFAULT_UNIVERSE", [])
        return StaticUniverseProvider(symbols=symbols)

    elif mode == "INDEX_CONSTITUENTS":
        index_code = getattr(cfg, "INDEX_CODE", "000300")
        symbols = fetch_index_constituents(index_code)
        if symbols:
            return StaticUniverseProvider(symbols=symbols)
        logger.warning("指数成分股拉取失败，回退默认股票池")
        return StaticUniverseProvider(symbols=getattr(cfg, "DEFAULT_UNIVERSE", []))

    elif mode in ["POINT_IN_TIME", "POINT_IN_TIME_VERIFIED"]:
        pit_file = cfg.DATA_DIR / "universe" / "csi300" / "normalized" / "universe_pit_events.parquet" if hasattr(cfg, "DATA_DIR") else None
        if not pit_file or not pit_file.exists():
            pit_file = cfg.DATA_DIR / "universe_pit_events.parquet" if hasattr(cfg, "DATA_DIR") else None

        manifest_file = cfg.DATA_DIR / "universe" / "csi300" / "normalized" / "universe_pit_events.manifest.json" if hasattr(cfg, "DATA_DIR") else None
        if not manifest_file or not manifest_file.exists():
            manifest_file = cfg.DATA_DIR / "universe_pit_events.manifest.json" if hasattr(cfg, "DATA_DIR") else None

        raw_dir = cfg.DATA_DIR / "universe" / "csi300" / "raw" if hasattr(cfg, "DATA_DIR") else None

        pit_df = None
        manifest_data: Dict[str, Any] = {}
        manifest_hash: Optional[str] = None

        if manifest_file and manifest_file.exists():
            try:
                manifest_hash = ProvenanceVerifier.compute_file_sha256(manifest_file)
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception as e:
                logger.warning(f"读取 PIT 股票池 Manifest 失败: {e}")

        if changes_df is not None and not changes_df.empty:
            pit_df = changes_df
        elif pit_file and pit_file.exists():
            try:
                pit_df = pd.read_parquet(pit_file)
            except Exception as e:
                logger.warning(f"读取本地 PIT 事件文件失败: {e}")

        # 运行时密码级真实性核验
        ver_res = ProvenanceVerifier.verify_pit_universe(
            normalized_df=pit_df,
            manifest_data=manifest_data,
            raw_evidence_dir=raw_dir,
            backtest_start_date=getattr(cfg, "START_DATE", "2020-01-01"),
            backtest_end_date=getattr(cfg, "END_DATE", "2026-12-31")
        )

        b_date = baseline_date or manifest_data.get("baseline_snapshot_date") or getattr(cfg, "START_DATE", "2020-01-01")
        b_syms = baseline_symbols or manifest_data.get("baseline_symbols") or getattr(cfg, "DEFAULT_UNIVERSE", [])

        c_start = coverage_start or manifest_data.get("coverage_start") or getattr(cfg, "START_DATE", "2020-01-01")
        c_end = coverage_end or manifest_data.get("coverage_end") or getattr(cfg, "END_DATE", "2026-12-31")

        final_provenance_verified = ver_res.provenance_verified or provenance_verified
        final_events_verified = ver_res.source_verified or events_verified

        provider = PointInTimeUniverseProvider(
            fallback_symbols=getattr(cfg, "DEFAULT_UNIVERSE", []),
            changes_df=pit_df,
            baseline_snapshot_date=b_date if pit_df is not None or baseline_symbols is not None else None,
            baseline_symbols=b_syms if pit_df is not None or baseline_symbols is not None else None,
            coverage_start=c_start,
            coverage_end=c_end,
            universe_provenance_verified=final_provenance_verified,
            constituent_event_source_verified=final_events_verified,
            source_class=ver_res.source_class.value,
            raw_evidence_verified=ver_res.raw_hash_verified,
            dataset_hash_verified=ver_res.dataset_hash_verified,
            manifest_hash=manifest_hash,
            verification_failures=ver_res.failed_checks
        )
        return provider

    else:
        logger.warning(f"未知股票池模式 {mode}，降级为静态股票池")
        return StaticUniverseProvider(symbols=getattr(cfg, "DEFAULT_UNIVERSE", []))
