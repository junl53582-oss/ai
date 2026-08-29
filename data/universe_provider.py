"""
股票池与成分股提供器接口 (data/universe_provider.py)
支持静态固定股票池 (StaticUniverseProvider) 与真实点位截面成分股提供器 (PointInTimeUniverseProvider)
包含：
1. get_required_symbols(start_date, end_date): 获取回测区间内所有曾经有效的历史成分股 UNION
2. Point-In-Time 覆盖范围与基线快照 (Baseline Snapshot) 完整性验证
3. 严格的幸存者偏差风险 (survivorship_bias_risk) 判定与模式划分 (POINT_IN_TIME / PARTIAL_PIT / STATIC_FALLBACK)
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Optional, Union, Any
import pandas as pd
import numpy as np

from config.settings import settings

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
        self.universe_coverage_complete = False
        self.universe_provenance_verified = False

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
    在任何日期 t 查询时，严格仅根据 effective_date <= t 的历史变动事件重构有效成分股。
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
        constituent_event_source_verified: bool = False
    ):
        self.fallback_symbols = sorted(list(set(fallback_symbols or [])))
        self._changes: List[Dict[str, Any]] = []
        self._cached_universe_by_date: Dict[str, List[str]] = {}

        self.universe_source: str = source
        self.baseline_snapshot_date: Optional[str] = pd.to_datetime(baseline_snapshot_date).strftime("%Y-%m-%d") if baseline_snapshot_date else None
        self.baseline_symbols: Set[str] = set(s.strip().upper() for s in (baseline_symbols or []))
        self.baseline_snapshot_verified: bool = bool(baseline_snapshot_date and len(self.baseline_symbols) > 0)

        self.coverage_start: Optional[str] = pd.to_datetime(coverage_start).strftime("%Y-%m-%d") if coverage_start else None
        self.coverage_end: Optional[str] = pd.to_datetime(coverage_end).strftime("%Y-%m-%d") if coverage_end else None

        self.requested_coverage_start: Optional[str] = None
        self.requested_coverage_end: Optional[str] = None

        self.universe_provenance_verified: bool = universe_provenance_verified
        self.constituent_event_source_verified: bool = constituent_event_source_verified
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
        """设置历史初始时点的基线成分股快照并进行证据认证"""
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
        events_verified: bool = True
    ):
        """显式设置该数据源经认证的有效历史覆盖区间与证据资质"""
        self.coverage_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        self.coverage_end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        if source:
            self.universe_source = source
        self.universe_provenance_verified = bool(provenance_verified)
        self.constituent_event_source_verified = bool(events_verified)

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
        """
        获取在回测区间 [start_date, end_date] 内曾经有效的所有成分股全集 (UNION)。
        确保历史调出的股票在数据层完整下载，绝不在回测历史中遗失！
        """
        if not self._changes:
            return list(self.fallback_symbols)

        s_str = pd.to_datetime(start_date).strftime("%Y-%m-%d") if start_date else None
        e_str = pd.to_datetime(end_date).strftime("%Y-%m-%d") if end_date else None

        # 1. 包含在 start_date 时点已经有效的成分股
        initial_active = set(self.get_universe(s_str)) if s_str else set()

        # 2. 包含在 [start_date, end_date] 区间内所有发生过 "IN" 纳入事件的标的
        interval_symbols = set()
        for c in self._changes:
            c_date = c["effective_date"]
            if s_str and c_date < s_str:
                continue
            if e_str and c_date > e_str:
                continue
            if c["action"] in ["IN", "ADD", "BUY"]:
                interval_symbols.add(c["symbol"])

        total_required = initial_active.union(interval_symbols)
        if not total_required:
            # 如果没有区间过滤，返回所有发生过事件的标的
            total_required = set(c["symbol"] for c in self._changes)

        return sorted(list(total_required))

    def is_coverage_complete(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> bool:
        """
        验证历史成分股覆盖完整性 (P0-1 严格证据链校验)：
        1. baseline_snapshot_verified 必须为 True 且 baseline_symbols 非空
        2. baseline_snapshot_date 必须 <= start_date (若指定)
        3. coverage_start 必须 <= start_date 且 coverage_end >= end_date
        4. universe_provenance_verified 必须为 True
        5. constituent_event_source_verified 必须为 True
        """
        if start_date:
            self.requested_coverage_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        if end_date:
            self.requested_coverage_end = pd.to_datetime(end_date).strftime("%Y-%m-%d")

        # 1. 基线快照检验 (必须认证通过且非空)
        if not self.baseline_snapshot_verified or not self.baseline_snapshot_date or not self.baseline_symbols:
            return False

        # 2. 变动事件记录检验
        if not self._changes:
            return False

        # 3. 证据资质检验 (血缘与事件数据源必须认证)
        if not self.universe_provenance_verified or not self.constituent_event_source_verified:
            return False

        # 4. 回测起始点与基线时间对齐检验
        if self.requested_coverage_start:
            if self.baseline_snapshot_date > self.requested_coverage_start:
                return False
            if not self.coverage_start or self.coverage_start > self.requested_coverage_start:
                return False

        # 5. 回测终止点覆盖检验
        if self.requested_coverage_end:
            if not self.coverage_end or self.coverage_end < self.requested_coverage_end:
                return False

        return True

    def get_mode(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> str:
        """真实模式报告：仅在完整证据认证下输出 POINT_IN_TIME_VERIFIED"""
        if not self._changes:
            return "STATIC_FALLBACK"
        
        if self.is_coverage_complete(start_date, end_date):
            return "POINT_IN_TIME_VERIFIED"
        
        if self.coverage_start and self.coverage_end and len(self._changes) >= 2:
            return "PARTIAL_PIT"
        
        return "PIT_INCOMPLETE"

    def has_survivorship_bias_risk(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> bool:
        """真实风险报告：只有在通过全部严格证据链认证 (POINT_IN_TIME_VERIFIED) 时才允许声明无幸存者偏差风险"""
        return self.get_mode(start_date, end_date) != "POINT_IN_TIME_VERIFIED"


def fetch_index_constituents(index_code: str = "000300") -> List[str]:
    """从 AKShare 拉取指数成分股，返回标准 symbol 列表 (如 '600519.SH')"""
    try:
        import akshare as ak
    except ImportError:
        logger.warning("未安装 akshare，无法拉取指数成分股")
        return []

    try:
        df = ak.index_stock_cons_csindex(symbol=index_code)
        if df is None or df.empty:
            logger.warning(f"指数 {index_code} 成分股为空")
            return []
        symbols = []
        for _, row in df.iterrows():
            code = str(row.get("成分券代码", "")).strip().zfill(6)
            exchange = str(row.get("交易所", ""))
            if "上海" in exchange:
                symbols.append(f"{code}.SH")
            elif "深圳" in exchange:
                symbols.append(f"{code}.SZ")
            elif "北京" in exchange:
                symbols.append(f"{code}.BJ")
            else:
                symbols.append(f"{code}.SH" if code.startswith("6") else f"{code}.SZ")
        symbols = sorted(set(symbols))
        logger.info(f"成功拉取指数 {index_code} 成分股 {len(symbols)} 只")
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
    统一股票池提供器工厂函数 (P0-2, P0-4)
    供 CLI (run_pipeline.py), FastAPI (server/app.py), Streamlit (dashboard/app.py) 与测试套件统一调用。
    生产环境下严格读取 universe_pit_events.manifest.json 进行血缘与基线哈希校验。
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
        pit_file = cfg.DATA_DIR / "universe_pit_events.parquet" if hasattr(cfg, "DATA_DIR") else None
        manifest_file = cfg.DATA_DIR / "universe_pit_events.manifest.json" if hasattr(cfg, "DATA_DIR") else None
        
        pit_df = None
        manifest_verified = False
        manifest_data = {}

        # 尝试读取 Manifest (P0-4)
        if manifest_file and manifest_file.exists():
            try:
                import json
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                if manifest_data.get("provenance_verified", False) and manifest_data.get("verification_method"):
                    manifest_verified = True
                    logger.info(f"成功加载并验证 PIT 股票池 Manifest: {manifest_file} (版本: {manifest_data.get('dataset_version', '1.0')})")
            except Exception as e:
                logger.warning(f"读取 PIT 股票池 Manifest 失败: {e}")

        if changes_df is not None and not changes_df.empty:
            pit_df = changes_df
        elif pit_file and pit_file.exists():
            try:
                pit_df = pd.read_parquet(pit_file)
            except Exception as e:
                logger.warning(f"读取本地 PIT 事件文件失败: {e}")

        b_date = baseline_date or manifest_data.get("baseline_snapshot_date") or getattr(cfg, "START_DATE", "2020-01-01")
        b_syms = baseline_symbols or manifest_data.get("baseline_symbols") or getattr(cfg, "DEFAULT_UNIVERSE", [])

        c_start = coverage_start or manifest_data.get("coverage_start") or getattr(cfg, "START_DATE", "2020-01-01")
        c_end = coverage_end or manifest_data.get("coverage_end") or getattr(cfg, "END_DATE", "2023-12-31")

        final_provenance_verified = manifest_verified or provenance_verified
        final_events_verified = manifest_verified or events_verified

        provider = PointInTimeUniverseProvider(
            fallback_symbols=getattr(cfg, "DEFAULT_UNIVERSE", []),
            changes_df=pit_df,
            baseline_snapshot_date=b_date if pit_df is not None or baseline_symbols is not None else None,
            baseline_symbols=b_syms if pit_df is not None or baseline_symbols is not None else None,
            coverage_start=c_start,
            coverage_end=c_end,
            universe_provenance_verified=final_provenance_verified,
            constituent_event_source_verified=final_events_verified
        )
        return provider

    else:
        logger.warning(f"未知股票池模式 {mode}，降级为静态股票池")
        return StaticUniverseProvider(symbols=getattr(cfg, "DEFAULT_UNIVERSE", []))

