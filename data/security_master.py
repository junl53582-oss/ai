"""
股票主数据与元信息管理模块 (Security Master) (data/security_master.py)
管理 A 股标的元数据：名称、真实上市日期、行业分类、所属板块、当前 ST 状态与按日期生效的历史 ST 状态 (Effective-Date As-Of)
严格禁止以今日 ST 状态反推过去！
支持持久化 ST Timeline 序列，重启或缓存重载不丢失。
"""
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Union, Set
import pandas as pd
import numpy as np

try:
    import akshare as ak
except ImportError:
    ak = None

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class STStatusEvent:
    """单次 ST 戴帽 / 摘帽生效事件"""
    symbol: str
    effective_date: str # YYYY-MM-DD
    is_st: bool
    source: str = "exchange_notice"


@dataclass
class StockMetadata:
    """单只股票基础元数据"""
    symbol: str
    name: str = ""
    list_date: Optional[str] = None          # 真实上市日期 YYYY-MM-DD
    industry: Optional[str] = None           # 行业分类
    board: str = "主板"                      # 所属板块: 主板/创业板/科创板/北交所
    current_is_st: bool = False              # 当前最新时点的 ST 状态
    historical_st_available: bool = False    # 是否有可靠的逐日历史 ST 数据库
    historical_st_timeline: Dict[str, bool] = field(default_factory=dict) # 按日期点位索引的 ST 状态
    st_events: List[STStatusEvent] = field(default_factory=list)          # 按生效日排列的 ST 变更流水


# 内置静态股票元数据映射 (代码 -> 名称 / 行业 / 板块)
# 用于网络不可达、AKShare 名称/行业接口失败时的兜底，确保看板仍能显示真实名称与行业
BUILTIN_SYMBOL_METADATA: Dict[str, Dict[str, str]] = {
    "600519.SH": {"name": "贵州茅台", "industry": "食品饮料", "board": "主板"},
    "000858.SZ": {"name": "五粮液",   "industry": "食品饮料", "board": "主板"},
    "601318.SH": {"name": "中国平安", "industry": "非银金融", "board": "主板"},
    "300750.SZ": {"name": "宁德时代", "industry": "电力设备", "board": "创业板"},
    "600036.SH": {"name": "招商银行", "industry": "银行",     "board": "主板"},
    "002594.SZ": {"name": "比亚迪",   "industry": "汽车",     "board": "主板"},
    "601899.SH": {"name": "紫金矿业", "industry": "有色金属", "board": "主板"},
    "600900.SH": {"name": "长江电力", "industry": "公用事业", "board": "主板"},
    "000333.SZ": {"name": "美的集团", "industry": "家用电器", "board": "主板"},
    "600276.SH": {"name": "恒瑞医药", "industry": "医药生物", "board": "主板"},
    "601166.SH": {"name": "兴业银行", "industry": "银行",     "board": "主板"},
    "002415.SZ": {"name": "海康威视", "industry": "电子",     "board": "主板"},
    "603288.SH": {"name": "海天味业", "industry": "食品饮料", "board": "主板"},
    "600887.SH": {"name": "伊利股份", "industry": "食品饮料", "board": "主板"},
    "000001.SZ": {"name": "平安银行", "industry": "银行",     "board": "主板"},
    "300059.SZ": {"name": "东方财富", "industry": "非银金融", "board": "创业板"},
    "600030.SH": {"name": "中信证券", "industry": "非银金融", "board": "主板"},
    "601012.SH": {"name": "隆基绿能", "industry": "电力设备", "board": "主板"},
    "002475.SZ": {"name": "立讯精密", "industry": "电子",     "board": "主板"},
    "601988.SH": {"name": "中国银行", "industry": "银行",     "board": "主板"},
    "600048.SH": {"name": "保利发展", "industry": "房地产",   "board": "主板"},
    "000725.SZ": {"name": "京东方A",  "industry": "电子",     "board": "主板"},
    "601668.SH": {"name": "中国建筑", "industry": "建筑装饰", "board": "主板"},
    "600104.SH": {"name": "上汽集团", "industry": "汽车",     "board": "主板"},
    "002714.SZ": {"name": "牧原股份", "industry": "农林牧渔", "board": "主板"},
    "600309.SH": {"name": "万华化学", "industry": "基础化工", "board": "主板"},
    "300760.SZ": {"name": "迈瑞医疗", "industry": "医药生物", "board": "创业板"},
    "601288.SH": {"name": "农业银行", "industry": "银行",     "board": "主板"},
    "601398.SH": {"name": "工商银行", "industry": "银行",     "board": "主板"},
    "601857.SH": {"name": "中国石油", "industry": "石油石化", "board": "主板"},
}


# 扩展元数据映射缓存 (惰性加载: 中证官网成分股名称 + 新浪行业分类)
_extended_maps_cache: Optional[Dict[str, Dict[str, str]]] = None


def _load_extended_maps() -> Dict[str, Dict[str, str]]:
    """加载扩展元数据映射 (中证官网名称 + 新浪行业)，返回 {6位代码: {name, industry}}"""
    global _extended_maps_cache
    if _extended_maps_cache is not None:
        return _extended_maps_cache

    merged: Dict[str, Dict[str, str]] = {}
    try:
        name_path = settings.DATA_DIR / "csindex_name_map.json"
        if name_path.exists():
            with open(name_path, "r", encoding="utf-8") as f:
                name_map = json.load(f)
            for code, name in name_map.items():
                merged.setdefault(code, {})["name"] = name
    except Exception as e:
        logger.debug(f"加载中证名称映射失败: {e}")

    try:
        ind_path = settings.DATA_DIR / "sina_industry_map.json"
        if ind_path.exists():
            with open(ind_path, "r", encoding="utf-8") as f:
                ind_map = json.load(f)
            for code, ind in ind_map.items():
                merged.setdefault(code, {})["industry"] = ind
    except Exception as e:
        logger.debug(f"加载新浪行业映射失败: {e}")

    _extended_maps_cache = merged
    logger.info(f"已加载扩展元数据映射 {len(merged)} 条 (中证名称 + 新浪行业)")
    return merged


def _apply_metadata_fallback(sym: str, name: str, industry: Optional[str], board: str):
    """用内置映射 + 扩展映射兜底填充名称/行业/板块，返回 (name, industry, board)"""
    code = sym.split(".")[0]
    builtin = BUILTIN_SYMBOL_METADATA.get(sym, {})
    extended = _load_extended_maps().get(code, {})

    if not name or name.strip() == "":
        name = builtin.get("name") or extended.get("name") or name
    if not industry or str(industry).strip() in ("", "nan", "UNKNOWN", "None"):
        industry = builtin.get("industry") or extended.get("industry") or industry
    if board in ("", "主板") and builtin.get("board"):
        board = builtin["board"]
    return name, industry, board


class SecurityMaster:
    """股票主数据管理器 (支持 Effective-Date As-Of 历史 ST 状态查询与持久化)"""

    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file or (settings.DATA_DIR / "security_master.parquet")
        self.manifest_file = self.cache_file.with_suffix(".json")
        self._metadata_map: Dict[str, StockMetadata] = {}
        self._loaded: bool = False

    def load_or_fetch(self, symbols: Optional[List[str]] = None, force_update: bool = False) -> Dict[str, StockMetadata]:
        """加载或从数据源抓取股票元信息"""
        if self._loaded and not force_update:
            return self._metadata_map

        if self.cache_file.exists() and not force_update:
            try:
                df = pd.read_parquet(self.cache_file)
                st_timeline_cache = {}
                if self.manifest_file.exists():
                    try:
                        with open(self.manifest_file, "r", encoding="utf-8") as f:
                            st_timeline_cache = json.load(f).get("st_timelines", {})
                    except Exception:
                        pass

                for _, row in df.iterrows():
                    sym = row["symbol"]
                    t_dict = st_timeline_cache.get(sym, {})
                    st_evs = [
                        STStatusEvent(symbol=sym, effective_date=d, is_st=bool(st_val))
                        for d, st_val in t_dict.items()
                    ]
                    st_evs.sort(key=lambda x: x.effective_date)
                    has_hist = bool(row.get("historical_st_available", False)) or (len(st_evs) > 0)

                    _name = str(row.get("name", ""))
                    _industry = str(row["industry"]) if pd.notna(row.get("industry")) else None
                    _board = str(row.get("board", "主板"))

                    # 缓存中的空名称/行业用内置 + 扩展映射兜底
                    _name, _industry, _board = _apply_metadata_fallback(sym, _name, _industry, _board)

                    self._metadata_map[sym] = StockMetadata(
                        symbol=sym,
                        name=_name,
                        list_date=str(row["list_date"]) if pd.notna(row.get("list_date")) else None,
                        industry=_industry,
                        board=_board,
                        current_is_st=bool(row.get("current_is_st", False)),
                        historical_st_available=has_hist,
                        historical_st_timeline=t_dict,
                        st_events=st_evs
                    )
                self._loaded = True
                logger.info(f"成功从缓存加载 {len(self._metadata_map)} 只股票元信息: {self.cache_file}")
                return self._metadata_map
            except Exception as e:
                logger.warning(f"读取 SecurityMaster 缓存失败: {e}，将重新拉取")

        symbols_to_query = symbols or settings.DEFAULT_UNIVERSE
        missing_symbols = [s for s in symbols_to_query if s not in self._metadata_map]
        
        if missing_symbols:
            logger.info(f"正在获取 {len(missing_symbols)} 只股票的真实元信息 (名称/上市日/行业/ST)...")
            for sym in missing_symbols:
                clean_code = sym.split(".")[0]
                board = "主板"
                if clean_code.startswith("688"):
                    board = "科创板"
                elif clean_code.startswith("300"):
                    board = "创业板"
                elif clean_code.startswith("8") or clean_code.startswith("4"):
                    board = "北交所"

                meta = StockMetadata(
                    symbol=sym,
                    board=board,
                    current_is_st=("ST" in sym.upper()),
                    historical_st_available=False
                )

                if ak is not None:
                    try:
                        info_df = ak.stock_individual_info_em(symbol=clean_code)
                        if info_df is not None and not info_df.empty:
                            info_dict = dict(zip(info_df.iloc[:, 0], info_df.iloc[:, 1]))
                            meta.name = str(info_dict.get("股票简称", meta.name))
                            meta.list_date = str(info_dict.get("上市时间", ""))
                            meta.industry = str(info_dict.get("行业", "")) if info_dict.get("行业") else None
                            meta.current_is_st = ("ST" in meta.name.upper()) or ("ST" in sym.upper())
                    except Exception as e:
                        logger.debug(f"获取 {sym} 详细个股信息异常: {e}")

                # 网络不可达 / 接口失败时，用内置 + 扩展映射兜底，确保名称与行业不丢失
                meta.name, meta.industry, meta.board = _apply_metadata_fallback(sym, meta.name, meta.industry, meta.board)

                self._metadata_map[sym] = meta

            self._save_cache()

        self._loaded = True
        return self._metadata_map

    def _save_cache(self):
        """持久化元数据到 Parquet 与 Manifest JSON"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            records = [
                {
                    "symbol": m.symbol,
                    "name": m.name,
                    "list_date": m.list_date,
                    "industry": m.industry,
                    "board": m.board,
                    "current_is_st": m.current_is_st,
                    "historical_st_available": m.historical_st_available
                }
                for m in self._metadata_map.values()
            ]
            if records:
                df_to_save = pd.DataFrame(records)
                df_to_save.to_parquet(self.cache_file, index=False)

            # 保存 st timeline manifest
            st_timelines = {m.symbol: m.historical_st_timeline for m in self._metadata_map.values() if m.historical_st_timeline}
            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump({"st_timelines": st_timelines}, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"保存 SecurityMaster 缓存失败: {e}")

    def get_info(self, symbol: str) -> Optional[StockMetadata]:
        """获取指定标的的元信息"""
        if not self._loaded or symbol not in self._metadata_map:
            self.load_or_fetch([symbol])
        return self._metadata_map.get(symbol)

    def register_historical_st_event(self, symbol: str, effective_date: str, is_st: bool, source: str = "exchange_notice"):
        """注册单次 ST 状态生效事件 (Effective-Date As-Of)"""
        meta = self.get_info(symbol)
        if not meta:
            meta = StockMetadata(symbol=symbol, historical_st_available=True)
            self._metadata_map[symbol] = meta
        d_str = pd.to_datetime(effective_date).strftime("%Y-%m-%d")
        ev = STStatusEvent(symbol=symbol, effective_date=d_str, is_st=is_st, source=source)
        meta.st_events.append(ev)
        meta.st_events.sort(key=lambda x: x.effective_date)
        meta.historical_st_timeline[d_str] = is_st
        meta.historical_st_available = True
        self._save_cache()

    def register_historical_st_timeline(self, symbol: str, timeline: Dict[str, bool]):
        """注册股票历史点位 ST 时间线 (自动转为 Effective-Date 事件并持久化)"""
        meta = self.get_info(symbol)
        if not meta:
            meta = StockMetadata(symbol=symbol, historical_st_available=True)
            self._metadata_map[symbol] = meta
        meta.historical_st_timeline = {}
        meta.st_events = []
        for d, st_val in timeline.items():
            d_str = pd.to_datetime(d).strftime("%Y-%m-%d")
            meta.historical_st_timeline[d_str] = bool(st_val)
            meta.st_events.append(STStatusEvent(symbol=symbol, effective_date=d_str, is_st=bool(st_val)))
        meta.st_events.sort(key=lambda x: x.effective_date)
        meta.historical_st_available = True
        self._save_cache()

    def get_st_status(self, symbol: str, date: Union[str, pd.Timestamp]) -> Optional[bool]:
        """
        获取指定日期 t 股票的历史真实 ST 状态 (Forward-As-Of 查找有效事件)：
        - 查询最近一条满足 effective_date <= t 的变动事件
        - 绝不读取未来事件
        - 若该日期前无任何事件：返回 None (显式未知)，严禁使用 current_is_st 反填！
        - 若历史数据完全缺失 (not historical_st_available)：返回 None (显式未知)
        """
        meta = self.get_info(symbol)
        if not meta:
            return None

        if meta.historical_st_available:
            query_date = pd.to_datetime(date).strftime("%Y-%m-%d")
            if meta.st_events:
                past_events = [ev for ev in meta.st_events if ev.effective_date <= query_date]
                if past_events:
                    latest_event = past_events[-1]
                    return latest_event.is_st
                return None

            if meta.historical_st_timeline:
                valid_dates = [d for d in meta.historical_st_timeline.keys() if d <= query_date]
                if valid_dates:
                    latest_d = max(valid_dates)
                    return meta.historical_st_timeline[latest_d]
                return None

            return None

        return None
