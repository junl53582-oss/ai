"""
ST 历史时间线提供器 (data/st_timeline.py)

背景 (Phase A / 2026-09-01):
    审计判定 HISTORICAL_ST 运行时覆盖 0%。经数据源排查:
    - 深市 (SZ): ak.stock_info_sz_change_name('简称变更') 返回全深市 7469 行
      改名历史, 含【变更日期】——ST/ST 进出日期完整可建 PIT 时间线 ✓
    - 沪市 (SH): 仅新浪"更名历史"纯名单 (无日期), SSE query API sqlId 不可公开发现
      —— 沪市 ST 日期缺口【文档化保留】

本模块:
    - fetch_sz_st_periods(): 全深市 ST/ST 周期表 (证券代码, 进入日期, 退出日期, 类型)
    - get_st_periods(symbol): 单标的 ST 周期查询
    - build_universe_coverage(symbols): 覆盖统计 (供认证 gate)

用法:
    from data.st_timeline import STTimelineProvider
    provider = STTimelineProvider()
    periods = provider.get_st_periods("000711.SZ")   # -> [(start, end, 'ST'), ...]
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)


class STTimelineProvider:
    """ST 历史时间线提供器 (深市带日期, 沪市缺口文档化)"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data_storage/st_timeline")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sz_table: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------ 拉取
    def fetch_sz_name_change_table(self, refresh: bool = False) -> pd.DataFrame:
        """全深市简称变更表 (含变更日期), 缓存后只读"""
        cache = self.cache_dir / "sz_name_changes.parquet"
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)
        df = ak.stock_info_sz_change_name(symbol="简称变更")
        df["变更日期"] = pd.to_datetime(df["变更日期"], errors="coerce")
        df = df.dropna(subset=["变更日期"]).sort_values(["证券代码", "变更日期"]).reset_index(drop=True)
        df.to_parquet(cache, index=False)
        logger.info(f"深市简称变更表缓存: {len(df)} 行 / {df['证券代码'].nunique()} 标的")
        return df

    def fetch_sz_st_periods(self, refresh: bool = False) -> pd.DataFrame:
        """由简称变更表推导 ST 周期: 名称含 'ST' 视为 ST 状态, 变更日切换"""
        table = self.fetch_sz_name_change_table(refresh=refresh)
        records = []
        for code, grp in table.groupby("证券代码"):
            is_st = grp["变更后简称"].astype(str).str.contains("ST", na=False)
            # 状态变化点: 每个变更日之后的状态
            for d, st in zip(grp["变更日期"], is_st):
                records.append({"证券代码": code, "变更日期": d, "is_st": bool(st)})
        st_df = pd.DataFrame(records)
        # 折叠为周期: 状态切换点
        periods = []
        for code, grp in st_df.sort_values("变更日期").groupby("证券代码"):
            cur_state = False
            start = None
            for _, row in grp.iterrows():
                if row["is_st"] and not cur_state:
                    cur_state = True
                    start = row["变更日期"]
                elif not row["is_st"] and cur_state:
                    periods.append({"证券代码": code, "st_start": start, "st_end": row["变更日期"], "st_type": "ST"})
                    cur_state = False
                    start = None
            if cur_state:  # 仍在 ST (到数据末尾)
                periods.append({"证券代码": code, "st_start": start, "st_end": None, "st_type": "ST"})
        out = pd.DataFrame(periods)
        out.to_parquet(self.cache_dir / "sz_st_periods.parquet", index=False)
        logger.info(f"深市 ST 周期推导完成: {len(out)} 段 / {out['证券代码'].nunique() if len(out) else 0} 标的")
        return out

    # ------------------------------------------------------------ 查询
    def get_st_periods(self, symbol: str) -> List[Tuple[str, Optional[str], str]]:
        """单标的 ST 周期 [(start, end|None, 'ST'), ...]; 沪市返回空并告警"""
        code = symbol.split(".")[0].strip()
        if symbol.split(".")[1] != "SZ":
            logger.warning(f"{symbol} 为沪市: ST 日期源缺失 (见模块文档), 返回空")
            return []
        periods = self.fetch_sz_st_periods()
        sub = periods[periods["证券代码"] == code]
        return [(r["st_start"].strftime("%Y-%m-%d"), r["st_end"].strftime("%Y-%m-%d") if pd.notna(r["st_end"]) else None, r["st_type"])
                for r in sub.to_dict("records")]

    # ------------------------------------------------------------ 覆盖
    def build_universe_coverage(self, symbols: List[str]) -> dict:
        """覆盖统计: 深市带日期全覆盖, 沪市缺口"""
        periods = self.fetch_sz_st_periods()
        sz_codes = set(periods["证券代码"]) if len(periods) else set()
        total = len(symbols)
        sz_with_st = sum(1 for s in symbols if s.split(".")[1] == "SZ" and s.split(".")[0] in sz_codes)
        sz_total = sum(1 for s in symbols if s.split(".")[1] == "SZ")
        sh_total = total - sz_total
        cov = {
            "requested_symbols": total,
            "sz_covered_with_dates": sz_with_st,
            "sz_total": sz_total,
            "sh_uncovered_no_date_source": sh_total,
            "coverage_ratio": round(sz_with_st / max(total, 1), 4),
            "note": "深市 ST 周期含进入/退出日期 (变更日); 沪市仅新浪改名名单无日期, 缺口文档化",
        }
        (self.cache_dir / "st_timeline_manifest.json").write_text(
            json.dumps({**cov, "created_at": datetime.now().isoformat(timespec="seconds")},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return cov
