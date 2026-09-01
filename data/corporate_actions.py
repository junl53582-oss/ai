"""
公司行为 (分红送转) PIT 事件提供器 (data/corporate_actions.py)

背景 (Phase A / 2026-09-01 架构审计):
    审计判定"公司行为运行时覆盖 0%" (RUNTIME_ATTESTATION: corporate_action_coverage_ratio=0.0):
    解析器在 source_registry 中存在且 Fail-Closed, 但没有任何已提交的 raw 证据与运行时数据源。
    本模块接入巨潮 (cninfo) 逐笔分红送转事件流 (ak.stock_dividend_cninfo), 在当前网络下实测可用。

数据语义 (PIT):
    - 每条记录 = 一次分红/送转方案: 公告日 / 股权登记日 / 除权除息日 / 送股比例 / 转增比例 / 派息比例
    - 事件在【公告日】即成为可感知信息; 除权日才是价格调整生效日
    - 与 PIT 复权因子事件表 (hfq-factor) 天然互补: 因子表给出价格调整的净效应,
      本表给出方案明细 (送/转/派 拆分) —— 二者应可交叉验证

用法:
    provider = CorporateActionProvider()
    events = provider.get_event_stream("600519.SH")       # 缓存后只读
    panel = provider.build_universe_panel(symbols)         # 全池事件面板 (供认证覆盖统计)
"""
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import akshare as ak
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["ex_date", "announce_date", "send_ratio", "transfer_ratio", "cash_ratio"]


class CorporateActionProvider:
    """巨潮分红送转事件提供器 (拉取 + 缓存 + PIT 校验)"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data_storage/corporate_actions")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return symbol.split(".")[0].strip()

    # ------------------------------------------------------------ 拉取层
    def fetch_events(self, symbol: str) -> pd.DataFrame:
        """拉取单标的全历史分红送转事件 (cninfo), 标准化为 REQUIRED_COLS 结构"""
        raw = ak.stock_dividend_cninfo(symbol=self._normalize_symbol(symbol))
        if raw is None or raw.empty:
            return pd.DataFrame(columns=REQUIRED_COLS + ["type", "record_date", "pay_date", "note"])

        # 列名映射 (cninfo 实测列)
        col_map = {
            "实施方案公告日期": "announce_date",
            "分红类型": "type",
            "送股比例": "send_ratio",
            "转增比例": "transfer_ratio",
            "派息比例": "cash_ratio",
            "股权登记日": "record_date",
            "除权日": "ex_date",
            "派息日": "pay_date",
            "实施方案分红说明": "note",
        }
        df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns}).copy()
        for c in ["announce_date", "record_date", "ex_date", "pay_date"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        for c in ["send_ratio", "transfer_ratio", "cash_ratio"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        df["symbol"] = symbol
        df = df.sort_values("ex_date").reset_index(drop=True) if "ex_date" in df.columns else df
        return df

    def get_event_stream(self, symbol: str, refresh: bool = False) -> pd.DataFrame:
        """缓存后读取单标的事件流 (PIT: 只读, 不修改)"""
        cache_file = self.cache_dir / f"{self._normalize_symbol(symbol)}.parquet"
        if cache_file.exists() and not refresh:
            return pd.read_parquet(cache_file)
        events = self.fetch_events(symbol)
        events.to_parquet(cache_file, index=False)
        logger.info(f"公司行为事件缓存: {symbol} ({len(events)} 笔)")
        return events

    # ------------------------------------------------------------ 校验
    @staticmethod
    def validate_events(events: pd.DataFrame) -> List[str]:
        """Fail-Closed 校验: 返回违规清单 (空=通过)"""
        violations = []
        if events.empty:
            return violations
        if "ex_date" in events.columns and events["ex_date"].notna().any():
            fut = events[events["ex_date"] > pd.Timestamp.now()]
            if not fut.empty:
                violations.append(f"{len(fut)} 条事件除权日在未来 (数据异常)")
        for c in ["send_ratio", "transfer_ratio", "cash_ratio"]:
            if c in events.columns and (events[c] < -1e-9).any():
                violations.append(f"比例列为负: {c}")
        return violations

    # ------------------------------------------------------------ 面板
    def build_universe_panel(self, symbols: List[str], refresh: bool = False) -> pd.DataFrame:
        """构建全池公司行为事件面板 (供认证覆盖统计)"""
        frames = []
        failed_symbols = []
        for sym in symbols:
            try:
                ev = self.get_event_stream(sym, refresh=refresh)
                violations = self.validate_events(ev)
                if violations:
                    logger.warning(f"{sym} 事件校验未通过: {violations}")
                if not ev.empty:
                    frames.append(ev)
            except Exception as e:
                failed_symbols.append(sym)
                logger.warning(f"{sym} 公司行为拉取失败: {e}")
        if not frames:
            return pd.DataFrame()
        panel = pd.concat(frames, ignore_index=True)
        # manifest (含未验证标的清单, 诚实记录覆盖缺口)
        manifest = {
            "dataset_name": "CORPORATE_ACTIONS_CNINFO",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "symbol_count": int(panel["symbol"].nunique()) if "symbol" in panel.columns else 0,
            "requested_symbol_count": int(len(symbols)),
            "event_count": int(len(panel)),
            "coverage_ratio": round(float(panel["symbol"].nunique()) / max(len(symbols), 1), 4),
            "coverage": float(panel["symbol"].nunique()) if "symbol" in panel.columns else 0.0,
            "unverified_symbols": failed_symbols,
            "source": "cninfo (ak.stock_dividend_cninfo)",
        }
        (self.cache_dir / "corporate_actions_manifest.json").write_text(
            __import__("json").dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return panel
