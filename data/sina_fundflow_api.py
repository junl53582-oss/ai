"""新浪个股资金流历史适配器 (data/sina_fundflow_api.py)

背景: 东财 push2* 对本机网络段封锁期间 (2026-09-09 起), 新浪 MoneyFlow 接口
全程可用且免登录。返回逐日主力净流入/超大单等字段, 分页取历史 (~120+ 日)。

字段映射 (Sina lscjfb -> 统一 schema, 与 EM 适配器对齐):
  netamount   主力净流入净额 (元)
  ratioamount 主力净流入占比 (小数, ×100 转百分比)
  r0_net      超大单净流入净额
  trade       收盘价
  changeratio 涨跌幅 (小数, ×100 转百分比)
"""
import logging
import time
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://finance.sina.com.cn/"}


def _to_sina_code(symbol: str) -> str:
    code = symbol.replace(".SH", "").replace(".SZ", "")
    return ("sh" if symbol.endswith(".SH") else "sz") + code


def fetch_sina_fundflow_history(symbol: str, min_days: int = 120,
                                max_pages: int = 8, timeout: int = 10) -> List[Dict[str, Any]]:
    """新浪个股资金流历史 (每页 60 日, 取到 min_days 或翻完)。失败返回 [] (Fail-Closed)。"""
    rows: List[Dict[str, Any]] = []
    seen = set()
    page = 1
    while page <= max_pages:
        try:
            r = requests.get(
                "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_qsfx_lscjfb",
                params={"page": page, "num": 60, "sort": "opendate", "asc": 0,
                        "daima": _to_sina_code(symbol)},
                timeout=timeout, headers=_HEADERS)
            data = r.json()
            if not data:
                break
            for d in data:
                od = d.get("opendate")
                if not od or od in seen:
                    continue
                seen.add(od)
                rows.append({
                    "date": od,
                    "main_net_inflow": float(d.get("netamount") or 0),
                    "main_net_ratio": float(d.get("ratioamount") or 0) * 100,
                    "super_net_inflow": float(d.get("r0_net") or 0),
                    "close": float(d.get("trade") or 0),
                    "pct_change": float(d.get("changeratio") or 0) * 100,
                })
            if len(rows) >= min_days or len(data) < 60:
                break
            page += 1
            time.sleep(0.8)  # 新浪礼貌限速
        except Exception as e:
            logger.warning(f"[Sina] 资金流 {symbol} 第 {page} 页失败: {e}")
            return rows if rows else []
    return sorted(rows, key=lambda r: r["date"])
