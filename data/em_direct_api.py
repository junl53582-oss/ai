"""东财直连适配器 (data/em_direct_api.py) — 绕过 akshare 解析层, 直调东财原始接口

背景 (2026-09-08 实测):
  - akshare 的 stock_news_em 因解析层损坏报 JSONDecodeError, 但东财原始接口存活
  - push2his 资金流历史接口必须走 HTTP 明文通道 (HTTPS 被网络层重置), lmt=0 返回 ~120 交易日
  - 因此所有适配器: HTTP 明文优先 + 超时重试 + 失败返回空 (Fail-Closed, 绝不编造)

接口:
  fetch_em_stock_news(symbol, num)       个股真实新闻 (搜索 API, 标题+日期+链接)
  fetch_em_fundflow_history(symbol)      个股资金流日K历史 (~120 交易日, 主力/超大单等)
"""
import json
import logging
import time
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _to_secid(symbol: str) -> str:
    code = symbol.replace(".SH", "").replace(".SZ", "")
    return ("1." if symbol.endswith(".SH") else "0.") + code


def fetch_em_stock_news(symbol: str, num: int = 3, timeout: int = 8) -> List[Dict[str, str]]:
    """东财搜索 API 个股真实新闻。失败返回 []。"""
    params = {
        "cb": "jQuery_em",
        "param": json.dumps({
            "uid": "", "keyword": symbol.replace(".SH", "").replace(".SZ", ""),
            "type": ["cmsArticleWebOld"], "client": "web", "clientType": "web",
            "clientVersion": "curr",
            "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "time",
                                           "pageIndex": 1, "pageSize": num,
                                           "preTag": "", "postTag": ""}},
        }),
    }
    for attempt in range(2):
        try:
            r = requests.get("http://search-api-web.eastmoney.com/search/jsonp",
                             params=params, timeout=timeout, headers=_HEADERS)
            txt = r.text
            payload = json.loads(txt[txt.index("(") + 1: txt.rindex(")")])
            arts = (payload.get("result") or {}).get("cmsArticleWebOld") or []
            return [{"title": (a.get("title") or "").replace("<em>", "").replace("</em>", ""),
                     "date": (a.get("date") or "")[:10],
                     "url": a.get("url", "")} for a in arts[:num] if a.get("title")]
        except Exception as e:
            logger.warning(f"[EM] 个股新闻尝试 {attempt+1} 失败 {symbol}: {e}")
            time.sleep(1)
    return []


def fetch_em_fundflow_history(symbol: str, timeout: int = 10) -> List[Dict[str, Any]]:
    """东财资金流日K历史 (~120 交易日): 日期/收盘涨跌/主力净流入等。失败返回 []。

    klines 字段序 (fields2 f51-f65): 日期,收盘价,涨跌幅,主力净流入,主力净占比,
    超大单净流入,超大单净占比,大单净流入,大单净占比,中单净流入,中单净占比,
    小单净流入,小单净占比,主力净流入-净额,主力净流入-净占比
    """
    params = {"lmt": 0, "klt": 101,
              "fields1": "f1,f2,f3,f7",
              "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
              "secid": _to_secid(symbol)}
    for attempt in range(2):
        for host in ("http://push2his.eastmoney.com", "http://push2.eastmoney.com"):
            try:
                r = requests.get(host + "/api/qt/stock/fflow/daykline/get",
                                 params=params, timeout=timeout, headers=_HEADERS)
                klines = ((r.json().get("data") or {}).get("klines")) or []
                out = []
                for k in klines:
                    p = k.split(",")
                    if len(p) >= 13:
                        out.append({"date": p[0],
                                    "main_net_inflow": float(p[1]),
                                    "small_net_inflow": float(p[2]),
                                    "mid_net_inflow": float(p[3]),
                                    "big_net_inflow": float(p[4]),
                                    "super_net_inflow": float(p[5]),
                                    "main_net_ratio": float(p[6]),
                                    "close": float(p[11]),
                                    "pct_change": float(p[12])})
                return out
            except Exception as e:
                logger.warning(f"[EM] 资金流 {host} 尝试 {attempt+1} 失败 {symbol}: {e}")
                time.sleep(0.5)
    return []


if __name__ == "__main__":
    print("新闻测试:", fetch_em_stock_news("688521.SH", num=2))
    ff = fetch_em_fundflow_history("688521.SH")
    print(f"资金流测试: {len(ff)} 条 | 最新: {ff[-1] if ff else '无'}")
