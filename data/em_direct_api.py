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

# ---- 全局限速与冷却 (2026-09-08 实测: 10 分钟 500 请求触发 EM 的 IP 级封禁) ----
_EM_MIN_INTERVAL = 1.5      # 相邻两次 EM 请求的最小间隔 (秒)
_EM_COOLDOWN_START = 60     # 首次连接被重置的冷却时长 (秒)
_EM_COOLDOWN_MAX = 1800     # 冷却上限 (秒)
_last_request_ts = 0.0
_cooldown_until = 0.0
_cooldown_secs = _EM_COOLDOWN_START


def _throttle():
    """全局限速 + 冷却等待: 所有 EM 请求前必须调用"""
    global _last_request_ts, _cooldown_until
    now = time.time()
    if now < _cooldown_until:
        wait = _cooldown_until - now
        logger.warning(f"[EM] 冷却中, 等待 {wait:.0f}s (连接被重置触发)")
        time.sleep(wait)
    elapsed = time.time() - _last_request_ts
    if elapsed < _EM_MIN_INTERVAL:
        time.sleep(_EM_MIN_INTERVAL - elapsed)
    globals()["_last_request_ts"] = time.time()


def _note_reset():
    """连接被重置: 进入指数冷却"""
    global _cooldown_until, _cooldown_secs
    _cooldown_secs = min(_cooldown_secs * 2, _EM_COOLDOWN_MAX)
    _cooldown_until = time.time() + _cooldown_secs
    logger.warning(f"[EM] 连接被重置, 冷却 {_cooldown_secs}s")


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
            _throttle()
            r = requests.get("http://search-api-web.eastmoney.com/search/jsonp",
                             params=params, timeout=timeout, headers=_HEADERS)
            txt = r.text
            payload = json.loads(txt[txt.index("(") + 1: txt.rindex(")")])
            arts = (payload.get("result") or {}).get("cmsArticleWebOld") or []
            return [{"title": (a.get("title") or "").replace("<em>", "").replace("</em>", ""),
                     "date": (a.get("date") or "")[:10],
                     "url": a.get("url", "")} for a in arts[:num] if a.get("title")]
        except Exception as e:
            if "Connection aborted" in str(e) or "RemoteDisconnected" in type(e).__name__:
                _note_reset()
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
                _throttle()
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
                                    "super_net_ratio": float(p[10]),
                                    "close": float(p[11]),
                                    "pct_change": float(p[12])})
                return out
            except Exception as e:
                if "Connection aborted" in str(e) or "RemoteDisconnected" in type(e).__name__:
                    _note_reset()
                logger.warning(f"[EM] 资金流 {host} 尝试 {attempt+1} 失败 {symbol}: {e}")
                time.sleep(0.5)
    return []


if __name__ == "__main__":
    print("新闻测试:", fetch_em_stock_news("688521.SH", num=2))
    ff = fetch_em_fundflow_history("688521.SH")
    print(f"资金流测试: {len(ff)} 条 | 最新: {ff[-1] if ff else '无'}")
