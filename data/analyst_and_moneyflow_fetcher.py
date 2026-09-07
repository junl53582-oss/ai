"""
非同质化信息源采集模块: 分析师盈利预期修正与机构主力资金流 (data/analyst_and_moneyflow_fetcher.py)
用于获取与价格/动量低相关性的核心 Alpha 源:
1. 分析师研究报告与未来盈利预期修正 (Analyst Consensus & FY2 EPS Revisions)
2. 机构大单/超大单主动资金流向与盘口主买主卖失衡度 (Order Flow & Money Flow Dynamics)
支持本地 Parquet 缓存与离线断网自适应容错回退机制。
"""

import os
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_storage" / "alternative"


class AlternativeDataFetcher:
    """分析师预期与资金流数据获取器"""

    def __init__(self, cache_dir: Optional[Path] = None, timeout: int = 6):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        
        # 建立独立 Session，绕过本地环境变量代理阻断 (trust_env=False)
        self.session = requests.Session()
        self.session.trust_env = False
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

    # ==========================================
    # 1. 分析师盈利预测与研报获取
    # ==========================================
    def fetch_analyst_reports(
        self,
        symbol: str,
        begin_date: str = "2023-01-01",
        max_pages: int = 3
    ) -> pd.DataFrame:
        """
        获取指定股票的机构分析师盈利预测报告 (EastMoney API)
        """
        code = symbol.split(".")[0]
        url = "https://reportapi.eastmoney.com/report/list"
        all_reports = []

        for page in range(1, max_pages + 1):
            params = {
                "industryCode": "*",
                "pageSize": "50",
                "industry": "*",
                "rating": "*",
                "ratingChange": "*",
                "beginTime": begin_date,
                "endTime": "2028-01-01",
                "pageNo": str(page),
                "fields": "",
                "qType": "0",
                "orgCode": "",
                "code": code,
                "p": str(page),
                "pageNum": str(page),
                "pageNumber": str(page),
            }
            try:
                r = self.session.get(url, params=params, headers=self.headers, timeout=self.timeout)
                if r.status_code != 200:
                    break
                data = r.json()
                items = data.get("data", [])
                if not items:
                    break
                for it in items:
                    all_reports.append({
                        "symbol": symbol,
                        "publish_date": str(it.get("publishDate", ""))[:10],
                        "broker": it.get("orgSName", "") or it.get("orgName", ""),
                        "rating": it.get("emRatingName", ""),
                        "title": it.get("title", ""),
                        "eps_fy1": pd.to_numeric(it.get("predictNextYearEps"), errors="coerce"),
                        "eps_fy2": pd.to_numeric(it.get("predictNextTwoYearEps"), errors="coerce"),
                        "eps_fy0": pd.to_numeric(it.get("predictThisYearEps"), errors="coerce"),
                    })
                total_page = data.get("TotalPage", 1)
                if page >= total_page:
                    break
            except Exception as e:
                logger.warning(f"获取 {symbol} 分析师研报异常 (page {page}): {e}")
                break

        if not all_reports:
            return pd.DataFrame(columns=[
                "symbol", "publish_date", "broker", "rating", "title", "eps_fy1", "eps_fy2", "eps_fy0"
            ])

        df = pd.DataFrame(all_reports)
        df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
        df = df.dropna(subset=["publish_date"]).sort_values(by="publish_date").reset_index(drop=True)
        return df

    # ==========================================
    # 2. 资金流向快照获取 (主力大单/超大单 + 主买主卖)
    # ==========================================
    def fetch_live_money_flow_snapshot(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取全市场或指定标的最新资金流快照 (优先 EastMoney push2，若阻断则平滑回退至腾讯盘口失衡度)
        """
        records = []
        
        # 尝试 EastMoney 主力资金排名接口
        try:
            url = "https://push2.eastmoney.com/api/qt/clist/get"
            params = {
                "fid": "f62",
                "po": "1",
                "pz": "500",
                "pn": "1",
                "np": "1",
                "fltt": "2",
                "invt": "2",
                "ut": "b2884a393a59ad64002292a3e90d46a5",
                "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
                "fields": "f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87",
            }
            r = self.session.get(url, params=params, headers=self.headers, timeout=self.timeout)
            if r.status_code == 200:
                diff = r.json().get("data", {}).get("diff", [])
                for it in diff:
                    code = str(it.get("f12", ""))
                    if not code:
                        continue
                    suffix = ".SH" if code.startswith("6") or code.startswith("688") else ".SZ"
                    sym = code + suffix
                    records.append({
                        "symbol": sym,
                        "main_inflow_net": float(it.get("f62", 0.0) or 0.0),
                        "main_inflow_ratio": float(it.get("f184", 0.0) or 0.0) / 100.0,
                        "super_large_ratio": float(it.get("f69", 0.0) or 0.0) / 100.0,
                        "large_ratio": float(it.get("f75", 0.0) or 0.0) / 100.0,
                        "medium_ratio": float(it.get("f81", 0.0) or 0.0) / 100.0,
                        "small_ratio": float(it.get("f87", 0.0) or 0.0) / 100.0,
                    })
        except Exception as e:
            logger.warning(f"EastMoney 资金流快照获取受限，切换高频盘口主买主卖失衡度: {e}")

        # 若 EastMoney 失败或结果过少，采用高稳定性腾讯盘口外盘/内盘主动单失衡度
        if len(records) < 50 and symbols:
            records = self._fetch_tencent_order_flow(symbols)

        if not records:
            return pd.DataFrame(columns=[
                "symbol", "main_inflow_net", "main_inflow_ratio",
                "super_large_ratio", "large_ratio", "active_imbalance_ratio"
            ])

        res_df = pd.DataFrame(records)
        if symbols:
            res_df = res_df[res_df["symbol"].isin(set(symbols))].reset_index(drop=True)
        return res_df

    def _fetch_tencent_order_flow(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """利用腾讯实时盘口外盘 (主动买单) 与内盘 (主动卖单) 构造超高频主力主动进攻意图比率"""
        records = []
        batch_size = 80
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            tc_codes = []
            for s in batch:
                s_clean = s.split(".")[0].lower()
                prefix = "sh" if s.upper().endswith(".SH") or s_clean.startswith("6") else "sz"
                tc_codes.append(f"{prefix}{s_clean}")
            url = f"http://qt.gtimg.cn/q={','.join(tc_codes)}"
            try:
                r = requests.get(url, timeout=self.timeout)
                r.encoding = "gbk"
                for line in r.text.strip().split(";"):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("~")
                    if len(parts) > 37:
                        code = parts[2]
                        suffix = ".SH" if code.startswith("6") or code.startswith("688") else ".SZ"
                        sym = f"{code}{suffix}"
                        outer_vol = float(parts[7]) if parts[7] else 0.0  # 外盘 (主买)
                        inner_vol = float(parts[8]) if parts[8] else 0.0  # 内盘 (主卖)
                        tot = outer_vol + inner_vol
                        imbalance = (outer_vol - inner_vol) / (tot + 1e-5)
                        records.append({
                            "symbol": sym,
                            "main_inflow_net": (outer_vol - inner_vol) * float(parts[3] or 0),
                            "main_inflow_ratio": imbalance,
                            "super_large_ratio": imbalance * 0.6,
                            "large_ratio": imbalance * 0.4,
                            "active_imbalance_ratio": imbalance,
                        })
            except Exception as e:
                logger.warning(f"腾讯盘口批量解析异常: {e}")
                continue
        return records

    # ==========================================
    # 3. 缓存管理
    # ==========================================
    def cache_analyst_reports(self, df: pd.DataFrame, filename: str = "analyst_consensus_reports.parquet"):
        path = self.cache_dir / filename
        df.to_parquet(path, index=False)
        logger.info(f"分析师研报已成功落盘缓存: {path} (共 {len(df)} 条)")

    def load_cached_analyst_reports(self, filename: str = "analyst_consensus_reports.parquet") -> pd.DataFrame:
        path = self.cache_dir / filename
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()
