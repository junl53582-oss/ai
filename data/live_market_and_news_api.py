"""
全自动官方行情与 7x24 财经消息直连 API 模块 (data/live_market_and_news_api.py)
功能:
1. LiveMarketAPI: 官方行情直连获取全市场股票最新实时量价 (毫秒级，无 Token 限制)
2. LiveNewsAPI: 直连新浪/财联社 7x24 全球财经快讯与个股最新公告资讯
3. AutoSyncEngine: 自动抓取并整合最新行情与新闻催化剂
"""
import time
import json
import re
import requests
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class LiveMarketAPI:
    """官方高频行情 CDN 直连 API 客户端"""
    
    @staticmethod
    def _format_symbol_for_tencent(symbol: str) -> str:
        sym = symbol.strip().lower()
        if sym.endswith('.sh'):
            return 'sh' + sym.replace('.sh', '')
        elif sym.endswith('.sz'):
            return 'sz' + sym.replace('.sz', '')
        elif sym.startswith('sh') or sym.startswith('sz'):
            return sym
        elif sym.startswith('6'):
            return 'sh' + sym
        else:
            return 'sz' + sym

    @classmethod
    def fetch_live_quotes(cls, symbols: List[str], timeout: int = 6) -> pd.DataFrame:
        """批量获取指定标的最新实时行情"""
        tc_symbols = [cls._format_symbol_for_tencent(s) for s in symbols]
        url = 'http://qt.gtimg.cn/q=' + ','.join(tc_symbols)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.encoding = 'gbk'
        except Exception as e:
            logger.error(f'行情接口请求失败: {e}')
            return pd.DataFrame()
            
        records = []
        for line in r.text.strip().split(';'):
            line = line.strip()
            if not line:
                continue
            parts = line.split('~')
            if len(parts) > 35:
                try:
                    name = parts[1]
                    code = parts[2]
                    # 补充后缀
                    suffix = '.SH' if code.startswith('6') or code.startswith('688') else '.SZ'
                    sym_full = code + suffix
                    cur_price = float(parts[3])
                    pre_close = float(parts[4])
                    pct_change = float(parts[32])
                    high = float(parts[33])
                    low = float(parts[34])
                    amount_wan = float(parts[37]) # 万元
                    trade_time = parts[30]
                    
                    records.append({
                        'symbol': sym_full,
                        'name': name,
                        'close': cur_price,
                        'pre_close': pre_close,
                        'pct_change': pct_change / 100.0,
                        'high': high,
                        'low': low,
                        'amount_yi': round(amount_wan / 10000.0, 2), # 亿元
                        'trade_time': trade_time
                    })
                except Exception as ex:
                    logger.warning(f'解析行情失败: {ex}')
                    continue
                    
        return pd.DataFrame(records)


class LiveNewsAPI:
    """7x24 实时财经资讯与个股新闻直连 API"""
    
    @staticmethod
    def fetch_7x24_telegraph(num: int = 15, timeout: int = 5) -> List[Dict[str, Any]]:
        """获取 7x24 实时快讯直播流"""
        url = f'https://zhibo.sina.com.cn/api/zhibo/feed?page=1&page_size={num}&zhibo_id=152'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            data = r.json()
            items = data.get('result', {}).get('data', {}).get('feed', {}).get('list', [])
            res = []
            for it in items:
                raw_text = it.get('rich_text', '').replace('\n', ' ').strip()
                clean_text = re.sub(r'<.*?>', '', raw_text)
                clean_text = re.sub(r'https?://\S+', '', clean_text)
                if len(clean_text) > 10:
                    res.append({
                        'time': it.get('create_time', time.strftime('%H:%M:%S')),
                        'content': clean_text[:120]
                    })
            return res
        except Exception as e:
            logger.error(f'7x24 资讯拉取异常: {e}')
            return []

    @staticmethod
    def fetch_stock_latest_news(keyword: str, num: int = 3, timeout: int = 5) -> List[str]:
        """抓取指定股票/行业的最新重要财经消息"""
        url = f'https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&k={keyword}&num={num}&page=1'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            data = r.json()
            items = data.get('result', {}).get('data', [])
            titles = []
            for it in items:
                title = it.get('title', '').strip()
                if title and len(title) > 6:
                    titles.append(title)
            return titles
        except Exception as e:
            logger.error(f'个股新闻拉取异常: {e}')
            return []


class AutoSyncEngine:
    """自动同步引擎：一键刷新最新行情与个股消息"""
    
    @classmethod
    def sync_picks_and_news(cls, picks_file: Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        if not picks_file.exists():
            return pd.DataFrame(), []
            
        df = pd.read_csv(picks_file)
        symbols = df['symbol'].tolist()
        
        # 1. 抓取最新官方实时行情
        logger.info(f'[*] 正在直连官方行情 CDN 自动获取 {len(symbols)} 支标的最新数据...')
        quote_df = LiveMarketAPI.fetch_live_quotes(symbols)
        
        if not quote_df.empty:
            quote_map = quote_df.set_index('symbol')
            for idx, r in df.iterrows():
                sym = r['symbol']
                if sym in quote_map.index:
                    q = quote_map.loc[sym]
                    df.at[idx, 'close'] = q['close']
                    df.at[idx, 'pct_change'] = q['pct_change']
                    
        # 2. 抓取 7x24 实时快讯
        logger.info('[*] 正在拉取 7x24 实时财经电报流...')
        telegraph = LiveNewsAPI.fetch_7x24_telegraph(num=12)
        
        # 3. 自动为每只股票拉取最新关联资讯 (如果匹配到实时新闻则动态升级)
        logger.info('[*] 正在匹配个股最新实时重大消息催化...')
        for idx, r in df.iterrows():
            nm = r['name']
            live_news = LiveNewsAPI.fetch_stock_latest_news(nm, num=2)
            if live_news:
                # 选取第一条作为最新实时催化
                df.at[idx, 'news_catalyst'] = f'【实时快讯】{live_news[0]}'
                
        # 4. 重新落盘
        df.to_csv(picks_file, index=False, encoding='utf-8-sig')
        logger.info(f'[+] 最新行情与消息已成功自动同步落盘至: {picks_file.name}')
        return df, telegraph
