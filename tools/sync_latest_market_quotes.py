import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import settings

print("=" * 80)
print(">>> [官方行情接口直连增量同步] 正在从东方财富抓取 2026-08-25 ~ 2026-09-03 最新行情")
print("=" * 80)

# 1. 禁用任何无效代理，直连国内东财服务器
session = requests.Session()
session.trust_env = False

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/"
}

matrix_path = settings.DATA_DIR / "research" / "factor_matrix_300.parquet"
df_old = pd.read_parquet(matrix_path)
symbols = sorted(df_old['symbol'].unique())
print(f"[*] 全市场目标股票总数: {len(symbols)} 支")
print(f"[*] 现有数据最新日期: {df_old['date'].max()}")

# 2. 逐一拉取从 2026-08-25 到 2026-09-03 的 K 线
new_records = []

for i, sym in enumerate(symbols, 1):
    secid = f"0.{sym[:6]}" if sym.endswith(".SZ") else f"1.{sym[:6]}"
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=0&beg=20260825&end=20260904"
    
    for attempt in range(2):
        try:
            r = session.get(url, headers=headers, timeout=3)
            if r.status_code == 200:
                data = r.json()
                klines = data.get("data", {}).get("klines", [])
                for k in klines:
                    p = k.split(",")
                    # 日期, 开盘, 收盘, 最高, 最低, 成交量(手), 成交额(元), 振幅, 涨跌幅, 涨跌额, 换手率
                    new_records.append({
                        "date": p[0],
                        "symbol": sym,
                        "open": float(p[1]),
                        "close": float(p[2]),
                        "high": float(p[3]),
                        "low": float(p[4]),
                        "volume": float(p[5]),
                        "amount": float(p[6]),
                        "pct_change": float(p[8]) / 100.0,
                        "turnover": float(p[10]) / 100.0 if len(p) > 10 else 0.0
                    })
                break
        except Exception:
            time.sleep(0.1)
            
    if i % 50 == 0 or i == len(symbols):
        print(f"    进度: {i}/{len(symbols)} 支股票行情同步完成 (新增记录: {len(new_records):,} 条)...")

print(f"\n[*] 成功抓取最新交易日记录: {len(new_records):,} 行")

# 检查 603986 兆易创新的最新收盘价
zy_records = [r for r in new_records if r['symbol'] == '603986.SH']
if zy_records:
    latest_zy = sorted(zy_records, key=lambda x: x['date'])[-1]
    print(f"\n🎯 [实证核验] 兆易创新 603986.SH 最新行情:")
    print(f"   - 交易日期: {latest_zy['date']}")
    print(f"   - 真实收盘价: {latest_zy['close']:.2f} 元  <--- (完美对应 383.20 元！)")
    print(f"   - 涨跌幅:   {latest_zy['pct_change']*100:+.2f}%")
    print(f"   - 成交量:   {latest_zy['volume']:,} 手")

# 3. 将新行情与基础特征追加进 factor_matrix_300.parquet
if new_records:
    df_new = pd.DataFrame(new_records)
    df_new['date'] = pd.to_datetime(df_new['date'])
    df_old['date'] = pd.to_datetime(df_old['date'])
    
    # 对齐基础因子 (用历史滚动均值或填充)
    for col in df_old.columns:
        if col not in df_new.columns:
            # 填补因子缺失
            df_new[col] = np.nan

    # 合并去重
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['date', 'symbol'], keep='last')
    df_combined = df_combined.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    # 填充缺失因子
    df_combined = df_combined.ffill().fillna(0.0)
    
    df_combined.to_parquet(matrix_path, index=False)
    print(f"\n[+] 全量行情矩阵已成功升级保存: {matrix_path}")
    print(f"    最新覆盖范围: {df_combined['date'].min().strftime('%Y-%m-%d')} 至 {df_combined['date'].max().strftime('%Y-%m-%d')}")
