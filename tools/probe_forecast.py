"""业绩预告接口探测: 字段完整性与可用性"""
import json
import sys
from pathlib import Path

import requests

r = requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',
                 params={'reportName': 'RPT_PUBLIC_OP_PREDICT', 'columns': 'ALL',
                         'filter': "(REPORTDATE='2026-06-30')",
                         'pageSize': 3, 'pageNumber': 1,
                         'sortColumns': 'NOTICE_DATE', 'sortTypes': '-1',
                         'source': 'WEB', 'client': 'WEB'},
                 timeout=12, headers={'User-Agent': 'Mozilla/5.0'})
d = r.json()
res = d.get('result') or {}
rows = res.get('data') or []
out = [f"HTTP {r.status_code} | 总数 {res.get('count')} | success={d.get('success')}"]
if rows:
    out.append("全部字段: " + ", ".join(sorted(rows[0].keys())))
    out.append("样本: " + json.dumps({k: rows[0][k] for k in list(rows[0])[:14]}, ensure_ascii=False))
Path(r'E:\股票预测\artifacts\forecast_probe.txt').write_text("\n".join(out), encoding='utf-8')
print("done")
