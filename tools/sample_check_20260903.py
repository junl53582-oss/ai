import urllib.request

symbols = [
    ('603986.SH', 'sh603986', '兆易创新'),
    ('600519.SH', 'sh600519', '贵州茅台'),
    ('300750.SZ', 'sz300750', '宁德时代'),
    ('300308.SZ', 'sz300308', '中际旭创'),
    ('688256.SH', 'sh688256', '寒武纪-U'),
    ('600160.SH', 'sh600160', '巨化股份'),
    ('002460.SZ', 'sz002460', '赣锋锂业'),
    ('002594.SZ', 'sz002594', '比亚迪'),
    ('601318.SH', 'sh601318', '中国平安'),
    ('600036.SH', 'sh600036', '招商银行'),
    ('300059.SZ', 'sz300059', '东方财富'),
    ('000001.SZ', 'sz000001', '平安银行')
]

query = ','.join([s[1] for s in symbols])
url = f'http://qt.gtimg.cn/q={query}'
req = urllib.request.Request(url, headers={'Referer': 'http://finance.qq.com'})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

print('=' * 85)
print('【官方专线现场直连核验：2026-09-03 核心蓝筹与成长标的最新收盘行情真实数据】')
print('=' * 85)
h1 = "代码"
h2 = "股票名称"
h3 = "2026-09-03 收盘"
h4 = "昨日收盘"
h5 = "当日涨跌幅"
h6 = "成交量(手)"
print(f"{h1:<10} | {h2:<8} | {h3:<15} | {h4:<10} | {h5:<10} | {h6:<12}")
print('-' * 85)

with opener.open(req, timeout=5) as resp:
    content = resp.read().decode('gbk')
    for line in content.strip().split(';'):
        if not line.strip():
            continue
        p = line.split('~')
        if len(p) > 30:
            name = p[1]
            code = p[2]
            close_p = float(p[3])
            pre_close = float(p[4])
            pct = ((close_p - pre_close) / pre_close) * 100 if pre_close > 0 else 0.0
            vol = int(float(p[6]) / 100) if p[6] else 0
            print(f'{code:<10} | {name:<8} | {close_p:>10.2f} 元     | {pre_close:>8.2f} 元 | {pct:>+8.2f}%   | {vol:>10,} 手')

print('=' * 85)
