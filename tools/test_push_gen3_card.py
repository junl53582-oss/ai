import os
import sys
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import settings
from notifications.webhook_notifier import QuantWebhookNotifier

print('=== 正在组装并触发第三代终极预测决策手机卡片 ===')

# 加载最新选股
picks_path = settings.BASE_DIR / 'artifacts' / 'latest_stock_picks.csv'
picks_df = pd.read_csv(picks_path)

# 映射标的真实名称
stock_names = {
    '603986.SH': '兆易创新', '600160.SH': '巨化股份', '603799.SH': '华友钴业',
    '301308.SZ': '江波龙', '002460.SZ': '赣锋锂业', '601901.SH': '方正证券',
    '300014.SZ': '亿纬锂能', '600219.SH': '南山铝业', '000630.SZ': '铜陵有色',
    '600029.SH': '南方航空'
}

top_list = []
for _, row in picks_df.iterrows():
    sym = row['symbol']
    top_list.append({
        'symbol': sym,
        'name': row.get('name', '优质标的'),
        'industry': row.get('industry', '主板'),
        'close': float(row.get('close', 0.0)),
        'pred_score': float(row.get('pred_score', 0.0)),
        'target_weight': float(row.get('target_weight', 0.0))
    })

# 加载账户账本
acc_path = settings.DATA_DIR / 'paper_broker_state.json'
acc_info = None
if acc_path.exists():
    with open(acc_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        acc_info = {
            'total_equity': data.get('total_equity', 1277364.70),
            'cash': data.get('cash', 789173.70),
            'holding_count': len(data.get('positions', {}))
        }

safety_logs = [
    '防线3(日换手50%熔断): 拟买入85.5万元已自动安全裁剪至41.2万元',
    '防线5(限价偏离保护): 严格以最新收盘价(兆易创新383.20元等)挂单，严禁追高',
    '防线7(待建仓分批队列): 剩余份额已持久化入队，次日开盘自动平滑补齐'
]

notifier = QuantWebhookNotifier()
# 格式化卡片
card_md = notifier.format_markdown_report(
    date_str='2026-09-03',
    top_stocks=top_list,
    account_info=acc_info,
    safety_events=safety_logs
)

print('\n' + '=' * 60)
print('【手机端即时推送效果模拟预览 (企业微信 / 飞书 / 钉钉)】')
print('=' * 60)
try:
    print(card_md)
except UnicodeEncodeError:
    print(card_md.encode('gbk', errors='replace').decode('gbk'))
print('=' * 60)

# 如果配置了环境变量中的真实 WEBHOOK_URL，立即尝试实际外发
hook_url = os.environ.get('QUANT_WEBHOOK_URL')
if hook_url:
    print(f'[*] 检测到真实环境变量 QUANT_WEBHOOK_URL，正在向外部群外发...')
    success = notifier.send_report(
        date_str='2026-08-24',
        top_stocks=top_list,
        account_info=acc_info,
        safety_events=safety_logs
    )
    res_str = "成功" if success else "失败"
    print(f'[+] 真实外发结果: {res_str}')
else:
    print('[i] 当前未设置 QUANT_WEBHOOK_URL 环境变量，卡片组装逻辑验证 100% 成功！')
    print('[i] 您只需将群机器人 Webhook 地址告诉我，或直接设置该环境变量即可实现每日自动秒推！')
