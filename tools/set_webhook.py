import os
import sys
from pathlib import Path

def set_webhook(url: str):
    env_file = Path(".env")
    lines = []
    found = False
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("QUANT_WEBHOOK_URL="):
                    lines.append(f"QUANT_WEBHOOK_URL={url.strip()}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"\nQUANT_WEBHOOK_URL={url.strip()}\n")
    with open(env_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[+] 成功配置手机端 Webhook: {url.strip()}")
    print("[*] 正在向该 Webhook 触发测试连接...")
    
    from notifications.webhook_notifier import QuantWebhookNotifier
    notifier = QuantWebhookNotifier(webhook_url=url.strip())
    test_stocks = [{'symbol': '603986.SH', 'name': '兆易创新', 'industry': '半导体', 'close': 383.20, 'pred_score': 0.4836, 'target_weight': 0.095}]
    succ = notifier.send_report(date_str="2026-09-03", top_stocks=test_stocks)
    print(f"[+] 测试推送结果: {'成功到达手机！' if succ else '发送失败，请检查URL是否有效'}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        set_webhook(sys.argv[1])
    else:
        print("使用说明: python tools/set_webhook.py <您的飞书/企微/钉钉机器人Webhook地址>")
