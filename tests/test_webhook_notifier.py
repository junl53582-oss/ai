import pytest
from notifications.webhook_notifier import QuantWebhookNotifier

def test_webhook_notifier_formatting_and_local_send():
    notifier = QuantWebhookNotifier(webhook_url=None)
    top_stocks = [
        {"symbol": "300308.SZ", "name": "中际旭创", "industry": "通信设备", "close": 870.22, "pred_score": 0.5155, "target_weight": 0.1641},
        {"symbol": "688256.SH", "name": "寒武纪", "industry": "半导体", "close": 969.05, "pred_score": 0.5136, "target_weight": 0.0950}
    ]
    account_info = {
        "total_equity": 999917.53,
        "cash": 670047.53,
        "holding_count": 5
    }
    safety_events = [
        "[GUARD] [防线3: 日换手熔断] 买入总额已触发等比例安全裁剪"
    ]
    
    md = notifier.format_markdown_report("2026-08-24", top_stocks, account_info, safety_events)
    assert "中际旭创" in md
    assert "寒武纪" in md
    assert "日换手熔断" in md
    assert "999,917.53" in md

    # 本地控制台安全发送
    success = notifier.send_report("2026-08-24", top_stocks, account_info, safety_events)
    assert success is True
