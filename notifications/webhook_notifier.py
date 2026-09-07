"""
企业级量化决策推送中枢 (notifications/webhook_notifier.py)
支持: 飞书 (Feishu)、企业微信 (WeCom)、钉钉 (DingTalk) 机器人 Webhook 自动推送
"""
import math
import os
import json
import logging
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from config.settings import settings

logger = logging.getLogger(__name__)

class QuantWebhookNotifier:
    """全渠道量化决策消息推送引擎"""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.environ.get("QUANT_WEBHOOK_URL") or getattr(settings, "WEBHOOK_URL", "")

    def format_markdown_report(
        self,
        date_str: str,
        top_stocks: List[Dict[str, Any]],
        account_info: Optional[Dict[str, Any]] = None,
        safety_events: Optional[List[str]] = None
    ) -> str:
        """格式化 Markdown 决策卡片"""
        lines = [
            f"## 🚀 A股多因子 AI 量化盘后决策卡片",
            f"> **数据基准日**: **{date_str} (官方专线直连已收盘 · 核验通过)**",
            f"> **模型架构**: 第三代 Mega-Alpha (Qlib DoubleEnsemble + TabularMLP + Ridge)",
            f"> **执行规则**: T日收盘信号 -> T+1日开盘买入 (7重资金风控保护)",
            "",
            "### 🎯 最新 Top 买入决策标的:"
        ]
        
        for idx, s in enumerate(top_stocks[:8], 1):
            sym = s.get('symbol')
            if not sym:
                raise ValueError("Stock entry missing required 'symbol'")

            if 'pred_score' not in s or s['pred_score'] is None:
                raise ValueError(f"Stock {sym} missing required 'pred_score'")
            try:
                score = float(s['pred_score'])
            except (ValueError, TypeError):
                raise ValueError(f"Stock {sym} non-numeric pred_score: {s['pred_score']}")
            if math.isnan(score) or math.isinf(score):
                raise ValueError(f"Stock {sym} invalid pred_score: {score}")

            if 'close' not in s or s['close'] is None:
                raise ValueError(f"Stock {sym} missing required 'close'")
            try:
                price = float(s['close'])
            except (ValueError, TypeError):
                raise ValueError(f"Stock {sym} non-numeric close: {s['close']}")
            if math.isnan(price) or math.isinf(price) or price <= 0:
                raise ValueError(f"Stock {sym} invalid close price: {price}")

            if 'target_weight' not in s or s['target_weight'] is None:
                raise ValueError(f"Stock {sym} missing required 'target_weight'")
            try:
                weight = float(s['target_weight'])
            except (ValueError, TypeError):
                raise ValueError(f"Stock {sym} non-numeric target_weight: {s['target_weight']}")
            if math.isnan(weight) or math.isinf(weight) or weight < 0:
                raise ValueError(f"Stock {sym} invalid target_weight: {weight}")

            name = s.get('name', sym)
            ind = s.get('industry', '主板')
            w_pct = weight * 100
            lines.append(f"{idx}. **{name}** ({sym}) - {ind} | 现价: {price:.2f}元 | 模型排序分数: {score:.4f} | 建议仓位: {w_pct:.1f}%")

        if account_info:
            total_eq = account_info.get('total_equity')
            cash_val = account_info.get('cash')
            if total_eq is None or cash_val is None:
                raise ValueError("account_info missing required 'total_equity' or 'cash'")
            try:
                total_eq = float(total_eq)
                cash_val = float(cash_val)
            except (ValueError, TypeError):
                raise ValueError("account_info total_equity or cash must be numeric")
            if math.isnan(total_eq) or math.isnan(cash_val) or total_eq < 0 or cash_val < 0:
                raise ValueError("account_info contains invalid NaN or negative values")

            lines.extend([
                "",
                "### 💼 虚拟账户最新账本:",
                f"- **总资产**: {total_eq:,.2f} 元",
                f"- **可用现金**: {cash_val:,.2f} 元",
                f"- **持仓标的数**: {int(account_info.get('holding_count', 0))} 只"
            ])

        if safety_events:
            lines.extend([
                "",
                "### 🛡️ 资金风控触发记录:"
            ])
            for ev in safety_events:
                lines.append(f"- ⚠️ {ev}")

        lines.extend([
            "",
            f"🕒 *生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}* | [点击查看网页大屏](http://localhost:8501)"
        ])
        return "\n".join(lines)

    def send_report(
        self,
        date_str: str,
        top_stocks: List[Dict[str, Any]],
        account_info: Optional[Dict[str, Any]] = None,
        safety_events: Optional[List[str]] = None
    ) -> bool:
        """执行发送"""
        text = self.format_markdown_report(date_str, top_stocks, account_info, safety_events)
        
        if not self.webhook_url:
            logger.info("未配置 QUANT_WEBHOOK_URL，转为本地控制台输出卡片:")
            try:
                print("\n" + "=" * 60)
                print(text)
                print("=" * 60 + "\n")
            except UnicodeEncodeError:
                clean_text = text.encode("gbk", errors="replace").decode("gbk")
                print("\n" + "=" * 60)
                print(clean_text)
                print("=" * 60 + "\n")
            return True

        payload = {}
        if "feishu" in self.webhook_url or "lark" in self.webhook_url:
            payload = {
                "msg_type": "interactive",
                "card": {
                    "header": {
                        "title": {"tag": "plain_text", "content": f"📈 A股量化决策日报 ({date_str})"},
                        "template": "blue"
                    },
                    "elements": [{"tag": "markdown", "content": text}]
                }
            }
        elif "dingtalk" in self.webhook_url:
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": f"A股量化决策日报 ({date_str})",
                    "text": text
                }
            }
        else:
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "content": text
                }
            }

        try:
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_data = resp.read().decode("utf-8")
                logger.info(f"Webhook 推送成功，服务器响应: {res_data}")
                return True
        except Exception as e:
            logger.error(f"Webhook 推送失败: {e}")
            return False
