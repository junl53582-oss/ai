"""
多渠道消息通知中心 (scheduler/notifier.py)
支持将每日盘后选股与次日调仓决策清单自动推送至飞书、企业微信、钉钉机器人与电子邮箱。
"""
import os
import json
import logging
from typing import Optional, Dict, Any, List
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class MessageNotifier:
    """量化决策多通道通知分发器"""

    @classmethod
    def send_feishu_card(cls, webhook_url: str, title: str, content_markdown: str) -> bool:
        """发送飞书交互式卡片消息"""
        if not webhook_url:
            return False
        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": "blue"
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": content_markdown
                    }
                ]
            }
        }
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"飞书推送异常: {e}")
            return False

    @classmethod
    def send_wechat_work(cls, webhook_url: str, content_markdown: str) -> bool:
        """发送企业微信 Markdown 消息"""
        if not webhook_url:
            return False
        payload = {
            "msg_type": "markdown",
            "markdown": {"content": content_markdown}
        }
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"企微推送异常: {e}")
            return False

    @classmethod
    def send_dingtalk(cls, webhook_url: str, title: str, content_markdown: str) -> bool:
        """发送钉钉 Markdown 消息"""
        if not webhook_url:
            return False
        payload = {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": content_markdown}
        }
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"钉钉推送异常: {e}")
            return False

    @classmethod
    def format_daily_report_markdown(
        cls,
        signal_date: str,
        execution_date: str,
        top_df: pd.DataFrame,
        macro_status: str = "正常持仓",
        perf_summary: Optional[Dict[str, Any]] = None
    ) -> str:
        """构建标准化量化交易决策 Markdown 报告"""
        lines = []
        lines.append(f"### 📈 **【A股量化系统 · 每日交易决策报告】**")
        lines.append(f"**📅 信号日期 (T日收盘)**: `{signal_date}` | **执行日期 (T+1日开盘)**: `{execution_date}`")
        lines.append(f"**🛡️ 组合风控状态**: <font color='green'>**{macro_status}**</font>")
        lines.append("---")

        if top_df is not None and not top_df.empty:
            lines.append(f"#### 🎯 **推荐目标持仓 (Top {len(top_df)} 标的)**")
            lines.append("| 排名 | 标的代码 | 股票名称 | 所属行业 | 预测上涨概率 | 目标权重 | 现价 |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for idx, row in top_df.reset_index(drop=True).iterrows():
                sym = row.get("symbol", "-")
                name = row.get("name", "-")
                ind = row.get("industry", "UNKNOWN")
                score = row.get("pred_score", 0.0)
                weight = row.get("target_weight", 0.0)
                price = row.get("close", 0.0)
                lines.append(
                    f"| {idx+1} | `{sym}` | **{name}** | {ind} | **{score*100:.1f}%** | `{weight*100:.1f}%` | {price:.2f}元 |"
                )
        else:
            lines.append("⚠️ 今日无满足条件的推荐标的（防守空仓或全量停牌）")

        lines.append("\n---")
        lines.append(f"💡 *注：T日收盘完成全量特征计算与走步预测，请于次日 09:25-09:30 集合竞价按目标权重执行挂单。*")
        return "\n".join(lines)
