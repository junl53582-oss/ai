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
        perf_summary: Optional[Dict[str, Any]] = None,
        is_calibrated_probability: Optional[bool] = None,
        calibration_evidence: Optional[Any] = None
    ) -> str:
        """构建标准化量化交易决策 Markdown 报告 (Zero-Fabrication 审计强化)"""
        import numpy as np
        from models.verified_metrics import ProbabilityCalibrationEvidence

        lines = []
        lines.append("### 📈 **【A股量化系统 · 每日交易决策报告】**")
        lines.append(f"**📅 信号日期 (T日收盘)**: `{signal_date}` | **执行日期 (T+1日开盘)**: `{execution_date}`")
        lines.append(f"**🛡️ 组合风控状态**: <font color='green'>**{macro_status}**</font>")
        lines.append("---")

        # 彻底移除宏观状态触发伪概率：概率校准必须且仅能依赖强类型已核验的 ProbabilityCalibrationEvidence
        is_calibrated = False
        if calibration_evidence is not None:
            if not isinstance(calibration_evidence, ProbabilityCalibrationEvidence) or not calibration_evidence.is_valid():
                raise ValueError(f"Invalid or unverified ProbabilityCalibrationEvidence: {calibration_evidence}")
            is_calibrated = True
        elif is_calibrated_probability is True:
            raise ValueError("Cannot claim calibrated probability without verified ProbabilityCalibrationEvidence")

        score_col_name = "预测上涨概率" if is_calibrated else "模型排序分数"

        if top_df is not None and not top_df.empty:
            lines.append(f"#### 🎯 **推荐目标持仓 (Top {len(top_df)} 标的)**")
            lines.append(f"| 排名 | 标的代码 | 股票名称 | 所属行业 | {score_col_name} | 目标权重 | 现价 |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for idx, row in top_df.reset_index(drop=True).iterrows():
                sym = row.get("symbol", "-")
                if not sym or pd.isna(sym) or sym == "-":
                    raise ValueError(f"Missing symbol at index {idx}")
                name = row.get("name", sym)
                ind = row.get("industry", "UNKNOWN")
                
                # 严格校验 pred_score，禁止缺失默认 0.0% 兜底与伪造概率
                if "pred_score" not in row or row["pred_score"] is None:
                    raise ValueError(f"Missing pred_score for symbol {sym} at index {idx}")
                raw_score = row["pred_score"]
                if pd.isna(raw_score) or np.isinf(raw_score):
                    raise ValueError(f"Invalid pred_score (NaN or Inf) for symbol {sym}: {raw_score}")
                try:
                    score = float(raw_score)
                except (ValueError, TypeError):
                    raise ValueError(f"Non-numeric pred_score for symbol {sym}: {raw_score}")

                if is_calibrated:
                    if not (0.0 <= score <= 1.0):
                        raise ValueError(f"Out-of-bounds calibrated probability for symbol {sym}: {score} (must be in [0, 1])")
                    score_str = f"**{score*100:.2f}%**"
                else:
                    score_str = f"**{score:.4f}**"

                # 严格校验 target_weight 与 close 现价，禁止静默 0.0 兜底
                if "target_weight" not in row or row["target_weight"] is None or pd.isna(row["target_weight"]):
                    raise ValueError(f"Missing or invalid target_weight for symbol {sym} at index {idx}")
                try:
                    weight = float(row["target_weight"])
                except (ValueError, TypeError):
                    raise ValueError(f"Non-numeric target_weight for symbol {sym}: {row['target_weight']}")
                if np.isnan(weight) or np.isinf(weight) or weight < 0:
                    raise ValueError(f"Invalid target_weight for symbol {sym}: {weight}")

                if "close" not in row or row["close"] is None or pd.isna(row["close"]):
                    raise ValueError(f"Missing close price for symbol {sym} at index {idx}")
                try:
                    price = float(row["close"])
                except (ValueError, TypeError):
                    raise ValueError(f"Non-numeric close price for symbol {sym}: {row['close']}")
                if np.isnan(price) or np.isinf(price) or price <= 0:
                    raise ValueError(f"Invalid close price for symbol {sym}: {price}")

                lines.append(
                    f"| {idx+1} | `{sym}` | **{name}** | {ind} | {score_str} | `{weight*100:.1f}%` | {price:.2f}元 |"
                )
        else:
            lines.append("⚠️ 今日无满足条件的推荐标的（防守空仓或全量停牌）")

        lines.append("\n---")
        lines.append("💡 *注：T日收盘完成全量特征计算与走步预测，请于次日 09:25-09:30 集合竞价按目标权重执行挂单。*")
        return "\n".join(lines)

