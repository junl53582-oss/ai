# 1. 确保 .env 配置文件存在并包含 Webhook 配置模版
import os
from pathlib import Path

env_path = Path(".env")
webhook_tpl = """# ==============================================================================
# 量化系统企业级配置文件 (.env)
# ==============================================================================
# 手机端消息推送 Webhook 地址 (支持 企业微信 / 飞书 / 钉钉 群机器人)
# 格式示例:
# 飞书:   https://open.feishu.cn/open-apis/bot/v2/hook/xxxx-xxxx
# 企微:   https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx-xxxx
# 钉钉:   https://oapi.dingtalk.com/robot/send?access_token=xxxx-xxxx
QUANT_WEBHOOK_URL=
"""
if not env_path.exists() or "QUANT_WEBHOOK_URL" not in env_path.read_text(encoding="utf-8", errors="ignore"):
    with open(env_path, "a", encoding="utf-8") as f:
        f.write("\n" + webhook_tpl)
print("[+] .env 配置文件已就绪: 支持直接填入手机端 Webhook")
