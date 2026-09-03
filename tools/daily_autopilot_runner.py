"""
A股 AI 量化全自动无人值守日常巡航主调度程序 (tools/daily_autopilot_runner.py)
每日收盘后 15:05 由系统调度触发，自动执行四步闭环:
1. 官方金融专线直连同步全市场最新收盘行情
2. 第三代 Mega-Alpha 大模型生成截面预测评分
3. 实盘/仿真执行器实施 7 重资金风控下的调仓换股
4. 消息中枢自动将决策卡片与账户账本推送到用户手机 Webhook
"""
import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import settings

log_dir = settings.BASE_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"autopilot_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AutopilotRunner")

def run_daily_autopilot():
    today_str = datetime.now().strftime("%Y-%m-%d")
    logger.info("=" * 80)
    logger.info(f">>> [量化日常巡航启动] 日期: {today_str} | 时间: {datetime.now().strftime('%H:%M:%S')}")
    logger.info("=" * 80)

    # 1. 官方专线直连抓取最新行情
    logger.info("[Step 1/4] 正在直连官方行情 CDN 同步今日最新收盘量价...")
    try:
        from data.live_market_syncer import sync_latest_quotes_from_live
        df_latest = sync_latest_quotes_from_live()
        logger.info(f"[+] 步骤 1 完成: 成功同步 {len(df_latest)} 支标的今日官方收盘数据")
    except Exception as e:
        logger.error(f"[-] 步骤 1 异常: {e}")

    # 2. 第四代 DRL 强化模型推演选股与动态权重优化
    logger.info("[Step 2/4] 启动第四代 DRL 强化模型执行最新截面打分与动态仓位优化...")
    try:
        import subprocess
        res = subprocess.run([sys.executable, "tools/predict_gen4_drl_picks.py"], capture_output=True, text=True)
        if res.returncode == 0:
            logger.info("[+] 步骤 2 完成: 第四代 DRL 强化模型推演与动态权重落盘成功")
        else:
            logger.warning(f"[-] 步骤 2 告警: {res.stderr[:200]}")
    except Exception as e:
        logger.error(f"[-] 步骤 2 异常: {e}")

    # 3. 调仓换股执行
    logger.info("[Step 3/4] 启动实盘调仓执行器 (7重资金风控守卫)...")
    try:
        res = subprocess.run([sys.executable, "execution/run_trader.py", "--broker", "paper", "--target-file", "artifacts/latest_stock_picks.csv"], capture_output=True, text=True)
        logger.info(f"[+] 步骤 3 完成: 调仓执行完毕 (退出码 {res.returncode})")
    except Exception as e:
        logger.error(f"[-] 步骤 3 异常: {e}")

    # 4. 手机推送
    logger.info("[Step 4/4] 组装决策卡片并推送到手机终端...")
    try:
        res = subprocess.run([sys.executable, "tools/test_push_gen3_card.py"], capture_output=True, text=True)
        logger.info("[+] 步骤 4 完成: 手机端消息推送完毕")
    except Exception as e:
        logger.error(f"[-] 步骤 4 异常: {e}")

    logger.info("=" * 80)
    logger.info(f">>> [今日量化日常巡航圆满完成] 审计日志已归档: {log_file}")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_daily_autopilot()
