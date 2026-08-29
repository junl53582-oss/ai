"""
盘后自动化调度与决策流水线 (scheduler/daily_runner.py)
每日 15:05 收盘后自动同步行情、构建多因子矩阵、模型预测、组合优化与次日买卖决策多渠道推送。
"""
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

# 加入项目根目录
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import settings
from data.universe_provider import create_universe_provider
from data.data_manager import DataManager
from factors.processor import FactorProcessor
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from models.evaluator import ModelEvaluator
from strategy.portfolio import PortfolioBuilder
from scheduler.notifier import MessageNotifier

logger = logging.getLogger(__name__)


def run_daily_automation(
    webhook_url: Optional[str] = None,
    channel: str = "feishu",
    optimizer_type: str = "equal",
    force_update: bool = False
) -> Dict[str, Any]:
    """执行每日盘后自动化任务并推送决策清单"""
    print("\n" + "=" * 70)
    print(">> 启动 A股盘后量化自动化任务 (Daily Automation Runner)")
    print("=" * 70)

    # 1. 同步最新数据
    print("[1/4] 同步全市场与成分股行情数据...")
    univ_provider = create_universe_provider(settings)
    data_manager = DataManager(universe_provider=univ_provider)
    market_df = data_manager.sync_and_build_dataset(force_update=force_update)

    # 2. 计算特征与中性化
    print("[2/4] 计算 Alpha158 + A股定制 + 另类因子并执行截面中性化...")
    processor = FactorProcessor()
    factor_df = processor.build_and_save_factor_matrix(market_df, force_update=force_update)
    labeler = TargetLabeler(horizon=settings.LABEL_HORIZON)
    factor_df = labeler.compute_excess_return_label(factor_df, canonical_dates=data_manager.get_trading_calendar())

    # 3. 走步训练与预测
    print("[3/4] 执行走步预测提取最新截面打分...")
    trainer = WalkForwardTrainer()
    oos_df, latest_model = trainer.run_walk_forward(factor_df)

    latest_date = oos_df["date"].max()
    daily_df = oos_df[oos_df["date"] == latest_date].copy()
    expected_exec_date = data_manager.get_next_trading_date(latest_date)
    exec_str = expected_exec_date.strftime("%Y-%m-%d") if expected_exec_date else "次一交易日"

    # 4. 组合优化与生成目标仓位
    print(f"[4/4] 运行组合优化器 ({optimizer_type}) 生成次日买卖决策...")
    builder = PortfolioBuilder(
        top_k_buy=settings.TOP_K_BUY,
        top_k_hold=settings.TOP_K_HOLD,
        weight_method=optimizer_type,
        universe_provider=univ_provider
    )
    top_df = builder.build_target_portfolio(daily_df, current_holdings=set(), date=latest_date)

    # 5. 构建报告并推送通知
    sig_str = latest_date.strftime("%Y-%m-%d")
    report_md = MessageNotifier.format_daily_report_markdown(
        signal_date=sig_str,
        execution_date=exec_str,
        top_df=top_df,
        macro_status="正常多头持仓"
    )

    print("\n" + "-" * 70)
    print(report_md)
    print("-" * 70)

    if webhook_url:
        print(f"正在通过 {channel} Webhook 分发消息通知...")
        if channel == "feishu":
            MessageNotifier.send_feishu_card(webhook_url, f"A股量化决策报告 ({sig_str})", report_md)
        elif channel == "wechat":
            MessageNotifier.send_wechat_work(webhook_url, report_md)
        elif channel == "dingtalk":
            MessageNotifier.send_dingtalk(webhook_url, f"A股量化决策报告 ({sig_str})", report_md)
        print("✅ 消息分发成功！")

    return {
        "signal_date": sig_str,
        "execution_date": exec_str,
        "top_portfolio": top_df,
        "markdown_report": report_md
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A股盘后量化自动化任务")
    parser.add_argument("--webhook", type=str, default=None, help="机器人 Webhook 地址")
    parser.add_argument("--channel", type=str, default="feishu", choices=["feishu", "wechat", "dingtalk"])
    parser.add_argument("--optimizer", type=str, default="equal", help="组合优化器类型")
    parser.add_argument("--force-update", action="store_true", help="强制更新数据")
    args = parser.parse_args()

    run_daily_automation(
        webhook_url=args.webhook,
        channel=args.channel,
        optimizer_type=args.optimizer,
        force_update=args.force_update
    )
