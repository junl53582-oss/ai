"""
盘后自动化调度与决策流水线 (scheduler/daily_runner.py)
每日 15:05 收盘后自动同步行情、构建多因子矩阵、模型预测、组合优化与次日买卖决策多渠道推送。

Phase A 健壮性加固:
- 分阶段异常捕获与自动重试 (网络类阶段重试 3 次, 指数退避)
- 任一阶段最终失败: 推送失败告警 (若配置 webhook) 后以非零状态抛出, 严禁静默死亡
- 支持 current_holdings 传入, 修复实盘决策路径滞回缓冲 (TOP_K_HOLD) 永不生效的缺陷
"""
import sys
import time
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Set
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


def _execute_stage(stage_name: str, fn: Callable[[], Any], retries: int = 1, retry_delay: float = 5.0) -> Any:
    """执行单个流水线阶段: 带重试与指数退避, 最终失败抛出 RuntimeError。

    Args:
        stage_name: 阶段名 (用于日志与错误定位)
        fn: 无参可调用对象
        retries: 最大重试次数 (总尝试 = 1 + retries)
        retry_delay: 首次重试延迟秒数, 之后指数退避 (x2)
    """
    attempt = 0
    delay = retry_delay
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt > retries:
                logger.error(f"❌ [{stage_name}] 最终失败 (共尝试 {attempt} 次): {e}\n{traceback.format_exc()}")
                raise RuntimeError(f"阶段 [{stage_name}] 重试 {retries} 次后仍失败: {e}") from e
            logger.warning(f"⚠️ [{stage_name}] 第 {attempt} 次尝试失败: {e} — {delay:.0f}s 后重试...")
            time.sleep(delay)
            delay *= 2


def _push_failure_alert(webhook_url: Optional[str], channel: str, stage_error: str):
    """流水线最终失败时的告警推送 (尽力而为, 不再抛错)"""
    if not webhook_url:
        logger.critical(f"🚨 盘后流水线失败且未配置 webhook: {stage_error}")
        return
    alert_md = f"## 🚨 A股盘后流水线失败告警\n\n```\n{stage_error[:1500]}\n```\n\n请尽快人工介入检查数据/模型/网络状态。"
    try:
        if channel == "feishu":
            MessageNotifier.send_feishu_card(webhook_url, "🚨 A股量化流水线失败告警", alert_md)
        elif channel == "wechat":
            MessageNotifier.send_wechat_work(webhook_url, alert_md)
        elif channel == "dingtalk":
            MessageNotifier.send_dingtalk(webhook_url, "🚨 A股量化流水线失败告警", alert_md)
        logger.info("失败告警已推送")
    except Exception as push_err:
        logger.critical(f"🚨 流水线失败且告警推送也失败: {push_err} | 原始错误: {stage_error}")


def run_daily_automation(
    webhook_url: Optional[str] = None,
    channel: str = "feishu",
    optimizer_type: str = "equal",
    force_update: bool = False,
    current_holdings: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """执行每日盘后自动化任务并推送决策清单

    Args:
        current_holdings: 当前真实持仓标的集合。不传则退化为空集 (滞回缓冲不生效,
                          新买入门槛为 TOP_K_BUY); 传入后已持仓标的按 TOP_K_HOLD 宽缓冲保留。
    """
    print("\n" + "=" * 70)
    print(">> 启动 A股盘后量化自动化任务 (Daily Automation Runner)")
    print("=" * 70)

    holdings = current_holdings if current_holdings is not None else set()
    if holdings:
        logger.info(f"已接入真实持仓 {len(holdings)} 只, 滞回缓冲 (TOP_K_HOLD={settings.TOP_K_HOLD}) 生效")
    else:
        logger.warning("⚠️ 未传入 current_holdings, 滞回缓冲不生效 (所有标的按新买入 TOP_K_BUY 门槛处理)")

    try:
        # 1. 同步最新数据 (网络阶段, 重试 3 次)
        print("[1/4] 同步全市场与成分股行情数据...")
        univ_provider = create_universe_provider(settings)
        data_manager = DataManager(universe_provider=univ_provider)
        market_df = _execute_stage("数据同步", lambda: data_manager.sync_and_build_dataset(force_update=force_update), retries=3, retry_delay=10.0)

        # 2. 计算特征与中性化 (本地计算, 重试 1 次)
        print("[2/4] 计算 Alpha158 + A股定制 + 另类因子并执行截面中性化...")
        processor = FactorProcessor()
        factor_df = _execute_stage("因子计算", lambda: processor.build_and_save_factor_matrix(market_df, force_update=force_update), retries=1)
        labeler = TargetLabeler(horizon=settings.LABEL_HORIZON)
        factor_df = _execute_stage("标签生成", lambda: labeler.compute_excess_return_label(factor_df, canonical_dates=data_manager.get_trading_calendar()), retries=1)

        # 3. 走步训练与预测 (重训阶段不自动重试——同数据重试无意义, 失败直接告警)
        print("[3/4] 执行走步预测提取最新截面打分...")
        trainer = WalkForwardTrainer()
        oos_df, latest_model = _execute_stage("走步训练", lambda: trainer.run_walk_forward(factor_df), retries=0)

        latest_date = oos_df["date"].max()
        daily_df = oos_df[oos_df["date"] == latest_date].copy()
        expected_exec_date = data_manager.get_next_trading_date(latest_date)
        exec_str = expected_exec_date.strftime("%Y-%m-%d") if expected_exec_date else "次一交易日"

        # 4. 组合优化与生成目标仓位 (Phase A 修复: 传入真实持仓, 激活滞回缓冲)
        print(f"[4/4] 运行组合优化器 ({optimizer_type}) 生成次日买卖决策...")
        builder = PortfolioBuilder(
            top_k_buy=settings.TOP_K_BUY,
            top_k_hold=settings.TOP_K_HOLD,
            weight_method=optimizer_type,
            universe_provider=univ_provider
        )
        top_df = _execute_stage(
            "组合构建",
            lambda: builder.build_target_portfolio(daily_df, current_holdings=holdings, date=latest_date),
            retries=1
        )

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
            try:
                if channel == "feishu":
                    MessageNotifier.send_feishu_card(webhook_url, f"A股量化决策报告 ({sig_str})", report_md)
                elif channel == "wechat":
                    MessageNotifier.send_wechat_work(webhook_url, report_md)
                elif channel == "dingtalk":
                    MessageNotifier.send_dingtalk(webhook_url, f"A股量化决策报告 ({sig_str})", report_md)
                print("✅ 消息分发成功！")
            except Exception as push_err:
                # 推送失败不应丢弃已生成的决策结果, 但必须显式告警
                logger.error(f"❌ 决策报告推送失败: {push_err} — 决策结果已生成, 请人工查看本地报告")

        return {
            "signal_date": sig_str,
            "execution_date": exec_str,
            "top_portfolio": top_df,
            "markdown_report": report_md
        }

    except Exception as stage_err:
        # 任何阶段最终失败: 推送告警后原样抛出 (让调度器/日志记录到非零退出)
        _push_failure_alert(webhook_url, channel, str(stage_err))
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A股盘后量化自动化任务")
    parser.add_argument("--webhook", type=str, default=None, help="机器人 Webhook 地址")
    parser.add_argument("--channel", type=str, default="feishu", choices=["feishu", "wechat", "dingtalk"])
    parser.add_argument("--optimizer", type=str, default="equal", help="组合优化器类型")
    parser.add_argument("--force-update", action="store_true", help="强制更新数据")
    parser.add_argument("--holdings", type=str, default=None, help="当前持仓标的, 逗号分隔 (如 600519.SH,000858.SZ); 不传则滞回缓冲不生效")
    args = parser.parse_args()

    holdings = {s.strip() for s in args.holdings.split(",") if s.strip()} if args.holdings else set()

    run_daily_automation(
        webhook_url=args.webhook,
        channel=args.channel,
        optimizer_type=args.optimizer,
        force_update=args.force_update,
        current_holdings=holdings
    )
