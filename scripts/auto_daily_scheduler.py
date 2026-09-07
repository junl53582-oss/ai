"""
盘后全自动无人值守调度引擎 (scripts/auto_daily_scheduler.py)
用于支持 Direction 3:
1. 交易日检查: 智能判断 A 股交易日 (周一至周五，规避非交易时段)
2. 行情与资讯拉取: 盘后 (15:05) 自动直连官方 CDN 获取最新收盘量价与 7x24 财经消息
3. 选股模型运行: 刷新沪深 300 与中证 500 最新预测清单与目标配置权重
4. 模拟对账更新: 驱动 PaperTradingLedger 自动进行收盘估值结算、模拟调仓与净值记账
5. 支持 --run-now (立即触发一次) 与 --daemon (后台定时守护)
"""

import sys
import time
import json
import logging
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Union

# 根目录引用
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config.settings import settings
from data.trading_calendar import CanonicalTradingCalendar
from data.live_market_and_news_api import AutoSyncEngine
from data.universe_csi500 import CSI500UniverseManager
from execution.paper_ledger import PaperTradingLedger, ShadowTradingLedger

logger = logging.getLogger("AutoScheduler")

def _setup_scheduler_logging():
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        log_file = settings.ARTIFACTS_DIR / "auto_daily_scheduler.log"
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        handlers=handlers
    )

_setup_scheduler_logging()


class DailyExecutionPipeline:
    """每日端到端自动化调度流水线"""

    @staticmethod
    def is_trading_day(dt: Optional[Union[date, datetime, str]] = None) -> bool:
        """检查指定日期是否为交易所法定交易日 (CanonicalTradingCalendar)"""
        cal = CanonicalTradingCalendar.get_instance()
        if dt is None:
            check_str = date.today().strftime("%Y-%m-%d")
        elif isinstance(dt, (date, datetime)):
            check_str = dt.strftime("%Y-%m-%d")
        else:
            check_str = str(dt)[:10]
        return cal.is_trading_day(check_str)

    @classmethod
    def execute_post_market_pipeline(
        cls,
        sync_online: bool = True,
        ledger_file: Optional[Union[str, Path]] = None,
        shadow_ledger_file: Optional[Union[str, Path]] = None,
        status_file: Optional[Union[str, Path]] = None,
        execution_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行盘后全套跑批任务"""
        start_t = time.time()
        now_dt = datetime.now()
        date_str = execution_date if execution_date else now_dt.strftime("%Y-%m-%d")
        time_str = now_dt.strftime("%H:%M:%S")

        is_trading = cls.is_trading_day(date_str)

        logger.info("=================================================================")
        logger.info(f"  启动盘后全自动跑批流水线 | 日期: {date_str} 时间: {time_str} | 交易日: {is_trading}")
        logger.info("=================================================================")

        results = {
            "date": date_str,
            "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "is_trading_day": is_trading,
            "steps": {}
        }

        # -------------------------------------------------------------
        # 步骤 1: 行情与快讯同步 (沪深300 + 中证500)
        # -------------------------------------------------------------
        logger.info(">> [Step 1] 抓取并同步全市场最新收盘量价与舆情消息...")
        picks_csi300 = settings.ARTIFACTS_DIR / "latest_stock_picks.csv"
        picks_csi500 = settings.ARTIFACTS_DIR / "csi500_stock_picks.csv"

        if not picks_csi500.exists():
            CSI500UniverseManager.generate_csi500_picks_file(picks_csi500)

        if sync_online:
            try:
                AutoSyncEngine.sync_picks_and_news(picks_csi300)
                AutoSyncEngine.sync_picks_and_news(picks_csi500)
                results["steps"]["sync_quotes"] = "SUCCESS"
                logger.info("[+] 沪深300 与 中证500 最新行情消息同步完毕")
            except Exception as e:
                logger.error(f"[-] 行情同步异常: {e}")
                results["steps"]["sync_quotes"] = f"ERROR: {e}"
        else:
            results["steps"]["sync_quotes"] = "SKIPPED"

        # -------------------------------------------------------------
        # 步骤 2: 读取最新 Top-K 选股决策
        # -------------------------------------------------------------
        logger.info(">> [Step 2] 提取最新优选持仓清单...")
        import pandas as pd
        if picks_csi300.exists():
            top_df = pd.read_csv(picks_csi300)
            picks_count = len(top_df[top_df.get("target_weight", 0) > 0])
            results["steps"]["load_picks"] = f"LOADED {len(top_df)} (Target Buys: {picks_count})"
        else:
            top_df = pd.DataFrame()
            results["steps"]["load_picks"] = "NO_FILE"

        # -------------------------------------------------------------
        # 步骤 3 & 4: 封存前瞻信号与执行观察撮合 (严格通过状态机驱动 Paper/Shadow)
        # -------------------------------------------------------------
        logger.info(">> [Step 3 & 4] 驱动前瞻状态机 (封存信号并撮合前一交易日观察)...")
        ledger = PaperTradingLedger(ledger_file=Path(ledger_file) if ledger_file else None)
        shadow = ShadowTradingLedger(ledger_file=Path(shadow_ledger_file) if shadow_ledger_file else None)
        cal = CanonicalTradingCalendar.get_instance()

        if is_trading:
            from models.registry import ModelRegistry
            try:
                reg = ModelRegistry()
                prod_m = reg.get_production()
                active_mid = prod_m.model_id if prod_m else "PRODUCTION_ACTIVE"
            except Exception:
                active_mid = "PRODUCTION_ACTIVE"

            from execution.prospective_state_machine import seal_signal, execute_observed
            sig_store_dir = Path(shadow_ledger_file).parent / "prospective_signals" if shadow_ledger_file else None

            # 4.1 当日信号封存 (SIGNAL_SEALED)
            if not top_df.empty:
                try:
                    if date_str <= "2026-08-24":
                        seal_rec = {
                            "status": "CALENDAR_COVERAGE_BLOCKED",
                            "reason": f"禁止回填历史日期 ({date_str} <= 2026-08-24) 冒充 prospective 信号",
                            "observed_trading_days_added": 0,
                        }
                    else:
                        seal_rec = seal_signal(
                            signal_date=date_str,
                            picks_df=top_df,
                            model_id=active_mid,
                            factor_matrix_path=settings.DATA_DIR / "features" / "factor_matrix_latest.parquet",
                            storage_dir=sig_store_dir,
                        )
                    results["steps"]["seal_signal"] = seal_rec
                except Exception as seal_err:
                    logger.warning(f"封存前瞻信号告警: {seal_err}")
                    results["steps"]["seal_signal"] = {"error": str(seal_err)}
            else:
                results["steps"]["seal_signal"] = "SKIPPED_EMPTY_PICKS"

            # 4.2 观察执行撮合 (通过 execute_observed 同步驱动 Paper 与 Shadow，严禁旁路直接 rebalance)
            if date_str <= "2026-08-24":
                results["steps"]["rebalance"] = {
                    "status": "CALENDAR_COVERAGE_BLOCKED",
                    "reason": f"历史日期 ({date_str} <= 2026-08-24) 禁止修改正式模拟账本",
                }
                results["steps"]["shadow_observation"] = {
                    "status": "CALENDAR_COVERAGE_BLOCKED",
                    "reason": f"历史日期 ({date_str} <= 2026-08-24) 禁止冒充前瞻观察样本",
                    "observed_trading_days_added": 0,
                }
                logger.warning(f"⚠️ 历史日期阻断: {date_str} 禁止回填正式 Paper/Shadow 账本")
            elif not cal.is_trading_day(date_str) or cal.next_trading_day(date_str) is None:
                results["steps"]["rebalance"] = {
                    "status": "CALENDAR_COVERAGE_BLOCKED",
                    "reason": "超出物理日历覆盖范围",
                }
                results["steps"]["shadow_observation"] = {
                    "status": "CALENDAR_COVERAGE_BLOCKED",
                    "reason": "超出物理日历覆盖范围 (2026-08-24)",
                    "observed_trading_days_added": 0,
                }
                logger.warning(f"⚠️ 日历覆盖阻断: {date_str} 超出物理交易日历覆盖范围")
            else:
                prev_day = cal.prev_trading_day(date_str)
                if prev_day and sig_store_dir:
                    prev_signal_file = sig_store_dir / f"SIGNAL_SEALED_{prev_day}.json"
                    if not prev_signal_file.exists():
                        logger.warning(f"⚠️ 前一交易日 ({prev_day}) 信号未真实封存，跳过观察撮合")
                        results["steps"]["rebalance"] = {
                            "status": "PREV_SIGNAL_NOT_FOUND",
                            "reason": f"未找到前一交易日 ({prev_day}) 封存信号制品",
                        }
                        results["steps"]["shadow_observation"] = {
                            "status": "PREV_SIGNAL_NOT_FOUND",
                            "reason": f"未找到前一交易日 ({prev_day}) 的真实封存信号制品，拒绝伪造历史信号",
                            "observed_trading_days_added": 0,
                        }
                    else:
                        quotes_df = top_df.copy()
                        obs_payload = execute_observed(
                            signal_date=prev_day,
                            execution_date=date_str,
                            quotes_df=quotes_df,
                            paper_ledger=ledger,
                            shadow_ledger=shadow,
                            storage_dir=sig_store_dir,
                        )
                        results["steps"]["rebalance"] = {
                            "status": "EXECUTED_VIA_STATE_MACHINE",
                            "total_trades": len(ledger.trade_history),
                            "nav": ledger.nav,
                        }
                        results["steps"]["shadow_observation"] = {
                            "observed_days": shadow.observed_trading_days,
                            "evidence_maturity": shadow.evidence_maturity,
                            "status": shadow.status,
                            "evidence_sha256": obs_payload.get("file_sha256") or obs_payload.get("details", {}).get("shadow", {}).get("evidence_sha256"),
                            "observed_trading_days_added": obs_payload.get("observed_trading_days_added", 0),
                        }
                        logger.info(f"[+] 模拟盘与影子观察经由 execute_observed 完成: 累计观察 {shadow.observed_trading_days} 天")
                else:
                    results["steps"]["rebalance"] = {"status": "CALENDAR_COVERAGE_BLOCKED"}
                    results["steps"]["shadow_observation"] = {
                        "status": "CALENDAR_COVERAGE_BLOCKED",
                        "observed_trading_days_added": 0,
                    }
        else:
            logger.info(f"今日 ({date_str}) 非法定交易日，跳过状态机撮合")
            try:
                is_weekend = datetime.strptime(date_str, "%Y-%m-%d").weekday() >= 5
            except Exception:
                is_weekend = False

            if date_str > "2026-08-24" and not is_weekend:
                results["steps"]["rebalance"] = {
                    "status": "CALENDAR_COVERAGE_BLOCKED",
                    "reason": f"超出物理交易日历覆盖范围 ({date_str} > 2026-08-24)",
                }
                results["steps"]["shadow_observation"] = {
                    "status": "CALENDAR_COVERAGE_BLOCKED",
                    "reason": f"超出物理交易日历覆盖范围 ({date_str} > 2026-08-24)",
                    "observed_trading_days_added": 0,
                }
            else:
                results["steps"]["rebalance"] = "SKIPPED_NON_TRADING_DAY"
                results["steps"]["snapshot"] = "SKIPPED_NON_TRADING_DAY"
                results["steps"]["shadow_observation"] = "SKIPPED_NON_TRADING_DAY_OR_EMPTY"

        results["summary"] = {
            "total_equity": ledger.total_equity,
            "cash": ledger.cash,
            "market_value": ledger.market_value,
            "nav": ledger.nav,
            "cum_return_pct": ledger.cum_return_pct,
            "positions_count": len(ledger.positions)
        }

        cost_time = time.time() - start_t
        results["elapsed_seconds"] = round(cost_time, 2)
        logger.info("=================================================================")
        logger.info(f"  盘后跑批全部完成! 耗时: {cost_time:.2f}s | 当前模拟账户总资产: {ledger.total_equity:,.2f} 元 (净值: {ledger.nav:.4f})")
        logger.info("=================================================================")

        # 输出状态文件
        target_status_file = Path(status_file) if status_file else settings.ARTIFACTS_DIR / "auto_scheduler_last_run.json"
        target_status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_status_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results


def run_daemon_loop():
    """常驻后台守护循环：每个交易日 15:05:00 自动触发"""
    logger.info(">> 启动后台无人值守守护模式 (Daemon Mode). 正在监听交易日 15:05 触发窗口...")
    while True:
        now = datetime.now()
        # 判断是否为交易日且在 15:05~15:06 之间
        if DailyExecutionPipeline.is_trading_day(now.date()):
            if now.hour == 15 and now.minute == 5:
                logger.info("⏰ 命中交易日 15:05 触发时间，开始执行盘后流水线...")
                try:
                    DailyExecutionPipeline.execute_post_market_pipeline(sync_online=True)
                except Exception as e:
                    logger.error(f"跑批发生未捕获异常: {e}")
                # 休眠 70 秒避开当分钟重复触发
                time.sleep(70)
        time.sleep(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股量化每日盘后自动化调度引擎")
    parser.add_argument("--run-now", action="store_true", help="立即执行一次完整的盘后流水线")
    parser.add_argument("--daemon", action="store_true", help="以常驻后台守护进程模式运行")
    parser.add_argument("--no-sync", action="store_true", help="跳过联网行情抓取 (用于快速离线测试)")
    args = parser.parse_args()

    if args.run_now or (not args.daemon):
        sync = not args.no_sync
        res = DailyExecutionPipeline.execute_post_market_pipeline(sync_online=sync)
        print(f"\n[*] 调度执行成功: 账户资产 {res['summary']['total_equity']:,.2f} 元, 耗时 {res['elapsed_seconds']}s")
    elif args.daemon:
        run_daemon_loop()
