"""
A股多因子股票预测与量化回测系统 - 一键自动化运行脚本 (run_pipeline.py)
支持从命令行执行全量化流程、启动 FastAPI 服务或启动 Streamlit 决策看板
"""
import sys
import io
import argparse
import logging
import subprocess
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import json
import pandas as pd

# 确保 UTF-8 输出
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 将根目录加入路径
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import settings
from data.universe_provider import create_universe_provider
from data.data_manager import DataManager
from factors.processor import FactorProcessor
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from models.evaluator import ModelEvaluator
from strategy.corporate_actions import create_corporate_action_provider
from strategy.portfolio import PortfolioBuilder
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer
from backtest.audit import AuditCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    force_update: bool = False,
    audit_json_path: Optional[str] = None,
    optimizer_type: str = "equal",
    model_type: str = "lightgbm",
    webhook_url: Optional[str] = None,
    channel: str = "feishu"
):
    """端到端执行全套量化预测与回测流水线"""
    print("\n" + "=" * 75)
    print(">> 启动 A股多因子机器学习预测与实盘回测流水线 (Enterprise Quant Pipeline)")
    print("=" * 75)

    # 1. 数据工程与 Parquet 缓存 (SecurityMaster 与真实上市日)
    print("\n[Step 1/5] 数据同步与清洗 (AKShare / SecurityMaster -> Parquet，双价格体系与真实交易日历)...")
    univ_provider = create_universe_provider(settings)
    data_manager = DataManager(universe_provider=univ_provider)

    try:
        market_df = data_manager.sync_and_build_dataset(force_update=force_update)
        if market_df is not None and not market_df.empty:
            act_start = pd.to_datetime(market_df["date"].min()).strftime("%Y-%m-%d")
            act_end = pd.to_datetime(market_df["date"].max()).strftime("%Y-%m-%d")
            data_manager.actual_backtest_start_date = act_start
            data_manager.actual_backtest_end_date = act_end
            data_manager.requested_backtest_start_date = settings.START_DATE
            data_manager.requested_backtest_end_date = settings.END_DATE
            if hasattr(univ_provider, "set_actual_backtest_window"):
                univ_provider.set_actual_backtest_window(act_start, act_end)
        print(f"   * 行情数据加载就绪: {len(market_df)} 条记录，覆盖 {len(market_df['symbol'].unique())} 只股票 (实际区间: {getattr(data_manager, 'actual_backtest_start_date', 'N/A')} -> {getattr(data_manager, 'actual_backtest_end_date', 'N/A')})")
        print(f"   * 上市日期覆盖率: {data_manager.listing_date_coverage_ratio*100:.1f}% | 行业覆盖率: {data_manager.industry_coverage_ratio*100:.1f}%")
        print(f"   * 数据源明细: {data_manager.data_source_breakdown}")
    except Exception as e:
        logger.warning(f"⚠️ 数据同步触发 Fail-Closed 拦截: {e}")
        if audit_json_path:
            out_p = Path(audit_json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fail_meta = AuditCollector.collect(data_manager=data_manager)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(fail_meta.to_dict(), f, ensure_ascii=False, indent=2)
            try:
                tool_p = root_dir / "tools" / "generate_audit_report.py"
                subprocess.run(
                    [sys.executable, str(tool_p),
                     "--pytest-xml", str(root_dir / "artifacts" / "pytest.xml"),
                     "--audit-json", str(out_p)],
                    check=True, capture_output=True, text=True, timeout=120
                )
            except Exception as re_err:
                logger.warning(f"报告生成异常: {re_err}")
        print(f"\n[Fail-Closed 保护已激活] 缺少生产证据或数据拉取受阻，已生成诚实的 HIGH_RISK 审计评级报告。\n错误详情: {e}")
        return

    # 1.5 基本面财务因子注入 (质量/成长异源信号, 季度->日频 PIT 对齐)
    if getattr(settings, "ENABLE_FUNDAMENTALS", False):
        from data.fundamentals import FundamentalsProvider
        logger.info("注入基本面财务因子 (质量/成长)...")
        fund = FundamentalsProvider(delay_days=settings.FUNDAMENTAL_DELAY_DAYS)
        fund_daily = fund.build_daily_fundamental_matrix(market_df, start_year=settings.FUNDAMENTAL_START_YEAR)
        before_cols = set(market_df.columns)
        market_df = market_df.merge(fund_daily, on=["symbol", "date"], how="left")
        new_cols = [c for c in fund_daily.columns if c not in before_cols and c not in ("symbol", "date")]
        cov = fund_daily[new_cols].notna().mean().mean() * 100 if new_cols else 0.0
        print(f"   * 基本面因子注入完成: {len(new_cols)} 个 (覆盖率均值 {cov:.1f}%, 拉取 {fund.source_counts})")

    # 2. 特征工程与截面标准化/中性化 (逐日行业 + 市值)
    print("\n[Step 2/5] 计算 Alpha 因子并执行【逐日行业 + 市值截面中性化】...")
    processor = FactorProcessor()
    factor_df = processor.build_and_save_factor_matrix(market_df, force_update=force_update)
    
    labeler = TargetLabeler(horizon=settings.LABEL_HORIZON)
    factor_df = labeler.compute_excess_return_label(factor_df, canonical_dates=data_manager.get_trading_calendar())
    factor_cols = FactorProcessor.get_all_factor_cols()
    if settings.is_classification:
        print(f"   * 特征与标签构建完成: 共 {len(factor_cols)} 个因子，目标预测未来 {settings.LABEL_HORIZON} 日涨跌方向 (跑赢/跑输基准)")
    else:
        print(f"   * 特征与标签构建完成: 共 {len(factor_cols)} 个因子，目标预测未来 {settings.LABEL_HORIZON} 日超额收益")
    print(f"   * 逐日行业中性化模式: {processor.industry_neutralization_enabled} (执行天数占比: {processor.industry_neutralized_day_ratio*100:.1f}%)")

    # 3. 走步滚动训练 (Walk-Forward + Purged Gap)
    print(f"\n[Step 3/5] 执行严格 Walk-Forward 滚动时序训练 (模型类型: {model_type}, 含 Purged Gap 隔离，杜绝未来函数)...")
    trainer = WalkForwardTrainer(model_type=model_type)
    oos_df, latest_model = trainer.run_walk_forward(factor_df)
    
    evaluator = ModelEvaluator()
    eval_metrics = evaluator.evaluate_predictions(oos_df)
    if settings.is_classification:
        print(
            f"   * OOS 分类评估结果: AUC = {eval_metrics['auc']:.4f} | "
            f"Accuracy = {eval_metrics['accuracy']*100:.2f}% | "
            f"Precision = {eval_metrics['precision']*100:.2f}% | "
            f"Recall = {eval_metrics['recall']*100:.2f}% | "
            f"F1 = {eval_metrics['f1']:.4f} | "
            f"Brier = {eval_metrics['brier_score']:.4f} | "
            f"上涨样本占比 = {eval_metrics['positive_rate']*100:.1f}%"
        )
    else:
        print(
            f"   * OOS 评估结果: Mean RankIC = {eval_metrics['rank_ic_mean']:+.4f} | "
            f"RankICIR = {eval_metrics['rank_icir']:.4f} (Newey-West: {eval_metrics['rank_icir_newey_west']:.4f}) | "
            f"RankIC>0 胜率 = {eval_metrics['rank_ic_win_rate']}% | "
            f"20D 滚动 RankIC = {eval_metrics['rolling_rank_ic_20d']:+.4f}"
        )

    # 4. Top-K 选股与最新信号 (支持现代优化器)
    print(f"\n[Step 4/5] 提取最新交易日 Top-K 选股信号 (优化器模式: {optimizer_type})...")
    latest_date = oos_df["date"].max()
    daily_df = oos_df[oos_df["date"] == latest_date].copy()
    
    builder = PortfolioBuilder(
        top_k_buy=settings.TOP_K_BUY, 
        top_k_hold=settings.TOP_K_HOLD, 
        weight_method=optimizer_type,
        universe_provider=univ_provider
    )
    top_df = builder.build_target_portfolio(daily_df, current_holdings=set(), date=latest_date)
    
    expected_exec_date = data_manager.get_next_trading_date(latest_date)
    exec_str = expected_exec_date.strftime("%Y-%m-%d") if expected_exec_date else "无下一交易日"

    print(f"   * 信号日期 (T日收盘): {latest_date.strftime('%Y-%m-%d')} | 预期执行日期 (T+1 真实交易日): {exec_str}")
    print(f"   * 行业上限约束: {'已启用 (30%硬上限)' if builder.sector_cap_enabled else '已关闭'} | UNKNOWN行业权重: {builder.unknown_industry_weight*100:.1f}%")
    print(f"   * 推荐买入 Top-{len(top_df)} 标的:")
    for idx, row in top_df.iterrows():
        s_name = f"({row['name']})" if "name" in row and pd.notna(row["name"]) else ""
        st_tag = "[ST]" if bool(row.get("current_is_st", False)) else ""
        ind_str = f"[{row.get('industry', 'UNKNOWN')}]"
        if settings.is_classification:
            score_str = f"{settings.LABEL_HORIZON}日上涨概率: {row['pred_score']*100:.1f}%"
        else:
            score_str = f"{settings.LABEL_HORIZON}日预期超额: {row['pred_score']*100:+.2f}%"
        print(f"     [{idx+1}] {row['symbol']} {s_name} {st_tag} {ind_str:<8} | {score_str} | 目标权重: {row['target_weight']*100:.1f}% | 收盘价: {row['close']:.2f}元")

    # 5. A股实盘级走步回测 (T日信号 -> T+1日开盘撮合)
    print("\n[Step 5/5] 执行 A股实盘级走步回测 (T+1 / 历史税费 / 流动性容量 / FIFO净胜率 / 公司行为)...")
    corp_provider = create_corporate_action_provider(settings)
    engine = BacktestEngine(
        initial_cash=settings.INITIAL_CASH,
        top_k_buy=settings.TOP_K_BUY,
        top_k_hold=settings.TOP_K_HOLD,
        rebalance_freq=settings.REBALANCE_FREQ,
        portfolio_builder=builder,
        corporate_actions=corp_provider
    )
    equity_df, orders_df = engine.run(oos_df)

    audit_obj = AuditCollector.collect(
        data_manager=data_manager,
        factor_processor=processor,
        portfolio_builder=engine.builder,
        trainer=trainer,
        engine=engine
    )

    if audit_json_path:
        out_p = Path(audit_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(audit_obj.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"   * 运行审计元数据已导出至: {out_p}")
        # 基于真实运行产物自动刷新 CAPABILITY / RUNTIME_ATTESTATION / MASTER 三份认证报告
        try:
            tool_p = root_dir / "tools" / "generate_audit_report.py"
            subprocess.run(
                [sys.executable, str(tool_p),
                 "--pytest-xml", str(root_dir / "artifacts" / "pytest.xml"),
                 "--audit-json", str(out_p)],
                check=True, capture_output=True, text=True, timeout=120
            )
            print(f"   * 认证报告已刷新: RUNTIME_ATTESTATION.md 评级 <- 实际运行审计")
        except Exception as e:
            logger.warning(f"认证报告刷新失败 (不影响回测结果): {e}")

    analyzer = PerformanceAnalyzer()
    perf = analyzer.calculate_metrics(equity_df, orders_df, closed_trades=engine.closed_trades, audit_info=audit_obj)
    audit = perf.get("audit_metadata", {})

    # 回测产物持久化: 指标JSON / 净值CSV / 订单CSV / 月度热力图PNG -> reports/
    saved_paths = analyzer.save_reports(equity_df, orders_df, perf)
    if saved_paths:
        print(f"\n   * 回测产物已落盘 ({len(saved_paths)} 个文件):")
        for label, pth in saved_paths.items():
            print(f"     - {label:<26}: {pth}")

    print("\n" + "=" * 75)
    print("📊 量化策略绩效回测总览 (Performance Summary - T+1 Open撮合)")
    print("=" * 75)
    print(f"  • 策略累计收益率   : {perf['cum_strategy_return']:+.2f}% (基准沪深300: {perf['cum_benchmark_return']:+.2f}%)")
    print(f"  • 超额累计收益 (Alpha): {perf['excess_return']:+.2f}%")
    print(f"  • 年化复合收益率 (CAGR): {perf['cagr']:+.2f}% (基准年化: {perf['benchmark_cagr']:+.2f}%)")
    print(f"  • 年化波动率       : {perf['annualized_volatility']:.2f}%")
    print(f"  • 夏普比率 (Sharpe) : {perf['sharpe_ratio']:.2f}")
    print(f"  • 索提诺比率 (Sortino): {perf['sortino_ratio']:.2f}")
    print(f"  • 最大回撤 (Max DD) : {perf['max_drawdown']:.2f}%")
    print(f"  • 卡玛比率 (Calmar) : {perf['calmar_ratio']:.2f}")
    print(f"  • 年化换手率 (Turnover): {perf['annualized_turnover']:.2f}x")
    print(f"  • 平均持仓天数     : {perf['average_holding_days']:.1f} 交易日 (FIFO 真实买卖配对)")
    print(f"  • 净胜率 (Net Win Rate) : {perf['net_win_rate']:.2f}% (扣除历史印花税、过户费与佣金后)")
    print(f"  • 毛胜率 (Gross Win Rate): {perf['gross_win_rate']:.2f}% (仅原始价差)")
    print(f"  • 净盈亏比 (Net P/L)    : {perf['profit_loss_ratio']:.2f}")
    print(f"  • 累计完成订单笔数 : {perf['total_trades']} 笔 (撤单: {audit.get('cancelled_order_count', 0)} 笔)")
    print(f"  • 累计摩擦成本统计 : {perf['total_costs']:,.2f} 元 (印花税: {perf['total_stamp_tax']:.2f}, 过户费: {perf.get('total_transfer_fee', 0):.2f}, 佣金: {perf['total_commission']:.2f}, 滑点: {perf['total_slippage']:.2f})")
    print("-" * 75)
    print(f"  • 数据与规则真实性审计 (Audit Metadata):")
    print(f"    - 数据来源明细  : {audit.get('data_source_breakdown')} (主要: {audit.get('data_source')})")
    print(f"    - 交易日历来源  : {audit.get('calendar_source')} (交易所官方认证: {'是' if audit.get('calendar_is_exchange_official') else '否 (第三方)'})")
    print(f"    - 逐日中性化模式: {audit.get('industry_neutralization_enabled')} (覆盖率均值: {round((audit.get('industry_coverage_ratio_mean') or 0)*100, 1)}%)")
    print(f"    - 行业集中度上限: {'已严格执行 (30%硬上限)' if audit.get('sector_cap_enabled') else '已关闭'}")
    print(f"    - 历史 ST 状态  : {'支持历史逐日' if audit.get('historical_st_available') else '无历史逐日ST (历史回测未伪造ST)'}")
    print(f"    - 股票池模式    : {audit.get('universe_mode')} (幸存者偏差风险: {'⚠️ 存在 (STATIC)' if audit.get('survivorship_bias_risk') else '已消除'})")
    print(f"    - 停牌超期警告  : {audit.get('stale_price_warning_events', 0)} 次 (影响 {len(audit.get('stale_price_affected_symbols', []))} 只标的)")
    print(f"    - 综合可信度评级: {audit.get('overall_backtest_reliability')}")
    print("=" * 75)

    top_feats = latest_model.get_feature_importance(top_n=10)
    print("\n🏆 Top 10 核心有效 Alpha 因子 (Gain 增益贡献):")
    for _, f_row in top_feats.iterrows():
        print(f"   • {f_row['feature']:<22} : {f_row['importance_pct']:.2f}%")

    # 6. 推送每日通知
    if webhook_url:
        from scheduler.notifier import MessageNotifier
        sig_str = latest_date.strftime("%Y-%m-%d")
        report_md = MessageNotifier.format_daily_report_markdown(
            signal_date=sig_str,
            execution_date=exec_str,
            top_df=top_df,
            macro_status="正常多头持仓"
        )
        print(f"\n正在通过 {channel} Webhook 推送决策通知...")
        if channel == "feishu":
            MessageNotifier.send_feishu_card(webhook_url, f"A股量化决策报告 ({sig_str})", report_md)
        elif channel == "wechat":
            MessageNotifier.send_wechat_work(webhook_url, report_md)
        elif channel == "dingtalk":
            MessageNotifier.send_dingtalk(webhook_url, f"A股量化决策报告 ({sig_str})", report_md)
        print("✅ 消息推送完成！")

    print("\n🎉 全流程运行完毕！")
    print("👉 启动 API 服务: python run_pipeline.py --serve-api")
    print("👉 启动 Web 看板: streamlit run dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股多因子量化预测与回测系统")
    parser.add_argument("--force-update", action="store_true", help="强制重新下载数据与计算因子")
    parser.add_argument(
        "--audit-json", type=str,
        default=str(root_dir / "artifacts" / "runtime_audit.json"),
        help="导出运行时审计 JSON 文件路径 (默认 artifacts/runtime_audit.json, 传空串禁用)"
    )
    parser.add_argument("--optimizer", type=str, default="equal", choices=["equal", "inv_vol", "score_weighted", "risk_parity", "qp"], help="组合优化器类型 (equal/inv_vol/score_weighted/risk_parity/qp)")
    parser.add_argument("--model-type", type=str, default="lightgbm", choices=["lightgbm", "ensemble", "double_ensemble", "mlp"], help="预测模型类型 (lightgbm/ensemble/double_ensemble/mlp)")
    parser.add_argument("--universe-profile", type=str, default="HS300_CORE", choices=["HS300_CORE", "ZZ500_GROWTH", "TECH_INNOVATION", "HIGH_DIVIDEND"], help="选股股票池 Profile")
    parser.add_argument("--auto-tune", action="store_true", help="启动贝叶斯超参数自适应寻优 (Bayesian Optimization)")
    parser.add_argument("--webhook", type=str, default=None, help="机器人 Webhook 地址 (飞书/企微/钉钉)")
    parser.add_argument("--channel", type=str, default="feishu", choices=["feishu", "wechat", "dingtalk"], help="通知渠道")
    parser.add_argument("--serve-api", action="store_true", help="启动 FastAPI 后端服务")
    parser.add_argument("--serve-dashboard", action="store_true", help="启动 Streamlit 前端看板")
    parser.add_argument("--port", type=int, default=8000, help="服务端口号")
    args = parser.parse_args()

    if args.universe_profile:
        settings.set_universe_profile(args.universe_profile)

    if args.serve_api:
        import uvicorn
        print(f"正在启动 FastAPI 服务 (http://localhost:{args.port})...")
        uvicorn.run("server.app:app", host="0.0.0.0", port=args.port, reload=False)
    elif args.serve_dashboard:
        import subprocess
        print("正在启动 Streamlit 决策看板...")
        subprocess.run(["streamlit", "run", "dashboard/app.py"])
    else:
        run_pipeline(
            force_update=args.force_update,
            audit_json_path=args.audit_json,
            optimizer_type=args.optimizer,
            model_type=args.model_type,
            webhook_url=args.webhook,
            channel=args.channel
        )
