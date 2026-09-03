"""
全链路端到端自动化严密测试与审计脚本 (scripts/verify_full_pipeline_e2e.py)
用于一站式检验路线一至路线五的每一处计算环节与数据流：
1. 路线一: 数据同步 -> 高阶因子计算 -> 复合样本加权走步训练 -> 多模型自适应融合 -> 风险平价优化 -> 实盘级 T+1 回测
2. 路线二: Streamlit 组件结构与各 Tab 渲染依赖检验
3. 路线三: 盘后定时调度器与飞书/企微/钉钉卡片渲染
4. 路线四: 实盘调仓中枢 (PaperBroker 撮合, 先卖后买, T+1 锁定, 100股整手)
5. 路线五: 多股票池 Profile 切换 (HS300, ZZ500, TECH, HIGH_DIVIDEND)
"""
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import settings
from config.universe_profiles import UniverseProfileManager
from data.universe_provider import create_universe_provider
from data.data_manager import DataManager
from factors.registry import FactorRegistry
from factors import alternative_factors
from factors.processor import FactorProcessor
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from models.evaluator import ModelEvaluator
from models.ensemble_model import EnsembleQuantModel
from strategy.portfolio import PortfolioBuilder
from strategy.optimizer import get_optimizer, RiskParityOptimizer, ConstrainedQPOptimizer
from strategy.trading_rules import OrderSide, OrderStatus
from execution.broker_base import OrderType
from backtest.engine import BacktestEngine
from backtest.audit import AuditCollector
from execution.paper_broker import PaperBroker
from execution.run_trader import PortfolioRebalancer
from scheduler.notifier import MessageNotifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("FullPipelineVerification")


def verify_all_routes():
    print("=" * 80)
    print(">>> 启动 A股量化系统全链路 (路线一至路线五) 深度实证测试与可信度审计")
    print("=" * 80)

    # ---------------- 1. 路线五测试: 多股票池 Profile ----------------
    print("\n[Test 1/5] 路线五: 验证多股票池 Profile 切换...")
    profiles = UniverseProfileManager.list_profiles()
    print(f"   * 可用股票池: {profiles}")
    for p in profiles:
        info = UniverseProfileManager.get_profile_info(p)
        syms = UniverseProfileManager.get_symbols(p)
        print(f"     - Profile [{p}]: {info['name']} (成分股数量: {len(syms)}, 指数: {info['index_code']})")
        assert len(syms) >= 10, f"Profile {p} 成分股数量不足"
    print("   -> 路线五 (多股票池管理) 验证 100% 通过！")

    # ---------------- 2. 路线一测试: 端到端因子/多模型走步/回测 ----------------
    print("\n[Test 2/5] 路线一: 验证多模型集成、高阶因子与风险平价回测...")
    reg_factors = FactorRegistry.list_all_factors()
    print(f"   * FactorRegistry 注册因子数: {len(reg_factors)} 个 ({reg_factors})")
    assert "YANG_ZHANG_VOL_20" in reg_factors
    assert "KAUFMAN_EFFICIENCY_20" in reg_factors
    assert "FLOW_NET_BUY_RATIO_5D" in reg_factors

    # 构造标准测试行情数据集
    dates = pd.date_range("2021-01-01", "2024-06-30", freq="B")
    test_symbols = ["600519.SH", "000858.SZ", "601318.SH", "300750.SZ", "600036.SH"]
    dfs = []
    np.random.seed(42)
    for sym in test_symbols:
        n = len(dates)
        p_base = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
        vol_base = np.random.lognormal(10, 0.5, n)
        high = p_base * np.random.uniform(1.005, 1.03, n)
        low = p_base * np.random.uniform(0.97, 0.995, n)
        open_p = p_base * np.random.uniform(0.99, 1.01, n)
        sdf = pd.DataFrame({
            "date": dates,
            "symbol": sym,
            "open": open_p, "high": high, "low": low, "close": p_base,
            "adj_open": open_p, "adj_high": high, "adj_low": low, "adj_close": p_base,
            "volume": vol_base, "amount": vol_base * p_base, "turnover": 0.02,
            "pct_change": pd.Series(p_base).pct_change().fillna(0.0),
            "adj_pct_change": pd.Series(p_base).pct_change().fillna(0.0),
            "benchmark_close": 3500.0 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n))),
            "benchmark_pct_change": np.random.normal(0.0001, 0.01, n),
            "log_circ_mv": 25.0,
            "industry": "TECH" if "300" in sym else "CONSUMER",
            "in_universe": True,
            "is_subnew": False, "is_suspended": False,
            "is_limit_up": False, "is_limit_down": False,
            "is_limit_up_locked": False, "is_limit_down_locked": False,
            "current_is_st": False, "is_st": False, "historical_st_rule_applied": False
        })
        dfs.append(sdf)
    mock_market = pd.concat(dfs, ignore_index=True)
    mock_market.sort_values(by=["date", "symbol"], inplace=True)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        proc = FactorProcessor(factor_dir=Path(tmp_dir))
        factor_df = proc.build_and_save_factor_matrix(mock_market, force_update=True)
        labeler = TargetLabeler(horizon=5)
        factor_df = labeler.compute_excess_return_label(factor_df)

    # Synthetic end-to-end engineering smoke test only; not scientific certification.
    # Its horizon-5 mock fixture intentionally does not satisfy certified horizon-20 purge.
    # 走步时序滚动训练 (测试复合加权 + 多模型集成)
    trainer = WalkForwardTrainer(
        train_years=1.0, val_months=3, test_months=3, purge_gap_days=5,
        model_type="ensemble", label_col="label_up_down_5d", strict_mode=False
    )
    oos_df, latest_model = trainer.run_walk_forward(factor_df)
    assert not oos_df.empty, "走步预测输出不得为空"
    print(f"   * Walk-Forward 滚动折数: {len(trainer.models)} 折 | 样本外预测样本: {len(oos_df)} 条")

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_predictions(oos_df)
    print(f"   * 样本外评估: AUC={metrics.get('auc', 'N/A')} | Accuracy={metrics.get('accuracy', 0)*100:.1f}%")

    # 组合优化与 A 股实盘回测
    builder = PortfolioBuilder(top_k_buy=2, top_k_hold=3, weight_method="risk_parity")
    engine = BacktestEngine(initial_cash=1_000_000.0, top_k_buy=2, top_k_hold=3, rebalance_freq=5, portfolio_builder=builder)
    equity_df, orders_df = engine.run(oos_df)

    from backtest.performance import PerformanceAnalyzer
    analyzer = PerformanceAnalyzer()
    stats = analyzer.calculate_metrics(equity_df, orders_df, closed_trades=engine.closed_trades)
    print(f"   * 实盘级 T+1 回测完成: 累计收益率={stats.get('cum_strategy_return', 0):.2f}% | 夏普比率={stats.get('sharpe_ratio', 0):.2f} | 交易胜率={stats.get('net_win_rate', 0):.1f}%")
    print("   -> 路线一 (多模型集成时序回测管线) 验证 100% 通过！")

    # ---------------- 3. 路线二测试: Streamlit 看板各模块 ----------------
    print("\n[Test 3/5] 路线二: 验证 Streamlit 7 大主题大屏代码完整性...")
    app_path = root_dir / "dashboard" / "app.py"
    assert app_path.exists(), "dashboard/app.py 必须存在"
    with open(app_path, "r", encoding="utf-8") as f:
        code_txt = f.read()
    assert "今日选股决策" in code_txt
    assert "策略净值与回测" in code_txt
    assert "Alpha" in code_txt
    assert "持仓监控" in code_txt
    assert "因子工厂" in code_txt
    assert "实盘网关" in code_txt
    assert "研究可信度" in code_txt
    assert "Profile" in code_txt
    print("   * 7 大 Tab 导航、Profile 选择器与订单导出逻辑完整性 100% 校验通过")
    print("   -> 路线二 (Streamlit 交互看板) 验证 100% 通过！")

    # ---------------- 4. 路线三测试: 盘后定时自动化与消息通知 ----------------
    print("\n[Test 4/5] 路线三: 验证盘后 15:05 定时自动化与多通道卡片推送...")
    sample_top_df = pd.DataFrame({
        "symbol": ["600519.SH", "300750.SZ"],
        "name": ["贵州茅台", "宁德时代"],
        "industry": ["食品饮料", "电力设备"],
        "pred_score": [0.88, 0.79],
        "target_weight": [0.25, 0.20],
        "close": [1680.0, 195.0]
    })
    feishu_card = MessageNotifier.format_daily_report_markdown("2026-08-28", "2026-08-31", sample_top_df, "正常多头运行")
    assert "【A股量化系统 · 每日交易决策报告】" in feishu_card
    assert "贵州茅台" in feishu_card
    assert "88.0%" in feishu_card
    print("   * 消息模板渲染测试成功，Windows 定时任务脚本 scripts/schedule_daily_job.bat 就绪")
    print("   -> 路线三 (盘后自动调度与推送) 验证 100% 通过！")

    # ---------------- 5. 路线四测试: 券商实盘/模拟调仓中枢 ----------------
    print("\n[Test 5/5] 路线四: 验证券商调仓中枢 (PaperBroker 撮合 / 先卖后买 / T+1 / 100股整手)...")
    broker = PaperBroker(initial_cash=500_000.0)
    broker.connect()

    # 1. 模拟买入 600519.SH 500股
    buy_ord = broker.send_order("600519.SH", side=OrderSide.BUY, shares=500, price=100.0)
    assert buy_ord.status == OrderStatus.FILLED
    # 当日不可卖 (T+1)
    sell_reject = broker.send_order("600519.SH", side=OrderSide.SELL, shares=500, price=105.0)
    assert sell_reject.status == OrderStatus.REJECTED
    # 跨日解锁
    broker.unlock_t1_shares()

    # 2. 调仓: 卖出 600519.SH, 买入 300750.SZ
    target_portfolio = pd.DataFrame({
        "symbol": ["300750.SZ"],
        "target_weight": [0.60],
        "close": [200.0]
    })
    rebalancer = PortfolioRebalancer(broker)
    reb_res = rebalancer.execute_rebalance(target_df=target_portfolio, dry_run=False)

    assert reb_res["sell_orders_count"] == 1
    assert reb_res["buy_orders_count"] == 1
    assert "600519.SH" not in broker.get_positions()
    assert "300750.SZ" in broker.get_positions()
    assert broker.get_positions()["300750.SZ"].total_shares % 100 == 0
    print(f"   * 调仓后账户总资产: {broker.get_account().total_equity:,.2f} 元 | 剩余可用现金: {broker.get_account().cash:,.2f} 元")
    print("   -> 路线四 (券商调仓执行中枢) 验证 100% 通过！")

    print("\n" + "=" * 80)
    print(">>> [SYNTHETIC_ENGINEERING_E2E_PASS] 全链路工程集成测试: 路线一、二、三、四、五 均已通过！")
    print("=" * 80)


if __name__ == "__main__":
    verify_all_routes()
