"""
小型端到端回测 Smoke Test 脚本 (tests/smoke_test.py)
快速验证全套数据工程、Alpha 因子计算 (逐日中性化)、Walk-Forward 走步训练、A股走步撮合回测与 Fail-Closed 审计元数据
"""
import sys
import io
import tempfile
from pathlib import Path

# 确保 UTF-8 输出 (仅直接运行脚本时生效，避免污染 pytest 的 capture 管道)
if __name__ == "__main__" and sys.platform.startswith("win"):
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import logging
import pandas as pd
import numpy as np

from config.settings import settings
from data.data_manager import DataManager, count_trading_days
from data.data_fetcher import DataFetcher
from data.security_master import SecurityMaster, StockMetadata
from factors.processor import FactorProcessor
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from models.evaluator import ModelEvaluator
from strategy.portfolio import PortfolioBuilder
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer
from backtest.audit import AuditCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def generate_mini_market_df(symbols, start_date="2021-01-01", end_date="2023-06-30"):
    """构造微型标准行情数据集 (4只股票, 真实行业分类)"""
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    dfs = []
    
    ind_map = {
        "600519.SH": "食品饮料",
        "000858.SZ": "食品饮料",
        "600036.SH": "银行",
        "601318.SH": "非银金融"
    }

    for sym in symbols:
        np.random.seed(abs(hash(sym)) % (2**32))
        n = len(dates)
        rets = np.random.normal(0.0003, 0.018, n)
        prices = 100.0 * np.exp(np.cumsum(rets))
        
        df = pd.DataFrame({
            "date": dates,
            "symbol": sym,
            "name": sym.split(".")[0],
            "industry": ind_map.get(sym, "UNKNOWN"),
            "board": "主板",
            "list_date": "2010-01-01",
            "current_is_st": False,
            "is_st": False,
            "historical_st_rule_applied": False,
            "is_subnew": False,
            "is_suspended": False,
            "in_universe": True,
            "is_limit_up": False,
            "is_limit_down": False,
            "is_limit_up_locked": False,
            "is_limit_down_locked": False,
            "open": prices * (1 + np.random.normal(0, 0.002, n)),
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "adj_open": prices * (1 + np.random.normal(0, 0.002, n)),
            "adj_high": prices * 1.01,
            "adj_low": prices * 0.99,
            "adj_close": prices,
            "volume": np.random.lognormal(14, 0.5, n),
            "amount": np.random.lognormal(18, 0.5, n),
            "turnover": np.random.uniform(0.01, 0.05, n),
            "pct_change": rets,
            "adj_pct_change": rets,
            "benchmark_close": 4000.0 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n))),
            "benchmark_pct_change": np.random.normal(0.0001, 0.01, n),
            "log_circ_mv": 25.0 + np.random.normal(0, 0.2, n)
        })
        dfs.append(df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.sort_values(by=["date", "symbol"], inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    return full_df


def run_smoke_test():
    print("\n" + "=" * 70)
    print(">> 启动小型端到端回测 Smoke Test")
    print("=" * 70)

    symbols = ["600519.SH", "000858.SZ", "600036.SH", "601318.SH"]
    
    print("\n[Step 1/5] 构建微型数据集 (4只股票, 2.5年跨度)...")
    market_df = generate_mini_market_df(symbols)
    print(f">> 行情记录数: {len(market_df)}, 覆盖标的: {market_df['symbol'].unique()}")

    print("\n[Step 2/5] 特征工程与【逐日行业 + 市值中性化】...")
    # 必须隔离到临时目录: 直接使用生产 FACTORS_DIR 会覆盖正式因子矩阵缓存
    # (曾发生: 4 只股票的测试数据把 300 只股票的生产缓存污染成 4555 行)
    with tempfile.TemporaryDirectory() as tmp_dir:
        processor = FactorProcessor(factor_dir=Path(tmp_dir))
        factor_df = processor.build_and_save_factor_matrix(market_df, force_update=True)
        labeler = TargetLabeler(horizon=settings.LABEL_HORIZON)
        factor_df = labeler.compute_excess_return_label(factor_df)
    print(f">> 逐日行业中性化模式: {processor.industry_neutralization_enabled} | 覆盖率均值: {(processor.industry_coverage_ratio_mean or 0)*100:.1f}%")

    print("\n[Step 3/5] Walk-Forward 走步滚动训练与评估...")
    trainer = WalkForwardTrainer(train_years=2, val_months=3, test_months=1, purge_gap_days=settings.PURGE_GAP_DAYS)
    oos_df, model = trainer.run_walk_forward(factor_df)
    evaluator = ModelEvaluator()
    eval_metrics = evaluator.evaluate_predictions(oos_df)
    if settings.is_classification:
        print(f">> OOS 分类评估: AUC = {eval_metrics['auc']:.4f} | Accuracy = {eval_metrics['accuracy']*100:.2f}% | F1 = {eval_metrics['f1']:.4f} | Brier = {eval_metrics['brier_score']:.4f}")
    else:
        print(f">> OOS 样本外评估: Mean RankIC = {eval_metrics['rank_ic_mean']:+.4f} | RankIC>0 胜率 = {eval_metrics['rank_ic_win_rate']}%")

    print("\n[Step 4/5] A股实盘级走步回测 (T日信号 -> T+1日开盘撮合)...")
    engine = BacktestEngine(
        initial_cash=1000000.0,
        top_k_buy=2,
        top_k_hold=3,
        rebalance_freq=5
    )
    equity_df, orders_df = engine.run(oos_df)

    audit_obj = AuditCollector.collect(
        factor_processor=processor,
        portfolio_builder=engine.builder,
        engine=engine
    )

    analyzer = PerformanceAnalyzer()
    perf = analyzer.calculate_metrics(equity_df, orders_df, closed_trades=engine.closed_trades, audit_info=audit_obj)

    print("\n[Step 5/5] 回测绩效与真实性审计报告:")
    print("-" * 70)
    print(f"  * 策略累计收益: {perf['cum_strategy_return']:+.2f}% | 基准收益: {perf['cum_benchmark_return']:+.2f}%")
    print(f"  * 年化复合收益率 (CAGR): {perf['cagr']:+.2f}% | 夏普比率: {perf['sharpe_ratio']:.2f}")
    print(f"  * 净胜率 (Net Win Rate): {perf['net_win_rate']:.2f}% | 毛胜率: {perf['gross_win_rate']:.2f}%")
    print(f"  * 完成订单数: {perf['total_trades']} 笔 | FIFO平仓批次: {perf['closed_pair_trades']} 笔")
    print(f"  * 审计元数据 (audit_metadata):\n    {perf['audit_metadata']}")
    print("-" * 70)
    print(">> 端到端微型回测 SMOKE TEST 100% 成功通过！\n")


if __name__ == "__main__":
    run_smoke_test()
