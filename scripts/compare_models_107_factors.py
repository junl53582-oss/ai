"""
107 因子全量多模型时序 Walk-Forward 实证对比评估 (scripts/compare_models_107_factors.py)
在 462,844 行标准面板上对比:
1. LightGBM (经典基线)
2. DoubleEnsemble (微软 Qlib 样本损失重加权 + 特征子空间扰动)
3. 风险平价组合优化与全量实盘级 T+1 撮合
"""
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# 根目录引用
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import settings
from data.data_manager import DataManager
from data.universe_provider import create_universe_provider
from factors.processor import FactorProcessor
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from models.evaluator import ModelEvaluator
from strategy.portfolio import PortfolioBuilder
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def run_model_experiment(model_type: str, factor_df: pd.DataFrame) -> dict:
    logger.info(f"\n=======================================================")
    logger.info(f">> 启动模型时序 Walk-Forward 滚动实测: {model_type}")
    logger.info(f"=======================================================")

    trainer = WalkForwardTrainer(
        train_years=settings.TRAIN_WINDOW_YEARS,
        val_months=settings.VAL_WINDOW_MONTHS,
        test_months=settings.TEST_WINDOW_MONTHS,
        purge_gap_days=settings.PURGE_GAP_DAYS,
        model_type=model_type
    )

    oos_df, latest_model = trainer.run_walk_forward(factor_df)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_predictions(oos_df)
    logger.info(f"[{model_type}] 样本外评估: AUC={metrics.get('auc', 0):.4f} | Accuracy={metrics.get('accuracy', 0)*100:.2f}% | Brier={metrics.get('brier_score', 0):.4f}")

    # A股实盘级 T+1 回测 (风险平价优化 + 迟滞缓冲区)
    builder = PortfolioBuilder(top_k_buy=settings.TOP_K_BUY, top_k_hold=settings.TOP_K_HOLD, weight_method="risk_parity")
    engine = BacktestEngine(
        initial_cash=settings.INITIAL_CASH,
        top_k_buy=settings.TOP_K_BUY,
        top_k_hold=settings.TOP_K_HOLD,
        rebalance_freq=settings.REBALANCE_FREQ,
        portfolio_builder=builder
    )
    equity_df, orders_df = engine.run(oos_df)

    analyzer = PerformanceAnalyzer()
    stats = analyzer.calculate_metrics(equity_df, orders_df, closed_trades=engine.closed_trades)
    logger.info(f"[{model_type}] 回测完成: 累计收益率={stats.get('cum_strategy_return', 0):.2f}% | 胜率={stats.get('net_win_rate', 0):.1f}% | 夏普={stats.get('sharpe_ratio', 0):.2f}")

    return {
        "model_type": model_type,
        "auc": metrics.get("auc", 0),
        "accuracy": metrics.get("accuracy", 0),
        "brier_score": metrics.get("brier_score", 0),
        "cum_strategy_return": stats.get("cum_strategy_return", 0),
        "cum_benchmark_return": stats.get("cum_benchmark_return", 0),
        "excess_return": stats.get("excess_return", 0),
        "cagr": stats.get("cagr", 0),
        "annualized_volatility": stats.get("annualized_volatility", 0),
        "max_drawdown": stats.get("max_drawdown", 0),
        "sharpe_ratio": stats.get("sharpe_ratio", 0),
        "net_win_rate": stats.get("net_win_rate", 0),
        "profit_loss_ratio": stats.get("profit_loss_ratio", 0),
        "total_trades": stats.get("total_trades", 0),
        "total_costs": stats.get("total_costs", 0)
    }


def main():
    logger.info(">> 加载 462,844 行 × 107 因子正式生产矩阵...")
    processor = FactorProcessor()
    factor_df = processor.load_factor_matrix()

    dm = DataManager(universe_provider=create_universe_provider(settings))
    labeler = TargetLabeler(horizon=settings.LABEL_HORIZON)
    factor_df = labeler.compute_excess_return_label(factor_df, canonical_dates=dm.get_trading_calendar())

    logger.info(f"因子矩阵加载完成: {len(factor_df)} 行 × {len(factor_df.columns)} 列")

    # 1. 运行 LightGBM
    lgb_results = run_model_experiment("lightgbm", factor_df)

    # 2. 运行 DoubleEnsemble
    de_results = run_model_experiment("double_ensemble", factor_df)

    comparison = {
        "dataset_rows": len(factor_df),
        "factor_count": len(FactorProcessor.get_all_factor_cols()),
        "models": {
            "lightgbm": lgb_results,
            "double_ensemble": de_results
        }
    }

    out_file = settings.REPORTS_DIR / "model_comparison_107_factors.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    logger.info(f"\n>> 实证对比完成！结果已保存至: {out_file}")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
