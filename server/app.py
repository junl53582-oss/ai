"""
FastAPI 企业级量化服务 RESTful API 接口 (server/app.py)
提供数据同步、因子计算、Walk-Forward 模型训练、最新选股信号输出、自定义回测服务与 Fail-Closed 真实性审计 (AuditMetadata)
"""
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

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

app = FastAPI(
    title="A-Share Enterprise Quantitative Prediction & Backtest API",
    description="基于 Qlib Alpha158Subset + A股定制因子 + LightGBM 严格走步回测与真实性审计的量化决策系统接口",
    version="1.5.0"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局内存缓存
GLOBAL_CACHE: Dict[str, Any] = {
    "market_df": None,
    "factor_df": None,
    "oos_df": None,
    "latest_model": None,
    "eval_metrics": None,
    "equity_df": None,
    "orders_df": None,
    "performance_metrics": None,
    "factor_processor": None
}


# ---------------- Pydantic 请求模型 ----------------
class SyncDataRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, description="股票代码列表")
    benchmark_symbol: str = Field(default=settings.BENCHMARK_SYMBOL, description="基准指数代码")
    start_date: str = Field(default=settings.START_DATE, description="起始日期 YYYY-MM-DD")
    end_date: Optional[str] = Field(default=settings.END_DATE, description="结束日期 YYYY-MM-DD (None为最新日)")
    force_update: bool = Field(default=False, description="是否强制重新下载")


class TrainModelRequest(BaseModel):
    train_years: int = Field(default=settings.TRAIN_WINDOW_YEARS, description="训练集跨度 (年)")
    val_months: int = Field(default=settings.VAL_WINDOW_MONTHS, description="验证集跨度 (月)")
    test_months: int = Field(default=settings.TEST_WINDOW_MONTHS, description="测试集跨度 (月)")
    purge_gap_days: int = Field(default=settings.PURGE_GAP_DAYS, description="验证与测试集 Purge 隔离天数")


class BacktestRunRequest(BaseModel):
    initial_cash: float = Field(default=settings.INITIAL_CASH, description="初始资金 (元)")
    top_k_buy: Optional[int] = Field(default=None, description="买入截面排名阈值 (Top K Buy)")
    top_k_hold: Optional[int] = Field(default=None, description="持仓缓冲区排名阈值 (Top K Hold)")
    top_k: Optional[int] = Field(default=None, description="兼容旧接口TopK参数")
    rebalance_freq: int = Field(default=settings.REBALANCE_FREQ, description="调仓周期 (交易日数)")
    stop_loss_pct: float = Field(default=settings.STOP_LOSS_PCT, description="个股硬止损阈值")
    trailing_stop_pct: float = Field(default=settings.TRAILING_STOP_PCT, description="跟踪止盈阈值")


# ---------------- API 端点 ----------------
@app.get("/", tags=["System"])
def root():
    """服务健康检查"""
    return {
        "status": "online",
        "service": "A-Share Quant Engine",
        "version": "1.5.0",
        "cached_data_ready": GLOBAL_CACHE["market_df"] is not None,
        "model_trained": GLOBAL_CACHE["latest_model"] is not None
    }


@app.post("/api/v1/data/sync", tags=["Data Engine"])
def sync_data(req: SyncDataRequest):
    """1. 同步股票与基准行情并缓存为 Parquet"""
    try:
        manager = DataManager()
        df = manager.sync_and_build_dataset(
            symbols=req.symbols,
            benchmark_symbol=req.benchmark_symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            force_update=req.force_update
        )
        GLOBAL_CACHE["market_df"] = df
        return {
            "status": "success",
            "message": f"成功同步并缓存 {len(df['symbol'].unique())} 只股票的历史行情",
            "total_records": len(df),
            "date_range": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
            "listing_date_coverage_ratio": manager.listing_date_coverage_ratio,
            "industry_coverage_ratio": manager.industry_coverage_ratio,
            "data_source_breakdown": manager.data_source_breakdown
        }
    except Exception as e:
        logger.error(f"数据同步失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/factors/compute", tags=["Feature Engineering"])
def compute_factors(force_update: bool = False):
    """2. 计算 Qlib Alpha158Subset (46个) + A股定制因子 (13个) 并执行逐日截面中性化"""
    try:
        manager = DataManager()
        market_df = GLOBAL_CACHE["market_df"]
        if market_df is None:
            market_df = manager.load_dataset()
            GLOBAL_CACHE["market_df"] = market_df

        processor = FactorProcessor()
        factor_df = processor.build_and_save_factor_matrix(market_df, force_update=force_update)
        
        cal = DataManager().get_trading_calendar()
        labeler = TargetLabeler()
        factor_df = labeler.compute_excess_return_label(factor_df, canonical_dates=cal)
        
        GLOBAL_CACHE["factor_df"] = factor_df
        GLOBAL_CACHE["factor_processor"] = processor
        factor_cols = FactorProcessor.get_all_factor_cols()
        
        return {
            "status": "success",
            "message": f"特征工程构建完成！共生成 {len(factor_cols)} 个 Alpha 因子",
            "total_factors": len(factor_cols),
            "industry_neutralization_enabled": processor.industry_neutralization_enabled,
            "industry_coverage_ratio_mean": processor.industry_coverage_ratio_mean,
            "industry_neutralized_day_ratio": processor.industry_neutralized_day_ratio,
            "factor_list": factor_cols[:15] + ["..."]
        }
    except Exception as e:
        logger.error(f"特征工程构建失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/model/train", tags=["Model & Training"])
def train_walk_forward(req: TrainModelRequest):
    """3. 启动严格 Walk-Forward 滚动时序训练并评估 IC/RankIC"""
    try:
        factor_df = GLOBAL_CACHE["factor_df"]
        if factor_df is None:
            processor = FactorProcessor()
            factor_df = processor.load_factor_matrix()
            cal = DataManager().get_trading_calendar()
            labeler = TargetLabeler()
            factor_df = labeler.compute_excess_return_label(factor_df, canonical_dates=cal)
            GLOBAL_CACHE["factor_df"] = factor_df
            GLOBAL_CACHE["factor_processor"] = processor

        trainer = WalkForwardTrainer(
            train_years=req.train_years,
            val_months=req.val_months,
            test_months=req.test_months,
            purge_gap_days=req.purge_gap_days
        )
        oos_df, latest_model = trainer.run_walk_forward(factor_df)
        
        evaluator = ModelEvaluator()
        eval_metrics = evaluator.evaluate_predictions(oos_df)
        top_features = latest_model.get_feature_importance(top_n=15).to_dict(orient="records")

        GLOBAL_CACHE["oos_df"] = oos_df
        GLOBAL_CACHE["latest_model"] = latest_model
        GLOBAL_CACHE["eval_metrics"] = eval_metrics

        if settings.is_classification:
            metrics_out = {
                "task_type": "classification",
                "auc": eval_metrics["auc"],
                "accuracy": eval_metrics["accuracy"],
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "f1": eval_metrics["f1"],
                "brier_score": eval_metrics["brier_score"],
                "log_loss": eval_metrics["log_loss"],
                "positive_rate": eval_metrics["positive_rate"],
                "confusion_matrix": eval_metrics["confusion_matrix"],
                "quantile_returns": eval_metrics["quantile_returns"]
            }
        else:
            metrics_out = {
                "task_type": "regression",
                "rank_ic_mean": eval_metrics["rank_ic_mean"],
                "icir": eval_metrics["icir"],
                "rank_icir": eval_metrics["rank_icir"],
                "rank_ic_win_rate": eval_metrics["rank_ic_win_rate"],
                "rolling_rank_ic_20d": eval_metrics["rolling_rank_ic_20d"],
                "total_evaluated_days": eval_metrics["total_evaluated_days"],
                "quantile_returns": eval_metrics["quantile_returns"]
            }

        return {
            "status": "success",
            "message": "Walk-Forward 走步训练完成",
            "metrics": metrics_out,
            "top_features": top_features
        }
    except Exception as e:
        logger.error(f"模型训练失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/signals/latest", tags=["Alpha Signals"])
def get_latest_signals(top_k_buy: Optional[int] = None, top_k_hold: Optional[int] = None):
    """4. 获取最新交易日 Top-K 选股池与调仓建议 (包含 Fail-Closed audit_metadata)"""
    try:
        oos_df = GLOBAL_CACHE["oos_df"]
        if oos_df is None:
            raise ValueError("尚未生成预测数据，请先执行 /api/v1/model/train 接口！")

        latest_date = oos_df["date"].max()
        daily_df = oos_df[oos_df["date"] == latest_date].copy()

        k_buy = top_k_buy or settings.TOP_K_BUY
        k_hold = top_k_hold or settings.TOP_K_HOLD

        builder = PortfolioBuilder(top_k_buy=k_buy, top_k_hold=k_hold)
        top_df = builder.build_target_portfolio(daily_df, current_holdings=set(), date=latest_date)

        expected_exec_date = DataManager().get_next_trading_date(latest_date)
        exec_str = expected_exec_date.strftime("%Y-%m-%d") if expected_exec_date else "无下一交易日"

        signals = []
        for idx, row in top_df.iterrows():
            signal = {
                "rank": idx + 1,
                "symbol": row["symbol"],
                "name": row.get("name", "未知"),
                "industry": row.get("industry", "UNKNOWN"),
                "is_st": bool(row.get("current_is_st", False)),
                "target_weight_pct": round(float(row["target_weight"]) * 100, 2),
                "close_price": round(float(row["close"]), 2)
            }
            if settings.is_classification:
                signal["up_probability_pct"] = round(float(row["pred_score"]) * 100, 2)
                signal["predicted_direction"] = "涨(跑赢基准)" if float(row["pred_score"]) >= 0.5 else "跌(跑输基准)"
            else:
                signal[f"predicted_{settings.LABEL_HORIZON}d_excess_pct"] = round(float(row["pred_score"]) * 100, 2)
            signals.append(signal)

        audit_meta = {
            "survivorship_bias_risk": True,
            "historical_st_bias_risk": True,
            "industry_neutralization_enabled": "DISABLED"
        }
        if GLOBAL_CACHE.get("factor_processor") is not None:
            fp = GLOBAL_CACHE["factor_processor"]
            audit_meta["industry_neutralization_enabled"] = fp.industry_neutralization_enabled

        return {
            "status": "success",
            "signal_date": latest_date.strftime("%Y-%m-%d"),
            "expected_execution_date": exec_str,
            "sector_cap_enabled": builder.sector_cap_enabled,
            "audit_metadata": audit_meta,
            "execution_note": "信号产生于收盘后，预计于下一真实交易日开盘 (Open) 执行撮合",
            "top_candidates_count": len(signals),
            "recommendations": signals
        }
    except Exception as e:
        logger.error(f"获取最新信号失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/backtest/run", tags=["Backtest & Execution"])
def run_backtest(req: BacktestRunRequest):
    """5. 执行 A股实盘级全仿真走步回测 (返回标准金融指标与 Fail-Closed 审计元数据)"""
    try:
        oos_df = GLOBAL_CACHE["oos_df"]
        if oos_df is None:
            raise ValueError("尚未生成预测数据，请先执行 /api/v1/model/train 接口！")

        k_buy = req.top_k_buy or req.top_k or settings.TOP_K_BUY
        k_hold = req.top_k_hold or (k_buy * 2) or settings.TOP_K_HOLD

        manager = DataManager()
        market_df = GLOBAL_CACHE["market_df"]
        if market_df is not None:
            manager._compute_coverage_stats(market_df)

        processor = GLOBAL_CACHE.get("factor_processor") or FactorProcessor()
        corp_provider = create_corporate_action_provider(settings)

        engine = BacktestEngine(
            initial_cash=req.initial_cash,
            top_k_buy=k_buy,
            top_k_hold=k_hold,
            rebalance_freq=req.rebalance_freq,
            corporate_actions=corp_provider
        )
        equity_df, orders_df = engine.run(oos_df)

        audit_obj = AuditCollector.collect(
            data_manager=manager,
            factor_processor=processor,
            portfolio_builder=engine.builder,
            engine=engine
        )

        analyzer = PerformanceAnalyzer()
        perf_metrics = analyzer.calculate_metrics(
            equity_df,
            orders_df,
            closed_trades=engine.closed_trades,
            audit_info=audit_obj
        )

        GLOBAL_CACHE["equity_df"] = equity_df
        GLOBAL_CACHE["orders_df"] = orders_df
        GLOBAL_CACHE["performance_metrics"] = perf_metrics

        return {
            "status": "success",
            "performance_summary": {k: v for k, v in perf_metrics.items() if k != "monthly_table"},
            "total_orders": len(orders_df),
            "closed_pair_trades": len(engine.closed_trades),
            "sample_orders": orders_df.head(10).to_dict(orient="records"),
            "audit_metadata": perf_metrics.get("audit_metadata")
        }
    except Exception as e:
        logger.error(f"回测执行失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/performance/summary", tags=["Performance"])
def get_performance_summary():
    """6. 获取回测净值曲线数据与详细指标"""
    equity_df = GLOBAL_CACHE["equity_df"]
    perf_metrics = GLOBAL_CACHE["performance_metrics"]
    if equity_df is None or perf_metrics is None:
        raise HTTPException(status_code=400, detail="暂无回测结果，请先调用 /api/v1/backtest/run")

    curve_data = equity_df[["date", "cum_strategy_return", "cum_benchmark_return", "total_equity", "benchmark_equity"]].copy()
    curve_data["date"] = curve_data["date"].dt.strftime("%Y-%m-%d")

    return {
        "metrics": perf_metrics,
        "equity_curve": curve_data.to_dict(orient="records")
    }


@app.get("/api/v1/risk/regime", tags=["Risk Management"])
def get_market_regime():
    """7. 获取实时宏观市场状态、波动率目标与风控建议"""
    from strategy.risk_manager import MarketRegimeDetector, DynamicDrawdownController
    market_df = GLOBAL_CACHE.get("market_df")
    equity_df = GLOBAL_CACHE.get("equity_df")

    if market_df is not None and "benchmark_close" in market_df.columns:
        bench_s = market_df.groupby("date")["benchmark_close"].first()
        regime_info = MarketRegimeDetector.detect_regime(bench_s)
    else:
        regime_info = {"regime": "未知 (未加载数据)", "recommended_gross_exposure": 0.8}

    if equity_df is not None and "total_equity" in equity_df.columns:
        dd_limit, dd_reason = DynamicDrawdownController.evaluate_drawdown_exposure_limit(equity_df["total_equity"])
    else:
        dd_limit, dd_reason = 1.0, "无回撤数据"

    return {
        "market_regime": regime_info,
        "drawdown_control": {
            "allowed_max_exposure": dd_limit,
            "status": dd_reason
        }
    }


@app.get("/api/v1/attribution/latest", tags=["Factor Attribution"])
def get_factor_attribution():
    """8. 获取当前投资组合 Barra 风格因子暴露与收益归因分解"""
    from factors.attribution import BarraFactorAttribution
    equity_df = GLOBAL_CACHE.get("equity_df")
    factor_df = GLOBAL_CACHE.get("factor_df")
    oos_df = GLOBAL_CACHE.get("oos_df")

    decomp = {}
    if equity_df is not None and len(equity_df) > 10:
        p_ret = equity_df["total_equity"].pct_change().dropna()
        b_ret = equity_df["benchmark_equity"].pct_change().dropna()
        decomp = BarraFactorAttribution.decompose_returns(p_ret, b_ret)

    exposures = {}
    if factor_df is not None and oos_df is not None:
        latest_d = oos_df["date"].max()
        top_slice = oos_df[oos_df["date"] == latest_d].head(settings.TOP_K_BUY)
        exposures = BarraFactorAttribution.compute_portfolio_style_exposure(top_slice, factor_df)

    return {
        "macro_return_decomposition": decomp,
        "barra_style_exposures": exposures
    }
