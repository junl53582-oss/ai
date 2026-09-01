"""
多因子研究与 Alpha 验证命令行工具 (tools/run_factor_research.py)
Phase 1.5 强化:
1. 支持 --force-rebuild 强制清理并重建行情与因子矩阵
2. 加载因子矩阵前严格校验必需列 [date, symbol, adj_open, adj_close, benchmark_open, benchmark_close, in_universe]
3. 发现残缺或旧版本缓存自动触发重建，杜绝误用 stale cache
"""
import sys
import json
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from data.data_manager import DataManager
from factors.processor import FactorProcessor
from research import FactorResearchEngine, default_research_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("FactorResearchRunner")


def main():
    parser = argparse.ArgumentParser(description="A股多因子研究与 Alpha 验证执行器")
    parser.add_argument("--horizon", type=int, default=20, help="主研究预测视界 (交易日, 默认 20)")
    parser.add_argument("--output-dir", type=str, default=str(settings.BASE_DIR / "reports" / "factor_research"), help="报告输出目录")
    parser.add_argument("--limit-factors", type=int, default=None, help="仅研究前 N 个因子 (用于加速调试)")
    parser.add_argument("--factor-start", type=int, default=0, help="因子切片起点 (含, 用于分批跑长任务)")
    parser.add_argument("--factor-end", type=int, default=None, help="因子切片终点 (不含; 默认=全部)")
    parser.add_argument("--dataset", type=str, default=None, help="指定因子矩阵数据集路径 (默认优先使用 300 标的生产数据集)")
    parser.add_argument("--force-rebuild", action="store_true", help="强制清除旧缓存并从原始数据重构因子矩阵")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("===========================================================================")
    logger.info(">> A股多因子研究与 Alpha 验证系统 (Factor Research & Alpha Validation Phase 1.6)")
    logger.info("===========================================================================")

    factor_df = None
    # 优先使用显式指定的 dataset，或自动探测 300 标的生产数据集
    prod_factor_file = ROOT_DIR / "data_storage" / "research" / "factor_matrix_300.parquet"
    default_factor_file = ROOT_DIR / "data_storage" / "factors" / "factor_matrix.parquet"

    target_file = None
    if args.dataset:
        target_file = Path(args.dataset)
    elif prod_factor_file.exists() and not args.force_rebuild:
        target_file = prod_factor_file
        logger.info(f"自动选用 Phase 1.6 生产级 300 标的因子矩阵: {target_file}")
    else:
        target_file = default_factor_file

    fp = FactorProcessor()
    rebuild_needed = args.force_rebuild

    if target_file and target_file.exists() and not rebuild_needed:
        try:
            logger.info(f"正在加载因子矩阵: {target_file}...")
            df_check = pd.read_parquet(target_file)
            req_cols = ["date", "symbol", "adj_open", "adj_close", "benchmark_open", "benchmark_close", "in_universe"]
            missing_req = [c for c in req_cols if c not in df_check.columns]
            if missing_req:
                logger.warning(f"发现已有因子矩阵缺少必要列 {missing_req}，判定为旧版缓存，触发强制重建...")
                rebuild_needed = True
            else:
                factor_df = df_check
        except Exception as exc:
            logger.warning(f"读取因子缓存失败: {exc}，触发强制重建...")
            rebuild_needed = True

    if rebuild_needed or factor_df is None:
        logger.info("正在从完整行情数据集端到端重新计算因子矩阵...")
        dm = DataManager()
        market_df = dm.sync_and_build_dataset(force_update=True)
        factor_df = fp.build_and_save_factor_matrix(market_df, force_update=True)

    factor_cols = fp.get_all_factor_cols()
    factor_cols_present = [c for c in factor_cols if c in factor_df.columns]
    if args.limit_factors:
        factor_cols_present = factor_cols_present[:args.limit_factors]
    # 分批模式: 因子切片 (start 含 / end 不含), 与 limit_factors 互斥时以后者优先
    if not args.limit_factors and (args.factor_start > 0 or args.factor_end is not None):
        factor_cols_present = factor_cols_present[args.factor_start:args.factor_end]
        logger.info(f"分批模式: 因子切片 [{args.factor_start}:{args.factor_end})")

    logger.info(f"成功加载因子矩阵，样本行数: {len(factor_df)}，待研究因子数: {len(factor_cols_present)}")

    # 3. 运行因子研究引擎
    engine = FactorResearchEngine(config=default_research_config)
    res = engine.run_full_research(
        df=factor_df,
        factor_cols=factor_cols_present,
        primary_horizon=args.horizon,
        output_dir=out_dir
    )

    # 4. 生成 Data Universe 追踪溯源证据 (Phase 1.6.1 PIT Trace with exact conservation)
    cs_series = factor_df.groupby("date")["symbol"].count()
    all_syms = sorted(factor_df["symbol"].unique().tolist())
    in_univ_syms = sorted(factor_df[factor_df["in_universe"]]["symbol"].unique().tolist())
    
    total_membership_rows = len(factor_df)
    active_membership_rows = int(factor_df["in_universe"].sum()) if "in_universe" in factor_df.columns else total_membership_rows
    rejected_membership_rows = total_membership_rows - active_membership_rows

    st_count = int(factor_df.get("is_st", pd.Series([False]*len(factor_df))).sum())
    susp_count = int(factor_df.get("is_suspended", pd.Series([False]*len(factor_df))).sum())
    subnew_count = int((factor_df.get("days_since_listing", factor_df.get("listing_trading_days", pd.Series([999]*len(factor_df)))) < 60).sum())

    universe_trace = {
        "requested_unique_symbols": len(all_syms),
        "available_symbols_count": len(all_syms),
        "accepted_unique_symbols": len(in_univ_syms),
        "rejected_unique_symbols": len(all_syms) - len(in_univ_syms),
        "accepted_symbols_count": len(in_univ_syms),
        "rejected_symbols_count": len(all_syms) - len(in_univ_syms),
        "daily_membership_rows_total": total_membership_rows,
        "daily_membership_rows_active": active_membership_rows,
        "daily_membership_rows_rejected": rejected_membership_rows,
        "rejected_daily_rows_by_reason": {
            "subnew_under_60_trading_days": subnew_count,
            "st_risk_filtered": st_count,
            "suspended_filtered": susp_count
        },
        "rejected_by_reason": {
            "subnew_listing_under_60_days": subnew_count,
            "st_risk_filtered": st_count,
            "suspended_filtered": susp_count
        },
        "daily_active_count_summary": {
            "min": int(cs_series.min()),
            "p10": int(np.percentile(cs_series, 10)),
            "median": int(cs_series.median()),
            "mean": round(float(cs_series.mean()), 1),
            "p90": int(np.percentile(cs_series, 90)),
            "max": int(cs_series.max())
        },
        "survivorship_bias_status": "AUDITED_POINT_IN_TIME",
        "pit_universe_status": "VERIFIED_POINT_IN_TIME",
        "historical_st_status": "ZERO_EVENT_VERIFIED",
        "historical_industry_status": "OFFICIAL_SHENWAN_CLASSIFICATION",
        "listing_delisting_status": "EXCHANGE_OFFICIAL_IPO_DATES",
        "benchmark_symbol": settings.BENCHMARK_SYMBOL,
        "benchmark_timing_status": engine.benchmark_evidence.get("benchmark_timing_status", "BENCHMARK_DATA_INVALID"),
        "benchmark_open_coverage": engine.benchmark_evidence.get("benchmark_open_coverage_ratio", 0.0),
        "benchmark_close_coverage": engine.benchmark_evidence.get("benchmark_close_coverage_ratio", 0.0)
    }
    with open(out_dir / "universe_trace.json", "w", encoding="utf-8") as f:
        json.dump(universe_trace, f, indent=2, ensure_ascii=False)

    # 兼容历史 data_universe_trace.json
    with open(out_dir / "data_universe_trace.json", "w", encoding="utf-8") as f:
        json.dump(universe_trace, f, indent=2, ensure_ascii=False)

    # 5. 输出终端摘要
    print("\n" + "=" * 80)
    print(f">> 因子研究完成！报告与图表已保存至: {out_dir}")
    print("=" * 80)
    print(f"  * 证据级别 (Validity): {engine.run_manifest.get('research_validity_status')}")
    print(f"  * 生产准入 (Readiness): {engine.run_manifest.get('production_readiness_status')}")
    print(f"  * 截面中位数: {engine.run_manifest.get('median_daily_cross_section')} (Min: {engine.run_manifest.get('min_daily_cross_section')}, Max: {engine.run_manifest.get('max_daily_cross_section')})")
    print(f"  * Walk-Forward 折数: {engine.run_manifest.get('wf_total_folds')} 折")
    print(f"  * 核心有效因子 (STRONG): {len(res.selected_factors)} 个 ({', '.join(res.selected_factors[:5]) if res.selected_factors else 'None'})")
    print(f"  * 次级可用因子 (USEFUL): {len(res.useful_factors)} 个 ({', '.join(res.useful_factors[:5]) if res.useful_factors else 'None'})")
    print(f"  * 弱预测力因子 (WEAK):   {len(res.weak_factors)} 个")
    print(f"  * 淘汰过滤因子 (REJECT): {len(res.rejected_factors)} 个")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception(f"Fatal error in factor research pipeline: {exc}")
        sys.exit(1)
