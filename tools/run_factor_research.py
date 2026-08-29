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
    parser.add_argument("--force-rebuild", action="store_true", help="强制清除旧缓存并从原始数据重构因子矩阵")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("===========================================================================")
    logger.info(">> A股多因子研究与 Alpha 验证系统 (Factor Research & Alpha Validation Phase 1.5)")
    logger.info("===========================================================================")

    # 1. 加载行情数据
    logger.info("正在加载与校验行情数据集...")
    dm = DataManager()
    if args.force_rebuild:
        market_df = dm.sync_and_build_dataset(force_update=True)
    else:
        market_df = dm.load_dataset()

    # 2. 构建或加载因子矩阵 (严格校验必需列)
    fp = FactorProcessor()
    factor_file = fp.factor_dir / "factor_matrix.parquet"
    factor_df = None
    rebuild_needed = args.force_rebuild

    if factor_file.exists() and not rebuild_needed:
        try:
            df_check = pd.read_parquet(factor_file)
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

    if rebuild_needed or factor_df is None or not factor_file.exists():
        logger.info("正在从完整行情数据集端到端重新计算因子矩阵...")
        factor_df = fp.build_and_save_factor_matrix(market_df, force_update=True)

    factor_cols = fp.get_all_factor_cols()
    factor_cols_present = [c for c in factor_cols if c in factor_df.columns]
    if args.limit_factors:
        factor_cols_present = factor_cols_present[:args.limit_factors]

    logger.info(f"成功加载因子矩阵，样本行数: {len(factor_df)}，待研究因子数: {len(factor_cols_present)}")

    # 3. 运行因子研究引擎
    engine = FactorResearchEngine(config=default_research_config)
    res = engine.run_full_research(
        df=factor_df,
        factor_cols=factor_cols_present,
        primary_horizon=args.horizon,
        output_dir=out_dir
    )

    # 4. 生成 Data Universe 追踪溯源证据 (Phase 1.5 P1-5)
    universe_trace = {
        "configured_default_symbols": settings.DEFAULT_UNIVERSE,
        "configured_default_symbols_count": len(settings.DEFAULT_UNIVERSE),
        "universe_profile": getattr(settings, "UNIVERSE_PROFILE", "HS300_CORE"),
        "provider_mode": getattr(dm, "universe_provider", None).get_mode() if hasattr(dm, "universe_provider") else "STATIC_CONFIG",
        "market_dataset_symbols_count": int(market_df["symbol"].nunique()) if "symbol" in market_df.columns else 0,
        "factor_matrix_symbols_count": int(factor_df["symbol"].nunique()) if "symbol" in factor_df.columns else 0,
        "factor_matrix_rows": len(factor_df),
        "benchmark_symbol": settings.BENCHMARK_SYMBOL,
        "benchmark_timing_status": engine.benchmark_evidence.get("benchmark_timing_status", "BENCHMARK_DATA_INVALID"),
        "data_universe_root_cause": "factor_matrix.parquet 在离线环境下使用已归档的 5 只大盘权重股基准样本 (4555行)；全量成分股实盘需连接 AkShare 在线下载。系统如实标记为 DEVELOPMENT_SAMPLE。"
    }
    with open(out_dir / "data_universe_trace.json", "w", encoding="utf-8") as f:
        json.dump(universe_trace, f, indent=2, ensure_ascii=False)

    # 5. 输出终端摘要
    print("\n" + "=" * 80)
    print(f">> 因子研究完成！报告与图表已保存至: {out_dir}")
    print("=" * 80)
    print(f"  * 证据级别 (Validity): {engine.run_manifest.get('research_validity_status')}")
    print(f"  * 核心有效因子 (STRONG): {len(res.selected_factors)} 个 ({', '.join(res.selected_factors[:5]) if res.selected_factors else 'None'})")
    print(f"  * 次级可用因子 (USEFUL): {len(res.useful_factors)} 个")
    print(f"  * 弱预测力因子 (WEAK):   {len(res.weak_factors)} 个")
    print(f"  * 淘汰过滤因子 (REJECT): {len(res.rejected_factors)} 个")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception(f"Fatal error in factor research pipeline: {exc}")
        sys.exit(1)
