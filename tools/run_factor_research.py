"""
多因子研究与 Alpha 验证命令行工具 (tools/run_factor_research.py)
用法:
    python tools/run_factor_research.py [--horizon 20] [--output-dir reports/factor_research]
"""
import sys
import argparse
import logging
from pathlib import Path

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
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("===========================================================================")
    logger.info(">> A股多因子研究与 Alpha 验证系统 (Factor Research & Alpha Validation)")
    logger.info("===========================================================================")

    # 1. 加载行情与因子数据
    logger.info("正在加载行情数据与因子矩阵...")
    dm = DataManager()
    market_df = dm.load_dataset()

    fp = FactorProcessor()
    factor_file = fp.factor_dir / "factor_matrix.parquet"
    if factor_file.exists():
        factor_df = fp.load_factor_matrix()
    else:
        factor_df = fp.build_and_save_factor_matrix(market_df)

    factor_cols = fp.get_all_factor_cols()
    factor_cols_present = [c for c in factor_cols if c in factor_df.columns]
    if args.limit_factors:
        factor_cols_present = factor_cols_present[:args.limit_factors]

    logger.info(f"成功加载因子矩阵，样本行数: {len(factor_df)}，待研究因子数: {len(factor_cols_present)}")

    # 2. 运行因子研究引擎
    engine = FactorResearchEngine(config=default_research_config)
    res = engine.run_full_research(
        df=factor_df,
        factor_cols=factor_cols_present,
        primary_horizon=args.horizon,
        output_dir=out_dir
    )

    # 3. 输出终端摘要
    print("\n" + "=" * 80)
    print(f">> 因子研究完成！报告与图表已保存至: {out_dir}")
    print("=" * 80)
    print(f"  * 核心有效因子 (STRONG): {len(res.selected_factors)} 个 ({', '.join(res.selected_factors[:5]) if res.selected_factors else 'None'})")
    print(f"  * 次级可用因子 (USEFUL): {len(res.useful_factors)} 个")
    print(f"  * 弱预测力因子 (WEAK):   {len(res.weak_factors)} 个")
    print(f"  * 淘汰过滤因子 (REJECT): {len(res.rejected_factors)} 个")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
