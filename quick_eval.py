"""
快速验证脚本: 仅跑到 OOS AUC 评估 (跳过实盘回测 Step5)，用于快速验证
"基本面因子 + 长持有期标签" 改造对预测力的提升，无需等待完整回测。
"""
import sys
import io
import argparse

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from config.settings import settings
from data.universe_provider import create_universe_provider
from data.data_manager import DataManager
from factors.processor import FactorProcessor
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from models.evaluator import ModelEvaluator


def main(force_update: bool = False):
    print(f"=== 快速验证 (LABEL_HORIZON={settings.LABEL_HORIZON}, "
          f"ENABLE_FUNDAMENTALS={settings.ENABLE_FUNDAMENTALS}) ===")

    univ = create_universe_provider(settings)
    dm = DataManager(universe_provider=univ)
    market_df = dm.sync_and_build_dataset(force_update=force_update)
    print(f"* 行情: {len(market_df)} 行, {market_df['symbol'].nunique()} 只")

    if settings.ENABLE_FUNDAMENTALS:
        from data.fundamentals import FundamentalsProvider
        fp = FundamentalsProvider(delay_days=settings.FUNDAMENTAL_DELAY_DAYS)
        fd = fp.build_daily_fundamental_matrix(market_df, start_year=settings.FUNDAMENTAL_START_YEAR)
        before = set(market_df.columns)
        market_df = market_df.merge(fd, on=["symbol", "date"], how="left")
        new_cols = [c for c in fd.columns if c not in before and c not in ("symbol", "date")]
        cov = fd[new_cols].notna().mean().mean() * 100 if new_cols else 0.0
        print(f"* 基本面: {len(new_cols)} 因子, 覆盖率均值 {cov:.1f}%, 拉取统计 {fp.source_counts}")

    proc = FactorProcessor()
    factor_df = proc.build_and_save_factor_matrix(market_df, force_update=force_update)
    labeler = TargetLabeler(horizon=settings.LABEL_HORIZON)
    factor_df = labeler.compute_excess_return_label(factor_df, canonical_dates=dm.get_trading_calendar())

    factor_cols = FactorProcessor.get_all_factor_cols()
    print(f"* 因子总数: {len(factor_cols)}, 中性化: {proc.industry_neutralization_enabled}")

    trainer = WalkForwardTrainer()
    oos_df, latest_model = trainer.run_walk_forward(factor_df)

    ev = ModelEvaluator()
    m = ev.evaluate_predictions(oos_df)
    if settings.is_classification:
        print("\n=== OOS 分类评估结果 ===")
        print(f"  AUC        = {m['auc']:.4f}")
        print(f"  Accuracy   = {m['accuracy']*100:.2f}%")
        print(f"  Precision  = {m['precision']*100:.2f}%")
        print(f"  Recall     = {m['recall']*100:.2f}%")
        print(f"  F1         = {m['f1']:.4f}")
        print(f"  Brier      = {m['brier_score']:.4f}")
        print(f"  上涨占比   = {m['positive_rate']*100:.1f}%")
    else:
        print(f"  RankIC = {m['rank_ic_mean']:+.4f}, IR = {m['rank_icir']:.4f}")

    top = latest_model.get_feature_importance(top_n=15)
    print("\n=== Top 15 因子重要性 ===")
    for _, r in top.iterrows():
        print(f"  {r['feature']:<22} : {r['importance_pct']:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-update", action="store_true")
    args = ap.parse_args()
    main(force_update=args.force_update)
