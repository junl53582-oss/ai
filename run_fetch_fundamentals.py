"""
独立脚本: 仅拉取并缓存全股票池基本面财务数据 (data_storage/fundamentals/*.parquet)
可后台运行，断点续拉；run_pipeline 重跑时命中缓存，避免重复网络请求。
"""
import sys
import io

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from config.settings import settings
from data.universe_provider import create_universe_provider
from data.data_manager import DataManager
from data.fundamentals import FundamentalsProvider


def main():
    dm = DataManager(universe_provider=create_universe_provider(settings))
    mdf = dm.sync_and_build_dataset()
    print(f"[fetch] market_df rows={len(mdf)} symbols={mdf['symbol'].nunique()}", flush=True)

    fp = FundamentalsProvider(delay_days=settings.FUNDAMENTAL_DELAY_DAYS)
    res = fp.build_daily_fundamental_matrix(mdf, start_year=settings.FUNDAMENTAL_START_YEAR)
    print(f"[fetch] FUND_DONE source_counts={fp.source_counts} coverage={fp.coverage_ratio}", flush=True)
    print("[fetch] 基本面缓存构建完成，可安全退出。", flush=True)


if __name__ == "__main__":
    main()
