"""
Deterministic Rebuilder for Stock Picks Artifacts (tools/rebuild_clean_stock_picks.py)
Rebuilds artifacts/latest_stock_picks.csv, aggressive_stock_picks.csv, and csi500_stock_picks.csv
from the authentic, trusted data slice (as of 2026-08-24) using canonical production inference:
FactorProcessor -> BatchInference -> PortfolioBuilder.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from factors.processor import FactorProcessor
from models.inference import BatchInference
from strategy.portfolio import PortfolioBuilder


def rebuild_all_picks():
    matrix_path = ROOT_DIR / "data_storage" / "research" / "factor_matrix_300.parquet"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing {matrix_path}")

    df = pd.read_parquet(matrix_path)
    df["date"] = pd.to_datetime(df["date"])

    # 锁定可信截止日: 2026-08-24
    trusted_date = pd.to_datetime("2026-08-24")
    sub_df = df[(df["date"] >= "2026-07-01") & (df["date"] <= trusted_date)].copy()
    sub_df = FactorProcessor.compute_advanced_alpha_features(sub_df)

    latest_date = sub_df["date"].max()
    print(f"[*] 依据可信基准日期构建: {latest_date.strftime('%Y-%m-%d')}")

    engine = BatchInference()
    scored_df = engine.predict(sub_df, date=latest_date)

    # 提取真实股票元信息 (名称从 SecurityMaster 权威关联，行业从 PIT 元信息关联)
    sm_path = ROOT_DIR / "data_storage" / "security_master.parquet"
    if sm_path.exists():
        sm_df = pd.read_parquet(sm_path)[["symbol", "name"]].drop_duplicates(subset=["symbol"])
    else:
        sm_df = pd.DataFrame(columns=["symbol", "name"])

    sym_meta = df[["symbol", "industry", "close"]].dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"], keep="last")
    sym_meta = sym_meta.merge(sm_df, on="symbol", how="left")

    scored_df = scored_df.drop(columns=[c for c in ["name", "industry", "close"] if c in scored_df.columns])
    scored_df = scored_df.merge(sym_meta, on="symbol", how="left")

    # 1. 核心大盘优选 latest_stock_picks.csv (Top-8)
    pb_core = PortfolioBuilder(top_k_buy=8, top_k_hold=15, weight_method="inv_vol")
    portfolio_core = pb_core.build_target_portfolio(scored_df, current_holdings=set(), date=latest_date)
    portfolio_core["date"] = latest_date.strftime("%Y-%m-%d")

    # 2. 进取型优选 aggressive_stock_picks.csv (Top-20 排序标的)
    pb_agg = PortfolioBuilder(top_k_buy=20, top_k_hold=30, weight_method="score_weighted")
    portfolio_agg = pb_agg.build_target_portfolio(scored_df, current_holdings=set(), date=latest_date)
    portfolio_agg["date"] = latest_date.strftime("%Y-%m-%d")

    # 3. CSI500 优选池 (如果存在专有池则对齐，否则从成长特征挑选)
    csi500_file = ROOT_DIR / "data_storage" / "research" / "csi500_sample.parquet"
    if csi500_file.exists():
        df_500 = pd.read_parquet(csi500_file)
        scored_500 = scored_df[scored_df["symbol"].isin(df_500["symbol"])].copy()
    else:
        # 从高弹性与非金融中选取 Top-15
        scored_500 = scored_df[~scored_df["industry"].isin(["银行", "非银金融", "房地产"])].copy()

    pb_500 = PortfolioBuilder(top_k_buy=15, top_k_hold=25, weight_method="inv_vol")
    portfolio_500 = pb_500.build_target_portfolio(scored_500, current_holdings=set(), date=latest_date)
    portfolio_500["date"] = latest_date.strftime("%Y-%m-%d")

    # 保存产物
    art_dir = ROOT_DIR / "artifacts"
    art_dir.mkdir(exist_ok=True)

    core_path = art_dir / "latest_stock_picks.csv"
    agg_path = art_dir / "aggressive_stock_picks.csv"
    csi500_path = art_dir / "csi500_stock_picks.csv"

    portfolio_core.to_csv(core_path, index=False)
    portfolio_agg.to_csv(agg_path, index=False)
    portfolio_500.to_csv(csi500_path, index=False)

    print(f"[+] 核心选股已保存: {core_path} ({len(portfolio_core)} 标的, 行业数: {portfolio_core['industry'].nunique()})")
    print(f"[+] 进取选股已保存: {agg_path} ({len(portfolio_agg)} 标的, 行业数: {portfolio_agg['industry'].nunique()})")
    print(f"[+] 成长选股已保存: {csi500_path} ({len(portfolio_500)} 标的, 行业数: {portfolio_500['industry'].nunique()})")


if __name__ == "__main__":
    rebuild_all_picks()
