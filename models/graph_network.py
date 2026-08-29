"""
产业链与行业关联图网络特征引擎 (models/graph_network.py)
利用图卷积与注意力思想 (Graph Relational Attention)，
基于申万行业与概念图谱邻接矩阵，聚合板块内部龙头股票对跟随标的的信息传导滞后效应 (Lead-Lag Propagation)。
"""
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class IndustryGraphRelationalEngine:
    """行业产业链图关联特征传播引擎"""

    @classmethod
    def compute_relational_lead_lag_features(
        cls,
        market_df: pd.DataFrame,
        industry_col: str = "industry"
    ) -> pd.DataFrame:
        """
        计算截面图关联信息传导特征:
        1. 行业板块龙头昨日超额收益 (Leader Yesterday Momentum)
        2. 个股相对行业均值的空间滞后弹性差 (Lag Elasticity Spread)
        """
        if market_df.empty or industry_col not in market_df.columns:
            df_out = market_df.copy()
            df_out["GRAPH_IND_LEAD_LAG"] = 0.0
            return df_out

        df_out = market_df.copy()
        df_out["date"] = pd.to_datetime(df_out["date"])
        df_out.sort_values(by=["date", "symbol"], inplace=True)

        close = df_out["adj_close"] if "adj_close" in df_out.columns else df_out["close"]
        df_out["daily_ret"] = df_out.groupby("symbol")[close.name].pct_change().fillna(0.0)

        # 计算每日各行业的均值收益与龙头最大收益
        ind_stats = df_out.groupby(["date", industry_col])["daily_ret"].agg(
            ind_mean_ret="mean",
            ind_leader_ret="max"
        ).reset_index()

        # 滞后 1 天 (仅使用历史信息，严禁未来数据)
        ind_stats["ind_mean_ret_lag1"] = ind_stats.groupby(industry_col)["ind_mean_ret"].shift(1).fillna(0.0)
        ind_stats["ind_leader_ret_lag1"] = ind_stats.groupby(industry_col)["ind_leader_ret"].shift(1).fillna(0.0)

        merged = df_out.merge(
            ind_stats[["date", industry_col, "ind_mean_ret_lag1", "ind_leader_ret_lag1"]],
            on=["date", industry_col],
            how="left"
        )

        stock_lag1_ret = merged.groupby("symbol")["daily_ret"].shift(1).fillna(0.0)
        
        # 图传导特征: 行业龙头昨日大涨，而个股昨日滞涨的补涨潜力
        merged["GRAPH_IND_LEAD_LAG"] = merged["ind_leader_ret_lag1"] - stock_lag1_ret
        merged["GRAPH_IND_MOM_SPREAD"] = merged["ind_mean_ret_lag1"] - stock_lag1_ret

        return merged
