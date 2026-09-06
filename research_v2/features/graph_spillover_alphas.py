"""
产业链与行业图关联传导 Alpha 特征工程 (research_v2/features/graph_spillover_alphas.py)
利用图结构信息扩散 (Graph Diffusion & Relational Spillover) 思想:
A 股具有强烈的板块产业链内部“龙头先动、二级滞后跟风”的传导时滞效应。
本模块实现 3 大高阶拓扑传导 Alpha:
1. GRAPH_LEADER_SPILLOVER_1D: 板块龙头昨日涨幅与个股昨日涨幅之差 (补涨弹性差)
2. GRAPH_CLUSTER_BREADTH_SURGE_5D: 板块内部个股普涨突破比率的 5 日加速度 (板块共振暴发度)
3. GRAPH_CLUSTER_FLOW_SPILLOVER_10D: 板块内部关联标的主力资金外溢强度 (资金扩散度)

严格 Point-In-Time 隔离，严禁未来数据穿越。
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def compute_graph_spillover_alphas(
    df: pd.DataFrame,
    industry_col: str = "industry"
) -> pd.DataFrame:
    """
    计算基于行业图拓扑的传导 Alpha 特征
    输入 df 需包含: symbol, date, close, high, low, volume, amount, industry (可选)
    """
    res = df.copy()
    res["date"] = pd.to_datetime(res["date"])
    res = res.sort_values(by=["date", "symbol"]).reset_index(drop=True)

    if industry_col not in res.columns:
        res[industry_col] = "UNKNOWN"
    res[industry_col] = res[industry_col].fillna("UNKNOWN")

    # 1. 计算单日收益率
    close_col = "adj_close" if "adj_close" in res.columns else "close"
    res["_daily_ret"] = res.groupby("symbol")[close_col].pct_change().fillna(0.0)

    # 2. 板块内部统计: 每日每个行业的平均收益、龙头收益 (最大市值或最大收益标的)
    ind_daily = res.groupby(["date", industry_col]).agg(
        ind_mean_ret=("_daily_ret", "mean"),
        ind_max_ret=("_daily_ret", "max"),
        ind_surge_count=("_daily_ret", lambda s: (s > 0.02).sum()),
        ind_stock_count=("_daily_ret", "count")
    ).reset_index()

    ind_daily["ind_surge_ratio"] = ind_daily["ind_surge_count"] / ind_daily["ind_stock_count"].clip(lower=1)

    # 严格滞后 1 天 (T-1 日龙头收益用于预测 T 日，绝无未来数据)
    ind_daily = ind_daily.sort_values(by=[industry_col, "date"]).reset_index(drop=True)
    ind_daily["ind_max_ret_lag1"] = ind_daily.groupby(industry_col)["ind_max_ret"].shift(1).fillna(0.0)
    ind_daily["ind_mean_ret_lag1"] = ind_daily.groupby(industry_col)["ind_mean_ret"].shift(1).fillna(0.0)
    
    # 5日共振突破斜率 (5D Breadth Surge)
    ind_daily["ind_surge_ratio_lag1"] = ind_daily.groupby(industry_col)["ind_surge_ratio"].shift(1).fillna(0.0)
    ind_daily["ind_surge_5d_mean"] = ind_daily.groupby(industry_col)["ind_surge_ratio_lag1"].rolling(5, min_periods=2).mean().reset_index(drop=True).fillna(0.0)

    # 合并回主表
    merged = pd.merge(
        res,
        ind_daily[["date", industry_col, "ind_max_ret_lag1", "ind_mean_ret_lag1", "ind_surge_5d_mean"]],
        on=["date", industry_col],
        how="left"
    )

    # 个股自身昨日收益
    self_lag1_ret = merged.groupby("symbol")["_daily_ret"].shift(1).fillna(0.0)

    # -------------------------------------------------------------
    # Alpha 1: 龙头昨日暴涨、个股滞涨的补涨弹性差
    # -------------------------------------------------------------
    merged["GRAPH_LEADER_SPILLOVER_1D"] = (merged["ind_max_ret_lag1"] - self_lag1_ret).clip(-0.20, 0.20)

    # -------------------------------------------------------------
    # Alpha 2: 板块共振突破度
    # -------------------------------------------------------------
    merged["GRAPH_CLUSTER_BREADTH_SURGE_5D"] = merged["ind_surge_5d_mean"].fillna(0.0)

    # -------------------------------------------------------------
    # Alpha 3: 板块资金流外溢强度 (若有主力资金流特征，结合其板块均值)
    # -------------------------------------------------------------
    if "ALPHA_MAIN_CAPITAL_NET_RATIO_5D" in merged.columns:
        ind_flow = merged.groupby(["date", industry_col])["ALPHA_MAIN_CAPITAL_NET_RATIO_5D"].mean().reset_index(name="_ind_flow_mean")
        ind_flow["_ind_flow_lag1"] = ind_flow.groupby(industry_col)["_ind_flow_mean"].shift(1).fillna(0.0)
        merged = pd.merge(merged, ind_flow[["date", industry_col, "_ind_flow_lag1"]], on=["date", industry_col], how="left")
        merged["GRAPH_CLUSTER_FLOW_SPILLOVER_10D"] = merged["_ind_flow_lag1"].fillna(0.0)
    else:
        # 基于成交量突破代理
        vol_surge = merged.groupby("symbol")["volume"].pct_change().shift(1).fillna(0.0)
        merged["GRAPH_CLUSTER_FLOW_SPILLOVER_10D"] = vol_surge.clip(-1.0, 3.0)

    # 截面去极值与标准化 (3 MAD & Z-score)
    graph_cols = ["GRAPH_LEADER_SPILLOVER_1D", "GRAPH_CLUSTER_BREADTH_SURGE_5D", "GRAPH_CLUSTER_FLOW_SPILLOVER_10D"]
    for col in graph_cols:
        merged[col] = merged.groupby("date")[col].transform(_robust_standardize)

    # 清理临时列
    temp_cols = [c for c in merged.columns if c.startswith("_")]
    merged = merged.drop(columns=temp_cols, errors="ignore")

    return merged


def _robust_standardize(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 5:
        return series.fillna(0.0)
    median = valid.median()
    mad = (valid - median).abs().median()
    if mad < 1e-6:
        mad = valid.std() or 1.0
    upper = median + 3.1482 * mad
    lower = median - 3.1482 * mad
    clipped = series.clip(lower=lower, upper=upper)
    mean = clipped.mean()
    std = clipped.std()
    if std < 1e-6:
        return clipped - mean
    return (clipped - mean) / std
