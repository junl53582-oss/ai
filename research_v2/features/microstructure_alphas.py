"""
日内与集合竞价微观结构 Alpha 特征库 (research_v2/features/microstructure_alphas.py)
实现 5 大捕捉主力日内真实意图与筹码聚集的前沿 Alpha:
1. ALPHA_AUCTION_GAP_THRUST: 早盘集合竞价跳空推力 (衡量隔夜预期与主力开盘试盘)
2. ALPHA_OPEN_ABSORPTION_RATIO: 开盘后洗盘与承接强弱比 (高开低走被砸 vs 低开高走承接)
3. ALPHA_TAIL_VWAP_TWAP_SPREAD: 日内均价与时间均价偏离度 (VWAP/TWAP 溢价，衡量尾盘抢筹)
4. ALPHA_INTRADAY_VOLATILITY_ASYMMETRY: 上行与下行空间阻力非对称度
5. ALPHA_CLOSE_THRUST_INTENSITY: 尾盘拉升强度与筹码换手活跃度 (尾盘推力乘换手率)

支持 Point-In-Time 严格截面去极值 (3 MAD) 与 Z-Score 标准化。
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def compute_microstructure_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于基础量价矩阵计算 5 大日内与竞价微观结构 Alpha
    输入 df 需包含: symbol, date, open, high, low, close, volume, amount, turnover (可选)
    """
    res = df.copy()
    res["date"] = pd.to_datetime(res["date"])
    res = res.sort_values(by=["symbol", "date"]).reset_index(drop=True)

    # 前收盘价
    res["_pre_close"] = res.groupby("symbol")["close"].shift(1)
    # 若首日缺失前收盘价，以开盘价替代
    res["_pre_close"] = res["_pre_close"].fillna(res["open"])

    # -------------------------------------------------------------
    # Alpha 1: 集合竞价跳空推力 (Auction Gap Thrust)
    # open_t / close_{t-1} - 1
    # -------------------------------------------------------------
    res["ALPHA_AUCTION_GAP_THRUST"] = (res["open"] / (res["_pre_close"] + 1e-6) - 1.0).clip(-0.25, 0.25)

    # -------------------------------------------------------------
    # Alpha 2: 开盘承接强弱比 (Open Absorption Ratio)
    # (close - open) / (high - low + eps)
    # -------------------------------------------------------------
    high_low_range = (res["high"] - res["low"]).clip(lower=1e-5)
    res["ALPHA_OPEN_ABSORPTION_RATIO"] = ((res["close"] - res["open"]) / high_low_range).clip(-1.0, 1.0)

    # -------------------------------------------------------------
    # Alpha 3: 尾盘均价溢价 (Tail VWAP/TWAP Spread)
    # VWAP = amount / volume, TWAP = (open + high + low + close) / 4
    # VWAP / TWAP - 1
    # -------------------------------------------------------------
    # 当量或额为0时安全保护
    vol_safe = res["volume"].clip(lower=1e-3)
    raw_vwap = res["amount"] / vol_safe
    raw_twap = (res["open"] + res["high"] + res["low"] + res["close"]) / 4.0
    # 防止由于除权除息造成的量价单位不匹配，以相对比率计算
    vwap_twap_ratio = raw_vwap / (raw_twap + 1e-6)
    # 正常该比率在 0.8 ~ 1.2 之间波动
    res["ALPHA_TAIL_VWAP_TWAP_SPREAD"] = (vwap_twap_ratio - 1.0).clip(-0.15, 0.15)

    # -------------------------------------------------------------
    # Alpha 4: 日内上下行阻力非对称度 (Volatility Asymmetry)
    # (high - open) / (open - low + eps)
    # -------------------------------------------------------------
    down_dist = (res["open"] - res["low"]).clip(lower=1e-5)
    up_dist = (res["high"] - res["open"]).clip(lower=1e-5)
    res["ALPHA_INTRADAY_VOLATILITY_ASYMMETRY"] = np.log((up_dist / down_dist).clip(0.05, 20.0))

    # -------------------------------------------------------------
    # Alpha 5: 尾盘拉升强度与活跃度 (Close Thrust Intensity)
    # ((2*close - high - low) / (high - low + eps)) * turnover
    # -------------------------------------------------------------
    turnover_col = res["turnover"] if "turnover" in res.columns else (res["volume"] / (res["volume"].median() or 1.0)).clip(0.1, 10.0)
    clv = (2.0 * res["close"] - res["high"] - res["low"]) / high_low_range
    res["ALPHA_CLOSE_THRUST_INTENSITY"] = clv * np.sqrt(turnover_col.clip(lower=0.01))

    # -------------------------------------------------------------
    # 截面去极值与标准化 (Winsorize 3 MAD & Z-Score per date)
    # -------------------------------------------------------------
    target_cols = [
        "ALPHA_AUCTION_GAP_THRUST",
        "ALPHA_OPEN_ABSORPTION_RATIO",
        "ALPHA_TAIL_VWAP_TWAP_SPREAD",
        "ALPHA_INTRADAY_VOLATILITY_ASYMMETRY",
        "ALPHA_CLOSE_THRUST_INTENSITY"
    ]

    for col in target_cols:
        res[col] = res.groupby("date")[col].transform(_robust_standardize)

    # 清理临时列
    res = res.drop(columns=[c for c in res.columns if c.startswith("_")], errors="ignore")
    return res


def _robust_standardize(series: pd.Series) -> pd.Series:
    """截面去极值与 Z-Score 标准化"""
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
