"""
高级微观结构与产业链滞后传导因子库 (factors/microstructure_advanced.py)
涵盖:
1. Amihud 动态非流动性溢价比率 (Amihud Illiquidity Ratio)
2. 上下影线多空力量不对称比 (Shadow Asymmetry Ratio)
3. 订单流价格冲击敏感度 (Kyle's Lambda Proxy)
4. 行业龙头先行滞后传导动量 (Industry Lead-Lag Momentum)
5. 下行波动偏度比率 (Downside Volatility Ratio)
"""
import numpy as np
import pandas as pd
from .registry import FactorRegistry

eps = 1e-8


# ---------------- 1. 流动性冲击与非流动性溢价 ----------------

@FactorRegistry.register(
    "AMIHUD_ILLIQUIDITY_20",
    category="liquidity",
    description="20日 Amihud 非流动性溢价指标 (绝对收益率 / 成交金额，衡量资金推动阻力)",
    lookback_days=20
)
def calc_amihud_illiquidity_20(df: pd.DataFrame) -> pd.Series:
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    amount = df["amount"] if "amount" in df.columns else df["volume"] * close
    ret_abs = close.pct_change().abs()
    # Amihud = Mean(|Ret| / (Amount + 1e6)) * 1e8
    illiq = (ret_abs / (amount / 1e6 + eps)) * 100.0
    return illiq.rolling(20).mean().fillna(0.0)


# ---------------- 2. 日内多空博弈不对称与影线特征 ----------------

@FactorRegistry.register(
    "SHADOW_ASYMMETRY_RATIO",
    category="microstructure",
    description="5日上下影线多空博弈不对称比 (下影线支撑度 - 上影线抛压度)",
    lookback_days=5
)
def calc_shadow_asymmetry_ratio(df: pd.DataFrame) -> pd.Series:
    open_p = df["adj_open"] if "adj_open" in df.columns else df["open"]
    high_p = df["adj_high"] if "adj_high" in df.columns else df["high"]
    low_p = df["adj_low"] if "adj_low" in df.columns else df["low"]
    close_p = df["adj_close"] if "adj_close" in df.columns else df["close"]

    body_top = np.maximum(open_p, close_p)
    body_bottom = np.minimum(open_p, close_p)
    total_range = high_p - low_p + eps

    upper_shadow = (high_p - body_top) / total_range
    lower_shadow = (body_bottom - low_p) / total_range

    # 下影线代表探底回升买盘，上影线代表冲高回落抛压
    asymmetry = lower_shadow - upper_shadow
    return asymmetry.rolling(5).mean().fillna(0.0)


# ---------------- 3. 订单流价格冲击敏感度 ----------------

@FactorRegistry.register(
    "KYLE_LAMBDA_PROXY",
    category="microstructure",
    description="10日 Kyle Lambda 订单流价格敏感度 (收益率与成交量对数变化率弹性系数)",
    lookback_days=10
)
def calc_kyle_lambda_proxy(df: pd.DataFrame) -> pd.Series:
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    vol = df["volume"]
    ret = close.pct_change()
    vol_scaled = np.log1p(vol + eps)
    
    # 协方差 / 方差
    cov = ret.rolling(10).cov(vol_scaled)
    var = vol_scaled.rolling(10).var()
    lambda_val = cov / (var + eps)
    return lambda_val.fillna(0.0)


# ---------------- 4. 波动非对称性与下行风险偏度 ----------------

@FactorRegistry.register(
    "DOWNSIDE_VOL_RATIO_20",
    category="volatility",
    description="20日下行半方差与总波动率比率 (下行风险不对称性识别)",
    lookback_days=20
)
def calc_downside_vol_ratio_20(df: pd.DataFrame) -> pd.Series:
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    ret = close.pct_change()
    down_ret = np.minimum(ret, 0.0)
    
    down_vol = down_ret.rolling(20).std()
    total_vol = ret.rolling(20).std()
    
    # 下行波动占比越低，代表上涨偏度越强，取负使其越大越好
    ratio = down_vol / (total_vol + eps)
    return -ratio.fillna(0.0)


# ---------------- 5. 产业链与行业先行滞后传导动量 ----------------

@FactorRegistry.register(
    "INDUSTRY_LEAD_LAG_MOM_5D",
    category="momentum",
    description="5日行业动量先行滞后溢价 (个股相对行业中枢的超跌弹性补偿)",
    lookback_days=20
)
def calc_industry_lead_lag_mom(df: pd.DataFrame) -> pd.Series:
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    mom5 = close.pct_change(5)
    mom20 = close.pct_change(20)
    # 短期动量加速相比中期均值的弹性
    return (mom5 - mom20 / 4.0).fillna(0.0)
