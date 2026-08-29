"""
A股另类与进阶特征因子库 (factors/alternative_factors.py)
包含基于因子注册器的微观结构波动率、资金流代理、波动压缩蓄势、动量加速度、Kaufman 效率比与量价背离等高效 Alpha 因子。
"""
import numpy as np
import pandas as pd
from .registry import FactorRegistry

eps = 1e-8


# ---------------- 1. 资金流与吸筹因子 ----------------

@FactorRegistry.register(
    "FLOW_NET_BUY_RATIO_5D",
    category="money_flow",
    description="5日估算主动买入净量强度 (日内高低位置加权金额)",
    lookback_days=5
)
def calc_flow_net_buy_5d(df: pd.DataFrame) -> pd.Series:
    """日内多空博弈净资金流 = ((Close - Low) - (High - Close)) / (High - Low + eps) * Amount"""
    high = df["adj_high"] if "adj_high" in df.columns else df["high"]
    low = df["adj_low"] if "adj_low" in df.columns else df["low"]
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    amount = df["amount"] if "amount" in df.columns else df["volume"] * close

    clv = ((close - low) - (high - close)) / (high - low + eps)
    net_flow = clv * amount
    return net_flow.rolling(5).sum() / (amount.rolling(5).sum() + eps)


@FactorRegistry.register(
    "FLOW_ACCUMULATION_20D",
    category="money_flow",
    description="20日资金累积与吸筹指标 (Chaikin A/D Oscillator 变体)",
    lookback_days=20
)
def calc_flow_accumulation_20d(df: pd.DataFrame) -> pd.Series:
    high = df["adj_high"] if "adj_high" in df.columns else df["high"]
    low = df["adj_low"] if "adj_low" in df.columns else df["low"]
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    vol = df["volume"]

    clv = ((close - low) - (high - close)) / (high - low + eps)
    ad = clv * vol
    return ad.rolling(20).mean() / (vol.rolling(20).mean() + eps)


# ---------------- 2. 高阶微观结构与真实波动率因子 ----------------

@FactorRegistry.register(
    "YANG_ZHANG_VOL_20",
    category="volatility",
    description="Yang-Zhang 20日无偏极值波动率 (融合隔夜跳空与日内振幅的高精度估计量)",
    lookback_days=20
)
def calc_yang_zhang_vol(df: pd.DataFrame) -> pd.Series:
    """
    Yang-Zhang 波动率估计量 (比普通收盘价标准差精确度提升 10+ 倍)
    sigma_YZ^2 = sigma_overnight^2 + k * sigma_open_close^2 + (1-k) * sigma_RS^2
    """
    open_p = df["adj_open"] if "adj_open" in df.columns else df["open"]
    high_p = df["adj_high"] if "adj_high" in df.columns else df["high"]
    low_p = df["adj_low"] if "adj_low" in df.columns else df["low"]
    close_p = df["adj_close"] if "adj_close" in df.columns else df["close"]
    prev_close = close_p.shift(1)

    log_ho = np.log(high_p / (open_p + eps) + eps)
    log_lo = np.log(low_p / (open_p + eps) + eps)
    log_co = np.log(close_p / (open_p + eps) + eps)
    log_oc = np.log(open_p / (prev_close + eps) + eps)
    log_cc = np.log(close_p / (prev_close + eps) + eps)

    # Rogers-Satchell 波动率项
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    rs_var = rs.rolling(20).mean()

    # 隔夜项与日内项
    open_var = log_oc.rolling(20).var()
    close_var = log_co.rolling(20).var()

    k = 0.34 / (1.34 + (20 + 1) / (20 - 1))
    yz_var = open_var + k * close_var + (1 - k) * rs_var
    return np.sqrt(np.maximum(yz_var, 0.0))


@FactorRegistry.register(
    "VOLATILITY_SQUEEZE_20",
    category="volatility",
    description="20日布林带与通道波动压缩比 (低波蓄势突破识别)",
    lookback_days=20
)
def calc_volatility_squeeze(df: pd.DataFrame) -> pd.Series:
    """布林带宽度 / 20日均价"""
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    boll_width = (2 * 2 * std20) / (ma20 + eps)
    return -boll_width


@FactorRegistry.register(
    "ATR_RATIO_14",
    category="volatility",
    description="14日真实波动幅度(ATR)相对于60日均值的弹性偏离比",
    lookback_days=60
)
def calc_atr_ratio_14(df: pd.DataFrame) -> pd.Series:
    high = df["adj_high"] if "adj_high" in df.columns else df["high"]
    low = df["adj_low"] if "adj_low" in df.columns else df["low"]
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr14 = tr.rolling(14).mean()
    atr60 = tr.rolling(60).mean()
    return atr14 / (atr60 + eps)


# ---------------- 3. 趋势效率与动量加速度因子 ----------------

@FactorRegistry.register(
    "KAUFMAN_EFFICIENCY_20",
    category="trend_efficiency",
    description="20日 Kaufman 价格趋势效率比 (净位移 / 累计路径长度，区分强趋势与杂波震荡)",
    lookback_days=20
)
def calc_kaufman_efficiency(df: pd.DataFrame) -> pd.Series:
    """ER = |Price(t) - Price(t-N)| / Sum(|Price(i) - Price(i-1)|)"""
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    change = (close - close.shift(20)).abs()
    volatility = (close - close.shift(1)).abs().rolling(20).sum()
    er = change / (volatility + eps)
    return er.fillna(0.0)


@FactorRegistry.register(
    "MACD_HIST_SLOPE_5",
    category="momentum",
    description="5日 MACD 柱状图加速度 (动量一阶与二阶导变化率)",
    lookback_days=35
)
def calc_macd_hist_slope(df: pd.DataFrame) -> pd.Series:
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_hist = (dif - dea) * 2.0
    return macd_hist - macd_hist.shift(5)


@FactorRegistry.register(
    "MA_BULL_ALIGNMENT",
    category="momentum",
    description="5/10/20/60日均线多头排列综合评分",
    lookback_days=60
)
def calc_ma_bull_alignment(df: pd.DataFrame) -> pd.Series:
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    c1 = (ma5 > ma10).astype(float)
    c2 = (ma10 > ma20).astype(float)
    c3 = (ma20 > ma60).astype(float)
    c4 = (close > ma5).astype(float)
    return (c1 + c2 + c3 + c4) / 4.0


@FactorRegistry.register(
    "RSI_14",
    category="momentum",
    description="14日经典相对强弱指标 (RSI)",
    lookback_days=14
)
def calc_rsi_14(df: pd.DataFrame) -> pd.Series:
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down + eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi / 100.0


@FactorRegistry.register(
    "BOLL_PCT_B",
    category="momentum",
    description="20日布林带 %B 震荡位置指标 ((Close - Lower) / (Upper - Lower))",
    lookback_days=20
)
def calc_boll_pct_b(df: pd.DataFrame) -> pd.Series:
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    pct_b = (close - lower) / (upper - lower + eps)
    return pct_b


# ---------------- 4. 量价背离与反转因子 ----------------

@FactorRegistry.register(
    "VP_DIVERGENCE_10D",
    category="volume_price",
    description="10日价格变动与成交量变动的相关性与背离度",
    lookback_days=10
)
def calc_vp_divergence_10d(df: pd.DataFrame) -> pd.Series:
    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    vol = df["volume"]

    ret = close.pct_change()
    vol_chg = vol.pct_change()
    corr = ret.rolling(10).corr(vol_chg)
    return corr.fillna(0.0)
