"""
全新异源 Alpha 因子库 (research_v2/alphas/novel_alphas.py)
涵盖:
1. 残差动量 (Residual Momentum)
2. 换手率异常冲击 (Turnover Surprise)
3. 复合非线性因子: 质量 x 动量 (Quality x Momentum)
4. 流动性冲击 x 波动率状态 (Liquidity x Volatility)
"""
import pandas as pd
import numpy as np

class NovelAlphaFactory:
    """新一代 Alpha 算子工厂"""

    @staticmethod
    def calc_residual_momentum(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """剥离基准指数收益后的纯个股残差动量"""
        df_sorted = df.sort_values(['symbol', 'date'])
        stock_ret = df_sorted.groupby('symbol')['pct_change'].rolling(window).sum().reset_index(0, drop=True)
        bm_ret = df_sorted.groupby('symbol')['benchmark_close'].pct_change(window).reset_index(0, drop=True)
        return (stock_ret - bm_ret).reindex(df.index)

    @staticmethod
    def calc_turnover_surprise(df: pd.DataFrame, short_w: int = 5, long_w: int = 20) -> pd.Series:
        """换手率突增比率 (Turnover Surprise)"""
        df_sorted = df.sort_values(['symbol', 'date'])
        to_col = 'turnover' if 'turnover' in df.columns else 'volume'
        short_to = df_sorted.groupby('symbol')[to_col].rolling(short_w).mean().reset_index(0, drop=True)
        long_to = df_sorted.groupby('symbol')[to_col].rolling(long_w).mean().reset_index(0, drop=True)
        return (short_to / (long_to + 1e-8) - 1.0).reindex(df.index)

    @staticmethod
    def calc_quality_x_momentum(df: pd.DataFrame) -> pd.Series:
        """复合非线性: 估值/市值残差 x 相对动量"""
        df_sorted = df.sort_values(['symbol', 'date'])
        mom = df_sorted.groupby('symbol')['pct_change'].rolling(20).sum().reset_index(0, drop=True)
        log_mv = df_sorted['LOG_CIRC_MV'] if 'LOG_CIRC_MV' in df.columns else np.log(df_sorted['close'] * df_sorted['volume'] + 1.0)
        # 截面 Rank 交互
        res = []
        temp = pd.DataFrame({'date': df_sorted['date'], 'mom': mom, 'log_mv': log_mv})
        for dt, grp in temp.groupby('date'):
            r_mom = grp['mom'].rank(pct=True)
            r_val = (-grp['log_mv']).rank(pct=True)
            interaction = r_mom * r_val
            res.append(interaction)
        return pd.concat(res).reindex(df.index)

    @staticmethod
    def calc_liquidity_x_volatility(df: pd.DataFrame) -> pd.Series:
        """复合非线性: 非流动性冲击 x 波动率收敛"""
        df_sorted = df.sort_values(['symbol', 'date'])
        amt = df_sorted['amount'] if 'amount' in df.columns else df_sorted['volume'] * df_sorted['close']
        amihud = df_sorted['pct_change'].abs() / (amt + 1.0)
        amihud_roll = df_sorted.groupby('symbol').apply(lambda g: (g['pct_change'].abs() / (g['amount'] + 1.0)).rolling(20).mean()).reset_index(0, drop=True)
        vol_20 = df_sorted.groupby('symbol')['pct_change'].rolling(20).std().reset_index(0, drop=True)
        return (amihud_roll * (vol_20 + 1e-6)).reindex(df.index)
