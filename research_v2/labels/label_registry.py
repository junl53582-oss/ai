"""
版本化标签注册表 (research_v2/labels/label_registry.py)
用于 Stage 3: Label Redesign，在完全公平统一的 Walk-Forward 样本上横向评测 5 大标签体系。
"""
from enum import Enum
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

class LabelVersion(str, Enum):
    LABEL_V1 = "LABEL_V1_RAW_20D"              # 经典 20 日收盘收益率
    LABEL_V2 = "LABEL_V2_EXECUTABLE_20D"       # T+1 开盘买入 -> T+21 开盘卖出可执行收益率
    LABEL_V3 = "LABEL_V3_BENCHMARK_RELATIVE"   # T+1 执行对齐基准超额收益率
    LABEL_V4 = "LABEL_V4_INDUSTRY_NEUTRAL"     # 行业中性化残差收益率
    LABEL_V5 = "LABEL_V5_COST_ADJUSTED"        # 扣除双边 20bps 摩擦成本的可实现 Alpha

class LabelRegistry:
    """版本化标签工厂与计算器"""

    @staticmethod
    def compute_label_v1(df: pd.DataFrame, horizon: int = 20) -> pd.Series:
        """LABEL_V1: 经典日频收盘价 20 日前瞻收益率 (未对齐撮合时序)"""
        df_sorted = df.sort_values(['symbol', 'date'])
        shifted_close = df_sorted.groupby('symbol')['adj_close'].shift(-horizon)
        return (shifted_close / df_sorted['adj_close'] - 1.0)

    @staticmethod
    def compute_label_v2(df: pd.DataFrame, horizon: int = 20) -> pd.Series:
        """LABEL_V2: T+1 开盘买入 -> T+1+horizon 开盘卖出可执行收益率"""
        df_sorted = df.sort_values(['symbol', 'date'])
        entry_price = df_sorted.groupby('symbol')['adj_open'].shift(-1)
        exit_price = df_sorted.groupby('symbol')['adj_open'].shift(-(1 + horizon))
        ret = (exit_price / entry_price - 1.0)
        # 涨停或停牌无法在 T+1 买入则置 NaN
        if 'is_limit_up_locked' in df.columns:
            limit_up_t1 = df_sorted.groupby('symbol')['is_limit_up_locked'].shift(-1)
            ret = ret.mask(limit_up_t1 == True, np.nan)
        if 'is_suspended' in df.columns:
            susp_t1 = df_sorted.groupby('symbol')['is_suspended'].shift(-1)
            ret = ret.mask(susp_t1 == True, np.nan)
        return ret

    @staticmethod
    def compute_label_v3(df: pd.DataFrame, horizon: int = 20) -> pd.Series:
        """LABEL_V3: T+1 执行对齐基准超额收益率"""
        stock_ret = LabelRegistry.compute_label_v2(df, horizon=horizon)
        df_sorted = df.sort_values(['symbol', 'date'])
        bm_entry = df_sorted.groupby('symbol')['benchmark_open'].shift(-1)
        bm_exit = df_sorted.groupby('symbol')['benchmark_open'].shift(-(1 + horizon))
        bm_ret = (bm_exit / bm_entry - 1.0)
        return stock_ret - bm_ret

    @staticmethod
    def compute_label_v4(df: pd.DataFrame, horizon: int = 20) -> pd.Series:
        """LABEL_V4: 逐日横截面行业中性化残差超额收益率"""
        v3_ret = LabelRegistry.compute_label_v3(df, horizon=horizon)
        temp_df = df[['date', 'industry']].copy()
        temp_df['v3_ret'] = v3_ret
        
        residuals = []
        for dt, grp in temp_df.groupby('date'):
            valid = grp.dropna(subset=['v3_ret'])
            if len(valid) < 10 or 'industry' not in valid.columns:
                residuals.append(grp['v3_ret'])
                continue
            # 行业均值中心化 (残差化)
            ind_mean = valid.groupby('industry')['v3_ret'].transform('mean')
            res = valid['v3_ret'] - ind_mean
            residuals.append(res)
        
        return pd.concat(residuals).reindex(df.index)

    @staticmethod
    def compute_label_v5(df: pd.DataFrame, horizon: int = 20, round_trip_cost_bps: float = 20.0) -> pd.Series:
        """LABEL_V5: 扣除双边 20bps 交易摩擦成本的可实现净 Alpha"""
        v3_ret = LabelRegistry.compute_label_v3(df, horizon=horizon)
        cost = round_trip_cost_bps / 10000.0
        return v3_ret - cost
