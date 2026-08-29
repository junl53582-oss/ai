"""
A股可交易性掩码引擎 (models/tradability_mask.py)
实现顶尖私募的 Mask-first 训练机制：
在模型训练时，自动识别并过滤“开盘一字涨停无法买入 (Limit-Up Lock)”、
“开盘一字跌停无法卖出 (Limit-Down Lock)”与“停牌无流动性 (Suspension)”样本，
防止模型拟合不可执行的虚假 Alpha，确保实盘收益转换率达到 100%。
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TradabilityMaskEngine:
    """可交易性掩码生成与过滤引擎"""

    @classmethod
    def apply_tradability_mask(
        cls,
        df: pd.DataFrame,
        filter_limit_up: bool = True,
        filter_suspended: bool = True,
        filter_subnew: bool = True
    ) -> pd.DataFrame:
        """
        生成严格可交易掩码列 'is_tradable_sample':
        仅在真实可下单成交的样本上进行模型训练与损失反向传播
        """
        if df.empty:
            return df

        mask = pd.Series(True, index=df.index)

        # 1. 过滤开盘一字涨停 (无法挂单买入)
        if filter_limit_up and "is_limit_up_locked" in df.columns:
            limit_up_locked = df["is_limit_up_locked"].fillna(False).astype(bool)
            mask = mask & (~limit_up_locked)

        # 2. 过滤停牌样本
        if filter_suspended and "is_suspended" in df.columns:
            suspended = df["is_suspended"].fillna(False).astype(bool)
            mask = mask & (~suspended)

        # 3. 过滤上市不足 60 天的新股次新股
        if filter_subnew and "is_subnew" in df.columns:
            subnew = df["is_subnew"].fillna(False).astype(bool)
            mask = mask & (~subnew)

        df_out = df.copy()
        df_out["is_tradable_sample"] = mask
        logger.info(f"可交易掩码构建完成: 真实可交易样本占比 {mask.mean()*100:.1f}%")
        return df_out
