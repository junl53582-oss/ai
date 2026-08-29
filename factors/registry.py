"""
因子注册器与元数据管理中心 (factors/registry.py)
提供基于装饰器的因子注册机制 (@register_factor)、因子分类索引、自动文档化与向量化批量计算。
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Set
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FactorMetadata:
    """因子元数据描述"""
    name: str
    func: Callable[[pd.DataFrame], pd.Series]
    category: str                         # momentum, volatility, volume, valuation, money_flow, etc.
    description: str
    required_cols: List[str] = field(default_factory=list)
    lookback_days: int = 0
    is_active: bool = True


class FactorRegistry:
    """因子注册中心单例"""
    _registry: Dict[str, FactorMetadata] = {}

    @classmethod
    def register(
        cls,
        name: str,
        category: str = "technical",
        description: str = "",
        required_cols: Optional[List[str]] = None,
        lookback_days: int = 0
    ):
        """
        因子注册装饰器
        用法:
        @FactorRegistry.register("ATR_RATIO_20", category="volatility", description="20日ATR与均线偏离度")
        def calc_atr_ratio(df: pd.DataFrame) -> pd.Series:
            ...
        """
        def decorator(func: Callable[[pd.DataFrame], pd.Series]):
            cls._registry[name] = FactorMetadata(
                name=name,
                func=func,
                category=category,
                description=description,
                required_cols=required_cols or ["close", "open", "high", "low", "volume"],
                lookback_days=lookback_days
            )
            return func
        return decorator

    @classmethod
    def get_factor(cls, name: str) -> Optional[FactorMetadata]:
        return cls._registry.get(name)

    @classmethod
    def list_all_factors(cls) -> List[str]:
        return sorted(list(cls._registry.keys()))

    @classmethod
    def get_factors_by_category(cls, category: str) -> List[str]:
        return sorted([k for k, v in cls._registry.items() if v.category == category])

    @classmethod
    def get_all_categories(cls) -> List[str]:
        return sorted(list(set(v.category for v in cls._registry.values())))

    @classmethod
    def get_metadata_df(cls) -> pd.DataFrame:
        """返回所有已注册因子的元数据 DataFrame 表"""
        rows = []
        for name, meta in cls._registry.items():
            rows.append({
                "factor_name": name,
                "category": meta.category,
                "description": meta.description,
                "lookback_days": meta.lookback_days,
                "is_active": meta.is_active
            })
        return pd.DataFrame(rows)

    @classmethod
    def compute_all_registered(cls, df: pd.DataFrame, categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        为输入的多股票 DataFrame 计算所有已注册的扩展因子。
        自动按股票分组计算，杜绝跨股票时序污染。
        """
        df = df.copy()
        if "symbol" not in df.columns:
            return df

        target_factors = [
            meta for name, meta in cls._registry.items()
            if meta.is_active and (categories is None or meta.category in categories)
        ]

        if not target_factors:
            return df

        logger.info(f"开始通过 FactorRegistry 批量计算 {len(target_factors)} 个扩展因子...")

        processed_dfs = []
        for sym, grp in df.groupby("symbol", group_keys=False):
            grp_copy = grp.copy()
            grp_copy.sort_values(by="date", inplace=True)
            for meta in target_factors:
                try:
                    grp_copy[meta.name] = meta.func(grp_copy)
                except Exception as e:
                    logger.warning(f"股票 {sym} 计算因子 {meta.name} 异常: {e}")
                    grp_copy[meta.name] = np.nan
            processed_dfs.append(grp_copy)

        res_df = pd.concat(processed_dfs, ignore_index=True) if processed_dfs else df
        res_df.sort_values(by=["date", "symbol"], inplace=True)
        res_df.reset_index(drop=True, inplace=True)
        return res_df
