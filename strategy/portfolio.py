"""
Top-K 截面选股与投资组合构建器 (strategy/portfolio.py)
支持 Top-K 缓冲区 (Hysteresis Buffer: Top_K_Buy vs Top_K_Hold) 降低换手率，
支持硬性行业持仓集中度上限约束 (MAX_SECTOR_EXPOSURE，包括 UNKNOWN 行业同样受限，超限不放宽直接保留现金)
支持严格基于 PointInTimeUniverseProvider 按日期过滤候选股票池
"""
import logging
from typing import List, Dict, Any, Optional, Set, Union
import pandas as pd
import numpy as np

from config.settings import settings
from .trading_rules import AShareTradingRules
from data.universe_provider import UniverseProvider

logger = logging.getLogger(__name__)


class PortfolioBuilder:
    """组合构建与调仓信号发生器 (支持 Hysteresis 缓冲区、硬性行业上限与 Point-In-Time 股票池过滤)"""

    def __init__(
        self,
        top_k_buy: int = settings.TOP_K_BUY,
        top_k_hold: int = settings.TOP_K_HOLD,
        max_sector_exposure: float = settings.MAX_SECTOR_EXPOSURE,
        weight_method: str = "equal",
        optimizer_type: Optional[str] = None,
        trading_rules: Optional[AShareTradingRules] = None,
        universe_provider: Optional[UniverseProvider] = None
    ):
        self.top_k_buy = top_k_buy
        self.top_k_hold = top_k_hold
        self.max_sector_exposure = max_sector_exposure
        self.weight_method = optimizer_type or weight_method
        self.rules = trading_rules or AShareTradingRules()
        self.universe_provider = universe_provider

        # 审计语义字段
        self.sector_cap_enabled: bool = True
        self.industry_data_available: bool = False
        self.unknown_industry_cap_applied: bool = False
        self.industry_coverage_ratio: float = 0.0
        self.unknown_industry_weight: float = 0.0
        self.sector_constraint_enabled: bool = True # 兼容旧字段

    def build_target_portfolio(
        self,
        daily_df: pd.DataFrame,
        current_holdings: Set[str],
        date: Optional[Union[str, pd.Timestamp]] = None
    ) -> pd.DataFrame:
        """
        在 T 日收盘截面上生成目标组合：
        1. 若配置了 universe_provider，先按当前 date 严格过滤 Point-In-Time 有效成分股
        2. 排除停牌标的
        3. 按 pred_score 降序排名 (Rank 1 为最优)
        4. 缓冲区规则：
           - 对于已有持仓：若排名 <= top_k_hold，继续持有；若排名 > top_k_hold，标记剔除
           - 对于未持仓：若排名 <= top_k_buy，候选买入
        5. 计算分配权重 target_weight 并执行硬性行业集中度约束 (MAX_SECTOR_EXPOSURE，包含 UNKNOWN 行业)
        """
        df = daily_df.copy()

        # 1. 严格 Point-In-Time 股票池过滤 (P0-4, P0-5)
        if self.universe_provider is not None:
            active_univ = set(self.universe_provider.get_universe(date))
            df = df[df["symbol"].isin(active_univ)].copy()
        elif "in_universe" in df.columns:
            in_univ_mask = df["in_universe"].fillna(False).astype(bool)
            df = df[in_univ_mask].copy()

        tradable_df = df[~df.get("is_suspended", False)].copy()
        if tradable_df.empty:
            return pd.DataFrame()

        sorted_df = tradable_df.sort_values(by="pred_score", ascending=False).reset_index(drop=True)
        sorted_df["rank"] = np.arange(len(sorted_df)) + 1

        selected_rows = []
        for _, row in sorted_df.iterrows():
            sym = row["symbol"]
            rank = row["rank"]
            is_held = sym in current_holdings

            if is_held:
                if rank <= self.top_k_hold:
                    selected_rows.append(row)
            else:
                if rank <= self.top_k_buy:
                    selected_rows.append(row)

        if not selected_rows:
            selected_df = sorted_df.head(self.top_k_buy).copy()
        else:
            selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
            if len(selected_df) > self.top_k_hold:
                selected_df = selected_df.head(self.top_k_hold).copy()

        # 统一缺失行业为 UNKNOWN
        if "industry" not in selected_df.columns:
            selected_df["industry"] = "UNKNOWN"
        else:
            selected_df["industry"] = selected_df["industry"].fillna("UNKNOWN")
            selected_df.loc[selected_df["industry"].astype(str).str.strip() == "", "industry"] = "UNKNOWN"

        # 统计行业覆盖与审计语义
        non_unknown_count = int((selected_df["industry"] != "UNKNOWN").sum())
        self.industry_coverage_ratio = round(non_unknown_count / max(len(selected_df), 1), 4)
        self.industry_data_available = bool(non_unknown_count > 0)
        self.sector_cap_enabled = True

        # 计算基础归一化权重 (总和为 1.0)
        raw_weights = self._compute_weights(selected_df)
        selected_df["target_weight"] = raw_weights.values

        # 行业暴露上限硬约束 (任何行业包括 UNKNOWN 均严格受限，超额留存现金)
        selected_df["target_weight"] = self._apply_sector_caps(selected_df)

        unknown_mask = (selected_df["industry"] == "UNKNOWN")
        self.unknown_industry_weight = round(float(selected_df.loc[unknown_mask, "target_weight"].sum()), 4)
        self.unknown_industry_cap_applied = bool(unknown_mask.any())
        self.sector_constraint_enabled = True

        return selected_df

    def _compute_weights(self, selected_df: pd.DataFrame) -> pd.Series:
        """计算组合目标持仓权重"""
        k = len(selected_df)
        if k == 0:
            return pd.Series(dtype=float)

        from .optimizer import get_optimizer
        optimizer = get_optimizer(self.weight_method)
        return optimizer.optimize(
            selected_df,
            max_sector_exposure=self.max_sector_exposure
        )

    def _apply_sector_caps(self, df: pd.DataFrame) -> pd.Series:
        """
        硬性行业暴露上限约束：
        对于任何行业 s（包括 UNKNOWN），若其总权重 raw_sec_w > max_sector_exposure：
        则将其行业总权重硬截断为 max_sector_exposure，行业内个股同比例缩放。
        严禁自动放宽上限！未分配的剩余权重自然保留为现金。
        """
        weights = df["target_weight"].copy().values.astype(float)
        industries = df["industry"].values

        unique_ind = np.unique(industries[pd.notna(industries)])
        if len(unique_ind) == 0:
            return pd.Series(weights, index=df.index)

        final_weights = weights.copy()
        for ind in unique_ind:
            mask = (industries == ind)
            sec_w = weights[mask].sum()
            if sec_w > self.max_sector_exposure:
                scale = self.max_sector_exposure / sec_w
                final_weights[mask] = weights[mask] * scale

        return pd.Series(final_weights, index=df.index)
