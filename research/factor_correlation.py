"""
因子相关性与高冗余聚集分析引擎 (research/factor_correlation.py)
计算因子截面特征相关性、日度 IC 序列相关性，并对高相关冗余因子组进行聚类识别。
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
import pandas as pd

from .config import ResearchConfig, default_research_config

logger = logging.getLogger(__name__)


@dataclass
class CorrelationAnalysisResult:
    """多因子相关性与冗余分析结果"""
    factor_value_corr_matrix: pd.DataFrame
    factor_ic_corr_matrix: pd.DataFrame
    redundancy_groups: List[List[str]] = field(default_factory=list)
    redundant_pairs: List[Dict[str, Any]] = field(default_factory=list) # [{"factor_a", "factor_b", "correlation"}]
    factor_to_group_id: Dict[str, int] = field(default_factory=dict)


class FactorCorrelationEngine:
    """因子相关性与冗余分析核心引擎"""

    @classmethod
    def compute_cross_sectional_correlation(
        cls,
        df: pd.DataFrame,
        factor_cols: List[str],
        date_col: str = "date"
    ) -> pd.DataFrame:
        """
        计算因子特征截面 Spearman 秩相关系数矩阵 (每日计算后取时间序列均值)
        """
        valid_cols = [c for c in factor_cols if c in df.columns]
        if len(valid_cols) < 2:
            return pd.DataFrame(1.0, index=valid_cols, columns=valid_cols)

        # 每日截面计算相关矩阵
        daily_corrs = []
        for _, grp in df.groupby(date_col):
            sub = grp[valid_cols].dropna(how="all")
            if len(sub) >= 5:
                corr_m = sub.corr(method="spearman")
                daily_corrs.append(corr_m)

        if not daily_corrs:
            return pd.DataFrame(np.eye(len(valid_cols)), index=valid_cols, columns=valid_cols)

        # 取时间平均
        avg_corr = pd.concat(daily_corrs).groupby(level=0).mean()
        avg_corr = avg_corr.loc[valid_cols, valid_cols].fillna(0.0)
        np.fill_diagonal(avg_corr.values, 1.0)
        return avg_corr.round(4)

    @classmethod
    def compute_ic_correlation(
        cls,
        daily_ic_dict: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        计算因子日度 IC 序列之间的相关性矩阵 (捕捉预测信号层面的冗余度)
        """
        if not daily_ic_dict:
            return pd.DataFrame()

        ic_df = pd.DataFrame(daily_ic_dict).dropna(how="all")
        if ic_df.empty or len(ic_df.columns) < 2:
            cols = list(daily_ic_dict.keys())
            return pd.DataFrame(1.0, index=cols, columns=cols)

        ic_corr = ic_df.corr(method="pearson").round(4)
        return ic_corr

    @classmethod
    def identify_redundancy(
        cls,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.85
    ) -> Tuple[List[List[str]], List[Dict[str, Any]], Dict[str, int]]:
        """
        识别高相关冗余对，并使用连通图算法划分冗余因子群组 (Redundancy Groups)
        """
        factors = list(corr_matrix.columns)
        redundant_pairs = []
        adj: Dict[str, Set[str]] = {f: set() for f in factors}

        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                f1 = factors[i]
                f2 = factors[j]
                val = float(corr_matrix.loc[f1, f2])
                if abs(val) >= threshold:
                    redundant_pairs.append({
                        "factor_a": f1,
                        "factor_b": f2,
                        "correlation": round(val, 4),
                        "abs_correlation": round(abs(val), 4)
                    })
                    adj[f1].add(f2)
                    adj[f2].add(f1)

        # 连通分量划分聚类群组
        visited = set()
        groups = []
        factor_to_group = {}

        group_id = 1
        for f in factors:
            if f not in visited:
                component = []
                queue = [f]
                visited.add(f)
                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                if len(component) > 1:
                    groups.append(sorted(component))
                    for m in component:
                        factor_to_group[m] = group_id
                    group_id += 1
                else:
                    factor_to_group[f] = 0 # 独立非冗余因子

        # 按关联度降序排列
        redundant_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
        return groups, redundant_pairs, factor_to_group

    @classmethod
    def analyze_correlation(
        cls,
        df: pd.DataFrame,
        factor_cols: List[str],
        daily_ic_dict: Dict[str, pd.Series],
        config: Optional[ResearchConfig] = None
    ) -> CorrelationAnalysisResult:
        """多因子相关性全套评估"""
        cfg = config or default_research_config
        val_corr = cls.compute_cross_sectional_correlation(df, factor_cols)
        ic_corr = cls.compute_ic_correlation(daily_ic_dict)
        groups, pairs, factor_map = cls.identify_redundancy(val_corr, threshold=cfg.REDUNDANCY_CORR_THRESHOLD)

        return CorrelationAnalysisResult(
            factor_value_corr_matrix=val_corr,
            factor_ic_corr_matrix=ic_corr,
            redundancy_groups=groups,
            redundant_pairs=pairs,
            factor_to_group_id=factor_map
        )
