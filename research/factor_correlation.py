"""
因子相关性、共线性与多维冗余聚类分析引擎 (research/factor_correlation.py)
Phase 1.3 P1-4: 综合因子值相关性与 IC 相关性，采用完全连接度聚类杜绝连通图链式误判
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import pandas as pd

from .config import ResearchConfig, default_research_config

logger = logging.getLogger(__name__)


@dataclass
class CorrelationAnalysisResult:
    """相关性与冗余分析结果"""
    factor_value_corr_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_ic_corr_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    high_corr_pairs: List[Dict[str, Any]] = field(default_factory=list)
    redundancy_groups: List[List[str]] = field(default_factory=list)
    factor_to_group_id: Dict[str, int] = field(default_factory=dict)


class FactorCorrelationEngine:
    """因子截面与时序相关性分析器"""

    @classmethod
    def analyze_correlation(
        cls,
        df: pd.DataFrame,
        factor_cols: List[str],
        daily_ic_dict: Optional[Dict[str, pd.Series]] = None,
        config: Optional[ResearchConfig] = None
    ) -> CorrelationAnalysisResult:
        """
        Phase 1.3 核心升级: 综合因子值相关性与 IC 相关性，基于完全连接度聚类
        """
        cfg = config or default_research_config
        valid_cols = [c for c in factor_cols if c in df.columns]
        if len(valid_cols) < 2:
            return CorrelationAnalysisResult()

        # 1. 因子值截面相关性矩阵
        sub_df = df[valid_cols].dropna(how="all")
        val_corr = sub_df.corr(method="spearman").fillna(0.0)

        # 2. 因子 IC 时序相关性矩阵
        if daily_ic_dict and len(daily_ic_dict) >= 2:
            ic_df = pd.DataFrame(daily_ic_dict).dropna(how="all")
            ic_corr = ic_df.corr(method="pearson").fillna(0.0)
        else:
            ic_corr = val_corr.copy()

        # 3. 识别高相关因子对 (综合 Value 与 IC 相关性)
        threshold = cfg.REDUNDANCY_CORR_THRESHOLD
        high_corr_pairs = []
        n = len(valid_cols)
        
        # 构建完全连接度相似矩阵
        is_pairwise_redundant = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(i + 1, n):
                f1 = valid_cols[i]
                f2 = valid_cols[j]
                
                v_c = float(val_corr.loc[f1, f2]) if f1 in val_corr.index and f2 in val_corr.columns else 0.0
                i_c = float(ic_corr.loc[f1, f2]) if f1 in ic_corr.index and f2 in ic_corr.columns else 0.0
                
                max_abs_corr = max(abs(v_c), abs(i_c))

                if max_abs_corr >= threshold:
                    is_pairwise_redundant[i, j] = True
                    is_pairwise_redundant[j, i] = True
                    high_corr_pairs.append({
                        "factor_1": f1,
                        "factor_2": f2,
                        "value_corr": round(v_c, 4),
                        "ic_corr": round(i_c, 4),
                        "max_corr": round(max_abs_corr, 4)
                    })

        # 4. 完全连接聚类 (Clique / Complete Linkage 防止连通图长链误伤 A~B~C 但 A!~C)
        groups: List[List[str]] = []
        assigned: Set[str] = set()

        for i, f1 in enumerate(valid_cols):
            if f1 in assigned:
                continue

            current_group = [f1]
            for j in range(i + 1, n):
                f2 = valid_cols[j]
                if f2 in assigned:
                    continue

                # 必须与当前 group 内所有成员均满足高相关 (Complete Linkage)
                can_join = True
                for member in current_group:
                    m_idx = valid_cols.index(member)
                    if not is_pairwise_redundant[m_idx, j]:
                        can_join = False
                        break

                if can_join:
                    current_group.append(f2)

            if len(current_group) > 1:
                groups.append(current_group)
                for member in current_group:
                    assigned.add(member)

        # 构建 factor -> group_id 映射
        factor_to_gid = {}
        for gid, grp in enumerate(groups, start=1):
            for f in grp:
                factor_to_gid[f] = gid

        return CorrelationAnalysisResult(
            factor_value_corr_matrix=val_corr,
            factor_ic_corr_matrix=ic_corr,
            high_corr_pairs=high_corr_pairs,
            redundancy_groups=groups,
            factor_to_group_id=factor_to_gid
        )
