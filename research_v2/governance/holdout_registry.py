"""
Research Holdout & Training Pool Governance Registry (research_v2/governance/holdout_registry.py)

1. Holdout Governance:
   - HISTORICAL_RESEARCH_OOS: 历史已被特征/超参/模型选择触碰过的数据区间，严禁伪装为未触碰的 holdout。
   - VALIDATION_RESEARCH_POOL: 走步验证与超参对比池。
   - PROSPECTIVE_HOLDOUT: 模型/策略完全冻结后新产生的严格未来盲测数据 (需等待实际未来交易日产生)。
   - 当前事实声明: FINAL_HOLDOUT_AVAILABLE = False (绝不伪造历史区间为新 holdout)。

2. Phase 2.1-C Two Training Pools:
   - OBJECTIVE_COMMON_TRAIN_POOL: 仅用于 Classification vs Regression vs LambdaRank controlled comparison (保持样本一致)。
   - REGRESSION_NATIVE_TRAIN_POOL: 用于未来 Phase 2.1-C 最大化连续回归学习能力 (label_valid & net_alpha.notna & in_universe & not_excluded，不要求分类极值标签非空)。
   - LAMBDARANK_NATIVE_TRAIN_POOL: 由全部有效 continuous samples 按日生成 relevance grade。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class DatasetGovernanceStatus(str, enum.Enum):
    HISTORICAL_RESEARCH_OOS = "HISTORICAL_RESEARCH_OOS"
    VALIDATION_RESEARCH_POOL = "VALIDATION_RESEARCH_POOL"
    PROSPECTIVE_HOLDOUT = "PROSPECTIVE_HOLDOUT"


class TrainingPoolType(str, enum.Enum):
    OBJECTIVE_COMMON_TRAIN_POOL = "OBJECTIVE_COMMON_TRAIN_POOL"
    REGRESSION_NATIVE_TRAIN_POOL = "REGRESSION_NATIVE_TRAIN_POOL"
    LAMBDARANK_NATIVE_TRAIN_POOL = "LAMBDARANK_NATIVE_TRAIN_POOL"


@dataclass(frozen=True)
class HoldoutGovernanceManifest:
    dataset_sha256: str
    date_start: str
    date_end: str
    created_at: str
    freeze_commit_sha: str
    feature_schema_hash: str
    model_config_hash: str
    selection_history: Dict[str, Any]
    ever_used_for_feature_selection: bool
    ever_used_for_hyperparameter_selection: bool
    ever_used_for_portfolio_parameter_selection: bool
    ever_used_for_model_selection: bool
    holdout_status: str
    final_holdout_available: bool = False


def build_objective_common_train_pool(
    df_labeled: pd.DataFrame,
    label_clf_col: str = "label_up_down_20d",
    exec_direction_col: str = "label_direction_20d"
) -> pd.Series:
    """
    构造 3-Arm 严格受控实验共同训练池：
    必须同时满足分类标签有效，用于公平比较目标函数。
    """
    in_univ = df_labeled["in_universe"].fillna(False).astype(bool) if "in_universe" in df_labeled.columns else pd.Series(True, index=df_labeled.index)
    not_excl = ~df_labeled["excluded_from_training"].fillna(False).astype(bool) if "excluded_from_training" in df_labeled.columns else pd.Series(True, index=df_labeled.index)

    return (
        df_labeled["label_valid"].fillna(False).astype(bool)
        & df_labeled[label_clf_col].notna()
        & df_labeled[exec_direction_col].notna()
        & in_univ
        & not_excl
    )


def build_regression_native_train_pool(
    df_labeled: pd.DataFrame,
    exec_label_col: str = "label_net_alpha_20d"
) -> pd.Series:
    """
    构造 Continuous Regression Native 训练池 (Phase 2.1-C):
    仅要求执行标签有效，不要求分类极值标签非空，最大化样本量与连续学习能力。
    """
    in_univ = df_labeled["in_universe"].fillna(False).astype(bool) if "in_universe" in df_labeled.columns else pd.Series(True, index=df_labeled.index)
    not_excl = ~df_labeled["excluded_from_training"].fillna(False).astype(bool) if "excluded_from_training" in df_labeled.columns else pd.Series(True, index=df_labeled.index)

    return (
        df_labeled["label_valid"].fillna(False).astype(bool)
        & df_labeled[exec_label_col].notna()
        & in_univ
        & not_excl
    )


def build_lambdarank_native_train_pool(
    df_labeled: pd.DataFrame,
    exec_label_col: str = "label_net_alpha_20d"
) -> pd.Series:
    """
    构造 LambdaRank Native 训练池:
    由全部有效连续收益样本在日截面内生成等级。
    """
    return build_regression_native_train_pool(df_labeled, exec_label_col)
