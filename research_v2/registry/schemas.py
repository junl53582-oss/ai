"""
V2 Research Experiment & Baseline Schemas (research_v2/registry/schemas.py)
标准科学实验元数据实体与单变量因果对比数据结构。
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class PredictionMetrics:
    mean_daily_rank_ic: float
    nw20_rank_icir: float
    auc: Optional[float]
    q5_minus_q1_spread: float
    common_ranking_rows: int
    common_oos_dates: int
    ndcg_at_5: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    top5_net_alpha: Optional[float] = None
    top10_net_alpha: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradingMetrics:
    cost_adjusted_excess_return: float
    sharpe_ratio: float
    max_drawdown: float
    fold_win_ratio: float
    annualized_turnover: float
    total_costs: float
    total_filled_trades: int
    sortino_ratio: Optional[float] = None
    worst_fold_return: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonVsBaseline:
    baseline_id: str
    delta_rank_ic: float
    delta_nw20_rankicir: float
    delta_excess_return: float
    delta_sharpe: float
    delta_max_drawdown: float
    delta_fold_win_ratio: float
    delta_turnover: float
    robust_improvement: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRecord:
    experiment_id: str
    phase: str
    status: str
    created_at: str
    parent_baseline_id: str
    source_commit: str
    dataset_id: str
    dataset_sha256: str
    feature_set_id: str
    feature_set_hash: str
    label_schema_id: str
    label_schema_hash: str
    model_id: str
    model_config_hash: str
    primary_change: str
    controlled_variables: Dict[str, Any]
    train_window_years: float = 1.5
    val_window_months: int = 3
    test_window_months: int = 2
    purge_gap_days: int = 25
    random_seed: int = 42
    prediction_metrics: Optional[Dict[str, Any]] = None
    trading_metrics: Optional[Dict[str, Any]] = None
    comparison_vs_baseline: Optional[Dict[str, Any]] = None
    artifact_paths: List[str] = field(default_factory=list)
    artifact_hashes: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.primary_change:
            raise ValueError("primary_change is required for single-variable research discipline")
        if not self.controlled_variables:
            raise ValueError("controlled_variables is required to guarantee reproducibility")
        if not self.parent_baseline_id:
            raise ValueError("parent_baseline_id is required")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineRecord:
    baseline_id: str
    baseline_status: str
    created_at: str
    model_evidence_source_commit: str
    dataset_sha256: str
    feature_schema_hash: str
    prediction_baseline: Dict[str, Any]
    trading_candidate: Dict[str, Any]
    legacy_label_semantics: Dict[str, Any]
    artifact_hashes: Dict[str, str]
    immutable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
