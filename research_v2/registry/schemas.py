"""
V2 Research Experiment & Baseline Schemas (research_v2/registry/schemas.py)
标准科学实验元数据实体与单变量因果对比数据结构。
"""
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


class MissingExperimentMetricError(ValueError):
    """缺失必要对比指标异常"""
    pass


class BaselineIntegrityError(RuntimeError):
    """基准完整性与防伪校验异常"""
    pass


def require_metric(metrics: Dict[str, Any], key: str) -> float:
    """提取必要指标，若缺失、None、NaN、Inf 则抛出异常"""
    if key not in metrics or metrics[key] is None:
        raise MissingExperimentMetricError(f"Missing required experiment metric '{key}'")
    val = metrics[key]
    try:
        f_val = float(val)
    except (ValueError, TypeError):
        raise MissingExperimentMetricError(f"Metric '{key}' cannot be converted to float: {val}")
    if math.isnan(f_val) or math.isinf(f_val):
        raise MissingExperimentMetricError(f"Metric '{key}' is invalid (NaN or Inf): {val}")
    return f_val


@dataclass
class PredictionMetrics:
    mean_daily_rank_ic: float
    nw20_rank_icir: float
    q5_minus_q1_spread: float
    common_ranking_rows: int
    common_oos_dates: int
    auc: Optional[float] = None
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
    comparison_status: str  # "COMPARABLE" | "NOT_COMPARABLE"
    delta_rank_ic: Optional[float] = None
    delta_nw20_rankicir: Optional[float] = None
    delta_excess_return: Optional[float] = None
    delta_sharpe: Optional[float] = None
    delta_max_drawdown: Optional[float] = None
    delta_fold_win_ratio: Optional[float] = None
    delta_turnover: Optional[float] = None
    robust_improvement: Optional[bool] = None
    missing_metrics: List[str] = field(default_factory=list)

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
        if not self.experiment_id or not isinstance(self.experiment_id, str):
            raise ValueError("experiment_id is required and must be a string")
        if not self.primary_change or not isinstance(self.primary_change, str):
            raise ValueError("primary_change is required for single-variable research discipline")
        if not self.controlled_variables or not isinstance(self.controlled_variables, dict):
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
    certification_logic_source_commit: str
    certified_artifact_commit: str
    dataset_sha256: str
    feature_schema_hash: str
    prediction_baseline: Dict[str, Any]
    trading_candidate: Dict[str, Any]
    legacy_label_semantics: Dict[str, Any]
    artifact_hash_manifest_sha256: str
    artifact_hashes: Dict[str, str]
    immutable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
