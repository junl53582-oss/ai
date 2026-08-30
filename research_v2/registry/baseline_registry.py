"""
V2 Baseline Registry (research_v2/registry/baseline_registry.py)
管理不可篡改的科学基准参照系 (如 LEGACY_BASELINE_V1)，防止历史真值被静默覆盖。
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .schemas import BaselineRecord, ComparisonVsBaseline

logger = logging.getLogger("baseline_registry")


class BaselineRegistry:
    """不可篡改基准注册表"""
    def __init__(self, baselines_dir: Optional[Path] = None):
        if baselines_dir is None:
            from config.settings import settings
            baselines_dir = settings.REPORTS_DIR / "baselines"
        self.baselines_dir = Path(baselines_dir)
        self._cache: Dict[str, BaselineRecord] = {}
        self._load_builtins()

    def _load_builtins(self):
        legacy_dir = self.baselines_dir / "legacy_v1"
        manifest_file = legacy_dir / "baseline_manifest.json"
        if manifest_file.exists():
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            record = BaselineRecord(
                baseline_id=data["baseline_id"],
                baseline_status=data["baseline_status"],
                created_at=data["created_at"],
                model_evidence_source_commit=data["model_evidence_source_commit"],
                dataset_sha256=data["dataset_sha256"],
                feature_schema_hash=data["feature_schema_hash"],
                prediction_baseline=data["prediction_baseline"],
                trading_candidate=data["trading_candidate"],
                legacy_label_semantics=data["legacy_label_semantics"],
                artifact_hashes=data.get("artifact_hashes", {}),
                immutable=True
            )
            self._cache["LEGACY_BASELINE_V1"] = record

    def register(self, record: BaselineRecord) -> None:
        if record.baseline_id in self._cache:
            existing = self._cache[record.baseline_id]
            if existing.immutable:
                raise ValueError(f"Baseline '{record.baseline_id}' is immutable and cannot be overwritten.")
        self._cache[record.baseline_id] = record

    def get(self, baseline_id: str) -> BaselineRecord:
        if baseline_id not in self._cache:
            self._load_builtins()
        if baseline_id not in self._cache:
            raise KeyError(f"Baseline '{baseline_id}' not found in registry.")
        return self._cache[baseline_id]

    def compare(
        self,
        candidate_pred: Dict[str, Any],
        candidate_trading: Dict[str, Any],
        baseline_id: str = "LEGACY_BASELINE_V1"
    ) -> ComparisonVsBaseline:
        """与指定基准进行因果差异比较"""
        base = self.get(baseline_id)
        b_pred = base.prediction_baseline
        b_trade = base.trading_candidate

        d_ic = candidate_pred.get("mean_daily_rank_ic", 0.0) - b_pred.get("mean_daily_rank_ic", 0.0)
        d_ir = candidate_pred.get("nw20_rank_icir", 0.0) - b_pred.get("nw20_rank_icir", 0.0)
        d_exc = candidate_trading.get("cost_adjusted_excess_return", 0.0) - b_trade.get("cost_adjusted_excess_return", 0.0)
        d_sharpe = candidate_trading.get("sharpe_ratio", 0.0) - b_trade.get("sharpe_ratio", 0.0)
        d_mdd = candidate_trading.get("max_drawdown", 0.0) - b_trade.get("max_drawdown", 0.0)
        d_win = candidate_trading.get("fold_win_ratio", 0.0) - b_trade.get("real_fold_win_ratio", b_trade.get("fold_win_ratio", 0.0))
        d_to = candidate_trading.get("annualized_turnover", 0.0) - b_trade.get("annualized_turnover_avg", b_trade.get("annualized_turnover", 0.0))

        robust = bool(d_ic > 0 and d_exc > 0 and d_sharpe > 0)
        return ComparisonVsBaseline(
            baseline_id=baseline_id,
            delta_rank_ic=round(d_ic, 5),
            delta_nw20_rankicir=round(d_ir, 4),
            delta_excess_return=round(d_exc, 2),
            delta_sharpe=round(d_sharpe, 2),
            delta_max_drawdown=round(d_mdd, 2),
            delta_fold_win_ratio=round(d_win, 4),
            delta_turnover=round(d_to, 2),
            robust_improvement=robust
        )
