"""
V2 Baseline Registry (research_v2/registry/baseline_registry.py)
管理不可篡改的科学基准参照系 (如 LEGACY_BASELINE_V1)，具备密码级文件防伪与完整性校验。
"""
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from .schemas import (
    BaselineRecord,
    ComparisonVsBaseline,
    BaselineIntegrityError,
    MissingExperimentMetricError,
    require_metric
)

logger = logging.getLogger("baseline_registry")


def compute_file_sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found for hash calculation: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BaselineRegistry:
    """不可篡改基准注册表（支持磁盘文件防伪完整性验证）"""
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
        hashes_file = legacy_dir / "artifact_hashes.json"

        if manifest_file.exists() and hashes_file.exists():
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            hashes_data = json.loads(hashes_file.read_text(encoding="utf-8"))
            
            artifacts_dict = hashes_data.get("artifacts", {})
            record = BaselineRecord(
                baseline_id=manifest_data["baseline_id"],
                baseline_status=manifest_data["baseline_status"],
                created_at=manifest_data["created_at"],
                model_evidence_source_commit=manifest_data["model_evidence_source_commit"],
                certification_logic_source_commit=manifest_data.get("certification_logic_source_commit", ""),
                certified_artifact_commit=manifest_data.get("certified_artifact_commit", ""),
                dataset_sha256=manifest_data["dataset_sha256"],
                feature_schema_hash=manifest_data["feature_schema_hash"],
                prediction_baseline=manifest_data["prediction_baseline"],
                trading_candidate=manifest_data["trading_candidate"],
                legacy_label_semantics=manifest_data["legacy_label_semantics"],
                artifact_hash_manifest_sha256=manifest_data.get("artifact_hash_manifest_sha256", ""),
                artifact_hashes=artifacts_dict,
                immutable=True
            )
            self._cache["LEGACY_BASELINE_V1"] = record

    def verify_integrity(self, baseline_id: str = "LEGACY_BASELINE_V1") -> bool:
        """
        全量检验基准文件防伪完整性：
        1. 验证 artifact_hashes.json SHA256 与 manifest 中记录一致
        2. 验证每个冻结产物文件在磁盘上的真实 SHA256 与 artifact_hashes.json 一致
        """
        legacy_dir = self.baselines_dir / "legacy_v1" if baseline_id == "LEGACY_BASELINE_V1" else self.baselines_dir / baseline_id.lower()
        manifest_file = legacy_dir / "baseline_manifest.json"
        hashes_file = legacy_dir / "artifact_hashes.json"

        if not manifest_file.exists():
            raise BaselineIntegrityError(f"Baseline manifest file not found: {manifest_file}")
        if not hashes_file.exists():
            raise BaselineIntegrityError(f"Artifact hashes manifest not found: {hashes_file}")

        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        expected_manifest_hash = manifest_data.get("artifact_hash_manifest_sha256")
        if not expected_manifest_hash:
            raise BaselineIntegrityError("baseline_manifest.json missing 'artifact_hash_manifest_sha256'")

        actual_hashes_file_sha = compute_file_sha256(hashes_file)
        if actual_hashes_file_sha != expected_manifest_hash:
            raise BaselineIntegrityError(
                f"artifact_hashes.json tamper detected! Expected: {expected_manifest_hash}, Actual: {actual_hashes_file_sha}"
            )

        hashes_data = json.loads(hashes_file.read_text(encoding="utf-8"))
        artifacts = hashes_data.get("artifacts", {})

        for filename, expected_sha in artifacts.items():
            f_path = legacy_dir / filename
            if not f_path.exists():
                raise BaselineIntegrityError(f"Frozen artifact '{filename}' missing on disk: {f_path}")
            actual_sha = compute_file_sha256(f_path)
            if actual_sha != expected_sha:
                raise BaselineIntegrityError(
                    f"Tamper detected in artifact '{filename}'! Expected SHA: {expected_sha}, Actual: {actual_sha}"
                )

        return True

    def register(self, record: BaselineRecord) -> None:
        if record.baseline_id in self._cache:
            existing = self._cache[record.baseline_id]
            if existing.immutable:
                raise ValueError(f"Baseline '{record.baseline_id}' is immutable and cannot be overwritten.")
        self._cache[record.baseline_id] = record

    def get(self, baseline_id: str, verify_integrity: bool = True) -> BaselineRecord:
        if verify_integrity:
            self.verify_integrity(baseline_id)
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
        """
        与指定基准进行因果差异比较 (Fail-Closed, Missing != 0):
        必要指标缺失时返回 NOT_COMPARABLE，禁止以 0 代替缺失值。
        """
        base = self.get(baseline_id, verify_integrity=True)
        b_pred = base.prediction_baseline
        b_trade = base.trading_candidate

        required_pred_keys = ["mean_daily_rank_ic", "nw20_rank_icir"]
        required_trade_keys = ["cost_adjusted_excess_return", "sharpe_ratio", "max_drawdown", "fold_win_ratio", "annualized_turnover"]

        missing_keys: List[str] = []
        for k in required_pred_keys:
            if k not in candidate_pred or candidate_pred[k] is None:
                missing_keys.append(f"pred.{k}")
        for k in required_trade_keys:
            if k not in candidate_trading or candidate_trading[k] is None:
                missing_keys.append(f"trading.{k}")

        if missing_keys:
            return ComparisonVsBaseline(
                baseline_id=baseline_id,
                comparison_status="NOT_COMPARABLE",
                robust_improvement=None,
                missing_metrics=missing_keys
            )

        try:
            cand_ic = require_metric(candidate_pred, "mean_daily_rank_ic")
            cand_ir = require_metric(candidate_pred, "nw20_rank_icir")
            cand_exc = require_metric(candidate_trading, "cost_adjusted_excess_return")
            cand_sharpe = require_metric(candidate_trading, "sharpe_ratio")
            cand_mdd = require_metric(candidate_trading, "max_drawdown")
            cand_win = require_metric(candidate_trading, "fold_win_ratio")
            cand_to = require_metric(candidate_trading, "annualized_turnover")
        except MissingExperimentMetricError as e:
            return ComparisonVsBaseline(
                baseline_id=baseline_id,
                comparison_status="NOT_COMPARABLE",
                robust_improvement=None,
                missing_metrics=[str(e)]
            )

        base_ic = require_metric(b_pred, "mean_daily_rank_ic")
        base_ir = require_metric(b_pred, "nw20_rank_icir")
        base_exc = require_metric(b_trade, "cost_adjusted_excess_return")
        base_sharpe = require_metric(b_trade, "sharpe_ratio")
        base_mdd = require_metric(b_trade, "max_drawdown")
        base_win = require_metric(b_trade, "real_fold_win_ratio") if "real_fold_win_ratio" in b_trade else require_metric(b_trade, "fold_win_ratio")
        base_to = require_metric(b_trade, "annualized_turnover_avg") if "annualized_turnover_avg" in b_trade else require_metric(b_trade, "annualized_turnover")

        d_ic = cand_ic - base_ic
        d_ir = cand_ir - base_ir
        d_exc = cand_exc - base_exc
        d_sharpe = cand_sharpe - base_sharpe
        d_mdd = cand_mdd - base_mdd
        d_win = cand_win - base_win
        d_to = cand_to - base_to

        robust = bool(d_ic > 0 and d_exc > 0 and d_sharpe > 0)
        return ComparisonVsBaseline(
            baseline_id=baseline_id,
            comparison_status="COMPARABLE",
            delta_rank_ic=round(d_ic, 5),
            delta_nw20_rankicir=round(d_ir, 4),
            delta_excess_return=round(d_exc, 2),
            delta_sharpe=round(d_sharpe, 2),
            delta_max_drawdown=round(d_mdd, 2),
            delta_fold_win_ratio=round(d_win, 4),
            delta_turnover=round(d_to, 2),
            robust_improvement=robust,
            missing_metrics=[]
        )
