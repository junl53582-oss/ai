"""
Tests for Experiment Registry & Baseline Registry
(tests/test_phase2_1_experiment_registry.py)
"""
import json
from pathlib import Path

import pytest

from research_v2.registry.schemas import (
    ExperimentRecord,
    ExperimentIntegrityError,
    BaselineIntegrityError,
)
from research_v2.registry.baseline_registry import BaselineRegistry
from research_v2.registry.experiment_registry import ExperimentRegistry


class _DummyBaselineRegistry:
    def get(self, baseline_id):
        if baseline_id != "LEGACY_BASELINE_V1":
            raise KeyError(baseline_id)
        return object()


def _make_experiment(experiment_id: str = "exp_001") -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=experiment_id,
        phase="2.1-A",
        status="PLANNED",
        created_at="2026-08-31T00:00:00Z",
        parent_baseline_id="LEGACY_BASELINE_V1",
        source_commit="a" * 40,
        dataset_id="dataset_v1",
        dataset_sha256="b" * 64,
        feature_set_id="FEAT_V1",
        feature_set_hash="c" * 64,
        label_schema_id="EXECUTION_ALIGNED_V1",
        label_schema_hash="d" * 64,
        model_id="lightgbm_clf_baseline",
        model_config_hash="e" * 64,
        primary_change="execution_aligned_label",
        controlled_variables={"feature_set": "frozen"},
    )


def test_baseline_registry_load_legacy_v1():
    reg = BaselineRegistry()
    base = reg.get("LEGACY_BASELINE_V1", verify_integrity=True)
    assert base.baseline_id == "LEGACY_BASELINE_V1"
    assert base.prediction_baseline["mean_daily_rank_ic"] == 0.0503
    assert base.trading_candidate["cost_adjusted_excess_return"] == 5.72


def test_baseline_registry_tamper_detection(tmp_path):
    src_dir = Path("reports/baselines")
    dst_dir = tmp_path / "baselines"
    import shutil

    shutil.copytree(src_dir, dst_dir)

    target_csv = dst_dir / "legacy_v1" / "model_comparison.csv"
    text = target_csv.read_text(encoding="utf-8")
    target_csv.write_text(
        text + "\n# TAMPERED_DATA",
        encoding="utf-8",
    )

    reg = BaselineRegistry(
        baselines_dir=dst_dir,
        project_root=Path.cwd(),
    )
    with pytest.raises(
        BaselineIntegrityError,
        match="Git seal mismatch|Tamper detected",
    ):
        reg.verify_integrity("LEGACY_BASELINE_V1")


def test_baseline_immutability():
    reg = BaselineRegistry()
    base = reg.get("LEGACY_BASELINE_V1", verify_integrity=True)
    with pytest.raises(
        ValueError,
        match="immutable and cannot be overwritten",
    ):
        reg.register(base)


def test_missing_metric_comparison_fail_closed():
    reg = BaselineRegistry()
    cand_pred = {
        "mean_daily_rank_ic": 0.0550,
        "nw20_rank_icir": 0.4500,
    }
    cand_trading_missing_sharpe = {
        "cost_adjusted_excess_return": 7.50,
        "max_drawdown": -12.00,
        "fold_win_ratio": 0.60,
        "annualized_turnover": 8.50,
    }
    comp = reg.compare(
        cand_pred,
        cand_trading_missing_sharpe,
        baseline_id="LEGACY_BASELINE_V1",
    )
    assert comp.comparison_status == "NOT_COMPARABLE"
    assert comp.robust_improvement is None
    assert "trading.sharpe_ratio" in comp.missing_metrics


def test_valid_baseline_comparison():
    reg = BaselineRegistry()
    cand_pred = {
        "mean_daily_rank_ic": 0.0550,
        "nw20_rank_icir": 0.4500,
    }
    cand_trading = {
        "cost_adjusted_excess_return": 7.50,
        "sharpe_ratio": 0.45,
        "max_drawdown": -12.00,
        "fold_win_ratio": 0.60,
        "annualized_turnover": 8.50,
    }
    comp = reg.compare(
        cand_pred,
        cand_trading,
        baseline_id="LEGACY_BASELINE_V1",
    )
    assert comp.comparison_status == "COMPARABLE"
    assert comp.delta_rank_ic == round(0.0550 - 0.0503, 5)
    assert comp.delta_excess_return == round(7.50 - 5.72, 2)
    assert comp.robust_improvement is True


def test_experiment_registry_rejects_malformed_manifest(tmp_path):
    exp_dir = tmp_path / "experiments" / "broken"
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment_manifest.json").write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentIntegrityError,
        match="Invalid experiment manifest",
    ):
        ExperimentRegistry(
            experiments_dir=tmp_path / "experiments",
            baseline_registry=_DummyBaselineRegistry(),
        )


def test_experiment_registry_rejects_path_id_mismatch(tmp_path):
    exp_dir = tmp_path / "experiments" / "folder_id"
    exp_dir.mkdir(parents=True)
    record = _make_experiment("manifest_id")
    (exp_dir / "experiment_manifest.json").write_text(
        json.dumps(record.to_dict(), indent=2),
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentIntegrityError,
        match="path/id mismatch",
    ):
        ExperimentRegistry(
            experiments_dir=tmp_path / "experiments",
            baseline_registry=_DummyBaselineRegistry(),
        )


def test_experiment_registry_persists_atomically_and_rejects_duplicate(tmp_path):
    registry = ExperimentRegistry(
        experiments_dir=tmp_path / "experiments",
        baseline_registry=_DummyBaselineRegistry(),
    )
    record = _make_experiment("exp_atomic")
    registry.register_experiment(record)

    manifest_path = (
        tmp_path
        / "experiments"
        / "exp_atomic"
        / "experiment_manifest.json"
    )
    assert manifest_path.exists()
    assert not (
        manifest_path.parent / ".experiment_manifest.json.tmp"
    ).exists()

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["experiment_id"] == "exp_atomic"

    with pytest.raises(ValueError, match="Duplicate experiment_id"):
        registry.register_experiment(record)
