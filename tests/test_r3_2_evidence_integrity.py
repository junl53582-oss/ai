import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_v2.governance.certification import evaluate_research_gates, validate_artifact_hash_chain, validate_final_run_pointer
from research_v2.governance.holdout_registry import build_regression_native_train_pool
from tools.run_model_research import paired_block_bootstrap
from models.walk_forward import WalkForwardTrainer


def _write_evidence(root: Path, *, seed_std=0.008, holdout=False):
    source = "a" * 40
    dates = ["2023-01-02", "2023-01-03", "2023-01-04"]
    fold = {k: 1 for k in ("test_dates_count", "candidate_oos_rows", "baseline_oos_rows", "candidate_trade_count", "baseline_trade_count")}
    fold.update({"fold_id": 1, "test_start": dates[0], "test_end": dates[-1], "candidate_prediction_sha256": "a", "baseline_prediction_sha256": "b", "candidate_backtest_run_id": "c", "baseline_backtest_run_id": "d", "candidate_equity_sha256": "e", "baseline_equity_sha256": "f", "candidate_orders_sha256": "g", "baseline_orders_sha256": "h", "candidate_strategy_return": 0.1, "baseline_strategy_return": 0.0, "candidate_benchmark_return": 0.0, "baseline_benchmark_return": 0.0, "candidate_excess_return": 0.1, "baseline_excess_return": 0.0, "candidate_minus_baseline": 0.1, "engine_config_hash": "i", "dataset_sha256": "j", "source_code_sha": source})
    payloads = {
        "fold_backtest_provenance.json": {"run_mode": "certified", "folds": [fold]},
        "production_isolation.json": {"before": {}, "after": {}, "file_sha_before": "", "file_sha_after": ""},
        "walk_forward_purge_audit.json": [{"configured_purge_gap_trading_days": 20, "label_horizon_trading_days": 20, "actual_train_val_gap_trading_days": 20, "actual_val_test_gap_trading_days": 20}],
        "calendar_metadata.json": {"run_mode": "certified", "calendar_source": "exchange", "calendar_artifact_sha256": "calendar-artifact", "dates": dates, "dataset_overlap_count": 3, "calendar_sha256": hashlib.sha256("\n".join(dates).encode()).hexdigest(), "source_code_sha": source},
        "fundamental_provenance_manifest.json": {"synthetic_delay_certified_count": 0, "invalid_chronology_count": 0, "official_announcement_rows": 1, "source_code_sha": source},
        "quantile_evaluation_summary.json": {"ranking_method": "average", "expected_n_groups": 5, "total_dates": 3, "valid_quantile_dates": 3, "invalid_quantile_dates": 0, "invalid_tie_dates": 0, "dates_missing_required_groups": 0, "aggregation_method": "mean_of_daily_quantile_returns", "daily_group_counts": {d: {f"Q{i}": 1 for i in range(1, 6)} for d in dates}},
        "multi_seed_robustness.json": {"seed_rankic_each": {"42": .01, "100": .02, "2024": .03}, "seed_rankic_std": seed_std},
        "bootstrap_comparison.json": [{"comparison_pair": "candidate_vs_baseline", "bootstrap_ci_95_lower": .01}],
        "governance_manifest.json": {"final_holdout_available": holdout, "historical_oos_status": True, "live_trading_ready": False, "production_model_promotion": False, "source_code_sha": source},
        "runtime_source_provenance.json": {"runtime_git_sha": source, "worktree_clean_before_run": True},
    }
    for name, payload in payloads.items(): (root / name).write_text(json.dumps(payload), encoding="utf-8")
    manifest = {}
    for path in root.iterdir():
        if path.is_file():
            manifest[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size, "generated_by": "test", "source_code_sha": source, "dataset_sha256": "dataset", "calendar_sha256": "calendar", "config_hash": "config", "run_id": "run"}
    (root / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _matrix(root):
    return evaluate_research_gates({}, {}, "", "", [], {}, {}, [], {}, {}, {}, {}, {}, root)


def test_missing_evidence_dir_cannot_pass():
    matrix = evaluate_research_gates({}, {}, "", "", [], {}, {}, [], {}, {}, {}, {}, {}, None)
    assert matrix["INFRASTRUCTURE_STATUS"] == "INSUFFICIENT_EVIDENCE"
    assert not matrix["GATES"]["CANONICAL_CALENDAR_PROVENANCE"]["passed"]


def test_missing_label_valid_fails_certified_pool():
    with pytest.raises(KeyError, match="label_valid"):
        build_regression_native_train_pool(pd.DataFrame({"label_net_alpha_20d": [.1]}))


def test_native_pool_missing_in_universe_fails():
    with pytest.raises(KeyError, match="in_universe"):
        build_regression_native_train_pool(pd.DataFrame({"label_valid": [True], "label_net_alpha_20d": [.1], "excluded_from_training": [False]}))


def test_native_pool_missing_exclusion_flag_fails():
    with pytest.raises(KeyError, match="excluded_from_training"):
        build_regression_native_train_pool(pd.DataFrame({"label_valid": [True], "label_net_alpha_20d": [.1], "in_universe": [True]}))


def test_holdout_true_must_fail(tmp_path):
    _write_evidence(tmp_path, holdout=True)
    assert not _matrix(tmp_path)["GATES"]["FINAL_HOLDOUT_GOVERNANCE"]["passed"]


def test_missing_purge_field_cannot_pass(tmp_path):
    _write_evidence(tmp_path)
    (tmp_path / "walk_forward_purge_audit.json").write_text("[{}]", encoding="utf-8")
    assert not _matrix(tmp_path)["GATES"]["WALKFORWARD_PURGE_GATE"]["passed"]


def test_boolean_claims_cannot_self_certify(tmp_path):
    _write_evidence(tmp_path)
    (tmp_path / "calendar_metadata.json").write_text(json.dumps({"calendar_provenance_verified": True}), encoding="utf-8")
    assert not _matrix(tmp_path)["GATES"]["CANONICAL_CALENDAR_PROVENANCE"]["passed"]


def test_bootstrap_90_and_95_interval_percentiles():
    s = pd.Series(np.arange(40, dtype=float), index=pd.bdate_range("2023-01-01", periods=40))
    result = paired_block_bootstrap(s + 1, s, "candidate", "baseline", n_bootstraps=100)
    assert result["bootstrap_ci_90_lower"] <= result["bootstrap_ci_95_lower"]
    assert result["bootstrap_ci_95_upper"] <= result["bootstrap_ci_90_upper"]


def test_seed_failure_does_not_fail_infrastructure(tmp_path):
    _write_evidence(tmp_path, seed_std=.008)
    matrix = _matrix(tmp_path)
    assert matrix["MODEL_EVIDENCE_STATUS"] == "MIXED_EVIDENCE_NOT_ROBUST"


def test_required_gate_evidence_hash_not_empty(tmp_path):
    _write_evidence(tmp_path)
    matrix = _matrix(tmp_path)
    for gate in matrix["GATES"].values():
        assert gate["evidence_artifacts"]
        assert gate["evidence_sha256"]


def test_strict_purge_below_label_horizon_fails():
    with pytest.raises(ValueError, match="below label horizon"):
        WalkForwardTrainer(purge_gap_days=5, strict_mode=True)


def test_artifact_manifest_tamper_fails(tmp_path):
    _write_evidence(tmp_path)
    (tmp_path / "bootstrap_comparison.json").write_text("[]", encoding="utf-8")
    assert not validate_artifact_hash_chain(tmp_path, next(iter(json.loads((tmp_path / "artifact_manifest.json").read_text()).values())))[0]


def test_artifact_manifest_missing_file_fails(tmp_path):
    _write_evidence(tmp_path)
    (tmp_path / "bootstrap_comparison.json").unlink()
    assert not validate_artifact_hash_chain(tmp_path, next(iter(json.loads((tmp_path / "artifact_manifest.json").read_text()).values())))[0]


def test_quantile_boolean_claims_without_runtime_counts_cannot_pass(tmp_path):
    _write_evidence(tmp_path)
    (tmp_path / "quantile_evaluation_summary.json").write_text(json.dumps({
        "ranking_method": "average", "daily_equal_weighted": True,
        "all_equal_dates_invalid": True, "row_shuffle_invariant": True}), encoding="utf-8")
    assert not _matrix(tmp_path)["GATES"]["QUANTILE_EVALUATION_INTEGRITY"]["passed"]


def test_synthetic_test_run_cannot_scientifically_certify(tmp_path):
    _write_evidence(tmp_path)
    (tmp_path / "fold_backtest_provenance.json").write_text(json.dumps({"run_mode": "synthetic_test", "folds": []}), encoding="utf-8")
    assert _matrix(tmp_path)["INFRASTRUCTURE_STATUS"] == "INSUFFICIENT_EVIDENCE"


def test_final_run_pointer_rejects_hash_or_provenance_mismatch(tmp_path):
    run = tmp_path / "runs" / "run-1"
    run.mkdir(parents=True)
    payload = run / "payload.json"
    payload.write_text("{}", encoding="utf-8")
    entry = {"sha256": hashlib.sha256(payload.read_bytes()).hexdigest(), "size_bytes": payload.stat().st_size,
             "generated_by": "test", "source_code_sha": "a" * 40, "dataset_sha256": "dataset",
             "calendar_sha256": "calendar", "config_hash": "config", "run_id": "run-1"}
    manifest = run / "artifact_manifest.json"
    manifest.write_text(json.dumps({"payload.json": entry}), encoding="utf-8")
    matrix = run / "audit_gate_matrix.json"
    matrix.write_text("{}", encoding="utf-8")
    pointer = tmp_path / "FINAL_RUN_POINTER.json"
    pointer_data = {"run_id": "run-1", "code_freeze_sha": "a" * 40, "dataset_sha256": "dataset",
                    "calendar_sha256": "calendar", "config_hash": "config",
                    "artifact_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "gate_matrix_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
                    "created_at": "2026-08-31T00:00:00", "run_status": "FAILED"}
    pointer.write_text(json.dumps(pointer_data), encoding="utf-8")
    assert validate_final_run_pointer(pointer, "a" * 40)[0]
    pointer_data["dataset_sha256"] = "tampered"
    pointer.write_text(json.dumps(pointer_data), encoding="utf-8")
    assert not validate_final_run_pointer(pointer, "a" * 40)[0]
