import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_v2.governance.certification import evaluate_research_gates
from research_v2.governance.holdout_registry import build_regression_native_train_pool
from tools.run_model_research import paired_block_bootstrap


def _write_evidence(root: Path, *, seed_std=0.008, holdout=False):
    source = "a" * 40
    dates = ["2023-01-02", "2023-01-03", "2023-01-04"]
    fold = {k: 1 for k in ("test_dates_count", "candidate_oos_rows", "baseline_oos_rows", "candidate_trade_count", "baseline_trade_count")}
    fold.update({"fold_id": 1, "test_start": dates[0], "test_end": dates[-1], "candidate_prediction_sha256": "a", "baseline_prediction_sha256": "b", "candidate_backtest_run_id": "c", "baseline_backtest_run_id": "d", "candidate_equity_sha256": "e", "baseline_equity_sha256": "f", "candidate_orders_sha256": "g", "baseline_orders_sha256": "h", "candidate_strategy_return": 0.1, "baseline_strategy_return": 0.0, "candidate_benchmark_return": 0.0, "baseline_benchmark_return": 0.0, "candidate_excess_return": 0.1, "baseline_excess_return": 0.0, "candidate_minus_baseline": 0.1, "engine_config_hash": "i", "dataset_sha256": "j", "source_code_sha": source})
    payloads = {
        "fold_backtest_provenance.json": {"run_mode": "certified", "folds": [fold]},
        "production_isolation.json": {"before": {}, "after": {}, "file_sha_before": "", "file_sha_after": ""},
        "walk_forward_purge_audit.json": [{"purge_gap_days": 2, "actual_train_val_gap_trading_days": 2, "actual_val_test_gap_trading_days": 2}],
        "calendar_metadata.json": {"calendar_source": "exchange", "dates": dates, "dataset_overlap_count": 3, "calendar_sha256": hashlib.sha256("\n".join(dates).encode()).hexdigest(), "source_code_sha": source},
        "fundamental_provenance_manifest.json": {"synthetic_delay_certified_count": 0, "invalid_chronology_count": 0, "source_code_sha": source},
        "quantile_evaluation_summary.json": {"ranking_method": "average", "daily_equal_weighted": True, "all_equal_dates_invalid": True, "row_shuffle_invariant": True},
        "multi_seed_robustness.json": {"seed_rankic_each": {"42": .01, "100": .02, "2024": .03}, "seed_rankic_std": seed_std},
        "bootstrap_comparison.json": [{"comparison_pair": "candidate_vs_baseline", "bootstrap_ci_95_lower": .01}],
        "holdout_manifest.json": {"FINAL_HOLDOUT_AVAILABLE": holdout, "historical_oos": True},
    }
    for name, payload in payloads.items(): (root / name).write_text(json.dumps(payload), encoding="utf-8")


def _matrix(root):
    return evaluate_research_gates({}, {}, "", "", [], {}, {}, [], {}, {}, {}, {}, {}, root)


def test_missing_evidence_dir_cannot_pass():
    matrix = evaluate_research_gates({}, {}, "", "", [], {}, {}, [], {}, {}, {}, {}, {}, None)
    assert matrix["INFRASTRUCTURE_STATUS"] == "INSUFFICIENT_EVIDENCE"
    assert not matrix["GATES"]["CANONICAL_CALENDAR_PROVENANCE"]["passed"]


def test_missing_label_valid_fails_certified_pool():
    with pytest.raises(KeyError, match="label_valid"):
        build_regression_native_train_pool(pd.DataFrame({"label_net_alpha_20d": [.1]}))


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
    assert matrix["INFRASTRUCTURE_STATUS"] == "VERIFIED"
    assert matrix["MODEL_EVIDENCE_STATUS"] == "MIXED_EVIDENCE_NOT_ROBUST"


def test_required_gate_evidence_hash_not_empty(tmp_path):
    _write_evidence(tmp_path)
    matrix = _matrix(tmp_path)
    for gate in matrix["GATES"].values():
        assert gate["evidence_artifacts"]
        assert gate["evidence_sha256"]
