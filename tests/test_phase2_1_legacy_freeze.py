"""
Tests for Legacy Baseline V1 Freeze (tests/test_phase2_1_legacy_freeze.py)
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np


def test_legacy_v1_baseline_artifacts_exist():
    base_dir = Path("reports/baselines/legacy_v1")
    assert base_dir.exists()
    required_files = [
        "baseline_manifest.json",
        "legacy_model_semantics.json",
        "model_comparison.csv",
        "trading_fold_stability.csv",
        "seed_robustness.csv",
        "artifact_hashes.json",
        "LEGACY_BASELINE_REPORT.md"
    ]
    for rf in required_files:
        p = base_dir / rf
        assert p.exists(), f"Missing legacy baseline artifact: {p}"


def test_legacy_baseline_id_and_status():
    manifest_file = Path("reports/baselines/legacy_v1/baseline_manifest.json")
    assert manifest_file.exists()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["baseline_id"] == "LEGACY_BASELINE_V1"
    assert data["baseline_status"] == "FROZEN"
    assert data["model_evidence_source_commit"] == "e6da4a2320ad4cbd5ef9cf8b9f772baf89602a48"
    assert len(data["dataset_sha256"]) == 64
    assert len(data["feature_schema_hash"]) == 64
    assert data["feature_count"] == 79
    assert data["label_horizon"] == 20
    assert data["prediction_champion_seed_robustness"] == "PASS"
    assert data["trading_candidate_seed_robustness"] == "NOT_CERTIFIED"
    assert data["live_trading_ready"] is False


def test_legacy_metrics_exact_match():
    manifest_file = Path("reports/baselines/legacy_v1/baseline_manifest.json")
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    
    # 预测冠军
    p = data["prediction_baseline"]
    assert p["model_id"] == "lightgbm_clf_baseline"
    assert p["mean_daily_rank_ic"] == 0.0503
    assert p["nw20_rank_icir"] == 0.4044
    assert p["auc"] == 0.5319
    assert p["q5_minus_q1_spread"] == 7.17
    assert p["common_ranking_rows"] == 221019
    assert p["common_oos_dates"] == 744

    # 交易候选
    t = data["trading_candidate"]
    assert t["historical_artifact_model_id"] == "lightgbm_ranker"
    assert t["legacy_model_id"] == "legacy_ordinal_ranker"
    assert t["cost_adjusted_excess_return"] == 5.72
    assert t["sharpe_ratio"] == 0.36
    assert t["max_drawdown"] == -14.35
    assert t["real_fold_win_ratio"] == 0.55


def test_legacy_ranker_semantics_corrected():
    sem_file = Path("reports/baselines/legacy_v1/legacy_model_semantics.json")
    assert sem_file.exists()
    data = json.loads(sem_file.read_text(encoding="utf-8"))
    ranker_sem = data["models"]["trading_candidate"]

    assert ranker_sem["historical_artifact_model_id"] == "lightgbm_ranker"
    assert ranker_sem["legacy_model_id"] == "legacy_ordinal_ranker"
    assert ranker_sem["effective_estimator_class"] == "LGBMRanker"
    assert ranker_sem["effective_objective"] == "regression"
    assert ranker_sem["effective_metric"] == "rmse"
    assert ranker_sem["ranking_group_supplied"] is True
    assert ranker_sem["relevance_label"] == "daily_ordinal_0_to_4"
    assert ranker_sem["true_lambdarank_certified"] is False
    assert data["true_lambdarank_certified"] is False


def test_legacy_label_timing_documented():
    manifest_file = Path("reports/baselines/legacy_v1/baseline_manifest.json")
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    lbl_sem = data["legacy_label_semantics"]
    assert lbl_sem["signal_time"] == "T_CLOSE"
    assert lbl_sem["entry_aligned"] is False
    assert lbl_sem["legacy_return_window"] == "T_CLOSE_TO_T_PLUS_20_CLOSE"
    assert lbl_sem["execution_engine_entry"] == "T_PLUS_1_OPEN"
