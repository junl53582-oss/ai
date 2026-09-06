import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.evaluator import ModelEvaluator
from research_v2.governance.certification import evaluate_research_gates


def _make_sample_predictions_df(n_days=10, stocks_per_day=50, random_seed=42):
    rng = np.random.default_rng(random_seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    records = []
    for d in dates:
        d_str = str(d.date())
        scores = rng.uniform(0.0, 1.0, stocks_per_day)
        labels = 0.05 * scores + rng.normal(0.0, 0.02, stocks_per_day)
        for i in range(stocks_per_day):
            records.append({
                "date": d_str,
                "symbol": f"{i:06d}.SZ",
                "pred_score": float(scores[i]),
                "label_excess_20d": float(labels[i]),
                "in_universe": True,
            })
    return pd.DataFrame(records)


def test_evaluator_metrics_contain_total_dates():
    df = _make_sample_predictions_df(n_days=5, stocks_per_day=30)
    ev = ModelEvaluator()
    metrics = ev.evaluate_predictions(df, label_col="label_excess_20d", task_type="regression")

    assert "total_dates" in metrics
    assert metrics["total_dates"] == 5
    assert metrics["valid_quantile_dates"] == 5
    assert metrics["invalid_quantile_dates"] == 0
    assert metrics["invalid_tie_dates"] == 0
    assert metrics["dates_missing_required_groups"] == 0
    assert metrics["ranking_method"] == "average"
    assert metrics["quantile_aggregation_method"] == "mean_of_daily_quantile_returns"
    assert metrics["total_dates"] == metrics["valid_quantile_dates"] + metrics["invalid_quantile_dates"]


def test_quantile_invariant_with_invalid_and_tie_dates():
    df = _make_sample_predictions_df(n_days=6, stocks_per_day=30)
    dates = sorted(df["date"].unique())

    # Date 1: Insufficient observations (< 5)
    df = df[~((df["date"] == dates[0]) & (df["symbol"] > "000002.SZ"))]

    # Date 2: Identical scores (tie collapse: unique scores = 1 < 5)
    mask_date2 = df["date"] == dates[1]
    df.loc[mask_date2, "pred_score"] = 0.5

    ev = ModelEvaluator()
    metrics = ev.evaluate_predictions(df, label_col="label_excess_20d", task_type="regression")

    assert metrics["total_dates"] == 6
    assert metrics["valid_quantile_dates"] == 4
    assert metrics["invalid_quantile_dates"] == 2
    assert metrics["invalid_tie_dates"] == 1
    assert metrics["dates_missing_required_groups"] == 2
    assert metrics["total_dates"] == metrics["valid_quantile_dates"] + metrics["invalid_quantile_dates"]


def test_quantile_row_shuffle_invariance():
    df = _make_sample_predictions_df(n_days=8, stocks_per_day=40, random_seed=123)
    ev = ModelEvaluator()
    metrics_original = ev.evaluate_predictions(df, label_col="label_excess_20d", task_type="regression")

    # Permute rows completely
    df_shuffled = df.sample(frac=1.0, random_state=999).reset_index(drop=True)
    metrics_shuffled = ev.evaluate_predictions(df_shuffled, label_col="label_excess_20d", task_type="regression")

    assert metrics_original["total_dates"] == metrics_shuffled["total_dates"]
    assert metrics_original["valid_quantile_dates"] == metrics_shuffled["valid_quantile_dates"]
    assert np.isclose(metrics_original["Q5_minus_Q1"], metrics_shuffled["Q5_minus_Q1"])
    assert np.isclose(metrics_original["mean_daily_q5_minus_q1"], metrics_shuffled["mean_daily_q5_minus_q1"])
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        assert np.isclose(
            metrics_original["quantile_returns"][q],
            metrics_shuffled["quantile_returns"][q]
        )


def test_quantile_cross_sectional_independence():
    df_base = _make_sample_predictions_df(n_days=3, stocks_per_day=30, random_seed=111)
    ev = ModelEvaluator()
    info_base = ev._compute_quantile_returns(df_base, label_col="label_excess_20d")

    df_extra = _make_sample_predictions_df(n_days=1, stocks_per_day=30, random_seed=222)
    df_extra["date"] = "2023-01-06"
    df_extra["pred_score"] = df_extra["pred_score"] * 1000.0

    df_combined = pd.concat([df_base, df_extra], ignore_index=True)
    info_combined = ev._compute_quantile_returns(df_combined, label_col="label_excess_20d")

    for d in df_base["date"].unique():
        assert info_base["daily_group_counts"][d] == info_combined["daily_group_counts"][d]


def test_quantile_no_future_label_leakage():
    df = _make_sample_predictions_df(n_days=2, stocks_per_day=25, random_seed=777)
    ev = ModelEvaluator()
    info1 = ev._compute_quantile_returns(df, label_col="label_excess_20d")

    df_mod_label = df.copy()
    df_mod_label["label_excess_20d"] = -df_mod_label["label_excess_20d"] * 10.0
    info2 = ev._compute_quantile_returns(df_mod_label, label_col="label_excess_20d")

    assert info1["daily_group_counts"] == info2["daily_group_counts"]
    assert info1["total_dates"] == info2["total_dates"]
    assert info1["valid_quantile_dates"] == info2["valid_quantile_dates"]


def test_governance_certification_quantile_gate_pass_and_fail_closed(tmp_path):
    dates = ["2023-01-02", "2023-01-03", "2023-01-04"]
    daily_counts = {d: {f"Q{i}": 10 for i in range(1, 6)} for d in dates}

    valid_payload = {
        "ranking_method": "average",
        "expected_n_groups": 5,
        "total_dates": 3,
        "valid_quantile_dates": 3,
        "invalid_quantile_dates": 0,
        "invalid_tie_dates": 0,
        "dates_missing_required_groups": 0,
        "aggregation_method": "mean_of_daily_quantile_returns",
        "daily_group_counts": daily_counts,
        "runtime_model_id": "baseline",
        "source_code_sha": "a" * 40,
    }

    evidence_file = tmp_path / "quantile_evaluation_summary.json"
    evidence_file.write_text(json.dumps(valid_payload), encoding="utf-8")

    from tests.test_r3_2_evidence_integrity import _write_evidence
    _write_evidence(tmp_path)
    evidence_file.write_text(json.dumps(valid_payload), encoding="utf-8")

    manifest_path = tmp_path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["quantile_evaluation_summary.json"]["sha256"] = hashlib.sha256(evidence_file.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    matrix = evaluate_research_gates({}, {}, "", "", [], {}, {}, [], {}, {}, {}, {}, {}, tmp_path)
    gate = matrix["GATES"]["QUANTILE_EVALUATION_INTEGRITY"]
    assert gate["passed"] is True, f"Gate failed unexpectedly: {gate['message']}"

    # Fail-Closed 1: total_dates is None
    payload_none_dates = copy.deepcopy(valid_payload)
    payload_none_dates["total_dates"] = None
    evidence_file.write_text(json.dumps(payload_none_dates), encoding="utf-8")
    manifest["quantile_evaluation_summary.json"]["sha256"] = hashlib.sha256(evidence_file.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    matrix = evaluate_research_gates({}, {}, "", "", [], {}, {}, [], {}, {}, {}, {}, {}, tmp_path)
    assert matrix["GATES"]["QUANTILE_EVALUATION_INTEGRITY"]["passed"] is False

    # Fail-Closed 2: Invariant mismatch: total_dates != valid + invalid
    payload_mismatch = copy.deepcopy(valid_payload)
    payload_mismatch["total_dates"] = 10
    evidence_file.write_text(json.dumps(payload_mismatch), encoding="utf-8")
    manifest["quantile_evaluation_summary.json"]["sha256"] = hashlib.sha256(evidence_file.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    matrix = evaluate_research_gates({}, {}, "", "", [], {}, {}, [], {}, {}, {}, {}, {}, tmp_path)
    assert matrix["GATES"]["QUANTILE_EVALUATION_INTEGRITY"]["passed"] is False

    # Fail-Closed 3: ranking_method != average
    payload_first = copy.deepcopy(valid_payload)
    payload_first["ranking_method"] = "first"
    evidence_file.write_text(json.dumps(payload_first), encoding="utf-8")
    manifest["quantile_evaluation_summary.json"]["sha256"] = hashlib.sha256(evidence_file.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    matrix = evaluate_research_gates({}, {}, "", "", [], {}, {}, [], {}, {}, {}, {}, {}, tmp_path)
    assert matrix["GATES"]["QUANTILE_EVALUATION_INTEGRITY"]["passed"] is False
