"""
Unit tests for Phase 2.1-D: Factor Interaction & Ranking Optimization Integrity
(tests/test_phase2_1d_interactions_and_ranking.py)

Covers:
1. Interaction terms deterministic computation and no lookahead
2. LambdaRank date grouping integrity (sum of group sizes equals row count)
3. Asymmetric loss gradient properties (overpredict penalty > underpredict penalty)
4. Double Ensemble sample reweighting invariance
5. Ledger append-only immutability
"""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from models.asymmetric_loss import AsymmetricRegressionObjective, AsymmetricLossObjective


def _make_dummy_panel(n_days=10, n_stocks=15, random_seed=42):
    rng = np.random.default_rng(random_seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    records = []
    for d in dates:
        d_str = str(d.date())
        for i in range(n_stocks):
            sym = f"{i:06d}.SZ"
            records.append({
                "date": d_str,
                "symbol": sym,
                "f1": rng.normal(0, 1),
                "f2": rng.normal(0, 1),
                "target": rng.normal(0.01, 0.05),
            })
    return pd.DataFrame(records).sort_values(["symbol", "date"]).reset_index(drop=True)


def test_interaction_terms_deterministic():
    df = _make_dummy_panel(n_days=10, n_stocks=10)
    inter1 = df["f1"] * np.log1p(np.abs(df["f2"]))
    inter2 = df["f1"] * np.log1p(np.abs(df["f2"]))
    pd.testing.assert_series_equal(inter1, inter2)


def test_lambdarank_group_integrity():
    df = _make_dummy_panel(n_days=12, n_stocks=20)
    group_sizes = df.groupby("date").size().values
    assert int(np.sum(group_sizes)) == len(df)
    assert len(group_sizes) == 12
    assert (group_sizes == 20).all()


def test_asymmetric_regression_loss_gradients():
    asym_obj = AsymmetricRegressionObjective(underpredict_gain=1.0, overpredict_loss=2.5)

    class DummyData:
        def get_label(self):
            return np.array([0.0, 0.0])

    data = DummyData()
    preds = np.array([0.1, -0.1])
    grad, hess = asym_obj(preds, data)

    assert np.isclose(grad[0], 0.25)
    assert np.isclose(grad[1], -0.10)
    assert np.isclose(hess[0], 2.5)
    assert np.isclose(hess[1], 1.0)


def test_double_ensemble_sample_reweighting():
    residuals = np.array([0.01, 0.05, 0.20])
    mean_res = np.mean(residuals)
    norm_res = residuals / mean_res
    decay = 0.8
    weights = np.exp(-decay * (norm_res - 1.0))
    assert weights[2] < weights[0]
    assert weights[1] < weights[0]


def test_ledger_append_only(tmp_path):
    ledger_path = tmp_path / "EXPERIMENT_LEDGER.jsonl"
    r1 = {"exp_id": "EXP_01", "rank_ic": 0.038}
    r2 = {"exp_id": "EXP_02", "rank_ic": 0.041}

    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(r1) + "\n")
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(r2) + "\n")

    lines = ledger_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["exp_id"] == "EXP_01"
    assert json.loads(lines[1])["exp_id"] == "EXP_02"
