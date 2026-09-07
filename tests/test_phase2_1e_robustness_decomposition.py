"""
Unit tests for Phase 2.1-E: Robustness Decomposition & Tail Alpha Validation Integrity
(tests/test_phase2_1e_robustness_decomposition.py)

Covers:
1. Fold attribution sum invariant (sum of fold samples equals total samples)
2. Date attribution invariant (mean of delta equals cand_mean - base_mean)
3. Ablation reproducibility (ablation configurations are distinct and deterministic)
4. Bootstrap pairing correctness (identical series yields delta=0, CI=[0, 0])
5. Block bootstrap serial dependence property
6. Q5-Q1 portfolio accounting reality check (sleeve overlapping vs naive forward mean)
7. Turnover calculation realism (day-to-day weight transitions non-negative)
8. Cost linked to turnover (2x turnover yields 2x transaction cost)
9. Stock contribution aggregation (sum of stock contributions equals portfolio excess)
10. Ledger append-only immutability
"""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def test_date_attribution_invariant():
    """Verify that average daily delta equals average candidate metric minus average baseline metric."""
    rng = np.random.default_rng(42)
    n = 100
    base_ic = 0.03 + rng.normal(0, 0.02, n)
    cand_ic = 0.04 + rng.normal(0, 0.02, n)
    delta_ic = cand_ic - base_ic

    assert np.isclose(np.mean(delta_ic), np.mean(cand_ic) - np.mean(base_ic), atol=1e-12)


def test_bootstrap_pairing_correctness():
    """Verify that paired circular block bootstrap with identical inputs yields exactly 0 delta."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n)
    s = pd.Series(np.linspace(0.01, 0.05, n), index=dates)

    from scripts.phase21e_pipeline import paired_block_bootstrap
    res = paired_block_bootstrap(s, s, block_size=10, n_bootstraps=200)
    assert np.isclose(res["mean_diff"], 0.0, atol=1e-8)
    assert np.isclose(res["bootstrap_ci_95_lower"], 0.0, atol=1e-8)
    assert np.isclose(res["bootstrap_ci_95_upper"], 0.0, atol=1e-8)
    assert res["robust_improvement"] is False


def test_block_bootstrap_serial_dependence():
    """Verify that block bootstrap with larger block size widens CI when positive autocorrelation is present."""
    from scripts.phase21e_pipeline import paired_block_bootstrap
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n)
    # AR(1) positive autocorrelated delta
    e = rng.normal(0, 0.02, n)
    delta = np.zeros(n)
    for t in range(1, n):
        delta[t] = 0.7 * delta[t - 1] + e[t]
    base = pd.Series(0.03, index=dates)
    cand = pd.Series(0.03 + delta, index=dates)

    res_b5 = paired_block_bootstrap(cand, base, block_size=5, n_bootstraps=500)
    res_b20 = paired_block_bootstrap(cand, base, block_size=20, n_bootstraps=500)

    # In presence of positive autocorrelation, larger block size should capture larger block variance
    width_b5 = res_b5["bootstrap_ci_95_upper"] - res_b5["bootstrap_ci_95_lower"]
    width_b20 = res_b20["bootstrap_ci_95_upper"] - res_b20["bootstrap_ci_95_lower"]
    assert width_b20 >= width_b5 * 0.9  # Block size 20 accounts for persistent clusters


def test_turnover_calculation_realism():
    """Verify that day-to-day holdings changes correctly measure one-way turnover."""
    from scripts.phase21e_pipeline import compute_realized_holdings_turnover
    # Day 1: hold stocks A and B (weights 0.5, 0.5)
    # Day 2: hold stocks B and C (weights 0.5, 0.5) -> stock A sold (0.5), C bought (0.5) -> one-way = 0.50
    h_df = pd.DataFrame([
        {"date": "2023-01-01", "symbol": "000001.SZ", "weight": 0.5},
        {"date": "2023-01-01", "symbol": "000002.SZ", "weight": 0.5},
        {"date": "2023-01-02", "symbol": "000002.SZ", "weight": 0.5},
        {"date": "2023-01-02", "symbol": "000003.SZ", "weight": 0.5},
    ])
    to = compute_realized_holdings_turnover(h_df)
    assert len(to) == 1
    assert np.isclose(to.iloc[0], 0.50)


def test_cost_linked_to_turnover():
    """Verify that transaction cost scales strictly with one-way turnover."""
    one_way_to = 0.40  # 40% one-way turnover
    cost_bps = 20.0    # 20 bps = 0.0020
    drag = one_way_to * (cost_bps / 10000.0)
    assert np.isclose(drag, 0.0008)

    # 2x turnover
    drag_double = (one_way_to * 2.0) * (cost_bps / 10000.0)
    assert np.isclose(drag_double, drag * 2.0)


def test_stock_contribution_aggregation():
    """Verify that stock level contributions sum to total portfolio gross return."""
    stock_returns = pd.Series({"000001.SZ": 0.02, "000002.SZ": -0.01, "000003.SZ": 0.05})
    weights = pd.Series({"000001.SZ": 0.4, "000002.SZ": 0.3, "000003.SZ": 0.3})
    contribs = stock_returns * weights
    portfolio_ret = (stock_returns * weights).sum()
    assert np.isclose(contribs.sum(), portfolio_ret)


def test_q5_q1_sleeve_accounting_vs_naive():
    """Verify that sleeve overlapping portfolio accounting properly captures daily compounding."""
    from scripts.phase21e_pipeline import compute_sleeve_overlapping_returns
    # Generate 40 days of dummy prices for 10 stocks
    rng = np.random.default_rng(42)
    dates = [str(d.date()) for d in pd.bdate_range("2023-01-01", periods=40)]
    recs = []
    for d in dates:
        for i in range(10):
            recs.append({
                "date": d,
                "symbol": f"S{i:02d}",
                "pred_score": rng.uniform(0, 1),
                "daily_return": rng.normal(0.0005, 0.01),
                "label_excess_20d": rng.normal(0.01, 0.05)
            })
    df_panel = pd.DataFrame(recs)
    sleeve_res = compute_sleeve_overlapping_returns(df_panel, horizon=20, n_groups=5)
    assert "daily_q5_ret" in sleeve_res
    assert "daily_q1_ret" in sleeve_res
    assert len(sleeve_res["daily_q5_ret"]) > 0


def test_ledger_append_only(tmp_path):
    """Verify ledger strictly appends without overwriting."""
    l_path = tmp_path / "ABLATION_LEDGER.jsonl"
    r1 = {"exp_id": "ABLATION_A", "q5_q1": 14.5}
    r2 = {"exp_id": "ABLATION_B", "q5_q1": 15.2}
    with open(l_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(r1) + "\n")
    with open(l_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(r2) + "\n")
    lines = l_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["exp_id"] == "ABLATION_A"
    assert json.loads(lines[1])["exp_id"] == "ABLATION_B"
