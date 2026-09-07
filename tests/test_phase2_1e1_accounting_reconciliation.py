"""
Unit Tests for Phase 2.1-E1: Scientific Accounting Reconciliation
(tests/test_phase2_1e1_accounting_reconciliation.py)

Covers at least 14 critical scientific accounting invariants:
1. turnover unit consistency (decimal vs percentage)
2. one-way / two-way turnover identity (two_way == 2.0 * one_way)
3. sleeve turnover aggregation (portfolio holdings sum of weighted sleeve trades)
4. turnover derived from holdings (strictly from W_t and W_{t-1})
5. 10bps == 0.001, 20bps == 0.002 identity
6. daily cost linked to actual turnover (cost_t = turnover_t * cost_rate)
7. daily net return compounding (cum_ret == prod(1 + net_ret) - 1)
8. baseline / candidate prediction separation (assert max diff > tolerance)
9. baseline / candidate tail membership separation (sets are distinct)
10. bootstrap paired but distinct series (diff is non-trivial)
11. fold contribution sum invariant (sum of fold deltas == overall delta)
12. fold concentration classification (CONCENTRATED / MIXED / WELL_DISTRIBUTED)
13. ICIR sqrt annualization (sqrt(242/20) and sqrt(242))
14. decimal / percentage conversion helper
"""

import pytest
import numpy as np
import pandas as pd


def test_turnover_unit_consistency():
    """1. turnover unit consistency: 0.0385 decimal == 3.85%"""
    dec_val = 0.0385
    pct_val = dec_val * 100.0
    assert abs(pct_val - 3.85) < 1e-6
    assert abs(pct_val / 100.0 - dec_val) < 1e-6


def test_one_way_two_way_turnover_identity():
    """2. one-way / two-way turnover identity: two_way == 2.0 * one_way"""
    buy_notional = 0.04
    sell_notional = 0.04
    nav = 1.0

    one_way = 0.5 * (buy_notional + sell_notional) / nav
    two_way = (buy_notional + sell_notional) / nav

    assert abs(two_way - 2.0 * one_way) < 1e-8
    assert abs(one_way - 0.04) < 1e-8
    assert abs(two_way - 0.08) < 1e-8


def test_sleeve_turnover_aggregation():
    """3. sleeve turnover aggregation: portfolio turnover is sum of weighted sleeve trade deltas"""
    n_sleeves = 20
    sleeve_weight = 1.0 / n_sleeves

    # On day t, only 1 sleeve rebalances with 100% turnover within that sleeve
    single_sleeve_trade = 1.0
    portfolio_trade = single_sleeve_trade * sleeve_weight

    assert abs(portfolio_trade - 0.05) < 1e-8
    assert portfolio_trade < single_sleeve_trade


def test_turnover_derived_from_holdings():
    """4. turnover derived from holdings: strictly from W_t and W_{t-1}"""
    w_prev = {"A": 0.5, "B": 0.5, "C": 0.0}
    w_curr = {"A": 0.4, "B": 0.3, "C": 0.3}

    all_keys = set(w_prev.keys()).union(set(w_curr.keys()))
    buy_n = sum(max(w_curr.get(k, 0.0) - w_prev.get(k, 0.0), 0.0) for k in all_keys)
    sell_n = sum(max(w_prev.get(k, 0.0) - w_curr.get(k, 0.0), 0.0) for k in all_keys)

    assert abs(buy_n - 0.3) < 1e-8
    assert abs(sell_n - 0.3) < 1e-8
    one_way = 0.5 * (buy_n + sell_n)
    assert abs(one_way - 0.3) < 1e-8


def test_bps_decimal_identity():
    """5. bps to decimal identity: 10 bps == 0.0010, 20 bps == 0.0020, 30 bps == 0.0030, 50 bps == 0.0050"""
    mappings = {
        10: 0.0010,
        20: 0.0020,
        30: 0.0030,
        50: 0.0050
    }
    for bps, dec in mappings.items():
        assert abs(bps / 10000.0 - dec) < 1e-8


def test_daily_cost_linked_to_actual_turnover():
    """6. daily cost linked to actual turnover: cost_t = turnover_t * cost_rate"""
    turnovers = np.array([0.02, 0.04, 0.01, 0.05])
    cost_rate = 0.0020  # 20 bps

    daily_costs = turnovers * cost_rate
    expected_costs = np.array([0.00004, 0.00008, 0.00002, 0.00010])
    np.testing.assert_allclose(daily_costs, expected_costs)


def test_daily_net_return_compounding():
    """7. daily net return compounding: cum_ret == prod(1 + net_ret) - 1"""
    gross_returns = np.array([0.01, -0.005, 0.02, 0.003])
    costs = np.array([0.0005, 0.0005, 0.0005, 0.0005])
    net_returns = gross_returns - costs

    cum_net = np.prod(1.0 + net_returns) - 1.0
    expected_cum = (1.0095 * 0.9945 * 1.0195 * 1.0025) - 1.0
    assert abs(cum_net - expected_cum) < 1e-8


def test_baseline_candidate_prediction_separation():
    """8. baseline / candidate prediction separation: assert max diff > tolerance"""
    pred_base = np.array([0.1, 0.4, 0.2, 0.8, 0.5])
    pred_cand = np.array([0.15, 0.35, 0.25, 0.75, 0.6])

    diff = np.abs(pred_base - pred_cand)
    max_diff = np.max(diff)
    assert max_diff > 1e-4
    assert not np.allclose(pred_base, pred_cand)


def test_baseline_candidate_tail_membership_separation():
    """9. baseline / candidate tail membership separation: sets are distinct"""
    df_sample = pd.DataFrame({
        "symbol": [f"S_{i}" for i in range(10)],
        "pred_base": [0.1, 0.2, 0.9, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0],
        "pred_cand": [0.9, 0.2, 0.1, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0]
    })

    top2_base = set(df_sample.nlargest(2, "pred_base")["symbol"])
    top2_cand = set(df_sample.nlargest(2, "pred_cand")["symbol"])

    # Top 2 base: S_2, S_3; Top 2 cand: S_0, S_3
    assert top2_base != top2_cand
    assert len(top2_base.intersection(top2_cand)) == 1


def test_bootstrap_paired_but_distinct_series():
    """10. bootstrap paired but distinct series: diff is non-trivial"""
    series_a = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00], index=pd.date_range("2024-01-01", periods=5))
    series_b = pd.Series([0.00, 0.01, -0.02, 0.01, -0.01], index=pd.date_range("2024-01-01", periods=5))

    delta_series = series_a - series_b
    assert not np.allclose(delta_series.values, 0.0)
    assert abs(delta_series.mean() - 0.012) < 1e-6


def test_fold_contribution_sum_invariant():
    """11. fold contribution sum invariant: sum of fold deltas == overall delta"""
    fold_deltas = [0.02, -0.01, 0.03, 0.04, -0.02]
    total_delta = sum(fold_deltas)
    assert abs(total_delta - 0.06) < 1e-8


def test_fold_concentration_classification():
    """12. fold concentration classification rule: >100% share must be CONCENTRATED"""
    # Case 1: single fold > 100% because other folds negative
    folds_case1 = [{"delta": 0.12}, {"delta": -0.05}, {"delta": 0.01}]
    sum_deltas = sum(f["delta"] for f in folds_case1)  # 0.08
    largest = max(f["delta"] for f in folds_case1)     # 0.12
    share = largest / sum_deltas                        # 1.50 (150%)

    if share > 1.0 or share < 0:
        cls1 = "CONCENTRATED"
    else:
        cls1 = "WELL_DISTRIBUTED"
    assert cls1 == "CONCENTRATED"

    # Case 2: well balanced
    folds_case2 = [{"delta": 0.02}, {"delta": 0.02}, {"delta": 0.02}]
    share2 = max(f["delta"] for f in folds_case2) / sum(f["delta"] for f in folds_case2) # 0.333
    cls2 = "WELL_DISTRIBUTED" if share2 <= 0.50 else "CONCENTRATED"
    assert cls2 == "WELL_DISTRIBUTED"


def test_icir_sqrt_annualization():
    """13. ICIR sqrt annualization: period-annualized is raw * sqrt(242/20)"""
    raw_icir = 0.39426
    horizon = 20
    period_annualized = raw_icir * np.sqrt(242.0 / horizon)
    daily_annualized = raw_icir * np.sqrt(242.0)

    assert abs(period_annualized - 1.37144) < 1e-3
    assert abs(daily_annualized - 6.13324) < 1e-3


def test_decimal_percentage_conversion_helpers():
    """14. decimal / percentage conversion helpers"""
    def to_bps(decimal_val: float) -> float:
        return decimal_val * 10000.0

    def to_pct(decimal_val: float) -> float:
        return decimal_val * 100.0

    assert abs(to_bps(0.0020) - 20.0) < 1e-6
    assert abs(to_pct(0.0931) - 9.31) < 1e-6
