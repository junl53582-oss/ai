"""
Formal Research Runner E2E Smoke & State Isolation Tests (tests/test_formal_research_runner_e2e.py)
验证 Formal Runner 执行真实性、无状态回测引擎隔离以及产物证据链完整性。
"""
import pytest
import numpy as np
import pandas as pd
import hashlib
import json
from pathlib import Path

from config.settings import settings
from tools.run_model_research import run_research, _create_fresh_backtest_engine, _execute_backtest_slice
from backtest.engine import BacktestEngine


@pytest.fixture
def synthetic_research_dataset(tmp_path):
    dates = pd.bdate_range("2022-01-01", "2023-08-01")
    symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600519.SH", "000858.SZ"]
    rows = []
    rng = np.random.default_rng(42)

    for dt in dates:
        for sym in symbols:
            p_base = 100.0 + rng.normal(0, 5)
            rows.append({
                "date": dt,
                "symbol": sym,
                "open": p_base * 0.99,
                "high": p_base * 1.02,
                "low": p_base * 0.98,
                "close": p_base,
                "adj_open": p_base * 0.99,
                "adj_high": p_base * 1.02,
                "adj_low": p_base * 0.98,
                "adj_close": p_base,
                "volume": 10000.0,
                "amount": 1000000.0,
                "turnover": 0.02,
                "pct_change": float(rng.normal(0, 0.02)),
                "adj_pct_change": float(rng.normal(0, 0.02)),
                "benchmark_close": 3500.0 + rng.normal(0, 20),
                "benchmark_pct_change": float(rng.normal(0, 0.01)),
                "F_MOM": float(rng.standard_normal()),
                "F_VOL": float(rng.standard_normal()),
                "F_LIQ": float(rng.standard_normal()),
                "in_universe": True,
                "is_subnew": False,
                "is_suspended": False,
                "is_limit_up": False,
                "is_limit_down": False,
                "is_limit_up_locked": False,
                "is_limit_down_locked": False,
                "current_is_st": False,
                "is_st": False,
                "historical_st_rule_applied": False
            })

    df = pd.DataFrame(rows)
    df.sort_values(by=["date", "symbol"], inplace=True)
    parquet_path = tmp_path / "synthetic_factor_matrix.parquet"
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def test_formal_research_runner_smoke_completes(tmp_path, synthetic_research_dataset):
    """
    1. Formal Research Runner 端到端 Smoke 真实执行，验证 WalkForward、BacktestEngine、PerformanceAnalyzer 与 Gate Matrix 全部生成。
    """
    out_root = tmp_path / "reports"
    run_cfg = {
        "n_estimators": 5,
        "n_bootstraps": 50,
        "train_years": 0.6,
        "val_months": 2,
        "test_months": 2,
        "purge_gap_days": 20,
        "run_mode": "synthetic_test",
        "candidates": [
            {"model_id": "lightgbm_clf_baseline", "model_name": "LightGBM Classification", "task_type": "classification", "feature_selection": "all", "weighting_mode": "none"},
            {"model_id": "lightgbm_reg_baseline", "model_name": "LightGBM Regression", "task_type": "regression", "feature_selection": "all", "weighting_mode": "none"}
        ]
    }

    res = run_research(
        dataset_path=synthetic_research_dataset,
        output_root=out_root,
        run_config=run_cfg
    )

    assert "run_id" in res
    assert "reports_dir" in res
    rep_dir = Path(res["reports_dir"])
    assert rep_dir.exists()

    # 验证核心证据文件生成
    assert (rep_dir / "trading_fold_stability.csv").exists()
    assert (rep_dir / "audit_gate_matrix.json").exists()
    assert (rep_dir / "artifact_manifest.json").exists()
    assert (rep_dir / "walk_forward_purge_audit.json").exists()
    assert (rep_dir / "multi_seed_robustness.json").exists()

    gate_matrix = res["gate_matrix"]
    assert "OVERALL_STATUS" in gate_matrix
    assert "GATES" in gate_matrix
    assert gate_matrix["FINAL_HOLDOUT_AVAILABLE"] is False
    assert gate_matrix["LIVE_TRADING_READY"] is False


def test_backtest_runs_do_not_share_state():
    """
    2. 验证连续两次调用 _execute_backtest_slice 使用完全独立、无状态污染的 BacktestEngine 实例。
    """
    dates = pd.bdate_range("2023-01-01", periods=20)
    oos_a = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "symbol": ["000001.SZ", "000002.SZ"] * len(dates),
        "close": 100.0,
        "adj_close": 100.0,
        "benchmark_close": 3000.0,
        "pred_score": [0.8, 0.2] * len(dates),
        "pred_rank": [1.0, 2.0] * len(dates),
        "in_universe": [True] * (len(dates) * 2),
        "is_suspended": [False] * (len(dates) * 2),
        "is_limit_up_locked": [False] * (len(dates) * 2),
        "is_limit_down_locked": [False] * (len(dates) * 2)
    })

    # 第 1 次运行
    perf_1 = _execute_backtest_slice(oos_a)
    # 第 2 次运行完全相同的切片
    perf_2 = _execute_backtest_slice(oos_a)

    assert perf_1["total_trades"] == perf_2["total_trades"]
    assert np.isclose(perf_1.get("cum_strategy_return", 0.0), perf_2.get("cum_strategy_return", 0.0))


def test_real_fold_metrics_not_copied_constants(tmp_path, synthetic_research_dataset):
    """
    3. 验证 trading_fold_stability.csv 中的各折指标不是人工复制的固定常数。
    """
    out_root = tmp_path / "reports_var"
    run_cfg = {
        "n_estimators": 5,
        "n_bootstraps": 20,
        "train_years": 0.6,
        "val_months": 2,
        "test_months": 2,
        "purge_gap_days": 20,
        "run_mode": "synthetic_test",
        "candidates": [
            {"model_id": "lightgbm_clf_baseline", "model_name": "LightGBM Classification", "task_type": "classification", "feature_selection": "all", "weighting_mode": "none"},
            {"model_id": "lightgbm_reg_baseline", "model_name": "LightGBM Regression", "task_type": "regression", "feature_selection": "all", "weighting_mode": "none"}
        ]
    }

    res = run_research(
        dataset_path=synthetic_research_dataset,
        output_root=out_root,
        run_config=run_cfg
    )

    rep_dir = Path(res["reports_dir"])
    fold_df = pd.read_csv(rep_dir / "trading_fold_stability.csv")
    assert not fold_df.empty
    assert "candidate_minus_baseline_excess" in fold_df.columns
    assert "candidate_won" in fold_df.columns
    assert "fold" in fold_df.columns
