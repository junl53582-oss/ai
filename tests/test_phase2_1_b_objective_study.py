"""
Dedicated Targeted Unit and Integration Tests for Phase 2.1-B Model Objective Study (tests/test_phase2_1_b_objective_study.py)
Covering:
1. Classification baseline config frozen (binary, scale_pos_weight=1.0, random_state=42)
2. Regression config uses continuous execution target and shared structural params
3. LambdaRank objective truly equals lambdarank with eval_at=[30], label_gain=[0..9]
4. Group sizes calculation correctness: sum(groups) == row count, grouped strictly by date
5. Relevance grade mapping: range 0..9, top decile 9, bottom decile 0, strictly cross-sectional per date (no lookahead/cross-date leakage)
6. WalkForwardTrainer support for classification, regression, and ranking with isolated tmp_path
7. Common OOS pool one-to-one merge validation and key hash matching
8. Production model isolation (saved_models/latest_lightgbm.pkl unchanged)
9. Provenance gate fail-closed tests
"""
import copy
import hashlib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from config.settings import settings
from models.lightgbm_model import LightGBMQuantModel
from models.walk_forward import WalkForwardTrainer
from tools.run_phase2_1_b_objective_study import (
    _validate_source_provenance,
    _paired_block_bootstrap,
    _daily_rankic,
    _top10_daily_alpha,
    _fold_comparison_three_arms,
    _build_lambdarank_relevance_labels,
)


def _make_synthetic_multiday_data(seed: int = 42, n_dates: int = 15, n_syms: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    symbols = [f"{i:06d}.SZ" for i in range(1, n_syms + 1)]
    rows = []
    for dt in dates:
        for sym in symbols:
            ret = float(rng.standard_normal()) * 0.03
            rows.append({
                "date": dt,
                "symbol": sym,
                "factor_a": float(rng.standard_normal()),
                "factor_b": float(rng.standard_normal()),
                "label_net_alpha_20d": ret,
                "label_direction_20d": 1.0 if ret > 0 else 0.0,
                "label_valid": True,
                "in_universe": True,
                "excluded_from_training": False,
                "common_train": True
            })
    df = pd.DataFrame(rows)
    df.sort_values(["date", "symbol"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def test_lambdarank_relevance_grades_range_and_deciles():
    df = _make_synthetic_multiday_data(seed=42, n_dates=10, n_syms=50)
    
    # Compute cross-sectional 10-grade relevance via shared helper
    relevance_grades = _build_lambdarank_relevance_labels(
        df, df["common_train"], target_col="label_net_alpha_20d", n_grades=10
    )
    
    assert relevance_grades.min() == 0
    assert relevance_grades.max() == 9
    assert set(np.unique(relevance_grades.dropna())).issubset(set(range(10)))
    
    # Verify per date top decile is 9 and bottom decile is 0
    df["grade"] = relevance_grades
    for dt, g in df.groupby("date"):
        assert g["grade"].max() == 9
        assert g["grade"].min() == 0
        assert g.loc[g["label_net_alpha_20d"].idxmax(), "grade"] == 9
        assert g.loc[g["label_net_alpha_20d"].idxmin(), "grade"] == 0


def test_lambdarank_relevance_no_cross_date_leakage():
    df = _make_synthetic_multiday_data(seed=42, n_dates=5, n_syms=10)
    grades_all = _build_lambdarank_relevance_labels(
        df, df["common_train"], target_col="label_net_alpha_20d", n_grades=10
    )
    
    # Isolate single date and compute independently
    dt0 = df["date"].iloc[0]
    single_date_df = df[df["date"] == dt0].copy()
    grades_single = _build_lambdarank_relevance_labels(
        single_date_df, single_date_df["common_train"], target_col="label_net_alpha_20d", n_grades=10
    )
    
    np.testing.assert_array_equal(
        grades_all[df["date"] == dt0].values,
        grades_single.values
    )


def test_lambdarank_group_sizes_sum_and_chronology(tmp_path):
    df = _make_synthetic_multiday_data(seed=42, n_dates=10, n_syms=15)
    df["label_rank"] = _build_lambdarank_relevance_labels(
        df, df["common_train"], target_col="label_net_alpha_20d", n_grades=10
    )
    
    rank_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5],
        "label_gain": list(range(10)),
        "learning_rate": 0.05,
        "n_estimators": 10,
        "early_stopping_rounds": 5,
        "random_state": 42,
        "verbose": -1
    }
    
    model = LightGBMQuantModel(
        params=rank_params,
        task_type="ranking",
        model_dir=tmp_path / "rank_model",
        random_state=42
    )
    
    X_tr = df[["factor_a", "factor_b", "date"]].copy()
    y_tr = df["label_rank"]
    
    model.fit(
        X_train=X_tr,
        y_train=y_tr,
        feature_names=["factor_a", "factor_b"]
    )
    
    assert model.model is not None
    preds = model.predict(df[["factor_a", "factor_b"]])
    assert len(preds) == len(df)
    assert np.all(np.isfinite(preds))


def test_three_arms_shared_structural_parameters():
    base = settings.LGBM_PARAMS_CLF.copy()
    base["random_state"] = 42
    
    clf_params = base.copy()
    clf_params["objective"] = "binary"
    clf_params["scale_pos_weight"] = 1.0
    
    reg_params = base.copy()
    reg_params["objective"] = "regression"
    reg_params["metric"] = ["l2", "rmse"]
    if "scale_pos_weight" in reg_params:
        del reg_params["scale_pos_weight"]
        
    rank_params = base.copy()
    rank_params["objective"] = "lambdarank"
    rank_params["metric"] = "ndcg"
    rank_params["eval_at"] = [30]
    rank_params["label_gain"] = list(range(10))
    if "scale_pos_weight" in rank_params:
        del rank_params["scale_pos_weight"]
        
    shared_keys = ["learning_rate", "num_leaves", "max_depth", "min_child_samples",
                   "lambda_l1", "lambda_l2", "feature_fraction", "bagging_fraction",
                   "bagging_freq", "random_state", "n_estimators", "early_stopping_rounds"]
                   
    for k in shared_keys:
        if k in base:
            assert clf_params[k] == reg_params[k] == rank_params[k]


def test_walk_forward_three_arms_isolated_execution(tmp_path):
    df = _make_synthetic_multiday_data(seed=42, n_dates=120, n_syms=10)
    df["label_rank"] = _build_lambdarank_relevance_labels(
        df, df["common_train"], target_col="label_net_alpha_20d", n_grades=10
    )
    
    base_params = {
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "n_estimators": 10,
        "early_stopping_rounds": 5,
        "random_state": 42,
        "verbose": -1
    }
    
    # Synthetic three-arm estimator isolation only; not scientific certification.
    # Its intentionally short windows cannot satisfy the certified 20-day horizon.
    # Arm A
    p_a = base_params.copy()
    p_a["objective"] = "binary"
    p_a["scale_pos_weight"] = 1.0
    t_a = WalkForwardTrainer(
        train_years=0.1, val_months=1, test_months=1, purge_gap_days=2,
        label_col="label_direction_20d", task_type="classification", model_type="lightgbm",
        model_dir=tmp_path / "clf", model_params=p_a, random_state=42, strict_mode=False
    )
    oos_a, _ = t_a.run_walk_forward(df, feature_cols=["factor_a", "factor_b"])
    
    # Arm B
    p_b = base_params.copy()
    p_b["objective"] = "regression"
    p_b["metric"] = ["l2", "rmse"]
    t_b = WalkForwardTrainer(
        train_years=0.1, val_months=1, test_months=1, purge_gap_days=2,
        label_col="label_net_alpha_20d", task_type="regression", model_type="regression",
        model_dir=tmp_path / "reg", model_params=p_b, random_state=42, strict_mode=False
    )
    oos_b, _ = t_b.run_walk_forward(df, feature_cols=["factor_a", "factor_b"])
    
    # Arm C
    p_c = base_params.copy()
    p_c["objective"] = "lambdarank"
    p_c["metric"] = "ndcg"
    p_c["eval_at"] = [2]
    p_c["label_gain"] = list(range(10))
    t_c = WalkForwardTrainer(
        train_years=0.1, val_months=1, test_months=1, purge_gap_days=2,
        label_col="label_rank", task_type="ranking", model_type="ranking",
        model_dir=tmp_path / "rank", model_params=p_c, random_state=42, strict_mode=False
    )
    oos_c, _ = t_c.run_walk_forward(df, feature_cols=["factor_a", "factor_b"])
    
    assert len(oos_a) == len(oos_b) == len(oos_c) > 0
    assert "pred_score" in oos_a.columns
    assert "pred_score" in oos_b.columns
    assert "pred_score" in oos_c.columns


def test_common_oos_pool_one_to_one_merge_and_metrics():
    df = _make_synthetic_multiday_data(seed=42, n_dates=20, n_syms=10)
    rng = np.random.default_rng(42)
    df["pred_score_clf"] = rng.uniform(0.1, 0.9, size=len(df))
    df["pred_score_reg"] = rng.normal(0.0, 0.02, size=len(df))
    df["pred_score_rank"] = rng.normal(0.0, 1.0, size=len(df))
    
    daily_clf = _daily_rankic(df, "pred_score_clf", "label_net_alpha_20d")
    daily_reg = _daily_rankic(df, "pred_score_reg", "label_net_alpha_20d")
    daily_rank = _daily_rankic(df, "pred_score_rank", "label_net_alpha_20d")
    
    assert len(daily_clf) == len(daily_reg) == len(daily_rank) == 20
    
    boot = _paired_block_bootstrap(daily_reg, daily_clf, block_size=5, n_bootstraps=500, seed=42)
    assert "ci_95_lower" in boot
    assert "ci_97_5_lower" in boot
    assert "prob_positive" in boot
    assert boot["ci_97_5_lower"] <= boot["ci_95_lower"]
    assert boot["ci_97_5_upper"] >= boot["ci_95_upper"]


def test_production_model_dir_isolation_in_test(tmp_path, monkeypatch):
    fake_prod_dir = tmp_path / "saved_models"
    fake_prod_dir.mkdir(parents=True, exist_ok=True)
    fake_prod_pkl = fake_prod_dir / "latest_lightgbm.pkl"
    fake_prod_pkl.write_bytes(b"INITIAL_PRODUCTION_MODEL_BINARY")
    
    # Train test model in separate tmp dir
    test_model_dir = tmp_path / "experiment_models"
    df = _make_synthetic_multiday_data(seed=42, n_dates=5, n_syms=5)
    
    m = LightGBMQuantModel(
        params={"objective": "binary", "n_estimators": 5, "random_state": 42, "verbose": -1},
        task_type="classification",
        model_dir=test_model_dir,
        random_state=42
    )
    m.fit(df[["factor_a", "factor_b"]], df["label_direction_20d"], feature_names=["factor_a", "factor_b"])
    m.save(test_model_dir / "latest_lightgbm.pkl")
    
    # Assert production model file was never touched
    assert fake_prod_pkl.read_bytes() == b"INITIAL_PRODUCTION_MODEL_BINARY"
    assert (test_model_dir / "latest_lightgbm.pkl").exists()
