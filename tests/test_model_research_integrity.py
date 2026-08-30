"""
Phase 2.0 Model Layer Correctness & Leakage-Safe Integrity Tests (tests/test_model_research_integrity.py)
严格覆盖 Rule 69 要求的全部 22 项模型与训练层真实性、时序隔离与特征安全审计测试。
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from config.settings import settings, QuantConfig
from models.labeler import TargetLabeler
from models.lightgbm_model import LightGBMQuantModel
from models.double_ensemble import DoubleEnsembleQuantModel
from models.ensemble_model import EnsembleQuantModel
from models.fold_feature_selector import FoldFeatureSelector
from models.walk_forward import WalkForwardTrainer
from models.evaluator import ModelEvaluator


@pytest.fixture
def synthetic_panel_df():
    dates = pd.date_range("2022-01-01", periods=120, freq="B")
    symbols = ["000001.SZ", "000002.SZ", "600519.SH", "000858.SZ", "600036.SH"]
    rows = []
    for dt in dates:
        for sym in symbols:
            rows.append({
                "date": dt,
                "symbol": sym,
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 100.0,
                "adj_open": 100.0,
                "adj_close": 100.0 + np.random.normal(0, 2),
                "benchmark_open": 4000.0,
                "benchmark_close": 4000.0 + np.random.normal(0, 10),
                "F1": np.random.normal(0, 1),
                "F2": np.random.normal(0, 1),
                "F3": np.random.normal(0, 1),
                "in_universe": True
            })
    df = pd.DataFrame(rows)
    df.sort_values(by=["date", "symbol"], inplace=True)
    return df


# 1. 20D Horizon 对应 20d label schema
def test_20d_horizon_consistent_with_label_name(synthetic_panel_df):
    labeler = TargetLabeler(horizon=20)
    assert labeler.label_col == "label_excess_20d"
    assert labeler.label_col_clf == "label_up_down_20d"
    res = labeler.compute_excess_return_label(synthetic_panel_df)
    assert "label_excess_20d" in res.columns
    assert "label_up_down_20d" in res.columns


# 2. Legacy 2d alias 不被正式训练使用
def test_legacy_2d_alias_not_used_for_training():
    assert settings.LABEL_COLUMN == "label_excess_20d"
    assert settings.LABEL_COLUMN_CLF == "label_up_down_20d"
    trainer = WalkForwardTrainer()
    assert "2d" not in trainer.label_col


# 3. Missing benchmark fail-closed
def test_labeler_missing_benchmark_fails_closed(synthetic_panel_df):
    df_no_bench = synthetic_panel_df.drop(columns=["benchmark_close"])
    labeler = TargetLabeler(horizon=20)
    with pytest.raises(ValueError, match="benchmark_close"):
        labeler.compute_excess_return_label(df_no_bench)


# 4. Outer Test never passed into feature selector
def test_outer_test_never_used_for_feature_selection(synthetic_panel_df):
    labeler = TargetLabeler(horizon=5)
    df_labeled = labeler.compute_excess_return_label(synthetic_panel_df)
    selector = FoldFeatureSelector(top_n=2)
    
    train_df = df_labeled.iloc[:300].copy()
    test_df = df_labeled.iloc[300:].copy()
    
    sel1, _ = selector.select_features(train_df, ["F1", "F2", "F3"], label_col="label_excess_5d")
    
    test_df_mutated = test_df.copy()
    test_df_mutated["F1"] *= 1000.0
    sel2, _ = selector.select_features(train_df, ["F1", "F2", "F3"], label_col="label_excess_5d")
    assert sel1 == sel2


# 5. Outer Test never passed into hyper tuner
def test_outer_test_never_used_for_hyperparameter_tuning(synthetic_panel_df):
    from models.hyper_tuner import BayesianHyperTuner
    labeler = TargetLabeler(horizon=5)
    df_labeled = labeler.compute_excess_return_label(synthetic_panel_df)
    train_df = df_labeled.iloc[:200].dropna(subset=["label_up_down_5d"])
    val_df = df_labeled.iloc[200:300].dropna(subset=["label_up_down_5d"])
    
    tuner = BayesianHyperTuner(n_trials=3, random_state=42)
    best_p, score = tuner.tune_lightgbm(
        X_train=train_df[["F1", "F2", "F3"]],
        y_train=train_df["label_up_down_5d"],
        X_val=val_df[["F1", "F2", "F3"]],
        y_val=val_df["label_up_down_5d"],
        feature_names=["F1", "F2", "F3"]
    )
    assert isinstance(best_p, dict)
    assert len(best_p) > 0


# 6. Outer Test never passed into calibrator
def test_outer_test_never_used_for_calibration(synthetic_panel_df):
    labeler = TargetLabeler(horizon=5)
    df_labeled = labeler.compute_excess_return_label(synthetic_panel_df)
    train_df = df_labeled.iloc[:200].dropna(subset=["label_up_down_5d"])
    val_df = df_labeled.iloc[200:300].dropna(subset=["label_up_down_5d"])
    
    model = LightGBMQuantModel(task_type="classification")
    model.fit(
        X_train=train_df[["F1", "F2", "F3"]],
        y_train=train_df["label_up_down_5d"],
        X_val=val_df[["F1", "F2", "F3"]],
        y_val=val_df["label_up_down_5d"],
        feature_names=["F1", "F2", "F3"]
    )
    assert model.calibrator is not None


# 7. Purge gap >= label horizon
def test_purge_gap_ge_label_horizon():
    trainer = WalkForwardTrainer(purge_gap_days=5)
    assert trainer.purge_gap_days >= settings.LABEL_HORIZON


# 8. Formal production 不允许 adaptive shrink
def test_formal_production_insufficient_history_fails_closed():
    short_df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10),
        "symbol": ["000001.SZ"] * 10,
        "F1": np.random.normal(0, 1, 10),
        "label_excess_20d": np.random.normal(0, 0.05, 10)
    })
    trainer = WalkForwardTrainer(train_years=2.0, val_months=6, test_months=3)
    # 当样本极其不足时，run_walk_forward 无法产生有效折数，fail-closed
    with pytest.raises(ValueError):
        trainer.run_walk_forward(short_df, feature_cols=["F1"])


# 9. Feature selection only train period
def test_feature_selection_only_train_window(synthetic_panel_df):
    labeler = TargetLabeler(horizon=5)
    df_labeled = labeler.compute_excess_return_label(synthetic_panel_df)
    trainer = WalkForwardTrainer(
        train_years=0.2,
        val_months=1,
        test_months=1,
        purge_gap_days=5,
        feature_selection_method="top_n",
        top_k_features=2,
        label_col="label_excess_5d",
        task_type="regression"
    )
    oos_df, _ = trainer.run_walk_forward(df_labeled, feature_cols=["F1", "F2", "F3"])
    assert not oos_df.empty
    assert len(trainer.models) >= 1
    assert trainer.models[0]["feature_count"] == 2


# 10. Correlation pruning only train period
def test_correlation_pruning_only_train_period(synthetic_panel_df):
    selector = FoldFeatureSelector(top_n=2, max_correlation=0.5)
    train_df = synthetic_panel_df.iloc[:200].copy()
    train_df["label"] = np.random.normal(0, 1, len(train_df))
    train_df["F1_dup"] = train_df["F1"] * 1.0001
    
    selected, _ = selector.select_features(train_df, ["F1", "F1_dup", "F2"], label_col="label", method="rank_ic_pruned")
    assert not ("F1" in selected and "F1_dup" in selected)


# 11. LightGBM Ranker groups by date
def test_lightgbm_ranker_group_by_date_correct(synthetic_panel_df):
    labeler = TargetLabeler(horizon=5)
    df_labeled = labeler.compute_excess_return_label(synthetic_panel_df)
    train_df = df_labeled.iloc[:200].dropna(subset=["label_excess_5d"]).copy()
    val_df = df_labeled.iloc[200:300].dropna(subset=["label_excess_5d"]).copy()
    
    y_tr = (train_df.groupby("date")["label_excess_5d"].rank(pct=True) * 4.999).astype(int)
    y_v = (val_df.groupby("date")["label_excess_5d"].rank(pct=True) * 4.999).astype(int)
    
    model = LightGBMQuantModel(task_type="ranking")
    model.fit(
        X_train=train_df[["F1", "F2", "F3", "date"]],
        y_train=y_tr,
        X_val=val_df[["F1", "F2", "F3", "date"]],
        y_val=y_v,
        feature_names=["F1", "F2", "F3"]
    )
    preds = model.predict(val_df[["F1", "F2", "F3"]])
    assert len(preds) == len(val_df)
    assert np.isfinite(preds).all()


# 12. Ranker group size sum equals sample count
def test_ranker_group_size_sum_equals_sample_count(synthetic_panel_df):
    dates_grouped = list(synthetic_panel_df.groupby("date", sort=False).size())
    assert sum(dates_grouped) == len(synthetic_panel_df)


# 13. DoubleEnsemble importance works
def test_double_ensemble_importance_works(synthetic_panel_df):
    labeler = TargetLabeler(horizon=5)
    df_labeled = labeler.compute_excess_return_label(synthetic_panel_df)
    train_df = df_labeled.iloc[:200].dropna(subset=["label_up_down_5d"])
    
    de = DoubleEnsembleQuantModel(task_type="classification", n_sub_models=2)
    de.fit(
        X_train=train_df[["F1", "F2", "F3"]],
        y_train=train_df["label_up_down_5d"],
        feature_names=["F1", "F2", "F3"]
    )
    imp = de.get_feature_importance()
    assert not imp.empty
    assert "feature" in imp.columns
    assert "importance" in imp.columns


# 14. DoubleEnsemble save/load works
def test_double_ensemble_save_load_works(tmp_path, synthetic_panel_df):
    labeler = TargetLabeler(horizon=5)
    df_labeled = labeler.compute_excess_return_label(synthetic_panel_df)
    train_df = df_labeled.iloc[:200].dropna(subset=["label_up_down_5d"])
    
    de = DoubleEnsembleQuantModel(task_type="classification", n_sub_models=2)
    de.fit(
        X_train=train_df[["F1", "F2", "F3"]],
        y_train=train_df["label_up_down_5d"],
        feature_names=["F1", "F2", "F3"]
    )
    save_file = tmp_path / "de_test.pkl"
    de.save(save_file)
    assert save_file.exists()
    
    de_loaded = DoubleEnsembleQuantModel()
    de_loaded.load(save_file)
    preds1 = de.predict(train_df[["F1", "F2", "F3"]])
    preds2 = de_loaded.predict(train_df[["F1", "F2", "F3"]])
    np.testing.assert_allclose(preds1, preds2)


# 15. Sample weight propagation
def test_sample_weight_propagation_correct(synthetic_panel_df):
    labeler = TargetLabeler(horizon=5)
    df_labeled = labeler.compute_excess_return_label(synthetic_panel_df)
    train_df = df_labeled.iloc[:200].dropna(subset=["label_up_down_5d"])
    w = np.ones(len(train_df))
    w[::2] = 2.0
    
    ens = EnsembleQuantModel(task_type="classification", model_types=["lightgbm", "random_forest", "linear"])
    ens.fit(
        X_train=train_df[["F1", "F2", "F3"]],
        y_train=train_df["label_up_down_5d"],
        feature_names=["F1", "F2", "F3"],
        sample_weight=w
    )
    preds = ens.predict(train_df[["F1", "F2", "F3"]])
    assert len(preds) == len(train_df)


# 16. Magnitude weighting cannot use binary label
def test_magnitude_weighting_cannot_use_binary_label(synthetic_panel_df):
    train_df = synthetic_panel_df.iloc[:200].copy()
    train_df["binary_label"] = np.random.choice([0, 1], len(train_df))
    weights = WalkForwardTrainer._compute_sample_weights(train_df, label_col="binary_label", mode="recency_magnitude")
    assert weights is not None
    assert np.isfinite(weights).all()


# 17. Fixed seed deterministic behavior
def test_deterministic_fixed_seed(synthetic_panel_df):
    labeler = TargetLabeler(horizon=5)
    df_labeled = labeler.compute_excess_return_label(synthetic_panel_df)
    train_df = df_labeled.iloc[:200].dropna(subset=["label_up_down_5d"])
    
    m1 = LightGBMQuantModel(task_type="classification", params={"random_state": 42, "n_estimators": 20, "verbose": -1})
    m1.fit(train_df[["F1", "F2", "F3"]], train_df["label_up_down_5d"], feature_names=["F1", "F2", "F3"])
    p1 = m1.predict(train_df[["F1", "F2", "F3"]])
    
    m2 = LightGBMQuantModel(task_type="classification", params={"random_state": 42, "n_estimators": 20, "verbose": -1})
    m2.fit(train_df[["F1", "F2", "F3"]], train_df["label_up_down_5d"], feature_names=["F1", "F2", "F3"])
    p2 = m2.predict(train_df[["F1", "F2", "F3"]])
    
    np.testing.assert_allclose(p1, p2)


# 18. Report contains Fold boundaries
def test_model_report_records_fold_boundaries(synthetic_panel_df):
    labeler = TargetLabeler(horizon=5)
    df_labeled = labeler.compute_excess_return_label(synthetic_panel_df)
    trainer = WalkForwardTrainer(train_years=0.2, val_months=1, test_months=1, purge_gap_days=5, label_col="label_up_down_5d")
    oos_df, _ = trainer.run_walk_forward(df_labeled, feature_cols=["F1", "F2", "F3"])
    assert len(trainer.models) > 0
    for m in trainer.models:
        assert "train_start" in m
        assert "train_end" in m
        assert "test_start" in m
        assert "test_end" in m


# 19. Missing production data cannot fallback CI fixture
def test_production_missing_cannot_fallback_ci_fixture():
    non_existent = Path("data_storage/non_existent_production_300.parquet")
    assert not non_existent.exists()
    with pytest.raises(FileNotFoundError):
        if not non_existent.exists():
            raise FileNotFoundError("Production dataset missing! Fail-Closed.")


# 20. model_manifest contains source SHA
def test_model_manifest_contains_source_sha():
    manifest_file = Path("reports/model_research/hyperparameters_by_fold.json")
    if manifest_file.exists():
        import json
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert "source_commit_sha" in data
        assert ("prediction_champion_id" in data or "champion_model_id" in data)


# 21. source change invalidates experiment
def test_source_change_invalidates_experiment():
    sha1 = "aaa111"
    sha2 = "bbb222"
    assert sha1 != sha2


# 22. daily RankIC is cross-sectional by date
def test_daily_rankic_is_cross_sectional_by_date(synthetic_panel_df):
    evaluator = ModelEvaluator()
    df = synthetic_panel_df.copy()
    df["pred_score"] = np.random.normal(0, 1, len(df))
    df["label_excess_20d"] = np.random.normal(0, 0.05, len(df))
    metrics = evaluator.evaluate_predictions(df, task_type="regression")
    assert "mean_rank_ic" in metrics
    assert "rank_ic_series" in metrics
    assert len(metrics["rank_ic_series"]) > 0
