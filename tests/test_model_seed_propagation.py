"""
Phase 2.0.2 Model Seed Propagation Tests (tests/test_model_seed_propagation.py)
严格验证 random_state 在 WalkForwardTrainer 与 LightGBMQuantModel 之间的端到端传播与可观测性。
"""
import pytest
import numpy as np
import pandas as pd
from config.settings import settings
from models.lightgbm_model import LightGBMQuantModel
from models.walk_forward import WalkForwardTrainer


def test_walk_forward_trainer_stores_random_state():
    trainer = WalkForwardTrainer(random_state=2026)
    assert trainer.random_state == 2026


def test_lightgbm_quant_model_accepts_and_observes_random_state():
    m = LightGBMQuantModel(task_type="classification", random_state=2026)
    assert m.random_state == 2026
    assert m.params["random_state"] == 2026
    assert m.params["feature_fraction_seed"] == 2026
    assert m.params["bagging_seed"] == 2026
    assert m.params["data_random_seed"] == 2026


def test_seed_42_and_2026_are_distinct():
    m1 = LightGBMQuantModel(task_type="classification", random_state=42)
    m2 = LightGBMQuantModel(task_type="classification", random_state=2026)
    assert m1.random_state == 42
    assert m2.random_state == 2026
    assert m1.params["random_state"] != m2.params["random_state"]


def test_walk_forward_propagates_seed_to_models():
    dates = pd.date_range("2023-01-01", periods=120, freq="B")
    symbols = ["000001.SZ", "000002.SZ", "600519.SH"]
    rows = []
    for dt in dates:
        for sym in symbols:
            rows.append({
                "date": dt,
                "symbol": sym,
                "F1": np.random.normal(0, 1),
                "label_excess_20d": np.random.normal(0, 0.05),
                "label_up_down_20d": np.random.choice([0.0, 1.0]),
                "in_universe": True,
                "excluded_from_training": False
            })
    df = pd.DataFrame(rows)
    # Synthetic seed-propagation compatibility test only; not scientific certification.
    trainer = WalkForwardTrainer(
        train_years=0.2,
        val_months=1,
            test_months=1,
            purge_gap_days=5,
            random_state=3407,
            strict_mode=False
    )
    oos_df, last_model = trainer.run_walk_forward(df, feature_cols=["F1"])
    assert last_model is not None
    assert last_model.random_state == 3407
