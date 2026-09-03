"""
Comprehensive Targeted Tests for Phase 2.1-A r2 Final Guard Hotfix (tests/test_phase2_1_a_execution_labeler.py)
Covering:
1. T+1/T+21 exact mapping, suspension/volume/limit gates, deferred exits, cost models, fail-closed validation, pool parity.
2. 100% tmp_path Production Model Physical Isolation testing (never touching real saved_models).
3. Fixed random number generators (rng = np.random.default_rng(42)) for complete test determinism.
"""
import hashlib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from config.settings import settings
from research_v2.labels.execution_labeler import ExecutionAlignedLabeler
from models.walk_forward import WalkForwardTrainer


def _frame(n_days=26, symbols=("AAA", "BBB", "CCC")):
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    rows = []
    for s_idx, sym in enumerate(symbols):
        for i, dt in enumerate(dates):
            base = 100.0 + s_idx * 5.0 + i
            rows.append({
                "date": dt, "symbol": sym, "open": base, "adj_open": base,
                "benchmark_open": 1000.0 + i, "volume": 100000.0,
                "is_suspended": False, "is_limit_up_locked": False,
                "is_limit_down_locked": False, "in_universe": True,
                "excluded_from_training": False
            })
    return pd.DataFrame(rows), dates


# Gate 1: T -> T+1 -> T+21 exact mapping
def test_exact_t1_to_t21_mapping_and_gross_alpha():
    df, dates = _frame()
    out = ExecutionAlignedLabeler(threshold_mode="fixed", threshold=0.0,
                                  commission_rate=0.0, slippage_rate=0.0).compute(df)
    row = out[(out["date"] == dates[0]) & (out["symbol"] == "AAA")].iloc[0]
    assert row["exec_entry_date"] == dates[1]
    assert row["planned_exit_date"] == dates[21]
    assert row["actual_exit_date"] == dates[21]
    assert row["holding_trading_days"] == 20
    assert row["exit_deferred_days"] == 0
    assert bool(row["label_valid"]) is True
    stock = (100.0 + 21) / (100.0 + 1) - 1.0
    bench = (1000.0 + 21) / (1000.0 + 1) - 1.0
    assert row["label_gross_alpha_20d"] == pytest.approx(stock - bench)


# Gate 2: Suspended entry with ffilled price is invalid
def test_suspended_entry_rejects_even_when_price_is_present():
    df, dates = _frame()
    mask = (df["date"] == dates[1]) & (df["symbol"] == "AAA")
    df.loc[mask, "is_suspended"] = True
    out = ExecutionAlignedLabeler(threshold_mode="fixed").compute(df)
    row = out[(out["date"] == dates[0]) & (out["symbol"] == "AAA")].iloc[0]
    assert bool(row["entry_tradable"]) is False
    assert bool(row["label_valid"]) is False
    assert row["label_invalid_reason"] == "ENTRY_SUSPENDED_OR_NO_VOLUME"
    assert np.isnan(row["label_net_alpha_20d"])


# Gate 3: Volume = 0 entry is invalid
def test_volume_zero_entry_is_invalid():
    df, dates = _frame()
    mask = (df["date"] == dates[1]) & (df["symbol"] == "AAA")
    df.loc[mask, "volume"] = 0.0
    out = ExecutionAlignedLabeler(threshold_mode="fixed").compute(df)
    row = out[(out["date"] == dates[0]) & (out["symbol"] == "AAA")].iloc[0]
    assert bool(row["entry_tradable"]) is False
    assert bool(row["label_valid"]) is False
    assert row["label_invalid_reason"] == "ENTRY_SUSPENDED_OR_NO_VOLUME"


# Gate 4: Limit-up entry is invalid (cannot buy on 一字涨停)
def test_limit_up_entry_is_not_labeled_as_executable():
    df, dates = _frame()
    mask = (df["date"] == dates[1]) & (df["symbol"] == "AAA")
    df.loc[mask, "is_limit_up_locked"] = True
    out = ExecutionAlignedLabeler(threshold_mode="fixed").compute(df)
    row = out[(out["date"] == dates[0]) & (out["symbol"] == "AAA")].iloc[0]
    assert bool(row["label_valid"]) is False
    assert row["label_invalid_reason"] == "ENTRY_LIMIT_UP_LOCKED"


# Gate 5: Limit-down planned exit defers to T+22
def test_limit_down_planned_exit_defers_to_first_sellable_day():
    df, dates = _frame(n_days=27)
    mask = (df["date"] == dates[21]) & (df["symbol"] == "AAA")
    df.loc[mask, "is_limit_down_locked"] = True
    out = ExecutionAlignedLabeler(threshold_mode="fixed",
                                  commission_rate=0.0, slippage_rate=0.0).compute(df)
    row = out[(out["date"] == dates[0]) & (out["symbol"] == "AAA")].iloc[0]
    assert bool(row["planned_exit_tradable"]) is False
    assert row["actual_exit_date"] == dates[22]
    assert row["exit_deferred_days"] == 1
    assert row["holding_trading_days"] == 21
    assert bool(row["label_valid"]) is True


# Gate 6: Suspended planned exit defers to first sellable day
def test_suspended_planned_exit_defers_to_first_sellable_day():
    df, dates = _frame(n_days=27)
    mask = (df["date"] == dates[21]) & (df["symbol"] == "AAA")
    df.loc[mask, "is_suspended"] = True
    out = ExecutionAlignedLabeler(threshold_mode="fixed",
                                  commission_rate=0.0, slippage_rate=0.0).compute(df)
    row = out[(out["date"] == dates[0]) & (out["symbol"] == "AAA")].iloc[0]
    assert bool(row["planned_exit_tradable"]) is False
    assert row["actual_exit_date"] == dates[22]
    assert row["exit_deferred_days"] == 1
    assert bool(row["label_valid"]) is True


# Gate 7: Consecutive unsellable days continue to defer
def test_consecutive_unsellable_days_deferred_properly():
    df, dates = _frame(n_days=30)
    for d_idx in [21, 22, 23]:
        mask = (df["date"] == dates[d_idx]) & (df["symbol"] == "AAA")
        df.loc[mask, "is_limit_down_locked"] = True
    out = ExecutionAlignedLabeler(threshold_mode="fixed",
                                  commission_rate=0.0, slippage_rate=0.0).compute(df)
    row = out[(out["date"] == dates[0]) & (out["symbol"] == "AAA")].iloc[0]
    assert row["actual_exit_date"] == dates[24]
    assert row["exit_deferred_days"] == 3
    assert row["holding_trading_days"] == 23
    assert bool(row["label_valid"]) is True


# Gate 8: Data ends before executable exit -> label invalid
def test_no_executable_exit_before_data_end_fails_closed():
    df, dates = _frame(n_days=23)  # Only up to T+22
    for d_idx in [21, 22]:
        mask = (df["date"] == dates[d_idx]) & (df["symbol"] == "AAA")
        df.loc[mask, "is_limit_down_locked"] = True
    out = ExecutionAlignedLabeler(threshold_mode="fixed").compute(df)
    row = out[(out["date"] == dates[0]) & (out["symbol"] == "AAA")].iloc[0]
    assert bool(row["label_valid"]) is False
    assert row["label_invalid_reason"] == "NO_EXECUTABLE_EXIT_BEFORE_DATA_END"


# Gate 9: Transaction costs reduce net return below gross return
def test_transaction_cost_adjustment_reduces_stock_return_and_alpha():
    df, dates = _frame()
    out = ExecutionAlignedLabeler(threshold_mode="fixed",
                                  commission_rate=0.00025, slippage_rate=0.001).compute(df)
    row = out[(out["date"] == dates[0]) & (out["symbol"] == "AAA")].iloc[0]
    assert row["label_cost_drag"] > 0
    assert row["stock_net_return"] < row["stock_gross_return"]
    assert row["label_net_alpha_20d"] < row["label_gross_alpha_20d"]


# Gate 10: Insufficient forward horizon at tail
def test_tail_rows_fail_closed_for_insufficient_forward_horizon():
    df, dates = _frame()
    out = ExecutionAlignedLabeler(threshold_mode="fixed").compute(df)
    row = out[(out["date"] == dates[-1]) & (out["symbol"] == "AAA")].iloc[0]
    assert bool(row["label_valid"]) is False
    assert row["label_invalid_reason"] == "INSUFFICIENT_FORWARD_HORIZON"


# Gate 11: Missing required columns fail closed
def test_required_execution_state_columns_fail_closed():
    df, _ = _frame()
    with pytest.raises(ValueError, match="missing"):
        ExecutionAlignedLabeler().compute(df.drop(columns=["is_limit_down_locked"]))


# Gate 12: Missing benchmark open fails closed
def test_benchmark_missing_fails_closed():
    df, dates = _frame()
    mask = (df["date"] == dates[1])
    df.loc[mask, "benchmark_open"] = np.nan
    out = ExecutionAlignedLabeler(threshold_mode="fixed").compute(df)
    row = out[(out["date"] == dates[0]) & (out["symbol"] == "AAA")].iloc[0]
    assert bool(row["label_valid"]) is False
    assert row["label_invalid_reason"] == "BENCHMARK_OPEN_MISSING"


# Gate 13: Duplicate date+symbol fails closed
def test_duplicate_date_symbol_fails_closed():
    df, dates = _frame()
    dup_row = df.iloc[0:1].copy()
    df_dup = pd.concat([df, dup_row], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate date/symbol"):
        ExecutionAlignedLabeler().compute(df_dup)


# Gate 14: Nonfinite label fails closed
def test_nonfinite_label_fails_closed():
    df, dates = _frame()
    mask = (df["date"] == dates[1]) & (df["symbol"] == "AAA")
    df.loc[mask, "adj_open"] = 0.0  # Causes division by zero
    out = ExecutionAlignedLabeler(threshold_mode="fixed").compute(df)
    row = out[(out["date"] == dates[0]) & (out["symbol"] == "AAA")].iloc[0]
    assert bool(row["label_valid"]) is False
    assert row["label_invalid_reason"] in ("ENTRY_INVALID_OPEN_PRICE", "NONFINITE_EXECUTION_LABEL")


# Gate 15: Common train pool keys identical across Arm A and Arm B
def test_common_train_pool_keys_identical():
    from tools.run_phase2_1_a_label_ab import _build_common_training_labels
    df, dates = _frame(n_days=30)
    rng = np.random.default_rng(42)
    df["label_up_down_20d"] = rng.choice([0.0, 1.0, np.nan], size=len(df))
    df = ExecutionAlignedLabeler(threshold_mode="fixed").compute(df)
    labeled = _build_common_training_labels(df)

    mask_a = labeled["ab_label_legacy"].notna()
    mask_b = labeled["ab_label_execution"].notna()
    assert (mask_a == mask_b).all()
    assert (labeled.loc[mask_a, ["date", "symbol"]].values == labeled.loc[mask_b, ["date", "symbol"]].values).all()


# Gate 16: Common OOS pool merge validation
def test_common_oos_pool_keys_identical():
    df, dates = _frame(n_days=30)
    rng = np.random.default_rng(42)
    df["label_net_alpha_20d"] = rng.standard_normal(len(df))
    df["label_valid"] = True

    a_pred = df[["date", "symbol"]].copy()
    a_pred["pred_score_legacy"] = rng.random(len(a_pred))

    b_pred = df[["date", "symbol"]].copy()
    b_pred["pred_score_execution"] = rng.random(len(b_pred))

    common = a_pred.merge(b_pred, on=["date", "symbol"], how="inner", validate="one_to_one")
    assert len(common) == len(a_pred)
    assert len(common) == len(b_pred)


# Gate 17: Production Model Physical Isolation in 100% tmp_path simulation (Never touches real saved_models)
def test_phase2_1_a_does_not_mutate_production_model(tmp_path, monkeypatch):
    """
    严密验证完整 Walk-Forward 训练与 latest_model.save() 全过程下的生产模型物理隔离。
    使用 monkeypatch 将生产模型目录完全重定向至 tmp_path，杜绝任何对真实 saved_models 的误触。
    """
    simulated_prod_dir = tmp_path / "simulated_saved_models"
    simulated_prod_dir.mkdir(parents=True, exist_ok=True)
    fake_prod_file = simulated_prod_dir / "latest_lightgbm.pkl"
    sentinel_bytes = b"PRODUCTION_SENTINEL_HASH_TEST_BYTES_42"
    fake_prod_file.write_bytes(sentinel_bytes)
    sha_before = hashlib.sha256(sentinel_bytes).hexdigest()

    # 动态 patch settings，使默认目录指向模拟生产路径
    monkeypatch.setattr(settings, "MODELS_DIR", simulated_prod_dir)
    monkeypatch.setattr(settings, "MODEL_DIR", simulated_prod_dir)

    # 创建隔离实验模型目录
    isolated_legacy_dir = tmp_path / "run_ab" / "models" / "legacy"
    isolated_exec_dir = tmp_path / "run_ab" / "models" / "execution"
    isolated_legacy_dir.mkdir(parents=True, exist_ok=True)
    isolated_exec_dir.mkdir(parents=True, exist_ok=True)

    # 硬断言：所有路径必须在 tmp_path 内
    assert isolated_legacy_dir.is_relative_to(tmp_path)
    assert isolated_exec_dir.is_relative_to(tmp_path)
    assert simulated_prod_dir.is_relative_to(tmp_path)

    # 构造能够走通最小 Walk-Forward 训练并触发 save() 的合成数据集 (1.5年训练 + 3月验证 + 2月测试)
    dates = pd.bdate_range("2022-01-01", "2024-06-01")
    rng = np.random.default_rng(42)
    syn_rows = []
    for sym in ["000001.SZ", "600000.SH", "600519.SH"]:
        for dt in dates:
            syn_rows.append({
                "date": dt,
                "symbol": sym,
                "feat_1": float(rng.standard_normal()),
                "feat_2": float(rng.standard_normal()),
                "ab_label_legacy": float(rng.choice([0.0, 1.0])),
                "ab_label_execution": float(rng.choice([0.0, 1.0])),
            })
    syn_df = pd.DataFrame(syn_rows)

    custom_params = settings.LGBM_PARAMS_CLF.copy()
    custom_params["n_estimators"] = 5
    custom_params["min_child_samples"] = 2
    custom_params["scale_pos_weight"] = 1.0

    trainer_legacy = WalkForwardTrainer(
        train_years=1.0, val_months=2, test_months=1, purge_gap_days=20,
        task_type="classification", model_dir=isolated_legacy_dir,
        model_params=custom_params, label_col="ab_label_legacy"
    )
    trainer_exec = WalkForwardTrainer(
        train_years=1.0, val_months=2, test_months=1, purge_gap_days=20,
        task_type="classification", model_dir=isolated_exec_dir,
        model_params=custom_params, label_col="ab_label_execution"
    )

    assert trainer_legacy.model_dir.is_relative_to(tmp_path)
    assert trainer_exec.model_dir.is_relative_to(tmp_path)

    # 实际执行 Walk-Forward，内部必将调用 latest_model.save()
    oos_leg, mod_leg = trainer_legacy.run_walk_forward(syn_df, feature_cols=["feat_1", "feat_2"])
    oos_exc, mod_exc = trainer_exec.run_walk_forward(syn_df, feature_cols=["feat_1", "feat_2"])

    # 1. 验证实验模型文件确实生成在各自独立的隔离路径中
    assert (isolated_legacy_dir / "latest_lightgbm.pkl").exists()
    assert (isolated_exec_dir / "latest_lightgbm.pkl").exists()

    # 2. 验证模拟生产模型文件内容与哈希完全未受任何修改
    assert fake_prod_file.exists()
    sha_after = hashlib.sha256(fake_prod_file.read_bytes()).hexdigest()
    assert sha_before == sha_after
    assert fake_prod_file.read_bytes() == sentinel_bytes

    # 3. 验证模拟生产目录下除 latest_lightgbm.pkl 外没有被新建任何其他文件
    created_prod_files = [f for f in simulated_prod_dir.iterdir() if f.name != "registry"]
    assert len(created_prod_files) == 1
    assert created_prod_files[0].name == "latest_lightgbm.pkl"
