"""
Tests for Experiment Registry & Baseline Registry (tests/test_phase2_1_experiment_registry.py)
"""
import pytest
from pathlib import Path
from research_v2.registry.schemas import ExperimentRecord
from research_v2.registry.baseline_registry import BaselineRegistry
from research_v2.registry.experiment_registry import ExperimentRegistry


def test_baseline_registry_load_legacy_v1():
    reg = BaselineRegistry()
    base = reg.get("LEGACY_BASELINE_V1")
    assert base.baseline_id == "LEGACY_BASELINE_V1"
    assert base.prediction_baseline["mean_daily_rank_ic"] == 0.0503
    assert base.trading_candidate["cost_adjusted_excess_return"] == 5.72


def test_baseline_immutability():
    reg = BaselineRegistry()
    base = reg.get("LEGACY_BASELINE_V1")
    with pytest.raises(ValueError, match="immutable and cannot be overwritten"):
        reg.register(base)


def test_experiment_registry_validation():
    base_reg = BaselineRegistry()
    exp_reg = ExperimentRegistry(baseline_registry=base_reg)

    # A. 缺少 primary_change
    with pytest.raises(ValueError, match="primary_change is required"):
        rec_bad1 = ExperimentRecord(
            experiment_id="EXP_TEST_001",
            phase="Phase 2.1-A",
            status="DRAFT",
            created_at="2026-08-31",
            parent_baseline_id="LEGACY_BASELINE_V1",
            source_commit="commit_1",
            dataset_id="DATA_300",
            dataset_sha256="sha_1",
            feature_set_id="FEAT_79",
            feature_set_hash="hash_1",
            label_schema_id="LBL_EXEC",
            label_schema_hash="hash_2",
            model_id="lightgbm_clf_baseline",
            model_config_hash="hash_3",
            primary_change="",
            controlled_variables={"dataset": "same"}
        )
        exp_reg.register_experiment(rec_bad1, save_to_disk=False)

    # B. 缺少 controlled_variables
    with pytest.raises(ValueError, match="controlled_variables is required"):
        rec_bad2 = ExperimentRecord(
            experiment_id="EXP_TEST_002",
            phase="Phase 2.1-A",
            status="DRAFT",
            created_at="2026-08-31",
            parent_baseline_id="LEGACY_BASELINE_V1",
            source_commit="commit_1",
            dataset_id="DATA_300",
            dataset_sha256="sha_1",
            feature_set_id="FEAT_79",
            feature_set_hash="hash_1",
            label_schema_id="LBL_EXEC",
            label_schema_hash="hash_2",
            model_id="lightgbm_clf_baseline",
            model_config_hash="hash_3",
            primary_change="LABEL_ENTRY_ALIGNMENT",
            controlled_variables={}
        )
        exp_reg.register_experiment(rec_bad2, save_to_disk=False)

    # C. 不存在的 parent_baseline_id
    with pytest.raises(ValueError, match="does not exist in BaselineRegistry"):
        rec_bad3 = ExperimentRecord(
            experiment_id="EXP_TEST_003",
            phase="Phase 2.1-A",
            status="DRAFT",
            created_at="2026-08-31",
            parent_baseline_id="UNKNOWN_BASELINE",
            source_commit="commit_1",
            dataset_id="DATA_300",
            dataset_sha256="sha_1",
            feature_set_id="FEAT_79",
            feature_set_hash="hash_1",
            label_schema_id="LBL_EXEC",
            label_schema_hash="hash_2",
            model_id="lightgbm_clf_baseline",
            model_config_hash="hash_3",
            primary_change="LABEL_ENTRY_ALIGNMENT",
            controlled_variables={"dataset": "same"}
        )
        exp_reg.register_experiment(rec_bad3, save_to_disk=False)

    # D. 正确注册与重复校验
    rec_good = ExperimentRecord(
        experiment_id="EXP_TEST_004",
        phase="Phase 2.1-A",
        status="DRAFT",
        created_at="2026-08-31",
        parent_baseline_id="LEGACY_BASELINE_V1",
        source_commit="commit_1",
        dataset_id="DATA_300",
        dataset_sha256="sha_1",
        feature_set_id="FEAT_79",
        feature_set_hash="hash_1",
        label_schema_id="LBL_EXEC",
        label_schema_hash="hash_2",
        model_id="lightgbm_clf_baseline",
        model_config_hash="hash_3",
        primary_change="LABEL_ENTRY_ALIGNMENT",
        controlled_variables={"dataset": "same", "features": "same"}
    )
    exp_reg.register_experiment(rec_good, save_to_disk=False)

    with pytest.raises(ValueError, match="Duplicate experiment_id"):
        exp_reg.register_experiment(rec_good, save_to_disk=False)


def test_baseline_comparison_delta_calculation():
    reg = BaselineRegistry()
    cand_pred = {"mean_daily_rank_ic": 0.0550, "nw20_rank_icir": 0.4500}
    cand_trading = {
        "cost_adjusted_excess_return": 7.50,
        "sharpe_ratio": 0.45,
        "max_drawdown": -12.00,
        "fold_win_ratio": 0.60,
        "annualized_turnover": 8.50
    }
    comp = reg.compare(cand_pred, cand_trading, baseline_id="LEGACY_BASELINE_V1")
    assert comp.delta_rank_ic == round(0.0550 - 0.0503, 5)
    assert comp.delta_excess_return == round(7.50 - 5.72, 2)
    assert comp.robust_improvement is True
