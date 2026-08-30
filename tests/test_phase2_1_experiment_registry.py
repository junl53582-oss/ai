"""
Tests for Experiment Registry & Baseline Registry (tests/test_phase2_1_experiment_registry.py)
"""
import json
import shutil
import pytest
from pathlib import Path
from research_v2.registry.schemas import ExperimentRecord, BaselineIntegrityError
from research_v2.registry.baseline_registry import BaselineRegistry
from research_v2.registry.experiment_registry import ExperimentRegistry


def test_baseline_registry_load_legacy_v1():
    reg = BaselineRegistry()
    base = reg.get("LEGACY_BASELINE_V1", verify_integrity=True)
    assert base.baseline_id == "LEGACY_BASELINE_V1"
    assert base.prediction_baseline["mean_daily_rank_ic"] == 0.0503
    assert base.trading_candidate["cost_adjusted_excess_return"] == 5.72


def test_baseline_registry_tamper_detection(tmp_path):
    # 复制一份 baselines 到 tmp_path
    src_dir = Path("reports/baselines")
    dst_dir = tmp_path / "baselines"
    shutil.copytree(src_dir, dst_dir)

    # 篡改 model_comparison.csv 的一个字符
    target_csv = dst_dir / "legacy_v1" / "model_comparison.csv"
    text = target_csv.read_text(encoding="utf-8")
    target_csv.write_text(text + "\n# TAMPERED_DATA", encoding="utf-8")

    reg = BaselineRegistry(baselines_dir=dst_dir)
    with pytest.raises(BaselineIntegrityError, match="Tamper detected"):
        reg.verify_integrity("LEGACY_BASELINE_V1")


def test_baseline_immutability():
    reg = BaselineRegistry()
    base = reg.get("LEGACY_BASELINE_V1", verify_integrity=True)
    with pytest.raises(ValueError, match="immutable and cannot be overwritten"):
        reg.register(base)


def test_missing_metric_comparison_fail_closed():
    reg = BaselineRegistry()
    # 缺少 sharpe_ratio
    cand_pred = {"mean_daily_rank_ic": 0.0550, "nw20_rank_icir": 0.4500}
    cand_trading_missing_sharpe = {
        "cost_adjusted_excess_return": 7.50,
        "max_drawdown": -12.00,
        "fold_win_ratio": 0.60,
        "annualized_turnover": 8.50
    }
    comp = reg.compare(cand_pred, cand_trading_missing_sharpe, baseline_id="LEGACY_BASELINE_V1")
    assert comp.comparison_status == "NOT_COMPARABLE"
    assert comp.robust_improvement is None
    assert "trading.sharpe_ratio" in comp.missing_metrics


def test_valid_baseline_comparison():
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
    assert comp.comparison_status == "COMPARABLE"
    assert comp.delta_rank_ic == round(0.0550 - 0.0503, 5)
    assert comp.delta_excess_return == round(7.50 - 5.72, 2)
    assert comp.robust_improvement is True
