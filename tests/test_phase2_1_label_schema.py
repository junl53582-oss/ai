"""
Tests for Label Schema & Feature Schema (tests/test_phase2_1_label_schema.py)
"""
import pytest
from research_v2.labels.schema import ExecutionAlignedLabelSchema
from research_v2.features.feature_set_schema import VersionedFeatureSetSchema


def test_execution_aligned_label_schema_valid():
    schema = ExecutionAlignedLabelSchema()
    schema.validate()
    assert schema.signal_time == "T_CLOSE"
    assert schema.entry_offset_trading_days == 1
    assert schema.holding_period_trading_days == 20
    assert schema.exit_offset_trading_days == 21
    assert schema.entry_offset_trading_days + schema.holding_period_trading_days == schema.exit_offset_trading_days
    
    h = schema.compute_hash()
    assert len(h) == 64


def test_invalid_timing_schema_fails():
    bad_schema = ExecutionAlignedLabelSchema(exit_offset_trading_days=20)
    with pytest.raises(ValueError, match="Timing mismatch"):
        bad_schema.validate()

    bad_signal = ExecutionAlignedLabelSchema(signal_time="T_OPEN")
    with pytest.raises(ValueError, match="signal_time must be T_CLOSE"):
        bad_signal.validate()


def test_feature_set_schema_validation():
    feat_schema = VersionedFeatureSetSchema(
        feature_set_id="FEAT_PROD_79",
        feature_groups={
            "price_volume": ["f1", "f2"],
            "volatility": ["f3"]
        },
        feature_names=["f1", "f2", "f3"],
        feature_count=3,
        created_from_commit="commit_sha_1"
    )
    feat_schema.validate()
    h = feat_schema.compute_hash()
    assert len(h) == 64

    # Count mismatch
    bad_feat = VersionedFeatureSetSchema(
        feature_set_id="FEAT_BAD",
        feature_groups={"price_volume": ["f1"]},
        feature_names=["f1", "f2"],
        feature_count=1,
        created_from_commit="commit_sha_1"
    )
    with pytest.raises(ValueError, match="Feature names in groups do not match"):
        bad_feat.validate()
