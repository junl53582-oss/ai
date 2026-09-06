import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json

from config.settings import settings
from factors.processor import FactorProcessor, ADVANCED_ALPHA_SPECS
from factors.registry import FactorRegistry
import factors.alternative_factors
from models.registry import ModelRegistry, ModelState
from models.inference import BatchInference, InferenceError
from strategy.portfolio import PortfolioBuilder

CANONICAL_32_FEATURES = [
    'ALPHA_RESIDUAL_MOMENTUM_20', 'ALPHA_TURNOVER_SURPRISE_5_20', 'ALPHA_QUALITY_X_MOMENTUM',
    'ALPHA_LIQUIDITY_X_VOL', 'ALPHA_SHORT_REVERSAL_5', 'ALPHA_IDIO_VOL_PENALTY',
    'ALPHA_MONEY_FLOW_DIV_10', 'adj_open', 'adj_high', 'adj_low', 'adj_close',
    'LOG_CIRC_MV', 'turnover', 'KMID', 'KLEN', 'KMID2', 'KUP', 'KLOW', 'KSFT',
    'ROC5', 'MAX_RATIO_5', 'MIN_RATIO_5', 'MA_RATIO_5', 'ROC10', 'MAX_RATIO_10',
    'MIN_RATIO_10', 'MA_RATIO_10', 'ROC20', 'MAX_RATIO_20', 'MIN_RATIO_20',
    'MA_RATIO_20', 'ROC30'
]
EXPECTED_SCHEMA_HASH = 'ad44898838817b0867f51e224377e1729ced530bd9a42f7b22cd81672e76b21e'


def test_advanced_alpha_specs_and_registry_integration():
    assert len(ADVANCED_ALPHA_SPECS) == 7
    expected_7 = [
        'ALPHA_RESIDUAL_MOMENTUM_20', 'ALPHA_TURNOVER_SURPRISE_5_20', 'ALPHA_QUALITY_X_MOMENTUM',
        'ALPHA_LIQUIDITY_X_VOL', 'ALPHA_SHORT_REVERSAL_5', 'ALPHA_IDIO_VOL_PENALTY',
        'ALPHA_MONEY_FLOW_DIV_10'
    ]
    for alpha_name in expected_7:
        assert alpha_name in ADVANCED_ALPHA_SPECS
        spec = ADVANCED_ALPHA_SPECS[alpha_name]
        assert 'inputs' in spec
        assert 'warmup' in spec
        assert spec.get('missing_policy') == 'warmup_nan'
        assert 'pit_causality' in spec
        assert spec.get('dtype') == 'float64'

    all_factor_cols = FactorProcessor.get_all_factor_cols()
    for alpha_name in expected_7:
        assert alpha_name in all_factor_cols, f'{alpha_name} must be in get_all_factor_cols'

    registered_factors = FactorRegistry.list_all_factors()
    for alpha_name in expected_7:
        assert alpha_name in registered_factors, f'{alpha_name} must be in FactorRegistry'


def test_canonical_feature_schema_hash():
    calculated_hash = FactorProcessor.compute_feature_schema_hash(CANONICAL_32_FEATURES)
    assert calculated_hash == EXPECTED_SCHEMA_HASH, f'Hash mismatch: {calculated_hash} != {EXPECTED_SCHEMA_HASH}'

    prod_dir = Path('saved_models/production/m_20260903_194757_hybrid_bagging_ridge')
    manifest_path = prod_dir / 'manifest.json'
    metadata_path = prod_dir / 'metadata.json'

    assert manifest_path.exists()
    assert metadata_path.exists()

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    assert manifest.get('feature_schema_hash') == EXPECTED_SCHEMA_HASH
    assert manifest.get('feature_count') == 32
    assert manifest.get('feature_names') == CANONICAL_32_FEATURES
    assert metadata.get('feature_schema_hash') == EXPECTED_SCHEMA_HASH
    assert metadata.get('feature_count') == 32


def test_real_production_inference_e2e():
    dataset_path = Path('data_storage/research/factor_matrix_300.parquet')
    assert dataset_path.exists()

    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])

    sub_df = df[df['date'] >= '2026-06-01'].copy()
    assert len(sub_df) > 0

    sub_df = FactorProcessor.compute_advanced_alpha_features(sub_df)
    latest_date = sub_df['date'].max()

    engine = BatchInference()
    assert engine.model_id == 'm_20260903_194757_hybrid_bagging_ridge'
    assert engine.expected_schema_hash == EXPECTED_SCHEMA_HASH

    scored_df = engine.predict(sub_df, date=latest_date)
    assert len(scored_df) > 0
    assert 'pred_score' in scored_df.columns
    assert 'pred_rank' in scored_df.columns
    assert 'feature_schema_hash' in scored_df.columns
    assert 'is_synthetic_demo' in scored_df.columns

    scores = scored_df['pred_score'].values
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert scored_df['feature_schema_hash'].iloc[0] == EXPECTED_SCHEMA_HASH
    assert not scored_df['is_synthetic_demo'].iloc[0]

    builder = PortfolioBuilder(
        top_k_buy=settings.TOP_K_BUY,
        top_k_hold=settings.TOP_K_HOLD,
        weight_method='inv_vol'
    )
    portfolio = builder.build_target_portfolio(scored_df, current_holdings=set(), date=latest_date)
    assert len(portfolio) > 0
    assert len(portfolio) <= settings.TOP_K_BUY
    assert 'target_weight' in portfolio.columns
    assert np.isclose(portfolio['target_weight'].sum(), 0.95, atol=0.05) or portfolio['target_weight'].sum() <= 1.0


def test_fail_closed_on_missing_advanced_alpha():
    dataset_path = Path('data_storage/research/factor_matrix_300.parquet')
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    sub_df = df[df['date'] >= '2026-08-01'].copy()
    sub_df = FactorProcessor.compute_advanced_alpha_features(sub_df)
    latest_date = sub_df['date'].max()

    engine = BatchInference()

    for alpha_col in [
        'ALPHA_RESIDUAL_MOMENTUM_20', 'ALPHA_TURNOVER_SURPRISE_5_20', 'ALPHA_QUALITY_X_MOMENTUM',
        'ALPHA_LIQUIDITY_X_VOL', 'ALPHA_SHORT_REVERSAL_5', 'ALPHA_IDIO_VOL_PENALTY',
        'ALPHA_MONEY_FLOW_DIV_10'
    ]:
        corrupt_df = sub_df.drop(columns=[alpha_col])
        with pytest.raises(InferenceError, match='Fail-Closed'):
            engine.predict(corrupt_df, date=latest_date)


def test_fail_closed_on_mismatched_schema_hash():
    dataset_path = Path('data_storage/research/factor_matrix_300.parquet')
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    sub_df = df[df['date'] >= '2026-08-01'].copy()
    sub_df = FactorProcessor.compute_advanced_alpha_features(sub_df)
    latest_date = sub_df['date'].max()

    engine = BatchInference()
    with pytest.raises(InferenceError, match='Fail-Closed'):
        engine.predict(sub_df, date=latest_date, expected_schema_hash='corrupted_hash_value_12345')


def test_fail_closed_on_non_numeric_feature():
    dataset_path = Path('data_storage/research/factor_matrix_300.parquet')
    df = pd.read_parquet(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    sub_df = df[df['date'] >= '2026-08-01'].copy()
    sub_df = FactorProcessor.compute_advanced_alpha_features(sub_df)
    latest_date = sub_df['date'].max()

    engine = BatchInference()
    corrupt_df = sub_df.copy()
    corrupt_df['turnover'] = corrupt_df['turnover'].astype(str)
    with pytest.raises(InferenceError):
        engine.predict(corrupt_df, date=latest_date)
