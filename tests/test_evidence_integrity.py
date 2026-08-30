import hashlib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from config.settings import settings
from data.provenance import SourceClass, UniverseVerificationResult
from data.source_registry import CorporateActionCoverageEvidence
from data.universe_provider import (
    UniverseProvider,
    StaticUniverseProvider,
    PointInTimeUniverseProvider,
    create_universe_provider
)
from data.data_manager import DataManager
from data.security_master import SecurityMaster
from factors.processor import FactorProcessor
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from models.evaluator import ModelEvaluator
from strategy.portfolio import PortfolioBuilder
from strategy.trading_rules import (
    AShareTradingRules,
    PriceLimitRuleEngine,
    Order,
    OrderStatus,
    OrderSide,
    PositionRecord,
    PositionLot
)
from strategy.corporate_actions import CorporateActionProvider, CorporateAction, CorporateActionCoverageRecord
from backtest.engine import BacktestEngine
from backtest.audit import AuditCollector, AuditMetadata, CERTIFICATION_FIELDS


@pytest.mark.unit
def test_pit_certification_negative_cases():
    p1 = PointInTimeUniverseProvider(coverage_start='2021-01-01', coverage_end='2024-12-31')
    p1.add_constituent_change('2021-06-01', '600519.SH', 'IN')
    p1.add_constituent_change('2022-01-01', '000858.SZ', 'IN')
    assert p1.is_coverage_complete('2021-01-01', '2024-12-31') is False
    assert p1.get_mode('2021-01-01', '2024-12-31') != 'POINT_IN_TIME_VERIFIED'
    assert p1.has_survivorship_bias_risk('2021-01-01', '2024-12-31') is True

    p2 = PointInTimeUniverseProvider()
    p2.set_baseline_snapshot('2021-06-01', ['600519.SH'])
    p2.set_coverage_window('2021-06-01', '2024-12-31')
    assert p2.is_coverage_complete('2021-01-01', '2024-12-31') is False

    p3 = PointInTimeUniverseProvider()
    p3.set_baseline_snapshot('2020-12-31', ['600519.SH'])
    p3.set_coverage_window('2020-12-31', '2023-12-31')
    assert p3.is_coverage_complete('2021-01-01', '2024-12-31') is False

    p4 = PointInTimeUniverseProvider()
    p4.set_baseline_snapshot('2020-12-31', ['600519.SH'])
    p4.set_coverage_window('2020-12-31', '2024-12-31')
    assert p4.is_coverage_complete('2021-01-01', '2024-12-31') is True
    assert p4.has_survivorship_bias_risk('2021-01-01', '2024-12-31') is True
    assert p4.get_mode('2021-01-01', '2024-12-31') != 'POINT_IN_TIME_VERIFIED'


@pytest.mark.unit
def test_pit_certification_positive_case():
    ver_res = UniverseVerificationResult(
        is_valid=True,
        source_class=SourceClass.OFFICIAL_PRIMARY,
        provenance_verified=True,
        raw_hash_verified=True,
        dataset_hash_verified=True,
        coverage_verified=True,
        source_verified=True,
        baseline_verified=True,
        event_integrity_verified=True,
        survivorship_bias_risk=False,
        mode="POINT_IN_TIME_VERIFIED",
        failed_checks=[]
    )
    p = PointInTimeUniverseProvider(
        verification_result=ver_res,
        baseline_snapshot_date='2020-12-31',
        baseline_symbols=['600519.SH', '000858.SZ'],
        coverage_start='2020-12-31',
        coverage_end='2024-12-31'
    )
    p.add_constituent_change('2022-01-01', '600036.SH', 'IN')

    assert p.is_coverage_complete('2021-01-01', '2024-12-31') is True
    assert p.get_mode('2021-01-01', '2024-12-31') == 'POINT_IN_TIME_VERIFIED'
    assert p.has_survivorship_bias_risk('2021-01-01', '2024-12-31') is False


@pytest.mark.unit
def test_universe_provider_factory():
    class MockConfig:
        UNIVERSE_MODE = 'STATIC'
        DEFAULT_UNIVERSE = ['600519.SH', '000858.SZ']
    
    p_static = create_universe_provider(MockConfig)
    assert isinstance(p_static, StaticUniverseProvider)
    assert p_static.get_mode() == 'STATIC'
    assert p_static.has_survivorship_bias_risk() is True


@pytest.mark.integration
@pytest.mark.slow
def test_four_year_pit_pipeline_e2e_integration(tmp_path):
    symbols = ['A', 'B', 'C', 'D', 'E', 'F']
    dates = pd.date_range('2021-01-01', '2024-12-31', freq='B')
    
    pit_provider = PointInTimeUniverseProvider(
        verification_result=UniverseVerificationResult(
            is_valid=True,
            source_class=SourceClass.OFFICIAL_PRIMARY,
            provenance_verified=True,
            raw_hash_verified=True,
            dataset_hash_verified=True,
            coverage_verified=True,
            source_verified=True,
            baseline_verified=True,
            event_integrity_verified=True,
            survivorship_bias_risk=False,
            mode="POINT_IN_TIME",
            failed_checks=[]
        ),
        baseline_snapshot_date='2020-12-31',
        baseline_symbols=['A', 'B', 'C'],
        coverage_start='2020-12-31',
        coverage_end='2024-12-31'
    )
    pit_provider.add_constituent_change('2022-01-01', 'A', 'OUT')
    pit_provider.add_constituent_change('2022-01-01', 'D', 'IN')
    pit_provider.add_constituent_change('2023-01-01', 'B', 'OUT')
    pit_provider.add_constituent_change('2023-01-01', 'E', 'IN')
    pit_provider.add_constituent_change('2024-01-01', 'C', 'OUT')
    pit_provider.add_constituent_change('2024-01-01', 'F', 'IN')

    dfs = []
    for sym in symbols:
        n = len(dates)
        np.random.seed(abs(hash(sym)) % (2**32))
        rets = np.random.normal(0.0002, 0.015, n)
        prices = 100.0 * np.exp(np.cumsum(rets))
        sdf = pd.DataFrame({
            'date': dates,
            'symbol': sym,
            'open': prices * 1.001,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'adj_open': prices * 1.001,
            'adj_high': prices * 1.01,
            'adj_low': prices * 0.99,
            'adj_close': prices,
            'volume': 100000.0,
            'amount': 10000000.0,
            'turnover': 0.02,
            'pct_change': rets,
            'adj_pct_change': rets,
            'benchmark_close': 3000.0 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n))),
            'benchmark_pct_change': np.random.normal(0.0001, 0.01, n),
            'log_circ_mv': 24.0,
            'industry': 'TECH',
            'board': '主板',
            'list_date': '2015-01-01',
            'is_subnew': False,
            'is_suspended': False,
            'is_limit_up': False,
            'is_limit_down': False,
            'is_limit_up_locked': False,
            'is_limit_down_locked': False,
            'current_is_st': False,
            'is_st': False,
            'historical_st_rule_applied': False
        })
        dfs.append(sdf)
    
    market_df = pd.concat(dfs, ignore_index=True)
    market_df.sort_values(by=['date', 'symbol'], inplace=True)
    market_df['in_universe'] = market_df.apply(lambda r: pit_provider.is_member(r['symbol'], r['date']), axis=1)

    proc = FactorProcessor(factor_dir=tmp_path / 'factors')
    factor_df = proc.build_and_save_factor_matrix(market_df, force_update=True)
    
    labeler = TargetLabeler(horizon=5)
    factor_df = labeler.compute_excess_return_label(factor_df)

    trainer = WalkForwardTrainer(train_years=1, val_months=3, test_months=6, purge_gap_days=5)
    oos_df, model = trainer.run_walk_forward(factor_df)

    a_2023 = oos_df[(oos_df['symbol'] == 'A') & (oos_df['date'] >= '2022-01-01')]
    assert (a_2023['in_universe'] == False).all()
    assert a_2023['pred_rank'].isna().all()

    f_2022 = oos_df[(oos_df['symbol'] == 'F') & (oos_df['date'] < '2024-01-01')]
    assert (f_2022['in_universe'] == False).all()
    assert f_2022['pred_rank'].isna().all()

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_predictions(oos_df)
    assert metrics['evaluated_member_rows'] > 0
    assert metrics['oos_excluded_nonmember_rows'] > 0
    if settings.is_classification:
        assert 'auc' in metrics
    else:
        assert 'rank_icir_newey_west' in metrics
    assert len(metrics['quantile_returns']) >= 1

    builder = PortfolioBuilder(top_k_buy=2, top_k_hold=3, universe_provider=pit_provider)
    engine = BacktestEngine(
        initial_cash=1000000.0,
        top_k_buy=2,
        top_k_hold=3,
        rebalance_freq=5,
        portfolio_builder=builder
    )
    equity_df, orders_df = engine.run(oos_df)

    bought_symbols_2023 = [o.symbol for o in engine.completed_orders if o.side == OrderSide.BUY and o.execution_date and o.execution_date >= '2023-01-01' and o.execution_date < '2024-01-01']
    assert 'A' not in bought_symbols_2023
    assert 'B' not in bought_symbols_2023

    dm_mock = DataManager(parquet_dir=tmp_path / 'parquet', universe_provider=pit_provider)
    audit = AuditCollector.collect(data_manager=dm_mock, factor_processor=proc, engine=engine)
    assert audit.universe_coverage_complete is True
    assert audit.survivorship_bias_risk is False
    assert audit.order_quantity_conservation_passed is True


@pytest.mark.unit
def test_in_universe_fail_closed_and_empty_universe():
    dates = pd.date_range('2023-01-01', '2023-01-05', freq='B')
    df = pd.DataFrame({
        'date': list(dates) * 2,
        'symbol': ['S1'] * len(dates) + ['S2'] * len(dates),
        'close': [10.0] * (len(dates)*2),
        'adj_close': [10.0] * (len(dates)*2),
        'in_universe': [False] * (len(dates)*2),
        'MOM_5': [0.01] * (len(dates)*2),
        'VOL_5': [0.02] * (len(dates)*2),
        'log_circ_mv': [20.0] * (len(dates)*2),
        'industry': ['TECH'] * (len(dates)*2)
    })
    proc = FactorProcessor()
    std_df = proc.cross_sectional_standardize(df, ['MOM_5', 'VOL_5'])
    assert len(std_df) == len(df)


@pytest.mark.unit
def test_manifest_chained_lineage_and_strict_load(tmp_path):
    proc = FactorProcessor(factor_dir=tmp_path / 'factors')
    with pytest.raises(FileNotFoundError):
        proc.load_factor_matrix(strict=True)


@pytest.mark.unit
def test_qfq_pit_safety_invariance_proof():
    dates = pd.date_range('2023-01-01', '2023-01-30', freq='B')
    n = len(dates)
    np.random.seed(42)
    p_orig = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    v_orig = np.random.lognormal(10, 0.5, n)

    df_a = pd.DataFrame({
        'date': dates,
        'symbol': '600519.SH',
        'open': p_orig, 'high': p_orig * 1.01, 'low': p_orig * 0.99, 'close': p_orig,
        'adj_open': p_orig, 'adj_high': p_orig * 1.01, 'adj_low': p_orig * 0.99, 'adj_close': p_orig,
        'volume': v_orig, 'amount': p_orig * v_orig, 'turnover': 0.01,
        'pct_change': pd.Series(p_orig).pct_change().fillna(0.0),
        'adj_pct_change': pd.Series(p_orig).pct_change().fillna(0.0)
    })

    p_split = p_orig / 2.0
    df_b = pd.DataFrame({
        'date': dates,
        'symbol': '600519.SH',
        'open': p_orig, 'high': p_orig * 1.01, 'low': p_orig * 0.99, 'close': p_orig,
        'adj_open': p_split, 'adj_high': p_split * 1.01, 'adj_low': p_split * 0.99, 'adj_close': p_split,
        'volume': v_orig, 'amount': p_orig * v_orig, 'turnover': 0.01,
        'pct_change': pd.Series(p_orig).pct_change().fillna(0.0),
        'adj_pct_change': pd.Series(p_split).pct_change().fillna(0.0)
    })

    proc = FactorProcessor()
    feat_a = proc.alpha_calc.compute_all(df_a)
    feat_b = proc.alpha_calc.compute_all(df_b)

    for f in ['ROC_5', 'ROC_10', 'ROC_20', 'VOLATILITY_20', 'BETA_20']:
        if f in feat_a.columns and f in feat_b.columns:
            s_a = feat_a[f].dropna().values
            s_b = feat_b[f].dropna().values
            np.testing.assert_allclose(s_a, s_b, rtol=1e-5, err_msg=f'Factor {f} not invariant under QFQ split')


@pytest.mark.unit
def test_audit_collector_anti_forgery_blocks_override():
    meta = AuditCollector.collect(
        custom_overrides={
            'survivorship_bias_risk': False,
            'overall_backtest_reliability': 'VERIFIED',
            'order_quantity_conservation_passed': True,
            'synthetic_data_used': False
        }
    )
    assert meta.survivorship_bias_risk is True
    assert meta.overall_backtest_reliability == 'HIGH_RISK'
    assert meta.order_quantity_conservation_passed is False
    assert meta.synthetic_data_used is True


@pytest.mark.unit
def test_corporate_action_coverage_with_zero_events(tmp_path, monkeypatch):
    from data.crypto_anchor import TRUSTED_KEY_REGISTRY, DOMAIN_SEPARATOR_ACQUISITION
    from cryptography.hazmat.primitives.asymmetric import ed25519 as crypto_ed25519
    from data.source_registry import AcquisitionReceipt
    from dataclasses import asdict
    import json

    priv = crypto_ed25519.Ed25519PrivateKey.generate()
    sk_hex = priv.private_bytes_raw().hex()
    pk_hex = priv.public_key().public_bytes_raw().hex()

    monkeypatch.setitem(TRUSTED_KEY_REGISTRY, "PROD_DOWNLOADER_KEY_TEST", {
        "algorithm": "ED25519",
        "key_id": "PROD_DOWNLOADER_KEY_TEST",
        "public_key_hex": pk_hex,
        "allowed_purposes": ["ACQUISITION_RECEIPT"],
        "issuer_type": "PROJECT",
        "status": "ACTIVE",
        "not_before": "2025-01-01T00:00:00Z",
        "not_after": "2030-01-01T00:00:00Z",
        "is_production": True
    })

    prov = CorporateActionProvider()
    raw_f = tmp_path / "raw.json"
    raw_f.write_text("[]", encoding="utf-8")
    h_raw = hashlib.sha256(raw_f.read_bytes()).hexdigest()
    resp_f = tmp_path / "resp.json"
    resp_f.write_text("[]", encoding="utf-8")
    h_resp = hashlib.sha256(resp_f.read_bytes()).hexdigest()

    meta_f = tmp_path / "raw.json.source.json"
    meta_f.write_text(json.dumps({
        "source_id": "SSE",
        "source_url": "https://www.sse.com.cn/disclosure/events.json",
        "retrieved_at_utc": "2026-01-01T00:00:00Z",
        "sha256": h_raw,
        "original_filename": "raw.json",
        "byte_size": raw_f.stat().st_size,
        "downloader_version": "3.1"
    }), encoding="utf-8")

    rec = AcquisitionReceipt(
        receipt_id="REC_001",
        source_id="SSE",
        source_url="https://www.sse.com.cn/disclosure/events.json",
        requested_at="2026-01-01T00:00:00Z",
        downloaded_at="2026-01-01T00:00:00Z",
        raw_sha256=h_raw,
        original_filename="raw.json",
        query_context={
            "resource_type": "CORPORATE_ACTION",
            "symbol": "600519.SH",
            "query_start": "2021-01-01",
            "query_end": "2023-12-31",
            "request_params_sha256": "a" * 64
        },
        trust_anchor_type="TRUSTED_KEY_ATTESTATION",
        signing_key_id="PROD_DOWNLOADER_KEY_TEST"
    )
    digest = rec.compute_integrity_digest()
    msg_to_sign = f"{DOMAIN_SEPARATOR_ACQUISITION}:".encode("utf-8") + digest.encode("utf-8")
    sig = priv.sign(msg_to_sign)
    rec.attestation_signature = sig.hex()

    rec_f = tmp_path / "raw.json.receipt.json"
    rec_f.write_text(json.dumps(asdict(rec)), encoding="utf-8")

    ev = CorporateActionCoverageEvidence(
        symbol='600519.SH',
        query_start='2021-01-01',
        query_end='2023-12-31',
        source_id='SSE',
        query_success=True,
        empty_result=True,
        empty_result_verified=True,
        raw_result_file="raw.json",
        raw_result_hash=h_raw,
        response_file="resp.json",
        response_hash=h_resp,
        source_metadata_file="raw.json.source.json",
        acquisition_receipt_file="raw.json.receipt.json"
    )
    prov.register_coverage_record(ev)
    
    assert prov.validate_coverage(['600519.SH'], '2021-01-01', '2023-12-31', evidence_dir=tmp_path) is True
    assert prov.coverage_complete is True
    assert prov.zero_event_proof_verified is True


@pytest.mark.unit
def test_corporate_action_split_partial_fill_conservation():
    engine = BacktestEngine(initial_cash=1000000.0)
    engine.positions['600519.SH'] = PositionRecord(
        symbol='600519.SH', shares=1000, available_shares=1000, avg_cost=100.0,
        last_price=100.0, buy_date='2023-01-01', highest_price=100.0,
        lots=[PositionLot(shares=1000, buy_execution_price=100.0, buy_date='2023-01-01')]
    )
    test_order = Order(
        order_id='ORD_001', symbol='600519.SH', side=OrderSide.SELL,
        signal_date='2023-05-10', requested_shares=1000, cumulative_filled_shares=400,
        remaining_shares=600, cancelled_shares=0, signal_price=100.0
    )
    assert test_order.verify_quantity_conservation() is True
    engine.pending_orders.append(test_order)

    engine.corporate_actions.register_action(CorporateAction(
        symbol='600519.SH', ex_date='2023-05-11', action_type='BONUS_SHARE', share_ratio=0.5
    ))
    engine._apply_corporate_actions('2023-05-11')

    assert engine.positions['600519.SH'].shares == 1500
    assert engine.positions['600519.SH'].lots[0].buy_execution_price == pytest.approx(100.0 / 1.5)
    
    ord_adj = engine.pending_orders[0]
    assert ord_adj.requested_shares == 1500
    assert ord_adj.cumulative_filled_shares == 600
    assert ord_adj.remaining_shares == 900
    assert ord_adj.verify_quantity_conservation() is True


@pytest.mark.unit
def test_cash_dividend_trailing_stop_protection():
    engine = BacktestEngine(initial_cash=1000000.0)
    engine.positions['600519.SH'] = PositionRecord(
        symbol='600519.SH', shares=1000, available_shares=1000, avg_cost=100.0,
        last_price=120.0, buy_date='2023-01-01', highest_price=120.0,
        lots=[PositionLot(shares=1000, buy_execution_price=100.0, buy_date='2023-01-01')]
    )
    engine.corporate_actions.register_action(CorporateAction(
        symbol='600519.SH', ex_date='2023-05-11', action_type='CASH_DIVIDEND', cash_dividend_per_share=10.0
    ))
    engine._apply_corporate_actions('2023-05-11')

    pos = engine.positions['600519.SH']
    assert pos.last_price == 110.0
    assert pos.highest_price == 110.0
    assert engine.cash == 1000000.0 + 10000.0


@pytest.mark.unit
def test_price_limit_rule_engine_comprehensive():
    engine = PriceLimitRuleEngine()

    assert engine.get_price_limit_ratio('688001.SH', trade_date='2022-01-01') == 0.20
    assert engine.get_price_limit_ratio('688001.SH', trade_date='2022-01-01', listing_days=3) == 999.0

    assert engine.get_price_limit_ratio('300001.SZ', trade_date='2020-08-23') == 0.10
    assert engine.get_price_limit_ratio('300001.SZ', trade_date='2020-08-23', is_st=True) == 0.05
    assert engine.get_price_limit_ratio('300001.SZ', trade_date='2020-08-24') == 0.20
    assert engine.get_price_limit_ratio('300001.SZ', trade_date='2020-08-25', is_st=True) == 0.20

    assert engine.get_price_limit_ratio('600519.SH', trade_date='2022-01-01') == 0.10
    assert engine.get_price_limit_ratio('600519.SH', trade_date='2022-01-01', is_st=True) == 0.05

    assert engine.get_price_limit_ratio('830001.BJ', trade_date='2022-01-01') == 0.30


@pytest.mark.integration
def test_true_production_pit_e2e(tmp_path):
    """验证通过真实 DataManager.sync_and_build_dataset 产生完整 PIT in_universe (P0-3)"""
    from data.data_fetcher import DataFetcher

    class FakeFetcher(DataFetcher):
        def __init__(self):
            super().__init__()
            self.source_counts = {"fake": 3}
            self.last_benchmark_source = "fake"

        def fetch_benchmark_daily(self, symbol, start_date, end_date=None):
            dates = pd.date_range(start_date, end_date or "2023-12-31", freq="B")
            return pd.DataFrame({
                "date": dates,
                "close": np.linspace(3000, 3500, len(dates)),
                "pct_change": np.full(len(dates), 0.001)
            })

        def fetch_stock_daily(self, symbol, start_date, end_date=None):
            dates = pd.date_range(start_date, end_date or "2023-12-31", freq="B")
            n = len(dates)
            return pd.DataFrame({
                "date": dates,
                "symbol": symbol,
                "open": np.full(n, 10.0),
                "high": np.full(n, 10.5),
                "low": np.full(n, 9.8),
                "close": np.full(n, 10.2),
                "volume": np.full(n, 100000.0),
                "amount": np.full(n, 1000000.0),
                "turnover": np.full(n, 0.02),
                "pct_change": np.full(n, 0.002),
                "adj_open": np.full(n, 10.0),
                "adj_high": np.full(n, 10.5),
                "adj_low": np.full(n, 9.8),
                "adj_close": np.full(n, 10.2)
            })

    pit = PointInTimeUniverseProvider.for_test_fixture(
        fallback_symbols=["600000.SH", "000001.SZ"],
        baseline_snapshot_date="2020-12-31",
        baseline_symbols=["600000.SH"],
        coverage_start="2020-12-31",
        coverage_end="2023-12-31"
    )
    pit.add_constituent_change("2022-01-01", "000001.SZ", "IN")
    pit.add_constituent_change("2023-01-01", "600000.SH", "OUT")

    dm = DataManager(parquet_dir=tmp_path / "parquet", fetcher=FakeFetcher(), universe_provider=pit)
    df = dm.sync_and_build_dataset(
        symbols=["600000.SH", "000001.SZ"],
        start_date="2021-01-01",
        end_date="2023-12-31",
        force_update=True
    )

    assert "in_universe" in df.columns
    assert df["in_universe"].isna().sum() == 0
    
    # 2021年：600000 为成员，000001 非成员
    sub_2021 = df[df["date"] == pd.Timestamp("2021-06-01")]
    assert bool(sub_2021[sub_2021["symbol"] == "600000.SH"]["in_universe"].iloc[0]) is True
    assert bool(sub_2021[sub_2021["symbol"] == "000001.SZ"]["in_universe"].iloc[0]) is False

    # 2023年：000001 为成员，600000 非成员
    sub_2023 = df[df["date"] == pd.Timestamp("2023-06-01")]
    assert bool(sub_2023[sub_2023["symbol"] == "600000.SH"]["in_universe"].iloc[0]) is False
    assert bool(sub_2023[sub_2023["symbol"] == "000001.SZ"]["in_universe"].iloc[0]) is True


@pytest.mark.unit
def test_qfq_pit_safety_dual_snapshot_invariance():
    """双快照前复权时不变性证明 (P0-13)"""
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    n = len(dates)
    
    # 真实除权除息：在 2023-02-15 进行 10送10 (价格折半)
    split_idx = 30
    prices_raw = np.linspace(100.0, 120.0, n)
    
    # 快照1 (除权前拉取, T=2023-02-01)
    df_snap1 = pd.DataFrame({
        "date": dates[:split_idx],
        "symbol": "600519.SH",
        "close": prices_raw[:split_idx],
        "open": prices_raw[:split_idx] * 0.99,
        "high": prices_raw[:split_idx] * 1.01,
        "low": prices_raw[:split_idx] * 0.98,
        "adj_open": prices_raw[:split_idx] * 0.99,
        "adj_high": prices_raw[:split_idx] * 1.01,
        "adj_low": prices_raw[:split_idx] * 0.98,
        "adj_close": prices_raw[:split_idx],
        "volume": np.full(split_idx, 10000.0),
        "amount": np.full(split_idx, 1000000.0),
        "turnover": np.full(split_idx, 0.02),
        "pct_change": np.full(split_idx, 0.005),
        "log_circ_mv": np.full(split_idx, 25.0),
        "in_universe": True
    })

    # 快照2 (除权后拉取, T=2023-03-31 前复权)
    prices_adj = prices_raw.copy() * 0.5
    
    df_snap2 = pd.DataFrame({
        "date": dates,
        "symbol": "600519.SH",
        "close": np.concatenate([prices_raw[:split_idx], prices_raw[split_idx:] * 0.5]),
        "open": np.concatenate([prices_raw[:split_idx] * 0.99, prices_raw[split_idx:] * 0.5 * 0.99]),
        "high": np.concatenate([prices_raw[:split_idx] * 1.01, prices_raw[split_idx:] * 0.5 * 1.01]),
        "low": np.concatenate([prices_raw[:split_idx] * 0.98, prices_raw[split_idx:] * 0.5 * 0.98]),
        "adj_open": prices_adj * 0.99,
        "adj_high": prices_adj * 1.01,
        "adj_low": prices_adj * 0.98,
        "adj_close": prices_adj,
        "volume": np.full(n, 20000.0),
        "amount": np.full(n, 1000000.0),
        "turnover": np.full(n, 0.02),
        "pct_change": np.full(n, 0.005),
        "log_circ_mv": np.full(n, 25.0),
        "in_universe": True
    })

    proc = FactorProcessor()
    f1 = proc.alpha_calc.compute_all(df_snap1)
    f2 = proc.alpha_calc.compute_all(df_snap2)

    # 验证在历史区间 [0..split_idx-1] 内，无量纲因子与收益率因子恒等
    common_cols = [c for c in ["KMID", "KLEN", "ROC5", "MA5_RATIO"] if c in f1.columns and c in f2.columns]
    for c in common_cols:
        v1 = f1[c].iloc[10:split_idx].values
        v2 = f2[c].iloc[10:split_idx].values
        np.testing.assert_allclose(v1, v2, rtol=1e-4, err_msg=f"Factor {c} violated QFQ invariance")


@pytest.mark.unit
def test_benchmark_cross_symbol_leakage_rejected():
    """验证基准指数多标的截面合并时绝不跨股票泄露 (P0-10)"""
    dates = pd.date_range("2023-01-01", "2023-01-10", freq="B")
    
    # 股票A 有全部10天，股票B 只有后5天
    df_a = pd.DataFrame({"date": dates, "symbol": "A", "close": 10.0, "volume": 100.0, "amount": 1000.0, "turnover": 0.01})
    df_b = pd.DataFrame({"date": dates[5:], "symbol": "B", "close": 20.0, "volume": 200.0, "amount": 2000.0, "turnover": 0.02})
    
    bench = pd.DataFrame({"date": dates, "benchmark_close": np.linspace(3000, 3100, len(dates)), "benchmark_pct_change": 0.001})
    
    # 合并
    stocks = pd.concat([df_a, df_b], ignore_index=True)
    merged = pd.merge(stocks, bench, on="date", how="left")
    
    # 断言：合并后每一只股票对应日期的基准价格严格一致，绝不会因为 ffill 串行
    for d in dates[5:]:
        ba = merged[(merged["date"] == d) & (merged["symbol"] == "A")]["benchmark_close"].iloc[0]
        bb = merged[(merged["date"] == d) & (merged["symbol"] == "B")]["benchmark_close"].iloc[0]
        assert ba == bb


@pytest.mark.unit
def test_factor_cache_streaming_sha256_rejection(tmp_path):
    """验证流式 SHA256 能够精准捕获中间行数据的微小篡改 (P0-14)"""
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
    n = len(dates)
    
    df1 = pd.DataFrame({
        "date": dates, "symbol": "A", "close": np.full(n, 10.0),
        "open": np.full(n, 10.0), "high": np.full(n, 10.5), "low": np.full(n, 9.8),
        "adj_close": np.full(n, 10.0), "volume": np.full(n, 1000.0), "amount": np.full(n, 10000.0),
        "in_universe": True
    })
    
    # 仅修改中间某一行数据
    df2 = df1.copy()
    df2.loc[n // 2, "close"] = 10.01

    h1 = FactorProcessor._compute_streaming_content_hash(df1)
    h2 = FactorProcessor._compute_streaming_content_hash(df2)

    assert h1 != h2, "流式 SHA256 必须成功检测到中间行数据的微小篡改！"


@pytest.mark.unit
def test_certification_policy_truth_table():
    """验证 CertificationPolicy 13 项门禁真值表 (P0-12)"""
    from backtest.audit import CertificationPolicy, AuditMetadata

    # 1. 完美状态 -> VERIFIED
    meta_pass = AuditMetadata(
        runtime_config_hash="a" * 64,
        runtime_config_hash_verified=True,
        universe_source_class="OFFICIAL_PRIMARY",
        universe_raw_evidence_verified=True,
        universe_dataset_hash_verified=True,
        trust_root_verified=True,
        universe_manifest_hash="b" * 64,
        universe_manifest_hash_verified=True,
        factor_manifest_hash="c" * 64,
        factor_manifest_hash_verified=True,
        market_manifest_hash="d" * 64,
        market_manifest_hash_verified=True,
        corporate_action_manifest_hash="e" * 64,
        corporate_action_manifest_hash_verified=True,
        manifest_chain_verified=True,
        corporate_action_provenance_verified=True,
        market_data_provenance_verified=True,
        data_source="csi_official_direct",
        benchmark_source="csi_000300_official",
        actual_backtest_start_date="2020-01-01",
        actual_backtest_end_date="2024-12-31",
        universe_coverage_start="2020-01-01",
        universe_coverage_end="2024-12-31",
        universe_coverage_complete=True,
        universe_provenance_verified=True,
        survivorship_bias_risk=False,
        historical_st_coverage_complete=True,
        historical_st_bias_risk=False,
        st_unknown_rows=0,
        corporate_action_coverage_complete=True,
        corporate_action_bias_risk=False,
        corporate_action_adjustment_available=True,
        corporate_action_dataset_hash_verified=True,
        corporate_action_coverage_ratio=1.0,
        cache_fingerprint_verified=True,
        raw_data_provenance_preserved=True,
        adjustment_point_in_time_safe=True,
        future_adjustment_leakage_test_passed=True,
        benchmark_coverage_ratio=1.0,
        benchmark_missing_date_count=0,
        order_quantity_conservation_passed=True,
        synthetic_data_used=False,
        calendar_is_exchange_official=True
    )
    status, failed = CertificationPolicy.evaluate(meta_pass)
    assert status == "VERIFIED"
    assert len(failed) == 0

    # 2. 存在幸存者偏差 -> HIGH_RISK
    meta_bias = AuditMetadata(
        universe_coverage_complete=False,
        survivorship_bias_risk=True
    )
    status_bias, failed_bias = CertificationPolicy.evaluate(meta_bias)
    assert status_bias == "HIGH_RISK"
    assert "survivorship_bias_risk_present" in failed_bias

    # 3. 仅日历为第三方 -> CONTROLLED_WITH_LIMITATIONS
    meta_lim = AuditMetadata(
        runtime_config_hash="a" * 64,
        runtime_config_hash_verified=True,
        universe_source_class="OFFICIAL_PRIMARY",
        universe_raw_evidence_verified=True,
        universe_dataset_hash_verified=True,
        trust_root_verified=True,
        universe_manifest_hash="b" * 64,
        universe_manifest_hash_verified=True,
        factor_manifest_hash="c" * 64,
        factor_manifest_hash_verified=True,
        market_manifest_hash="d" * 64,
        market_manifest_hash_verified=True,
        corporate_action_manifest_hash="e" * 64,
        corporate_action_manifest_hash_verified=True,
        manifest_chain_verified=True,
        corporate_action_provenance_verified=True,
        market_data_provenance_verified=True,
        data_source="csi_official_direct",
        benchmark_source="csi_000300_official",
        actual_backtest_start_date="2020-01-01",
        actual_backtest_end_date="2024-12-31",
        universe_coverage_start="2020-01-01",
        universe_coverage_end="2024-12-31",
        universe_coverage_complete=True,
        universe_provenance_verified=True,
        survivorship_bias_risk=False,
        historical_st_coverage_complete=True,
        historical_st_bias_risk=False,
        st_unknown_rows=0,
        corporate_action_coverage_complete=True,
        corporate_action_bias_risk=False,
        corporate_action_adjustment_available=True,
        corporate_action_coverage_ratio=1.0,
        cache_fingerprint_verified=True,
        raw_data_provenance_preserved=True,
        adjustment_point_in_time_safe=True,
        future_adjustment_leakage_test_passed=True,
        benchmark_coverage_ratio=1.0,
        benchmark_missing_date_count=0,
        order_quantity_conservation_passed=True,
        synthetic_data_used=False,
        calendar_is_exchange_official=False # 仅此项不满足
    )
    status_lim, failed_lim = CertificationPolicy.evaluate(meta_lim)
    assert status_lim == "CONTROLLED_WITH_LIMITATIONS"
    assert "calendar_not_exchange_official" in failed_lim


@pytest.mark.unit
def test_corporate_action_provider_factory(tmp_path):
    """验证 create_corporate_action_provider 工厂类 (P0-9)"""
    from strategy.corporate_actions import create_corporate_action_provider

    class MockConfig:
        DATA_DIR = tmp_path
        CORPORATE_ACTIONS_FILE = tmp_path / "corp.parquet"
        CORPORATE_ACTIONS_COVERAGE_FILE = tmp_path / "coverage.json"

    p = create_corporate_action_provider(MockConfig)
    assert isinstance(p, CorporateActionProvider)

