"""
Unit and integration tests for Stage 3:
- Cryptographic structured evidence objects in ModelRegistry.promote
- Rejection of boolean evidence flags
- Strict Fail-Closed Live Trading Hard Gate (LIVE_TRADING_READY = False)
- MiniQMT broker connect/send_order fail-closed enforcement
- run_trader CLI fail-closed enforcement
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import settings
from models.registry import (
    ModelRegistry,
    ModelState,
    PromotionError,
    ProspectiveEvidence,
    PaperTradingEvidence,
)
from execution.miniqmt_broker import MiniQMTBroker
from execution.broker_base import OrderSide, OrderType


@pytest.fixture()
def registry(tmp_path):
    return ModelRegistry(root=tmp_path / "registry")


@pytest.fixture()
def dummy_artifact(tmp_path) -> Path:
    from models.lightgbm_model import LightGBMQuantModel
    model = LightGBMQuantModel(task_type="classification")
    X = pd.DataFrame({"f1": np.random.rand(60), "f2": np.random.rand(60)})
    y = pd.Series(np.random.randint(0, 2, 60))
    model.fit(X, y)
    path = tmp_path / "research_model.pkl"
    model.save(filepath=path)
    return path


def _register(registry, artifact, **kw):
    return registry.register_research_artifact(
        artifact_path=artifact,
        model_type="lightgbm",
        task_type="classification",
        metrics=kw.pop("metrics", {"auc": 0.53, "rank_ic": 0.05}),
        dataset_sha256=kw.pop("dataset_sha256", "a" * 64),
        feature_count=2,
        feature_schema_hash="dummy_schema_hash_1234",
        **kw,
    )


def test_live_trading_ready_default_is_false():
    """Verify default live trading status is LOCKED and flag is False."""
    assert settings.LIVE_TRADING_READY is False
    assert settings.LIVE_TRADING_STATUS == "LOCKED"


def test_promote_rejects_boolean_evidence(registry, dummy_artifact):
    """Verify promote rejects boolean True/False for prospective_validation and paper_trading."""
    mid = _register(registry, dummy_artifact)
    registry.promote(mid, ModelState.CANDIDATE, approver="researcher")
    registry.promote(mid, ModelState.APPROVED, approver="linjun", evidence={"certification_ref": "c1.json"})

    # Test 1: boolean prospective_validation
    with pytest.raises(PromotionError) as exc_info:
        registry.promote(
            mid,
            ModelState.PRODUCTION,
            approver="linjun",
            evidence={
                "certification_ref": "c1.json",
                "prospective_validation": True,
                "paper_trading": {"ref": "p_001", "ledger_hash": "sha_1", "trade_count": 10},
            },
        )
    assert "布尔值" in str(exc_info.value) or "boolean" in str(exc_info.value).lower()

    # Test 2: boolean paper_trading
    with pytest.raises(PromotionError) as exc_info:
        registry.promote(
            mid,
            ModelState.PRODUCTION,
            approver="linjun",
            evidence={
                "certification_ref": "c1.json",
                "prospective_validation": {"ref": "v_001", "dataset_sha256": "sha_2", "metrics": {}},
                "paper_trading": True,
            },
        )
    assert "布尔值" in str(exc_info.value) or "boolean" in str(exc_info.value).lower()

    # Test 3: empty dict for prospective_validation
    with pytest.raises(PromotionError):
        registry.promote(
            mid,
            ModelState.PRODUCTION,
            approver="linjun",
            evidence={
                "certification_ref": "c1.json",
                "prospective_validation": {},
                "paper_trading": {"ref": "p_001", "ledger_hash": "sha_1", "trade_count": 10},
            },
        )

    # Test 4: dict missing key fields for prospective_validation
    with pytest.raises(PromotionError) as exc_info:
        registry.promote(
            mid,
            ModelState.PRODUCTION,
            approver="linjun",
            evidence={
                "certification_ref": "c1.json",
                "prospective_validation": {"unrelated_field": 123},
                "paper_trading": {"ref": "p_001", "ledger_hash": "sha_1", "trade_count": 10},
            },
        )
    assert "有效字段" in str(exc_info.value)

    # Test 5: empty dict for paper_trading
    with pytest.raises(PromotionError):
        registry.promote(
            mid,
            ModelState.PRODUCTION,
            approver="linjun",
            evidence={
                "certification_ref": "c1.json",
                "prospective_validation": {"ref": "v_001", "dataset_sha256": "sha_2", "metrics": {}},
                "paper_trading": {},
            },
        )

    # Test 6: dict missing key fields for paper_trading
    with pytest.raises(PromotionError) as exc_info:
        registry.promote(
            mid,
            ModelState.PRODUCTION,
            approver="linjun",
            evidence={
                "certification_ref": "c1.json",
                "prospective_validation": {"ref": "v_001", "dataset_sha256": "sha_2", "metrics": {}},
                "paper_trading": {"unrelated_field": 123},
            },
        )
    assert "有效字段" in str(exc_info.value)


def test_promote_succeeds_with_valid_structured_evidence(registry, dummy_artifact, tmp_path, monkeypatch):
    """Verify promote succeeds when valid structured evidence objects or dataclasses are provided."""
    from data.crypto_anchor import TRUSTED_KEY_REGISTRY, generate_keypair
    from models.registry import EvidenceArtifact
    from datetime import datetime
    sk, pk = generate_keypair()
    key_id = "STAGE3_TEST_SIGNER_KEY"
    entry = {
        "algorithm": "ED25519",
        "key_id": key_id,
        "public_key_hex": pk.hex(),
        "allowed_purposes": ["MODEL_PROMOTION", "RUNTIME_ATTESTATION"],
        "status": "ACTIVE",
        "is_production": False,
    }
    monkeypatch.setitem(TRUSTED_KEY_REGISTRY, key_id, entry)

    mid = _register(registry, dummy_artifact)
    rec = registry.get(mid)
    registry.promote(mid, ModelState.CANDIDATE, approver="researcher")
    registry.promote(mid, ModelState.APPROVED, approver="linjun", evidence={"certification_ref": "c1.json"})

    p1 = tmp_path / "pv_ev.json"
    prospective_evidence = EvidenceArtifact(
        evidence_type="PROSPECTIVE_VALIDATION",
        artifact_path=str(p1.resolve()),
        artifact_sha256="",
        schema_version="evidence_v1",
        model_id=mid,
        dataset_sha256=rec.dataset_sha256,
        feature_schema_hash=rec.feature_schema_hash,
        created_at=datetime.now().isoformat(),
        observation_start="2026-07-27",
        observation_end="2026-08-24",
        observed_trading_days=21,
        status="MATURE",
        approver="linjun",
        signer_key_id=key_id,
        signature="",
        metrics={"rank_ic": 0.082, "annualized_ic": 0.082},
    )
    prospective_evidence.sign(private_key_hex=sk.hex())
    prospective_evidence.save(p1)

    p2 = tmp_path / "pt_ev.json"
    paper_evidence = EvidenceArtifact(
        evidence_type="PAPER_TRADING",
        artifact_path=str(p2.resolve()),
        artifact_sha256="",
        schema_version="evidence_v1",
        model_id=mid,
        dataset_sha256=rec.dataset_sha256,
        feature_schema_hash=rec.feature_schema_hash,
        created_at=datetime.now().isoformat(),
        observation_start="2026-07-27",
        observation_end="2026-08-24",
        observed_trading_days=21,
        status="MATURE",
        approver="linjun",
        signer_key_id=key_id,
        signature="",
        metrics={"sharpe": 2.1, "max_drawdown": 0.028},
    )
    paper_evidence.sign(private_key_hex=sk.hex())
    paper_evidence.save(p2)

    promoted_rec = registry.promote(
        mid,
        ModelState.PRODUCTION,
        approver="linjun",
        evidence={
            "certification_ref": "c1.json",
            "prospective_validation": prospective_evidence,
            "paper_trading": paper_evidence,
        },
        note="Stage 3 verification test pass",
    )

    assert promoted_rec.state == ModelState.PRODUCTION
    last_promo = promoted_rec.promotion_history[-1]
    ev = last_promo["evidence"]
    assert isinstance(ev["prospective_validation"], dict)
    assert ev["prospective_validation"]["status"] == "MATURE"
    assert ev["prospective_validation"]["metrics"]["rank_ic"] == 0.082
    assert isinstance(ev["paper_trading"], dict)
    assert ev["paper_trading"]["status"] == "MATURE"
    assert ev["paper_trading"]["observed_trading_days"] == 21


def test_miniqmt_broker_fail_closed_connect():
    """Verify MiniQMTBroker.connect raises RuntimeError when LIVE_TRADING_READY is False."""
    broker = MiniQMTBroker()
    with pytest.raises(RuntimeError, match="LIVE_TRADING_READY is False"):
        broker.connect()


def test_miniqmt_broker_fail_closed_send_order():
    """Verify MiniQMTBroker.send_order raises RuntimeError when LIVE_TRADING_READY is False."""
    broker = MiniQMTBroker()
    # Even if mocked as connected
    broker.is_connected = True
    broker.trader = MagicMock()
    with pytest.raises(RuntimeError, match="LIVE_TRADING_READY=False"):
        broker.send_order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            shares=100,
            price=12.5,
            order_type=OrderType.LIMIT,
        )


def test_run_trader_cli_blocks_live_trading(monkeypatch):
    """Verify running run_trader CLI with --broker miniqmt raises RuntimeError."""
    from execution.run_trader import run_trader_cli

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_trader.py", "--broker", "miniqmt"],
    )
    with pytest.raises(RuntimeError, match="LIVE_TRADING_READY=False"):
        run_trader_cli()


def test_run_trader_no_walk_forward_trainer_in_inference():
    """Verify run_trader does not call WalkForwardTrainer."""
    import execution.run_trader as rt
    src_path = Path(rt.__file__)
    src_content = src_path.read_text(encoding="utf-8")
    assert "WalkForwardTrainer" not in src_content
    assert "BatchInference" in src_content
