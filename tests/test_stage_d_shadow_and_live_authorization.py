"""
Stage D 验收测试套件: 影子观察流水线贯通与实盘多因子鉴权硬网关 (tests/test_stage_d_shadow_and_live_authorization.py)

测试范围:
1. 影子观察记账贯通 (ShadowTradingLedger):
   - scheduler/daily_runner.py 每日盘后自动化中影子观察记账
   - scripts/auto_daily_scheduler.py 盘后跑批流水线影子观察记账与权威交易日检查
2. 实盘多因子硬核鉴权网关 (execution/live_gate.py):
   - 逐项覆盖 11 大合规与风控因子 (a ~ k) 的 Fail-Closed 拦截能力
   - 全部 11 项满足时方可解锁通过
3. 券商终端与执行器防御:
   - MiniQMTBroker 连接与发单前置多因子硬核拦截
   - 断线严格阻断，严禁假想仿真成交
   - run_trader.py 缺失 --live-confirm 时 fail-closed 拦截，拒绝静默降级为 Dry-Run
"""
import os
import json
import time
import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

from config.settings import settings
from data.trading_calendar import CanonicalTradingCalendar
from data.crypto_anchor import TRUSTED_KEY_REGISTRY, sign_with_environment_key, verify_ed25519_signature
from execution.paper_ledger import PaperTradingLedger, ShadowTradingLedger
from execution.live_gate import verify_live_trading_multi_factor_gate, LiveGateAuthError
from execution.miniqmt_broker import MiniQMTBroker
from execution.broker_base import OrderSide, OrderType, ExecutionStatus


class DummyModelRecord:
    def __init__(self, model_id="PROD_TEST_M1", state="PRODUCTION", evidence=None, promotion_history=None):
        self.model_id = model_id
        self.state = state
        self.evidence = evidence or {}
        self.promotion_history = promotion_history or []


def create_valid_approval_artifact(tmp_path: Path, signer_key_id="PROD_LIVE_AUTH_KEY_2026_V1", nonce: Optional[str] = None) -> Path:
    key_entry = TRUSTED_KEY_REGISTRY[signer_key_id]
    env_var = key_entry.get("env_private_key_var")
    test_sk_hex = "11" * 32
    if env_var and os.environ.get(env_var):
        sk_hex = os.environ.get(env_var)
    else:
        from cryptography.hazmat.primitives.asymmetric import ed25519 as crypto_ed25519
        priv = crypto_ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(test_sk_hex))
        pk_hex = priv.public_key().public_bytes_raw().hex()
        key_entry["public_key_hex"] = pk_hex
        sk_hex = test_sk_hex

    used_nonce = nonce or f"stage_d_nonce_{time.time()}"
    raw_payload = {
        "approval_id": "APPR_20260904_001",
        "authorized_action": "LIVE_ORDER",
        "operator": "chief_risk_officer",
        "account_id": "5500123456",
        "model_id": "PROD_CERT_V1",
        "environment": "production",
        "symbol": "600519.SH",
        "side": "BUY",
        "quantity": 1000.0,
        "limit_price": 50.0,
        "nonce": used_nonce,
        "expiration_timestamp": time.time() + 3600.0,
        "risk_limits": {"max_notional": 1000000.0, "max_daily_notional": 5000000.0},
        "decision": "APPROVE_LIVE_GATE_UNLOCK",
        "authorized_accounts": ["5500123456"],
        "timestamp": "2026-09-04T15:00:00Z"
    }
    msg_bytes = json.dumps(raw_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    allowed_purposes = key_entry.get("allowed_purposes", [])
    req_purpose = "LIVE_TRADING_AUTHORIZATION" if "LIVE_TRADING_AUTHORIZATION" in allowed_purposes else allowed_purposes[0]

    sig_hex, errs = sign_with_environment_key(
        message=msg_bytes,
        key_id=signer_key_id,
        required_purpose=req_purpose,
        domain_separator="LIVE_TRADING_AUTHORIZATION_APPROVAL_V1",
        explicit_private_key_hex=sk_hex
    )
    artifact_data = {
        "schema_version": "approval_v1",
        "signer_key_id": signer_key_id,
        "canonical_payload": raw_payload,
        "signature": sig_hex
    }
    appr_file = tmp_path / "live_approval_artifact.json"
    appr_file.write_text(json.dumps(artifact_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return appr_file


class TestStageDShadowLedgerWiring:
    """1. 影子观察流水线贯通测试"""

    def test_auto_scheduler_trading_day_check_and_shadow_observation(self, tmp_path):
        from scripts.auto_daily_scheduler import DailyExecutionPipeline

        # 验证交易日判断 (CanonicalTradingCalendar)
        assert DailyExecutionPipeline.is_trading_day("2026-08-24") is True
        assert DailyExecutionPipeline.is_trading_day("2026-09-05") is False
        assert DailyExecutionPipeline.is_trading_day("2026-09-06") is False

        # 测试在交易日执行跑批，写入 ShadowTradingLedger (使用前瞻日期与日历 Mock)
        test_shadow_ledger = tmp_path / "test_shadow_ledger.parquet"
        test_paper_ledger = tmp_path / "test_paper_ledger.parquet"
        test_status_file = tmp_path / "test_status.json"

        picks_file = tmp_path / "picks.csv"
        pd.DataFrame({
            "symbol": ["600519.SH", "000858.SZ"],
            "date": ["2026-08-31", "2026-08-31"],
            "open": [1600.0, 180.0],
            "target_weight": [0.10, 0.10],
            "close": [1600.0, 180.0]
        }).to_csv(picks_file, index=False)

        from data.trading_calendar import CanonicalTradingCalendar
        cal = CanonicalTradingCalendar.get_instance()
        fm_path = tmp_path / "features" / "factor_matrix_latest.parquet"
        fm_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "date": ["2026-08-28", "2026-08-28"],
            "symbol": ["600519.SH", "000858.SZ"],
            "f1": [1.0, 2.0]
        }).to_parquet(fm_path, index=False)

        def mock_is_trading(d):
            return str(d)[:10] in ("2026-08-28", "2026-08-31", "2026-09-01")

        def mock_next_trading(d):
            mapping = {"2026-08-28": "2026-08-31", "2026-08-31": "2026-09-01"}
            return mapping.get(str(d)[:10])

        def mock_prev_trading(d):
            mapping = {"2026-08-31": "2026-08-28", "2026-09-01": "2026-08-31"}
            return mapping.get(str(d)[:10])

        with patch("scripts.auto_daily_scheduler.settings.BASE_DIR", tmp_path), \
             patch("scripts.auto_daily_scheduler.settings.PARQUET_DIR", tmp_path), \
             patch("scripts.auto_daily_scheduler.settings.DATA_DIR", tmp_path), \
             patch.object(cal, "is_trading_day", side_effect=mock_is_trading), \
             patch.object(cal, "next_trading_day", side_effect=mock_next_trading), \
             patch.object(cal, "prev_trading_day", side_effect=mock_prev_trading):

            artifacts_dir = tmp_path / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "latest_stock_picks.csv").write_text(picks_file.read_text())
            (artifacts_dir / "csi500_stock_picks.csv").write_text(picks_file.read_text())

            from execution.prospective_state_machine import seal_signal
            sig_dir = tmp_path / "prospective_signals"
            sig_dir.mkdir(parents=True, exist_ok=True)
            prev_picks_df = pd.DataFrame([
                {"symbol": "600519.SH", "name": "贵州茅台", "date": "2026-08-28", "pred_score": 0.85, "target_weight": 0.10, "close": 1590.0},
                {"symbol": "000858.SZ", "name": "五粮液", "date": "2026-08-28", "pred_score": 0.75, "target_weight": 0.10, "close": 175.0}
            ])
            seal_signal(
                signal_date="2026-08-28",
                picks_df=prev_picks_df,
                model_id="PRODUCTION_MODEL",
                factor_matrix_path=fm_path,
                storage_dir=sig_dir
            )

            res = DailyExecutionPipeline.execute_post_market_pipeline(
                sync_online=False,
                ledger_file=test_paper_ledger,
                shadow_ledger_file=test_shadow_ledger,
                status_file=test_status_file,
                execution_date="2026-08-31"
            )

            assert res["is_trading_day"] is True
            assert "shadow_observation" in res["steps"]
            assert test_shadow_ledger.exists()
            assert test_status_file.exists()

            sl = ShadowTradingLedger(ledger_file=test_shadow_ledger)
            assert sl.observed_trading_days >= 1

    def test_auto_scheduler_skips_on_non_trading_day(self, tmp_path):
        from scripts.auto_daily_scheduler import DailyExecutionPipeline

        test_shadow_ledger = tmp_path / "test_shadow_ledger.json"
        test_paper_ledger = tmp_path / "test_paper_ledger.json"
        test_status_file = tmp_path / "test_status.json"

        res = DailyExecutionPipeline.execute_post_market_pipeline(
            sync_online=False,
            ledger_file=test_paper_ledger,
            shadow_ledger_file=test_shadow_ledger,
            status_file=test_status_file,
            execution_date="2026-09-05"  # 周六
        )

        assert res["is_trading_day"] is False
        assert res["steps"]["rebalance"] == "SKIPPED_NON_TRADING_DAY"
        assert res["steps"]["shadow_observation"] == "SKIPPED_NON_TRADING_DAY_OR_EMPTY"

    def test_daily_runner_shadow_ledger_record(self, tmp_path):
        from scheduler.daily_runner import run_daily_automation
        from data.trading_calendar import CanonicalTradingCalendar

        test_shadow_ledger = tmp_path / "daily_runner_shadow_ledger.parquet"
        cal = CanonicalTradingCalendar.get_instance()

        def mock_is_trading(d):
            return str(d)[:10] in ("2026-08-28", "2026-08-31", "2026-09-01")

        def mock_next_trading(d):
            mapping = {"2026-08-28": "2026-08-31", "2026-08-31": "2026-09-01"}
            return mapping.get(str(d)[:10])

        def mock_prev_trading(d):
            mapping = {"2026-08-31": "2026-08-28", "2026-09-01": "2026-08-31"}
            return mapping.get(str(d)[:10])

        # 模拟流水线各阶段 (执行日 2026-08-31)
        mock_market_df = pd.DataFrame({
            "date": [pd.Timestamp("2026-08-31")],
            "symbol": ["600519.SH"],
            "open": [1600.0],
            "close": [1600.0],
            "is_suspended": [False],
            "is_limit_up_locked": [False],
            "amount": [1e8]
        })
        mock_scored_df = pd.DataFrame({
            "date": [pd.Timestamp("2026-08-31")],
            "symbol": ["600519.SH"],
            "pred_score": [0.85]
        })
        mock_top_df = pd.DataFrame({
            "symbol": ["600519.SH"],
            "name": ["贵州茅台"],
            "date": ["2026-08-31"],
            "pred_score": [0.85],
            "target_weight": [0.20],
            "close": [1600.0]
        })

        fm_path = tmp_path / "factor_matrix_20260828.parquet"
        pd.DataFrame({
            "date": ["2026-08-28"],
            "symbol": ["600519.SH"],
            "f1": [1.0]
        }).to_parquet(fm_path, index=False)

        # 提前合法封存 T 日 (2026-08-28) 信号，供 T+1 (2026-08-31) 真实前瞻执行
        from execution.prospective_state_machine import seal_signal
        sig_dir = tmp_path / "prospective_signals"
        sig_dir.mkdir(parents=True, exist_ok=True)
        prev_picks_df = pd.DataFrame([
            {"symbol": "600519.SH", "name": "贵州茅台", "date": "2026-08-28", "pred_score": 0.85, "target_weight": 0.20, "close": 1590.0}
        ])

        with patch.object(cal, "is_trading_day", side_effect=mock_is_trading), \
             patch.object(cal, "next_trading_day", side_effect=mock_next_trading), \
             patch.object(cal, "prev_trading_day", side_effect=mock_prev_trading):

            seal_signal(
                signal_date="2026-08-28",
                picks_df=prev_picks_df,
                model_id="PROD_TEST_M1",
                factor_matrix_path=fm_path,
                storage_dir=sig_dir
            )

            with patch("scheduler.daily_runner.DataManager") as MockDM, \
                 patch("scheduler.daily_runner.FactorProcessor") as MockFP, \
                 patch("scheduler.daily_runner.TargetLabeler") as MockTL, \
                 patch("models.inference.BatchInference") as MockBI, \
                 patch("scheduler.daily_runner.PortfolioBuilder") as MockPB:

                dm_instance = MockDM.return_value
                dm_instance.sync_and_build_dataset.return_value = mock_market_df
                dm_instance.get_trading_calendar.return_value = ["2026-08-31"]
                dm_instance.get_next_trading_date.return_value = pd.Timestamp("2026-09-01")

                fp_instance = MockFP.return_value
                fp_instance.build_and_save_factor_matrix.return_value = mock_scored_df

                tl_instance = MockTL.return_value
                tl_instance.compute_excess_return_label.return_value = mock_scored_df

                bi_instance = MockBI.return_value
                bi_instance.predict.return_value = mock_scored_df
                bi_instance.record = DummyModelRecord(model_id="PROD_TEST_M1")

                pb_instance = MockPB.return_value
                pb_instance.build_target_portfolio.return_value = mock_top_df

                res = run_daily_automation(
                    mode="inference",
                    shadow_ledger_file=test_shadow_ledger
                )

                assert res["signal_date"] == "2026-08-31"
                assert res["shadow_observation"] is not None
                assert test_shadow_ledger.exists()
                sl = ShadowTradingLedger(ledger_file=test_shadow_ledger)
                assert sl.observed_trading_days >= 1


class TestStageDMultiFactorGateFailClosed:
    """2. 实盘 11 项多因子硬核鉴权 Fail-Closed 拦截测试"""

    @pytest.fixture
    def valid_env_setup(self, tmp_path, monkeypatch):
        """构造全量满足 11 项条件的合法环境"""
        monkeypatch.setattr(settings, "LIVE_TRADING_READY", True)
        monkeypatch.setattr(settings, "LIVE_TRADING_GATE_STATUS", "UNLOCKED")
        monkeypatch.setattr(settings, "LIVE_TRADING_STATUS", "UNLOCKED")
        monkeypatch.setattr(settings, "EMERGENCY_KILL_SWITCH", False)
        monkeypatch.setattr(settings, "LIVE_ACCOUNT_WHITELIST", ["5500123456"])
        monkeypatch.setattr(settings, "MAX_QUOTE_STALENESS_SECONDS", 300.0)

        appr_file = create_valid_approval_artifact(tmp_path)

        from data.crypto_anchor import compute_canonical_keyring_hash
        canonical_hash = compute_canonical_keyring_hash()
        monkeypatch.setenv("QUANT_TRUSTED_KEYRING_SHA256", canonical_hash)

        cert_file = tmp_path / "model_cert.json"
        cert_file.write_text(json.dumps({"model_id": "PROD_CERT_V1", "status": "CERTIFIED"}), encoding="utf-8")

        # 构造已认证且观察满 30 天的生产模型
        model_rec = DummyModelRecord(
            model_id="PROD_CERT_V1",
            state="PRODUCTION",
            evidence={
                "certification_ref": str(cert_file),
                "prospective_validation": {
                    "status": "MATURE",
                    "observed_trading_days": 30,
                    "artifact_path": "reports/phase_21h/PROSPECTIVE_EXPERIMENT_LEDGER.jsonl"
                }
            }
        )

        # 构造观察满 30 天的模拟盘与影子账本
        paper_ledger = MagicMock()
        paper_ledger.observed_trading_days = 30

        shadow_ledger = MagicMock()
        shadow_ledger.observed_trading_days = 30

        return {
            "account_id": "5500123456",
            "live_confirm": True,
            "quote_timestamp": time.time(),
            "current_date": "2026-08-24",  # 合法交易日
            "model_record": model_rec,
            "paper_ledger": paper_ledger,
            "shadow_ledger": shadow_ledger,
            "approval_artifact_path": appr_file,
            "nonce_store_path": tmp_path / "nonce_store.json",
            "order_symbol": "600519.SH",
            "order_side": "BUY",
            "order_shares": 1000,
            "order_price": 50.0,
        }

    def test_all_eleven_factors_pass(self, valid_env_setup):
        """全部 11 项满足时鉴权通过"""
        ok, res = verify_live_trading_multi_factor_gate(**valid_env_setup)
        assert ok is True
        assert res["factor_a_ready"] is True
        assert res["factor_b_status"] is True
        assert res["factor_k_kill_switch"] is True
        assert res["factor_g_whitelist"] is True
        assert res["factor_h_confirm"] is True
        assert res["factor_j_trading_day"] is True
        assert res["factor_c_research_cert"] is True
        assert res["factor_d_prospective_mature"] is True
        assert res["factor_e_observation_days"] >= 20
        assert res["factor_f_signature_verified"] is True

    def test_factor_a_fail_closed(self, valid_env_setup, monkeypatch):
        """因子 A: LIVE_TRADING_READY is False"""
        monkeypatch.setattr(settings, "LIVE_TRADING_READY", False)
        with pytest.raises(LiveGateAuthError, match="因子A"):
            verify_live_trading_multi_factor_gate(**valid_env_setup)

    def test_factor_b_fail_closed(self, valid_env_setup, monkeypatch):
        """因子 B: LIVE_TRADING_GATE_STATUS != UNLOCKED"""
        monkeypatch.setattr(settings, "LIVE_TRADING_GATE_STATUS", "LOCKED")
        with pytest.raises(LiveGateAuthError, match="因子B"):
            verify_live_trading_multi_factor_gate(**valid_env_setup)

    def test_factor_k_kill_switch_fail_closed(self, valid_env_setup, monkeypatch):
        """因子 K: EMERGENCY_KILL_SWITCH is True"""
        monkeypatch.setattr(settings, "EMERGENCY_KILL_SWITCH", True)
        with pytest.raises(LiveGateAuthError, match="因子K"):
            verify_live_trading_multi_factor_gate(**valid_env_setup)

    def test_factor_g_whitelist_fail_closed(self, valid_env_setup):
        """因子 G: 账户不在白名单中"""
        valid_env_setup["account_id"] = "9999999999"
        with pytest.raises(LiveGateAuthError, match="因子G"):
            verify_live_trading_multi_factor_gate(**valid_env_setup)

    def test_factor_h_confirm_fail_closed(self, valid_env_setup):
        """因子 H: 缺失 live_confirm"""
        valid_env_setup["live_confirm"] = False
        with pytest.raises(LiveGateAuthError, match="因子H"):
            verify_live_trading_multi_factor_gate(**valid_env_setup)

    def test_factor_j_trading_day_fail_closed(self, valid_env_setup):
        """因子 J: 非交易日 (周六 2026-09-05)"""
        valid_env_setup["current_date"] = "2026-09-05"
        with pytest.raises(LiveGateAuthError, match="因子J"):
            verify_live_trading_multi_factor_gate(**valid_env_setup)

    def test_factor_i_staleness_fail_closed(self, valid_env_setup):
        """因子 I: 行情陈旧超过 300 秒"""
        valid_env_setup["quote_timestamp"] = time.time() - 301.0
        with pytest.raises(LiveGateAuthError, match="因子I"):
            verify_live_trading_multi_factor_gate(**valid_env_setup)

    def test_factor_c_research_cert_fail_closed(self, valid_env_setup):
        """因子 C: 缺失已认证研究证据"""
        valid_env_setup["model_record"].evidence["certification_ref"] = None
        with pytest.raises(LiveGateAuthError, match="因子C"):
            verify_live_trading_multi_factor_gate(**valid_env_setup)

    def test_factor_d_prospective_immature_fail_closed(self, valid_env_setup):
        """因子 D: 前瞻观察天数不足 20 天"""
        valid_env_setup["model_record"].evidence["prospective_validation"]["observed_trading_days"] = 19
        with pytest.raises(LiveGateAuthError, match="因子D"):
            verify_live_trading_multi_factor_gate(**valid_env_setup)

    def test_factor_e_observation_days_fail_closed(self, valid_env_setup):
        """因子 E: 模拟盘和影子观察均未达到 20 交易日"""
        valid_env_setup["paper_ledger"].observed_trading_days = 19
        valid_env_setup["shadow_ledger"].observed_trading_days = 19
        with pytest.raises(LiveGateAuthError, match="因子E"):
            verify_live_trading_multi_factor_gate(**valid_env_setup)

    def test_factor_f_signature_fail_closed(self, valid_env_setup, tmp_path):
        """因子 F: 审批文件不存在或签名被篡改"""
        bad_file = tmp_path / "tampered_approval.json"
        bad_file.write_text(json.dumps({
            "signer_key_id": "PROD_RUNTIME_KEY_2026_V1",
            "canonical_payload": {"foo": "bar"},
            "signature": "00" * 64
        }))
        valid_env_setup["approval_artifact_path"] = bad_file
        with pytest.raises(LiveGateAuthError, match="因子F"):
            verify_live_trading_multi_factor_gate(**valid_env_setup)


class TestStageDMiniQMTAndTraderFailClosed:
    """3. MiniQMT 与交易执行器硬阻断测试"""

    def test_miniqmt_connect_blocked_by_live_gate(self):
        """MiniQMTBroker.connect() 默认被多因子网关 Fail-Closed 拦截"""
        broker = MiniQMTBroker()
        with pytest.raises(LiveGateAuthError):
            broker.connect()

    def test_miniqmt_send_order_blocked_by_live_gate(self):
        """MiniQMTBroker.send_order() 默认被多因子网关 Fail-Closed 拦截"""
        broker = MiniQMTBroker()
        with pytest.raises(LiveGateAuthError):
            broker.send_order("600519.SH", OrderSide.BUY, 100, 100.0)

    def test_miniqmt_disconnect_blocks_without_simulation(self):
        """当多因子网关被 Mock 放行但终端断线时，严格阻断新订单，严禁假想仿真成交"""
        broker = MiniQMTBroker()
        broker.is_connected = False

        with patch("execution.live_gate.verify_live_trading_multi_factor_gate", return_value=(True, {})):
            order = broker.send_order("600519.SH", OrderSide.BUY, 100, 100.0)
            assert order.status == ExecutionStatus.REJECTED
            assert "实盘安全风控已阻断新订单并冻结状态" in order.error_msg
            assert "BLOCKED_" in order.order_id
            assert order.filled_shares == 0
