"""
Stage R4: Sealed Prospective T+1 State Machine Verification Suite
(tests/test_r4_prospective_signal_execution_state_machine.py)

验证:
1. SIGNAL_SEALED:
   - 封存 T 日信号，生成不可篡改 JSON 与 SHA-256 manifest
   - 观察天数增加 0 (observed_trading_days + 0)
   - 校验特征矩阵截面日期一致性，错配时 fail-closed
2. EXECUTION_OBSERVED:
   - 物理核验 T 日 sealed 信号制品 SHA-256 签名
   - 篡改信号制品内容时 fail-closed (SHA256 校验失败)
   - T+1 真实行情物理撮合 (停牌、涨停锁死不可成交)
   - 标的必须全量覆盖行情，任一标的缺行情整日抛错拒绝 (Fail-Closed)
   - 观察天数增加 1 (observed_trading_days + 1)
3. T+1 相邻交易日约束与跨期拦截:
   - 信号日与执行日非严格 T+1 相邻交易日时 fail-closed
   - 严禁将 2026-08-24 旧信号在 2026-09 月记成新 Paper/Shadow 样本
   - 非前瞻历史回填拦截 (<= 2026-08-24)
"""

import json
import pytest
import pandas as pd
from pathlib import Path

from data.trading_calendar import CanonicalTradingCalendar, TradeDateError
from execution.paper_ledger import PaperTradingLedger, ShadowTradingLedger
from execution.prospective_state_machine import ProspectiveStateMachine, seal_signal, execute_observed


@pytest.fixture
def prospective_calendar(monkeypatch):
    """模拟后 2026-08-24 的真实前瞻交易日历"""
    cal = CanonicalTradingCalendar.get_instance()
    trading_days = {"2026-08-28", "2026-08-31", "2026-09-01"}
    orig_is_trading = cal.is_trading_day
    orig_next = cal.next_trading_day

    def mock_is_trading(d):
        d_str = str(d)[:10]
        if d_str in trading_days:
            return True
        return orig_is_trading(d)

    def mock_next(d):
        d_str = str(d)[:10]
        if d_str == "2026-08-28":
            return "2026-08-31"
        if d_str == "2026-08-31":
            return "2026-09-01"
        return orig_next(d)

    monkeypatch.setattr(cal, "is_trading_day", mock_is_trading)
    monkeypatch.setattr(cal, "next_trading_day", mock_next)
    return cal


@pytest.fixture
def mock_matrix_file(tmp_path):
    """构造特征矩阵用于截面日期核验 (截面日期 2026-08-28)"""
    matrix_path = tmp_path / "test_factor_matrix.parquet"
    df = pd.DataFrame({
        "date": ["2026-08-27", "2026-08-28"],
        "symbol": ["600026.SH", "601138.SH"],
        "f1": [1.0, 2.0]
    })
    df.to_parquet(matrix_path, index=False)
    return matrix_path


@pytest.fixture
def sample_picks():
    return pd.DataFrame([
        {"symbol": "600026.SH", "name": "中远海能", "date": "2026-08-28", "pred_score": 0.85, "target_weight": 0.50, "close": 20.0},
        {"symbol": "601138.SH", "name": "工业富联", "date": "2026-08-28", "pred_score": 0.75, "target_weight": 0.50, "close": 60.0}
    ])


@pytest.fixture
def sample_quotes_t1():
    return pd.DataFrame([
        {"symbol": "600026.SH", "date": "2026-08-31", "open": 20.5, "close": 21.0, "is_suspended": False, "is_limit_up_locked": False},
        {"symbol": "601138.SH", "date": "2026-08-31", "open": 61.0, "close": 62.0, "is_suspended": False, "is_limit_up_locked": False}
    ])


def test_r4_signal_sealed_creates_manifest_and_adds_zero_days(tmp_path, mock_matrix_file, sample_picks, prospective_calendar):
    """测试阶段一: SIGNAL_SEALED 生成不可篡改制品且观察天数不增加"""
    sig_dir = tmp_path / "prospective_signals"
    
    res = seal_signal(
        signal_date="2026-08-28",
        picks_df=sample_picks,
        model_id="TEST_MODEL_V1",
        factor_matrix_path=mock_matrix_file,
        storage_dir=sig_dir,
    )
    
    assert res["event"] == "SIGNAL_SEALED"
    assert res["observed_trading_days_added"] == 0
    assert res["signal_date"] == "2026-08-28"
    
    sealed_file = sig_dir / "SIGNAL_SEALED_2026-08-28.json"
    manifest_file = sig_dir / "SIGNAL_SEALED_2026-08-28.manifest.json"
    assert sealed_file.exists()
    assert manifest_file.exists()
    
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["is_sealed"] is True
    assert manifest["observed_trading_days_added"] == 0
    assert len(manifest["file_sha256"]) == 64


def test_r4_signal_sealed_rejects_mismatched_factor_matrix_date(tmp_path, sample_picks, prospective_calendar):
    """测试阶段一: 特征矩阵最大日期与信号日错配时 fail-closed"""
    sig_dir = tmp_path / "prospective_signals"
    matrix_path = tmp_path / "mismatched_matrix.parquet"
    df = pd.DataFrame({"date": ["2026-08-27"], "symbol": ["600000.SH"]})
    df.to_parquet(matrix_path, index=False)
    
    with pytest.raises(ValueError, match="特征矩阵最大日期.*与信号日.*不匹配"):
        seal_signal(
            signal_date="2026-08-28",
            picks_df=sample_picks,
            model_id="TEST_MODEL_V1",
            factor_matrix_path=matrix_path,
            storage_dir=sig_dir,
        )


def test_r4_execution_observed_verifies_hash_and_increments_one_day(
    tmp_path, mock_matrix_file, sample_picks, sample_quotes_t1, prospective_calendar
):
    """测试阶段二: EXECUTION_OBSERVED 物理验证 hash 并严格撮合，观察天数加 1"""
    sig_dir = tmp_path / "prospective_signals"
    paper_file = tmp_path / "paper_ledger.parquet"
    shadow_file = tmp_path / "shadow_ledger.parquet"
    
    paper = PaperTradingLedger(ledger_file=paper_file, initial_cash=1_000_000.0)
    shadow = ShadowTradingLedger(ledger_file=shadow_file, initial_cash=1_000_000.0)
    assert paper.observed_trading_days == 0
    assert shadow.observed_trading_days == 0
    
    # 1. T 日 (2026-08-28) 封存信号
    seal_signal(
        signal_date="2026-08-28",
        picks_df=sample_picks,
        model_id="TEST_MODEL_V1",
        factor_matrix_path=mock_matrix_file,
        storage_dir=sig_dir,
    )
    
    # 2. T+1 日 (2026-08-31) 真实物理执行
    exec_res = execute_observed(
        signal_date="2026-08-28",
        execution_date="2026-08-31",
        quotes_df=sample_quotes_t1,
        paper_ledger=paper,
        shadow_ledger=shadow,
        storage_dir=sig_dir,
    )
    
    assert exec_res["event"] == "EXECUTION_OBSERVED"
    assert exec_res["observed_trading_days_added"] == 1
    assert paper.observed_trading_days == 1
    assert shadow.observed_trading_days == 1
    assert len(paper.trade_history) >= 2


def test_r4_execution_observed_detects_tampered_sealed_file(
    tmp_path, mock_matrix_file, sample_picks, sample_quotes_t1, prospective_calendar
):
    """测试阶段二: 封存信号文件被篡改时，物理校验 SHA-256 拒绝执行 (fail-closed)"""
    sig_dir = tmp_path / "prospective_signals"
    
    seal_signal(
        signal_date="2026-08-28",
        picks_df=sample_picks,
        model_id="TEST_MODEL_V1",
        factor_matrix_path=mock_matrix_file,
        storage_dir=sig_dir,
    )
    
    sealed_file = sig_dir / "SIGNAL_SEALED_2026-08-28.json"
    content = json.loads(sealed_file.read_text(encoding="utf-8"))
    content["picks"].append({"symbol": "000001.SZ", "name": "平安银行", "target_weight": 0.1})
    sealed_file.write_text(json.dumps(content), encoding="utf-8")
    
    with pytest.raises(ValueError, match="封存制品 SHA256 校验失败"):
        execute_observed(
            signal_date="2026-08-28",
            execution_date="2026-08-31",
            quotes_df=sample_quotes_t1,
            storage_dir=sig_dir,
            paper_ledger=PaperTradingLedger(ledger_file=tmp_path / "p.parquet"),
        )


def test_r4_rejects_non_t_plus_one_adjacent_dates(
    tmp_path, mock_matrix_file, sample_picks, sample_quotes_t1, prospective_calendar
):
    """测试阶段二: 严格核验 A 股 T+1 相邻交易日约束"""
    sig_dir = tmp_path / "prospective_signals"
    
    seal_signal(
        signal_date="2026-08-28",
        picks_df=sample_picks,
        model_id="TEST_MODEL_V1",
        factor_matrix_path=mock_matrix_file,
        storage_dir=sig_dir,
    )
    
    # 试图跨期到 2026-09-01 执行 (T+1 应为 2026-08-31)
    with pytest.raises(TradeDateError, match=r"违反 A 股 T\+1 规则"):
        execute_observed(
            signal_date="2026-08-28",
            execution_date="2026-09-01",
            quotes_df=sample_quotes_t1,
            storage_dir=sig_dir,
            paper_ledger=PaperTradingLedger(ledger_file=tmp_path / "p.parquet"),
        )


def test_r4_strictly_forbids_backfilling_20260824_as_september_samples(tmp_path, sample_picks, sample_quotes_t1):
    """测试规则 7: 严禁把 2026-08-24 旧信号记成 2026-09 月的新 Paper/Shadow 样本"""
    sig_dir = tmp_path / "prospective_signals"
    
    with pytest.raises(ValueError, match="严禁把 2026-08-24 的旧信号记成 2026-09 月的新 Paper/Shadow 样本"):
        execute_observed(
            signal_date="2026-08-24",
            execution_date="2026-09-07",
            quotes_df=sample_quotes_t1,
            storage_dir=sig_dir,
        )


def test_r4_untradable_quote_filtering(tmp_path, mock_matrix_file, sample_picks, prospective_calendar):
    """测试真实物理行情约束: 标的包含停牌或一字涨停但具备真实行情"""
    sig_dir = tmp_path / "prospective_signals"
    paper_file = tmp_path / "paper_ledger.parquet"
    paper = PaperTradingLedger(ledger_file=paper_file, initial_cash=1_000_000.0)
    
    seal_signal(
        signal_date="2026-08-28",
        picks_df=sample_picks,
        model_id="TEST_MODEL_V1",
        factor_matrix_path=mock_matrix_file,
        storage_dir=sig_dir,
    )
    
    untradable_quotes = pd.DataFrame([
        {"symbol": "600026.SH", "date": "2026-08-31", "open": 20.0, "close": 20.0, "is_suspended": True, "is_limit_up_locked": False},
        {"symbol": "601138.SH", "date": "2026-08-31", "open": 60.0, "close": 60.0, "is_suspended": False, "is_limit_up_locked": True}
    ])
    
    exec_res = execute_observed(
        signal_date="2026-08-28",
        execution_date="2026-08-31",
        quotes_df=untradable_quotes,
        paper_ledger=paper,
        storage_dir=sig_dir,
    )
    
    assert exec_res["details"]["paper"]["trades_count"] == 0
    assert len(paper.positions) == 0


def test_r4_missing_any_quote_fails_closed_and_keeps_ledger_intact(
    tmp_path, mock_matrix_file, sample_picks, prospective_calendar
):
    """测试 S2 核心反例: 目标标的若有任意一只缺失行情，整日抛错拒绝，账本文件与天数完全不变"""
    sig_dir = tmp_path / "prospective_signals"
    paper_file = tmp_path / "paper_ledger.parquet"
    paper = PaperTradingLedger(ledger_file=paper_file, initial_cash=1_000_000.0)
    init_days = paper.observed_trading_days
    
    seal_signal(
        signal_date="2026-08-28",
        picks_df=sample_picks,
        model_id="TEST_MODEL_V1",
        factor_matrix_path=mock_matrix_file,
        storage_dir=sig_dir,
    )
    
    # 仅提供 600026.SH 行情，故意缺失 601138.SH
    partial_quotes = pd.DataFrame([
        {"symbol": "600026.SH", "date": "2026-08-31", "open": 20.0, "close": 20.0, "is_suspended": False, "is_limit_up_locked": False}
    ])
    
    with pytest.raises(ValueError, match="缺少执行日.*真实行情数据，全天拒绝撮合"):
        execute_observed(
            signal_date="2026-08-28",
            execution_date="2026-08-31",
            quotes_df=partial_quotes,
            paper_ledger=paper,
            storage_dir=sig_dir,
        )
    
    assert paper.observed_trading_days == init_days
    assert not (sig_dir / "EXECUTION_OBSERVED_2026-08-31.json").exists()
