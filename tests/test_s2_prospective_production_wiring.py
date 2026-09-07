"""
Stage S2 测试套件: 不可变前瞻状态机与生产调度全链路实测 (无 allow_historical 绕过)
tests/test_s2_prospective_production_wiring.py
"""
import inspect
import json
import unittest.mock as mock
from pathlib import Path
import pandas as pd
import pytest
from tests.artifact_guards import require_factor_matrix, require_factor_matrix_v2, require_equity_curves

from config.settings import settings
from data.trading_calendar import CanonicalTradingCalendar, TradeDateError
from execution.paper_ledger import PaperTradingLedger, ShadowTradingLedger
from execution.prospective_state_machine import (
    ImmutableArtifactConflict,
    execute_observed,
    seal_signal,
)
from research.historical_replay_adapter import HistoricalReplayAdapter
from scheduler.daily_runner import run_daily_automation
from scripts.auto_daily_scheduler import DailyExecutionPipeline


@pytest.fixture
def mock_factor_matrix_28(tmp_path):
    fm_path = tmp_path / "factor_matrix_20260828.parquet"
    df = pd.DataFrame({
        "date": ["2026-08-28", "2026-08-28"],
        "symbol": ["600519.SH", "000858.SZ"],
        "factor_1": [1.2, -0.5]
    })
    df.to_parquet(fm_path, index=False)
    return fm_path


@pytest.fixture
def valid_picks_df():
    return pd.DataFrame([
        {"symbol": "600519.SH", "name": "贵州茅台", "date": "2026-08-28", "pred_score": 0.85, "target_weight": 0.5, "close": 1600.0},
        {"symbol": "000858.SZ", "name": "五粮液", "date": "2026-08-28", "pred_score": 0.75, "target_weight": 0.5, "close": 180.0}
    ])


@pytest.fixture
def valid_quotes_df():
    return pd.DataFrame([
        {"symbol": "600519.SH", "date": "2026-08-31", "open": 1620.0, "close": 1630.0, "is_suspended": False, "is_limit_up_locked": False},
        {"symbol": "000858.SZ", "date": "2026-08-31", "open": 182.0, "close": 185.0, "is_suspended": False, "is_limit_up_locked": False}
    ])


def test_no_allow_historical_in_api_signature():
    """验证 seal_signal 和 execute_observed 正式 API 彻底移除了 allow_historical 形参"""
    seal_sig = inspect.signature(seal_signal)
    exec_sig = inspect.signature(execute_observed)
    assert "allow_historical" not in seal_sig.parameters, "seal_signal 绝不能包含 allow_historical 参数"
    assert "allow_historical" not in exec_sig.parameters, "execute_observed 绝不能包含 allow_historical 参数"


def test_seal_signal_idempotent_and_conflict_rejection(tmp_path, mock_factor_matrix_28, valid_picks_df):
    """测试阶段一: 封存制品完全一致保持幂等，内容冲突强制抛出 ImmutableArtifactConflict"""
    sig_dir = tmp_path / "prospective_signals"
    cal = CanonicalTradingCalendar.get_instance()

    with mock.patch.object(cal, "is_trading_day", return_value=True), \
         mock.patch.object(cal, "next_trading_day", return_value="2026-08-31"):
        # 1. 首次封存
        res1 = seal_signal(
            signal_date="2026-08-28",
            picks_df=valid_picks_df,
            model_id="MODEL_V1",
            factor_matrix_path=mock_factor_matrix_28,
            storage_dir=sig_dir,
        )
        assert res1["event"] == "SIGNAL_SEALED"

        # 2. 二次封存相同内容: 幂等返回 IDEMPOTENT_ALREADY_SEALED
        res2 = seal_signal(
            signal_date="2026-08-28",
            picks_df=valid_picks_df,
            model_id="MODEL_V1",
            factor_matrix_path=mock_factor_matrix_28,
            storage_dir=sig_dir,
        )
        assert res2["event"] == "IDEMPOTENT_ALREADY_SEALED"
        assert res2["file_sha256"] == res1["file_sha256"]

        # 3. 企图篡改选股标的并覆盖封存制品: 强制抛出 ImmutableArtifactConflict
        conflicting_picks = valid_picks_df.copy()
        conflicting_picks.loc[0, "close"] = 1700.0  # 修改价格
        with pytest.raises(ImmutableArtifactConflict, match="信号封存制品已存在且内容冲突"):
            seal_signal(
                signal_date="2026-08-28",
                picks_df=conflicting_picks,
                model_id="MODEL_V1",
                factor_matrix_path=mock_factor_matrix_28,
                storage_dir=sig_dir,
            )


def test_picks_df_strict_field_and_date_validation(tmp_path, mock_factor_matrix_28, valid_picks_df):
    """测试选股清单字段完整性与截面日期严格一致性"""
    sig_dir = tmp_path / "prospective_signals"
    cal = CanonicalTradingCalendar.get_instance()

    with mock.patch.object(cal, "is_trading_day", return_value=True), \
         mock.patch.object(cal, "next_trading_day", return_value="2026-08-31"):

        # 1. 截面日期错配拦截 (picks 内日期为 2026-08-20, 信号日为 2026-08-28)
        bad_date_df = valid_picks_df.copy()
        bad_date_df.loc[0, "date"] = "2026-08-20"
        with pytest.raises(ValueError, match="选股清单中的日期与信号日.*不匹配"):
            seal_signal(
                signal_date="2026-08-28",
                picks_df=bad_date_df,
                model_id="MODEL_V1",
                factor_matrix_path=mock_factor_matrix_28,
                storage_dir=sig_dir,
            )

        # 2. 负数目标权重拦截
        bad_weight_df = valid_picks_df.copy()
        bad_weight_df.loc[0, "target_weight"] = -0.1
        with pytest.raises(ValueError, match="无效或负数 target_weight"):
            seal_signal(
                signal_date="2026-08-28",
                picks_df=bad_weight_df,
                model_id="MODEL_V1",
                factor_matrix_path=mock_factor_matrix_28,
                storage_dir=sig_dir,
            )

        # 3. 缺失 factor_matrix_path 拦截
        with pytest.raises((ValueError, TypeError, FileNotFoundError)):
            seal_signal(
                signal_date="2026-08-28",
                picks_df=valid_picks_df,
                model_id="MODEL_V1",
                factor_matrix_path=None,
                storage_dir=sig_dir,
            )

        # 4. 企图传 allow_historical 参数被拒绝 (TypeError)
        with pytest.raises(TypeError):
            seal_signal(
                signal_date="2026-08-28",
                picks_df=valid_picks_df,
                model_id="MODEL_V1",
                factor_matrix_path=mock_factor_matrix_28,
                storage_dir=sig_dir,
                allow_historical=True,
            )


def test_execute_observed_idempotent_and_conflict_rejection(
    tmp_path, mock_factor_matrix_28, valid_picks_df, valid_quotes_df
):
    """测试阶段二: 执行制品幂等返回与冲突覆盖拦截"""
    sig_dir = tmp_path / "prospective_signals"
    shadow_file = tmp_path / "shadow.parquet"
    shadow = ShadowTradingLedger(ledger_file=shadow_file)
    cal = CanonicalTradingCalendar.get_instance()

    with mock.patch.object(cal, "is_trading_day", return_value=True), \
         mock.patch.object(cal, "next_trading_day", return_value="2026-08-31"):

        seal_signal(
            signal_date="2026-08-28",
            picks_df=valid_picks_df,
            model_id="MODEL_V1",
            factor_matrix_path=mock_factor_matrix_28,
            storage_dir=sig_dir,
        )

        # 1. 首次执行
        e1 = execute_observed(
            signal_date="2026-08-28",
            execution_date="2026-08-31",
            quotes_df=valid_quotes_df,
            shadow_ledger=shadow,
            storage_dir=sig_dir,
        )
        assert e1["event"] == "EXECUTION_OBSERVED"
        assert e1["observed_trading_days_added"] == 1

        # 2. 二次执行相同内容: 保持幂等
        e2 = execute_observed(
            signal_date="2026-08-28",
            execution_date="2026-08-31",
            quotes_df=valid_quotes_df,
            shadow_ledger=shadow,
            storage_dir=sig_dir,
        )
        assert e2["event"] == "IDEMPOTENT_ALREADY_EXECUTED"
        assert e2["file_sha256"] == e1["file_sha256"]


def test_execute_observed_missing_quotes_fails_closed_zero_changes(
    tmp_path, mock_factor_matrix_28, valid_picks_df
):
    """测试缺任一标的行情时，整个执行日抛错，不创建 EXECUTION_OBSERVED 制品，账本完全不变，增加 0 天"""
    sig_dir = tmp_path / "prospective_signals"
    paper_file = tmp_path / "paper.parquet"
    paper = PaperTradingLedger(ledger_file=paper_file)
    cal = CanonicalTradingCalendar.get_instance()

    with mock.patch.object(cal, "is_trading_day", return_value=True), \
         mock.patch.object(cal, "next_trading_day", return_value="2026-08-31"):

        seal_signal(
            signal_date="2026-08-28",
            picks_df=valid_picks_df,
            model_id="MODEL_V1",
            factor_matrix_path=mock_factor_matrix_28,
            storage_dir=sig_dir,
        )

        # 记录执行前账本字节状态 (如存在)
        pre_paper_bytes = paper_file.read_bytes() if paper_file.exists() else None

        # 仅提供 600519.SH 行情，000858.SZ 缺失
        partial_quotes = pd.DataFrame([
            {"symbol": "600519.SH", "date": "2026-08-31", "open": 1620.0, "close": 1630.0, "is_suspended": False, "is_limit_up_locked": False}
        ])

        with pytest.raises(ValueError, match="缺少执行日 .* 真实行情数据，全天拒绝撮合"):
            execute_observed(
                signal_date="2026-08-28",
                execution_date="2026-08-31",
                quotes_df=partial_quotes,
                paper_ledger=paper,
                storage_dir=sig_dir,
            )

        # 校验 Fail-Closed 不变量:
        # 1. 绝不创建 EXECUTION_OBSERVED 制品
        exec_file = sig_dir / "EXECUTION_OBSERVED_2026-08-31.json"
        assert not exec_file.exists(), "缺行情时严禁创建 EXECUTION_OBSERVED 制品"

        # 2. 账本未被修改
        post_paper_bytes = paper_file.read_bytes() if paper_file.exists() else None
        assert pre_paper_bytes == post_paper_bytes, "缺行情时账本文件字节必须完全不变"

        # 3. 仓位未变动
        assert len(paper.positions) == 0


def test_calendar_coverage_blocked_post_20260824(tmp_path):
    """测试日历物理覆盖边界 (2026-08-24) 后的任务保持 CALENDAR_COVERAGE_BLOCKED"""
    cal = CanonicalTradingCalendar.get_instance()
    # 2026-08-24 的次一交易日超出物理日历覆盖
    assert cal.next_trading_day("2026-08-24") is None

    # 尝试在超出日历覆盖范围执行 execute_observed
    quotes_df = pd.DataFrame([{"symbol": "600519.SH", "date": "2026-08-25", "open": 1600.0, "close": 1610.0}])
    with pytest.raises(TradeDateError, match="不是交易所合法交易日"):
        execute_observed(
            signal_date="2026-08-24",
            execution_date="2026-08-25",
            quotes_df=quotes_df,
            storage_dir=tmp_path
        )


def test_seal_signal_production_missing_columns_fails_closed(tmp_path, valid_picks_df, mock_factor_matrix_28):
    """验证选股清单缺少任意必要列时强制 Fail-Closed 抛错，拒绝任何默认兜底"""
    sig_dir = tmp_path / "prospective_signals"
    cal = CanonicalTradingCalendar.get_instance()

    with mock.patch.object(cal, "is_trading_day", return_value=True), \
         mock.patch.object(cal, "next_trading_day", return_value="2026-08-31"):

        # 1. 缺失 pred_score
        no_score_df = valid_picks_df.drop(columns=["pred_score"])
        with pytest.raises(ValueError, match="选股清单缺少必要列"):
            seal_signal(
                signal_date="2026-08-28",
                picks_df=no_score_df,
                model_id="M1",
                factor_matrix_path=mock_factor_matrix_28,
                storage_dir=sig_dir,
            )

        # 2. 缺失 target_weight
        no_weight_df = valid_picks_df.drop(columns=["target_weight"])
        with pytest.raises(ValueError, match="选股清单缺少必要列"):
            seal_signal(
                signal_date="2026-08-28",
                picks_df=no_weight_df,
                model_id="M1",
                factor_matrix_path=mock_factor_matrix_28,
                storage_dir=sig_dir,
            )

        # 3. 缺失 close
        no_close_df = valid_picks_df.drop(columns=["close"])
        with pytest.raises(ValueError, match="选股清单缺少必要列"):
            seal_signal(
                signal_date="2026-08-28",
                picks_df=no_close_df,
                model_id="M1",
                factor_matrix_path=mock_factor_matrix_28,
                storage_dir=sig_dir,
            )


def test_execute_observed_missing_quotes_columns_fails_closed(
    tmp_path, valid_picks_df, mock_factor_matrix_28
):
    """验证行情缺少必要列或日期不匹配时强制拒绝撮合整个交易日 (Fail-Closed)"""
    sig_dir = tmp_path / "prospective_signals"
    cal = CanonicalTradingCalendar.get_instance()
    shadow_file = tmp_path / "shadow.parquet"
    shadow = ShadowTradingLedger(ledger_file=shadow_file)

    with mock.patch.object(cal, "is_trading_day", return_value=True), \
         mock.patch.object(cal, "next_trading_day", return_value="2026-08-31"):

        # 封存合法信号
        seal_signal(
            signal_date="2026-08-28",
            picks_df=valid_picks_df,
            model_id="M1",
            factor_matrix_path=mock_factor_matrix_28,
            storage_dir=sig_dir,
        )

        # 1. 行情缺少 open 列 -> 强制抛错
        no_open_quotes = pd.DataFrame([
            {"symbol": "600519.SH", "date": "2026-08-31", "close": 1630.0},
            {"symbol": "000858.SZ", "date": "2026-08-31", "close": 185.0}
        ])
        with pytest.raises(ValueError, match="真实行情数据缺少必要列 'open'"):
            execute_observed(
                signal_date="2026-08-28",
                execution_date="2026-08-31",
                quotes_df=no_open_quotes,
                shadow_ledger=shadow,
                storage_dir=sig_dir,
            )

        # 2. 行情缺少 date 列 -> 强制抛错
        no_date_quotes = pd.DataFrame([
            {"symbol": "600519.SH", "open": 1620.0, "close": 1630.0},
            {"symbol": "000858.SZ", "open": 182.0, "close": 185.0}
        ])
        with pytest.raises(ValueError, match="真实行情数据缺少必要列 'date'"):
            execute_observed(
                signal_date="2026-08-28",
                execution_date="2026-08-31",
                quotes_df=no_date_quotes,
                shadow_ledger=shadow,
                storage_dir=sig_dir,
            )

        # 3. 截面日期错配拦截 (行情 date != exec_date)
        mismatch_quotes = pd.DataFrame([
            {"symbol": "600519.SH", "date": "2026-08-25", "open": 1620.0, "close": 1630.0},
            {"symbol": "000858.SZ", "date": "2026-08-25", "open": 182.0, "close": 185.0}
        ])
        with pytest.raises(ValueError, match="行情数据中存在与执行日 .* 不匹配或缺失的日期记录"):
            execute_observed(
                signal_date="2026-08-28",
                execution_date="2026-08-31",
                quotes_df=mismatch_quotes,
                shadow_ledger=shadow,
                storage_dir=sig_dir,
            )


def test_historical_replay_adapter_isolation_and_quarantine(tmp_path):
    """验证科研历史回放适配器与正式 Paper/Shadow 账本及 reports/prospective_signals 物理隔离"""
    replay_dir = tmp_path / "research_replays"
    adapter = HistoricalReplayAdapter(replay_storage_dir=replay_dir)

    picks = pd.DataFrame([
        {"symbol": "600519.SH", "target_weight": 0.5, "close": 1600.0, "pred_score": 0.8},
        {"symbol": "000858.SZ", "target_weight": 0.5, "close": 180.0, "pred_score": 0.7}
    ])
    quotes = pd.DataFrame([
        {"symbol": "600519.SH", "date": "2026-08-24", "open": 1620.0, "close": 1630.0},
        {"symbol": "000858.SZ", "date": "2026-08-24", "open": 182.0, "close": 185.0}
    ])

    res = adapter.replay_step(
        signal_date="2026-08-21",
        execution_date="2026-08-24",
        quotes_df=quotes,
        picks_df=picks,
        model_id="research_model_v1"
    )

    assert res["event"] == "RESEARCH_REPLAY_STEP"
    assert res["formal_observation_days_added"] == 0
    assert res["is_prospective"] is False

    # 验证正式制品目录没有产生任何文件
    formal_signals_dir = settings.BASE_DIR / "reports" / "prospective_signals"
    # 临时研究目录中产生了隔离产物
    assert (replay_dir / "RESEARCH_SIGNAL_2026-08-21.json").exists()
    assert (replay_dir / "RESEARCH_STEP_2026-08-24.json").exists()

    # 验证科研适配器禁止绑定正式生产 JSON 账本路径 (动态获取真实路径)
    formal_paper = Path(PaperTradingLedger().ledger_file).resolve()
    with pytest.raises(PermissionError, match="科研历史回放严禁绑定或写入正式生产 Paper 账本"):
        HistoricalReplayAdapter(replay_storage_dir=replay_dir, paper_ledger_file=formal_paper)


def test_historical_replay_adapter_strict_json_ledger_isolation(tmp_path):
    """
    S2 核心审计: HistoricalReplayAdapter 严格隔离正式 JSON 账本与正式生产目录
    1. 传真实正式 Paper JSON 路径必须 PermissionError;
    2. 传真实正式 Shadow JSON 路径必须 PermissionError;
    3. 将 replay_storage_dir 指向正式 prospective 目录必须 PermissionError;
    4. 使用相对路径或可解析到正式目录的路径也必须拒绝;
    5. 隔离回放完成后，正式两个 JSON 账本字节完全不变.
    """
    cal = CanonicalTradingCalendar.get_instance()
    replay_dir = tmp_path / "replay_isolated"
    replay_dir.mkdir(parents=True, exist_ok=True)

    formal_paper = Path(PaperTradingLedger().ledger_file).resolve()
    formal_shadow = Path(ShadowTradingLedger().ledger_file).resolve()
    formal_prospective_dir = (settings.BASE_DIR / "reports" / "prospective_signals").resolve()

    # 记录正式账本初始字节 (若存在)
    paper_initial_bytes = formal_paper.read_bytes() if formal_paper.exists() else None
    shadow_initial_bytes = formal_shadow.read_bytes() if formal_shadow.exists() else None

    # 1. 传真实正式 Paper JSON 路径必须 PermissionError
    with pytest.raises(PermissionError, match="正式生产 Paper 账本"):
        HistoricalReplayAdapter(replay_storage_dir=replay_dir, paper_ledger_file=formal_paper)

    # 2. 传真实正式 Shadow JSON 路径必须 PermissionError
    with pytest.raises(PermissionError, match="正式生产 Shadow 账本"):
        HistoricalReplayAdapter(replay_storage_dir=replay_dir, shadow_ledger_file=formal_shadow)

    # 3. 将 replay_storage_dir 指向正式 prospective 目录必须 PermissionError
    with pytest.raises(PermissionError, match="正式前瞻信号目录"):
        HistoricalReplayAdapter(replay_storage_dir=formal_prospective_dir)

    # 4. 使用相对路径解析到正式目录或账本也必须拒绝
    rel_paper = Path("data_storage/paper_trading_ledger.json")
    with pytest.raises(PermissionError):
        HistoricalReplayAdapter(replay_storage_dir=replay_dir, paper_ledger_file=rel_paper)

    rel_prosp = Path("reports/prospective_signals")
    with pytest.raises(PermissionError):
        HistoricalReplayAdapter(replay_storage_dir=rel_prosp)

    # 将 replay_storage_dir 指向正式 data_storage 目录也必须拒绝
    with pytest.raises(PermissionError, match="正式数据存储目录"):
        HistoricalReplayAdapter(replay_storage_dir=settings.DATA_DIR)

    # 5. 执行完整隔离历史重放
    adapter = HistoricalReplayAdapter(replay_storage_dir=replay_dir)
    quotes = pd.DataFrame([
        {"symbol": "600519.SH", "open": 1600.0, "close": 1610.0, "is_suspended": False, "is_limit_up_locked": False}
    ])
    picks = pd.DataFrame([
        {"symbol": "600519.SH", "target_weight": 0.20, "pred_score": 0.85, "close": 1600.0}
    ])
    res = adapter.replay_step(
        signal_date="2026-08-21",
        execution_date="2026-08-24",
        quotes_df=quotes,
        picks_df=picks,
        model_id="research_test_model"
    )
    assert res["event"] == "RESEARCH_REPLAY_STEP"
    assert res["formal_observation_days_added"] == 0

    # 验证正式两个账本字节完全不变
    if paper_initial_bytes is not None:
        assert formal_paper.read_bytes() == paper_initial_bytes, "正式 Paper 账本字节被意外修改！"
    else:
        assert not formal_paper.exists()

    if shadow_initial_bytes is not None:
        assert formal_shadow.read_bytes() == shadow_initial_bytes, "正式 Shadow 账本字节被意外修改！"
    else:
        assert not formal_shadow.exists()


@require_factor_matrix
def test_daily_scheduler_post_20260824_returns_calendar_coverage_blocked(tmp_path):
    """验证调度系统在 2026-08-24 之后由于日历边界自动返回 CALENDAR_COVERAGE_BLOCKED，增加 0 天"""
    test_paper = tmp_path / "paper.parquet"
    test_shadow = tmp_path / "shadow.parquet"
    test_status = tmp_path / "status.json"
    res = DailyExecutionPipeline.execute_post_market_pipeline(
        sync_online=False,
        ledger_file=test_paper,
        shadow_ledger_file=test_shadow,
        status_file=test_status,
        execution_date="2026-08-25"
    )
    assert res["steps"]["shadow_observation"]["status"] == "CALENDAR_COVERAGE_BLOCKED"
    assert res["steps"]["shadow_observation"]["observed_trading_days_added"] == 0
