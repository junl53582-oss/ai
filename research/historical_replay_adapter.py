"""
科研历史回放与离线仿真隔离适配器 (research/historical_replay_adapter.py)
用于离线科研、策略历史回放与测试仿真。
严格遵守物理隔离原则:
1. 输出仅允许写入隔离的 reports/research_replays/ 或调用方显式指定的临时研究目录；
2. 严禁写入 reports/prospective_signals、严禁触碰正式生产 PaperTradingLedger 或 ShadowTradingLedger 账本；
3. 绝不调用正式 prospective_state_machine 的 seal_signal 与 execute_observed，杜绝历史数据污染前瞻成熟度与实盘证据链。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from config.settings import settings
from data.trading_calendar import CanonicalTradingCalendar
from execution.paper_ledger import PaperTradingLedger, ShadowTradingLedger

logger = logging.getLogger("HistoricalReplayAdapter")

DEFAULT_REPLAY_DIR = settings.BASE_DIR / "reports" / "research_replays"


def _is_same_or_subpath(p: Path, target: Path) -> bool:
    try:
        p_res = p.resolve()
        target_res = target.resolve()
        if p_res == target_res:
            return True
        p_res.relative_to(target_res)
        return True
    except (ValueError, RuntimeError):
        return False


class HistoricalReplayAdapter:
    """科研历史回放专用隔离适配器"""

    def __init__(
        self,
        replay_storage_dir: Optional[Union[str, Path]] = None,
        paper_ledger_file: Optional[Union[str, Path]] = None,
        shadow_ledger_file: Optional[Union[str, Path]] = None,
    ):
        # 动态通过正式类实例获取真实正式生产账本物理路径并 resolve (不得硬编码)
        formal_paper = Path(PaperTradingLedger().ledger_file).resolve()
        formal_shadow = Path(ShadowTradingLedger().ledger_file).resolve()
        formal_prospective_dir = (settings.BASE_DIR / "reports" / "prospective_signals").resolve()
        formal_data_dir = Path(settings.DATA_DIR).resolve()

        # 1. 检验 replay_storage_dir 物理隔离性 (防止相对路径、符号链接或正式目录覆盖)
        storage_path = Path(replay_storage_dir or DEFAULT_REPLAY_DIR)
        storage_res = storage_path.resolve()

        if _is_same_or_subpath(storage_res, formal_prospective_dir):
            raise PermissionError("Fail-Closed: 科研历史回放严禁指向正式前瞻信号目录 (reports/prospective_signals)！")
        if _is_same_or_subpath(storage_res, formal_data_dir):
            raise PermissionError("Fail-Closed: 科研历史回放目录严禁指向正式数据存储目录 (data_storage)！")
        if storage_res in (formal_paper, formal_shadow):
            raise PermissionError("Fail-Closed: 科研历史回放目录严禁覆盖正式生产账本！")

        self.storage_dir = storage_path
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 2. 检验 paper_ledger_file 路径安全 (精确比对真实正式 JSON 账本与正式目录)
        if paper_ledger_file:
            p_res = Path(paper_ledger_file).resolve()
            if p_res in (formal_paper, formal_shadow):
                raise PermissionError("Fail-Closed: 科研历史回放严禁绑定或写入正式生产 Paper 账本！")
            if _is_same_or_subpath(p_res, formal_data_dir):
                raise PermissionError("Fail-Closed: 科研历史回放账本严禁置于正式数据目录 (data_storage)！")
            if _is_same_or_subpath(p_res, formal_prospective_dir):
                raise PermissionError("Fail-Closed: 科研历史回放账本严禁置于正式前瞻目录！")
            self.paper_file = Path(paper_ledger_file)
        else:
            self.paper_file = self.storage_dir / "research_paper_ledger.json"

        # 3. 检验 shadow_ledger_file 路径安全 (精确比对真实正式 JSON 账本与正式目录)
        if shadow_ledger_file:
            s_res = Path(shadow_ledger_file).resolve()
            if s_res in (formal_shadow, formal_paper):
                raise PermissionError("Fail-Closed: 科研历史回放严禁绑定或写入正式生产 Shadow 账本！")
            if _is_same_or_subpath(s_res, formal_data_dir):
                raise PermissionError("Fail-Closed: 科研历史回放账本严禁置于正式数据目录 (data_storage)！")
            if _is_same_or_subpath(s_res, formal_prospective_dir):
                raise PermissionError("Fail-Closed: 科研历史回放账本严禁置于正式前瞻目录！")
            self.shadow_file = Path(shadow_ledger_file)
        else:
            self.shadow_file = self.storage_dir / "research_shadow_ledger.json"

        self.paper_ledger = PaperTradingLedger(ledger_file=self.paper_file)
        self.shadow_ledger = ShadowTradingLedger(ledger_file=self.shadow_file)
        self.cal = CanonicalTradingCalendar.get_instance()

    def record_historical_signal(
        self,
        signal_date: str,
        picks_df: pd.DataFrame,
        model_id: str = "research_model",
    ) -> Dict[str, Any]:
        """记录离线科研历史信号到隔离目录 (RESEARCH_REPLAY_SIGNAL)"""
        sig_date = str(signal_date).strip()[:10]
        if not self.cal.is_trading_day(sig_date):
            raise ValueError(f"科研历史回放信号日 ({sig_date}) 不是合法交易所交易日！")

        if picks_df is None or picks_df.empty:
            raise ValueError("科研选股清单为空")

        out_file = self.storage_dir / f"RESEARCH_SIGNAL_{sig_date}.json"
        records = picks_df.to_dict(orient="records")
        payload = {
            "event": "RESEARCH_REPLAY_SIGNAL",
            "signal_date": sig_date,
            "model_id": model_id,
            "picks_count": len(records),
            "picks": records,
            "is_prospective": False,
            "quarantine_note": "RESEARCH_ONLY_ISOLATED_FROM_PROSPECTIVE",
        }
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def replay_step(
        self,
        signal_date: str,
        execution_date: str,
        quotes_df: pd.DataFrame,
        picks_df: pd.DataFrame,
        model_id: str = "research_model",
    ) -> Dict[str, Any]:
        """执行单步历史撮合仿真 (更新隔离的 research 账本，绝不增加正式观察天数)"""
        sig_date = str(signal_date).strip()[:10]
        exec_date = str(execution_date).strip()[:10]

        if not self.cal.is_trading_day(sig_date) or not self.cal.is_trading_day(exec_date):
            raise ValueError(f"信号日 ({sig_date}) 或执行日 ({exec_date}) 非交易日")

        expected_exec = self.cal.next_trading_day(sig_date)
        if expected_exec != exec_date:
            raise ValueError(f"T+1 错配: 预期 {expected_exec}, 实际 {exec_date}")

        # 记录科研信号
        sig_rec = self.record_historical_signal(sig_date, picks_df, model_id=model_id)

        # 撮合隔离 Paper
        price_col = "open" if "open" in quotes_df.columns else "close"
        price_dict = dict(zip(quotes_df["symbol"].astype(str), quotes_df[price_col].astype(float)))
        paper_res = self.paper_ledger.rebalance(
            target_df=picks_df,
            current_prices=price_dict,
            trade_date=exec_date,
            signal_date=sig_date,
            allow_historical=True,
        )

        # 撮合隔离 Shadow
        shadow_res = self.shadow_ledger.record_shadow_observation(
            target_df=picks_df,
            model_id=model_id,
            data_date=exec_date,
            market_df=quotes_df,
            signal_date=sig_date,
            allow_historical=True,
        )

        replay_payload = {
            "event": "RESEARCH_REPLAY_STEP",
            "signal_date": sig_date,
            "execution_date": exec_date,
            "model_id": model_id,
            "paper_trades": paper_res.get("total_trades", 0),
            "paper_nav": paper_res.get("nav", 1.0),
            "shadow_fills": shadow_res.get("shadow_fills_count", 0),
            "is_prospective": False,
            "formal_observation_days_added": 0,
        }
        step_file = self.storage_dir / f"RESEARCH_STEP_{exec_date}.json"
        step_file.write_text(json.dumps(replay_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return replay_payload
