"""
Prospective T+1 Execution State Machine (execution/prospective_state_machine.py)

两阶段物理前瞻状态机:
1. SIGNAL_SEALED (T 日盘后):
   - 检验 T 日合法交易日、特征矩阵最大日期 == signal_date (无未来信息或截面陈旧)
   - 生成不可篡改的封存预测制品: reports/prospective_signals/SIGNAL_SEALED_{signal_date}.json 及 SHA256 manifest
   - 【观察天数增加 0】 (observed_trading_days + 0)
   - 严禁回填历史 (<= 2026-08-24) 冒充前瞻信号
   - 独占式文件写入与幂等性防护: 已存在且一致返回 IDEMPOTENT_ALREADY_SEALED，内容冲突抛出 ImmutableArtifactConflict

2. EXECUTION_OBSERVED (T+1 日开盘/收盘):
   - 物理核验 T 日 sealed 信号制品存在且 SHA-256 校验通过 (防篡改)
   - 严格核验 canonical A 股 T+1 关系: cal.next_trading_day(signal_date) == execution_date
     - 拒绝跨度不符的非法执行 (例如拿 2026-08-24 旧信号在 2026-09-07 执行)
     - 严禁把 2026-08-24 的旧信号或历史数据直接记成 2026-09 月的新 Paper/Shadow 样本
   - 物理核验 T+1 日真实行情数据:
     - 校验真实行情存在 (open / close / price > 0)
     - 缺失真实行情标的严格标记 REJECTED_MISSING_QUOTE 并拒绝撮合 (禁止回退至 T 日收盘价)
     - 停牌 (is_suspended) 标的无法买卖
     - 一字涨停锁死 (is_limit_up_locked) 标的无法买入
     - 一字跌停锁死 (is_limit_down_locked) 标的无法卖出
   - 计算真实摩擦、滑点与规费成本，撮合更新 PaperTradingLedger 与 ShadowTradingLedger (至少提供一个账本)
   - 生成不可篡改的 EXECUTION_OBSERVED_{execution_date}.json 审计制品
   - 【观察天数增加由账本增量动态确定】 (observed_trading_days_added)
"""

import os
import sys
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config.settings import settings
from data.trading_calendar import CanonicalTradingCalendar, TradeDateError

logger = logging.getLogger("ProspectiveStateMachine")

DEFAULT_PROSPECTIVE_DIR = settings.BASE_DIR / "reports" / "prospective_signals"


class ImmutableArtifactConflict(ValueError):
    """当尝试篡改或覆盖已封存/已执行的不可变制品时抛出"""
    pass


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """计算物理文件的 SHA-256 哈希值"""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class ProspectiveStateMachine:
    """前瞻两阶段执行状态机"""

    @staticmethod
    def seal_signal(
        signal_date: str,
        picks_df: pd.DataFrame,
        model_id: str,
        factor_matrix_path: Optional[Union[str, Path]] = None,
        storage_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        阶段一: SIGNAL_SEALED (T 日盘后封存)
        - 校验 T 日为合法交易日
        - 必须校验其最大截面日期 == signal_date
        - 强制提供 factor_matrix_path 进行截面与哈希核验
        - 校验选股清单合法性 (必须包含 symbol, date, pred_score, target_weight, close 且全量 date == signal_date)
        - 严禁任何默认值兜底 (0.5/0.0/1.0/补日期一律拒绝)
        - 独占式不可变封存: 冲突抛出 ImmutableArtifactConflict，完全一致返回 IDEMPOTENT_ALREADY_SEALED
        - 【观察天数增加 0】
        """
        sig_date = str(signal_date).strip()[:10]
        cal = CanonicalTradingCalendar.get_instance()

        if not cal.is_trading_day(sig_date):
            raise TradeDateError(f"Fail-Closed: 信号日 ({sig_date}) 不是交易所合法交易日！")

        if sig_date <= "2026-08-24":
            raise ValueError(f"Fail-Closed: 禁止回填历史日期 ({sig_date} <= 2026-08-24) 冒充 prospective 信号！")

        if factor_matrix_path is None:
            raise ValueError("Fail-Closed: 封存前瞻信号必须提供 factor_matrix_path 进行截面日期与哈希核验！")

        fm_path = Path(factor_matrix_path)
        if not fm_path.exists():
            raise FileNotFoundError(f"Fail-Closed: 特征矩阵文件不存在: {fm_path}")
        df_dates = pd.read_parquet(fm_path, columns=["date"])
        max_date_in_matrix = str(df_dates["date"].max())[:10]
        if max_date_in_matrix != sig_date:
            raise ValueError(
                f"Fail-Closed: 特征矩阵最大日期 ({max_date_in_matrix}) 与信号日 ({sig_date}) 不匹配！严禁截面错配！"
            )
        matrix_sha = compute_file_sha256(fm_path)

        if picks_df is None or len(picks_df) == 0:
            raise ValueError("Fail-Closed: 选股决策清单为空，拒绝封存信号！")

        if "symbol" not in picks_df.columns:
            raise ValueError("Fail-Closed: 选股决策清单缺少必要列 'symbol'")

        picks_clean = picks_df.copy()

        # 严格要求全部 5 项核心字段，禁止任何静默默认值兜底 (Fail-Closed)
        required_picks_cols = ["symbol", "date", "pred_score", "target_weight", "close"]
        missing_cols = [c for c in required_picks_cols if c not in picks_clean.columns]
        if missing_cols:
            raise ValueError(f"Fail-Closed: 选股清单缺少必要列: {missing_cols}，禁止默认值兜底！")

        mismatches = picks_clean[picks_clean["date"].astype(str).str[:10] != sig_date]
        if not mismatches.empty or picks_clean["date"].isna().any():
            raise ValueError(f"Fail-Closed: 选股清单中的日期与信号日 ({sig_date}) 不匹配或缺失！严禁截面串期！")

        scores = pd.to_numeric(picks_clean["pred_score"], errors="coerce")
        if scores.isna().any() or np.isinf(scores).any():
            raise ValueError("Fail-Closed: 选股清单中存在无效 pred_score (NaN 或 Inf)")

        weights = pd.to_numeric(picks_clean["target_weight"], errors="coerce")
        if weights.isna().any() or (weights < 0).any():
            raise ValueError("Fail-Closed: 选股清单中存在无效或负数 target_weight")

        prices = pd.to_numeric(picks_clean["close"], errors="coerce")
        if prices.isna().any() or (prices <= 0).any():
            raise ValueError("Fail-Closed: 选股清单中存在无效或非正数 close 价格")

        picks_records = picks_clean.to_dict(orient="records")
        picks_records = sorted(picks_records, key=lambda x: str(x.get("symbol", "")))

        target_dir = Path(storage_dir) if storage_dir else DEFAULT_PROSPECTIVE_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        signal_file = target_dir / f"SIGNAL_SEALED_{sig_date}.json"
        manifest_file = target_dir / f"SIGNAL_SEALED_{sig_date}.manifest.json"

        # 独占式文件写入与不可变制品冲突防范
        if signal_file.exists():
            existing_bytes = signal_file.read_bytes()
            existing_sha = hashlib.sha256(existing_bytes).hexdigest()
            try:
                existing_data = json.loads(existing_bytes.decode("utf-8"))
            except Exception:
                existing_data = {}

            # 比较关键业务负载 (忽略 sealed_at 时间戳微秒级差异)
            is_same = (
                existing_data.get("signal_date") == sig_date
                and existing_data.get("model_id") == model_id
                and existing_data.get("picks") == picks_records
                and existing_data.get("factor_matrix_sha256") == matrix_sha
            )
            if is_same:
                logger.info(f"封存制品已存在且内容完全一致，保持幂等 (IDEMPOTENT_ALREADY_SEALED): {signal_file.name}")
                return {
                    "event": "IDEMPOTENT_ALREADY_SEALED",
                    "signal_date": sig_date,
                    "model_id": model_id,
                    "file_sha256": existing_sha,
                    "observed_trading_days_added": 0,
                    "status": "SEALED_PENDING_EXECUTION",
                    "sealed_payload": existing_data,
                }
            else:
                raise ImmutableArtifactConflict(
                    f"Fail-Closed: 信号封存制品已存在且内容冲突，禁止覆盖已封存制品: {signal_file.name}"
                )

        sealed_payload = {
            "event": "SIGNAL_SEALED",
            "signal_date": sig_date,
            "model_id": model_id,
            "picks_count": len(picks_records),
            "picks": picks_records,
            "factor_matrix_sha256": matrix_sha,
            "sealed_at": datetime.now().isoformat(),
            "observed_trading_days_added": 0,
            "status": "SEALED_PENDING_EXECUTION",
        }

        content_bytes = json.dumps(sealed_payload, indent=2, ensure_ascii=False).encode("utf-8")
        file_sha256 = hashlib.sha256(content_bytes).hexdigest()

        signal_file.write_bytes(content_bytes)

        manifest = {
            "file_name": signal_file.name,
            "file_sha256": file_sha256,
            "signal_date": sig_date,
            "model_id": model_id,
            "sealed_at": sealed_payload["sealed_at"],
            "is_sealed": True,
            "observed_trading_days_added": 0,
        }
        manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"T日信号已密码学封存: {signal_file.name} (SHA: {file_sha256[:8]}, 观察天数+0)")
        sealed_payload["file_sha256"] = file_sha256
        return sealed_payload

    @staticmethod
    def execute_observed(
        signal_date: str,
        execution_date: str,
        quotes_df: pd.DataFrame,
        paper_ledger: Optional[Any] = None,
        shadow_ledger: Optional[Any] = None,
        storage_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        阶段二: EXECUTION_OBSERVED (T+1 日物理执行与观察结算)
        - 必须至少提供一个账本 (paper_ledger 或 shadow_ledger)
        - 物理验证 T 日 sealed 信号制品存在且 sha256 校验通过
        - 严格核验 canonical A 股 T+1 约束: cal.next_trading_day(signal_date) == execution_date
        - 严禁将 2026-08-24 旧信号在 2026-09 月执行为新 Paper/Shadow 样本
        - 物理验证 T+1 真实行情 (有效正价格，检查开盘/执行价格)
        - 标的必须全量覆盖行情，任一标的缺行情整日抛错拒绝 (Fail-Closed)，禁止部分执行
        - 撮合更新 Paper/Shadow 账本，观察天数增量依据账本变化真实确定
        """
        sig_date = str(signal_date).strip()[:10]
        exec_date = str(execution_date).strip()[:10]

        # 严禁将 2026-08-24 旧信号在 2026-09 月执行 (最高优先级拦截)
        if sig_date <= "2026-08-24" and exec_date >= "2026-09-01":
            raise ValueError("Fail-Closed: 严禁把 2026-08-24 的旧信号记成 2026-09 月的新 Paper/Shadow 样本！")

        cal = CanonicalTradingCalendar.get_instance()

        if not cal.is_trading_day(sig_date):
            raise TradeDateError(f"Fail-Closed: 信号日 ({sig_date}) 不是交易所合法交易日！")
        if not cal.is_trading_day(exec_date):
            raise TradeDateError(f"Fail-Closed: 执行日 ({exec_date}) 不是交易所合法交易日！")

        if sig_date <= "2026-08-24" or exec_date <= "2026-08-24":
            raise ValueError(
                f"Fail-Closed: 禁止回填历史日期 (信号日: {sig_date}, 执行日: {exec_date} <= 2026-08-24) 冒充 prospective 执行！"
            )

        # 严格 A 股 T+1 相邻交易日约束
        expected_exec = cal.next_trading_day(sig_date)
        if expected_exec != exec_date:
            raise TradeDateError(
                f"Fail-Closed: 违反 A 股 T+1 规则! 信号日 {sig_date} 的合法次一交易日必须是 {expected_exec}, 实际传入执行日为 {exec_date}"
            )

        # 物理验证 T 日 sealed 信号制品存在且 SHA-256 校验通过
        target_dir = Path(storage_dir) if storage_dir else DEFAULT_PROSPECTIVE_DIR
        signal_file = target_dir / f"SIGNAL_SEALED_{sig_date}.json"
        manifest_file = target_dir / f"SIGNAL_SEALED_{sig_date}.manifest.json"

        if not signal_file.exists():
            raise FileNotFoundError(f"Fail-Closed: 未找到信号日 ({sig_date}) 的封存信号制品: {signal_file}")
        if not manifest_file.exists():
            raise FileNotFoundError(f"Fail-Closed: 未找到信号日 ({sig_date}) 的封存清单: {manifest_file}")

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        content_bytes = signal_file.read_bytes()
        calc_sha = hashlib.sha256(content_bytes).hexdigest()
        if calc_sha != manifest.get("file_sha256"):
            raise ValueError(
                f"Fail-Closed: 信号日 ({sig_date}) 封存制品 SHA256 校验失败! 预期 {manifest.get('file_sha256')}, 实际 {calc_sha}"
            )

        sealed_payload = json.loads(content_bytes.decode("utf-8"))
        picks_records = sealed_payload.get("picks", [])
        model_id = sealed_payload.get("model_id", "UNKNOWN")

        # 独占式执行制品幂等 vs 篡改冲突核验
        exec_file = target_dir / f"EXECUTION_OBSERVED_{exec_date}.json"
        exec_manifest_file = target_dir / f"EXECUTION_OBSERVED_{exec_date}.manifest.json"
        if exec_file.exists():
            existing_exec_bytes = exec_file.read_bytes()
            existing_exec_sha = hashlib.sha256(existing_exec_bytes).hexdigest()
            try:
                existing_exec_data = json.loads(existing_exec_bytes.decode("utf-8"))
            except Exception:
                existing_exec_data = {}

            is_same_exec = (
                existing_exec_data.get("signal_date") == sig_date
                and existing_exec_data.get("execution_date") == exec_date
                and existing_exec_data.get("model_id") == model_id
                and existing_exec_data.get("sealed_signal_sha256") == calc_sha
            )
            if is_same_exec:
                logger.info(f"执行制品已存在且内容一致，保持幂等 (IDEMPOTENT_ALREADY_EXECUTED): {exec_file.name}")
                return {
                    "event": "IDEMPOTENT_ALREADY_EXECUTED",
                    "signal_date": sig_date,
                    "execution_date": exec_date,
                    "model_id": model_id,
                    "file_sha256": existing_exec_sha,
                    "observed_trading_days_added": 0,
                    "status": "COMPLETED",
                    "details": existing_exec_data.get("details", {}),
                }
            else:
                raise ImmutableArtifactConflict(
                    f"Fail-Closed: 执行制品已存在且内容冲突，禁止覆盖已执行制品: {exec_file.name}"
                )

        # 物理验证 T+1 当天真实行情数据
        if quotes_df is None or len(quotes_df) == 0:
            raise ValueError(f"Fail-Closed: 执行日 ({exec_date}) 真实行情数据为空，无法撮合！")
        if "symbol" not in quotes_df.columns:
            raise ValueError("Fail-Closed: 真实行情数据缺少必要列 'symbol'")

        # 行情数据必须完整包含 date, symbol, open, close (Fail-Closed)
        for req_q_col in ["symbol", "date", "open", "close"]:
            if req_q_col not in quotes_df.columns:
                raise ValueError(f"Fail-Closed: 真实行情数据缺少必要列 '{req_q_col}'！")

        price_col = "open"
        if "date" in quotes_df.columns:
            mismatches = quotes_df[quotes_df["date"].dropna().astype(str).str[:10] != exec_date]
            if not mismatches.empty or quotes_df["date"].isna().any():
                raise ValueError(f"Fail-Closed: 行情数据中存在与执行日 ({exec_date}) 不匹配或缺失的日期记录！")

        valid_quotes = quotes_df[quotes_df[price_col] > 0]
        if valid_quotes.empty:
            raise ValueError(f"Fail-Closed: 执行日 ({exec_date}) 真实行情数据无任何有效正价格！")

        # 建立行情速查字典
        quote_map = {}
        for _, qr in quotes_df.iterrows():
            s = str(qr["symbol"])
            p = float(qr[price_col])
            if p <= 0:
                continue
            quote_map[s] = {
                "symbol": s,
                "price": p,
                "is_suspended": bool(qr.get("is_suspended", False)),
                "is_limit_up_locked": bool(qr.get("is_limit_up_locked", False)),
                "is_limit_down_locked": bool(qr.get("is_limit_down_locked", False)),
                "amount": float(qr.get("amount", 1e8)),
            }

        # 检查选股清单中缺失行情的标的 (任一标的缺失行情，全天抛错拒绝撮合，Fail-Closed)
        missing_quote_symbols = []
        tradable_picks = []
        for p in picks_records:
            sym = str(p["symbol"])
            q_info = quote_map.get(sym)
            if not q_info:
                missing_quote_symbols.append(sym)
                logger.warning(f"标的 {sym} 缺少 T+1 ({exec_date}) 真实开盘行情，拒绝撮合 (REJECTED_MISSING_QUOTE)")
                continue
            if q_info["is_suspended"] or q_info["is_limit_up_locked"]:
                continue
            p_copy = dict(p)
            p_copy["close"] = q_info["price"]
            tradable_picks.append(p_copy)

        if missing_quote_symbols:
            raise ValueError(
                f"Fail-Closed: 选股标的 {missing_quote_symbols} 缺少执行日 ({exec_date}) 真实行情数据，全天拒绝撮合！"
            )

        exec_details = {
            "rejected_missing_quotes": missing_quote_symbols
        }

        # 撮合更新账本 (必须至少提供一个账本进行物理撮合与增量记录)
        if paper_ledger is None and shadow_ledger is None:
            raise ValueError("Fail-Closed: execute_observed 必须至少传入一个账本 (paper_ledger 或 shadow_ledger) 进行物理撮合！")

        # 记录撮合前账本天数
        init_paper_days = paper_ledger.observed_trading_days if paper_ledger is not None else 0
        init_shadow_days = shadow_ledger.observed_trading_days if shadow_ledger is not None else 0

        # 撮合 PaperTradingLedger (严禁 allow_historical)
        if paper_ledger is not None:
            target_df = (
                pd.DataFrame(tradable_picks)
                if tradable_picks
                else pd.DataFrame(columns=["symbol", "target_weight", "close"])
            )
            price_dict = {s: d["price"] for s, d in quote_map.items()}

            reb_res = paper_ledger.rebalance(
                target_df,
                current_prices=price_dict,
                trade_date=exec_date,
                signal_date=sig_date,
                allow_historical=False,
            )
            final_p_days = paper_ledger.observed_trading_days
            exec_details["paper"] = {
                "trades_count": reb_res.get("total_trades", 0),
                "trades": reb_res.get("trades", []),
                "nav": reb_res.get("nav", 1.0),
                "observed_trading_days": final_p_days,
                "observed_trading_days_added": max(0, final_p_days - init_paper_days),
            }

        # 撮合 ShadowTradingLedger (严禁 allow_historical)
        if shadow_ledger is not None:
            picks_df = pd.DataFrame(picks_records)
            obs_res = shadow_ledger.record_shadow_observation(
                target_df=picks_df,
                model_id=model_id,
                data_date=exec_date,
                market_df=quotes_df,
                signal_date=sig_date,
                allow_historical=False,
            )
            final_s_days = shadow_ledger.observed_trading_days
            exec_details["shadow"] = {
                "evidence_sha256": obs_res.get("evidence_sha256"),
                "shadow_fills_count": obs_res.get("shadow_fills_count", 0),
                "observed_trading_days": final_s_days,
                "observed_trading_days_added": max(0, final_s_days - init_shadow_days),
            }

        final_paper_days = paper_ledger.observed_trading_days if paper_ledger is not None else 0
        final_shadow_days = shadow_ledger.observed_trading_days if shadow_ledger is not None else 0
        paper_delta = max(0, final_paper_days - init_paper_days)
        shadow_delta = max(0, final_shadow_days - init_shadow_days)

        # 观察天数增量严格依据物理账本真实增加，禁止任何强制 +1 兜底；缺失行情时增量恒为 0
        if missing_quote_symbols:
            days_added = 0
        else:
            days_added = max(paper_delta, shadow_delta)

        # 生成不可篡改的执行结果制品与 manifest
        exec_payload = {
            "event": "EXECUTION_OBSERVED",
            "signal_date": sig_date,
            "execution_date": exec_date,
            "model_id": model_id,
            "sealed_signal_sha256": calc_sha,
            "observed_trading_days_added": days_added,
            "executed_at": datetime.now().isoformat(),
            "status": "COMPLETED",
            "details": exec_details,
        }

        exec_bytes = json.dumps(exec_payload, indent=2, ensure_ascii=False).encode("utf-8")
        exec_sha = hashlib.sha256(exec_bytes).hexdigest()

        exec_file.write_bytes(exec_bytes)
        exec_manifest = {
            "file_name": exec_file.name,
            "file_sha256": exec_sha,
            "signal_date": sig_date,
            "execution_date": exec_date,
            "sealed_signal_sha256": calc_sha,
            "executed_at": exec_payload["executed_at"],
            "observed_trading_days_added": days_added,
        }
        exec_manifest_file.write_text(json.dumps(exec_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"T+1执行与观察已记录: {exec_file.name} (SHA: {exec_sha[:8]}, 观察天数+{days_added})")
        exec_payload["file_sha256"] = exec_sha
        return exec_payload


# 顶层便捷导出函数
seal_signal = ProspectiveStateMachine.seal_signal
execute_observed = ProspectiveStateMachine.execute_observed
