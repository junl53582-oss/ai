"""
实盘模拟跟踪对账账本引擎 (execution/paper_ledger.py)
用于支持 Direction 2:
1. 真实仿真记录从当前截面开始的实盘模拟账户 (初始本金 1,000,000 元)
2. 遵循 A 股 T+1 规则、100 股整手向下取整约束与印花税/佣金真实摩擦
3. 记录持仓明细、浮动盈亏、已实现盈亏、每日净值序列与成交交割单
4. 持久化存储至 data_storage/paper_trading_ledger.json
"""

import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np

from config.settings import settings

logger = logging.getLogger("PaperLedger")


from data.trading_calendar import CanonicalTradingCalendar, TradeDateError
from execution.prospective_state_machine import ProspectiveStateMachine, seal_signal, execute_observed


class PaperTradingLedger:
    """实盘模拟跟踪对账账本管理类"""

    DEFAULT_INITIAL_CASH = 1_000_000.0  # 100 万元初始虚拟本金
    COMMISSION_RATE = 0.00025  # 佣金万分之 2.5 (最低 5 元)
    STAMP_DUTY_RATE = 0.0005   # 印花税千分之 0.5 (仅卖出收取)

    def __init__(self, ledger_file: Optional[Path] = None, initial_cash: float = DEFAULT_INITIAL_CASH):
        self.ledger_file = ledger_file or (settings.DATA_DIR / "paper_trading_ledger.json")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.daily_nav_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.realized_pnl: float = 0.0
        self.created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at: str = self.created_at

        if self.ledger_file.exists():
            self.load()
        else:
            self._init_fresh_ledger()

    def _init_fresh_ledger(self):
        """初始化空白账本 (严禁自动快照，杜绝非交易日初始化污染账本)"""
        self.cash = self.initial_cash
        self.positions = {}
        self.realized_pnl = 0.0
        self.daily_nav_history = []
        self.trade_history = []
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at = self.created_at
        self.save()

    @property
    def observed_trading_days(self) -> int:
        """真实有效交易日快照天数 (排除非交易日与被隔离/失效快照)"""
        from data.trading_calendar import CanonicalTradingCalendar
        cal = CanonicalTradingCalendar.get_instance()
        valid_dates = set()
        for snap in self.daily_nav_history:
            d = snap.get("date")
            if not d:
                continue
            if snap.get("status") == "INVALID_NON_TRADING_DAY" or snap.get("excluded_from_evidence"):
                continue
            if cal.is_trading_day(d):
                valid_dates.add(d)
        return len(valid_dates)

    def save(self):
        """持久化保存账本至 JSON 文件"""
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        days = self.observed_trading_days
        data = {
            "account_id": "PAPER_SIM_01",
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "positions": self.positions,
            "daily_nav_history": self.daily_nav_history,
            "trade_history": self.trade_history[-200:],  # 保留最近 200 笔交割单
            "observed_trading_days": days,
            "evidence_maturity": "MATURE" if days >= 20 else ("EARLY" if days >= 5 else "IMMATURE"),
            "status": "NOT_STARTED" if days == 0 else "OBSERVING",
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"模拟账本已更新落盘: {self.ledger_file}")

    def validate_ledger_integrity(self, strict: bool = True) -> List[str]:
        """
        核验当前账本记录的合法性 (Fail-Closed 审计强化):
        1. 严禁包含在非交易所交易日执行的买卖成交记录
        2. 严禁未排除的非交易日资产快照
        """
        from data.trading_calendar import CanonicalTradingCalendar
        cal = CanonicalTradingCalendar.get_instance()
        violations = []

        for trade in self.trade_history:
            ts = trade.get("timestamp") or trade.get("date") or ""
            t_date = str(ts)[:10]
            if t_date and not cal.is_trading_day(t_date):
                violations.append(f"Trade {trade.get('trade_id', 'UNKNOWN')} executed on non-trading day: {t_date}")

        for snap in self.daily_nav_history:
            s_date = snap.get("date")
            is_excluded = snap.get("excluded_from_evidence", False) or snap.get("status") == "INVALID_NON_TRADING_DAY"
            if s_date and not is_excluded and not cal.is_trading_day(s_date):
                violations.append(f"Unexcluded NAV snapshot on non-trading day: {s_date}")

        if violations and strict:
            raise TradeDateError(f"FATAL: Ledger integrity validation failed with {len(violations)} non-trading day violations: {violations}")

        return violations

    def load(self, strict_validate: bool = True):
        """从 JSON 文件加载账本并强制校验真实交易日合法性"""
        try:
            with open(self.ledger_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.initial_cash = float(data.get("initial_cash", self.DEFAULT_INITIAL_CASH))
            self.cash = float(data.get("cash", self.initial_cash))
            self.realized_pnl = float(data.get("realized_pnl", 0.0))
            self.positions = data.get("positions", {})
            self.daily_nav_history = data.get("daily_nav_history", [])
            self.trade_history = data.get("trade_history", [])
            self.created_at = data.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.updated_at = data.get("updated_at", self.created_at)
        except TradeDateError:
            raise
        except Exception as e:
            logger.error(f"读取模拟盘账本失败 ({e})，重新初始化")
            self._init_fresh_ledger()
            return

        if strict_validate:
            self.validate_ledger_integrity(strict=True)

    @classmethod
    def quarantine_contaminated_ledger(
        cls,
        src_ledger_file: Union[str, Path],
        quarantine_dir: Union[str, Path],
        reason: str
    ) -> Dict[str, Any]:
        """将受污染的账本物理隔离至指定目录并生成防伪审计清单"""
        import hashlib
        quarantine_dir = Path(quarantine_dir)
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        src_path = Path(src_ledger_file)
        if not src_path.exists():
            raise FileNotFoundError(f"Source ledger not found: {src_path}")

        content_bytes = src_path.read_bytes()
        file_sha256 = hashlib.sha256(content_bytes).hexdigest()
        raw_data = json.loads(content_bytes.decode("utf-8"))

        from data.trading_calendar import CanonicalTradingCalendar
        cal = CanonicalTradingCalendar.get_instance()
        invalid_dates = set()
        invalid_ids = []
        for t in raw_data.get("trade_history", []):
            d = str(t.get("timestamp") or t.get("date"))[:10]
            if not cal.is_trading_day(d):
                invalid_dates.add(d)
                invalid_ids.append(t.get("trade_id"))

        quarantine_name = f"paper_ledger_invalid_weekend_{min(invalid_dates).replace('-', '') if invalid_dates else 'quarantine'}"
        target_json = quarantine_dir / f"{quarantine_name}.json"
        target_manifest = quarantine_dir / f"{quarantine_name}.manifest.json"

        target_json.write_bytes(content_bytes)
        manifest = {
            "original_path": str(src_path),
            "original_sha256": file_sha256,
            "invalid_trade_dates": sorted(list(invalid_dates)),
            "invalid_trade_ids": invalid_ids,
            "reason": reason,
            "quarantined_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "excluded_from_all_evidence": True
        }
        target_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest


    @property
    def market_value(self) -> float:
        """持仓总市值"""
        return sum(float(pos.get("market_value", 0.0)) for pos in self.positions.values())

    @property
    def total_equity(self) -> float:
        """账户总资产 (现金 + 市值)"""
        return self.cash + self.market_value

    @property
    def nav(self) -> float:
        """累计单位净值 (起点 1.0)"""
        return self.total_equity / self.initial_cash

    @property
    def cum_return_pct(self) -> float:
        """累计收益率百分比"""
        return (self.nav - 1.0) * 100.0

    def update_quotes(self, quotes: Union[pd.DataFrame, Dict[str, float]]):
        """用最新行情价格刷新所有持仓标的市值与浮动盈亏"""
        if isinstance(quotes, pd.DataFrame):
            price_map = {}
            for _, r in quotes.iterrows():
                sym = str(r["symbol"])
                p = float(r.get("close", r.get("price", 0.0)))
                if p > 0:
                    price_map[sym] = p
        else:
            price_map = quotes

        for sym, pos in self.positions.items():
            if sym in price_map and price_map[sym] > 0:
                cur_p = price_map[sym]
                pos["current_price"] = cur_p
                pos["market_value"] = round(cur_p * pos["total_shares"], 2)
                cost_val = pos["avg_cost"] * pos["total_shares"]
                pos["unrealized_pnl"] = round(pos["market_value"] - cost_val, 2)
                pos["unrealized_pnl_pct"] = round((cur_p / pos["avg_cost"] - 1.0) * 100.0, 2) if pos["avg_cost"] > 0 else 0.0

        self.save()

    def record_daily_snapshot(self, date_str: Optional[str] = None):
        """记录当日收盘后的账户资产快照 (必须显式指定交易所合法交易日)"""
        if not date_str:
            raise TradeDateError("必须显式指定 trade_date，严禁隐式自动记录非交易日快照")

        date_str = str(date_str).strip()[:10]
        from data.trading_calendar import CanonicalTradingCalendar
        cal = CanonicalTradingCalendar.get_instance()
        if not cal.is_trading_day(date_str):
            raise TradeDateError(f"非交易所交易日严禁记录资产净值快照: {date_str} (fail-closed)")

        tot_eq = round(self.total_equity, 2)
        cur_nav = round(self.nav, 4)
        cum_ret = round(self.cum_return_pct, 2)

        prev_nav = 1.0
        if self.daily_nav_history:
            prev_nav = self.daily_nav_history[-1].get("nav", 1.0)
        daily_ret = round((cur_nav / prev_nav - 1.0) * 100.0, 2) if prev_nav > 0 else 0.0

        existing_idx = None
        for i, snap in enumerate(self.daily_nav_history):
            if snap.get("date") == date_str:
                existing_idx = i
                break

        snapshot = {
            "date": date_str,
            "total_equity": tot_eq,
            "cash": round(self.cash, 2),
            "market_value": round(self.market_value, 2),
            "nav": cur_nav,
            "daily_return_pct": daily_ret,
            "cum_return_pct": cum_ret,
            "position_ratio_pct": round((self.market_value / tot_eq) * 100.0, 1) if tot_eq > 0 else 0.0
        }

        if existing_idx is not None:
            self.daily_nav_history[existing_idx] = snapshot
        else:
            self.daily_nav_history.append(snapshot)

        self.save()

    def rebalance(
        self,
        target_df: pd.DataFrame,
        current_prices: Optional[Dict[str, float]] = None,
        trade_date: Optional[str] = None,
        signal_date: Optional[str] = None,
        allow_historical: bool = True
    ) -> Dict[str, Any]:
        """
        根据目标投资组合清单执行仿真调仓撮合:
        1. 整手约束: 严格按 100 股向下取整
        2. 先卖后买: 释放可用资金后再建仓
        3. 交易规费: 万 2.5 佣金 (最低 5 元) + 千 0.5 卖出印花税
        4. 交易日硬门禁: 必须在合法交易日调仓并快照
        5. T+1 相邻交易日约束与严禁旧信号跨期执行
        """
        from data.trading_calendar import CanonicalTradingCalendar
        cal = CanonicalTradingCalendar.get_instance()
        if trade_date:
            trade_date_str = str(trade_date).strip()[:10]
        else:
            trade_date_str = datetime.now().strftime("%Y-%m-%d")

        if signal_date:
            sig_str = str(signal_date).strip()[:10]
            if sig_str <= "2026-08-24" and trade_date_str >= "2026-09-01":
                raise ValueError("Fail-Closed: 严禁把 2026-08-24 的旧信号记成 2026-09 月的新 Paper/Shadow 样本！")
            expected_exec = cal.next_trading_day(sig_str)
            if expected_exec != trade_date_str:
                raise TradeDateError(f"Fail-Closed: 模拟调仓违反 A 股 T+1 规则! 信号日 {sig_str} 的合法次一交易日必须是 {expected_exec}, 实际传入 {trade_date_str}")

        if not cal.is_trading_day(trade_date_str):
            raise TradeDateError(f"非交易所交易日禁止执行调仓仿真: {trade_date_str} (fail-closed)")

        if not allow_historical and trade_date_str <= "2026-08-24":
            raise ValueError(f"Fail-Closed: 禁止回填历史日期 ({trade_date_str} <= 2026-08-24) 冒充实盘模拟交易！")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target_symbols = set()
        target_map: Dict[str, Dict[str, Any]] = {}

        if not target_df.empty:
            for _, r in target_df.iterrows():
                sym = str(r["symbol"])
                w = float(r.get("target_weight", 0.0))
                if w > 0:
                    target_symbols.add(sym)
                    target_map[sym] = {
                        "name": str(r.get("name", sym)),
                        "weight": w,
                        "price": float(r.get("close", 0.0))
                    }

        prices: Dict[str, float] = {}
        if current_prices:
            prices.update(current_prices)
        for sym, item in target_map.items():
            if sym not in prices and item["price"] > 0:
                prices[sym] = item["price"]
        for sym, pos in self.positions.items():
            if sym not in prices and pos.get("current_price", 0.0) > 0:
                prices[sym] = pos["current_price"]

        trades_executed = []
        equity_base = self.total_equity
        investable_equity = equity_base * 0.95  # 预留 5% 现金防御与滑点缓冲

        # 步骤 1: 先卖出
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            cur_p = prices.get(sym, pos.get("current_price", 0.0))
            if cur_p <= 0:
                continue

            cur_shares = pos["total_shares"]
            if sym not in target_symbols:
                sell_shares = cur_shares
            else:
                tgt_w = target_map[sym]["weight"]
                tgt_value = investable_equity * tgt_w
                tgt_shares = int(tgt_value / (cur_p * 100)) * 100
                sell_shares = max(0, cur_shares - tgt_shares)

            if sell_shares >= 100:
                gross_amt = sell_shares * cur_p
                comm = max(5.0, gross_amt * self.COMMISSION_RATE)
                stamp = gross_amt * self.STAMP_DUTY_RATE
                net_amt = gross_amt - comm - stamp

                cost_basis = sell_shares * pos["avg_cost"]
                realized = net_amt - cost_basis
                self.realized_pnl += realized

                self.cash += net_amt
                pos["total_shares"] -= sell_shares
                pos["market_value"] = round(pos["total_shares"] * cur_p, 2)

                trade_record = {
                    "trade_id": str(uuid.uuid4())[:8],
                    "timestamp": now_str,
                    "action": "SELL",
                    "symbol": sym,
                    "name": pos["name"],
                    "shares": sell_shares,
                    "price": cur_p,
                    "amount": round(gross_amt, 2),
                    "fee": round(comm + stamp, 2),
                    "realized_pnl": round(realized, 2)
                }
                self.trade_history.append(trade_record)
                trades_executed.append(trade_record)

                if pos["total_shares"] <= 0:
                    del self.positions[sym]

        # 步骤 2: 后买入
        for sym, item in target_map.items():
            cur_p = prices.get(sym, item.get("price", 0.0))
            if cur_p <= 0:
                continue

            tgt_w = item["weight"]
            tgt_value = investable_equity * tgt_w
            tgt_shares = int(tgt_value / (cur_p * 100)) * 100

            cur_shares = self.positions.get(sym, {}).get("total_shares", 0)
            buy_shares = max(0, tgt_shares - cur_shares)

            if buy_shares >= 100:
                gross_amt = buy_shares * cur_p
                comm = max(5.0, gross_amt * self.COMMISSION_RATE)
                total_required = gross_amt + comm

                if self.cash >= total_required:
                    self.cash -= total_required
                    if sym not in self.positions:
                        self.positions[sym] = {
                            "symbol": sym,
                            "name": item["name"],
                            "total_shares": buy_shares,
                            "available_shares": buy_shares,
                            "avg_cost": round(total_required / buy_shares, 2),
                            "current_price": cur_p,
                            "market_value": round(buy_shares * cur_p, 2),
                            "unrealized_pnl": 0.0,
                            "unrealized_pnl_pct": 0.0
                        }
                    else:
                        old_pos = self.positions[sym]
                        old_val = old_pos["total_shares"] * old_pos["avg_cost"]
                        new_tot_shares = old_pos["total_shares"] + buy_shares
                        old_pos["avg_cost"] = round((old_val + total_required) / new_tot_shares, 2)
                        old_pos["total_shares"] = new_tot_shares
                        old_pos["available_shares"] = new_tot_shares
                        old_pos["market_value"] = round(new_tot_shares * cur_p, 2)

                    trade_record = {
                        "trade_id": str(uuid.uuid4())[:8],
                        "timestamp": now_str,
                        "action": "BUY",
                        "symbol": sym,
                        "name": item["name"],
                        "shares": buy_shares,
                        "price": cur_p,
                        "amount": round(gross_amt, 2),
                        "fee": round(comm, 2),
                        "realized_pnl": 0.0
                    }
                    self.trade_history.append(trade_record)
                    trades_executed.append(trade_record)

        self.update_quotes(prices)
        self.record_daily_snapshot(trade_date_str)
        self.save()

        return {
            "timestamp": now_str,
            "total_trades": len(trades_executed),
            "trades": trades_executed,
            "cash": round(self.cash, 2),
            "market_value": round(self.market_value, 2),
            "total_equity": round(self.total_equity, 2),
            "nav": round(self.nav, 4)
        }

    def get_summary(self) -> Dict[str, Any]:
        """获取供仪表盘前端渲染的账本全景摘要"""
        pos_list = []
        for sym, p in self.positions.items():
            pos_list.append({
                "symbol": sym,
                "name": p.get("name", sym),
                "total_shares": p.get("total_shares", 0),
                "avg_cost": p.get("avg_cost", 0.0),
                "current_price": p.get("current_price", 0.0),
                "market_value": p.get("market_value", 0.0),
                "unrealized_pnl": p.get("unrealized_pnl", 0.0),
                "unrealized_pnl_pct": p.get("unrealized_pnl_pct", 0.0),
                "weight_pct": round((p.get("market_value", 0.0) / self.total_equity) * 100.0, 1) if self.total_equity > 0 else 0.0
            })

        pos_df = pd.DataFrame(pos_list) if pos_list else pd.DataFrame(columns=[
            "symbol", "name", "total_shares", "avg_cost", "current_price", "market_value", "unrealized_pnl", "unrealized_pnl_pct", "weight_pct"
        ])

        nav_df = pd.DataFrame(self.daily_nav_history) if self.daily_nav_history else pd.DataFrame(columns=[
            "date", "total_equity", "cash", "market_value", "nav", "daily_return_pct", "cum_return_pct", "position_ratio_pct"
        ])

        trades_df = pd.DataFrame(self.trade_history[::-1]) if self.trade_history else pd.DataFrame(columns=[
            "trade_id", "timestamp", "action", "symbol", "name", "shares", "price", "amount", "fee", "realized_pnl"
        ])

        tot_eq = self.total_equity
        pos_ratio = (self.market_value / tot_eq * 100.0) if tot_eq > 0 else 0.0
        days = self.observed_trading_days

        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "market_value": self.market_value,
            "total_equity": tot_eq,
            "realized_pnl": self.realized_pnl,
            "cum_return_pct": self.cum_return_pct,
            "nav": self.nav,
            "position_ratio_pct": pos_ratio,
            "positions_count": len(self.positions),
            "observed_trading_days": days,
            "evidence_maturity": "MATURE" if days >= 20 else ("EARLY" if days >= 5 else "IMMATURE"),
            "status": "NOT_STARTED" if days == 0 else "OBSERVING",
            "positions_df": pos_df,
            "nav_df": nav_df,
            "trades_df": trades_df,
            "updated_at": self.updated_at
        }


class ShadowTradingLedger:
    """
    影子观察对账账本引擎 (Shadow Trading Ledger)
    职责:
    1. 独立于 Paper Trading 分账存储 (data_storage/shadow_trading_ledger.json)
    2. 记录真实可成交性 (True Feasibility) 与影子撮合观察，绝不下单
    3. 严格记录: model_id, signals, target_weights, shadow_fills, costs, slippage, NAV, data_date, evidence_sha256
    4. 不伪造历史记录，样本天数不足 20 天时必须保持 IMMATURE/NOT_STARTED
    """
    DEFAULT_INITIAL_CASH = 1_000_000.0
    COMMISSION_RATE = 0.00025
    STAMP_DUTY_RATE = 0.0005
    BASE_SLIPPAGE_BPS = 5.0

    def __init__(self, ledger_file: Optional[Path] = None, initial_cash: float = DEFAULT_INITIAL_CASH):
        self.ledger_file = ledger_file or (settings.DATA_DIR / "shadow_trading_ledger.json")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.daily_observations: List[Dict[str, Any]] = []
        self.daily_nav_history: List[Dict[str, Any]] = []
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at: str = self.created_at

        if self.ledger_file.exists():
            self.load()
        else:
            self._init_fresh_ledger()

    def _init_fresh_ledger(self):
        self.cash = self.initial_cash
        self.positions = {}
        self.daily_observations = []
        self.daily_nav_history = []
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at = self.created_at
        self.save()

    def load(self):
        try:
            with open(self.ledger_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.initial_cash = float(data.get("initial_cash", self.DEFAULT_INITIAL_CASH))
            self.cash = float(data.get("cash", self.initial_cash))
            self.positions = data.get("positions", {})
            self.daily_observations = data.get("daily_observations", [])
            self.daily_nav_history = data.get("daily_nav_history", [])
            self.created_at = data.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.updated_at = data.get("updated_at", self.created_at)
        except Exception as e:
            logger.error(f"读取影子账本失败 ({e})，重新初始化")
            self._init_fresh_ledger()

    @property
    def observed_trading_days(self) -> int:
        """真实有效交易日快照天数 (排除非交易日与被隔离/失效快照)"""
        from data.trading_calendar import CanonicalTradingCalendar
        cal = CanonicalTradingCalendar.get_instance()
        valid_dates = set()
        for snap in self.daily_nav_history:
            d = snap.get("date")
            if not d:
                continue
            if snap.get("status") == "INVALID_NON_TRADING_DAY" or snap.get("excluded_from_evidence"):
                continue
            if cal.is_trading_day(d):
                valid_dates.add(d)
        return len(valid_dates)

    def save(self):
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        observed_days = self.observed_trading_days
        if observed_days >= 60:
            maturity = "MATURE"
        elif observed_days >= 20:
            maturity = "EARLY"
        else:
            maturity = "IMMATURE"

        data = {
            "account_id": "SHADOW_OBS_01",
            "account_type": "SHADOW",
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 2),
            "positions": self.positions,
            "daily_observations": self.daily_observations[-200:],
            "daily_nav_history": self.daily_nav_history,
            "observed_trading_days": observed_days,
            "evidence_maturity": maturity,
            "status": "NOT_STARTED" if observed_days == 0 else "OBSERVING",
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"影子观察账本已更新落盘: {self.ledger_file}")

    @property
    def total_equity(self) -> float:
        market_val = sum(float(p.get("market_value", 0.0)) for p in self.positions.values())
        return self.cash + market_val

    @property
    def nav(self) -> float:
        return self.total_equity / self.initial_cash if self.initial_cash > 0 else 1.0

    @property
    def evidence_maturity(self) -> str:
        observed_days = self.observed_trading_days
        if observed_days >= 60:
            return "MATURE"
        elif observed_days >= 20:
            return "EARLY"
        return "IMMATURE"

    @property
    def status(self) -> str:
        return "NOT_STARTED" if self.observed_trading_days == 0 else "OBSERVING"

    def record_shadow_observation(
        self,
        target_df: pd.DataFrame,
        model_id: str,
        data_date: str,
        market_df: Optional[pd.DataFrame] = None,
        signal_date: Optional[str] = None,
        allow_historical: bool = True
    ) -> Dict[str, Any]:
        """
        执行影子观察记录:
        评估真实可成交性 (流动性、涨跌停锁死、停牌状态、滑点与规费成本)，记录影子撮合而不实际向任何经纪端发单
        """
        import hashlib
        from data.trading_calendar import CanonicalTradingCalendar
        cal = CanonicalTradingCalendar.get_instance()
        data_date_str = str(data_date).strip()[:10]

        if signal_date:
            sig_str = str(signal_date).strip()[:10]
            if sig_str <= "2026-08-24" and data_date_str >= "2026-09-01":
                raise ValueError("Fail-Closed: 严禁把 2026-08-24 的旧信号记成 2026-09 月的新 Paper/Shadow 样本！")
            expected_exec = cal.next_trading_day(sig_str)
            if expected_exec != data_date_str:
                raise TradeDateError(f"Fail-Closed: 影子观察违反 T+1 规则! 信号日 {sig_str} 的合法次一交易日必须是 {expected_exec}, 实际传入 {data_date_str}")

        if not cal.is_trading_day(data_date_str):
            raise TradeDateError(f"非交易所交易日禁止记录影子观察: {data_date_str} (fail-closed)")

        if not allow_historical and data_date_str <= "2026-08-24":
            raise ValueError(f"Fail-Closed: 禁止回填历史日期 ({data_date_str} <= 2026-08-24) 冒充影子观察样本！")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tot_eq = self.total_equity

        # 检查是否已存在当日记录 (幂等性)
        for obs in self.daily_observations:
            if obs.get("date") == data_date and obs.get("model_id") == model_id:
                logger.info(f"影子账本已存在 {data_date} / {model_id} 记录，保持幂等")
                return obs

        shadow_fills = []
        total_slippage_cost = 0.0
        total_fee_cost = 0.0

        market_info_map = {}
        if market_df is not None and not market_df.empty:
            for _, mr in market_df.iterrows():
                market_info_map[str(mr["symbol"])] = mr

        for _, r in target_df.iterrows():
            sym = str(r["symbol"])
            w = float(r.get("target_weight", 0.0))
            price = float(r.get("close", 10.0))
            m_row = market_info_map.get(sym)

            # 可成交性审查
            tradable = True
            untradable_reason = ""
            slippage_bps = self.BASE_SLIPPAGE_BPS

            if m_row is not None:
                if bool(m_row.get("is_suspended", False)):
                    tradable = False
                    untradable_reason = "SUSPENDED"
                elif bool(m_row.get("is_limit_up_locked", False)):
                    tradable = False
                    untradable_reason = "LIMIT_UP_LOCKED"
                elif float(m_row.get("amount", 1e8)) < 1e7:
                    slippage_bps = 20.0

            target_val = tot_eq * w
            target_shares = int(target_val / (price * 100)) * 100 if price > 0 else 0
            shadow_shares = target_shares if tradable else 0

            slip_cost = (shadow_shares * price) * (slippage_bps / 10000.0) if tradable else 0.0
            fee_cost = max((shadow_shares * price) * self.COMMISSION_RATE, 5.0) if tradable and shadow_shares > 0 else 0.0

            total_slippage_cost += slip_cost
            total_fee_cost += fee_cost

            shadow_fills.append({
                "symbol": sym,
                "target_weight": w,
                "target_shares": target_shares,
                "shadow_fill_shares": shadow_shares,
                "price": price,
                "tradable": tradable,
                "untradable_reason": untradable_reason,
                "slippage_bps": slippage_bps,
                "slippage_cost": round(slip_cost, 2),
                "fee_cost": round(fee_cost, 2)
            })

        # 密码级证据哈希
        obs_payload = {
            "date": data_date,
            "model_id": model_id,
            "target_weights": {str(r["symbol"]): float(r.get("target_weight", 0.0)) for _, r in target_df.iterrows()},
            "fills": shadow_fills,
            "costs": {"fee": round(total_fee_cost, 2), "slippage": round(total_slippage_cost, 2)},
            "nav": round(self.nav, 4)
        }
        evidence_sha256 = hashlib.sha256(json.dumps(obs_payload, sort_keys=True).encode("utf-8")).hexdigest()

        obs_record = {
            "date": data_date,
            "timestamp": now_str,
            "model_id": model_id,
            "signal_count": len(target_df),
            "shadow_fills_count": sum(1 for f in shadow_fills if f["tradable"]),
            "shadow_fills": shadow_fills,
            "total_fee_cost": round(total_fee_cost, 2),
            "total_slippage_cost": round(total_slippage_cost, 2),
            "nav": round(self.nav, 4),
            "evidence_sha256": evidence_sha256
        }
        self.daily_observations.append(obs_record)

        # 记录 NAV 快照
        nav_entry = {
            "date": data_date,
            "total_equity": round(tot_eq, 2),
            "nav": round(self.nav, 4),
            "evidence_sha256": evidence_sha256
        }
        if not any(n["date"] == data_date for n in self.daily_nav_history):
            self.daily_nav_history.append(nav_entry)

        self.save()
        return obs_record

    def get_summary(self) -> Dict[str, Any]:
        observed_days = self.observed_trading_days
        if observed_days >= 60:
            maturity = "MATURE"
        elif observed_days >= 20:
            maturity = "EARLY"
        else:
            maturity = "IMMATURE"

        return {
            "account_id": "SHADOW_OBS_01",
            "account_type": "SHADOW",
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "total_equity": self.total_equity,
            "nav": self.nav,
            "observed_trading_days": observed_days,
            "evidence_maturity": maturity,
            "status": "NOT_STARTED" if observed_days == 0 else "OBSERVING",
            "daily_observations_count": len(self.daily_observations),
            "updated_at": self.updated_at
        }
