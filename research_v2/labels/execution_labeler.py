"""Execution-aligned forward label generator for Phase 2.1-A.

Signal timing:
    T close -> enter at T+1 open -> planned exit at T+21 open.

The generator deliberately separates:
- raw open: tradability / execution-state checks,
- adjusted open: return calculation across corporate actions,
- benchmark open: excess-return benchmark,
- planned exit vs actual executable exit: limit-down / suspension deferral.

It never forward-fills missing execution states on its own. If the source dataset
contains ffilled prices, the explicit suspension/volume/limit-lock fields still
gate label validity.

Research Integrity Hardened:
- Max Exit Defer Trading Days Policy (MAX_EXIT_DEFER_TRADING_DAYS fail-closed: EXIT_DEFER_EXCEEDS_POLICY)
- Require Canonical Calendar gate in certified mode
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from config.settings import settings
from research_v2.labels.schema import ExecutionAlignedLabelSchema
from strategy.trading_rules import TradingFeeSchedule


@dataclass(frozen=True)
class ExecutionLabelColumns:
    gross_alpha: str = "label_gross_alpha_20d"
    net_alpha: str = "label_net_alpha_20d"
    direction: str = "label_direction_20d"
    rank: str = "label_rank_20d"
    downside: str = "label_downside_20d"


class ExecutionAlignedLabeler:
    """Build executable T+1-open -> T+21-open labels without mutating legacy labels."""

    REQUIRED_COLUMNS = {
        "date", "symbol", "open", "adj_open", "benchmark_open", "volume",
        "is_suspended", "is_limit_up_locked", "is_limit_down_locked",
    }

    def __init__(
        self,
        schema: Optional[ExecutionAlignedLabelSchema] = None,
        threshold_mode: Optional[str] = None,
        threshold: Optional[float] = None,
        extreme_quantile: Optional[float] = None,
        commission_rate: Optional[float] = None,
        slippage_rate: Optional[float] = None,
        max_exit_defer_trading_days: Optional[int] = 5,
        require_canonical_calendar: bool = False
    ) -> None:
        self.schema = schema or ExecutionAlignedLabelSchema()
        self.schema.validate()
        self.threshold_mode = threshold_mode or settings.LABEL_THRESHOLD_MODE
        self.threshold = settings.LABEL_THRESHOLD if threshold is None else float(threshold)
        self.extreme_quantile = float(getattr(settings, "LABEL_EXTREME_QUANTILE", 0.30)) if extreme_quantile is None else float(extreme_quantile)
        self.commission_rate = float(settings.COMMISSION_RATE) if commission_rate is None else float(commission_rate)
        self.slippage_rate = float(settings.SLIPPAGE_RATE) if slippage_rate is None else float(slippage_rate)
        self.max_exit_defer_trading_days = int(max_exit_defer_trading_days) if max_exit_defer_trading_days is not None else None
        self.require_canonical_calendar = bool(require_canonical_calendar)
        self.cols = ExecutionLabelColumns()
        if not 0.0 < self.extreme_quantile < 0.5:
            raise ValueError("extreme_quantile must be in (0, 0.5)")
        if self.threshold_mode not in {"fixed", "cross_sectional_median", "cross_sectional_extreme"}:
            raise ValueError(f"Unsupported threshold_mode: {self.threshold_mode}")

    @staticmethod
    def _normalize_calendar(df: pd.DataFrame, canonical_dates: Optional[Iterable[pd.Timestamp]]) -> pd.DatetimeIndex:
        if canonical_dates is None:
            dates = pd.DatetimeIndex(pd.to_datetime(df["date"].dropna().unique()))
        else:
            dates = pd.DatetimeIndex(pd.to_datetime(list(canonical_dates)))
            if len(dates):
                lo, hi = pd.to_datetime(df["date"]).min(), pd.to_datetime(df["date"]).max()
                dates = dates[(dates >= lo) & (dates <= hi)]
        return pd.DatetimeIndex(sorted(pd.unique(dates)))

    @staticmethod
    def _positive_numeric(series: pd.Series) -> pd.Series:
        vals = pd.to_numeric(series, errors="coerce")
        return vals.notna() & np.isfinite(vals) & (vals > 0)

    @staticmethod
    def _bool(series: pd.Series) -> pd.Series:
        return series.astype("boolean").fillna(False).astype(bool)

    @staticmethod
    def _historical_fee_rate_map(dates: pd.Series, getter) -> pd.Series:
        unique_dates = pd.DatetimeIndex(pd.to_datetime(dates.dropna().unique()))
        mapping = {d: float(getter(d)) for d in unique_dates}
        return pd.to_datetime(dates).map(mapping)

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = sorted(self.REQUIRED_COLUMNS.difference(df.columns))
        if missing:
            raise ValueError("Execution-aligned labeling requires explicit execution-state columns; missing: " + str(missing))
        if df.duplicated(["date", "symbol"]).any():
            dup = df.loc[df.duplicated(["date", "symbol"], keep=False), ["date", "symbol"]].head(5)
            raise ValueError("Duplicate date/symbol rows would make execution mapping ambiguous: " + str(dup.to_dict(orient="records")))

    def compute(
        self,
        df: pd.DataFrame,
        canonical_dates: Optional[Iterable[pd.Timestamp]] = None,
        require_canonical_calendar: Optional[bool] = None
    ) -> pd.DataFrame:
        must_require_cal = self.require_canonical_calendar if require_canonical_calendar is None else bool(require_canonical_calendar)
        if must_require_cal and (canonical_dates is None or len(list(canonical_dates)) == 0):
            raise RuntimeError(
                "FATAL: Canonical exchange trading calendar is required in certified mode! "
                "Deducing trading calendar from filtered stock rows is disallowed."
            )

        self._validate_input(df)
        base = df.copy()
        base["date"] = pd.to_datetime(base["date"])
        base.sort_values(["date", "symbol"], inplace=True)
        base.reset_index(drop=True, inplace=True)
        market_dates = self._normalize_calendar(base, canonical_dates)
        if len(market_dates) <= self.schema.exit_offset_trading_days:
            out = base.copy()
            self._initialize_empty_outputs(out)
            out["label_invalid_reason"] = "INSUFFICIENT_FORWARD_HORIZON"
            return out

        date_map = pd.DataFrame({"date": market_dates})
        entry_offset, exit_offset = self.schema.entry_offset_trading_days, self.schema.exit_offset_trading_days
        date_map["exec_entry_date"] = pd.NaT
        date_map["planned_exit_date"] = pd.NaT
        date_map.loc[: len(market_dates) - entry_offset - 1, "exec_entry_date"] = market_dates[entry_offset:].values
        date_map.loc[: len(market_dates) - exit_offset - 1, "planned_exit_date"] = market_dates[exit_offset:].values

        out = base.merge(date_map, on="date", how="left", validate="many_to_one")
        out.rename(columns={"date": "signal_date"}, inplace=True)

        entry_lookup = base[["symbol", "date", "open", "adj_open", "benchmark_open", "volume", "is_suspended", "is_limit_up_locked"]].rename(columns={
            "date": "_entry_key_date", "open": "exec_entry_raw_open", "adj_open": "exec_entry_adj_open",
            "benchmark_open": "benchmark_entry_open", "volume": "entry_volume", "is_suspended": "entry_is_suspended",
            "is_limit_up_locked": "entry_is_limit_up_locked"})
        out = out.merge(entry_lookup, left_on=["symbol", "exec_entry_date"], right_on=["symbol", "_entry_key_date"], how="left", validate="many_to_one")
        out.drop(columns=["_entry_key_date"], inplace=True)

        planned_lookup = base[["symbol", "date", "open", "adj_open", "benchmark_open", "volume", "is_suspended", "is_limit_down_locked"]].rename(columns={
            "date": "_planned_exit_key_date", "open": "planned_exit_raw_open", "adj_open": "planned_exit_adj_open",
            "benchmark_open": "planned_benchmark_exit_open", "volume": "planned_exit_volume", "is_suspended": "planned_exit_is_suspended",
            "is_limit_down_locked": "planned_exit_is_limit_down_locked"})
        out = out.merge(planned_lookup, left_on=["symbol", "planned_exit_date"], right_on=["symbol", "_planned_exit_key_date"], how="left", validate="many_to_one")
        out.drop(columns=["_planned_exit_key_date"], inplace=True)

        entry_present = out["exec_entry_adj_open"].notna() & out["exec_entry_raw_open"].notna()
        entry_price_ok = self._positive_numeric(out["exec_entry_adj_open"]) & self._positive_numeric(out["exec_entry_raw_open"])
        out["entry_tradable"] = entry_present & entry_price_ok & (pd.to_numeric(out["entry_volume"], errors="coerce").fillna(0.0) > 0.0) & ~self._bool(out["entry_is_suspended"]) & ~self._bool(out["entry_is_limit_up_locked"])

        planned_present = out["planned_exit_adj_open"].notna() & out["planned_exit_raw_open"].notna()
        planned_price_ok = self._positive_numeric(out["planned_exit_adj_open"]) & self._positive_numeric(out["planned_exit_raw_open"])
        out["planned_exit_tradable"] = planned_present & planned_price_ok & (pd.to_numeric(out["planned_exit_volume"], errors="coerce").fillna(0.0) > 0.0) & ~self._bool(out["planned_exit_is_suspended"]) & ~self._bool(out["planned_exit_is_limit_down_locked"])

        self._attach_first_executable_exit(out, base)
        date_to_pos = {pd.Timestamp(d): i for i, d in enumerate(market_dates)}
        planned_pos = pd.to_datetime(out["planned_exit_date"]).map(date_to_pos)
        actual_pos = pd.to_datetime(out["actual_exit_date"]).map(date_to_pos)
        entry_pos = pd.to_datetime(out["exec_entry_date"]).map(date_to_pos)
        out["exit_deferred_days"] = (actual_pos - planned_pos).astype("Float64")
        out["holding_trading_days"] = (actual_pos - entry_pos).astype("Float64")

        self._compute_returns_and_costs(out)
        self._assign_validity(out)
        self._build_cross_sectional_targets(out)

        out.rename(columns={"signal_date": "date"}, inplace=True)
        out.sort_values(["date", "symbol"], inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out

    def _initialize_empty_outputs(self, out: pd.DataFrame) -> None:
        out["exec_entry_date"] = pd.NaT
        out["planned_exit_date"] = pd.NaT
        out["actual_exit_date"] = pd.NaT
        for c in ["exec_entry_raw_open", "exec_entry_adj_open", "exec_exit_raw_open", "exec_exit_adj_open",
                  "benchmark_entry_open", "benchmark_exit_open", "exit_deferred_days", "holding_trading_days",
                  "stock_gross_return", "stock_net_return", "benchmark_return", "label_cost_drag",
                  self.cols.gross_alpha, self.cols.net_alpha, self.cols.direction, self.cols.rank, self.cols.downside]:
            out[c] = np.nan
        out["entry_tradable"] = False
        out["planned_exit_tradable"] = False
        out["label_valid"] = False
        out["label_invalid_reason"] = ""

    def _attach_first_executable_exit(self, out: pd.DataFrame, base: pd.DataFrame) -> None:
        sellable_mask = self._positive_numeric(base["open"]) & self._positive_numeric(base["adj_open"]) & (pd.to_numeric(base["volume"], errors="coerce").fillna(0.0) > 0.0) & ~self._bool(base["is_suspended"]) & ~self._bool(base["is_limit_down_locked"])
        sellable = base.loc[sellable_mask, ["symbol", "date", "open", "adj_open", "benchmark_open"]].sort_values(["symbol", "date"])

        out["actual_exit_date"] = pd.NaT
        out["exec_exit_raw_open"] = np.nan
        out["exec_exit_adj_open"] = np.nan
        out["benchmark_exit_open"] = np.nan
        grouped = {sym: g.reset_index(drop=True) for sym, g in sellable.groupby("symbol", sort=False)}

        for sym, row_idx in out.groupby("symbol", sort=False).groups.items():
            candidates = grouped.get(sym)
            if candidates is None or candidates.empty:
                continue
            planned = pd.to_datetime(out.loc[row_idx, "planned_exit_date"])
            valid_plan = planned.notna().to_numpy()
            if not valid_plan.any():
                continue
            candidate_dates = candidates["date"].to_numpy(dtype="datetime64[ns]")
            planned_np = planned.to_numpy(dtype="datetime64[ns]")
            positions = np.searchsorted(candidate_dates, planned_np[valid_plan], side="left")
            target_rows = np.asarray(list(row_idx))[valid_plan]
            in_range = positions < len(candidates)
            if not in_range.any():
                continue
            target_rows, positions = target_rows[in_range], positions[in_range]
            chosen = candidates.iloc[positions]
            out.loc[target_rows, "actual_exit_date"] = chosen["date"].to_numpy()
            out.loc[target_rows, "exec_exit_raw_open"] = chosen["open"].to_numpy(dtype=float)
            out.loc[target_rows, "exec_exit_adj_open"] = chosen["adj_open"].to_numpy(dtype=float)
            out.loc[target_rows, "benchmark_exit_open"] = chosen["benchmark_open"].to_numpy(dtype=float)

    def _compute_returns_and_costs(self, out: pd.DataFrame) -> None:
        entry_adj = pd.to_numeric(out["exec_entry_adj_open"], errors="coerce")
        exit_adj = pd.to_numeric(out["exec_exit_adj_open"], errors="coerce")
        bench_entry = pd.to_numeric(out["benchmark_entry_open"], errors="coerce")
        bench_exit = pd.to_numeric(out["benchmark_exit_open"], errors="coerce")

        stock_gross = np.where(self._positive_numeric(entry_adj) & self._positive_numeric(exit_adj), exit_adj / entry_adj - 1.0, np.nan)
        benchmark_ret = np.where(self._positive_numeric(bench_entry) & self._positive_numeric(bench_exit), bench_exit / bench_entry - 1.0, np.nan)

        entry_transfer = self._historical_fee_rate_map(out["exec_entry_date"], TradingFeeSchedule.get_transfer_fee_rate).astype(float)
        exit_transfer = self._historical_fee_rate_map(out["actual_exit_date"], TradingFeeSchedule.get_transfer_fee_rate).astype(float)
        exit_stamp = self._historical_fee_rate_map(out["actual_exit_date"], TradingFeeSchedule.get_stamp_duty_rate).astype(float)

        buy_multiplier = (1.0 + self.slippage_rate) * (1.0 + self.commission_rate + entry_transfer)
        sell_multiplier = (1.0 - self.slippage_rate) * (1.0 - self.commission_rate - exit_transfer - exit_stamp)
        stock_net = np.where(self._positive_numeric(entry_adj) & self._positive_numeric(exit_adj), (exit_adj * sell_multiplier) / (entry_adj * buy_multiplier) - 1.0, np.nan)

        out["stock_gross_return"] = stock_gross
        out["stock_net_return"] = stock_net
        out["benchmark_return"] = benchmark_ret
        out["label_cost_drag"] = out["stock_gross_return"] - out["stock_net_return"]
        out[self.cols.gross_alpha] = out["stock_gross_return"] - out["benchmark_return"]
        out[self.cols.net_alpha] = out["stock_net_return"] - out["benchmark_return"]

    def _assign_validity(self, out: pd.DataFrame) -> None:
        reason = pd.Series("", index=out.index, dtype="object")
        insufficient = out["exec_entry_date"].isna() | out["planned_exit_date"].isna()
        reason.loc[insufficient] = "INSUFFICIENT_FORWARD_HORIZON"

        entry_missing = ~insufficient & (out["exec_entry_raw_open"].isna() | out["exec_entry_adj_open"].isna())
        reason.loc[entry_missing] = "ENTRY_MARKET_ROW_MISSING"

        entry_suspended = (reason == "") & (self._bool(out["entry_is_suspended"]) | (pd.to_numeric(out["entry_volume"], errors="coerce").fillna(0.0) <= 0.0))
        reason.loc[entry_suspended] = "ENTRY_SUSPENDED_OR_NO_VOLUME"

        entry_limit_up = (reason == "") & self._bool(out["entry_is_limit_up_locked"])
        reason.loc[entry_limit_up] = "ENTRY_LIMIT_UP_LOCKED"

        entry_bad_price = (reason == "") & ~(self._positive_numeric(out["exec_entry_raw_open"]) & self._positive_numeric(out["exec_entry_adj_open"]))
        reason.loc[entry_bad_price] = "ENTRY_INVALID_OPEN_PRICE"

        no_exit = (reason == "") & out["actual_exit_date"].isna()
        reason.loc[no_exit] = "NO_EXECUTABLE_EXIT_BEFORE_DATA_END"

        # 延期退出上限策略检查 (Exit Deferral Exceeds Policy Gate)
        if self.max_exit_defer_trading_days is not None:
            defer_exceeded = (reason == "") & (pd.to_numeric(out["exit_deferred_days"], errors="coerce").fillna(0) > self.max_exit_defer_trading_days)
            reason.loc[defer_exceeded] = "EXIT_DEFER_EXCEEDS_POLICY"

        bad_benchmark = (reason == "") & ~(self._positive_numeric(out["benchmark_entry_open"]) & self._positive_numeric(out["benchmark_exit_open"]))
        reason.loc[bad_benchmark] = "BENCHMARK_OPEN_MISSING"

        bad_label = (reason == "") & ~np.isfinite(pd.to_numeric(out[self.cols.net_alpha], errors="coerce"))
        reason.loc[bad_label] = "NONFINITE_EXECUTION_LABEL"

        out["label_valid"] = reason.eq("")
        out["label_invalid_reason"] = reason
        out.loc[~out["label_valid"], [self.cols.gross_alpha, self.cols.net_alpha]] = np.nan

    def _build_cross_sectional_targets(self, out: pd.DataFrame) -> None:
        net = self.cols.net_alpha
        valid_net = out[net].where(out["label_valid"])
        out[self.cols.rank] = valid_net.groupby(out["signal_date"]).rank(pct=True)
        out[self.cols.downside] = np.minimum(valid_net, 0.0)
        out[self.cols.direction] = np.nan

        if self.threshold_mode == "cross_sectional_median":
            med = valid_net.groupby(out["signal_date"]).transform("median")
            mask = valid_net.notna() & med.notna()
            out.loc[mask, self.cols.direction] = (valid_net[mask] > med[mask]).astype(float)
        elif self.threshold_mode == "cross_sectional_extreme":
            rank_pct = valid_net.groupby(out["signal_date"]).rank(pct=True)
            top = rank_pct > 1.0 - self.extreme_quantile
            bottom = rank_pct < self.extreme_quantile
            out.loc[top & valid_net.notna(), self.cols.direction] = 1.0
            out.loc[bottom & valid_net.notna(), self.cols.direction] = 0.0
        else:
            mask = valid_net.notna()
            out.loc[mask, self.cols.direction] = (valid_net[mask] > self.threshold).astype(float)

        out.loc[~out["label_valid"], [self.cols.direction, self.cols.rank, self.cols.downside]] = np.nan
