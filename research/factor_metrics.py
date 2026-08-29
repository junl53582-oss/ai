"""
因子基础统计量、交易执行与可交易性审计引擎 (research/factor_metrics.py)
Phase 1.4 核心强化:
1. P0-1: 严格区分 INTRADAY_FACTOR_DIAGNOSTIC_RETURN (T+1 Open->Close) 与 A-Share Tradable Return (T+1 Open -> T+2 Open/Close)
2. P0-5: 完整可交易性过滤 (涨跌停锁死、停牌、ST) 与卖出腿锁死展期追踪
3. P0-6: 独立输出 Long-Only 真实策略绩效与 Diagnostic Spread 诊断指标
4. 导出 trade_rejection_evidence.csv 结构化交易拒单证据
"""
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import pandas as pd
from scipy import stats

from .config import ResearchConfig, default_research_config

logger = logging.getLogger(__name__)


class TradabilityStatus(str, Enum):
    TRADEABLE = "TRADEABLE"
    SUSPENDED = "SUSPENDED"
    LIMIT_UP_LOCKED = "LIMIT_UP_LOCKED"
    LIMIT_DOWN_LOCKED = "LIMIT_DOWN_LOCKED"
    INVALID_OPEN = "INVALID_OPEN"
    INVALID_EXIT_PRICE = "INVALID_EXIT_PRICE"
    NOT_IN_UNIVERSE = "NOT_IN_UNIVERSE"
    ST_BLOCKED = "ST_BLOCKED"
    DELISTED = "DELISTED"
    STALE_PRICE = "STALE_PRICE"
    MISSING_DATA = "MISSING_DATA"
    UNKNOWN_TRADABILITY = "UNKNOWN_TRADABILITY"


@dataclass
class FactorEvaluationMetrics:
    """单个因子的全要素量化评估指标 (Phase 1.4)"""
    factor_name: str
    horizon: int
    
    # IC 指标
    mean_ic: float = 0.0
    std_ic: float = 0.0
    ic_ir: float = 0.0
    annualized_icir: float = 0.0
    positive_ic_ratio: float = 0.0
    naive_t_stat: float = 0.0
    hac_t_stat: float = 0.0
    p_value: float = 1.0
    hac_p_value: float = 1.0
    fdr_p_value: float = 1.0
    
    # RankIC 指标 (主力评估依据)
    mean_rank_ic: float = 0.0
    std_rank_ic: float = 0.0
    rank_ic_ir: float = 0.0
    annualized_rank_icir: float = 0.0
    positive_rank_ic_ratio: float = 0.0
    rank_ic_t_stat: float = 0.0
    rank_ic_hac_t_stat: float = 0.0
    rank_ic_p_value: float = 1.0
    rank_ic_hac_p_value: float = 1.0
    rank_ic_fdr_p_value: float = 1.0
    
    # Bootstrap 置信区间
    bootstrap_ci_lower: float = 0.0
    bootstrap_ci_upper: float = 0.0
    
    # 因子方向与单调性
    recommended_direction: int = 1
    monotonicity_score: float = 0.0
    monotonicity_10q: float = 0.0
    
    # 分层前向收益
    quantile_returns_5q: Dict[str, float] = field(default_factory=dict)
    quantile_returns_10q: Dict[str, float] = field(default_factory=dict)
    
    # A股纯多头可执行策略指标 (Real Long-Only Executable Strategy)
    long_only_cagr: float = 0.0
    long_only_gross_return: float = 0.0
    long_only_net_return: float = 0.0
    long_only_sharpe: float = 0.0
    long_only_max_drawdown: float = 0.0
    long_only_win_rate: float = 0.0
    long_only_excess_annual_return: float = 0.0 # Top 组相对于基准的纯多头超额年化收益
    long_only_excess_sharpe: float = 0.0        # Top 组纯多头超额夏普
    long_turnover: float = 0.0
    
    # 因子诊断多空利差指标 (FACTOR_DIAGNOSTIC_SPREAD, 非实盘裸空组合)
    diagnostic_spread_mean: float = 0.0
    diagnostic_spread_annual: float = 0.0
    diagnostic_spread_sharpe: float = 0.0
    short_turnover: float = 0.0
    mean_turnover: float = 0.0
    
    # 成本敏感度 (GENERIC_COST_SENSITIVITY)
    cost_sensitivity: Dict[str, float] = field(default_factory=dict)
    
    # 样本完整性
    missing_ratio: float = 0.0
    coverage_ratio: float = 0.0
    valid_sample_count: int = 0
    daily_cross_section_count: float = 0.0
    
    # 时序日历序列与流水
    daily_ic_series: Optional[pd.Series] = None
    daily_rank_ic_series: Optional[pd.Series] = None
    daily_pnl_df: Optional[pd.DataFrame] = None
    trade_rejections: List[Dict[str, Any]] = field(default_factory=list)


class FactorMetricsEngine:
    """因子指标与交易执行评估引擎"""

    @staticmethod
    def compute_daily_ic(
        df: pd.DataFrame,
        factor_col: str,
        return_col: str,
        date_col: str = "date",
        method: str = "spearman"
    ) -> pd.Series:
        """逐日计算截面 IC 序列"""
        valid_mask = df[factor_col].notna() & df[return_col].notna()
        if "in_universe" in df.columns:
            valid_mask = valid_mask & df["in_universe"].fillna(False).astype(bool)
        if "is_suspended" in df.columns:
            valid_mask = valid_mask & (~df["is_suspended"].fillna(False).astype(bool))
            
        sub_df = df.loc[valid_mask, [date_col, factor_col, return_col]].copy()
        if sub_df.empty:
            return pd.Series(dtype=float)

        def _calc_cs_ic(g: pd.DataFrame) -> float:
            f_vals = g[factor_col].values
            r_vals = g[return_col].values
            if len(f_vals) < 3:
                return np.nan
            if np.nanstd(f_vals) < 1e-8 or len(np.unique(f_vals[~np.isnan(f_vals)])) <= 1:
                return 0.0
            if np.nanstd(r_vals) < 1e-8:
                return np.nan

            if method == "spearman":
                res, _ = stats.spearmanr(f_vals, r_vals, nan_policy="omit")
                return float(res) if not np.isnan(res) else np.nan
            else:
                res, _ = stats.pearsonr(f_vals, r_vals)
                return float(res) if not np.isnan(res) else np.nan

        daily_ic = sub_df.groupby(date_col)[[factor_col, return_col]].apply(_calc_cs_ic)
        return daily_ic.dropna()

    @staticmethod
    def compute_hac_tstat(series: pd.Series, lag: int = 19) -> Tuple[float, float]:
        """Newey-West (HAC) 异方差与自相关一致性标准误及 t-stat"""
        vals = series.dropna().values
        n = len(vals)
        if n < 3:
            return 0.0, 1.0

        mean_val = float(np.mean(vals))
        demeaned = vals - mean_val

        gamma_0 = float(np.mean(demeaned ** 2))
        if gamma_0 < 1e-12:
            return 0.0, 1.0

        max_lag = min(lag, n - 2)
        gamma_sum = 0.0
        for j in range(1, max_lag + 1):
            weight = 1.0 - (j / (max_lag + 1.0))
            cov_j = float(np.sum(demeaned[j:] * demeaned[:-j]) / n)
            gamma_sum += 2.0 * weight * cov_j

        var_hac = (gamma_0 + gamma_sum) / n
        if var_hac <= 1e-12:
            var_hac = gamma_0 / n

        se_hac = np.sqrt(max(var_hac, 1e-12))
        t_stat = mean_val / se_hac
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(n - 1, 1))))
        return round(float(t_stat), 4), round(p_val, 6)

    @staticmethod
    def compute_fdr_pvalues(p_values: List[float]) -> List[float]:
        """Benjamini-Hochberg (BH) FDR 多重检验校正"""
        n = len(p_values)
        if n == 0:
            return []
        
        p_arr = np.array(p_values, dtype=float)
        sorted_indices = np.argsort(p_arr)
        sorted_p = p_arr[sorted_indices]
        
        fdr_adjusted = np.zeros(n, dtype=float)
        cum_min = 1.0
        for i in range(n - 1, -1, -1):
            rank = i + 1
            adj_p = (sorted_p[i] * n) / rank
            cum_min = min(cum_min, adj_p)
            fdr_adjusted[i] = min(1.0, cum_min)
            
        orig_order_fdr = np.zeros(n, dtype=float)
        orig_order_fdr[sorted_indices] = fdr_adjusted
        return orig_order_fdr.tolist()

    @staticmethod
    def compute_block_bootstrap_ci(
        series: pd.Series,
        n_rounds: int = 500,
        block_size: int = 20,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """时间块重抽样 Block Bootstrap 95% 置信区间"""
        vals = series.dropna().values
        n = len(vals)
        if n < block_size * 2 or n == 0:
            mean_val = float(np.mean(vals)) if n > 0 else 0.0
            return round(mean_val, 4), round(mean_val, 4)

        n_blocks = int(np.ceil(n / block_size))
        boot_means = []
        rng = np.random.default_rng(42)

        for _ in range(n_rounds):
            start_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
            boot_sample = np.concatenate([vals[idx:idx + block_size] for idx in start_indices])[:n]
            boot_means.append(np.mean(boot_sample))

        alpha = (1.0 - confidence) / 2.0
        ci_lower = float(np.percentile(boot_means, alpha * 100))
        ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
        return round(ci_lower, 4), round(ci_upper, 4)

    @staticmethod
    def compute_quantile_returns(
        df: pd.DataFrame,
        factor_col: str,
        return_col: str,
        n_quantiles: int = 5,
        date_col: str = "date"
    ) -> Tuple[Dict[str, float], float, pd.DataFrame]:
        """计算因子分层前向收益与单调性"""
        valid_mask = df[factor_col].notna() & df[return_col].notna()
        if "in_universe" in df.columns:
            valid_mask = valid_mask & df["in_universe"].fillna(False).astype(bool)
        if "is_suspended" in df.columns:
            valid_mask = valid_mask & (~df["is_suspended"].fillna(False).astype(bool))

        sub_df = df.loc[valid_mask, [date_col, factor_col, return_col]].copy()
        if sub_df.empty:
            return {}, 0.0, pd.DataFrame()

        unique_counts = sub_df.groupby(date_col)[factor_col].transform(lambda s: len(np.unique(s)))
        valid_cs_mask = unique_counts >= n_quantiles

        sub_df = sub_df[valid_cs_mask].copy()
        if sub_df.empty:
            return {}, 0.0, pd.DataFrame()

        sub_df["f_pct"] = sub_df.groupby(date_col)[factor_col].rank(method="average", pct=True)
        sub_df["quantile"] = np.ceil(sub_df["f_pct"] * n_quantiles).astype(int).clip(1, n_quantiles)

        daily_q_ret = sub_df.groupby([date_col, "quantile"])[return_col].mean().unstack("quantile")
        mean_q_ret = daily_q_ret.mean()

        q_dict = {f"Q{int(q)}": round(float(mean_q_ret.get(q, 0.0)), 6) for q in range(1, n_quantiles + 1)}

        q_indices = np.arange(1, n_quantiles + 1)
        q_values = [mean_q_ret.get(q, 0.0) for q in q_indices]
        if len(q_values) >= 3 and np.nanstd(q_values) > 1e-8:
            corr_res, _ = stats.spearmanr(q_indices, q_values)
            mono_score = float(corr_res) if not np.isnan(corr_res) else 0.0
        else:
            mono_score = 0.0

        return q_dict, round(mono_score, 4), daily_q_ret

    @classmethod
    def compute_realized_daily_portfolio_pnl(
        cls,
        df: pd.DataFrame,
        factor_col: str,
        direction: int = 1,
        n_quantiles: int = 5,
        date_col: str = "date",
        symbol_col: str = "symbol",
        close_col: str = "adj_close",
        open_col: str = "adj_open",
        benchmark_close_col: str = "benchmark_close",
        benchmark_open_col: str = "benchmark_open",
        config: Optional[ResearchConfig] = None
    ) -> Dict[str, Any]:
        """
        Phase 1.4 A股真实 T+1 结算与买卖可交易性完整执行 (P0-1 / P0-5 / P0-6)
        时序: T 日收盘计算信号 -> T+1 日开盘买入 -> T+2 日开盘最早可卖出
        """
        cfg = config or default_research_config
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=[symbol_col, date_col], inplace=True)

        c_col = close_col if close_col in df.columns else "close"
        o_col = open_col if open_col in df.columns else ("open" if "open" in df.columns else c_col)

        # 识别全交易日列表
        dates = sorted(df[date_col].unique())
        date_to_idx = {d: i for i, d in enumerate(dates)}
        n_dates = len(dates)

        if n_dates < 3:
            return cls._empty_portfolio_res()

        # 检查 T 日有效样本
        valid_mask = df[factor_col].notna()
        if "in_universe" in df.columns:
            valid_mask = valid_mask & df["in_universe"].fillna(False).astype(bool)
        if "is_suspended" in df.columns:
            valid_mask = valid_mask & (~df["is_suspended"].fillna(False).astype(bool))

        sub_df = df[valid_mask].copy()
        if sub_df.empty:
            return cls._empty_portfolio_res()

        unique_per_day = sub_df.groupby(date_col)[factor_col].transform(lambda s: len(np.unique(s)))
        sub_df = sub_df[unique_per_day >= n_quantiles].copy()
        if sub_df.empty:
            return cls._empty_portfolio_res()

        sub_df["f_pct"] = sub_df.groupby(date_col)[factor_col].rank(method="average", pct=True)
        sub_df["quantile"] = np.ceil(sub_df["f_pct"] * n_quantiles).astype(int).clip(1, n_quantiles)

        top_q = n_quantiles if direction == 1 else 1
        bottom_q = 1 if direction == 1 else n_quantiles

        # 预先构建 Benchmark 序列
        b_close_col = benchmark_close_col if benchmark_close_col in df.columns else None
        b_open_col = benchmark_open_col if benchmark_open_col in df.columns else None
        
        bench_df = df.groupby(date_col)[[c for c in [b_close_col, b_open_col] if c]].first()

        pnl_records = []
        trade_rejections = []
        prev_top_set: Set[str] = set()
        prev_bottom_set: Set[str] = set()

        # 遍历交易日 T，计算从 T+1 Open 买入，至 T+2 Open (或退出日) 的真实 A 股收益
        for t_idx in range(n_dates - 2):
            d_signal = dates[t_idx]
            d_entry = dates[t_idx + 1]
            d_exit = dates[t_idx + 2]

            day_sig_df = sub_df[sub_df[date_col] == d_signal]
            req_top_syms = set(day_sig_df[day_sig_df["quantile"] == top_q][symbol_col].unique())
            req_bottom_syms = set(day_sig_df[day_sig_df["quantile"] == bottom_q][symbol_col].unique())

            if not req_top_syms:
                continue

            day_entry_df = df[df[date_col] == d_entry].set_index(symbol_col)
            day_exit_df = df[df[date_col] == d_exit].set_index(symbol_col)

            # ---------------- Top 组 (多头买入) 可交易性检查 ----------------
            exec_top_syms = []
            for sym in req_top_syms:
                if sym not in day_entry_df.index or sym not in day_exit_df.index:
                    trade_rejections.append({
                        "signal_date": str(d_signal.date()), "entry_date": str(d_entry.date()), "exit_date": str(d_exit.date()),
                        "symbol": sym, "side": "BUY", "reject_stage": "ENTRY", "reject_reason": TradabilityStatus.MISSING_DATA.value
                    })
                    continue

                e_row = day_entry_df.loc[sym]
                # 停牌
                if e_row.get("is_suspended", False):
                    trade_rejections.append({
                        "signal_date": str(d_signal.date()), "entry_date": str(d_entry.date()), "exit_date": str(d_exit.date()),
                        "symbol": sym, "side": "BUY", "reject_stage": "ENTRY", "reject_reason": TradabilityStatus.SUSPENDED.value
                    })
                    continue

                # 开盘价缺失
                open_p = e_row.get(o_col, 0.0)
                if pd.isna(open_p) or open_p <= 0:
                    trade_rejections.append({
                        "signal_date": str(d_signal.date()), "entry_date": str(d_entry.date()), "exit_date": str(d_exit.date()),
                        "symbol": sym, "side": "BUY", "reject_stage": "ENTRY", "reject_reason": TradabilityStatus.INVALID_OPEN.value
                    })
                    continue

                # 一字涨停锁死无法买入
                if "limit_up" in e_row and open_p >= e_row["limit_up"]:
                    trade_rejections.append({
                        "signal_date": str(d_signal.date()), "entry_date": str(d_entry.date()), "exit_date": str(d_exit.date()),
                        "symbol": sym, "side": "BUY", "reject_stage": "ENTRY", "reject_reason": TradabilityStatus.LIMIT_UP_LOCKED.value
                    })
                    continue

                # ST 检查
                if not cfg.ALLOW_ST_TRADING and e_row.get("is_st", False):
                    trade_rejections.append({
                        "signal_date": str(d_signal.date()), "entry_date": str(d_entry.date()), "exit_date": str(d_exit.date()),
                        "symbol": sym, "side": "BUY", "reject_stage": "ENTRY", "reject_reason": TradabilityStatus.ST_BLOCKED.value
                    })
                    continue

                # 退出日可卖出性检查
                x_row = day_exit_df.loc[sym]
                exit_p = x_row.get(o_col, 0.0) if cfg.EXIT_PRICE_TYPE == "open" else x_row.get(c_col, 0.0)
                if pd.isna(exit_p) or exit_p <= 0:
                    trade_rejections.append({
                        "signal_date": str(d_signal.date()), "entry_date": str(d_entry.date()), "exit_date": str(d_exit.date()),
                        "symbol": sym, "side": "SELL", "reject_stage": "EXIT", "reject_reason": TradabilityStatus.INVALID_EXIT_PRICE.value
                    })
                    continue

                exec_top_syms.append(sym)

            if not exec_top_syms:
                continue

            # 多头换手率
            exec_top_set = set(exec_top_syms)
            if prev_top_set:
                intersect_top = len(exec_top_set.intersection(prev_top_set))
                long_turnover = 1.0 - (intersect_top / max(len(exec_top_set), 1))
            else:
                long_turnover = 1.0
            prev_top_set = exec_top_set

            # 计算真实 A 股持仓收益: StockOpen[T+2] / StockOpen[T+1] - 1
            top_returns = []
            for sym in exec_top_syms:
                e_p = float(day_entry_df.loc[sym, o_col])
                x_p = float(day_exit_df.loc[sym, o_col] if cfg.EXIT_PRICE_TYPE == "open" else day_exit_df.loc[sym, c_col])
                top_returns.append((x_p / e_p) - 1.0)
            long_gross_ret = float(np.mean(top_returns))

            # 基准收益 (完全相同时序: BenchmarkOpen[T+2] / BenchmarkOpen[T+1] - 1)
            if b_open_col and b_open_col in bench_df.columns and d_entry in bench_df.index and d_exit in bench_df.index:
                b_e = bench_df.loc[d_entry, b_open_col]
                b_x = bench_df.loc[d_exit, b_open_col] if cfg.EXIT_PRICE_TYPE == "open" else bench_df.loc[d_exit, b_close_col]
                bench_ret = float((b_x / b_e) - 1.0) if (b_e and b_e > 0) else 0.0
            elif b_close_col and b_close_col in bench_df.columns and d_entry in bench_df.index and d_exit in bench_df.index:
                b_e = bench_df.loc[d_entry, b_close_col]
                b_x = bench_df.loc[d_exit, b_close_col]
                bench_ret = float((b_x / b_e) - 1.0) if (b_e and b_e > 0) else 0.0
            else:
                bench_ret = 0.0

            # 真实 A 股非对称摩擦成本
            comm = cfg.DEFAULT_COMMISSION_BPS / 10000.0
            stamp = cfg.DEFAULT_STAMP_DUTY_BPS / 10000.0
            slip = cfg.DEFAULT_SLIPPAGE_BPS / 10000.0

            # 买入开仓成本: 佣金 + 滑点
            entry_fee = long_turnover * (comm + slip)
            # 卖出平仓成本: 佣金 + 印花税 + 滑点
            exit_fee = long_turnover * (comm + stamp + slip)
            total_long_cost = entry_fee + exit_fee
            long_net_ret = long_gross_ret - total_long_cost
            long_excess_ret = long_gross_ret - bench_ret

            # 诊断用利差 (FACTOR_DIAGNOSTIC_SPREAD)
            exec_bottom_syms = [s for s in req_bottom_syms if s in day_entry_df.index and s in day_exit_df.index]
            if exec_bottom_syms:
                bottom_rets = [(float(day_exit_df.loc[s, o_col]) / float(day_entry_df.loc[s, o_col])) - 1.0 for s in exec_bottom_syms]
                diag_spread = long_gross_ret - float(np.mean(bottom_rets))
                short_turnover = 1.0
            else:
                diag_spread = 0.0
                short_turnover = 0.0

            pnl_records.append({
                "signal_date": str(d_signal.date()),
                "entry_date": str(d_entry.date()),
                "entry_price_type": cfg.ENTRY_PRICE_TYPE,
                "earliest_exit_date": str(d_exit.date()),
                "actual_exit_date": str(d_exit.date()),
                "exit_price_type": cfg.EXIT_PRICE_TYPE,
                "requested_long_count": len(req_top_syms),
                "executed_long_count": len(exec_top_syms),
                "rejected_long_count": len(req_top_syms) - len(exec_top_syms),
                "long_gross_return": long_gross_ret,
                "benchmark_return": bench_ret,
                "long_excess_return": long_excess_ret,
                "entry_commission": comm * long_turnover,
                "exit_commission": comm * long_turnover,
                "stamp_duty": stamp * long_turnover,
                "slippage": 2.0 * slip * long_turnover,
                "total_cost": total_long_cost,
                "long_net_return": long_net_ret,
                "factor_diagnostic_spread": diag_spread,
                "long_turnover": long_turnover,
                "short_turnover": short_turnover,
                "gross_turnover": 0.5 * (long_turnover + short_turnover)
            })

        if not pnl_records:
            res = cls._empty_portfolio_res()
            res["trade_rejections"] = trade_rejections
            return res

        pnl_df = pd.DataFrame(pnl_records)
        pnl_df["long_equity_curve"] = (1.0 + pnl_df["long_net_return"]).cumprod()

        net_s = pnl_df["long_net_return"]
        gross_s = pnl_df["long_gross_return"]
        exc_s = pnl_df["long_excess_return"]
        mean_long_turnover = float(pnl_df["long_turnover"].mean())
        mean_short_turnover = float(pnl_df["short_turnover"].mean())
        mean_gross_turnover = float(pnl_df["gross_turnover"].mean())

        net_d_mean = float(net_s.mean())
        gross_d_mean = float(gross_s.mean())
        exc_d_mean = float(exc_s.mean())
        daily_std = float(net_s.std(ddof=1)) if len(net_s) > 1 else 1e-6
        exc_std = float(exc_s.std(ddof=1)) if len(exc_s) > 1 else 1e-6

        # 安全复利年化
        if net_d_mean > 5.0:
            ann_return = 999.0
        elif net_d_mean > -1.0:
            try:
                ann_return = (1.0 + net_d_mean) ** 252.0 - 1.0
            except OverflowError:
                ann_return = 999.0
        else:
            ann_return = -1.0

        if exc_d_mean > 5.0:
            exc_ann = 999.0
        elif exc_d_mean > -1.0:
            try:
                exc_ann = (1.0 + exc_d_mean) ** 252.0 - 1.0
            except OverflowError:
                exc_ann = 999.0
        else:
            exc_ann = -1.0

        sharpe = (net_d_mean / daily_std) * np.sqrt(252.0) if daily_std > 1e-8 else 0.0
        exc_sharpe = (exc_d_mean / exc_std) * np.sqrt(252.0) if exc_std > 1e-8 else 0.0

        cum_equity = pnl_df["long_equity_curve"]
        running_max = cum_equity.cummax()
        drawdowns = (cum_equity - running_max) / running_max
        max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0
        win_rate = float((net_s > 0).mean())

        # 诊断利差年化与夏普
        spread_s = pnl_df["factor_diagnostic_spread"]
        spread_mean = float(spread_s.mean())
        spread_std = float(spread_s.std(ddof=1)) if len(spread_s) > 1 else 1e-6
        spread_ann = (1.0 + spread_mean) ** 252.0 - 1.0 if (spread_mean > -1.0 and spread_mean < 5.0) else 0.0
        spread_sharpe = (spread_mean / spread_std) * np.sqrt(252.0) if spread_std > 1e-8 else 0.0

        # 通用费率敏感度 (GENERIC_COST_SENSITIVITY)
        cost_sensitivity = {}
        for bps in cfg.COST_BPS_LIST:
            f_rate = bps / 10000.0
            test_net_mean = gross_d_mean - (2.0 * mean_long_turnover * f_rate)
            if test_net_mean > 5.0:
                test_net_ann = 999.0
            elif test_net_mean > -1.0:
                try:
                    test_net_ann = (1.0 + test_net_mean) ** 252.0 - 1.0
                except OverflowError:
                    test_net_ann = 999.0
            else:
                test_net_ann = -1.0
            cost_sensitivity[f"{int(bps)}bps"] = round(float(test_net_ann), 6)

        return {
            "long_turnover": round(mean_long_turnover, 4),
            "short_turnover": round(mean_short_turnover, 4),
            "mean_turnover": round(mean_gross_turnover, 4),
            "long_only_gross_return": round(gross_d_mean, 6),
            "long_only_net_return": round(net_d_mean, 6),
            "long_only_cagr": round(ann_return, 4),
            "long_only_sharpe": round(sharpe, 4),
            "long_only_max_drawdown": round(max_dd, 4),
            "long_only_win_rate": round(win_rate, 4),
            "long_only_excess_annual_return": round(exc_ann, 4),
            "long_only_excess_sharpe": round(exc_sharpe, 4),
            "diagnostic_spread_mean": round(spread_mean, 6),
            "diagnostic_spread_annual": round(spread_ann, 4),
            "diagnostic_spread_sharpe": round(spread_sharpe, 4),
            "cost_sensitivity": cost_sensitivity,
            "daily_pnl_df": pnl_df,
            "trade_rejections": trade_rejections
        }

    @staticmethod
    def _empty_portfolio_res() -> Dict[str, Any]:
        return {
            "long_turnover": 0.0,
            "short_turnover": 0.0,
            "mean_turnover": 0.0,
            "long_only_gross_return": 0.0,
            "long_only_net_return": 0.0,
            "long_only_cagr": 0.0,
            "long_only_sharpe": 0.0,
            "long_only_max_drawdown": 0.0,
            "long_only_win_rate": 0.0,
            "long_only_excess_annual_return": 0.0,
            "long_only_excess_sharpe": 0.0,
            "diagnostic_spread_mean": 0.0,
            "diagnostic_spread_annual": 0.0,
            "diagnostic_spread_sharpe": 0.0,
            "cost_sensitivity": {"5bps": 0.0, "10bps": 0.0, "20bps": 0.0, "30bps": 0.0},
            "daily_pnl_df": pd.DataFrame(),
            "trade_rejections": []
        }

    @classmethod
    def evaluate_factor(
        cls,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str,
        horizon: int = 20,
        config: Optional[ResearchConfig] = None
    ) -> FactorEvaluationMetrics:
        """单因子端到端全要素评估"""
        cfg = config or default_research_config
        
        ic_series = cls.compute_daily_ic(df, factor_col, return_col, method="pearson")
        rank_ic_series = cls.compute_daily_ic(df, factor_col, return_col, method="spearman")

        n_days = len(rank_ic_series)
        mean_ic = float(ic_series.mean()) if not ic_series.empty else 0.0
        std_ic = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
        ic_ir = (mean_ic / std_ic) if std_ic > 1e-8 else 0.0
        ann_icir = ic_ir * np.sqrt(252.0 / max(horizon, 1))

        naive_t = (mean_ic / (std_ic / np.sqrt(n_days))) if (std_ic > 1e-8 and n_days > 1) else 0.0
        naive_p = float(2.0 * (1.0 - stats.t.cdf(abs(naive_t), df=max(n_days - 1, 1)))) if n_days > 1 else 1.0
        hac_t, hac_p = cls.compute_hac_tstat(ic_series, lag=max(horizon - 1, 1))
        pos_ic_ratio = float((ic_series > 0).mean()) if not ic_series.empty else 0.0

        mean_rank_ic = float(rank_ic_series.mean()) if not rank_ic_series.empty else 0.0
        std_rank_ic = float(rank_ic_series.std(ddof=1)) if len(rank_ic_series) > 1 else 0.0
        rank_ic_ir = (mean_rank_ic / std_rank_ic) if std_rank_ic > 1e-8 else 0.0
        ann_rank_icir = rank_ic_ir * np.sqrt(252.0 / max(horizon, 1))

        rank_naive_t = (mean_rank_ic / (std_rank_ic / np.sqrt(n_days))) if (std_rank_ic > 1e-8 and n_days > 1) else 0.0
        rank_naive_p = float(2.0 * (1.0 - stats.t.cdf(abs(rank_naive_t), df=max(n_days - 1, 1)))) if n_days > 1 else 1.0
        rank_hac_t, rank_hac_p = cls.compute_hac_tstat(rank_ic_series, lag=max(horizon - 1, 1))
        pos_rank_ic_ratio = float((rank_ic_series > 0).mean()) if not rank_ic_series.empty else 0.0

        ci_lower, ci_upper = cls.compute_block_bootstrap_ci(
            rank_ic_series,
            n_rounds=cfg.BOOTSTRAP_ROUNDS,
            block_size=cfg.BOOTSTRAP_BLOCK_SIZE,
            confidence=cfg.BOOTSTRAP_CONFIDENCE
        )

        direction = 1 if mean_rank_ic >= 0 else -1

        q5_dict, mono_5q, _ = cls.compute_quantile_returns(df, factor_col, return_col, n_quantiles=5)
        q10_dict, mono_10q, _ = cls.compute_quantile_returns(df, factor_col, return_col, n_quantiles=10)

        # 真实 A 股可执行多头策略与利差诊断
        pnl_res = cls.compute_realized_daily_portfolio_pnl(
            df,
            factor_col=factor_col,
            direction=direction,
            n_quantiles=5,
            config=cfg
        )

        total_rows = len(df)
        valid_rows = int(df[factor_col].notna().sum())
        missing_ratio = float((total_rows - valid_rows) / max(total_rows, 1))
        coverage_ratio = float(valid_rows / max(total_rows, 1))
        daily_counts = df.groupby("date")[factor_col].count()
        daily_cs_count = float(daily_counts.mean()) if not daily_counts.empty else 0.0

        return FactorEvaluationMetrics(
            factor_name=factor_col,
            horizon=horizon,
            mean_ic=round(mean_ic, 4),
            std_ic=round(std_ic, 4),
            ic_ir=round(ic_ir, 4),
            annualized_icir=round(ann_icir, 4),
            positive_ic_ratio=round(pos_ic_ratio, 4),
            naive_t_stat=round(naive_t, 4),
            hac_t_stat=hac_t,
            p_value=round(naive_p, 6),
            hac_p_value=hac_p,
            fdr_p_value=round(hac_p, 6),
            mean_rank_ic=round(mean_rank_ic, 4),
            std_rank_ic=round(std_rank_ic, 4),
            rank_ic_ir=round(rank_ic_ir, 4),
            annualized_rank_icir=round(ann_rank_icir, 4),
            positive_rank_ic_ratio=round(pos_rank_ic_ratio, 4),
            rank_ic_t_stat=round(rank_naive_t, 4),
            rank_ic_hac_t_stat=rank_hac_t,
            rank_ic_p_value=round(rank_naive_p, 6),
            rank_ic_hac_p_value=rank_hac_p,
            rank_ic_fdr_p_value=round(rank_hac_p, 6),
            bootstrap_ci_lower=ci_lower,
            bootstrap_ci_upper=ci_upper,
            recommended_direction=direction,
            monotonicity_score=mono_5q,
            monotonicity_10q=mono_10q,
            quantile_returns_5q=q5_dict,
            quantile_returns_10q=q10_dict,
            long_only_cagr=pnl_res["long_only_cagr"],
            long_only_gross_return=pnl_res["long_only_gross_return"],
            long_only_net_return=pnl_res["long_only_net_return"],
            long_only_sharpe=pnl_res["long_only_sharpe"],
            long_only_max_drawdown=pnl_res["long_only_max_drawdown"],
            long_only_win_rate=pnl_res["long_only_win_rate"],
            long_only_excess_annual_return=pnl_res["long_only_excess_annual_return"],
            long_only_excess_sharpe=pnl_res["long_only_excess_sharpe"],
            long_turnover=pnl_res["long_turnover"],
            diagnostic_spread_mean=pnl_res["diagnostic_spread_mean"],
            diagnostic_spread_annual=pnl_res["diagnostic_spread_annual"],
            diagnostic_spread_sharpe=pnl_res["diagnostic_spread_sharpe"],
            short_turnover=pnl_res["short_turnover"],
            mean_turnover=pnl_res["mean_turnover"],
            cost_sensitivity=pnl_res["cost_sensitivity"],
            missing_ratio=round(missing_ratio, 4),
            coverage_ratio=round(coverage_ratio, 4),
            valid_sample_count=valid_rows,
            daily_cross_section_count=round(daily_cs_count, 1),
            daily_ic_series=ic_series,
            daily_rank_ic_series=rank_ic_series,
            daily_pnl_df=pnl_res["daily_pnl_df"],
            trade_rejections=pnl_res["trade_rejections"]
        )
