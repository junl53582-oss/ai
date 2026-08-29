"""
因子基础统计量、交易执行与可交易性审计引擎 (research/factor_metrics.py)
Phase 1.5 核心强化:
1. P0-4: 生产 Schema 对齐 (is_limit_up_locked, is_limit_down_locked, limit_up_price, limit_down_price)
2. P0-5: 真实 Delayed Exit 展期卖出 (T+2跌停/停牌顺延至T+k，最长 MAX_UNEXECUTED_EXIT_DAYS，超时记录 UNEXECUTED_TIMEOUT)
3. 交易成本严格发生在 actual_exit_date
4. P1-3: 严格几何增长复利 CAGR (final_equity / initial_equity)**(252/N) - 1
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


class ExitStatus(str, Enum):
    EXECUTED_ON_TIME = "EXECUTED_ON_TIME"
    DELAYED_LIMIT_DOWN = "DELAYED_LIMIT_DOWN"
    DELAYED_SUSPENSION = "DELAYED_SUSPENSION"
    UNEXECUTED_TIMEOUT = "UNEXECUTED_TIMEOUT"


@dataclass
class FactorEvaluationMetrics:
    """单个因子的全要素量化评估指标 (Phase 1.5)"""
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
    long_only_cagr: Optional[float] = 0.0
    long_only_gross_return: float = 0.0
    long_only_net_return: float = 0.0
    long_only_sharpe: float = 0.0
    long_only_max_drawdown: float = 0.0
    long_only_win_rate: float = 0.0
    long_only_excess_annual_return: Optional[float] = 0.0 # Top 组相对于基准的纯多头超额年化收益
    long_only_excess_sharpe: Optional[float] = 0.0        # Top 组纯多头超额夏普
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
        Phase 1.5 A股真实 T+1 结算、生产 Schema 对齐与 Delayed Exit (P0-4 / P0-5 / P1-3)
        时序: T 日收盘计算信号 -> T+1 日开盘买入 -> T+2 日开盘最早卖出 (若无法卖出则顺延至 T+k)
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

        # 快速按日期建立个股字典索引
        df_by_date = {dt: g.set_index(symbol_col) for dt, g in df.groupby(date_col)}

        pnl_records = []
        trade_rejections = []
        prev_top_set: Set[str] = set()
        prev_bottom_set: Set[str] = set()

        # 遍历交易日 T，计算从 T+1 Open 买入，至 T+2 (或顺延 T+k) 的真实 A 股收益
        for t_idx in range(n_dates - 2):
            d_signal = dates[t_idx]
            d_entry = dates[t_idx + 1]
            d_earliest_exit = dates[t_idx + 2]

            day_sig_df = sub_df[sub_df[date_col] == d_signal]
            req_top_syms = set(day_sig_df[day_sig_df["quantile"] == top_q][symbol_col].unique())
            req_bottom_syms = set(day_sig_df[day_sig_df["quantile"] == bottom_q][symbol_col].unique())

            if not req_top_syms:
                continue

            day_entry_df = df_by_date.get(d_entry, pd.DataFrame())
            if day_entry_df.empty:
                continue

            # ---------------- Top 组 (多头买入) 可交易性检查 (P0-4) ----------------
            exec_top_positions = []
            for sym in req_top_syms:
                if sym not in day_entry_df.index:
                    trade_rejections.append({
                        "signal_date": str(d_signal.date()), "entry_date": str(d_entry.date()), "earliest_exit_date": str(d_earliest_exit.date()),
                        "symbol": sym, "side": "BUY", "reject_stage": "ENTRY", "reject_reason": TradabilityStatus.MISSING_DATA.value
                    })
                    continue

                e_row = day_entry_df.loc[sym]
                
                # 停牌
                if e_row.get("is_suspended", False):
                    trade_rejections.append({
                        "signal_date": str(d_signal.date()), "entry_date": str(d_entry.date()), "earliest_exit_date": str(d_earliest_exit.date()),
                        "symbol": sym, "side": "BUY", "reject_stage": "ENTRY", "reject_reason": TradabilityStatus.SUSPENDED.value
                    })
                    continue

                # 开盘价缺失
                open_p = e_row.get(o_col, 0.0)
                if pd.isna(open_p) or open_p <= 0:
                    trade_rejections.append({
                        "signal_date": str(d_signal.date()), "entry_date": str(d_entry.date()), "earliest_exit_date": str(d_earliest_exit.date()),
                        "symbol": sym, "side": "BUY", "reject_stage": "ENTRY", "reject_reason": TradabilityStatus.INVALID_OPEN.value
                    })
                    continue

                # 一字涨停锁死无法买入 (支持生产字段 is_limit_up_locked 与 limit_up_price)
                is_l_up = e_row.get("is_limit_up_locked", False)
                l_up_p = e_row.get("limit_up_price", e_row.get("limit_up", np.nan))
                if is_l_up or (not pd.isna(l_up_p) and l_up_p > 0 and open_p >= l_up_p):
                    trade_rejections.append({
                        "signal_date": str(d_signal.date()), "entry_date": str(d_entry.date()), "earliest_exit_date": str(d_earliest_exit.date()),
                        "symbol": sym, "side": "BUY", "reject_stage": "ENTRY", "reject_reason": TradabilityStatus.LIMIT_UP_LOCKED.value
                    })
                    continue

                # ST 检查
                if not cfg.ALLOW_ST_TRADING and e_row.get("is_st", False):
                    trade_rejections.append({
                        "signal_date": str(d_signal.date()), "entry_date": str(d_entry.date()), "earliest_exit_date": str(d_earliest_exit.date()),
                        "symbol": sym, "side": "BUY", "reject_stage": "ENTRY", "reject_reason": TradabilityStatus.ST_BLOCKED.value
                    })
                    continue

                # ---------------- 真实 Delayed Exit 寻找退出时点 (P0-5) ----------------
                actual_exit_dt = None
                actual_exit_p = None
                exit_status_str = None
                exit_delay = 0
                attempt_cnt = 0

                max_step = min(cfg.MAX_UNEXECUTED_EXIT_DAYS, n_dates - (t_idx + 2))
                for step in range(max_step):
                    attempt_cnt += 1
                    cand_dt = dates[t_idx + 2 + step]
                    cand_day_df = df_by_date.get(cand_dt, pd.DataFrame())
                    if cand_day_df.empty or sym not in cand_day_df.index:
                        continue
                    
                    x_row = cand_day_df.loc[sym]
                    # 检查是否停牌
                    if x_row.get("is_suspended", False):
                        continue
                    
                    # 检查价格有效性
                    cand_p = x_row.get(o_col, 0.0) if cfg.EXIT_PRICE_TYPE == "open" else x_row.get(c_col, 0.0)
                    if pd.isna(cand_p) or cand_p <= 0:
                        continue
                    
                    # 检查是否一字跌停锁死 (支持生产字段 is_limit_down_locked 与 limit_down_price)
                    is_l_dn = x_row.get("is_limit_down_locked", False)
                    l_dn_p = x_row.get("limit_down_price", x_row.get("limit_down", np.nan))
                    if is_l_dn or (not pd.isna(l_dn_p) and l_dn_p > 0 and cand_p <= l_dn_p):
                        continue

                    # 找到第一个可卖交易日！
                    actual_exit_dt = cand_dt
                    actual_exit_p = float(cand_p)
                    exit_delay = step
                    if step == 0:
                        exit_status_str = ExitStatus.EXECUTED_ON_TIME.value
                    else:
                        # 判断顺延主因
                        first_x_row = df_by_date.get(d_earliest_exit, pd.DataFrame()).loc[sym] if sym in df_by_date.get(d_earliest_exit, pd.DataFrame()).index else {}
                        if first_x_row.get("is_suspended", False):
                            exit_status_str = ExitStatus.DELAYED_SUSPENSION.value
                        else:
                            exit_status_str = ExitStatus.DELAYED_LIMIT_DOWN.value
                    break

                if actual_exit_dt is None:
                    # 超过最大展期天数仍未卖出
                    timeout_dt = dates[min(t_idx + 2 + max_step - 1, n_dates - 1)]
                    timeout_df = df_by_date.get(timeout_dt, pd.DataFrame())
                    if not timeout_df.empty and sym in timeout_df.index:
                        actual_exit_p = float(timeout_df.loc[sym].get(c_col, open_p))
                    else:
                        actual_exit_p = float(open_p)
                    actual_exit_dt = timeout_dt
                    exit_delay = max_step
                    exit_status_str = ExitStatus.UNEXECUTED_TIMEOUT.value

                exec_top_positions.append({
                    "symbol": sym,
                    "entry_price": float(open_p),
                    "exit_price": actual_exit_p,
                    "earliest_exit_date": d_earliest_exit,
                    "actual_exit_date": actual_exit_dt,
                    "exit_delay_days": exit_delay,
                    "exit_attempt_count": attempt_cnt,
                    "exit_status": exit_status_str,
                    "stock_return": (actual_exit_p / float(open_p)) - 1.0
                })

            if not exec_top_positions:
                continue

            exec_top_syms = [p["symbol"] for p in exec_top_positions]
            exec_top_set = set(exec_top_syms)

            # 多头换手率
            if prev_top_set:
                intersect_top = len(exec_top_set.intersection(prev_top_set))
                long_turnover = 1.0 - (intersect_top / max(len(exec_top_set), 1))
            else:
                long_turnover = 1.0
            prev_top_set = exec_top_set

            # 纯多头持仓收益
            top_returns = [p["stock_return"] for p in exec_top_positions]
            long_gross_ret = float(np.mean(top_returns))

            # 对应退出日的基准收益
            max_act_exit_dt = max(p["actual_exit_date"] for p in exec_top_positions)
            if b_open_col and b_open_col in bench_df.columns and d_entry in bench_df.index and max_act_exit_dt in bench_df.index:
                b_e = bench_df.loc[d_entry, b_open_col]
                b_x = bench_df.loc[max_act_exit_dt, b_open_col] if cfg.EXIT_PRICE_TYPE == "open" else bench_df.loc[max_act_exit_dt, b_close_col]
                bench_ret = float((b_x / b_e) - 1.0) if (b_e and b_e > 0) else 0.0
            elif b_close_col and b_close_col in bench_df.columns and d_entry in bench_df.index and max_act_exit_dt in bench_df.index:
                b_e = bench_df.loc[d_entry, b_close_col]
                b_x = bench_df.loc[max_act_exit_dt, b_close_col]
                bench_ret = float((b_x / b_e) - 1.0) if (b_e and b_e > 0) else 0.0
            else:
                bench_ret = 0.0

            # 真实 A 股非对称摩擦成本 (严格绑定实际成交)
            comm = cfg.DEFAULT_COMMISSION_BPS / 10000.0
            stamp = cfg.DEFAULT_STAMP_DUTY_BPS / 10000.0
            slip = cfg.DEFAULT_SLIPPAGE_BPS / 10000.0

            entry_fee = long_turnover * (comm + slip)
            exit_fee = long_turnover * (comm + stamp + slip)
            total_long_cost = entry_fee + exit_fee
            long_net_ret = long_gross_ret - total_long_cost
            long_excess_ret = long_gross_ret - bench_ret

            # 诊断用利差
            exec_bottom_syms = [s for s in req_bottom_syms if s in day_entry_df.index]
            if exec_bottom_syms:
                bottom_day_exit_df = df_by_date.get(d_earliest_exit, pd.DataFrame())
                bottom_rets = []
                for s in exec_bottom_syms:
                    if not bottom_day_exit_df.empty and s in bottom_day_exit_df.index:
                        b_ep = float(day_entry_df.loc[s, o_col])
                        b_xp = float(bottom_day_exit_df.loc[s, o_col] if cfg.EXIT_PRICE_TYPE == "open" else bottom_day_exit_df.loc[s, c_col])
                        bottom_rets.append((b_xp / b_ep) - 1.0)
                diag_spread = long_gross_ret - (float(np.mean(bottom_rets)) if bottom_rets else 0.0)
                short_turnover = 1.0
            else:
                diag_spread = 0.0
                short_turnover = 0.0

            mean_delay = float(np.mean([p["exit_delay_days"] for p in exec_top_positions]))
            mean_attempts = float(np.mean([p["exit_attempt_count"] for p in exec_top_positions]))
            primary_status = exec_top_positions[0]["exit_status"]

            pnl_records.append({
                "signal_date": str(d_signal.date()),
                "entry_date": str(d_entry.date()),
                "entry_price_type": cfg.ENTRY_PRICE_TYPE,
                "earliest_exit_date": str(d_earliest_exit.date()),
                "actual_exit_date": str(max_act_exit_dt.date()),
                "exit_delay_days": int(round(mean_delay)),
                "exit_attempt_count": int(round(mean_attempts)),
                "exit_status": primary_status,
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

        # 严格几何 CAGR: (final_equity / initial_equity)**(252 / N) - 1 (P1-3)
        n_periods = len(pnl_df)
        final_eq = float(pnl_df["long_equity_curve"].iloc[-1])
        if final_eq > 0 and n_periods > 0:
            cagr = (final_eq ** (252.0 / n_periods)) - 1.0
        else:
            cagr = -1.0

        # 基准几何收益
        pnl_df["bench_equity_curve"] = (1.0 + pnl_df["benchmark_return"]).cumprod()
        final_bench_eq = float(pnl_df["bench_equity_curve"].iloc[-1])
        if final_bench_eq > 0 and n_periods > 0:
            bench_cagr = (final_bench_eq ** (252.0 / n_periods)) - 1.0
        else:
            bench_cagr = -1.0
        exc_cagr = cagr - bench_cagr

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

        # 通用费率敏感度
        cost_sensitivity = {}
        for bps in cfg.COST_BPS_LIST:
            f_rate = bps / 10000.0
            test_net_series = gross_s - (2.0 * mean_long_turnover * f_rate)
            test_eq = (1.0 + test_net_series).cumprod()
            test_final = float(test_eq.iloc[-1]) if not test_eq.empty else 0.0
            test_cagr = (test_final ** (252.0 / n_periods)) - 1.0 if (test_final > 0 and n_periods > 0) else -1.0
            cost_sensitivity[f"{int(bps)}bps"] = round(float(test_cagr), 6)

        return {
            "long_turnover": round(mean_long_turnover, 4),
            "short_turnover": round(mean_short_turnover, 4),
            "mean_turnover": round(mean_gross_turnover, 4),
            "long_only_gross_return": round(gross_d_mean, 6),
            "long_only_net_return": round(net_d_mean, 6),
            "long_only_cagr": round(cagr, 4),
            "long_only_sharpe": round(sharpe, 4),
            "long_only_max_drawdown": round(max_dd, 4),
            "long_only_win_rate": round(win_rate, 4),
            "long_only_excess_annual_return": round(exc_cagr, 4),
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
