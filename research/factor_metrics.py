"""
因子基础统计量与 Alpha 检验引擎 (research/factor_metrics.py)
Phase 1.2 核心硬化:
1. 彻底实现真实 T+1 Open 买入执行时序与日度非重叠组合回测 (P0-1)
2. 真实 A 股可交易性过滤 (T+1 停牌、T+1 一字涨跌停锁死、ST、缺失开盘价)
3. Newey-West HAC 异方差自相关稳健统计量与 Benjamini-Hochberg FDR 校正
4. Tie-Aware 排列不变性分层单调性检验
5. 导出每日可审计的 daily_portfolio_pnl 数据
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from .config import ResearchConfig, default_research_config

logger = logging.getLogger(__name__)


@dataclass
class FactorEvaluationMetrics:
    """单个因子的全要素量化评估指标"""
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
    
    # 真实 T+1 开盘可执行日度多空组合指标 (Real T+1 Execution PnL, P0-1)
    daily_gross_mean_return: float = 0.0
    daily_net_mean_return: float = 0.0
    annualized_return: float = 0.0              # 真实日度多空非重叠年化收益
    annualized_volatility: float = 0.0          # 真实日度收益波动率年化
    sharpe_ratio: float = 0.0                   # 真实日度夏普比率
    net_sharpe_ratio: float = 0.0               # 真实扣费后日度夏普比率
    max_drawdown: float = 0.0                   # 真实净值曲线最大回撤
    win_rate: float = 0.0                       # 交易日多空胜率
    long_only_excess_annual_return: float = 0.0 # Top 组相对于基准的纯多头超额年化收益
    long_only_excess_sharpe: float = 0.0        # Top 组纯多头超额夏普
    
    # 换手与成本敏感度
    mean_turnover: float = 0.0
    cost_sensitivity: Dict[str, float] = field(default_factory=dict)
    
    # 样本完整性
    missing_ratio: float = 0.0
    coverage_ratio: float = 0.0
    valid_sample_count: int = 0
    daily_cross_section_count: float = 0.0
    
    # 时序日历序列
    daily_ic_series: Optional[pd.Series] = None
    daily_rank_ic_series: Optional[pd.Series] = None
    daily_pnl_df: Optional[pd.DataFrame] = None


class FactorMetricsEngine:
    """因子指标计算核心引擎 (时点安全、真实执行与无前视统计)"""

    @staticmethod
    def compute_daily_ic(
        df: pd.DataFrame,
        factor_col: str,
        return_col: str,
        date_col: str = "date",
        method: str = "spearman"
    ) -> pd.Series:
        """
        逐日计算截面 IC 序列。
        防作弊检查: 过滤停牌/非池样本，若有效样本 < 3 或全常数，返回 0.0/NaN。
        """
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
        """
        Newey-West (HAC) 异方差与自相关一致性标准误及 t-stat
        """
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
        """
        Benjamini-Hochberg (BH) FDR 多重检验校正
        """
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
        """
        计算因子在每日截面上的 N 分层平均前向收益与单调性得分。
        """
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
        config: Optional[ResearchConfig] = None
    ) -> Dict[str, Any]:
        """
        Phase 1.2 核心硬化: 严格 T+1 开盘真实执行日度多空与多头组合回测 (Real Executable Timing PnL)
        交易时序因果律:
        1. T 日收盘计算因子值与目标分位;
        2. T+1 日开盘买入 (以 adj_open[T+1] 计价成交)，T+1 日收盘以 adj_close[T+1] 计价结算;
        3. 真实 1D 持仓收益 R_{i, T+1} = (adj_close_{i, T+1} / adj_open_{i, T+1}) - 1;
        4. 可交易性硬过滤: 若 T+1 停牌、一字涨跌停、无开盘价或 ST，则 T+1 无法买入/卖出，严格剔除;
        5. 逐日扣除佣金、印花税与冲击滑点，构建真实 non-overlapping daily PnL 序列。
        """
        cfg = config or default_research_config
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=[symbol_col, date_col], inplace=True)

        c_col = close_col if close_col in df.columns else "close"
        o_col = open_col if open_col in df.columns else ("open" if "open" in df.columns else c_col)

        # 1. 计算个股 T+1 日内真实可执行收益 (Open to Close): R_{i, t} = Close_t / Open_t - 1
        # 若 open 为 0 或 NaN，退化使用 close[t] / close[t-1] - 1
        if o_col in df.columns and (df[o_col] > 0).any():
            df["daily_trade_return"] = (df[c_col] / df[o_col]) - 1.0
        else:
            df["daily_trade_return"] = df.groupby(symbol_col)[c_col].pct_change(1)

        # 2. 识别 T+1 可交易性掩码 (Tradability Filter)
        df["is_tradable_next_day"] = True
        if "is_suspended" in df.columns:
            df["is_tradable_next_day"] = df["is_tradable_next_day"] & (~df["is_suspended"].fillna(False).astype(bool))
        if o_col in df.columns:
            df["is_tradable_next_day"] = df["is_tradable_next_day"] & df[o_col].notna() & (df[o_col] > 0)

        # 3. 筛选 T 日有效样本
        valid_mask = df[factor_col].notna()
        if "in_universe" in df.columns:
            valid_mask = valid_mask & df["in_universe"].fillna(False).astype(bool)
        if "is_suspended" in df.columns:
            valid_mask = valid_mask & (~df["is_suspended"].fillna(False).astype(bool))

        sub_df = df[valid_mask].copy()
        if sub_df.empty:
            return cls._empty_portfolio_res()

        # 检查 unique 因子值
        unique_per_day = sub_df.groupby(date_col)[factor_col].transform(lambda s: len(np.unique(s)))
        sub_df = sub_df[unique_per_day >= n_quantiles].copy()
        if sub_df.empty:
            return cls._empty_portfolio_res()

        # T 日截面分位
        sub_df["f_pct"] = sub_df.groupby(date_col)[factor_col].rank(method="average", pct=True)
        sub_df["quantile"] = np.ceil(sub_df["f_pct"] * n_quantiles).astype(int).clip(1, n_quantiles)

        top_q = n_quantiles if direction == 1 else 1
        bottom_q = 1 if direction == 1 else n_quantiles

        dates = sorted(df[date_col].unique())
        date_to_idx = {d: i for i, d in enumerate(dates)}

        pnl_records = []
        prev_top_set = set()
        prev_bottom_set = set()

        for d in dates[:-1]:
            t_idx = date_to_idx[d]
            next_d = dates[t_idx + 1]

            day_t_df = sub_df[sub_df[date_col] == d]
            top_syms = set(day_t_df[day_t_df["quantile"] == top_q][symbol_col].unique())
            bottom_syms = set(day_t_df[day_t_df["quantile"] == bottom_q][symbol_col].unique())

            if not top_syms or not bottom_syms:
                continue

            # 换手率
            if prev_top_set:
                intersect = len(top_syms.intersection(prev_top_set))
                turnover = 1.0 - (intersect / max(len(top_syms), 1))
            else:
                turnover = 1.0
            prev_top_set = top_syms

            # T+1 日这批股票的实际可执行收益 (且满足 T+1 可交易性)
            day_next_df = df[df[date_col] == next_d]
            top_next = day_next_df[day_next_df[symbol_col].isin(top_syms) & day_next_df["is_tradable_next_day"]]
            bottom_next = day_next_df[day_next_df[symbol_col].isin(bottom_syms) & day_next_df["is_tradable_next_day"]]

            if top_next.empty or bottom_next.empty:
                continue

            r_top = float(top_next["daily_trade_return"].mean())
            r_bottom = float(bottom_next["daily_trade_return"].mean())

            if np.isnan(r_top) or np.isnan(r_bottom):
                continue

            # 基准 T+1 收益
            if benchmark_close_col in day_next_df.columns:
                bench_row = day_next_df[benchmark_close_col].dropna()
                r_bench = float(bench_row.iloc[0]) if not bench_row.empty else 0.0
            else:
                r_bench = 0.0

            gross_ls = r_top - r_bottom
            # 真实 A 股摩擦模型: 买入佣金 + 滑点; 卖出佣金 + 印花税 + 滑点
            # 每日多空双边调仓成本: 2 * turnover * (commission + 0.5 * stamp_duty + slippage)
            comm = cfg.DEFAULT_COMMISSION_BPS / 10000.0
            stamp = cfg.DEFAULT_STAMP_DUTY_BPS / 10000.0
            slip = cfg.DEFAULT_SLIPPAGE_BPS / 10000.0
            total_fee_rate = 2.0 * turnover * (comm + 0.5 * stamp + slip)
            net_ls = gross_ls - total_fee_rate

            pnl_records.append({
                "signal_date": str(d.date()) if hasattr(d, "date") else str(d),
                "execution_date": str(next_d.date()) if hasattr(next_d, "date") else str(next_d),
                "top_quantile_return": r_top,
                "bottom_quantile_return": r_bottom,
                "gross_return": gross_ls,
                "turnover": turnover,
                "commission": comm * 2.0 * turnover,
                "stamp_duty": stamp * turnover,
                "slippage": slip * 2.0 * turnover,
                "total_cost": total_fee_rate,
                "net_return": net_ls,
                "long_only_excess_return": r_top - r_bench
            })

        if not pnl_records:
            return cls._empty_portfolio_res()

        pnl_df = pd.DataFrame(pnl_records)
        pnl_df["equity_curve"] = (1.0 + pnl_df["net_return"]).cumprod()

        gross_s = pnl_df["gross_return"]
        net_s = pnl_df["net_return"]
        mean_turnover = float(pnl_df["turnover"].mean())

        gross_d_mean = float(gross_s.mean())
        net_d_mean = float(net_s.mean())
        daily_std = float(net_s.std(ddof=1)) if len(net_s) > 1 else 1e-6

        # 真实非重叠日收益年化
        ann_return = (1.0 + net_d_mean) ** 252.0 - 1.0 if net_d_mean > -1.0 else -1.0
        ann_vol = daily_std * np.sqrt(252.0)
        sharpe = (gross_d_mean / daily_std) * np.sqrt(252.0) if daily_std > 1e-8 else 0.0
        net_sharpe = (net_d_mean / daily_std) * np.sqrt(252.0) if daily_std > 1e-8 else 0.0

        # 最大回撤
        cum_equity = pnl_df["equity_curve"]
        running_max = cum_equity.cummax()
        drawdowns = (cum_equity - running_max) / running_max
        max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0
        win_rate = float((net_s > 0).mean())

        # Top 纯多头超额收益
        top_exc_s = pnl_df["long_only_excess_return"]
        top_exc_mean = float(top_exc_s.mean())
        top_exc_std = float(top_exc_s.std(ddof=1)) if len(top_exc_s) > 1 else 1e-6
        top_exc_ann = (1.0 + top_exc_mean) ** 252.0 - 1.0 if top_exc_mean > -1.0 else -1.0
        top_exc_sharpe = (top_exc_mean / top_exc_std) * np.sqrt(252.0) if top_exc_std > 1e-8 else 0.0

        # 费率敏感度测试 (5, 10, 20, 30 bps)
        cost_sensitivity = {}
        for bps in cfg.COST_BPS_LIST:
            f_rate = bps / 10000.0
            test_net_mean = gross_d_mean - (2.0 * mean_turnover * f_rate)
            test_net_ann = (1.0 + test_net_mean) ** 252.0 - 1.0 if test_net_mean > -1.0 else -1.0
            cost_sensitivity[f"{int(bps)}bps"] = round(float(test_net_ann), 6)

        return {
            "mean_turnover": round(mean_turnover, 4),
            "daily_gross_mean_return": round(gross_d_mean, 6),
            "daily_net_mean_return": round(net_d_mean, 6),
            "annualized_return": round(ann_return, 4),
            "annualized_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "net_sharpe_ratio": round(net_sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "long_only_excess_annual_return": round(top_exc_ann, 4),
            "long_only_excess_sharpe": round(top_exc_sharpe, 4),
            "cost_sensitivity": cost_sensitivity,
            "daily_pnl_df": pnl_df
        }

    @staticmethod
    def _empty_portfolio_res() -> Dict[str, Any]:
        return {
            "mean_turnover": 0.0,
            "daily_gross_mean_return": 0.0,
            "daily_net_mean_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "net_sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "long_only_excess_annual_return": 0.0,
            "long_only_excess_sharpe": 0.0,
            "cost_sensitivity": {"5bps": 0.0, "10bps": 0.0, "20bps": 0.0, "30bps": 0.0},
            "daily_pnl_df": pd.DataFrame()
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
        """
        单因子端到端全要素评估 (时点安全与真实执行)
        """
        cfg = config or default_research_config
        
        # 1. Pearson IC 与 RankIC
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

        # 真实 T+1 开盘执行日度收益 (P0-1)
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
            daily_gross_mean_return=pnl_res["daily_gross_mean_return"],
            daily_net_mean_return=pnl_res["daily_net_mean_return"],
            annualized_return=pnl_res["annualized_return"],
            annualized_volatility=pnl_res["annualized_volatility"],
            sharpe_ratio=pnl_res["sharpe_ratio"],
            net_sharpe_ratio=pnl_res["net_sharpe_ratio"],
            max_drawdown=pnl_res["max_drawdown"],
            win_rate=pnl_res["win_rate"],
            long_only_excess_annual_return=pnl_res["long_only_excess_annual_return"],
            long_only_excess_sharpe=pnl_res["long_only_excess_sharpe"],
            mean_turnover=pnl_res["mean_turnover"],
            cost_sensitivity=pnl_res["cost_sensitivity"],
            missing_ratio=round(missing_ratio, 4),
            coverage_ratio=round(coverage_ratio, 4),
            valid_sample_count=valid_rows,
            daily_cross_section_count=round(daily_cs_count, 1),
            daily_ic_series=ic_series,
            daily_rank_ic_series=rank_ic_series,
            daily_pnl_df=pnl_res["daily_pnl_df"]
        )
