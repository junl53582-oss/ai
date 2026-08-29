"""
因子基础统计量与 Alpha 检验引擎 (research/factor_metrics.py)
涵盖:
1. Pearson IC, Spearman RankIC, Naive / HAC (Newey-West) t-stat 与 FDR 校正
2. Block Bootstrap 95% 置信区间
3. 区分 Label 预测研究与真实 Non-overlapping 每日执行组合 PnL (P0-1)
4. 防作弊 Tie-aware 分层单调性检验与 Permutation Invariance (P0-5)
5. 真实日度换手率与费率敏感度测试 (P1-4)
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
    recommended_direction: int = 1               # 1 为正向, -1 为反向
    monotonicity_score: float = 0.0             # 分层单调性得分 [-1, 1]
    monotonicity_10q: float = 0.0
    
    # 分层前向收益 (用于单调性与研究诊断)
    quantile_returns_5q: Dict[str, float] = field(default_factory=dict)
    quantile_returns_10q: Dict[str, float] = field(default_factory=dict)
    
    # 真实日度可交易多空组合指标 (Real Non-Overlapping Daily PnL, P0-1)
    daily_gross_mean_return: float = 0.0
    daily_net_mean_return: float = 0.0
    annualized_return: float = 0.0              # 真实非重叠日收益年化: (1+mean)^252 - 1
    annualized_volatility: float = 0.0          # 真实日收益波动年化: std * sqrt(252)
    sharpe_ratio: float = 0.0                   # 真实每日 PnL 夏普
    net_sharpe_ratio: float = 0.0               # 扣除 10 bps 后真实每日 PnL 夏普
    max_drawdown: float = 0.0                   # 真实每日净值曲线最大回撤
    win_rate: float = 0.0                       # 交易日胜率
    
    # 换手与成本敏感度 (Generic Friction Stress Test, P1-4)
    mean_turnover: float = 0.0
    cost_sensitivity: Dict[str, float] = field(default_factory=dict) # bps -> net_annualized_return
    
    # 样本完整性
    missing_ratio: float = 0.0
    coverage_ratio: float = 0.0
    valid_sample_count: int = 0
    daily_cross_section_count: float = 0.0
    
    # 时序日历序列
    daily_ic_series: Optional[pd.Series] = None
    daily_rank_ic_series: Optional[pd.Series] = None
    daily_realized_pnl_series: Optional[pd.Series] = None


class FactorMetricsEngine:
    """因子指标计算核心引擎 (向量化、时点安全与无前视统计)"""

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
        防作弊逻辑 (P0-5):
        1. 过滤非 in_universe 与停牌股票;
        2. 若当日有效标的小于 3 或因子值为全常数 (标准差近 0 或 unique <= 1)，返回 NaN，杜绝伪造 IC。
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
            # 常数因子防作弊检查
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
        Newey-West (HAC) 异方差与自相关一致性标准误及 t-stat (P1-2)
        应对重叠 Forward Return (如 20D) 导致的序列自相关性
        """
        vals = series.dropna().values
        n = len(vals)
        if n < 3:
            return 0.0, 1.0

        mean_val = float(np.mean(vals))
        demeaned = vals - mean_val

        # gamma_0 (样本方差)
        gamma_0 = float(np.mean(demeaned ** 2))
        if gamma_0 < 1e-12:
            return 0.0, 1.0

        # 加权自相关协方差求和 (Bartlett 权重)
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
        Benjamini-Hochberg (BH) FDR 多重检验校正 (P1-1)
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
        """
        时间块重抽样 Block Bootstrap 估计均值 95% 置信区间
        """
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
        防作弊与行排列不变性 (P0-5):
        1. 若截面 unique 因子值 < n_quantiles，说明无法形成有效分组，当日置 NaN，杜绝通过 method='first' 制造假分层;
        2. 使用 method='average' 计算秩次，确保 DataFrame 行乱序排列结果逐位一致。
        """
        valid_mask = df[factor_col].notna() & df[return_col].notna()
        if "in_universe" in df.columns:
            valid_mask = valid_mask & df["in_universe"].fillna(False).astype(bool)
        if "is_suspended" in df.columns:
            valid_mask = valid_mask & (~df["is_suspended"].fillna(False).astype(bool))

        sub_df = df.loc[valid_mask, [date_col, factor_col, return_col]].copy()
        if sub_df.empty:
            return {}, 0.0, pd.DataFrame()

        # 检查每日 unique 数量
        unique_counts = sub_df.groupby(date_col)[factor_col].transform(lambda s: len(np.unique(s)))
        valid_cs_mask = unique_counts >= n_quantiles

        sub_df = sub_df[valid_cs_mask].copy()
        if sub_df.empty:
            return {}, 0.0, pd.DataFrame()

        # 每日截面切分 N 分位 (采用 rank(method='average') 保证排序置换不变性)
        sub_df["f_pct"] = sub_df.groupby(date_col)[factor_col].rank(method="average", pct=True)
        sub_df["quantile"] = np.ceil(sub_df["f_pct"] * n_quantiles).astype(int).clip(1, n_quantiles)

        # 逐日各组平均收益
        daily_q_ret = sub_df.groupby([date_col, "quantile"])[return_col].mean().unstack("quantile")
        mean_q_ret = daily_q_ret.mean()

        q_dict = {f"Q{int(q)}": round(float(mean_q_ret.get(q, 0.0)), 6) for q in range(1, n_quantiles + 1)}

        # 计算分层单调性得分 (Spearman 秩相关系数)
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
        cost_bps_list: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        P0-1 核心加固: 构建真实非重叠日度可交易多空组合收益 (Real Daily PnL Series)。
        交易时序原则 (P0-3):
        1. T 日收盘计算因子值与分位数划分;
        2. T+1 日开盘/收盘执行，持有标的获得 T+1 日真实已实现日收益率 R_{i, T+1} = (P_{i, T+1} / P_{i, T}) - 1;
        3. 严禁把 20D Forward Return 直接当每日 PnL 乘以 252 年化！
        4. 统计每日真实多空净值曲线、年化收益、年化波动、真实夏普比率、最大回撤与真实日均换手率。
        """
        cost_bps_list = cost_bps_list or [5.0, 10.0, 20.0, 30.0]
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=[symbol_col, date_col], inplace=True)

        p_col = close_col if close_col in df.columns else "close"
        # 1. 计算个股真实日收益率 (非重叠 1D Return): R_{i, t} = P_{i, t} / P_{i, t-1} - 1
        df["daily_ret_realized"] = df.groupby(symbol_col)[p_col].pct_change(1)

        # 2. 筛选 T 日因子有效且在池样本
        valid_mask = df[factor_col].notna()
        if "in_universe" in df.columns:
            valid_mask = valid_mask & df["in_universe"].fillna(False).astype(bool)
        if "is_suspended" in df.columns:
            valid_mask = valid_mask & (~df["is_suspended"].fillna(False).astype(bool))

        sub_df = df[valid_mask].copy()
        if sub_df.empty:
            return cls._empty_portfolio_res(cost_bps_list)

        # 3. 检查每日 unique 因子值 (P0-5 常数因子防作弊)
        unique_per_day = sub_df.groupby(date_col)[factor_col].transform(lambda s: len(np.unique(s)))
        sub_df = sub_df[unique_per_day >= n_quantiles].copy()
        if sub_df.empty:
            return cls._empty_portfolio_res(cost_bps_list)

        # T 日截面分位
        sub_df["f_pct"] = sub_df.groupby(date_col)[factor_col].rank(method="average", pct=True)
        sub_df["quantile"] = np.ceil(sub_df["f_pct"] * n_quantiles).astype(int).clip(1, n_quantiles)

        top_q = n_quantiles if direction == 1 else 1
        bottom_q = 1 if direction == 1 else n_quantiles

        # 4. 构建 T 日信号 -> T+1 日真实收益的映射
        dates = sorted(df[date_col].unique())
        date_to_idx = {d: i for i, d in enumerate(dates)}

        daily_long_ret = []
        daily_short_ret = []
        turnover_list = []
        pnl_dates = []
        prev_top_set = set()

        for d in dates[:-1]:
            t_idx = date_to_idx[d]
            next_d = dates[t_idx + 1]

            # T 日持仓名单
            day_t_df = sub_df[sub_df[date_col] == d]
            top_syms = set(day_t_df[day_t_df["quantile"] == top_q][symbol_col].unique())
            bottom_syms = set(day_t_df[day_t_df["quantile"] == bottom_q][symbol_col].unique())

            if not top_syms or not bottom_syms:
                continue

            # 换手率 (Top 组合变动)
            if prev_top_set:
                intersect = len(top_syms.intersection(prev_top_set))
                turnover = 1.0 - (intersect / max(len(top_syms), 1))
                turnover_list.append(turnover)
            prev_top_set = top_syms

            # T+1 日这批股票的实际日收益
            day_next_df = df[df[date_col] == next_d]
            r_top = day_next_df[day_next_df[symbol_col].isin(top_syms)]["daily_ret_realized"].mean()
            r_bottom = day_next_df[day_next_df[symbol_col].isin(bottom_syms)]["daily_ret_realized"].mean()

            if np.isnan(r_top) or np.isnan(r_bottom):
                continue

            daily_long_ret.append(r_top)
            daily_short_ret.append(r_bottom)
            pnl_dates.append(next_d)

        if not daily_long_ret:
            return cls._empty_portfolio_res(cost_bps_list)

        ls_series = pd.Series(np.array(daily_long_ret) - np.array(daily_short_ret), index=pd.to_datetime(pnl_dates))
        mean_turnover = float(np.mean(turnover_list)) if turnover_list else 0.0

        gross_daily_mean = float(ls_series.mean())
        daily_std = float(ls_series.std(ddof=1)) if len(ls_series) > 1 else 0.0

        # 真实非重叠日收益年化公式 (P0-1)
        # Annualized Return = (1 + mean_daily)^252 - 1 (复合) 或 mean_daily * 252 (线性)
        ann_return = (1.0 + gross_daily_mean) ** 252.0 - 1.0 if gross_daily_mean > -1.0 else -1.0
        ann_vol = daily_std * np.sqrt(252.0)
        sharpe = (gross_daily_mean / daily_std) * np.sqrt(252.0) if daily_std > 1e-8 else 0.0

        # 费率敏感度测试 (Generic Friction Stress Test)
        cost_sensitivity = {}
        for bps in cost_bps_list:
            fee_rate = bps / 10000.0
            # 每日扣除多空调仓摩擦: 2 * turnover * fee_rate
            net_d_mean = gross_daily_mean - (2.0 * mean_turnover * fee_rate)
            net_ann = (1.0 + net_d_mean) ** 252.0 - 1.0 if net_d_mean > -1.0 else -1.0
            cost_sensitivity[f"{int(bps)}bps"] = round(float(net_ann), 6)

        default_fee = 0.0010
        net_daily_mean = gross_daily_mean - (2.0 * mean_turnover * default_fee)
        net_ann_return = (1.0 + net_daily_mean) ** 252.0 - 1.0 if net_daily_mean > -1.0 else -1.0
        net_sharpe = (net_daily_mean / daily_std) * np.sqrt(252.0) if daily_std > 1e-8 else 0.0

        # 最大回撤 (基于非重叠日度累计净值曲线)
        cum_equity = (1.0 + ls_series).cumprod()
        running_max = cum_equity.cummax()
        drawdowns = (cum_equity - running_max) / running_max
        max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0
        win_rate = float((ls_series > 0).mean()) if not ls_series.empty else 0.0

        return {
            "mean_turnover": round(mean_turnover, 4),
            "daily_gross_mean_return": round(gross_daily_mean, 6),
            "daily_net_mean_return": round(net_daily_mean, 6),
            "annualized_return": round(ann_return, 4),
            "annualized_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "net_sharpe_ratio": round(net_sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "cost_sensitivity": cost_sensitivity,
            "daily_pnl_series": ls_series
        }

    @staticmethod
    def _empty_portfolio_res(cost_bps_list: List[float]) -> Dict[str, Any]:
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
            "cost_sensitivity": {f"{int(b)}bps": 0.0 for b in cost_bps_list},
            "daily_pnl_series": pd.Series(dtype=float)
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
        单因子端到端全指标评估 (融合 HAC 检验与真实日度 PnL)
        """
        cfg = config or default_research_config
        
        # 1. 计算 Pearson IC 与 RankIC
        ic_series = cls.compute_daily_ic(df, factor_col, return_col, method="pearson")
        rank_ic_series = cls.compute_daily_ic(df, factor_col, return_col, method="spearman")

        n_days = len(rank_ic_series)
        mean_ic = float(ic_series.mean()) if not ic_series.empty else 0.0
        std_ic = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
        ic_ir = (mean_ic / std_ic) if std_ic > 1e-8 else 0.0
        ann_icir = ic_ir * np.sqrt(252.0 / max(horizon, 1))

        # Naive vs HAC t-stat (P1-2)
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

        # 2. Block Bootstrap 置信区间
        ci_lower, ci_upper = cls.compute_block_bootstrap_ci(
            rank_ic_series,
            n_rounds=cfg.BOOTSTRAP_ROUNDS,
            block_size=cfg.BOOTSTRAP_BLOCK_SIZE,
            confidence=cfg.BOOTSTRAP_CONFIDENCE
        )

        # 3. 确定因子推荐方向
        direction = 1 if mean_rank_ic >= 0 else -1

        # 4. 计算 5 分位与 10 分位前向收益及单调性
        q5_dict, mono_5q, _ = cls.compute_quantile_returns(df, factor_col, return_col, n_quantiles=5)
        q10_dict, mono_10q, _ = cls.compute_quantile_returns(df, factor_col, return_col, n_quantiles=10)

        # 5. 真实日度非重叠多空组合收益 (P0-1)
        pnl_res = cls.compute_realized_daily_portfolio_pnl(
            df,
            factor_col=factor_col,
            direction=direction,
            n_quantiles=5,
            cost_bps_list=cfg.COST_BPS_LIST
        )

        # 6. 样本完整性统计
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
            fdr_p_value=round(hac_p, 6), # 待全因子统合 FDR 校正
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
            mean_turnover=pnl_res["mean_turnover"],
            cost_sensitivity=pnl_res["cost_sensitivity"],
            missing_ratio=round(missing_ratio, 4),
            coverage_ratio=round(coverage_ratio, 4),
            valid_sample_count=valid_rows,
            daily_cross_section_count=round(daily_cs_count, 1),
            daily_ic_series=ic_series,
            daily_rank_ic_series=rank_ic_series,
            daily_realized_pnl_series=pnl_res["daily_pnl_series"]
        )
