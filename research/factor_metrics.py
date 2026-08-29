"""
因子基础统计量与 Alpha 检验引擎 (research/factor_metrics.py)
涵盖 Pearson IC, Spearman RankIC, ICIR, FDR 检验, Block Bootstrap 置信区间,
分层单调性, 多空组合收益, 换手率及费率敏感度测试。
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
    t_stat: float = 0.0
    p_value: float = 1.0
    fdr_p_value: float = 1.0
    
    # RankIC 指标 (主力评估依据)
    mean_rank_ic: float = 0.0
    std_rank_ic: float = 0.0
    rank_ic_ir: float = 0.0
    annualized_rank_icir: float = 0.0
    positive_rank_ic_ratio: float = 0.0
    rank_ic_t_stat: float = 0.0
    rank_ic_p_value: float = 1.0
    rank_ic_fdr_p_value: float = 1.0
    
    # Bootstrap 置信区间
    bootstrap_ci_lower: float = 0.0
    bootstrap_ci_upper: float = 0.0
    
    # 因子方向与单调性
    recommended_direction: int = 1               # 1 为正向, -1 为反向
    monotonicity_score: float = 0.0             # 分层单调性得分 [-1, 1]
    monotonicity_10q: float = 0.0
    
    # 分层收益 (5分位与10分位)
    quantile_returns_5q: Dict[str, float] = field(default_factory=dict)
    quantile_returns_10q: Dict[str, float] = field(default_factory=dict)
    
    # 多空组合指标 (Top - Bottom)
    gross_long_short_return: float = 0.0
    net_long_short_return: float = 0.0          # 默认 10 bps 成本后收益
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    net_sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    
    # 换手与成本敏感度
    mean_turnover: float = 0.0
    cost_sensitivity: Dict[str, float] = field(default_factory=dict) # bps -> net_return
    
    # 样本完整性
    missing_ratio: float = 0.0
    coverage_ratio: float = 0.0
    valid_sample_count: int = 0
    daily_cross_section_count: float = 0.0
    
    # 时序日历序列
    daily_ic_series: Optional[pd.Series] = None
    daily_rank_ic_series: Optional[pd.Series] = None
    daily_long_short_returns: Optional[pd.Series] = None


class FactorMetricsEngine:
    """因子指标计算核心引擎 (向量化与时点安全)"""

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
        使用高效向量化分组，过滤掉不足 3 个有效截面样本的日期。
        """
        valid_mask = df[factor_col].notna() & df[return_col].notna()
        if "in_universe" in df.columns:
            valid_mask = valid_mask & df["in_universe"].fillna(False).astype(bool)
        if "is_suspended" in df.columns:
            valid_mask = valid_mask & (~df["is_suspended"].fillna(False).astype(bool))
            
        sub_df = df.loc[valid_mask, [date_col, factor_col, return_col]].copy()
        if sub_df.empty:
            return pd.Series(dtype=float)

        if method == "spearman":
            # 截面秩次相关性 (RankIC)
            sub_df["f_rank"] = sub_df.groupby(date_col)[factor_col].rank()
            sub_df["r_rank"] = sub_df.groupby(date_col)[return_col].rank()
            daily_ic = sub_df.groupby(date_col).apply(
                lambda g: g["f_rank"].corr(g["r_rank"]) if len(g) >= 3 else np.nan,
                include_groups=False if hasattr(pd, "__version__") and pd.__version__ >= "2.2" else None
            )
        else:
            # 截面皮尔逊相关性 (Pearson IC)
            daily_ic = sub_df.groupby(date_col).apply(
                lambda g: g[factor_col].corr(g[return_col]) if len(g) >= 3 else np.nan,
                include_groups=False if hasattr(pd, "__version__") and pd.__version__ >= "2.2" else None
            )
        return daily_ic.dropna()

    @staticmethod
    def compute_fdr_pvalues(p_values: List[float]) -> List[float]:
        """
        Benjamini-Hochberg (BH) FDR 多重检验 p 值校正
        保证控制整体虚假发现率 (False Discovery Rate)
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
            
        # 恢复原序列顺序
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
        适应金融时间序列自相关性，避免正态性假定失效
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
        计算因子在每日截面上的 N 分层平均收益与单调性得分。
        返回: (各分组平均收益字典, 单调性得分, 逐日各组收益DataFrame)
        """
        valid_mask = df[factor_col].notna() & df[return_col].notna()
        if "in_universe" in df.columns:
            valid_mask = valid_mask & df["in_universe"].fillna(False).astype(bool)
        if "is_suspended" in df.columns:
            valid_mask = valid_mask & (~df["is_suspended"].fillna(False).astype(bool))

        sub_df = df.loc[valid_mask, [date_col, factor_col, return_col]].copy()
        if sub_df.empty:
            return {}, 0.0, pd.DataFrame()

        # 每日截面切分 N 分位 (全向量化)
        sub_df["f_pct"] = sub_df.groupby(date_col)[factor_col].rank(method="first", pct=True)
        sub_df["quantile"] = np.ceil(sub_df["f_pct"] * n_quantiles).astype(int).clip(1, n_quantiles)

        # 逐日各组平均收益
        daily_q_ret = sub_df.groupby([date_col, "quantile"])[return_col].mean().unstack("quantile")
        mean_q_ret = daily_q_ret.mean()

        q_dict = {f"Q{int(q)}": round(float(mean_q_ret.get(q, 0.0)), 6) for q in range(1, n_quantiles + 1)}

        # 计算分层单调性得分 (Spearman 秩相关系数)
        q_indices = np.arange(1, n_quantiles + 1)
        q_values = [mean_q_ret.get(q, 0.0) for q in q_indices]
        if len(q_values) >= 3 and np.std(q_values) > 1e-8:
            corr_res, _ = stats.spearmanr(q_indices, q_values)
            mono_score = float(corr_res) if not np.isnan(corr_res) else 0.0
        else:
            mono_score = 0.0

        return q_dict, round(mono_score, 4), daily_q_ret

    @staticmethod
    def compute_turnover_and_long_short(
        df: pd.DataFrame,
        factor_col: str,
        return_col: str,
        n_quantiles: int = 5,
        direction: int = 1,
        date_col: str = "date",
        cost_bps_list: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        计算 Top 组合换手率、多空组合表现及费率敏感度。
        多头分位: 若 direction==1 取 Q_max，若 direction==-1 取 Q_1
        空头分位: 与多头相反
        """
        cost_bps_list = cost_bps_list or [5.0, 10.0, 20.0, 30.0]
        valid_mask = df[factor_col].notna() & df[return_col].notna()
        if "in_universe" in df.columns:
            valid_mask = valid_mask & df["in_universe"].fillna(False).astype(bool)
        if "is_suspended" in df.columns:
            valid_mask = valid_mask & (~df["is_suspended"].fillna(False).astype(bool))

        sub_df = df.loc[valid_mask, [date_col, "symbol", factor_col, return_col]].copy()
        if sub_df.empty:
            return {
                "mean_turnover": 0.0,
                "gross_long_short_return": 0.0,
                "net_long_short_return": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "net_sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "cost_sensitivity": {f"{int(b)}bps": 0.0 for b in cost_bps_list},
                "daily_ls_series": pd.Series(dtype=float)
            }

        # 划分各日分位数 (全向量化)
        sub_df["f_pct"] = sub_df.groupby(date_col)[factor_col].rank(method="first", pct=True)
        sub_df["quantile"] = np.ceil(sub_df["f_pct"] * n_quantiles).astype(int).clip(1, n_quantiles)

        top_q = n_quantiles if direction == 1 else 1
        bottom_q = 1 if direction == 1 else n_quantiles

        dates = sorted(sub_df[date_col].unique())
        prev_top_symbols = set()
        turnover_list = []
        ls_return_list = []
        date_records = []

        for d in dates:
            day_df = sub_df[sub_df[date_col] == d]
            top_df = day_df[day_df["quantile"] == top_q]
            bottom_df = day_df[day_df["quantile"] == bottom_q]

            if top_df.empty or bottom_df.empty:
                continue

            curr_top_symbols = set(top_df["symbol"].unique())
            if prev_top_symbols:
                # 换手率 = 0.5 * sum |w_t - w_{t-1}| = 1 - (重合数 / 规模)
                intersection = len(curr_top_symbols.intersection(prev_top_symbols))
                denom = max(len(curr_top_symbols), 1)
                turnover = 1.0 - (intersection / denom)
                turnover_list.append(turnover)
            prev_top_symbols = curr_top_symbols

            r_top = top_df[return_col].mean()
            r_bottom = bottom_df[return_col].mean()
            ls_ret = r_top - r_bottom
            ls_return_list.append(ls_ret)
            date_records.append(d)

        ls_series = pd.Series(ls_return_list, index=pd.to_datetime(date_records)) if ls_return_list else pd.Series(dtype=float)
        mean_turnover = float(np.mean(turnover_list)) if turnover_list else 0.0
        gross_return = float(ls_series.mean()) if not ls_series.empty else 0.0

        # 费率敏感度测试 (Net Return = Gross Return - 2 * Turnover * fee_rate)
        cost_sensitivity = {}
        for bps in cost_bps_list:
            fee_rate = bps / 10000.0
            net_ret = gross_return - (2.0 * mean_turnover * fee_rate)
            cost_sensitivity[f"{int(bps)}bps"] = round(float(net_ret), 6)

        # 基准成本 (10 bps) 下统计指标 (以 252 交易日年化)
        default_fee = 0.0010
        net_ls_series = ls_series - (2.0 * mean_turnover * default_fee)
        std_return = float(ls_series.std()) if len(ls_series) > 1 else 0.0

        ann_return = gross_return * 252.0
        ann_vol = std_return * np.sqrt(252.0)
        sharpe = (ann_return / ann_vol) if ann_vol > 1e-6 else 0.0

        net_ann_return = float(net_ls_series.mean()) * 252.0
        net_sharpe = (net_ann_return / ann_vol) if ann_vol > 1e-6 else 0.0

        # 最大回撤
        if not ls_series.empty:
            cum_ret = (1.0 + ls_series).cumprod()
            peak = cum_ret.cummax()
            dd = (cum_ret - peak) / peak
            max_dd = float(dd.min())
            win_rate = float((ls_series > 0).mean())
        else:
            max_dd = 0.0
            win_rate = 0.0

        return {
            "mean_turnover": round(mean_turnover, 4),
            "gross_long_short_return": round(gross_return, 6),
            "net_long_short_return": round(float(net_ls_series.mean()), 6),
            "annualized_return": round(ann_return, 4),
            "annualized_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "net_sharpe_ratio": round(net_sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "cost_sensitivity": cost_sensitivity,
            "daily_ls_series": ls_series
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
        单因子端到端全指标评估
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

        t_stat = (mean_ic / (std_ic / np.sqrt(n_days))) if (std_ic > 1e-8 and n_days > 1) else 0.0
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(n_days - 1, 1)))) if n_days > 1 else 1.0
        pos_ic_ratio = float((ic_series > 0).mean()) if not ic_series.empty else 0.0

        mean_rank_ic = float(rank_ic_series.mean()) if not rank_ic_series.empty else 0.0
        std_rank_ic = float(rank_ic_series.std(ddof=1)) if len(rank_ic_series) > 1 else 0.0
        rank_ic_ir = (mean_rank_ic / std_rank_ic) if std_rank_ic > 1e-8 else 0.0
        ann_rank_icir = rank_ic_ir * np.sqrt(252.0 / max(horizon, 1))

        rank_t_stat = (mean_rank_ic / (std_rank_ic / np.sqrt(n_days))) if (std_rank_ic > 1e-8 and n_days > 1) else 0.0
        rank_p_val = float(2.0 * (1.0 - stats.t.cdf(abs(rank_t_stat), df=max(n_days - 1, 1)))) if n_days > 1 else 1.0
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

        # 4. 计算 5 分位与 10 分位收益及单调性
        q5_dict, mono_5q, _ = cls.compute_quantile_returns(df, factor_col, return_col, n_quantiles=5)
        q10_dict, mono_10q, _ = cls.compute_quantile_returns(df, factor_col, return_col, n_quantiles=10)

        # 5. 换手率、多空表现与成本敏感度
        ls_res = cls.compute_turnover_and_long_short(
            df,
            factor_col=factor_col,
            return_col=return_col,
            n_quantiles=5,
            direction=direction,
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
            t_stat=round(t_stat, 4),
            p_value=round(p_val, 6),
            fdr_p_value=round(p_val, 6), # 待多因子集合统一校正
            mean_rank_ic=round(mean_rank_ic, 4),
            std_rank_ic=round(std_rank_ic, 4),
            rank_ic_ir=round(rank_ic_ir, 4),
            annualized_rank_icir=round(ann_rank_icir, 4),
            positive_rank_ic_ratio=round(pos_rank_ic_ratio, 4),
            rank_ic_t_stat=round(rank_t_stat, 4),
            rank_ic_p_value=round(rank_p_val, 6),
            rank_ic_fdr_p_value=round(rank_p_val, 6),
            bootstrap_ci_lower=ci_lower,
            bootstrap_ci_upper=ci_upper,
            recommended_direction=direction,
            monotonicity_score=mono_5q,
            monotonicity_10q=mono_10q,
            quantile_returns_5q=q5_dict,
            quantile_returns_10q=q10_dict,
            gross_long_short_return=ls_res["gross_long_short_return"],
            net_long_short_return=ls_res["net_long_short_return"],
            annualized_return=ls_res["annualized_return"],
            annualized_volatility=ls_res["annualized_volatility"],
            sharpe_ratio=ls_res["sharpe_ratio"],
            net_sharpe_ratio=ls_res["net_sharpe_ratio"],
            max_drawdown=ls_res["max_drawdown"],
            win_rate=ls_res["win_rate"],
            mean_turnover=ls_res["mean_turnover"],
            cost_sensitivity=ls_res["cost_sensitivity"],
            missing_ratio=round(missing_ratio, 4),
            coverage_ratio=round(coverage_ratio, 4),
            valid_sample_count=valid_rows,
            daily_cross_section_count=round(daily_cs_count, 1),
            daily_ic_series=ic_series,
            daily_rank_ic_series=rank_ic_series,
            daily_long_short_returns=ls_res["daily_ls_series"]
        )
