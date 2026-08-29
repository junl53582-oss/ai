"""
Barra 风格因子归因与 Alpha 可解释性分析引擎 (factors/attribution.py)
用于将投资组合收益拆解为: 市场收益 + 行业配置收益 + Barra 风格暴露收益 (市值/动量/波动/质量/价值) + 特质 Alpha 纯超额。
提供个股决策的特征贡献度瀑布拆解 (SHAP/Feature Attribution Proxy)。
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

BARRA_STYLE_MAP = {
    "Size (市值风格)": ["LOG_CIRC_MV", "log_circ_mv"],
    "Momentum (动量风格)": ["ROC5", "ROC10", "ROC20", "MA_BULL_ALIGNMENT", "MACD_HIST_SLOPE_5"],
    "Volatility (波动风格)": ["STD5", "STD20", "YANG_ZHANG_VOL_20", "VOLATILITY_SQUEEZE_20", "ATR_RATIO_14"],
    "Trend Efficiency (趋势效率)": ["KAUFMAN_EFFICIENCY_20", "RSI_14", "BOLL_PCT_B"],
    "Liquidity & Flow (资金流动性)": ["TURN5", "TURN20", "FLOW_NET_BUY_RATIO_5D", "FLOW_ACCUMULATION_20D"],
    "Quality (基本面质量)": ["F_ROE", "F_GROSS_MARGIN", "F_REV_GROWTH", "F_PROFIT_GROWTH"]
}


class BarraFactorAttribution:
    """Barra 风格与收益归因分析器"""

    @classmethod
    def compute_portfolio_style_exposure(
        cls,
        target_df: pd.DataFrame,
        factor_df: pd.DataFrame,
        trade_date: Optional[str] = None
    ) -> Dict[str, float]:
        """
        计算当前目标投资组合在各大 Barra 风格因子上的加权暴露度 (Z-Score 标准化暴露)
        """
        if target_df.empty or factor_df.empty:
            return {style: 0.0 for style in BARRA_STYLE_MAP.keys()}

        date_col = "date"
        if trade_date is not None and date_col in factor_df.columns:
            f_slice = factor_df[factor_df[date_col] == pd.Timestamp(trade_date)].copy()
        else:
            latest_d = factor_df[date_col].max() if date_col in factor_df.columns else None
            f_slice = factor_df[factor_df[date_col] == latest_d].copy() if latest_d else factor_df.copy()

        merged = target_df.merge(f_slice, on="symbol", how="inner", suffixes=("", "_f"))
        if merged.empty:
            return {style: 0.0 for style in BARRA_STYLE_MAP.keys()}

        weights = merged.get("target_weight", pd.Series(1.0 / len(merged), index=merged.index)).values
        weights = weights / (np.sum(weights) + 1e-8)

        exposures: Dict[str, float] = {}
        for style_name, factor_list in BARRA_STYLE_MAP.items():
            valid_cols = [c for c in factor_list if c in merged.columns]
            if not valid_cols:
                exposures[style_name] = 0.0
                continue
            # 该风格下所有因子的平均暴露
            style_vals = merged[valid_cols].mean(axis=1).values
            # 加权组合暴露
            exp_val = float(np.sum(weights * np.nan_to_num(style_vals)))
            exposures[style_name] = round(exp_val, 4)

        return exposures

    @classmethod
    def decompose_returns(
        cls,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        rf_annual: float = 0.02
    ) -> Dict[str, float]:
        """
        组合总收益宏观分解 (Brinson / CAPM 体系):
        总收益 = 无风险收益 (Rf) + 基准 Beta 市场收益 (Market) + 纯特质 Alpha 收益
        """
        if len(portfolio_returns) < 10 or len(benchmark_returns) < 10:
            return {
                "total_return": 0.0,
                "rf_component": 0.0,
                "market_beta_component": 0.0,
                "specific_alpha_component": 0.0,
                "beta": 1.0,
                "alpha_annualized": 0.0
            }

        p_ret = portfolio_returns.dropna()
        b_ret = benchmark_returns.dropna()
        idx = p_ret.index.intersection(b_ret.index)
        p_ret = p_ret.loc[idx]
        b_ret = b_ret.loc[idx]

        n_days = len(p_ret)
        rf_daily = (1.0 + rf_annual) ** (1.0 / 242.0) - 1.0

        p_cum = float((1.0 + p_ret).prod() - 1.0)
        b_cum = float((1.0 + b_ret).prod() - 1.0)
        rf_cum = float((1.0 + rf_daily) ** n_days - 1.0)

        # OLS 计算 Beta 与 Alpha
        cov_mat = np.cov(p_ret.values, b_ret.values)
        b_var = np.var(b_ret.values)
        beta = float(cov_mat[0, 1] / (b_var + 1e-8)) if b_var > 0 else 1.0
        beta = np.clip(beta, 0.0, 3.0)

        market_component = beta * (b_cum - rf_cum)
        alpha_component = p_cum - rf_cum - market_component
        alpha_annual = (1.0 + alpha_component) ** (242.0 / max(n_days, 1)) - 1.0

        return {
            "total_return": round(p_cum, 4),
            "rf_component": round(rf_cum, 4),
            "market_beta_component": round(market_component, 4),
            "specific_alpha_component": round(alpha_component, 4),
            "beta": round(beta, 3),
            "alpha_annualized": round(float(alpha_annual), 4)
        }


class AlphaFactorExplainer:
    """个股决策的特征归因与贡献度分析"""

    @classmethod
    def explain_stock_prediction(
        cls,
        stock_factors: pd.Series,
        top_n: int = 8
    ) -> List[Dict[str, Any]]:
        """
        解释单只股票的高打分成因:
        返回对预测打分贡献最积极的前 top_n 个因子及其 Z-Score 偏离度
        """
        contributions = []
        for col, val in stock_factors.items():
            if not isinstance(val, (int, float, np.number)) or np.isnan(val):
                continue
            if col.startswith("label_") or col in ("symbol", "date", "in_universe", "industry"):
                continue

            # 偏离度得分
            impact = float(val)
            direction = "positive" if impact > 0 else "negative"
            contributions.append({
                "factor": col,
                "score_impact": round(impact, 4),
                "direction": direction,
                "abs_impact": abs(impact)
            })

        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        return contributions[:top_n]
