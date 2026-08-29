"""
现代投资组合凸优化引擎 (strategy/optimizer.py)
基于 Scipy 原生优化库构建，提供风险平价 (Risk Parity)、逆波动率加权、打分 Softmax、
以及带行业集中度上限与换手率惩罚的二次规划 (Bounded QP Optimization)。
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class BasePortfolioOptimizer(ABC):
    """组合优化器抽象基类"""

    @abstractmethod
    def optimize(
        self,
        df: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        max_sector_exposure: float = 0.30,
        max_stock_weight: float = 0.20
    ) -> pd.Series:
        """返回标的代码与目标权重的 Series"""
        pass


class EqualWeightOptimizer(BasePortfolioOptimizer):
    """等权基准分配"""

    def optimize(
        self,
        df: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        max_sector_exposure: float = 0.30,
        max_stock_weight: float = 0.20
    ) -> pd.Series:
        n = len(df)
        if n == 0:
            return pd.Series(dtype=float)
        w = pd.Series(1.0 / n, index=df.index)
        return w


class InverseVolOptimizer(BasePortfolioOptimizer):
    """波动率倒数加权 (Inverse Volatility)"""

    def optimize(
        self,
        df: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        max_sector_exposure: float = 0.30,
        max_stock_weight: float = 0.20
    ) -> pd.Series:
        n = len(df)
        if n == 0:
            return pd.Series(dtype=float)
        vol_col = "STD20" if "STD20" in df.columns else "std"
        if vol_col in df.columns:
            vols = df[vol_col].replace(0, np.nan).fillna(0.02)
            inv_v = 1.0 / (vols + 1e-6)
            w = inv_v / inv_v.sum()
            return pd.Series(w.values, index=df.index)
        return pd.Series(1.0 / n, index=df.index)


class ScoreWeightedOptimizer(BasePortfolioOptimizer):
    """预测置信度 Softmax 加权"""

    def __init__(self, temperature: float = 2.0):
        self.temperature = temperature

    def optimize(
        self,
        df: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        max_sector_exposure: float = 0.30,
        max_stock_weight: float = 0.20
    ) -> pd.Series:
        n = len(df)
        if n == 0:
            return pd.Series(dtype=float)
        scores = df["pred_score"].values if "pred_score" in df.columns else np.ones(n)
        exp_s = np.exp((scores - np.mean(scores)) * self.temperature)
        w = exp_s / np.sum(exp_s)
        return pd.Series(w, index=df.index)


class RiskParityOptimizer(BasePortfolioOptimizer):
    """风险平价优化器 (Risk Parity / Equal Risk Contribution)"""

    def optimize(
        self,
        df: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        max_sector_exposure: float = 0.30,
        max_stock_weight: float = 0.20
    ) -> pd.Series:
        n = len(df)
        if n <= 1:
            return pd.Series(1.0 if n == 1 else [], index=df.index)

        # 构建近似协方差矩阵 (基于 STD20 估算，若无则使用标准方差)
        vol_col = "STD20" if "STD20" in df.columns else "std"
        if vol_col in df.columns:
            vols = df[vol_col].replace(0, np.nan).fillna(0.02).values
        else:
            vols = np.full(n, 0.02)

        # 假设跨资产相关系数常数 rho = 0.3
        rho = 0.3
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        cov = np.outer(vols, vols) * corr

        def risk_budget_objective(w):
            w = np.array(w)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)) + 1e-8)
            marginal_contrib = np.dot(cov, w) / port_vol
            risk_contrib = w * marginal_contrib
            target_risk = port_vol / n
            return np.sum((risk_contrib - target_risk) ** 2) * 1e4

        init_w = np.ones(n) / n
        bounds = [(0.0, max_stock_weight) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(
            risk_budget_objective,
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6}
        )

        if res.success:
            w = res.x / np.sum(res.x)
            return pd.Series(w, index=df.index)
        else:
            logger.warning(f"Risk Parity 求解未完全收敛 ({res.message})，回退至逆波动率加权")
            inv_v = 1.0 / (vols + 1e-6)
            return pd.Series(inv_v / np.sum(inv_v), index=df.index)


class ConstrainedQPOptimizer(BasePortfolioOptimizer):
    """
    带行业集中度上限与换手率惩罚的现代二次规划优化器 (Mean-Variance QP)
    目标: max  w^T * mu - 0.5 * gamma * w^T * Sigma * w - lambda_turnover * ||w - w_prev||_1
    """

    def __init__(self, risk_aversion: float = 1.0, turnover_penalty: float = 0.05):
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty

    def optimize(
        self,
        df: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        max_sector_exposure: float = 0.30,
        max_stock_weight: float = 0.20
    ) -> pd.Series:
        n = len(df)
        if n == 0:
            return pd.Series(dtype=float)
        if n == 1:
            return pd.Series([1.0], index=df.index)

        # 预期收益向量 mu
        if "pred_score" in df.columns:
            mu = df["pred_score"].values
            mu = (mu - np.mean(mu)) / (np.std(mu) + 1e-6)
        else:
            mu = np.zeros(n)

        # 估算方差
        vol_col = "STD20" if "STD20" in df.columns else "std"
        vols = df[vol_col].replace(0, np.nan).fillna(0.02).values if vol_col in df.columns else np.full(n, 0.02)
        cov = np.diag(vols ** 2)

        # 前一日持仓权重向量 w_prev
        symbols = df["symbol"].values if "symbol" in df.columns else np.arange(n)
        w_prev = np.zeros(n)
        if current_weights:
            for i, sym in enumerate(symbols):
                w_prev[i] = current_weights.get(sym, 0.0)

        industries = df["industry"].fillna("UNKNOWN").values if "industry" in df.columns else np.full(n, "UNKNOWN")
        unique_inds = np.unique(industries)

        def objective(w):
            ret = np.dot(w, mu)
            risk = 0.5 * self.risk_aversion * np.dot(w.T, np.dot(cov, w))
            turnover = self.turnover_penalty * np.sum(np.abs(w - w_prev))
            return -(ret - risk - turnover)

        init_w = np.ones(n) / n
        bounds = [(0.0, max_stock_weight) for _ in range(n)]

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        # 行业上限约束
        for ind in unique_inds:
            ind_mask = (industries == ind)
            constraints.append({
                "type": "ineq",
                "fun": lambda w, m=ind_mask: max_sector_exposure - np.sum(w[m])
            })

        res = minimize(
            objective,
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6}
        )

        if res.success:
            w = res.x
            w = np.clip(w, 0.0, max_stock_weight)
            w = w / np.sum(w)
            return pd.Series(w, index=df.index)
        else:
            logger.warning(f"QP 凸优化求解未收敛 ({res.message})，回退至基础等权")
            return pd.Series(1.0 / n, index=df.index)


def get_optimizer(method_name: str) -> BasePortfolioOptimizer:
    """优化器工厂函数"""
    method_name = method_name.lower()
    if method_name in ["equal", "equal_weight"]:
        return EqualWeightOptimizer()
    elif method_name in ["inv_vol", "inverse_volatility"]:
        return InverseVolOptimizer()
    elif method_name in ["score", "score_weighted"]:
        return ScoreWeightedOptimizer()
    elif method_name in ["risk_parity", "equal_risk"]:
        return RiskParityOptimizer()
    elif method_name in ["qp", "constrained_qp", "mean_variance"]:
        return ConstrainedQPOptimizer()
    else:
        logger.info(f"未指定或未知优化器: {method_name}，默认采用等权优化器")
        return EqualWeightOptimizer()
