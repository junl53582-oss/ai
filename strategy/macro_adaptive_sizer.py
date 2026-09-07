"""
宏观自适应动态总仓位调节与 ATR 风险平价分配引擎 (strategy/macro_adaptive_sizer.py)

核心原理:
传统量化策略常年保持固定高仓位 (如 95% 或 100%)，在结构性熊市或系统性流动性危机中，
即使选出全市场最抗跌的 8 只股票，也难逃贝塔下行泥沙俱下，产生 25%~40% 的深幅回撤。

本模块实现两级资金自适应管理:
1. 宏观环境自适应总仓位管理 (Macro-Adaptive Exposure Sizing, E_t in [0.25, 1.00]):
   - 依据全市场 20MA 宽度 (Market Breadth)、基准 20 日动量、20 日已实现波动率
   - 多头进攻期: E_t = 0.95 ~ 1.00 (满仓主攻)
   - 震荡平衡期: E_t = 0.65 ~ 0.80 (适度控制)
   - 系统性下行/流动性危机: E_t = 0.25 ~ 0.40 (自动保留 60%~75% 现金避险)
2. ATR 波动率倒数风险平价分配 (ATR Inverse Volatility Allocation):
   - 个股权重与其相对真实波幅 ATR% 成反比，高波概念股轻仓、低波稳健龙头重仓
   - 严格执行 30% 行业暴露硬上限
3. 多日自适应净值与回撤仿真回测 (含 20bps 交易成本核算)
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MacroAdaptiveExposureManager:
    """宏观自适应总仓位与 ATR 风险头寸管理器"""

    def __init__(
        self,
        min_exposure: float = 0.25,
        max_exposure: float = 1.00,
        neutral_exposure: float = 0.75,
        atr_window: int = 14,
        max_sector_exposure: float = 0.30,
        fee_bps: float = 20.0
    ):
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.neutral_exposure = neutral_exposure
        self.atr_window = atr_window
        self.max_sector_exposure = max_sector_exposure
        self.fee_rate = fee_bps / 10000.0

    def compute_daily_macro_exposure(
        self,
        market_df: pd.DataFrame,
        date_col: str = "date"
    ) -> pd.DataFrame:
        """
        计算每日市场宏观状态与推荐总仓位 E_t
        输入 df 需包含: date, symbol, close
        """
        df = market_df[["date", "symbol", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])

        # 1. 计算 20 日均线与全市场宽度
        df["ma20"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        df["above_ma20"] = (df["close"] > df["ma20"]).astype(float)
        daily_breadth = df.groupby("date")["above_ma20"].mean().rename("market_breadth")

        # 2. 市场平均收益率作为指数代理
        daily_mkt_ret = df.groupby("date")["close"].apply(lambda s: np.nanmean(s.pct_change())).fillna(0.0)
        daily_bm_mom20 = daily_mkt_ret.rolling(20, min_periods=5).mean().rename("benchmark_mom20")
        daily_bm_vol20 = daily_mkt_ret.rolling(20, min_periods=5).std().rename("benchmark_vol20")

        regime_df = pd.concat([daily_breadth, daily_bm_mom20, daily_bm_vol20], axis=1).reset_index()

        # 3. 判定宏观状态并输出自适应仓位 E_t
        exposures = []
        regimes = []
        vol_med = regime_df["benchmark_vol20"].median() or 0.015

        for _, row in regime_df.iterrows():
            b = row["market_breadth"]
            m = row["benchmark_mom20"]
            v = row["benchmark_vol20"]

            if b >= 0.55 and m >= 0.0:
                # 多头主升期
                exp = self.max_exposure
                reg = "BULL_FULL_EXPOSURE"
            elif b < 0.38 or m < -0.002 or v > vol_med * 1.5:
                # 破位熊市或恐慌杀跌
                exp = self.min_exposure
                reg = "BEAR_DEFENSIVE_CASH"
            else:
                # 震荡平衡市
                exp = self.neutral_exposure
                reg = "NEUTRAL_BALANCED"

            exposures.append(exp)
            regimes.append(reg)

        regime_df["target_gross_exposure"] = exposures
        regime_df["cash_ratio"] = 1.0 - np.array(exposures)
        regime_df["market_regime"] = regimes

        return regime_df[["date", "market_breadth", "benchmark_mom20", "benchmark_vol20", "target_gross_exposure", "cash_ratio", "market_regime"]]

    def allocate_portfolio_weights(
        self,
        selected_stocks_df: pd.DataFrame,
        target_gross_exposure: float,
        industry_col: str = "industry"
    ) -> pd.DataFrame:
        """
        在入选股票池上执行 ATR 波动率倒数加权，并缩放至 target_gross_exposure，严格遵守行业上限
        输入 selected_stocks_df 需包含: symbol, close, high, low, industry
        """
        df = selected_stocks_df.copy()
        n = len(df)
        if n == 0:
            return df

        # 计算相对振幅 proxy: (high - low) / close
        if "high" in df.columns and "low" in df.columns and "close" in df.columns:
            rel_vol = ((df["high"] - df["low"]) / (df["close"] + 1e-5)).clip(0.01, 0.20)
            inv_vol = 1.0 / rel_vol
            raw_weights = inv_vol / inv_vol.sum()
        else:
            raw_weights = np.full(n, 1.0 / n)

        # 缩放到目标总仓位
        scaled_weights = raw_weights * target_gross_exposure
        df["target_weight"] = scaled_weights

        # 行业上限裁剪 (单行业权重不得超过 max_sector_exposure)
        if industry_col in df.columns:
            df[industry_col] = df[industry_col].fillna("UNKNOWN")
            for ind, g_idx in df.groupby(industry_col).groups.items():
                sec_w = df.loc[g_idx, "target_weight"].sum()
                if sec_w > self.max_sector_exposure:
                    df.loc[g_idx, "target_weight"] *= (self.max_sector_exposure / sec_w)

        return df

    def simulate_adaptive_backtest(
        self,
        predictions_df: pd.DataFrame,
        macro_exposure_df: pd.DataFrame,
        returns_col: str = "target_ret_1d",
        top_k: int = 8,
        date_col: str = "date"
    ) -> pd.DataFrame:
        """
        按时间序列执行多日宏观自适应动态仓位回测
        """
        dates = sorted(predictions_df[date_col].unique())
        exp_map = macro_exposure_df.set_index(date_col)["target_gross_exposure"].to_dict()
        reg_map = macro_exposure_df.set_index(date_col)["market_regime"].to_dict()

        history_records = []
        prev_weights: Dict[str, float] = {}

        for d in dates:
            daily_df = predictions_df[predictions_df[date_col] == d].copy()
            daily_df = daily_df.sort_values(by="pred_score", ascending=False).reset_index(drop=True)
            top_df = daily_df.head(top_k).copy()

            target_exp = exp_map.get(pd.to_datetime(d), self.max_exposure)
            regime_name = reg_map.get(pd.to_datetime(d), "NEUTRAL_BALANCED")

            weighted_top = self.allocate_portfolio_weights(top_df, target_exp)
            curr_weights = dict(zip(weighted_top["symbol"], weighted_top["target_weight"]))

            # 计算换手率
            all_syms = set(prev_weights.keys()).union(set(curr_weights.keys()))
            turnover = 0.5 * sum(abs(curr_weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in all_syms)

            # 计算收益
            if returns_col in daily_df.columns:
                ret_map = daily_df.set_index("symbol")[returns_col].to_dict()
                gross_ret = sum(curr_weights.get(s, 0.0) * ret_map.get(s, 0.0) for s in curr_weights)
            else:
                gross_ret = 0.0

            net_ret = gross_ret - turnover * self.fee_rate

            history_records.append({
                "date": d,
                "gross_exposure": target_exp,
                "cash_ratio": 1.0 - target_exp,
                "market_regime": regime_name,
                "turnover": turnover,
                "gross_return": gross_ret,
                "net_return": net_ret
            })

            prev_weights = curr_weights

        res_df = pd.DataFrame(history_records)
        # 计算累计净值与最大回撤
        res_df["cumulative_net"] = (1.0 + res_df["net_return"]).cumprod()
        res_df["cummax"] = res_df["cumulative_net"].cummax()
        res_df["drawdown"] = (res_df["cumulative_net"] - res_df["cummax"]) / res_df["cummax"]

        return res_df
