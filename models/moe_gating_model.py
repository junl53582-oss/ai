"""
宏观环境门控混合专家模型 (Macro-Gated Mixture of Experts - MoE) (models/moe_gating_model.py)

核心原理:
单一全局模型无论在牛市还是熊市都使用同一种参数假设，必然导致牛市进攻乏力、熊市防御踩雷。
MoE 混合专家架构包含:
1. 宏观环境门控网络 (MacroRegimeGatingNetwork):
   - 基于全市场 20MA 宽度 (Market Breadth)、基准 20 日动量趋势、20 日已实现波动率
   - 动态输出每日多空/攻防权重 w_t in [0.10, 0.90] (规避极端单模型崩塌)
2. 进攻型专家 (AggressiveExpert):
   - 擅长高贝塔、量价动量突破、主力资金流入的非线性树模型 (LightGBM/Tree)
3. 防御型专家 (DefensiveExpert):
   - 擅长低波动、高质量、风格中性化与基本面稳健性的 L2 正则化 Ridge 模型
4. 专家动态聚合预测 (MoEPredictor):
   - 截面预测分数 = w_t * Aggressive_Score + (1 - w_t) * Defensive_Score
   - 提供完整的可解释性诊断信息 (Regime Type, Gating Weight, Expert Delta)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb

logger = logging.getLogger(__name__)


class MacroRegimeGatingNetwork:
    """宏观环境门控网络: 根据全市场截面广度与基准波动率输出攻防门控权重"""

    def __init__(
        self,
        breadth_weight: float = 2.0,
        momentum_weight: float = 3.0,
        vol_penalty_weight: float = 2.5,
        min_gate_weight: float = 0.10,
        max_gate_weight: float = 0.90
    ):
        self.breadth_weight = breadth_weight
        self.momentum_weight = momentum_weight
        self.vol_penalty_weight = vol_penalty_weight
        self.min_w = min_gate_weight
        self.max_w = max_gate_weight

    def compute_regime_weights(self, daily_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算每日宏观攻防门控权重与市场状态分类。
        输入 df 需包含: date, close, benchmark_close (可选)
        """
        df = daily_features_df[["date", "symbol", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])

        # 1. 计算个股 20 日均线
        df["ma20"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        df["above_ma20"] = (df["close"] > df["ma20"]).astype(float)

        # 2. 每日全市场宽度 (Market Breadth: 位于 20MA 上方的股票比例)
        daily_breadth = df.groupby("date")["above_ma20"].mean().rename("market_breadth")
        
        # 3. 市场中位数收益率作为基准代理
        daily_market_ret = df.groupby("date")["close"].apply(lambda s: np.nanmean(s.pct_change())).fillna(0.0)
        daily_bm_mom20 = daily_market_ret.rolling(20, min_periods=5).mean().rename("benchmark_mom20")
        daily_bm_vol20 = daily_market_ret.rolling(20, min_periods=5).std().rename("benchmark_vol20")

        regime_df = pd.concat([daily_breadth, daily_bm_mom20, daily_bm_vol20], axis=1).reset_index()

        # 4. 计算门控得分与 Sigmoid 转换
        breadth_dev = regime_df["market_breadth"].fillna(0.5) - 0.5
        mom_dev = regime_df["benchmark_mom20"].fillna(0.0)
        vol_med = regime_df["benchmark_vol20"].median() or 0.015
        vol_dev = (regime_df["benchmark_vol20"].fillna(vol_med) - vol_med) / (vol_med + 1e-4)

        raw_score = (
            self.breadth_weight * breadth_dev +
            self.momentum_weight * mom_dev -
            self.vol_penalty_weight * vol_dev
        )

        # Sigmoid 映射到 [min_w, max_w]
        prob_aggressive = 1.0 / (1.0 + np.exp(-raw_score.clip(-10.0, 10.0)))
        w_agg = self.min_w + (self.max_w - self.min_w) * prob_aggressive
        regime_df["weight_aggressive"] = w_agg
        regime_df["weight_defensive"] = 1.0 - w_agg

        # 市场状态划分
        conditions = [
            regime_df["weight_aggressive"] >= 0.58,
            regime_df["weight_aggressive"] <= 0.42
        ]
        choices = ["BULL_AGGRESSIVE", "BEAR_DEFENSIVE"]
        regime_df["market_regime"] = np.select(conditions, choices, default="NEUTRAL_BALANCED")

        return regime_df[["date", "market_breadth", "benchmark_mom20", "benchmark_vol20", "weight_aggressive", "weight_defensive", "market_regime"]]


class AggressiveExpert:
    """进攻型专家 (LightGBM GBDT, 优化高贝塔与弹性突破)"""

    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.05, num_leaves: int = 31):
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        clean_X = X.fillna(0.0)
        clean_y = y.fillna(0.0)
        self.model.fit(clean_X, clean_y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("AggressiveExpert has not been fitted.")
        clean_X = X.fillna(0.0)
        return self.model.predict(clean_X)


class DefensiveExpert:
    """防御型专家 (L2 正则化 Ridge，优化低波动、抗跌与风格安全边际)"""

    def __init__(self, alpha: float = 100.0):
        self.model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        clean_X = X.fillna(0.0)
        clean_y = y.fillna(0.0)
        self.model.fit(clean_X, clean_y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("DefensiveExpert has not been fitted.")
        clean_X = X.fillna(0.0)
        return self.model.predict(clean_X)


class MoEPredictor:
    """混合专家集成预测器 (宏观门控 + 进攻专家 + 防御专家)"""

    def __init__(
        self,
        gating_network: Optional[MacroRegimeGatingNetwork] = None,
        aggressive_expert: Optional[AggressiveExpert] = None,
        defensive_expert: Optional[DefensiveExpert] = None
    ):
        self.gating = gating_network or MacroRegimeGatingNetwork()
        self.agg_expert = aggressive_expert or AggressiveExpert()
        self.def_expert = defensive_expert or DefensiveExpert()
        self.regime_cache: Dict[str, Dict[str, Any]] = {}

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str], label_col: str):
        """同时训练进攻专家与防御专家"""
        X = train_df[feature_cols]
        y = train_df[label_col]
        logger.info(f"正在训练 MoE 进攻型专家 (样本量: {len(X)})...")
        self.agg_expert.fit(X, y)
        logger.info(f"正在训练 MoE 防御型专家 (样本量: {len(X)})...")
        self.def_expert.fit(X, y)
        return self

    def predict(
        self,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        regime_info_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        执行门控混合专家动态融合预测
        """
        res = test_df.copy()
        X = res[feature_cols]

        pred_agg = self.agg_expert.predict(X)
        pred_def = self.def_expert.predict(X)

        res["pred_aggressive"] = pred_agg
        res["pred_defensive"] = pred_def

        # 计算或合并每日宏观门控权重
        if regime_info_df is None:
            regime_info_df = self.gating.compute_regime_weights(res)

        regime_dict = regime_info_df.set_index("date")["weight_aggressive"].to_dict()
        regime_type_dict = regime_info_df.set_index("date")["market_regime"].to_dict()

        res["date"] = pd.to_datetime(res["date"])
        res["w_agg"] = res["date"].map(regime_dict).fillna(0.50)
        res["market_regime"] = res["date"].map(regime_type_dict).fillna("NEUTRAL_BALANCED")

        # 动态专家权重加权
        res["pred_score"] = res["w_agg"] * res["pred_aggressive"] + (1.0 - res["w_agg"]) * res["pred_defensive"]
        return res
