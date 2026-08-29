"""
A股本土化定制特征因子计算器 (共 13 个特色因子)
针对 A 股市场的特有生态（换手率异动、涨跌停连板、资金量能冲击、市值风格与日内影线博弈）构建专属因子
"""
import logging
from typing import List
import pandas as pd
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class AShareFactorCalculator:
    """A股专属特色因子计算引擎"""

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输入包含多只股票基础行情的 DataFrame，计算 A 股专属特色特征
        """
        logger.info("开始计算 A 股定制本土化因子 (13个特色因子)...")
        df = df.copy()
        df.sort_values(by=["symbol", "date"], inplace=True)

        processed_dfs = []
        for sym, grp in df.groupby("symbol"):
            processed_dfs.append(self._compute_single_stock(grp))

        features_df = pd.concat(processed_dfs, ignore_index=True) if processed_dfs else df
        features_df.sort_values(by=["date", "symbol"], inplace=True)
        features_df.reset_index(drop=True, inplace=True)

        logger.info(f"A 股特色因子计算完成，新增特征数: {len(self.get_factor_names())}")
        return features_df

    def _compute_single_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """单只股票的 A 股特色指标计算"""
        df = df.copy()
        eps = 1e-8
        
        close_p = df["adj_close"] if "adj_close" in df.columns else df["close"]
        open_p = df["adj_open"] if "adj_open" in df.columns else df["open"]
        high_p = df["adj_high"] if "adj_high" in df.columns else df["high"]
        low_p = df["adj_low"] if "adj_low" in df.columns else df["low"]
        raw_close = df["close"] if "close" in df.columns else close_p
        amount = df["amount"] if "amount" in df.columns else (df["volume"] * close_p)
        
        # 换手率提取或近似计算
        if "turnover" in df.columns:
            turnover = df["turnover"]
        elif "LOG_CIRC_MV" in df.columns:
            turnover = amount / (np.exp(df["LOG_CIRC_MV"]) + eps)
        elif "volume" in df.columns:
            turnover = df["volume"] / (df["volume"].rolling(20).mean() * 100 + eps)
        else:
            turnover = pd.Series(0.01, index=df.index)

        pct_chg = df["adj_pct_change"] if "adj_pct_change" in df.columns else (df["pct_change"] if "pct_change" in df.columns else close_p.pct_change())
        symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) > 0 else "000001.SZ"

        limit_rate = settings.PRICE_LIMIT_CHINEXT if (symbol.startswith("300") or symbol.startswith("688")) else settings.PRICE_LIMIT_MAIN

        # ---------------- 1. 换手率异动与流动性因子 (4个) ----------------
        df["TURNOVER_SURGE_5"] = turnover / (turnover.rolling(5).mean() + eps)
        df["TURNOVER_SURGE_20"] = turnover / (turnover.rolling(20).mean() + eps)
        df["TURNOVER_STD_20"] = turnover.rolling(20).std()
        df["AMOUNT_RATIO_5_20"] = amount.rolling(5).mean() / (amount.rolling(20).mean() + eps)

        # ---------------- 2. 涨跌停博弈与连板因子 (3个) ----------------
        limit_up_price = df.get("limit_up_price", raw_close.shift(1) * (1 + limit_rate))
        df["LIMIT_UP_SPACE"] = np.maximum(0.0, (limit_up_price - raw_close) / (raw_close + eps))
        
        # 滞后 1 日涨跌停状态 (优先使用 is_limit_up_locked / is_limit_down_locked)
        is_lup = df["is_limit_up_locked"] if "is_limit_up_locked" in df.columns else df.get("is_limit_up", pd.Series(False, index=df.index))
        is_ldown = df["is_limit_down_locked"] if "is_limit_down_locked" in df.columns else df.get("is_limit_down", pd.Series(False, index=df.index))
        df["IS_LIMIT_UP_LAG1"] = is_lup.shift(1).eq(True).astype(float)
        df["IS_LIMIT_DOWN_LAG1"] = is_ldown.shift(1).eq(True).astype(float)

        # ---------------- 3. 日内多空博弈 (4个) ----------------
        entity = (close_p - open_p).abs()
        total_range = high_p - low_p + eps
        
        # 上影线占比 (抛压强度)
        df["UPPER_SHADOW_RATIO"] = (high_p - np.maximum(open_p, close_p)) / total_range
        # 下影线占比 (承接力度)
        df["LOWER_SHADOW_RATIO"] = (np.minimum(open_p, close_p) - low_p) / total_range
        # 实体占比
        df["ENTITY_RATIO"] = entity / total_range
        # 日内收盘在全天振幅中的相对分位
        df["INTRADAY_CLOSE_LOC"] = (close_p - low_p) / total_range

        # ---------------- 4. 市值与量价效率因子 ----------------
        if "log_circ_mv" in df.columns:
            df["LOG_CIRC_MV"] = df["log_circ_mv"]
        else:
            est_mv = amount / (turnover + eps)
            df["LOG_CIRC_MV"] = np.log(np.maximum(est_mv, 1e8))

        # 价格趋势效率 (Kaufman 效率比率: 净位移 / 总位移)
        net_move_10 = (close_p - close_p.shift(10)).abs()
        total_move_10 = (close_p - close_p.shift(1)).abs().rolling(10).sum() + eps
        df["EFFICIENCY_RATIO_10"] = net_move_10 / total_move_10

        # ---------------- 5. 流动性因子 (Amihud 非流动性) ----------------
        # Amihud = |日收益率| / 成交额: 单位成交金额驱动的价格冲击，越高越不流动。
        # A股小盘/低流动性股票存在显著溢价，是量价体系缺失的流动性维度。
        abs_ret = pct_chg.abs()
        amihud_raw = abs_ret / (amount + eps)
        df["AMIHUD_20"] = amihud_raw.rolling(20).mean()
        # 对数化压缩极值 (非流动性跨度可达数个数量级)
        df["AMIHUD_20_LN"] = np.log1p(df["AMIHUD_20"].clip(lower=0))

        return df

    @classmethod
    def get_factor_names(cls) -> List[str]:
        """获取所有 A 股特色因子列表 (15个)"""
        return [
            "TURNOVER_SURGE_5", "TURNOVER_SURGE_20", "TURNOVER_STD_20", "AMOUNT_RATIO_5_20",
            "LIMIT_UP_SPACE", "IS_LIMIT_UP_LAG1", "IS_LIMIT_DOWN_LAG1",
            "UPPER_SHADOW_RATIO", "LOWER_SHADOW_RATIO", "ENTITY_RATIO", "INTRADAY_CLOSE_LOC",
            "LOG_CIRC_MV", "EFFICIENCY_RATIO_10", "AMIHUD_20", "AMIHUD_20_LN"
        ]
