"""
Qlib Alpha158 核心多因子子集 (Alpha158Subset) 向量化计算引擎
共 64 个精选因子 (K线形态、动量收益、长周期动量、风险调整动量、动量加速度、
均线偏离、波动率极值、量价相关性与成交量衍生)
统一基于前复权价格 (adj_close, adj_open, etc.) 计算，杜绝分红除权产生的虚假阶跃信号
"""
import logging
from typing import List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Alpha158Subset:
    """Qlib Alpha158 核心子集因子计算器 (共 64 个因子)"""

    # 动量/均线窗口 (扩展到长周期 120/250 以捕捉中长趋势)
    WINDOWS: List[int] = [5, 10, 20, 30, 60, 120, 250]
    # 波动率窗口 (覆盖全部动量窗口)
    STD_WINDOWS: List[int] = WINDOWS
    # 真实波幅窗口
    ATR_WINDOWS: List[int] = [5, 20, 60]
    # 量能窗口
    VOL_WINDOWS: List[int] = [5, 10, 20, 60]
    # 量价相关窗口
    CORR_WINDOWS: List[int] = [5, 10, 20]
    # 风险调整动量窗口 (收益 / 波动)
    ROC_STD_WINDOWS: List[int] = [20, 60, 120, 250]
    # 动量加速度 (短期均线与更长周期均线偏离的变化)
    MOM_ACC_PAIRS: List[tuple] = [(20, 60), (60, 120)]

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """输入包含多只股票时序的 DataFrame，按股票分组向量化计算 Alpha 特征"""
        logger.info("开始计算 Qlib Alpha158 精选子集因子 (64个核心因子)...")
        df = df.copy()
        df.sort_values(by=["symbol", "date"], inplace=True)

        processed_dfs = []
        for sym, grp in df.groupby("symbol"):
            processed_dfs.append(self._compute_single_stock(grp))

        features_df = pd.concat(processed_dfs, ignore_index=True) if processed_dfs else df
        features_df.sort_values(by=["date", "symbol"], inplace=True)
        features_df.reset_index(drop=True, inplace=True)

        logger.info(f"Alpha158Subset 计算完成，特征总数: {len(self.get_factor_names())}")
        return features_df

    def _compute_single_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """单只股票的向量化因子计算（优先使用前复权价格）"""
        df = df.copy()

        open_p = df["adj_open"] if "adj_open" in df.columns else df["open"]
        high_p = df["adj_high"] if "adj_high" in df.columns else df["high"]
        low_p = df["adj_low"] if "adj_low" in df.columns else df["low"]
        close_p = df["adj_close"] if "adj_close" in df.columns else df["close"]
        pct_chg = df["adj_pct_change"] if "adj_pct_change" in df.columns else df["pct_change"]
        vol = df["volume"]

        eps = 1e-8

        # ---------------- 1. K线基本形态算子 (6个) ----------------
        df["KMID"] = (close_p - open_p) / (open_p + eps)
        df["KLEN"] = (high_p - low_p) / (open_p + eps)
        df["KMID2"] = (close_p - open_p) / (high_p - low_p + eps)
        df["KUP"] = (high_p - np.maximum(open_p, close_p)) / (open_p + eps)
        df["KLOW"] = (np.minimum(open_p, close_p) - low_p) / (open_p + eps)
        df["KSFT"] = (2 * close_p - high_p - low_p) / (open_p + eps)

        # ---------------- 2. 动量与收益率算子 (ROC/MAX/MIN/MA, 4*7=28个) ----------------
        for w in self.WINDOWS:
            df[f"ROC{w}"] = (close_p - close_p.shift(w)) / (close_p.shift(w) + eps)
            df[f"MAX_RATIO_{w}"] = close_p.rolling(w).max() / (close_p + eps)
            df[f"MIN_RATIO_{w}"] = close_p.rolling(w).min() / (close_p + eps)
            df[f"MA_RATIO_{w}"] = close_p.rolling(w).mean() / (close_p + eps)

        # ---------------- 3. 价格波动率算子 (STD/ATR, 7+3=10个) ----------------
        for w in self.STD_WINDOWS:
            df[f"STD{w}"] = pct_chg.rolling(w).std()
        for w in self.ATR_WINDOWS:
            tr = np.maximum(
                high_p - low_p,
                np.maximum((high_p - close_p.shift(1)).abs(), (low_p - close_p.shift(1)).abs())
            )
            df[f"ATR_RATIO_{w}"] = tr.rolling(w).mean() / (close_p + eps)

        # ---------------- 3b. 风险调整动量 (ROC/STD, 4个) ----------------
        # 收益除以其波动率，剔除纯噪声波动，凸显稳健趋势信号
        for w in self.ROC_STD_WINDOWS:
            std_w = df[f"STD{w}"]
            df[f"ROC_STD_{w}"] = df[f"ROC{w}"] / (std_w + eps)

        # ---------------- 3c. 动量加速度 (2个) ----------------
        # 短期均线偏离相对长周期的变化: 捕捉趋势加速/减速
        for s, l in self.MOM_ACC_PAIRS:
            df[f"MOM_ACC_{s}_{l}"] = df[f"MA_RATIO_{s}"] - df[f"MA_RATIO_{l}"]

        # ---------------- 4. 成交量与量能衍生算子 (VMA/VSTD, 8个) ----------------
        for w in self.VOL_WINDOWS:
            df[f"VMA_RATIO_{w}"] = vol.rolling(w).mean() / (vol + eps)
            df[f"VSTD_{w}"] = vol.rolling(w).std() / (vol.rolling(w).mean() + eps)

        # ---------------- 5. 量价相关性与量能加权 (CORR/WVMA, 6个) ----------------
        for w in self.CORR_WINDOWS:
            df[f"CORR_PV_{w}"] = close_p.rolling(w).corr(vol).fillna(0.0)
            vol_sum = vol.rolling(w).sum() + eps
            vol_weight = vol / vol_sum
            df[f"WVMA_{w}"] = (pct_chg.abs() * vol_weight).rolling(w).sum()

        return df

    @classmethod
    def get_factor_names(cls) -> List[str]:
        """获取 Alpha158Subset 的因子名称列表 (动态生成, 共 64 个)"""
        names: List[str] = ["KMID", "KLEN", "KMID2", "KUP", "KLOW", "KSFT"]
        for w in cls.WINDOWS:
            names.extend([f"ROC{w}", f"MAX_RATIO_{w}", f"MIN_RATIO_{w}", f"MA_RATIO_{w}"])
        for w in cls.STD_WINDOWS:
            names.append(f"STD{w}")
        for w in cls.ATR_WINDOWS:
            names.append(f"ATR_RATIO_{w}")
        for w in cls.ROC_STD_WINDOWS:
            names.append(f"ROC_STD_{w}")
        for s, l in cls.MOM_ACC_PAIRS:
            names.append(f"MOM_ACC_{s}_{l}")
        for w in cls.VOL_WINDOWS:
            names.extend([f"VMA_RATIO_{w}", f"VSTD_{w}"])
        for w in cls.CORR_WINDOWS:
            names.extend([f"CORR_PV_{w}", f"WVMA_{w}"])
        return names


Alpha158Calculator = Alpha158Subset
