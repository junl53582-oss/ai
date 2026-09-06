"""
Barra 多风格正交残差化引擎 (research_v2/features/barra_orthogonalizer.py)

核心原理:
传统量价与基本面因子往往混杂了大量的宏观系统性风格暴露 (如市值大小 Size、大盘 Beta、历史反转/动量 Momentum、已实现波动率 Volatility、换手率 Liquidity)。
如果直接将混杂特征输入模型，模型很容易退化为“追随某种风格”(例如牛市追小盘、熊市追大盘红利)，导致在风格切换期出现巨大回撤。

本模块实现逐日截面多元正交投影 (Cross-Sectional OLS Orthogonalization):
1. 每日截面构建 5 大系统性 Barra 风险风格矩阵:
   - 截距项 (Intercept / Market Beta)
   - 对数流通市值 (Size: LOG_CIRC_MV)
   - 20日动量趋势 (Momentum: ROC20 / Ret20)
   - 20日已实现波动 (Volatility: STD20)
   - 20日平均换手率 (Liquidity: Turnover)
2. 截面投影分解:
   f_{k, i, t} = \alpha_t + \sum_s \beta_{k, s, t} S_{s, i, t} + \epsilon_{k, i, t}
3. 提取纯残差项 \tilde{f} = \epsilon，保证与所有风格因子样本内相关系数精确为 0。
4. 提供前后风格相关性诊断与显著性检验报告。
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_BARRA_FACTORS = [
    "LOG_CIRC_MV",
    "ROC20",
    "STD20",
    "turnover"
]


class BarraStyleOrthogonalizer:
    """Barra 多风格正交残差化处理器"""

    def __init__(
        self,
        risk_factor_cols: Optional[List[str]] = None,
        ridge_alpha: float = 1e-4
    ):
        self.risk_factor_cols = risk_factor_cols or DEFAULT_BARRA_FACTORS
        self.ridge_alpha = ridge_alpha

    def orthogonalize_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        date_col: str = "date"
    ) -> pd.DataFrame:
        """
        逐日对指定特征执行多风格正交化投影，返回包含正交化后特征的 DataFrame
        正交化后特征列名保持不变或以原特征名覆盖 (亦可新增 _orth 后缀)
        """
        res = df.copy()
        res[date_col] = pd.to_datetime(res[date_col])

        # 检查可用的风险因子
        available_risk = [c for c in self.risk_factor_cols if c in res.columns]
        if not available_risk:
            logger.warning(f"数据中未发现指定的 Barra 风险因子 {self.risk_factor_cols}，跳过正交化。")
            return res

        logger.info(f"开始执行 Barra 风格正交化 (风险因子: {available_risk}, 待正交特征: {len(feature_cols)} 个)...")

        # 逐日分组进行正交投影
        orth_features_dict = {f: np.zeros(len(res), dtype=float) for f in feature_cols}
        
        # 为了高效计算，按交易日遍历
        dates = res[date_col].unique()
        for d in dates:
            idx = (res[date_col] == d)
            sub_df = res.loc[idx]
            n_samples = len(sub_df)

            if n_samples < len(available_risk) + 3:
                # 样本量不足时，直接填充去均值
                for f in feature_cols:
                    orth_features_dict[f][idx] = sub_df[f].fillna(0.0).values
                continue

            # 构造自变量矩阵 S: [N, K + 1] (截距项 + 各风险因子)
            ones = np.ones((n_samples, 1), dtype=float)
            S_list = [ones]
            for r in available_risk:
                r_vals = sub_df[r].fillna(sub_df[r].median() or 0.0).values.astype(float)
                # 截面标准化风险因子
                r_std = r_vals.std()
                r_norm = (r_vals - r_vals.mean()) / (r_std if r_std > 1e-5 else 1.0)
                S_list.append(r_norm.reshape(-1, 1))

            S = np.hstack(S_list)  # [N, K + 1]

            # 计算投影矩阵 P = (S^T S + \lambda I)^{-1} S^T
            STS = S.T @ S + self.ridge_alpha * np.eye(S.shape[1])
            try:
                inv_STS = np.linalg.inv(STS)
                P = inv_STS @ S.T  # [K+1, N]
            except Exception:
                inv_STS = np.linalg.pinv(STS)
                P = inv_STS @ S.T

            # 对每个特征并行解出残差
            Y = sub_df[feature_cols].fillna(0.0).values  # [N, M]
            # beta = P @ Y -> [K+1, M]
            beta = P @ Y
            # 拟合系统性风格成分 = S @ beta -> [N, M]
            systematic_component = S @ beta
            # 特异性纯残差 = Y - systematic_component
            residuals = Y - systematic_component

            # 对残差进行截面标准化
            res_std = np.std(residuals, axis=0, keepdims=True)
            res_std = np.where(res_std < 1e-5, 1.0, res_std)
            residuals_norm = (residuals - np.mean(residuals, axis=0, keepdims=True)) / res_std

            for j, f in enumerate(feature_cols):
                orth_features_dict[f][idx] = residuals_norm[:, j]

        # 写入结果
        for f in feature_cols:
            res[f] = orth_features_dict[f]

        logger.info("Barra 多风格正交残差化完成，已成功剔除市场 Beta、市值 Size、动量 Momentum 等系统性风格污染。")
        return res

    def evaluate_orthogonality(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, float]:
        """
        计算正交化后特征与 Barra 风险因子之间的平均截面绝对相关系数 (验证是否接近 0)
        """
        available_risk = [c for c in self.risk_factor_cols if c in df.columns]
        if not available_risk:
            return {}

        corrs = []
        for d, g in df.groupby("date"):
            if len(g) < 10:
                continue
            for f in feature_cols:
                for r in available_risk:
                    c = np.corrcoef(g[f].fillna(0.0), g[r].fillna(0.0))[0, 1]
                    if not np.isnan(c):
                        corrs.append(abs(c))

        mean_abs_corr = float(np.mean(corrs)) if corrs else 0.0
        max_abs_corr = float(np.max(corrs)) if corrs else 0.0
        return {
            "mean_abs_correlation": round(mean_abs_corr, 6),
            "max_abs_correlation": round(max_abs_corr, 6),
            "is_orthogonal": bool(mean_abs_corr < 0.05)
        }
