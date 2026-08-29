"""
截面施密特因子正交化引擎 (factors/orthogonalizer.py)
采用 Gram-Schmidt 逐日截面正交化投影，剥离高共线性因子之间的同质化冗余信息，
提取纯净的增量残差 Alpha (Pure Incremental Residual Alpha)。
"""
import logging
from typing import List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class GramSchmidtOrthogonalizer:
    """因子施密特正交化处理器"""

    @classmethod
    def orthogonalize_cross_section(
        cls,
        df: pd.DataFrame,
        factor_cols: List[str],
        date_col: str = "date"
    ) -> pd.DataFrame:
        """
        对指定因子集合执行逐日截面 Gram-Schmidt 正交化处理:
        令 f_1' = f_1
        令 f_k' = f_k - sum_{j=1}^{k-1} (f_k . f_j' / ||f_j'||^2) * f_j'
        """
        if df.empty or len(factor_cols) <= 1:
            return df

        valid_factors = [c for c in factor_cols if c in df.columns]
        if len(valid_factors) <= 1:
            return df

        df_out = df.copy()

        def _ortho_one_day(day_df: pd.DataFrame) -> pd.DataFrame:
            mat = day_df[valid_factors].values.astype(float)
            n_rows, n_cols = mat.shape
            if n_rows < 3:
                return day_df

            # 均值中心化
            mat_centered = mat - np.nanmean(mat, axis=0, keepdims=True)
            mat_clean = np.nan_to_num(mat_centered, nan=0.0)

            # QR 分解实现 Gram-Schmidt 正交化
            # Q 矩阵的各列相互正交且范数为 1
            try:
                Q, R = np.linalg.qr(mat_clean)
                # 恢复原始方差尺度
                stds = np.nanstd(mat, axis=0, keepdims=True)
                ortho_mat = Q * stds * np.sqrt(n_rows)
                day_df[valid_factors] = ortho_mat
            except Exception:
                pass
            return day_df

        if date_col in df_out.columns:
            df_out[valid_factors] = df_out.groupby(date_col, group_keys=False)[valid_factors].apply(lambda g: _ortho_one_day(g)[valid_factors])
        else:
            df_out[valid_factors] = _ortho_one_day(df_out)[valid_factors]

        return df_out
