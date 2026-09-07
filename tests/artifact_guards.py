"""CI 大产物缺失守卫。

背景: factor_matrix_300.parquet (~311MB) 等属于《确定性重建》产物,
被 .gitignore 明确排除入库 (data_storage/research/*.parquet、reports/**/*.parquet)。
全新 CI clone 天然不存在这些物理文件, 相关完整性测试在本地持有数据的环境强制校验,
在无产物环境以显式 skip 跳过 (不伪造通过, 也不误报回归)。
"""
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _skipif_missing(rel_path: str, label: str):
    return pytest.mark.skipif(
        not (_REPO_ROOT / rel_path).exists(),
        reason=(
            f"[CI-SKIP] {label} 为不入库重建产物 ({rel_path}), 本环境不存在; "
            f"完整性校验仅在持有物理数据的环境执行"
        ),
    )


require_factor_matrix = _skipif_missing(
    "data_storage/research/factor_matrix_300.parquet", "factor_matrix_300.parquet"
)
require_factor_matrix_v2 = _skipif_missing(
    "data_storage/research/factor_matrix_300_v2.parquet", "factor_matrix_300_v2.parquet"
)
require_equity_curves = _skipif_missing(
    "reports/equity_curves_all_generations.parquet", "equity_curves_all_generations.parquet"
)
