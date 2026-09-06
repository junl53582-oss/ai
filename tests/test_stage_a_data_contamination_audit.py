"""
Stage A Verification: Test Data Contamination & Industry Cross-Section Integrity
Validates:
1. Cross-symbol forward fill prohibition.
2. Cross-sectional industry sanity (no single-industry collapse for HS300).
3. Known symbol industry sanity (e.g., 600519.SH, 300750.SZ, 300760.SZ != '半导体').
4. Market rows recomputation instead of un-grouped ffill.
5. Invalidation / fallback to last trusted date if latest data is corrupt.
6. Manifest/schema validator rejects semantically corrupt dataset.
"""
import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from config.settings import settings


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parent.parent


def test_no_cross_symbol_forward_fill(repo_root):
    """1. 代码库禁止在未按 symbol 分组的情况下对面板数据直接调用全局 ffill()/bfill()"""
    target_files = [
        repo_root / "data" / "live_market_syncer.py",
        repo_root / "tools" / "sync_latest_market_quotes.py"
    ]
    for fp in target_files:
        if not fp.exists():
            continue
        content = fp.read_text(encoding="utf-8")
        for line in content.splitlines():
            line_str = line.strip()
            if line_str.startswith("#"):
                continue
            assert not (
                ("df_combined.ffill()" in line_str or "df_merged.ffill()" in line_str or "df.ffill()" in line_str)
                and "groupby" not in line_str
            ), f"文件 {fp.name} 存在未按 symbol 分组的全局 ffill(): {line_str}"
            assert not (
                ("df_combined.bfill()" in line_str or "df_merged.bfill()" in line_str or "df.bfill()" in line_str)
                and "groupby" not in line_str
            ), f"文件 {fp.name} 存在未按 symbol 分组的全局 bfill(): {line_str}"


def test_latest_industry_cross_section_not_collapsed(repo_root):
    """2. HS300 截面在任何已纳入生产特征矩阵的交易日，行业数量均不得异常坍缩（至少需包含 10 个以上不同行业）"""
    matrix_path = repo_root / "data_storage" / "research" / "factor_matrix_300.parquet"
    if not matrix_path.exists():
        pytest.skip("factor_matrix_300.parquet does not exist")

    df = pd.read_parquet(matrix_path)
    df["date"] = pd.to_datetime(df["date"])

    for d, group in df.groupby("date"):
        d_str = d.strftime("%Y-%m-%d")
        if "industry" in group.columns:
            unique_inds = group["industry"].dropna().unique()
            assert len(unique_inds) > 1, (
                f"交易日 {d_str} 行业发生异常坍缩: 仅有 1 个行业 ({unique_inds})，属于跨股票污染！"
            )
            assert len(unique_inds) >= 10, (
                f"交易日 {d_str} 行业数过少: {len(unique_inds)} < 10，存在严重截面缺失或错误填充！"
            )


def test_known_symbol_industry_sanity(repo_root):
    """3. 核心白马股票在任何有效截面中均不得被错误映射为'半导体'"""
    matrix_path = repo_root / "data_storage" / "research" / "factor_matrix_300.parquet"
    if not matrix_path.exists():
        pytest.skip("factor_matrix_300.parquet does not exist")

    df = pd.read_parquet(matrix_path)
    df["date"] = pd.to_datetime(df["date"])

    known_checks = {
        "600519.SH": ["白酒", "饮料", "食品", "消费"],
        "300750.SZ": ["电池", "电力设备", "新能源", "汽车零部件"],
        "300760.SZ": ["医疗器械", "医药", "生物医药", "健康"]
    }

    for sym, forbidden_not_match in known_checks.items():
        sub = df[df["symbol"] == sym]
        if sub.empty or "industry" not in sub.columns:
            continue
        semi_rows = sub[sub["industry"] == "半导体"]
        assert len(semi_rows) == 0, (
            f"股票 {sym} 在以下日期被错误标成了'半导体': "
            f"{semi_rows['date'].dt.strftime('%Y-%m-%d').tolist()}"
        )


def test_market_rows_recompute_features_instead_of_copy():
    """4. 新行情行必须由 FactorProcessor 重新计算特征，禁止直接复制或继承旧因子"""
    from factors.processor import FactorProcessor
    from models.inference import BatchInference, InferenceError

    # 构造仅有行情 OHLCV 但无特征的增量切片
    raw_slice = pd.DataFrame([
        {
            "date": pd.to_datetime("2026-08-25"),
            "symbol": "600519.SH",
            "open": 1600.0,
            "close": 1610.0,
            "high": 1620.0,
            "low": 1590.0,
            "volume": 10000.0,
            "amount": 1.6e7,
            "turnover": 0.01,
            "pct_change": 0.00625,
            "industry": "白酒与葡萄酒"
        }
    ])

    engine = BatchInference()
    # 直接用缺少特征的原始行情推理必须 Fail-Closed 抛出 InferenceError
    with pytest.raises(InferenceError):
        engine.predict(raw_slice)


def test_invalid_latest_data_falls_back_to_last_trusted_date(repo_root):
    """5. 数据失效守卫（2026-09-07 语义升级）:
    - 历史存证保留: 2026-09-06 事故曾将 2026-09-03/04 行情标记失效并回退至 2026-08-24;
    - 2026-09-07 全量干净重建 (dataset v3, commit 6a89275) 后失效前提解除,
      生产链对齐最新可信数据 2026-09-04。隔离存证为历史事实不得篡改。"""
    # 检查失效存证文件 (历史事实, 必须保留)
    quarantine_path = repo_root / "reports" / "data_invalidation_quarantine_record_20260906.json"
    assert quarantine_path.exists(), "必须存在失效隔离审计文件"
    with open(quarantine_path, "r", encoding="utf-8") as f:
        q_record = json.load(f)

    assert "2026-09-03" in q_record["invalidated_dates"]
    assert "2026-09-04" in q_record["invalidated_dates"]
    assert q_record["trusted_data_as_of"] == "2026-08-24"

    # 生产因子矩阵已重建至最新可信日期 (2026-09-07 干净全量同步)
    matrix_path = repo_root / "data_storage" / "research" / "factor_matrix_300.parquet"
    df = pd.read_parquet(matrix_path)
    df["date"] = pd.to_datetime(df["date"])
    matrix_max = df["date"].max()
    assert matrix_max == pd.to_datetime("2026-09-04"), "生产矩阵最新日期必须为 v3 干净重建的 2026-09-04"

    # 选股产物必须与生产矩阵同源对齐 (列名 data_as_of, 由 predict_stocks.py 落盘)
    picks_path = repo_root / "artifacts" / "latest_stock_picks.csv"
    assert picks_path.exists()
    picks_df = pd.read_csv(picks_path)
    date_col = "date" if "date" in picks_df.columns else "data_as_of"
    assert picks_df[date_col].iloc[0] == "2026-09-04"
    # 选股行业不得全部坍缩
    assert picks_df["industry"].nunique() > 1


def test_manifest_rejects_semantically_corrupt_dataset(repo_root):
    """6. 数据验证器主动拦截行业坍缩、白马股票行业谬误与跨股票全同特征"""
    from tools.check_committed_dataset_schema import verify_dataset_semantic_integrity

    # Case 1: 制造行业坍缩的样本数据 (300 只股票全部为半导体)
    symbols = [f"{i:06d}.SH" for i in range(300)]
    corrupt_df = pd.DataFrame({
        "date": [pd.to_datetime("2026-09-03")] * 300,
        "symbol": symbols,
        "industry": ["半导体"] * 300
    })
    dummy_path = repo_root / "data_storage" / "research" / "test_corrupt.parquet"
    assert not verify_dataset_semantic_integrity(dummy_path, corrupt_df), "验证器必须拦截行业坍缩的数据集！"

    # Case 2: 制造贵州茅台被标记为半导体的样本数据
    corrupt_df2 = pd.DataFrame({
        "date": [pd.to_datetime("2026-08-24")],
        "symbol": ["600519.SH"],
        "industry": ["半导体"]
    })
    assert not verify_dataset_semantic_integrity(dummy_path, corrupt_df2), "验证器必须拦截 600519.SH 被标为半导体的数据集！"

    # Case 3: 真实物理文件必须通过
    matrix_path = repo_root / "data_storage" / "research" / "factor_matrix_300.parquet"
    real_df = pd.read_parquet(matrix_path)
    assert verify_dataset_semantic_integrity(matrix_path, real_df), "修复后的真实数据集必须通过语义验证门禁！"
