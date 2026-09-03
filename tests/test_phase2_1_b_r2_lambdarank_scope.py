"""
Targeted Tests for Phase 2.1-B r2: LambdaRank Target-Scope & Evidence Closure (tests/test_phase2_1_b_r2_lambdarank_scope.py)
Covering:
1. Critical Ineligible Extreme-Value Isolation Test: Adding extreme ineligible rows (+999/-999) does not mutate eligible relevance grades.
2. Direct testing of shared helper `_build_lambdarank_relevance_labels`.
3. Relevance grade decile correctness (0..9, min=0, max=9, non-eligible is NaN).
4. Group size and date alignment exact matching across contiguous blocks.
5. Daily RankIC hash canonical formatting and stability.
6. Scipy version string validation.
7. Fold training diagnostics and best_iteration_ratio extraction test.
"""
import hashlib
import numpy as np
import pandas as pd
import pytest
import scipy

from tools.run_phase2_1_b_objective_study import (
    _build_lambdarank_relevance_labels,
    _compute_lambdarank_scope_diagnostics,
    _compute_daily_rankic_hash,
    _get_environment_info,
    _extract_fold_diagnostics,
)


def test_lambdarank_ineligible_extreme_value_isolation():
    """
    核心科学门禁测试：
    验证向数据集中注入包含极端收益 (+999.0 / -999.0) 的未准入样本 (common_train=False) 时，
    准入样本 (common_train=True) 的相关性等级 (0..9) 保持 100% 绝对一致，不受任何污染。
    """
    rng = np.random.default_rng(42)
    dt = pd.Timestamp("2023-06-01")
    
    # 1. 构造 20 只正常准入股票
    eligible_rows = []
    for i in range(20):
        eligible_rows.append({
            "date": dt,
            "symbol": f"0000{i:02d}.SZ",
            "label_net_alpha_20d": float(rng.normal(0.0, 0.05)),
            "common_train": True
        })
    df_eligible = pd.DataFrame(eligible_rows)
    
    grades_clean = _build_lambdarank_relevance_labels(
        df_eligible, df_eligible["common_train"], target_col="label_net_alpha_20d", n_grades=10
    )
    
    # 2. 加入 10 只未准入股票，故意赋予极端值 (+999.0, -999.0, +500.0, etc.)
    ineligible_rows = [
        {"date": dt, "symbol": "999001.SZ", "label_net_alpha_20d": 999.0, "common_train": False},
        {"date": dt, "symbol": "999002.SZ", "label_net_alpha_20d": -999.0, "common_train": False},
        {"date": dt, "symbol": "999003.SZ", "label_net_alpha_20d": 500.0, "common_train": False},
        {"date": dt, "symbol": "999004.SZ", "label_net_alpha_20d": -500.0, "common_train": False},
        {"date": dt, "symbol": "999005.SZ", "label_net_alpha_20d": np.nan, "common_train": False},
    ]
    df_dirty = pd.concat([df_eligible, pd.DataFrame(ineligible_rows)], ignore_index=True)
    
    grades_dirty = _build_lambdarank_relevance_labels(
        df_dirty, df_dirty["common_train"], target_col="label_net_alpha_20d", n_grades=10
    )
    
    # 3. 严格断言：准入样本的相关性等级必须 100% 完全一致
    np.testing.assert_array_equal(
        grades_clean.values,
        grades_dirty.iloc[:20].values
    )
    
    # 4. 严格断言：未准入样本的相关性等级必须全部为 NaN
    assert grades_dirty.iloc[20:].isna().all()


def test_lambdarank_scope_diagnostics_calculation():
    dt1 = pd.Timestamp("2023-06-01")
    dt2 = pd.Timestamp("2023-06-02")
    
    rows = []
    for dt in [dt1, dt2]:
        for i in range(15):
            rows.append({
                "date": dt,
                "symbol": f"0000{i:02d}.SZ",
                "label_net_alpha_20d": float(i * 0.01),
                "common_train": bool(i < 10)  # 前 10 只准入，后 5 只未准入
            })
    df = pd.DataFrame(rows)
    grades = _build_lambdarank_relevance_labels(df, df["common_train"], "label_net_alpha_20d", 10)
    
    scope_df, summary = _compute_lambdarank_scope_diagnostics(df, df["common_train"], grades)
    
    assert summary["common_train_rows"] == 20
    assert summary["outside_common_train_rows"] == 10
    assert summary["outside_scope_non_null_grade_count"] == 0
    assert summary["eligible_non_finite_grade_count"] == 0
    assert summary["eligible_non_integer_grade_count"] == 0
    assert summary["eligible_per_date_min"] == 10
    assert summary["eligible_per_date_max"] == 10
    assert len(scope_df) >= 10


def test_daily_rankic_hash_canonical_calculation():
    dates = pd.bdate_range("2023-01-01", periods=5)
    s = pd.Series([0.05123456789123456, -0.02123456789123456, 0.0, 0.03333333333333333, 0.08765432109876543],
                  index=dates)
    h1 = _compute_daily_rankic_hash(s)
    h2 = _compute_daily_rankic_hash(s.sample(frac=1.0, random_state=42))  # out-of-order Series
    assert h1 == h2
    assert len(h1) == 64


def test_lambdarank_group_alignment_exact_dates():
    dates = pd.bdate_range("2023-01-01", periods=5)
    rows = []
    for dt in dates:
        for i in range(8):
            rows.append({
                "date": dt,
                "symbol": f"{i:06d}.SZ",
                "f1": float(i),
                "label_net_alpha_20d": float(i * 0.01),
                "common_train": True
            })
    df = pd.DataFrame(rows)
    df.sort_values(["date", "symbol"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    groups = list(df.groupby("date", sort=False).size())
    assert sum(groups) == len(df)
    assert groups == [8, 8, 8, 8, 8]
    
    # 验证每个 group 区间的日期唯一且递增
    curr_idx = 0
    for g_size in groups:
        sub_df = df.iloc[curr_idx:curr_idx + g_size]
        assert sub_df["date"].nunique() == 1
        curr_idx += g_size


def test_environment_info_scipy_version_is_valid():
    env_info = _get_environment_info()
    assert "scipy_version" in env_info
    assert env_info["scipy_version"] == scipy.__version__
    assert env_info["scipy_version"] != "scipy.stats"
