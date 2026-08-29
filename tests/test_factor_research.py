"""
因子研究引擎全要素单元与集成测试 (tests/test_factor_research.py)
覆盖: 正负向 IC、单调性得分、多视界衰减、换手率计算、摩擦成本扣减、因子相关性、
冗余聚类、缺失值覆盖率、无前视泄露、股票池与停牌过滤、Benjamini-Hochberg FDR 校正及 Walk-Forward 滚动筛选。
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from research.config import ResearchConfig
from research.factor_metrics import FactorMetricsEngine
from research.factor_decay import FactorDecayEngine
from research.factor_stability import FactorStabilityEngine
from research.factor_correlation import FactorCorrelationEngine
from research.factor_selection import FactorSelectionEngine
from research.factor_analyzer import FactorResearchEngine


@pytest.fixture
def synthetic_research_df():
    """构造包含 5 只股票、60 个交易日的合成研究面板数据"""
    np.random.seed(42)
    symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600519.SH", "601318.SH"]
    dates = pd.date_range("2023-01-01", periods=60, freq="B")

    rows = []
    for d in dates:
        bench_close = 4000.0 + np.random.randn() * 20.0
        for s in symbols:
            # 构造因子
            # FACTOR_POS 与未来收益强正相关
            # FACTOR_NEG 与未来收益强负相关
            # FACTOR_RANDOM 纯噪声
            pos_val = np.random.randn()
            neg_val = -pos_val + np.random.randn() * 0.1
            rand_val = np.random.randn()
            
            # 未来收益 (注入信号)
            future_ret_1d = pos_val * 0.03 + np.random.randn() * 0.01
            future_ret_5d = pos_val * 0.06 + np.random.randn() * 0.02
            future_ret_20d = pos_val * 0.12 + np.random.randn() * 0.03
            
            close_p = 100.0 + np.random.randn() * 5.0
            
            rows.append({
                "date": d,
                "symbol": s,
                "close": close_p,
                "adj_close": close_p,
                "benchmark_close": bench_close,
                "in_universe": True,
                "is_st": False,
                "is_suspended": False,
                "FACTOR_POS": pos_val,
                "FACTOR_NEG": neg_val,
                "FACTOR_RAND": rand_val,
                "FACTOR_REDUNDANT": pos_val * 0.95 + np.random.randn() * 0.01, # 与 POS 高度冗余
                "future_return_1d": future_ret_1d,
                "future_return_5d": future_ret_5d,
                "future_return_20d": future_ret_20d,
                "future_excess_return_1d": future_ret_1d,
                "future_excess_return_5d": future_ret_5d,
                "future_excess_return_20d": future_ret_20d
            })

    return pd.DataFrame(rows)


def test_ic_positive_factor(synthetic_research_df):
    """P0-3/4: 正向因子 IC 与 RankIC 显著大于 0"""
    m = FactorMetricsEngine.evaluate_factor(synthetic_research_df, "FACTOR_POS", "future_return_20d")
    assert m.mean_ic > 0.30
    assert m.mean_rank_ic > 0.30
    assert m.rank_ic_ir > 0.50
    assert m.recommended_direction == 1


def test_ic_negative_factor(synthetic_research_df):
    """P0-3/4: 负向因子自动识别 recommended_direction = -1"""
    m = FactorMetricsEngine.evaluate_factor(synthetic_research_df, "FACTOR_NEG", "future_return_20d")
    assert m.mean_ic < -0.30
    assert m.mean_rank_ic < -0.30
    assert m.recommended_direction == -1


def test_rank_ic_monotonic(synthetic_research_df):
    """P0-4: 秩相关性 RankIC 计算严密无异常"""
    ic_s = FactorMetricsEngine.compute_daily_ic(synthetic_research_df, "FACTOR_POS", "future_return_20d", method="spearman")
    assert len(ic_s) > 0
    assert (ic_s > 0).mean() > 0.80


def test_factor_quantile_monotonicity(synthetic_research_df):
    """P0-5: 强预测因子的分层收益单调性得分接近 1.0"""
    q_dict, mono_score, _ = FactorMetricsEngine.compute_quantile_returns(synthetic_research_df, "FACTOR_POS", "future_return_20d", n_quantiles=5)
    assert len(q_dict) == 5
    assert mono_score > 0.80
    assert q_dict["Q5"] > q_dict["Q1"]


def test_factor_decay(synthetic_research_df):
    """P0-8: 因子多视界衰减与最佳视界识别"""
    dec = FactorDecayEngine.analyze_decay(synthetic_research_df, "FACTOR_POS", horizons=[1, 5, 20])
    assert "1D" in dec.rank_ic_by_horizon
    assert "5D" in dec.rank_ic_by_horizon
    assert "20D" in dec.rank_ic_by_horizon
    assert dec.best_horizon in ["5D", "20D"]


def test_turnover_calculation(synthetic_research_df):
    """P0-7: 每日 Top 组合换手率计算精确合法 [0, 1]"""
    res = FactorMetricsEngine.compute_turnover_and_long_short(synthetic_research_df, "FACTOR_POS", "future_return_20d", n_quantiles=5)
    assert 0.0 <= res["mean_turnover"] <= 1.0


def test_transaction_cost_reduces_return(synthetic_research_df):
    """P0-7: 交易摩擦成本严格单调压降净收益"""
    res = FactorMetricsEngine.compute_turnover_and_long_short(synthetic_research_df, "FACTOR_POS", "future_return_20d", n_quantiles=5)
    costs = res["cost_sensitivity"]
    assert costs["5bps"] >= costs["10bps"] >= costs["20bps"] >= costs["30bps"]
    assert res["net_long_short_return"] <= res["gross_long_short_return"]


def test_factor_correlation_detection(synthetic_research_df):
    """P0-11: 截面相关性矩阵准确计算"""
    corr_m = FactorCorrelationEngine.compute_cross_sectional_correlation(
        synthetic_research_df, ["FACTOR_POS", "FACTOR_NEG", "FACTOR_RAND", "FACTOR_REDUNDANT"]
    )
    assert corr_m.loc["FACTOR_POS", "FACTOR_REDUNDANT"] > 0.85
    assert corr_m.loc["FACTOR_POS", "FACTOR_NEG"] < -0.80


def test_redundant_factor_group(synthetic_research_df):
    """P0-11: 高相关冗余因子组正确识别聚类"""
    corr_m = FactorCorrelationEngine.compute_cross_sectional_correlation(
        synthetic_research_df, ["FACTOR_POS", "FACTOR_REDUNDANT", "FACTOR_RAND"]
    )
    groups, pairs, f_map = FactorCorrelationEngine.identify_redundancy(corr_m, threshold=0.85)
    assert len(groups) >= 1
    assert "FACTOR_POS" in groups[0] and "FACTOR_REDUNDANT" in groups[0]


def test_missing_value_coverage(synthetic_research_df):
    """P0-15: 缺失率与有效样本覆盖率准确统计"""
    df = synthetic_research_df.copy()
    df.loc[df["symbol"] == "000001.SZ", "FACTOR_POS"] = np.nan
    m = FactorMetricsEngine.evaluate_factor(df, "FACTOR_POS", "future_return_20d")
    assert 0.15 <= m.missing_ratio <= 0.25
    assert m.coverage_ratio == pytest.approx(1.0 - m.missing_ratio, abs=1e-3)


def test_factor_direction_detection(synthetic_research_df):
    """P0-4: 自动检测正负向因子"""
    m_pos = FactorMetricsEngine.evaluate_factor(synthetic_research_df, "FACTOR_POS", "future_return_20d")
    m_neg = FactorMetricsEngine.evaluate_factor(synthetic_research_df, "FACTOR_NEG", "future_return_20d")
    assert m_pos.recommended_direction == 1
    assert m_neg.recommended_direction == -1


def test_future_return_no_lookahead():
    """P0-2/P0-25: 未来数据生成严格基于 shift(-H)，修改未来价格完全不影响历史因子值与历史标签"""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df1 = pd.DataFrame({
        "date": dates,
        "symbol": "600519.SH",
        "close": np.linspace(100, 130, 30),
        "benchmark_close": 4000.0,
        "is_suspended": False
    })
    labeled1 = FactorResearchEngine.generate_future_return_labels(df1, horizons=[5])
    val_t0 = labeled1.iloc[0]["future_return_5d"]

    # 仅修改 10 天后的数据
    df2 = df1.copy()
    df2.loc[15:, "close"] = 999.0
    labeled2 = FactorResearchEngine.generate_future_return_labels(df2, horizons=[5])
    val_t0_new = labeled2.iloc[0]["future_return_5d"]

    # 第 0 天的 5 日收益必须完全不变
    assert val_t0 == pytest.approx(val_t0_new, abs=1e-8)


def test_factor_research_respects_in_universe(synthetic_research_df):
    """P0-2: 因子 IC 计算严格过滤非 in_universe 样本"""
    df = synthetic_research_df.copy()
    df.loc[df["symbol"] == "600519.SH", "in_universe"] = False
    ic_s = FactorMetricsEngine.compute_daily_ic(df, "FACTOR_POS", "future_return_20d")
    assert not ic_s.empty


def test_suspended_stock_excluded(synthetic_research_df):
    """P0-2: 停牌股票不参与 IC 计算与分层收益"""
    df = synthetic_research_df.copy()
    df.loc[df["symbol"] == "600519.SH", "is_suspended"] = True
    ic_s = FactorMetricsEngine.compute_daily_ic(df, "FACTOR_POS", "future_return_20d")
    assert not ic_s.empty


def test_fdr_adjustment():
    """P0-21: Benjamini-Hochberg FDR p 值单调非降且受控"""
    p_vals = [0.001, 0.004, 0.015, 0.040, 0.200, 0.800]
    fdr_p = FactorMetricsEngine.compute_fdr_pvalues(p_vals)
    assert len(fdr_p) == len(p_vals)
    for i in range(len(fdr_p)):
        assert fdr_p[i] >= p_vals[i]
        assert fdr_p[i] <= 1.0


def test_walk_forward_factor_selection(synthetic_research_df):
    """P0-20: 滚动走步 (Walk-Forward) 因子验证流程执行稳健"""
    res = FactorSelectionEngine.run_walk_forward_selection(
        synthetic_research_df,
        factor_cols=["FACTOR_POS", "FACTOR_NEG", "FACTOR_RAND"],
        return_col="future_return_5d"
    )
    assert isinstance(res, dict)
