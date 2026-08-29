"""
遗传规划、贝叶斯超参优化、产业链图网络与可交易掩码测试套件 (tests/test_genetic_and_graph.py)
"""
import pytest
import pandas as pd
import numpy as np

from factors.genetic_miner import (
    ts_delay,
    ts_delta,
    ts_rank,
    ts_corr,
    ts_std,
    decay_linear,
    safe_div,
    GeneticAlphaMiner
)
from models.hyper_tuner import BayesianHyperTuner
from models.graph_network import IndustryGraphRelationalEngine
from models.tradability_mask import TradabilityMaskEngine


class TestGeneticAlphaMiner:

    def test_time_series_operators(self):
        s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        s2 = pd.Series([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0])

        d_res = ts_delay(s1, 2)
        assert d_res.iloc[2] == 1.0

        delta_res = ts_delta(s1, 1)
        assert delta_res.iloc[1] == 1.0

        r_res = ts_rank(s1, 5)
        assert r_res.iloc[-1] == 1.0

        corr_res = ts_corr(s1, s2, 5)
        assert len(corr_res) == 10
        assert not np.isnan(corr_res.iloc[-1])

        decay_res = decay_linear(s1, 3)
        assert len(decay_res) == 10

    def test_genetic_miner_execution(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        df = pd.DataFrame({
            "open": np.linspace(10, 20, 50),
            "close": np.linspace(11, 21, 50),
            "volume": np.full(50, 1000.0)
        })
        miner = GeneticAlphaMiner(population_size=10, generations=2)
        alphas = miner.mine_alphas(df, top_k_export=2)
        assert len(alphas) == 2
        assert "formula" in alphas[0]
        assert "mean_rank_ic" in alphas[0]


class TestBayesianHyperTuner:

    def test_hyper_tuner_optimization(self):
        np.random.seed(42)
        n = 100
        X_train = pd.DataFrame(np.random.randn(n, 4), columns=["F1", "F2", "F3", "F4"])
        y_train = pd.Series(np.random.choice([0, 1], size=n))
        X_val = pd.DataFrame(np.random.randn(30, 4), columns=["F1", "F2", "F3", "F4"])
        y_val = pd.Series(np.random.choice([0, 1], size=30))

        tuner = BayesianHyperTuner(n_trials=3, task_type="classification")
        best_p, best_s = tuner.tune_lightgbm(X_train, y_train, X_val, y_val)

        assert "learning_rate" in best_p
        assert "num_leaves" in best_p
        assert best_s > 0


class TestIndustryGraphAndMaskEngine:

    def test_industry_graph_relational_engine(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame({
            "date": list(dates) * 2,
            "symbol": ["A"] * 10 + ["B"] * 10,
            "industry": ["TECH"] * 20,
            "close": np.linspace(10, 20, 20)
        })
        res = IndustryGraphRelationalEngine.compute_relational_lead_lag_features(df)
        assert "GRAPH_IND_LEAD_LAG" in res.columns
        assert "GRAPH_IND_MOM_SPREAD" in res.columns

    def test_tradability_mask_engine(self):
        df = pd.DataFrame({
            "close": [10.0, 11.0, 12.0, 13.0],
            "is_limit_up_locked": [False, True, False, False],
            "is_suspended": [False, False, True, False],
            "is_subnew": [False, False, False, True]
        })
        masked_df = TradabilityMaskEngine.apply_tradability_mask(df)
        assert "is_tradable_sample" in masked_df.columns
        # 仅有第 1 行没有任何限制
        assert list(masked_df["is_tradable_sample"]) == [True, False, False, False]
