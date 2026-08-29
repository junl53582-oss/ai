"""
高级微观因子、施密特正交化与非对称损失函数测试套件 (tests/test_advanced_optimizations.py)
"""
import pytest
import pandas as pd
import numpy as np

from factors.registry import FactorRegistry
from factors import microstructure_advanced
from factors.orthogonalizer import GramSchmidtOrthogonalizer
from models.asymmetric_loss import AsymmetricLossObjective, AsymmetricRegressionObjective
from models.lightgbm_model import LightGBMQuantModel


class TestAdvancedMicrostructureFactors:

    def test_microstructure_factors_registered(self):
        factors = FactorRegistry.list_all_factors()
        assert "AMIHUD_ILLIQUIDITY_20" in factors
        assert "SHADOW_ASYMMETRY_RATIO" in factors
        assert "KYLE_LAMBDA_PROXY" in factors
        assert "DOWNSIDE_VOL_RATIO_20" in factors
        assert "INDUSTRY_LEAD_LAG_MOM_5D" in factors

    def test_compute_microstructure_factors(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "symbol": "600519.SH",
            "open": np.linspace(100, 120, 60),
            "high": np.linspace(105, 125, 60),
            "low": np.linspace(95, 115, 60),
            "close": np.linspace(102, 122, 60),
            "volume": np.full(60, 10000.0),
            "amount": np.full(60, 1000000.0)
        })
        res = FactorRegistry.compute_all_registered(df)
        assert "AMIHUD_ILLIQUIDITY_20" in res.columns
        assert "SHADOW_ASYMMETRY_RATIO" in res.columns
        assert not res["AMIHUD_ILLIQUIDITY_20"].isna().all()


class TestGramSchmidtOrthogonalizer:

    def test_cross_sectional_orthogonalization(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        # 构造两组高度共线性的因子
        syms = ["A", "B", "C", "D", "E"]
        rows = []
        for d in dates:
            for s in syms:
                f1 = np.random.normal(0, 1)
                f2 = 0.95 * f1 + np.random.normal(0, 0.1)  # 极高相关性
                rows.append({"date": d, "symbol": s, "F1": f1, "F2": f2})
        df = pd.DataFrame(rows)

        ortho_df = GramSchmidtOrthogonalizer.orthogonalize_cross_section(df, factor_cols=["F1", "F2"])
        assert "F1" in ortho_df.columns
        assert "F2" in ortho_df.columns

        # 检验单日截面相关性是否归零
        day1 = ortho_df[ortho_df["date"] == dates[0]]
        corr = np.corrcoef(day1["F1"], day1["F2"])[0, 1]
        assert abs(corr) < 1e-2, "正交化后截面相关性必须接近 0"


class TestAsymmetricLossObjective:

    def test_asymmetric_classification_objective(self):
        obj = AsymmetricLossObjective(false_positive_penalty=3.0)
        
        class MockTrainData:
            def get_label(self):
                return np.array([1, 0, 1, 0])

        preds = np.array([2.0, 2.0, -2.0, -2.0])  # raw logits
        grad, hess = obj(preds, MockTrainData())
        
        assert len(grad) == 4
        assert len(hess) == 4
        # 对于第 2 个样本 (实际为0，但预测为高 logits 2.0)，施加 3.0x 惩罚，其梯度应大于标准梯度
        assert grad[1] > 0.8 * 3.0

    def test_lightgbm_with_asymmetric_loss(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(80, 5), columns=[f"F_{i}" for i in range(5)])
        y = pd.Series(np.random.choice([0, 1], size=80))

        model = LightGBMQuantModel(task_type="classification", use_asymmetric_loss=True)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 80
        assert (preds >= 0.0).all() and (preds <= 1.0).all()
