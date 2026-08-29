"""
V8.0 新增核心特性自动化单元测试套件 (tests/test_v8_features.py)
测试范围:
1. 因子注册器 (FactorRegistry) 与另类因子 (AlternativeFactors)
2. 多模型动态自适应融合器 (EnsembleQuantModel)
3. 现代组合凸优化器 (Equal, InvVol, ScoreWeighted, RiskParity, ConstrainedQP)
4. 仿真交易沙盒 (PaperBroker) 与 MiniQMT 网关适配
5. 多渠道消息报告格式化 (MessageNotifier)
"""
import pytest
import numpy as np
import pandas as pd

from factors.registry import FactorRegistry
from factors import alternative_factors
from models.ensemble_model import EnsembleQuantModel
from strategy.optimizer import (
    EqualWeightOptimizer,
    InverseVolOptimizer,
    ScoreWeightedOptimizer,
    RiskParityOptimizer,
    ConstrainedQPOptimizer,
    get_optimizer
)
from execution.broker_base import OrderSide, OrderType, ExecutionStatus
from execution.paper_broker import PaperBroker
from execution.miniqmt_broker import MiniQMTBroker
from scheduler.notifier import MessageNotifier


class TestFactorRegistryAndAlternatives:
    """1. 因子工厂与另类因子测试"""

    def test_factor_registry_metadata(self):
        factors = FactorRegistry.list_all_factors()
        assert len(factors) >= 6, "已注册因子数应不小于 6 个"
        assert "FLOW_NET_BUY_RATIO_5D" in factors
        assert "VOLATILITY_SQUEEZE_20" in factors
        assert "MA_BULL_ALIGNMENT" in factors
        assert "VP_DIVERGENCE_10D" in factors

        meta_df = FactorRegistry.get_metadata_df()
        assert not meta_df.empty
        assert "category" in meta_df.columns
        assert "description" in meta_df.columns

    def test_compute_all_registered_factors(self):
        # 构造单只股票虚拟时序行情
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "symbol": "600519.SH",
            "open": np.linspace(100, 150, 80),
            "high": np.linspace(105, 155, 80),
            "low": np.linspace(95, 145, 80),
            "close": np.linspace(102, 152, 80),
            "volume": np.random.uniform(10000, 50000, 80),
            "amount": np.random.uniform(1000000, 5000000, 80)
        })
        res_df = FactorRegistry.compute_all_registered(df)
        assert "FLOW_NET_BUY_RATIO_5D" in res_df.columns
        assert "VOLATILITY_SQUEEZE_20" in res_df.columns
        assert "MA_BULL_ALIGNMENT" in res_df.columns
        # 检查数值不是全部 NaN
        assert not res_df["MA_BULL_ALIGNMENT"].dropna().empty


class TestEnsembleQuantModel:
    """2. 多模型动态自适应融合器测试"""

    def test_ensemble_classification_fit_and_predict(self):
        np.random.seed(42)
        n_samples = 200
        n_features = 8
        feature_names = [f"feat_{i}" for i in range(n_features)]
        
        X_train = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)
        y_train = pd.Series(np.random.choice([0, 1], size=n_samples))

        X_val = pd.DataFrame(np.random.randn(50, n_features), columns=feature_names)
        y_val = pd.Series(np.random.choice([0, 1], size=50))

        ensemble = EnsembleQuantModel(task_type="classification", model_types=["lightgbm", "random_forest", "linear"])
        ensemble.fit(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feature_names)

        assert len(ensemble.models) == 3
        assert len(ensemble.model_weights) == 3
        assert sum(ensemble.model_weights.values()) == pytest.approx(1.0, abs=1e-2)

        preds = ensemble.predict(X_val)
        assert len(preds) == 50
        assert (preds >= 0.0).all() and (preds <= 1.0).all()

        imp_df = ensemble.get_feature_importance(top_n=5)
        assert not imp_df.empty
        assert len(imp_df) <= 5

    def test_double_ensemble_and_mlp(self):
        from models.double_ensemble import DoubleEnsembleQuantModel
        from models.deep_tabular import TabularMLPQuantModel

        np.random.seed(42)
        n_samples = 120
        n_features = 10
        feature_names = [f"FEAT_{i}" for i in range(n_features)]
        X_train = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)
        y_train = pd.Series(np.random.choice([0, 1], size=n_samples))
        X_val = pd.DataFrame(np.random.randn(40, n_features), columns=feature_names)
        y_val = pd.Series(np.random.choice([0, 1], size=40))

        # 1. DoubleEnsemble 测试
        de = DoubleEnsembleQuantModel(task_type="classification", n_sub_models=3)
        de.fit(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feature_names)
        de_preds = de.predict(X_val)
        assert len(de_preds) == 40
        assert (de_preds >= 0.0).all() and (de_preds <= 1.0).all()

        # 2. TabularMLP 测试
        mlp = TabularMLPQuantModel(task_type="classification", hidden_layer_sizes=(32, 16), max_iter=50)
        mlp.fit(X_train, y_train, feature_names=feature_names)
        mlp_preds = mlp.predict(X_val)
        assert len(mlp_preds) == 40
        assert (mlp_preds >= 0.0).all() and (mlp_preds <= 1.0).all()


class TestPortfolioOptimizers:
    """3. 组合优化器测试"""

    @pytest.fixture
    def sample_portfolio_df(self):
        return pd.DataFrame({
            "symbol": ["600519.SH", "000858.SZ", "601318.SH", "300750.SZ", "600036.SH"],
            "industry": ["食品饮料", "食品饮料", "非银金融", "电力设备", "银行"],
            "pred_score": [0.85, 0.78, 0.72, 0.68, 0.65],
            "STD20": [0.015, 0.018, 0.022, 0.035, 0.012],
            "close": [1600.0, 140.0, 48.0, 190.0, 32.0]
        })

    def test_equal_weight_optimizer(self, sample_portfolio_df):
        opt = EqualWeightOptimizer()
        w = opt.optimize(sample_portfolio_df)
        assert len(w) == 5
        assert w.sum() == pytest.approx(1.0, abs=1e-4)
        assert (w == 0.2).all()

    def test_inverse_vol_optimizer(self, sample_portfolio_df):
        opt = InverseVolOptimizer()
        w = opt.optimize(sample_portfolio_df)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)
        # 波动率最小的 600036.SH (银行) 权重应该最大
        assert w.iloc[4] > w.iloc[3]

    def test_score_weighted_optimizer(self, sample_portfolio_df):
        opt = ScoreWeightedOptimizer()
        w = opt.optimize(sample_portfolio_df)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)
        # 预测最高分 600519.SH 权重应该最大
        assert w.iloc[0] > w.iloc[4]

    def test_risk_parity_optimizer(self, sample_portfolio_df):
        opt = RiskParityOptimizer()
        w = opt.optimize(sample_portfolio_df, max_stock_weight=0.30)
        assert w.sum() == pytest.approx(1.0, abs=1e-3)
        assert (w >= 0.0).all()
        assert (w <= 0.3001).all()

    def test_constrained_qp_optimizer(self, sample_portfolio_df):
        opt = ConstrainedQPOptimizer()
        w = opt.optimize(sample_portfolio_df, max_sector_exposure=0.35, max_stock_weight=0.25)
        assert w.sum() == pytest.approx(1.0, abs=1e-3)
        assert (w >= 0.0).all()
        assert (w <= 0.2501).all()

    def test_get_optimizer_factory(self):
        assert isinstance(get_optimizer("equal"), EqualWeightOptimizer)
        assert isinstance(get_optimizer("inv_vol"), InverseVolOptimizer)
        assert isinstance(get_optimizer("score_weighted"), ScoreWeightedOptimizer)
        assert isinstance(get_optimizer("risk_parity"), RiskParityOptimizer)
        assert isinstance(get_optimizer("qp"), ConstrainedQPOptimizer)


class TestExecutionGateway:
    """4. 实盘与仿真交易网关测试"""

    def test_paper_broker_lifecycle(self):
        broker = PaperBroker(initial_cash=500_000.0)
        assert broker.connect()
        acc = broker.get_account()
        assert acc.cash == 500_000.0

        # 1. 模拟买入 1000 股 (10手)
        buy_ord = broker.send_order(symbol="600519.SH", side=OrderSide.BUY, shares=1000, price=100.0)
        assert buy_ord.status == ExecutionStatus.FILLED
        assert buy_ord.filled_shares == 1000

        # T+1: 当日可用为 0
        pos = broker.get_positions()["600519.SH"]
        assert pos.total_shares == 1000
        assert pos.available_shares == 0

        # 2. 尝试当日卖出 (应被拒绝)
        sell_reject = broker.send_order(symbol="600519.SH", side=OrderSide.SELL, shares=500, price=105.0)
        assert sell_reject.status == ExecutionStatus.REJECTED

        # 3. 跨日解锁
        broker.unlock_t1_shares()
        assert broker.get_positions()["600519.SH"].available_shares == 1000

        # 4. 次日卖出 500 股
        sell_ord = broker.send_order(symbol="600519.SH", side=OrderSide.SELL, shares=500, price=110.0)
        assert sell_ord.status == ExecutionStatus.FILLED
        assert broker.get_positions()["600519.SH"].total_shares == 500

    def test_miniqmt_broker_mock_fallback(self):
        qmt = MiniQMTBroker()
        # 未连接时走安全 Mock 保护
        ord_res = qmt.send_order(symbol="600519.SH", side=OrderSide.BUY, shares=200, price=150.0)
        assert ord_res is not None
        assert ord_res.requested_shares == 200


class TestMessageNotification:
    """5. 多渠道通知格式化测试"""

    def test_daily_report_formatting(self):
        top_df = pd.DataFrame({
            "symbol": ["600519.SH", "300750.SZ"],
            "name": ["贵州茅台", "宁德时代"],
            "industry": ["食品饮料", "电力设备"],
            "pred_score": [0.852, 0.741],
            "target_weight": [0.10, 0.08],
            "close": [1650.0, 195.0]
        })
        md_text = MessageNotifier.format_daily_report_markdown(
            signal_date="2026-08-28",
            execution_date="2026-08-31",
            top_df=top_df,
            macro_status="正常多头持仓"
        )
        assert "贵州茅台" in md_text
        assert "宁德时代" in md_text
        assert "85.2%" in md_text
        assert "2026-08-28" in md_text


class TestWalkForwardEnhancements:
    """6. 走步训练样本复合加权与多模型支持测试"""

    def test_sample_weight_computation(self):
        from models.walk_forward import WalkForwardTrainer
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "label_up_down_2d": np.random.choice([0, 1], size=100),
            "label_excess_2d": np.random.normal(0, 0.05, size=100)
        })
        weights = WalkForwardTrainer._compute_sample_weights(df, label_col="label_excess_2d", half_life_days=60)
        assert len(weights) == 100
        assert (weights > 0).all()
        # 近期大幅波动样本权重应该显著大于远期小幅样本
        assert weights[-1] > weights[0]
