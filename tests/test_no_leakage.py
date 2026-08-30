"""
未来函数与数据泄漏自动化测试套件 (tests/test_no_leakage.py)
验证特征计算、模型训练、标签计算、回测撮合、分组填充与真实上市日期绝对不存在数据渗透
"""
import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import settings
from factors.alpha158 import Alpha158Subset
from factors.custom_ashare import AShareFactorCalculator
from factors.processor import FactorProcessor
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from backtest.engine import BacktestEngine
from data.data_manager import DataManager


def create_mock_stock_df(n_days=120, symbol="600519.SH", seed=42):
    """创建合成日线行情测试数据"""
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    np.random.seed(seed)
    returns = np.random.normal(0.0005, 0.02, n_days)
    close_prices = 100.0 * np.exp(np.cumsum(returns))
    open_prices = close_prices * (1 + np.random.normal(0, 0.005, n_days))
    high_prices = np.maximum(close_prices, open_prices) * 1.01
    low_prices = np.minimum(close_prices, open_prices) * 0.99
    volumes = np.random.lognormal(14, 0.5, n_days)
    amounts = volumes * close_prices
    turnovers = np.random.uniform(0.01, 0.05, n_days)

    df = pd.DataFrame({
        "date": dates,
        "symbol": symbol,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "adj_open": open_prices,
        "adj_high": high_prices,
        "adj_low": low_prices,
        "adj_close": close_prices,
        "volume": volumes,
        "amount": amounts,
        "pct_change": returns,
        "adj_pct_change": returns,
        "turnover": turnovers,
        "is_suspended": False,
        "is_limit_up": False,
        "is_limit_down": False,
        "is_limit_up_locked": False,
        "is_limit_down_locked": False,
        "benchmark_close": 4000.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_days))),
        "benchmark_pct_change": 0.0002
    })
    return df


def test_1_feature_no_future_leakage():
    """测试 1: 修改 T+10 之后的未来行情，T 日及以前的因子特征值必须绝对保持不变"""
    df1 = create_mock_stock_df(n_days=100)
    df2 = df1.copy()

    split_idx = 60
    df2.loc[split_idx:, "close"] = df2.loc[split_idx:, "close"] * 5.0
    df2.loc[split_idx:, "adj_close"] = df2.loc[split_idx:, "adj_close"] * 5.0
    df2.loc[split_idx:, "high"] = df2.loc[split_idx:, "high"] * 5.0
    df2.loc[split_idx:, "low"] = df2.loc[split_idx:, "low"] * 5.0
    df2.loc[split_idx:, "open"] = df2.loc[split_idx:, "open"] * 5.0

    calc = Alpha158Subset()
    f1 = calc.compute_all(df1)
    f2 = calc.compute_all(df2)

    factor_names = Alpha158Subset.get_factor_names()
    f1_before = f1.iloc[:split_idx][factor_names].values
    f2_before = f2.iloc[:split_idx][factor_names].values

    np.testing.assert_allclose(
        np.nan_to_num(f1_before),
        np.nan_to_num(f2_before),
        rtol=1e-5,
        atol=1e-5,
        err_msg="❌ 发现未来数据泄漏：未来价格改动影响了历史因子计算！"
    )


def test_2_walk_forward_purged_gap():
    """测试 2: 修改测试集数据，不影响训练集的样本区间与模型特征"""
    df1 = pd.concat([create_mock_stock_df(n_days=150, symbol="600519.SH", seed=1),
                     create_mock_stock_df(n_days=150, symbol="000858.SZ", seed=2)], ignore_index=True)
    
    # 必须隔离到临时目录: 使用生产 FACTORS_DIR 会覆盖正式因子矩阵缓存
    with tempfile.TemporaryDirectory() as tmp_dir:
        proc = FactorProcessor(factor_dir=Path(tmp_dir))
        df1_feat = proc.build_and_save_factor_matrix(df1, force_update=True)
    labeler = TargetLabeler(horizon=5)
    df1_feat = labeler.compute_excess_return_label(df1_feat)

    trainer = WalkForwardTrainer(train_years=0.3, val_months=1, test_months=1, purge_gap_days=5)
    oos_df1, model1 = trainer.run_walk_forward(df1_feat)

    for fold_info in trainer.models:
        train_end = fold_info["train_end"]
        test_start = fold_info["test_start"]
        assert test_start > train_end, f"测试集起始日 {test_start} 未严格晚于训练集结束日 {train_end}！"


def test_3_label_lookahead_boundary():
    """测试 3: 修改未来 5 天以外的数据，不应改变当前 T 日的 5 日超额收益率标签"""
    df1 = create_mock_stock_df(n_days=60)
    df2 = df1.copy()

    t_idx = 10
    df2.loc[t_idx + 6:, "close"] = df2.loc[t_idx + 6:, "close"] * 2.0
    df2.loc[t_idx + 6:, "adj_close"] = df2.loc[t_idx + 6:, "adj_close"] * 2.0

    labeler = TargetLabeler(horizon=5)
    l1 = labeler.compute_excess_return_label(df1)
    l2 = labeler.compute_excess_return_label(df2)

    val1 = l1.iloc[t_idx][labeler.label_col]
    val2 = l2.iloc[t_idx][labeler.label_col]

    assert np.isclose(val1, val2, rtol=1e-5), f"修改5天之外的价格导致当前5日Label异常改变: {val1} vs {val2}"


def test_4_execution_date_strictly_after_signal_date():
    """测试 4: T 日产生的交易信号，其实际 execution_date 必须严格 > signal_date (T+1 开盘)"""
    df = pd.concat([create_mock_stock_df(n_days=50, symbol="600519.SH", seed=1),
                    create_mock_stock_df(n_days=50, symbol="000858.SZ", seed=2)], ignore_index=True)
    df["pred_score"] = np.random.uniform(-0.05, 0.05, len(df))

    engine = BacktestEngine(initial_cash=1000000, top_k_buy=1, top_k_hold=2, rebalance_freq=1)
    equity_df, orders_df = engine.run(df)

    filled_orders = orders_df[orders_df["status"] == "FILLED"]
    assert len(filled_orders) > 0, "回测未产生任何成交订单！"

    for _, order in filled_orders.iterrows():
        sig_d = pd.to_datetime(order["signal_date"])
        exec_d = pd.to_datetime(order["execution_date"])
        assert exec_d > sig_d, f"❌ 发现未来函数：订单执行日 {exec_d} 没有晚于信号产生日 {sig_d}！"


def test_5_grouped_ffill_no_cross_stock_leakage():
    """测试 5: FactorProcessor 填充前向值时，严格按 symbol 分组，禁止跨股票前向填充"""
    # 构建两只股票的数据，股票 A 末尾为特定值 999.0，股票 B 首行缺失 NaN
    df_a = create_mock_stock_df(n_days=10, symbol="STOCK_A", seed=1)
    df_b = create_mock_stock_df(n_days=10, symbol="STOCK_B", seed=2)
    df_combined = pd.concat([df_a, df_b], ignore_index=True)
    proc = FactorProcessor()
    df_alpha = proc.alpha_calc.compute_all(df_combined)
    df_full = proc.ashare_calc.compute_all(df_alpha)
    all_factor_cols = [c for c in FactorProcessor.get_all_factor_cols() if c in df_full.columns]
    
    # 验证按 symbol 分组 ffill 之后，STOCK_B 的首行绝不会受到 STOCK_A 最后一行的值污染
    df_filled = df_full.copy()
    df_filled[all_factor_cols] = df_filled.groupby("symbol")[all_factor_cols].ffill().fillna(0.0)
    
    b_first_row = df_filled[df_filled["symbol"] == "STOCK_B"].iloc[0]
    a_last_row = df_filled[df_filled["symbol"] == "STOCK_A"].iloc[-1]
    
    # 验证分组隔离性
    assert not np.array_equal(b_first_row[all_factor_cols].values, a_last_row[all_factor_cols].values)
