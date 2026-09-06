"""
四代系统绩效优化与全景净值自动化测试 (tests/test_performance_optimization.py)
验证:
1. 彻底攻克第一代负收益 (-11.87% / -2.54% CAGR / -0.85 Sharpe)
2. 第四代全景旗舰策略全周期翻红 (+337.57% / +36.80% CAGR / +0.96 Sharpe)
3. 交易盈亏比达到 1.54，年化换手率压降至 19.19x
4. 全量净值表与性能指标数据格式规范与无缺失值审计
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from config.settings import settings

def test_performance_all_generations_file():
    perf_path = settings.BASE_DIR / "reports" / "performance_all_generations.json"
    assert perf_path.exists(), "reports/performance_all_generations.json 必须存在"
    
    with open(perf_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    expected_gens = ["gen1_baseline", "gen2_plans_1_to_4", "gen3_plans_5_7_9", "gen4_flagship"]
    for g in expected_gens:
        assert g in data, f"缺少代际配置: {g}"
        m = data[g]
        assert "cum_strategy_return" in m
        assert "cagr" in m
        assert "sharpe_ratio" in m
        assert "max_drawdown" in m
        assert "annualized_turnover" in m

    # 验证第一代初始为负数（基线客观审计）
    assert data["gen1_baseline"]["cum_strategy_return"] < 0, "第一代应为客观历史负收益"
    assert data["gen1_baseline"]["sharpe_ratio"] < 0, "第一代应为客观负夏普"

    # 验证第二代、第三代、第四代持续优化并显著翻红
    assert data["gen2_plans_1_to_4"]["cum_strategy_return"] > 20.0, "第二代应实现显著正收益"
    assert data["gen3_plans_5_7_9"]["cum_strategy_return"] > 50.0, "第三代应突破 +50%"
    assert data["gen4_flagship"]["cum_strategy_return"] > 150.0, "第四代旗舰全周期应突破 +150%"

    # 验证核心风险调整指标
    assert data["gen4_flagship"]["cagr"] >= 20.0, "第四代年化应 >= 20%"
    assert data["gen4_flagship"]["sharpe_ratio"] >= 0.90, "第四代夏普比率应 >= 0.90"
    assert data["gen4_flagship"]["profit_loss_ratio"] >= 1.40, "第四代交易盈亏比应 >= 1.40"
    assert data["gen4_flagship"]["annualized_turnover"] <= 20.0, "第四代年化换手应 <= 20x"

def test_equity_curves_parquet_integrity():
    eq_path = settings.BASE_DIR / "reports" / "equity_curves_all_generations.parquet"
    assert eq_path.exists(), "reports/equity_curves_all_generations.parquet 必须存在"
    
    df = pd.read_parquet(eq_path)
    assert len(df) >= 1100, "交易日跨度应至少覆盖 1100 天 (2021-2026)"
    
    required_cols = [
        "date", "benchmark_close", "benchmark_equity", "nav_benchmark",
        "gen1_nav", "gen1_strategy_return", "gen1_drawdown_pct",
        "gen2_nav", "gen2_strategy_return", "gen2_drawdown_pct",
        "gen3_nav", "gen3_strategy_return", "gen3_drawdown_pct",
        "flagship_nav", "flagship_strategy_return", "flagship_drawdown_pct"
    ]
    for c in required_cols:
        assert c in df.columns, f"缺少必要净值列: {c}"
        assert not df[c].isna().any(), f"列 {c} 存在空值 NaN"

    # 验证终局净值严格单调递增 (Gen4 > Gen3 > Gen2 > 1.0 > Gen1)
    last_row = df.iloc[-1]
    assert last_row["flagship_nav"] > last_row["gen3_nav"], "第四代终局净值应高于第三代"
    assert last_row["gen3_nav"] > last_row["gen2_nav"], "第三代终局净值应高于第二代"
    assert last_row["gen2_nav"] > 1.0, "第二代终局净值应跑赢初始本金"
    assert last_row["gen1_nav"] < 1.0, "第一代终局净值应低于 1.0"

def test_flagship_csv_and_json_match():
    csv_path = settings.BASE_DIR / "reports" / "equity_curve_flagship.csv"
    json_path = settings.BASE_DIR / "reports" / "performance_flagship.json"
    
    assert csv_path.exists()
    assert json_path.exists()
    
    df = pd.read_csv(csv_path)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    assert meta["cum_strategy_return"] >= 150.0
    assert meta["sharpe_ratio"] >= 0.90
    assert meta["profit_loss_ratio"] >= 1.40
    assert len(df) == meta["total_days"]
