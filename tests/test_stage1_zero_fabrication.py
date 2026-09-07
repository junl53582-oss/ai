"""
阶段 1 自动化验证套件 (tests/test_stage1_zero_fabrication.py)
验证目标:
1. 生产展示与核心策略中无固定 pred_score、固定 price_seeds、固定日期与人工 priority_map
2. 缺失合法模型输出时系统严格 Fail-Closed，拒绝生成默认 0.75 或伪造预测
3. 20 日分类概率绝不被包装为 5 日收益率或目标价格
4. 真实决策清单具备完整的 Provenance 元数据且无 synthetic 标记
"""

import re
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import settings
from strategy.aggressive_alpha_engine import AggressiveAlphaEngine
from data.universe_csi500 import CSI500UniverseManager


def test_no_hardcoded_predictions_in_production_code():
    """验证生产关键模块中无人工伪造代码特征"""
    app_path = settings.BASE_DIR / "dashboard" / "app.py"
    strat_path = settings.BASE_DIR / "strategy" / "aggressive_alpha_engine.py"
    csi_path = settings.BASE_DIR / "data" / "universe_csi500.py"

    app_text = app_path.read_text(encoding="utf-8")
    strat_text = strat_path.read_text(encoding="utf-8")
    csi_text = csi_path.read_text(encoding="utf-8")

    # 1. 严禁 default 0.75 胜率回退
    assert "pred_score', 0.75" not in app_text, "app.py 仍存在默认 0.75 预测分回退"
    assert "pred_win_prob = 0.75" not in app_text, "app.py 仍存在硬编码 0.75 胜率"

    # 2. 严禁强制正收益逻辑 max(0.045, ...)
    assert "max(0.045" not in app_text, "app.py 仍存在 max(0.045 强制正收益逻辑"

    # 3. 严禁固定价格种子字典
    assert "price_seeds = {" not in csi_text, "universe_csi500.py 仍存在固定 price_seeds 字典"

    # 4. 严禁固定 scores 数组
    assert "scores = [0.785" not in csi_text, "universe_csi500.py 仍存在固定 scores 数组"

    # 5. 严禁在策略执行中使用 priority_map
    assert "priority_map = {" not in strat_text, "aggressive_alpha_engine.py 仍存在人工 priority_map 字典"

    # 6. 严禁在生产代码中使用 linspace 生成 pred_score
    for match in re.finditer(r"np\.linspace\(.*?\)", strat_text):
        matched_str = match.group(0)
        assert "probs" not in matched_str and "pred_score" not in matched_str, f"发现 linspace 用于预测分: {matched_str}"


def test_fail_closed_when_pred_score_missing():
    """验证缺失模型预测分时，策略引擎严格 Fail-Closed 拒绝输出而非生成假数据"""
    dates = pd.date_range("2026-09-01", periods=2, freq="D")
    df = pd.DataFrame([
        {"date": dates[0], "symbol": "300308.SZ", "close": 800.0, "pct_change": 0.02},
        {"date": dates[1], "symbol": "300308.SZ", "close": 810.0, "pct_change": 0.012}
    ])

    # 缺 pred_score 时必须抛出 ValueError
    with pytest.raises(ValueError, match="Fail-Closed"):
        AggressiveAlphaEngine.generate_aggressive_portfolio(df, "2026-09-02")


def test_fail_closed_on_empty_factor_data():
    """验证空数据时策略引擎严格 Fail-Closed"""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Fail-Closed"):
        AggressiveAlphaEngine.generate_aggressive_portfolio(empty_df, "2026-09-03")


def test_classification_probabilities_not_converted_to_forward_target_prices():
    """验证 20 日分类概率不被篡改为 5 日目标价和置信区间光晕"""
    app_path = settings.BASE_DIR / "dashboard" / "app.py"
    app_text = app_path.read_text(encoding="utf-8")

    # 检验不存在 5 日目标价外推循环
    assert "exp_5d_pct" not in app_text, "app.py 仍存在 5 日收益率几何外推 exp_5d_pct"
    assert "90% 置信预测区间" not in app_text, "app.py 仍存在伪造的 90% 置信区间光晕带"
    assert "5日目标: ¥" not in app_text, "app.py 仍存在未经校准的 5 日目标价虚假展示"


def test_provenance_metadata_in_picks_files():
    """验证落盘决策清单具有完整合规元数据且无合成标签"""
    for fname in ["latest_stock_picks.csv", "aggressive_stock_picks.csv", "csi500_stock_picks.csv"]:
        fpath = settings.BASE_DIR / "artifacts" / fname
        assert fpath.exists(), f"决策文件 {fname} 不存在"

        df = pd.read_csv(fpath)
        assert len(df) > 0, f"{fname} 为空"
        assert "model_id" in df.columns, f"{fname} 缺少 model_id"
        assert "is_synthetic_demo" in df.columns, f"{fname} 缺少 is_synthetic_demo"
        assert "pred_score" in df.columns, f"{fname} 缺少 pred_score"

        # 验证 is_synthetic_demo 全部为 False
        assert not df["is_synthetic_demo"].any(), f"{fname} 包含了 synthetic demo 标记为 True 的记录"

        # 验证 pred_score 不是简单的线性递减 (非 np.linspace)
        scores = df["pred_score"].dropna().values
        diffs = np.diff(scores)
        if len(diffs) > 2:
            assert np.std(diffs) > 1e-6, f"{fname} 中的 pred_score 疑似由 linspace 线性等距构造"
