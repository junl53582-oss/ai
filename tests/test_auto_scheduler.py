"""
自动调度与中证500股票池自动化测试 (tests/test_auto_scheduler.py)
验证:
1. 交易日判断逻辑覆盖与有效性
2. 自动化盘后跑批流水线 (execute_post_market_pipeline) 端到端执行
3. 中证 500 高弹性股票池清单格式与指标有效性
4. 调度审计日志与状态持久化文件一致性
"""
import pytest
import json
from datetime import date
from pathlib import Path
from config.settings import settings
from scripts.auto_daily_scheduler import DailyExecutionPipeline
from data.universe_csi500 import CSI500UniverseManager

def test_is_trading_day():
    # 2026-08-21 是周五 (物理交易日)
    fri = date(2026, 8, 21)
    assert DailyExecutionPipeline.is_trading_day(fri) is True
    
    # 2026-08-22 是周六 (非交易日)
    sat = date(2026, 8, 22)
    assert DailyExecutionPipeline.is_trading_day(sat) is False
    
    # 2026-08-23 是周日 (非交易日)
    sun = date(2026, 8, 23)
    assert DailyExecutionPipeline.is_trading_day(sun) is False

    # 2026-09-04 超出物理日历范围 (fail-closed 返回 False)
    fut = date(2026, 9, 4)
    assert DailyExecutionPipeline.is_trading_day(fut) is False

def test_csi500_universe_generator(tmp_path):
    out_file = tmp_path / "csi500_stock_picks.csv"
    p = CSI500UniverseManager.generate_csi500_picks_file(out_file)
    assert p.exists()
    
    import pandas as pd
    df = pd.read_csv(p)
    assert len(df) >= 15
    required = ["symbol", "name", "industry", "pred_score", "target_weight", "close", "sentiment_stage"]
    for c in required:
        assert c in df.columns, f"缺少必要列: {c}"
        
    # 验证满仓目标权重 (95%)
    assert abs(df["target_weight"].sum() - 0.95) < 1e-4

def test_daily_pipeline_execution(tmp_path):
    test_paper = tmp_path / "paper_ledger.json"
    test_shadow = tmp_path / "shadow_ledger.json"
    status_file = tmp_path / "auto_scheduler_last_run.json"
    res = DailyExecutionPipeline.execute_post_market_pipeline(
        sync_online=False,
        ledger_file=test_paper,
        shadow_ledger_file=test_shadow,
        status_file=status_file,
        execution_date="2026-08-24"
    )
    assert res is not None
    assert "date" in res
    assert "steps" in res
    assert "summary" in res
    assert res["summary"]["total_equity"] > 0
    assert res["summary"]["nav"] > 0
    
    assert status_file.exists()
    with open(status_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["date"] == res["date"]
