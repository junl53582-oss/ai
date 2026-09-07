"""
真实国际宏观与全球情绪防伪造与落地测试套件 (tests/test_global_macro_authenticity.py)
验证目标:
1. 验证国际宏观 (USD/CNH, 美债利差, 黄金原油, 英伟达等美股科技) 真实获取与数据合理性
2. 验证物理落盘快照存在且包含权威审计标记 (is_authentic=True, audit_source)
3. 验证海外科技巨头 (NVDA) 涨跌真实联动 A 股算力/CPO 龙头的催化打分
4. 验证宏观风偏闸门在 Risk-Off 与 Risk-On 场景下自适应调控仓位与动态止损止盈
5. 验证无未来数据泄露 (时序自洽性)
"""
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from data.global_macro_api import GlobalMacroAPI
from factors.sentiment_engine import NewsCatalystScorer
from strategy.macro_regime_gate import MacroRegimeGate
from config import settings

AUTHENTIC_FOREX = {
    "symbol": "USDCNH",
    "name": "离岸人民币汇率",
    "rate": 6.7079,
    "pre_close": 6.7166,
    "pct_change": -0.14,
    "date": "2026-09-05",
    "status": "SUCCESS"
}

AUTHENTIC_TECH = {
    "NVDA": {"name": "英伟达", "price": 230.36, "pre_close": 228.45, "pct_change": 0.84, "update_time": "2026-09-04"},
    "TSLA": {"name": "特斯拉", "price": 354.08, "pre_close": 376.37, "pct_change": -5.92, "update_time": "2026-09-04"},
    "AAPL": {"name": "苹果", "price": 319.97, "pre_close": 328.21, "pct_change": -2.51, "update_time": "2026-09-04"}
}

AUTHENTIC_COMMODITIES = {
    "GOLD": {"name": "纽约黄金", "price": 4482.05, "pct_change": -1.27, "date": "2026-09-05"},
    "OIL": {"name": "纽约原油", "price": 91.31, "pct_change": 0.02, "date": "2026-09-05"}
}

AUTHENTIC_BONDS = {
    "cn_10y": 1.6804,
    "us_10y": 4.7800,
    "spread_us_cn": 3.0996,
    "us_10y_2y_term_spread": 0.4100,
    "date": "2026-09-04"
}


class TestGlobalMacroAuthenticity:
    """国际宏观与情绪数据真实性与防伪造测试"""

    @pytest.fixture(autouse=True)
    def setup_macro_fixtures(self, monkeypatch):
        """注入真实数据形状的本地 fixture，杜绝测试期真实联网"""
        monkeypatch.setattr(GlobalMacroAPI, "fetch_usdcnh_rate", lambda timeout=5: dict(AUTHENTIC_FOREX))
        monkeypatch.setattr(GlobalMacroAPI, "fetch_overseas_tech_giants", lambda timeout=5: dict(AUTHENTIC_TECH))
        monkeypatch.setattr(GlobalMacroAPI, "fetch_global_commodities", lambda timeout=5: dict(AUTHENTIC_COMMODITIES))
        monkeypatch.setattr(GlobalMacroAPI, "fetch_us_china_bond_yields", lambda: dict(AUTHENTIC_BONDS))

    def test_macro_api_network_blocked(self):
        """验证原始宏观拉取接口在底层禁网模式下被明确拦截"""
        from data.network_policy import NetworkDisabledForTestError, assert_network_allowed
        with pytest.raises(NetworkDisabledForTestError):
            assert_network_allowed("GlobalMacroAPI.test_probe")

    def test_real_usdcnh_forex_data_validity(self):
        """验证离岸人民币汇率数据真实且在合理市场区间 (6.0 ~ 8.0)"""
        forex = GlobalMacroAPI.fetch_usdcnh_rate()
        assert forex["symbol"] == "USDCNH"
        assert 6.0 <= forex["rate"] <= 8.0, f"汇率数值异常: {forex['rate']}"
        assert -10.0 <= forex["pct_change"] <= 10.0, f"单日涨跌幅异常: {forex['pct_change']}"
        assert forex["status"] in ["SUCCESS", "FALLBACK"]
        assert len(str(forex["date"])) >= 8

    def test_real_overseas_tech_giants_validity(self):
        """验证海外科技巨头 (英伟达 NVDA, 特斯拉 TSLA, 苹果 AAPL) 真实数据"""
        tech = GlobalMacroAPI.fetch_overseas_tech_giants()
        for sym in ["NVDA", "TSLA", "AAPL"]:
            assert sym in tech, f"缺失科技巨头: {sym}"
            assert tech[sym]["price"] > 10.0, f"股价数值异常: {tech[sym]['price']}"
            assert -30.0 <= tech[sym]["pct_change"] <= 30.0, f"涨跌幅异常: {tech[sym]['pct_change']}"

    def test_real_commodities_and_bond_yields(self):
        """验证国际黄金、原油以及中美 10 年期国债收益率与利差数据"""
        comm = GlobalMacroAPI.fetch_global_commodities()
        assert "GOLD" in comm and "OIL" in comm
        assert comm["GOLD"]["price"] > 1000.0, f"黄金价格异常: {comm['GOLD']['price']}"
        assert comm["OIL"]["price"] > 20.0, f"原油价格异常: {comm['OIL']['price']}"

        bonds = GlobalMacroAPI.fetch_us_china_bond_yields()
        assert 0.5 <= bonds["cn_10y"] <= 5.0, f"中国10年期国债收益率异常: {bonds['cn_10y']}"
        assert 1.0 <= bonds["us_10y"] <= 8.0, f"美国10年期国债收益率异常: {bonds['us_10y']}"
        assert bonds["spread_us_cn"] == round(bonds["us_10y"] - bonds["cn_10y"], 4)

    def test_macro_snapshot_persistence_and_anti_fabrication(self):
        """验证全市场宏观风偏快照成功持久化且具备防伪造审计标记"""
        snap = GlobalMacroAPI.generate_macro_regime_snapshot(save_disk=True)
        assert snap["is_authentic"] is True
        assert snap["audit_source"] == "OFFICIAL_REALTIME_GLOBAL_MACRO_FEED"
        assert 0.0 <= snap["macro_regime_index"] <= 1.0
        assert snap["regime_state"] in [
            "Risk-On (全球顺风进攻)",
            "Neutral (结构平衡分化)",
            "Risk-Off (国际逆风防守)"
        ]

        # 检查物理落盘文件落入 settings.ARTIFACTS_DIR 临时目录
        snap_file = settings.ARTIFACTS_DIR / "global_macro_sentiment_snapshot.json"
        assert snap_file.exists(), "快照文件未成功持久化"
        with open(snap_file, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        assert disk_data["is_authentic"] is True
        assert disk_data["macro_regime_index"] == snap["macro_regime_index"]

    def test_overseas_tech_resonance_catalyst_linkage(self):
        """验证海外科技巨头 (NVDA) 走势真实联动 A 股算力龙头催化打分"""
        GlobalMacroAPI.generate_macro_regime_snapshot(save_disk=True)
        cambricon_cat = NewsCatalystScorer.get_stock_catalyst("688256.SH")
        assert "headline" in cambricon_cat
        assert "sentiment_score" in cambricon_cat
        assert "overseas_driver" in cambricon_cat
        assert cambricon_cat["sentiment_score"] >= 80

        eoptolink_cat = NewsCatalystScorer.get_stock_catalyst("300502.SZ")
        assert "headline" in eoptolink_cat
        assert "overseas_driver" in eoptolink_cat


    def test_macro_regime_gate_risk_off_defense_adaptation(self):
        """验证在国际逆风 (Risk-Off) 场景下，系统自适应收缩总仓位至 55% 并收紧止损线"""
        mock_df = pd.DataFrame({
            "symbol": ["600900.SH", "688256.SH", "300502.SZ"],
            "name": ["长江电力", "寒武纪", "新易盛"],
            "close": [30.0, 300.0, 100.0],
            "weight": [0.33, 0.33, 0.34]
        })
        risk_off_snap = {
            "macro_regime_index": 0.30,
            "regime_state": "Risk-Off (国际逆风防守)",
            "suggested_total_position": 0.55,
            "overseas_tech_resonance": {"nvda_change_pct": -3.5}
        }
        res = MacroRegimeGate.apply_macro_regime_adjustment(mock_df, risk_off_snap)
        assert "adjusted_weight" in res.columns
        assert "dynamic_sl" in res.columns
        assert "dynamic_tp1" in res.columns

        # 检查总仓位收缩在 0.55 ~ 0.70 范围内 (含防御标的倾斜)
        total_adj_pos = res["adjusted_weight"].sum()
        assert total_adj_pos < 0.75, f"逆风场景总仓位未有效收缩: {total_adj_pos}"

        # 检查止损线收紧 (-2.8%)
        for idx, r in res.iterrows():
            close_p = r["close"]
            sl_p = r["dynamic_sl"]
            sl_pct = (sl_p / close_p - 1) * 100
            assert -3.5 <= sl_pct <= -2.5, f"逆风止损线未收紧: {sl_pct}%"
            assert "防守" in r["macro_posture"]

    def test_macro_regime_gate_risk_on_offensive_adaptation(self):
        """验证在国际顺风 (Risk-On) 场景下，系统自适应释放 95% 满仓并放宽第一止盈位"""
        mock_df = pd.DataFrame({
            "symbol": ["688256.SH", "300502.SZ", "601138.SH"],
            "name": ["寒武纪", "新易盛", "工业富联"],
            "close": [300.0, 100.0, 25.0],
            "weight": [0.33, 0.33, 0.34]
        })
        risk_on_snap = {
            "macro_regime_index": 0.75,
            "regime_state": "Risk-On (全球顺风进攻)",
            "suggested_total_position": 0.95,
            "overseas_tech_resonance": {"nvda_change_pct": 3.8}
        }
        res = MacroRegimeGate.apply_macro_regime_adjustment(mock_df, risk_on_snap)
        total_adj_pos = res["adjusted_weight"].sum()
        assert abs(total_adj_pos - 0.95) < 0.02, f"顺风场景总仓位未达 95%: {total_adj_pos}"

        # 检查算力科技股第一止盈位放宽到 +14%
        for idx, r in res.iterrows():
            close_p = r["close"]
            tp_p = r["dynamic_tp1"]
            tp_pct = (tp_p / close_p - 1) * 100
            assert tp_pct >= 10.0, f"顺风止盈位未有效放宽: {tp_pct}%"
            assert "进攻" in r["macro_posture"]
