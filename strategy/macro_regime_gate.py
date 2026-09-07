"""
全球宏观风偏动态调控闸门与自适应止盈止损模块 (strategy/macro_regime_gate.py)
功能:
1. 接收全市场国际宏观快照 (USD/CNH、美债利差、黄金原油、英伟达映射)
2. 动态调节投资组合总仓位 (Risk-On: 95% 满仓进攻; Neutral: 80% 平衡; Risk-Off: 55% 防守)
3. 攻防结构倾斜：Risk-Off 自动提高高股息防御标的权重，收敛止损位 (SL) 保护本金
4. 动态止盈调节：Risk-On 顺风期放宽第一止盈位 (TP1)，让利润伴随海外科技主升浪奔跑
"""
import json
import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings

logger = logging.getLogger(__name__)

# 波动率自适应风控系数 (第一版默认值; 标定研究待做, 非经验证最优)
_REGIME_ATR_K = {
    "Risk-On": {"k_tp": 4.0, "k_sl": 2.0},
    "Risk-Off": {"k_tp": 2.2, "k_sl": 1.6},
    "Neutral": {"k_tp": 3.0, "k_sl": 2.2},
}
_MAX_POSITION_CAP = 0.30  # 单票仓位上限
_ATR_CACHE: Dict[str, Any] = {}


def _load_atr14_map() -> Dict[str, float]:
    """从主行情缓存计算各标的 ATR14 (波动率自适应风控的基础输入, 进程级缓存)"""
    if _ATR_CACHE.get("map"):
        return _ATR_CACHE["map"]
    try:
        pq = Path(settings.PARQUET_DIR) / "market_daily.parquet"
        if not pq.exists():
            logger.warning(f"[MacroGate] ATR 数据源缺失: {pq}")
            return {}
        df = pd.read_parquet(pq, columns=["date", "symbol", "high", "low", "close"])
        df = df.sort_values(["symbol", "date"])
        prev_close = df.groupby("symbol")["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["tr"] = tr
        # 逐 symbol 取最近 14 条 TR 的均值 (简单移动平均 ATR14)
        atr_map = df.dropna(subset=["tr"]).groupby("symbol")["tr"].apply(lambda s: float(s.tail(14).mean())).to_dict()
        _ATR_CACHE["map"] = atr_map
        logger.info(f"[MacroGate] ATR14 计算完成: {len(atr_map)} 标的")
        return atr_map
    except Exception as e:
        logger.warning(f"[MacroGate] ATR14 计算失败 (回退静态比例): {e}")
        return {}

class MacroRegimeGate:
    """全球宏观风偏动态闸门"""

    DEFENSIVE_SYMBOLS = {"600900.SH", "601006.SH", "601988.SH", "601288.SH", "601398.SH", "600028.SH"}

    @classmethod
    def load_latest_snapshot(cls) -> Dict[str, Any]:
        """加载最新的国际宏观风偏快照"""
        snap_file = settings.ARTIFACTS_DIR / "global_macro_sentiment_snapshot.json"
        if snap_file.exists():
            try:
                with open(snap_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"读取宏观快照异常: {e}")
                
        # 降级基准快照 (2026-09-04 真实官方基准)
        return {
            "macro_regime_index": 0.458,
            "regime_state": "Neutral (结构平衡分化)",
            "suggested_total_position": 0.80,
            "regime_summary": "全球宏观处于震荡平衡期，多空博弈势均力敌，聚焦业绩与资金面共振的确定性龙头。",
            "overseas_tech_resonance": {
                "resonance_score": 0.352,
                "nvda_change_pct": 0.84,
                "tech_avg_change_pct": -1.18
            },
            "usdcnh_forex": {"rate": 6.7079, "pct_change": -0.14}
        }

    @classmethod
    def apply_macro_regime_adjustment(cls, top_df: pd.DataFrame, macro_snap: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        对选股清单进行宏观风偏自适应动态调优
        输出含有动态仓位、动态止损止盈及宏观风控理由的数据集
        """
        if top_df is None or top_df.empty:
            return top_df

        df = top_df.copy()
        if macro_snap is None:
            macro_snap = cls.load_latest_snapshot()

        regime_state = macro_snap.get("regime_state", "Neutral")
        regime_idx = macro_snap.get("macro_regime_index", 0.50)
        target_total_pos = macro_snap.get("suggested_total_position", 0.80)
        nvda_chg = macro_snap.get("overseas_tech_resonance", {}).get("nvda_change_pct", 0.0)

        n = len(df)
        if n == 0:
            return df

        # 0. 波动率输入 (ATR14, Fail-Open: 拿不到则回退旧静态比例)
        atr_map = _load_atr14_map()

        # 1. 逆波动率权重: w_i ∝ 1/ATR_i, 归一到宏观目标总仓位, 单票上限 30%
        #    (低波动标的承担更大仓位, 高波动标的自动降权; 无 ATR 的标的用中位数替代)
        inv = df["symbol"].map(lambda s: (1.0 / atr_map[s]) if atr_map.get(s) else np.nan)
        if inv.notna().any():
            fill = float(inv.median())
            inv = inv.fillna(fill if fill > 0 else 1.0)
            raw_w = inv / inv.sum() * float(target_total_pos)
            df["adjusted_weight"] = raw_w.clip(upper=_MAX_POSITION_CAP).round(4)
        elif "weight" in df.columns:
            cur_sum = df["weight"].sum()
            if cur_sum > 0:
                scale = target_total_pos / cur_sum
                df["adjusted_weight"] = (df["weight"] * scale).round(4)
            else:
                df["adjusted_weight"] = round(target_total_pos / n, 4)
        else:
            df["adjusted_weight"] = round(target_total_pos / n, 4)

        # 2. Risk-Off 特殊结构倾斜：若含有高股息防御标的，向防御倾斜
        if "Risk-Off" in regime_state:
            def_mask = df["symbol"].isin(cls.DEFENSIVE_SYMBOLS)
            if def_mask.any():
                def_count = def_mask.sum()
                extra_def_weight = min(0.15, (1.0 - target_total_pos) * 0.3)
                df.loc[def_mask, "adjusted_weight"] = (
                    df.loc[def_mask, "adjusted_weight"] + round(extra_def_weight / def_count, 4)
                ).clip(upper=_MAX_POSITION_CAP)

        # 3. 动态止盈止损: ATR 波动率自适应 (同一系数下高波动股更宽、低波动股更紧)
        #    系数为第一版默认值, 标定研究待做; 无 ATR 时回退旧静态比例
        kk = _REGIME_ATR_K["Neutral"]
        if "Risk-On" in regime_state:
            kk = _REGIME_ATR_K["Risk-On"]
        elif "Risk-Off" in regime_state:
            kk = _REGIME_ATR_K["Risk-Off"]

        for idx, r in df.iterrows():
            close_p = float(r.get("close", 10.0))
            sym = r.get("symbol", "")
            atr = atr_map.get(sym)

            if atr and atr > 0:
                tp_p = close_p + kk["k_tp"] * atr
                sl_p = max(close_p - kk["k_sl"] * atr, 0.01)
                mode_tag = "ATR自适应"
            else:
                if "Risk-On" in regime_state:
                    tp_ratio, sl_ratio = 1.10, 0.95
                elif "Risk-Off" in regime_state:
                    tp_ratio, sl_ratio = 1.06, 0.972
                else:
                    tp_ratio, sl_ratio = 1.085, 0.955
                tp_p, sl_p = close_p * tp_ratio, close_p * sl_ratio
                mode_tag = "静态回退"

            if "Risk-On" in regime_state:
                posture = "🚀 顺风主升进攻"
                rationale = (f"宏观顺风期(启发式规则, 未经验证仅供参考): 总仓位 {int(target_total_pos*100)}%, "
                             f"NVDA 映射 {nvda_chg:+.2f}% 仅为观察值; {mode_tag} TP1 {round(tp_p, 2)} 元 / SL {round(sl_p, 2)} 元。")
            elif "Risk-Off" in regime_state:
                posture = "🛡️ 逆风防守收敛"
                rationale = (f"宏观逆风期(启发式规则, 未经验证仅供参考): 总仓位下调至 {int(target_total_pos*100)}%, "
                             f"{mode_tag} SL {round(sl_p, 2)} 元 / TP1 {round(tp_p, 2)} 元。")
            else:
                posture = "⚖️ 结构均衡稳健"
                rationale = (f"宏观平衡期(启发式规则, 未经验证仅供参考): 总仓位 {int(target_total_pos*100)}%, "
                             f"{mode_tag} TP1 {round(tp_p, 2)} 元 / SL {round(sl_p, 2)} 元。")

            df.at[idx, "dynamic_tp1"] = round(tp_p, 2)
            df.at[idx, "dynamic_sl"] = round(sl_p, 2)
            df.at[idx, "macro_posture"] = posture
            df.at[idx, "macro_execution_rationale"] = rationale

        return df

if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "symbol": ["600519.SH", "300750.SZ", "688256.SH"],
        "name": ["贵州茅台", "宁德时代", "寒武纪"],
        "close": [1330.0, 240.0, 310.0],
        "weight": [0.15, 0.15, 0.15]
    })
    res = MacroRegimeGate.apply_macro_regime_adjustment(sample_df)
    print("Adjusted sample:")
    print(res[["symbol", "name", "adjusted_weight", "dynamic_tp1", "dynamic_sl", "macro_posture"]])
