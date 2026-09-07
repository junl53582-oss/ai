"""
中证 500 高弹性成长股票池定义与预测发生器 (data/universe_csi500.py)
用于支持 Direction 4:
1. 提供中证 500 (CSI 500) 核心高弹性科技、半导体、机器人、高端制造龙头股票池
2. 结合第四代旗舰多模态模型生成专属的高弹性主线选股清单 (artifacts/csi500_stock_picks.csv)
3. 与 Streamlit 前端 Tab 1 无缝联动，实现双池自由切换
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from config.settings import settings

logger = logging.getLogger("CSI500Universe")

# 中证500高弹性成长主线精选龙头标的池 (覆盖 AI芯片、高端装备、工业母机、卫星互联网、新材料、光通信)
CSI500_CORE_GROWTH_UNIVERSE = [
    {"symbol": "688256.SH", "name": "寒武纪", "industry": "AI算力芯片", "beta": 1.85, "concept": "思元系列国产大模型训练加速卡"},
    {"symbol": "300502.SZ", "name": "新易盛", "industry": "光通信模块", "beta": 1.75, "concept": "北美云巨头800G/1.6T硅光模块核心定点"},
    {"symbol": "300394.SZ", "name": "天孚通信", "industry": "光器件/引擎", "beta": 1.70, "concept": "高速光引擎精密制造与AI算力互联"},
    {"symbol": "300476.SZ", "name": "胜宏科技", "industry": "AI算力PCB", "beta": 1.68, "concept": "英伟达AI加速卡高端6阶HDI核心直供"},
    {"symbol": "688041.SH", "name": "海光信息", "industry": "国产CPU/DCU", "beta": 1.65, "concept": "深算系列高算力DCU国产替代龙头"},
    {"symbol": "688012.SH", "name": "中微公司", "industry": "半导体设备", "beta": 1.60, "concept": "高端CCP/ICP刻蚀机国产化绝对中坚"},
    {"symbol": "601689.SH", "name": "拓普集团", "industry": "人形机器人", "beta": 1.55, "concept": "北美车企轻量化底盘与机器人执行器"},
    {"symbol": "688036.SH", "name": "传音控股", "industry": "智能终端出海", "beta": 1.48, "concept": "非洲新兴市场AI手机软硬件生态霸主"},
    {"symbol": "300124.SZ", "name": "汇川技术", "industry": "工业自动化", "beta": 1.45, "concept": "工控伺服系统与机器人高精度减速电机"},
    {"symbol": "600160.SH", "name": "巨化股份", "industry": "算力液冷/氟化工", "beta": 1.50, "concept": "数据中心浸没式电子级冷却液全球供应商"},
    {"symbol": "002837.SZ", "name": "英维克", "industry": "数据中心温控", "beta": 1.62, "concept": "AI超算中心高密液冷散热全链条解决方案"},
    {"symbol": "688008.SH", "name": "澜起科技", "industry": "内存接口芯片", "beta": 1.58, "concept": "DDR5内存接口芯片与PCIe Retimer全球双寡头"},
    {"symbol": "301308.SZ", "name": "江波龙", "industry": "存储模组/主控", "beta": 1.65, "concept": "企业级SSD固态存储与车规级存储出海"},
    {"symbol": "600522.SH", "name": "中天科技", "industry": "海风海缆/光纤", "beta": 1.42, "concept": "深远海超高压海缆与海上风电新一轮招标"},
    {"symbol": "601138.SH", "name": "工业富联", "industry": "AI算力代工", "beta": 1.52, "concept": "GB200算力机柜与高速交换机全球代工"},
]


class CSI500UniverseManager:
    """中证 500 高弹性标的池管理与预测发生器 (零伪造、真实模型推理)"""

    @classmethod
    def get_core_symbols(cls) -> List[str]:
        return [item["symbol"] for item in CSI500_CORE_GROWTH_UNIVERSE]

    @classmethod
    def generate_csi500_picks_file(cls, out_path: Optional[Path] = None, factor_df: Optional[pd.DataFrame] = None, date: Optional[str] = None) -> Path:
        """根据真实行情与已注册生产模型推理生成中证 500 决策清单。
        
        严禁使用固定 scores 数组、固定 price_seeds、固定交易日期以及人工 priority_map。
        若无法完成真实推理或行情缺失，严格 Fail-Closed 抛出异常。
        """
        if out_path is None:
            out_path = settings.ARTIFACTS_DIR / "csi500_stock_picks.csv"

        # 1. 获取基础数据
        if factor_df is not None and not factor_df.empty:
            df = factor_df.copy()
        else:
            matrix_path = settings.BASE_DIR / "data_storage" / "research" / "factor_matrix_300.parquet"
            if not matrix_path.exists():
                raise FileNotFoundError(f"Fail-Closed: Factor matrix not found at {matrix_path}")
            df = pd.read_parquet(matrix_path)

        if 'date' not in df.columns:
            raise ValueError("Fail-Closed: Dataset missing required 'date' column.")

        df['date'] = pd.to_datetime(df['date'])
        target_date = pd.to_datetime(date) if date is not None else df['date'].max()

        # 针对每个股票提取 <= target_date 的最新一条真实切片
        matched_rows = []
        for item in CSI500_CORE_GROWTH_UNIVERSE:
            sym = item["symbol"]
            sym_df = df[(df["symbol"] == sym) & (df["date"] <= target_date)]
            if not sym_df.empty:
                latest_row = sym_df.sort_values(by="date").iloc[-1].to_dict()
                latest_row["name"] = item["name"]
                latest_row["industry"] = item["industry"]
                latest_row["concept"] = item["concept"]
                latest_row["beta"] = item["beta"]
                matched_rows.append(latest_row)

        if not matched_rows:
            raise ValueError(f"Fail-Closed: No market data found for CSI 500 universe on or before {target_date}.")

        sub_df = pd.DataFrame(matched_rows)

        # 2. 获取或计算真实 pred_score
        model_id = "unknown"
        if "pred_score" in sub_df.columns and sub_df["pred_score"].notna().all():
            logger.info("Using precomputed pred_score from input factor_df.")
        else:
            # 加载已注册生产模型执行真实推理
            prod_model_path = settings.BASE_DIR / "saved_models" / "production" / "m_20260903_194757_hybrid_bagging_ridge" / "model.pkl"
            if not prod_model_path.exists():
                prod_model_path = settings.BASE_DIR / "saved_models" / "latest_lightgbm.pkl"
            if not prod_model_path.exists():
                raise FileNotFoundError("Fail-Closed: No registered production model found for inference.")

            try:
                import joblib
                model = joblib.load(prod_model_path)
                model_id = "m_20260903_194757_hybrid_bagging_ridge" if "m_20260903_194757" in str(prod_model_path) else "latest_production_model"
            except Exception as e:
                raise RuntimeError(f"Fail-Closed: Failed to load production model: {e}")

            feature_names = getattr(model, "feature_names", None)
            if feature_names is None:
                feature_names = getattr(model, "feature_name_", None)
            if feature_names is None:
                raise ValueError("Fail-Closed: Production model lacks feature_names definition.")

            missing_feats = [f for f in feature_names if f not in sub_df.columns]
            if missing_feats:
                try:
                    from research_v2.alphas.novel_alphas import NovelAlphaFactory
                    sub_df['ALPHA_RESIDUAL_MOMENTUM_20'] = NovelAlphaFactory.calc_residual_momentum(df, window=20).loc[sub_df.index]
                    sub_df['ALPHA_TURNOVER_SURPRISE_5_20'] = NovelAlphaFactory.calc_turnover_surprise(df, short_w=5, long_w=20).loc[sub_df.index]
                    sub_df['ALPHA_QUALITY_X_MOMENTUM'] = NovelAlphaFactory.calc_quality_x_momentum(df).loc[sub_df.index]
                    sub_df['ALPHA_LIQUIDITY_X_VOL'] = NovelAlphaFactory.calc_liquidity_x_volatility(df).loc[sub_df.index]
                    sub_df['ALPHA_SHORT_REVERSAL_5'] = NovelAlphaFactory.calc_short_term_reversal(df, window=5).loc[sub_df.index]
                    sub_df['ALPHA_IDIO_VOL_PENALTY'] = NovelAlphaFactory.calc_idio_vol_penalty(df, window=20).loc[sub_df.index]
                    sub_df['ALPHA_MONEY_FLOW_DIV_10'] = NovelAlphaFactory.calc_money_flow_divergence(df, window=10).loc[sub_df.index]
                except Exception as e:
                    logger.warning(f"Feature calculation fallback: {e}")

            still_missing = [f for f in feature_names if f not in sub_df.columns]
            if still_missing:
                raise ValueError(f"Fail-Closed: Missing required model features: {still_missing}")

            X = sub_df[feature_names].fillna(0.0)
            preds = model.predict(X)
            sub_df["pred_score"] = [float(p) for p in preds]

        # 3. 严格依真实 pred_score 降序排序
        sub_df = sub_df.sort_values(by="pred_score", ascending=False).reset_index(drop=True)

        # 4. 配置目标权重：前 6 支执行满仓 95% 进攻分配，其余为观察池
        core_weights = [0.22, 0.20, 0.18, 0.14, 0.12, 0.09]
        weights = [0.0] * len(sub_df)
        for i in range(min(len(core_weights), len(sub_df))):
            weights[i] = core_weights[i]
        sub_df["target_weight"] = weights

        # 5. 格式化输出字段
        sub_df["date"] = sub_df["date"].astype(str).str[:10]
        sub_df["sentiment_stage"] = [
            "第一梯队核心" if i < 3 else ("第二梯队进攻" if i < 6 else "战略储备池")
            for i in range(len(sub_df))
        ]
        sub_df["model_id"] = model_id
        sub_df["is_synthetic_demo"] = False
        sub_df["data_as_of"] = str(target_date)[:10]

        out_cols = [
            "date", "symbol", "name", "industry", "concept", "beta", "close", "pct_change",
            "pred_score", "target_weight", "sentiment_stage", "model_id", "is_synthetic_demo", "data_as_of"
        ]
        final_df = sub_df[[c for c in out_cols if c in sub_df.columns]].copy()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"中证 500 真实模型决策清单已生成落盘: {out_path} (标的数: {len(final_df)})")
        return out_path
