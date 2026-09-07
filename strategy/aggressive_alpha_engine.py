"""
高弹性进取型主升浪 Alpha 决策引擎 (strategy/aggressive_alpha_engine.py)
涵盖 30 只全市场核心成长科技、算力、芯片、新材料、高端制造与出海主线标的
支持自由输出 Top 8 / Top 15 / Top 20 / Top 30 完整选股决策池
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

OFFENSIVE_SYMBOL_METADATA = {
    '600026.SH': {'name': '中远海能', 'industry': '能源航运',     'concept': '全球VLCC油运运价高弹性超级周期', 'beta': 1.45},
    '000301.SZ': {'name': '东方盛虹', 'industry': '新材料成长',   'concept': '光伏EVA/POE粒子与特种高分子龙头', 'beta': 1.40},
    '601138.SH': {'name': '工业富联', 'industry': 'AI硬件服务',   'concept': '英伟达GB200算力服务器全球代工', 'beta': 1.50},
    '688981.SH': {'name': '中芯国际', 'industry': '晶圆代工',     'concept': '国内先进制程芯片晶圆代工核心基石', 'beta': 1.45},
    '300308.SZ': {'name': '中际旭创', 'industry': 'AI算力/光模块', 'concept': '800G/1.6T高速光模块全球龙头', 'beta': 1.65},
    '603259.SH': {'name': '药明康德', 'industry': '创新药/CXO',   'concept': '全球小分子CRDMO外包研发服务龙头', 'beta': 1.40},
    '300750.SZ': {'name': '宁德时代', 'industry': '电力设备',     'concept': '全球动力电池与储能绝对霸主', 'beta': 1.40},
    '002594.SZ': {'name': '比亚迪',   'industry': '新能源整车',   'concept': '全球新能源汽车与整车智能化巨头', 'beta': 1.35},
    '603986.SH': {'name': '兆易创新', 'industry': '半导体芯片',   'concept': '存储芯片与MCU龙头', 'beta': 1.55},
    '688012.SH': {'name': '中微公司', 'industry': '半导体设备',   'concept': '高端等离子体刻蚀机自主可控', 'beta': 1.50},
    '601872.SH': {'name': '招商轮船', 'industry': '交通运输/航运', 'concept': '油散双轮驱动海运景气复苏', 'beta': 1.35},
    '300124.SZ': {'name': '汇川技术', 'industry': '工业机器人',   'concept': '工控自动化与人形机器人核心部件', 'beta': 1.35},
    '002475.SZ': {'name': '立讯精密', 'industry': '消费电子制造', 'concept': '声学可穿戴与AI精密制造绝对龙头', 'beta': 1.30},
    '300274.SZ': {'name': '阳光电源', 'industry': '光伏与储能',   'concept': '光伏逆变器与大型工商业储能出海', 'beta': 1.42},
    '688256.SH': {'name': '寒武纪',   'industry': 'AI算力芯片',   'concept': '思元系列云端AI训练推理芯片', 'beta': 1.70},
    '000977.SZ': {'name': '浪潮信息', 'industry': 'AI计算服务器', 'concept': '全球AI算力服务器核心出货商', 'beta': 1.48},
    '603501.SH': {'name': '韦尔股份', 'industry': 'CMOS图像芯片', 'concept': '车载与高端手机CIS传感器反转', 'beta': 1.40},
    '601899.SH': {'name': '紫金矿业', 'industry': '有色金属资源', 'concept': '铜金资源扩张与大宗商品超级周期', 'beta': 1.30},
    '600309.SH': {'name': '万华化学', 'industry': '高端化工新材料','concept': '全球聚氨酯MDI与高分子精细化工霸主', 'beta': 1.25},
    '601689.SH': {'name': '拓普集团', 'industry': '智能汽车/机器人','concept': '汽车轻量化底盘与人形机器人执行器', 'beta': 1.45},
    '300476.SZ': {'name': '胜宏科技', 'industry': 'AI算力PCB板',  'concept': '英伟达AI加速卡高阶HDI/PCB核心供应商', 'beta': 1.58},
    '300502.SZ': {'name': '新易盛',   'industry': '光通信模块',   'concept': '北美AI数据中心高端光模块大单交付', 'beta': 1.62},
    '300394.SZ': {'name': '天孚通信', 'industry': '光器件/引擎',  'concept': '高速光引擎与光通信精密元器件龙头', 'beta': 1.55},
    '688036.SH': {'name': '传音控股', 'industry': '智能终端出海', 'concept': '新兴市场智能手机之王与AI硬件渗透', 'beta': 1.35},
    '002371.SZ': {'name': '北方华创', 'industry': '半导体平台装备','concept': '刻蚀/薄膜沉积/清洗全系列半导体设备', 'beta': 1.45},
    '300014.SZ': {'name': '亿纬锂能', 'industry': '大圆柱锂电池', 'concept': '动储双轮驱动与高镍大圆柱电池量产', 'beta': 1.38},
    '600570.SH': {'name': '恒生电子', 'industry': '金融科技软件', 'concept': '证券基金核心交易柜台与AI大模型赋能', 'beta': 1.35},
    '002463.SZ': {'name': '沪电股份', 'industry': '算力网络PCB',  'concept': 'AI服务器与高速网络交换机高多层板', 'beta': 1.50},
    '300661.SZ': {'name': '圣邦股份', 'industry': '模拟芯片芯片', 'concept': '高性能信号链与电源管理芯片自研替代', 'beta': 1.45},
    '002241.SZ': {'name': '歌尔股份', 'industry': 'XR智能硬件',   'concept': 'XR头显设备与微电子精密声学代工', 'beta': 1.35}
}

class AggressiveAlphaEngine:
    """高弹性进取型量化引擎 (零伪造、严格依真实模型打分排序)"""
    
    @staticmethod
    def generate_aggressive_portfolio(factor_df: pd.DataFrame = None, date: str = None, top_k_buy: int = 8) -> pd.DataFrame:
        """从截面中筛选高弹性进攻型标的并生成进取仓位。
        
        严格坚持 Fail-Closed：输入数据必须提供真实模型预测分 'pred_score'，
        严禁使用 np.linspace 伪造胜率或人工 priority_map 强制顶替。
        """
        sub = None
        if factor_df is not None and not factor_df.empty:
            if 'date' in factor_df.columns:
                target_dt = pd.to_datetime(date) if date is not None else factor_df['date'].max()
                sub = factor_df[pd.to_datetime(factor_df['date']) == target_dt].copy()
            else:
                sub = factor_df.copy()
            
        if sub is None or sub.empty:
            matrix_path = Path('data_storage/research/factor_matrix_300.parquet')
            if matrix_path.exists() and date is not None:
                full_df = pd.read_parquet(matrix_path)
                full_df['date'] = pd.to_datetime(full_df['date'])
                sub = full_df[full_df['date'] == pd.to_datetime(date)].copy()
                
        if sub is None or sub.empty:
            raise ValueError("Fail-Closed: Input factor_df is empty or date not found. Refusing synthetic portfolio generation.")

        # 严格检查真实预测分
        if 'pred_score' not in sub.columns:
            raise ValueError("Fail-Closed: 'pred_score' missing from input data. Artificial/synthetic prediction scores are strictly prohibited.")

        dt_str = str(date) if date is not None else (str(sub['date'].iloc[0])[:10] if 'date' in sub.columns else "")

        records = []
        for sym, meta in OFFENSIVE_SYMBOL_METADATA.items():
            stock_rows = sub[sub['symbol'] == sym]
            if stock_rows.empty:
                continue
            r = stock_rows.iloc[0]
            pred_score = r['pred_score']
            if pd.isna(pred_score):
                continue
            
            close = float(r['close']) if 'close' in r and pd.notna(r['close']) else np.nan
            pct = float(r['pct_change']) if 'pct_change' in r and pd.notna(r['pct_change']) else 0.0
            
            records.append({
                'date': dt_str,
                'symbol': sym,
                'name': meta['name'],
                'industry': meta['industry'],
                'concept': meta['concept'],
                'beta': meta['beta'],
                'close': close,
                'pct_change': pct,
                'pred_score': float(pred_score)
            })
            
        if not records:
            raise ValueError("Fail-Closed: No symbols from offensive universe have valid non-null 'pred_score'.")

        res_df = pd.DataFrame(records)
        
        # 严格依据模型预测得分 pred_score 降序排序，零人工 priority_map
        res_df = res_df.sort_values(by=['pred_score'], ascending=[False]).reset_index(drop=True)
        
        n_total = len(res_df)
        core_weights = [0.18, 0.16, 0.14, 0.12, 0.11, 0.09, 0.08, 0.07]
        weights = [0.0] * n_total
        
        num_alloc = min(top_k_buy, n_total, len(core_weights))
        allocated_core = core_weights[:num_alloc]
        total_alloc_weight = sum(allocated_core)
        # 若标的不足 8 支，按比例缩放使得前部核心标的仓位合计不超过 0.95
        scale = 0.95 / total_alloc_weight if total_alloc_weight > 0 and num_alloc < len(core_weights) else 1.0
        for i in range(num_alloc):
            weights[i] = round(allocated_core[i] * min(scale, 1.0), 4)
            
        res_df['target_weight'] = weights
        return res_df
