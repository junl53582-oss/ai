"""
因子研究与 Alpha 验证全局配置 (research/config.py)
定义因子研究视界、分层数、交易摩擦成本假设、评分权重与分级门禁阈值。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ResearchConfig:
    # ---------------- 预测视界与收益定义 ----------------
    HORIZONS: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    PRIMARY_HORIZON: int = 20                   # 核心评估视界 (天)
    USE_EXCESS_RETURN: bool = True               # 是否优先使用基准超额收益 (去 Beta)
    
    # ---------------- 分层与组合回测 ----------------
    NUM_QUANTILES: int = 5                      # 主分层组数 (Q1..Q5)
    DETAILED_QUANTILES: int = 10                # 细分层组数 (Q1..Q10)
    
    # ---------------- 交易摩擦成本假设 (bps) ----------------
    COST_BPS_LIST: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0, 30.0])
    DEFAULT_COST_BPS: float = 10.0              # 默认基准摩擦成本 (10 bps = 0.1%)
    
    # ---------------- 滚动与稳定性参数 ----------------
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [60, 120, 252])
    MARKET_REGIME_WINDOW: int = 20              # 划分牛熊震荡的窗口
    MARKET_REGIME_THRESHOLD: float = 0.03       # 20日收益 > +3% 牛市, < -3% 熊市, 其它震荡
    
    # ---------------- 多重检验与 Bootstrap ----------------
    FDR_ALPHA: float = 0.05                     # Benjamini-Hochberg FDR 显著性水平
    BOOTSTRAP_ROUNDS: int = 500                 # Block Bootstrap 抽样轮数
    BOOTSTRAP_BLOCK_SIZE: int = 20              # 时间块长度 (20交易日)
    BOOTSTRAP_CONFIDENCE: float = 0.95          # 置信区间
    
    # ---------------- 因子筛选评分权重 ----------------
    # 综合得分 = 30% RankIC + 20% ICIR + 15% Monotonicity + 15% Stability + 10% Net Sharpe - 5% Turnover - 5% Missing - 10% Redundancy
    WEIGHT_RANK_IC: float = 0.30
    WEIGHT_IC_IR: float = 0.20
    WEIGHT_MONOTONICITY: float = 0.15
    WEIGHT_STABILITY: float = 0.15
    WEIGHT_NET_SHARPE: float = 0.10
    PENALTY_TURNOVER: float = 0.05
    PENALTY_MISSING: float = 0.05
    PENALTY_REDUNDANCY: float = 0.10
    
    # ---------------- 因子状态分级阈值 ----------------
    STRONG_RANK_IC: float = 0.04
    STRONG_IC_IR: float = 0.50
    STRONG_SIGN_STABILITY: float = 0.70
    STRONG_COVERAGE: float = 0.80
    
    USEFUL_RANK_IC: float = 0.02
    USEFUL_IC_IR: float = 0.30
    USEFUL_COVERAGE: float = 0.60
    
    WEAK_RANK_IC: float = 0.01
    
    # ---------------- 冗余相关性阈值 ----------------
    REDUNDANCY_CORR_THRESHOLD: float = 0.85     # 截面相关性 > 0.85 标记为冗余
    
    # ---------------- Walk-Forward 因子选择参数 ----------------
    WF_TRAIN_YEARS: float = 2.0
    WF_VALIDATION_YEARS: float = 1.0


default_research_config = ResearchConfig()
