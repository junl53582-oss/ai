"""
因子研究与 Alpha 验证全局配置 (research/config.py)
定义因子研究视界、分层数、交易摩擦成本假设、评分权重、门禁阈值、截面最小门禁与 Purged Walk-Forward 参数。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ResearchConfig:
    # ---------------- 预测视界与交易时序定义 (Phase 1.2 P0-1) ----------------
    HORIZONS: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    PRIMARY_HORIZON: int = 20                   # 核心评估视界 (天)
    USE_EXCESS_RETURN: bool = True               # 是否优先使用基准超额收益 (去 Beta)
    ENTRY_PRICE_TYPE: str = "open"               # 最早执行价格类型 (T+1 开盘价 open)
    EXIT_PRICE_TYPE: str = "close"              # 退出价格类型 (T+H 收盘价 close)
    
    # ---------------- 分层与组合回测 ----------------
    NUM_QUANTILES: int = 5                      # 主分层组数 (Q1..Q5)
    DETAILED_QUANTILES: int = 10                # 细分层组数 (Q1..Q10)
    
    # ---------------- 样本与截面最小门禁 (Phase 1.2 P0-2) ----------------
    MIN_RESEARCH_SYMBOLS: int = 50              # 正式实证标的数门槛 (低于此值标记为 DEVELOPMENT_SAMPLE)
    MIN_DAILY_CROSS_SECTION: int = 4            # 每日最小截面有效样本数
    MIN_NEUTRALIZATION_CROSS_SECTION: int = 10  # 真实中性化最小截面样本数 (不足则 fail closed 返回 None)
    MIN_QUANTILE_CROSS_SECTION: int = 5         # 分层回测最小截面唯一因子值数
    MIN_VALID_DAYS_PER_YEAR: int = 60           # 年度稳定性评估最小有效交易日数
    
    # ---------------- 交易摩擦成本假设 ----------------
    COST_BPS_LIST: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0, 30.0])
    DEFAULT_COMMISSION_BPS: float = 3.0         # 券商佣金 (万3 = 3 bps)
    DEFAULT_STAMP_DUTY_BPS: float = 5.0         # 卖出印花税 (万5 = 5 bps)
    DEFAULT_SLIPPAGE_BPS: float = 2.0           # 冲击滑点 (万2 = 2 bps)
    
    # ---------------- 滚动与稳定性参数 ----------------
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [60, 120, 252])
    MARKET_REGIME_WINDOW: int = 20              # 划分牛熊震荡的窗口
    MARKET_REGIME_THRESHOLD: float = 0.03       # 20日收益 > +3% 牛市, < -3% 熊市, 其它震荡
    
    # ---------------- 多重检验与 Bootstrap ----------------
    FDR_ALPHA: float = 0.05                     # Benjamini-Hochberg FDR 显著性水平 (STRONG 门禁)
    FDR_USEFUL_ALPHA: float = 0.20              # USEFUL 因子宽松 FDR 显著性水平
    BOOTSTRAP_ROUNDS: int = 500                 # Block Bootstrap 抽样轮数
    BOOTSTRAP_BLOCK_SIZE: int = 20              # 时间块长度 (20交易日)
    BOOTSTRAP_CONFIDENCE: float = 0.95          # 置信区间
    
    # ---------------- 因子筛选评分权重 ----------------
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
    STRONG_IC_IR: float = 0.50                  # 年化 RankIC IR
    STRONG_SIGN_STABILITY: float = 0.70
    STRONG_COVERAGE: float = 0.80
    
    USEFUL_RANK_IC: float = 0.02
    USEFUL_IC_IR: float = 0.30
    USEFUL_COVERAGE: float = 0.60
    
    WEAK_RANK_IC: float = 0.01
    
    # ---------------- 冗余相关性阈值 ----------------
    REDUNDANCY_CORR_THRESHOLD: float = 0.85     # 截面或 IC 相关性 > 0.85 标记为冗余
    
    # ---------------- Purged Walk-Forward 因子选择参数 (P0-5) ----------------
    WF_TRAIN_YEARS: float = 2.0
    WF_VALIDATION_YEARS: float = 1.0
    WF_PURGE_DAYS: int = 25                     # Purge 隔离天数 >= max(HORIZONS)=20，杜绝跨区泄漏
    WF_EMBARGO_DAYS: int = 5                    # Embargo 滞后缓冲天数
    MIN_WF_FOLDS_FOR_CERTIFICATION: int = 3     # 达到 OOS_VALIDATED 所需的最少 Fold 门禁


default_research_config = ResearchConfig()
