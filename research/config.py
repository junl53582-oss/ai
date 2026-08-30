"""
因子研究与 Alpha 验证全局配置 (research/config.py)
Phase 1.5: 严密缓存失效、生产交易状态对齐与延迟卖出 (Delayed Exit) 规则
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ResearchConfig:
    # ---------------- 交易与结算规则定义 (Phase 1.4/1.5 P0-1/P0-5) ----------------
    SETTLEMENT_RULE: str = "A_SHARE_T_PLUS_1_NO_SAME_DAY_SELL"
    SIGNAL_TIMESTAMP: str = "T_close"            # 信号生成时间: T 日收盘后
    ENTRY_OFFSET: int = 1                        # 买入时点: T+1 交易日
    ENTRY_PRICE_TYPE: str = "open"               # 买入价格类型: 开盘价 open
    EARLIEST_EXIT_OFFSET: int = 2                # 最早卖出时点: T+2 交易日 (A股 T+1 禁止日内回转)
    EXIT_PRICE_TYPE: str = "open"                # 卖出价格类型: 开盘价 open (或 close)
    MAX_UNEXECUTED_EXIT_DAYS: int = 10           # 跌停/停牌导致无法卖出时，最长展期等待交易日数 (Delayed Exit)
    
    # ---------------- 预测视界与收益定义 ----------------
    HORIZONS: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    PRIMARY_HORIZON: int = 20                    # 核心评估视界 (交易日)
    USE_EXCESS_RETURN: bool = True               # 是否优先使用基准超额收益 (去 Beta)
    BENCHMARK_CLOSE_COL: str = "benchmark_close" # 基准收盘价列名
    BENCHMARK_OPEN_COL: str = "benchmark_open"   # 基准开盘价列名
    ALLOW_BENCHMARK_FALLBACK_FOR_TESTS: bool = False # 生产环境严格禁用基准缺省回退 (Fail-Closed)
    
    # ---------------- 交易权限与可交易性过滤 (Phase 1.5 P0-4 / P0-5) ----------------
    ALLOW_ST_TRADING: bool = False               # 是否允许交易 ST / *ST 股票
    
    # ---------------- 分层与组合回测 ----------------
    NUM_QUANTILES: int = 5                       # 主分层组数 (Q1..Q5)
    DETAILED_QUANTILES: int = 10                 # 细分层组数 (Q1..Q10)
    
    # ---------------- 样本与截面生产硬门槛 (Phase 1.5 P1-2) ----------------
    MIN_RESEARCH_SYMBOLS: int = 50               # 实证研究最低标的数门限
    MIN_PRODUCTION_SYMBOLS: int = 300            # 生产级实盘推荐标的数门限 (如 CSI300)
    MIN_DAILY_CROSS_SECTION: int = 4             # 每日最小截面有效样本数
    MIN_DAILY_CROSS_SECTION_PRODUCTION: int = 150 # 生产级每日最小截面中位数样本数 (CSI300 核心)
    MIN_NEUTRALIZATION_CROSS_SECTION: int = 10   # 截面多元 OLS 中性化最小有效样本数
    MIN_VALID_DAYS_PER_YEAR: int = 60            # 年度稳定性评估最小有效交易日数
    
    # ---------------- 交易摩擦成本假设 (A股真实非对称费率) ----------------
    COST_BPS_LIST: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0, 30.0])
    DEFAULT_COMMISSION_BPS: float = 3.0          # 券商佣金 (双边 万3 = 3 bps)
    DEFAULT_STAMP_DUTY_BPS: float = 5.0          # 卖出印花税 (单边卖出 万5 = 5 bps, 买入为 0)
    DEFAULT_SLIPPAGE_BPS: float = 2.0            # 冲击滑点 (双边 万2 = 2 bps)
    
    # ---------------- 滚动与稳定性参数 ----------------
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [60, 120, 252])
    MARKET_REGIME_WINDOW: int = 20               # 划分牛熊震荡的窗口
    MARKET_REGIME_THRESHOLD: float = 0.03        # 20日收益 > +3% 牛市, < -3% 熊市, 其它震荡
    
    # ---------------- 多重检验与 Bootstrap ----------------
    FDR_ALPHA: float = 0.05                      # Benjamini-Hochberg FDR 显著性水平 (STRONG 门禁)
    FDR_USEFUL_ALPHA: float = 0.20               # USEFUL 因子宽松 FDR 显著性水平
    BOOTSTRAP_ROUNDS: int = 500                  # Block Bootstrap 抽样轮数
    BOOTSTRAP_BLOCK_SIZE: int = 20               # 时间块长度 (20交易日)
    BOOTSTRAP_CONFIDENCE: float = 0.95           # 置信区间
    
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
    STRONG_IC_IR: float = 0.50                   # 年化 RankIC IR
    STRONG_SIGN_STABILITY: float = 0.70
    STRONG_COVERAGE: float = 0.80
    
    USEFUL_RANK_IC: float = 0.02
    USEFUL_IC_IR: float = 0.30
    USEFUL_COVERAGE: float = 0.60
    
    WEAK_RANK_IC: float = 0.01
    
    # ---------------- 冗余相关性阈值 ----------------
    REDUNDANCY_CORR_THRESHOLD: float = 0.85      # 截面或 IC 相关性 > 0.85 标记为冗余
    
    # ---------------- Purged Walk-Forward 因子选择参数 ----------------
    WF_TRAIN_YEARS: float = 1.5
    WF_VALIDATION_YEARS: float = 0.5
    WF_PURGE_DAYS: int = 25                      # Purge 隔离天数 >= max(HORIZONS)+1
    WF_EMBARGO_DAYS: int = 5                     # Embargo 滞后缓冲天数
    MIN_WF_FOLDS_FOR_CERTIFICATION: int = 3      # 达到 OOS_VALIDATED 所需的最少 Fold 门禁


default_research_config = ResearchConfig()
