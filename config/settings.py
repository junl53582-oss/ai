"""
全局量化系统配置模块
包含数据路径、股票池、A股交易规则参数、模型超参数以及风控阈值
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# A股行情数据源 (东方财富 eastmoney.com / 新浪 sina.com.cn 等) 均为国内站点。
# 若系统存在全局代理 (如 Clash / VPN 的 127.0.0.1:7890)，requests 会把国内流量
# 转发至海外节点，导致 AKShare 连接失败 (ProxyError / ConnectionError)。
# 因此在项目内自动清除代理环境变量，让 AKShare 直连国内数据源。
# 注意：该操作仅在当前 Python 进程内生效，不影响系统级或其他程序的代理设置。
_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")


def _clear_proxy_for_domestic_data() -> None:
    for key in _PROXY_ENV_KEYS:
        os.environ.pop(key, None)


_clear_proxy_for_domestic_data()

# 全局网络超时兜底: 东财/中证/新浪等接口偶发不响应会让整条管线永久挂起 (已实测多次)。
# 在最早导入的 settings 模块设置，使所有 requests 调用默认 30 秒超时，由各调用方重试/跳过。
import socket
socket.setdefaulttimeout(30)


@dataclass
class QuantConfig:
    # ---------------- 路径配置 ----------------
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_DIR: Path = field(init=False)
    RAW_DATA_DIR: Path = field(init=False)
    PARQUET_DIR: Path = field(init=False)
    FACTOR_DIR: Path = field(init=False)
    MODELS_DIR: Path = field(init=False)
    REPORTS_DIR: Path = field(init=False)

    # ---------------- 标的与时间范围 ----------------
    BENCHMARK_SYMBOL: str = "000300.SH"  # 沪深300指数作为超额收益基准
    START_DATE: str = "2020-01-01"
    END_DATE: Optional[str] = None       # None 表示自动同步到最新可用交易日
    
    # 是否允许在网络/接口失败时使用模拟仿真数据（生产环境必须为 False，避免误用）
    ALLOW_SYNTHETIC_DATA: bool = False

    # 股票池模式与幸存者偏差控制: "POINT_IN_TIME" (已认证动态时点成分股，零幸存者偏差), "INDEX_CONSTITUENTS", "STATIC"
    UNIVERSE_MODE: str = "POINT_IN_TIME"
    # 指数成分股模式使用的指数代码 (沪深300)
    INDEX_CODE: str = "000300"

    # 默认选股股票池 Profile ("HS300_CORE" / "ZZ500_GROWTH" / "TECH_INNOVATION" / "HIGH_DIVIDEND")
    UNIVERSE_PROFILE: str = "HS300_CORE"

    # 默认选股股票池（代表性蓝筹与行业龙头）
    DEFAULT_UNIVERSE: List[str] = field(default_factory=lambda: [
        "600519.SH", "000858.SZ", "601318.SH", "300750.SZ", "600036.SH",
        "002594.SZ", "601899.SH", "600900.SH", "000333.SZ", "600276.SH",
        "601166.SH", "002415.SZ", "603288.SH", "600887.SH", "000001.SZ",
        "300059.SZ", "600030.SH", "601012.SH", "002475.SZ", "601988.SH",
        "600048.SH", "000725.SZ", "601668.SH", "600104.SH", "002714.SZ",
        "600309.SH", "300760.SZ", "601288.SH", "601398.SH", "601857.SH"
    ])

    def set_universe_profile(self, profile_key: str):
        """一键切换选股股票池 Profile"""
        from .universe_profiles import UniverseProfileManager
        self.UNIVERSE_PROFILE = profile_key
        self.DEFAULT_UNIVERSE = UniverseProfileManager.get_symbols(profile_key)
        info = UniverseProfileManager.get_profile_info(profile_key)
        self.INDEX_CODE = info.get("index_code", "000300")

    # 股票可投资性过滤
    MIN_LISTING_DAYS: int = 60         # 过滤上市不足 60 个交易日的新股（次新股）
    MAX_STALE_PRICE_DAYS: int = 60     # 停牌/无行情最长可用最后价格估值天数（超期输出 WARNING）

    # 历史 ST 状态处理模式: "disable_st_rule" (默认历史无逐日ST时关闭), "strict", "unknown_as_normal"
    HISTORICAL_ST_MODE: str = "disable_st_rule"

    # ---------------- 预测与标签配置 ----------------
    # 标签持有期: 由 2 日提升至 20 日。
    # 依据: 2 日收益近似随机游走，信噪比极低；质量/成长等基本面信号为低频季度信号，
    # 对 2 日涨跌 IC≈0，必须在更长持有期 (20 日) 才能释放预测力。更长持有期同时
    # 显著降低量价噪声，是提升 OOS AUC 的关键杠杆。
    LABEL_HORIZON: int = 20            # 预测未来 20 个交易日的涨跌方向 (跑赢/跑输基准)
    # 任务类型: "classification" (涨跌二分类) | "regression" (连续收益回归) | "ranking" (排序学习)
    TASK_TYPE: str = "classification"
    # 回归模式标签列名 (连续超额收益率)
    LABEL_COLUMN: str = "label_excess_20d"
    # 分类模式标签列名 (1=涨/跑赢基准, 0=跌/跑输基准)
    LABEL_COLUMN_CLF: str = "label_up_down_20d"
    # 二分类阈值: 未来超额收益 > 此值判定为 1 (上涨/跑赢基准)
    LABEL_THRESHOLD: float = 0.0
    # 阈值模式: "fixed" (固定阈值 LABEL_THRESHOLD) | "cross_sectional_median" (每日截面中位数，市场中性)
    #        | "cross_sectional_extreme" (每日截面极端分组: top/bottom q 分位，剔除中间样本，拉大类别差距)
    # 横截面中位数阈值可消除大盘整体涨跌造成的标签偏斜，使正负样本天然均衡、更稳健
    # 极端分组进一步拉大正负样本的信号差，用于诊断模型对"强跑赢 vs 强跑输"的区分能力
    LABEL_THRESHOLD_MODE: str = "cross_sectional_extreme"
    # 极端分组分位: 每日截面 top/bottom 各 LABEL_EXTREME_QUANTILE 分位分别标 1/0，中间 1-2q 标记为 NaN(剔除)
    LABEL_EXTREME_QUANTILE: float = 0.30
    # 概率校准: 是否对模型输出做 Isotonic 概率校准 (提升概率可信度，通常能改善 Brier Score)
    PROBABILITY_CALIBRATION: bool = True

    # ---------------- 基本面财务因子 (质量/成长) 配置 ----------------
    # 启用后接入 AKShare 业绩报表 (stock_yjbb_em) 构建 ROE/毛利率/EPS/营收成长/利润成长 因子，
    # 这是纯量价系统完全缺失的异源 alpha 信号 (A股最 robust 的横截面溢价来源之一)。
    # 接口: stock_yjbb_em (东财「业绩报表」)，按报告期批量返回全市场数据，当前网络可通。
    # 自带「最新公告日期」字段，可做 Point-In-Time 披露时点对齐，杜绝未来函数。
    # 提供质量(ROE/毛利率/EPS) + 成长(营收/利润同比) 异源信号，是纯量价系统缺失的核心 alpha。
    ENABLE_FUNDAMENTALS: bool = True
    FUNDAMENTAL_DELAY_DAYS: int = 110   # PIT 时点延迟: 报告期 + 110 天 ≈ 实际披露窗口，杜绝未来函数
    FUNDAMENTAL_START_YEAR: str = "2018"  # 财务拉取起始年 (覆盖训练期所需的早期财报)
    # 高阶因子注册表 (另类/微观结构/遗传算法挖掘 共 19 个) 开关。
    # 实测: 因子数 85→107 后 OOS AUC 0.5284→0.5273、Alpha +20.47%→+12.16%，
    # 说明这 19 个弱特征稀释了信噪比。此处留出开关以便 A/B 隔离验证。
    ENABLE_REGISTRY_FACTORS: bool = False

    @property
    def is_classification(self) -> bool:
        """是否为分类模式"""
        return self.TASK_TYPE == "classification"

    @property
    def active_label_col(self) -> str:
        """当前任务类型使用的标签列"""
        return self.LABEL_COLUMN_CLF if self.is_classification else self.LABEL_COLUMN

    # ---------------- 走步训练 (Walk-Forward) 参数 ----------------
    TRAIN_WINDOW_YEARS: float = 1.5    # 训练集时间跨度 (年)，缩短以加速训练
    VAL_WINDOW_MONTHS: int = 3         # 验证集时间跨度 (月)
    TEST_WINDOW_MONTHS: int = 2        # 测试/实盘交易时间跨度 (月)，扩大以减半 Fold 数加速
    # Purged 隔离天数必须 >= LABEL_HORIZON，否则训练样本的未来收益会跨越到验证/测试期造成泄漏
    PURGE_GAP_DAYS: int = 25           # >= LABEL_HORIZON(20)，严格杜绝标签前视泄漏

    # ---------------- 计算性能参数 ----------------
    # 截面中性化并行进程数: 0 = 自动 (CPU核数-1, 上限8, 仅当交易日数>=60时启用),
    # 1 = 强制单线程串行, >=2 = 指定进程数。并行与串行的数值结果完全一致。
    NEUTRALIZATION_N_JOBS: int = 0
    NEUTRALIZATION_CHUNK_DAYS: int = 25  # 并行任务按天分块大小 (摊薄进程间序列化开销)

    # ---------------- LightGBM 超参数 ----------------
    # 分类模式参数 (涨跌二分类)
    # 参考 Qlib 官方经验: 金融数据信噪比极低，需强正则化 (lambda_l1/l2) + 较小树深防过拟合
    LGBM_PARAMS_CLF: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "min_child_samples": 150,
        "lambda_l1": 10.0,
        "lambda_l2": 20.0,
        "n_estimators": 800,
        "early_stopping_rounds": 80,
        "random_state": 42,
        "verbose": -1,
        # Phase A (2026-09-01): n_jobs 从 -1 (全核) 收敛为 4 —— 研究/回测大量 LightGBM
        # 训练下全核线程风暴在 Windows+libomp 触发原生崩溃 (进程静默消失, 无 traceback,
        # 无事件记录; 已致 6 次研究进程死亡)。限核后单次训练稍慢但稳定, 结果不变。
        "n_jobs": 4
    })
    # 回归模式参数 (连续超额收益)
    LGBM_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "random_state": 42,
        "verbose": -1,
        # Phase A: 同上, 限核防原生崩溃
        "n_jobs": 4
    })

    # ---------------- A股交易规则与组合配置 ----------------
    INITIAL_CASH: float = 1_000_000.0  # 初始本金 100 万人民币
    TOP_K_BUY: int = 8                 # 每日截面允许买入的最高排名 (如 Top 8)
    TOP_K_HOLD: int = 24               # 允许继续持仓的缓冲区排名 (验证 40/32 更差后回退: 20/24 为最优)
    REBALANCE_FREQ: int = 20           # 调仓周期 (验证 40 更差后回退: 20 为最优)
    LOT_SIZE: int = 100                # A股最小买入单位（1手=100股）
    
    # 涨跌停幅度基准
    PRICE_LIMIT_MAIN: float = 0.10     # 主板涨跌停幅度 10%
    PRICE_LIMIT_CHINEXT: float = 0.20  # 创业板/科创板涨跌停幅度 20%
    PRICE_LIMIT_ST: float = 0.05       # ST股票涨跌停幅度 5%

    # ---------------- 交易摩擦成本 ----------------
    STAMP_DUTY: float = 0.0005         # 印花税：卖出单边 0.05% (A股现行标准)
    COMMISSION_RATE: float = 0.00025   # 券商佣金：双边万分之 2.5
    MIN_COMMISSION: float = 5.0        # 佣金最低 5 元
    SLIPPAGE_RATE: float = 0.001       # 预期滑点：双边 0.1%

    # ---------------- 风控阈值 ----------------
    STOP_LOSS_PCT: float = 0.08        # 个股硬止损线：-8%
    TRAILING_STOP_PCT: float = 0.05    # 跟踪止盈回撤线：高点回撤 5%
    MAX_DRAWDOWN_LIMIT: float = 0.12   # 策略最大回撤阈值：12%
    CIRCUIT_TARGET_EXPOSURE: float = 0.30 # 触发回撤熔断后的目标持仓暴露上限 30%
    MAX_SECTOR_EXPOSURE: float = 0.30  # 单一行业最大持仓上限 30% (硬约束)

    def __post_init__(self):
        self.DATA_DIR = self.BASE_DIR / "data_storage"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PARQUET_DIR = self.DATA_DIR / "parquet"
        self.FACTOR_DIR = self.DATA_DIR / "factors"
        self.FACTORS_DIR = self.FACTOR_DIR
        self.MODELS_DIR = self.BASE_DIR / "saved_models"
        self.MODEL_DIR = self.MODELS_DIR
        self.REPORTS_DIR = self.BASE_DIR / "reports"

        # 自动创建目录
        for path in [self.DATA_DIR, self.RAW_DATA_DIR, self.PARQUET_DIR, 
                     self.FACTOR_DIR, self.MODELS_DIR, self.REPORTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)


# 全局单例配置
settings = QuantConfig()
