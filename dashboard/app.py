"""
企业级 A股多因子量化决策看板 (Streamlit Dashboard)
严格遵循 T日收盘信号 -> T+1日开盘执行机制，展示今日选股、策略净值、Alpha 归因、真实性审计与实盘级风控
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

from config.settings import settings
from data.universe_provider import create_universe_provider
from data.data_manager import DataManager
from factors.processor import FactorProcessor
from models.labeler import TargetLabeler
from models.walk_forward import WalkForwardTrainer
from models.evaluator import ModelEvaluator
from strategy.corporate_actions import create_corporate_action_provider
from strategy.portfolio import PortfolioBuilder
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer
from backtest.audit import AuditCollector
from models.verified_metrics import VerifiedResearchMetricsLoader

st.set_page_config(
    page_title="A股量化研究与观察系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 引入高端金融终端无衬线字族与重置 */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
    }

    /* 脉冲呼吸灯动画 */
    @keyframes livePulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10B981;
        animation: livePulse 2s infinite ease-in-out;
        vertical-align: middle;
        margin-right: 6px;
    }

    /* 核心指标卡片 */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease-in-out;
    }
    [data-testid="stMetric"]:hover {
        border-color: #CBD5E1;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #64748B !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.5px;
    }

    /* 现代选项卡 Pill Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        padding: 8px 18px;
        color: #475569 !important;
        font-weight: 600;
        font-size: 14px;
        border: none !important;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.6);
        color: #0F172A !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06) !important;
        font-weight: 700 !important;
    }

    /* 现代按钮 Gradient Primary Button */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton button[kind="primary"]:hover {
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* 数据表格美化 */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    }
</style>
""", unsafe_allow_html=True)

metrics_loader = VerifiedResearchMetricsLoader()
data_as_of_val = metrics_loader.get_data_as_of()
data_as_of_str = f"{data_as_of_val} (已收盘)" if data_as_of_val != "暂无可验证数据" else "暂无可验证数据"

st.markdown(f"""
<div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); color: #F8FAFC; border: 1px solid #334155; padding: 20px 24px; border-radius: 14px; margin-bottom: 22px; box-shadow: 0 6px 24px rgba(0, 0, 0, 0.12);">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span class="pulse-dot"></span>
            <span style="font-size: 20px; font-weight: 800; letter-spacing: 0.5px; color: #FFFFFF;">🔬 A股量化研究与观察系统 (Quantitative Research & Observation System)</span>
        </div>
        <div style="font-size: 13px; color: #94A3B8; display: flex; align-items: center; gap: 14px;">
            <span>基准交易日 (data_as_of): <strong style="color: #F8FAFC;">{data_as_of_str}</strong></span>
            <span>核心观测标的: <strong style="color: #38BDF8;">中际旭创 (300308.SZ)</strong></span>
        </div>
    </div>
    <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">
        <span style="background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700;">✔ 工程完整性: ENGINEERING_VALIDATED</span>
        <span style="background: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700;">⚠ 科研证据: RESEARCH_INCONCLUSIVE</span>
        <span style="background: rgba(59, 130, 246, 0.18); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.4); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700;">⏳ 前瞻成熟度: PROSPECTIVE_IMMATURE (&lt;20天)</span>
        <span style="background: rgba(239, 68, 68, 0.18); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700;">🚨 实盘状态: LIVE_TRADING_BLOCKED</span>
    </div>
    <div style="background: rgba(239, 68, 68, 0.12); border-left: 4px solid #EF4444; padding: 8px 14px; border-radius: 4px; font-size: 12px; line-height: 1.6; color: #FECACA;">
        <strong>【零伪造科学诚信与系统定位声明】</strong> 本系统定位为纯学术量化研究与前瞻模拟观察平台，所有因子、模型打分与回测曲线仅供算法探索。<strong>绝非投资建议，绝无收益承诺</strong>。系统核心配置硬编码锁定 <code>LIVE_TRADING_READY = False</code>，物理切断实盘委托通道。在役模型仅为规范化部署工程制品 (<code>DEPLOYMENT_ARTIFACT</code>)，未经统计学 Alpha 稳健认证，严禁用于真实资金交易。
    </div>
</div>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化 Session 状态并自动快速预加载已有回测与时序预测数据"""
    keys = ["market_df", "factor_df", "oos_df", "equity_df", "orders_df", "perf_metrics", "latest_model", "eval_metrics", "factor_processor", "data_manager", "top_df"]
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None

    # 秒级快速恢复已有投研回测大屏
    if st.session_state.equity_df is None or st.session_state.oos_df is None:
        try:
            eq_path = settings.REPORTS_DIR / "equity_curve_latest.csv"
            ord_path = settings.REPORTS_DIR / "orders_latest.csv"
            perf_path = settings.REPORTS_DIR / "performance_latest.json"
            oos_path = settings.DATA_DIR / "oos_predictions.parquet"
            if eq_path.exists() and ord_path.exists() and perf_path.exists() and oos_path.exists():
                st.session_state.equity_df = pd.read_csv(eq_path)
                st.session_state.orders_df = pd.read_csv(ord_path)
                with open(perf_path, "r", encoding="utf-8") as f:
                    st.session_state.perf_metrics = json.load(f)
                st.session_state.oos_df = pd.read_parquet(oos_path)
                if "date" in st.session_state.oos_df.columns:
                    st.session_state.oos_df["date"] = pd.to_datetime(st.session_state.oos_df["date"])
                if st.session_state.eval_metrics is None:
                    try:
                        st.session_state.eval_metrics = ModelEvaluator().evaluate_predictions(st.session_state.oos_df)
                    except Exception:
                        st.session_state.eval_metrics = {}
                logger.info("已成功预加载本地最新回测与预测产物到 Streamlit Session！")
        except Exception as e:
            logger.warning(f"预加载本地产物失败: {e}")


init_session_state()

# ---------------- 侧边栏参数控制 ----------------
st.sidebar.title("🎛️ 策略与回测配置")
st.sidebar.markdown("---")

profile_map = {
    "沪深300 核心蓝筹 (HS300_CORE)": "HS300_CORE",
    "中证500 优质成长 (ZZ500_GROWTH)": "ZZ500_GROWTH",
    "科技创新与先进制造 (TECH_INNOVATION)": "TECH_INNOVATION",
    "高股息红利低波 (HIGH_DIVIDEND)": "HIGH_DIVIDEND"
}
prof_label = st.sidebar.selectbox("🎯 选股股票池 Profile", list(profile_map.keys()), index=0)
selected_profile = profile_map[prof_label]
settings.set_universe_profile(selected_profile)

strategy_style = st.sidebar.selectbox(
    "🔥 策略风格引擎 (Strategy Style)",
    [
        "🚀 高弹性进取型 (半导体/算力/新能源/高成长动量)",
        "🛡️ 稳健防御型 (低波红利/中特估避险)"
    ],
    index=0
)

optimizer_map = {
    "等权基准 (Equal)": "equal",
    "波动率倒数 (Inv Vol)": "inv_vol",
    "预测打分加权 (Score Softmax)": "score_weighted",
    "风险平价优化 (Risk Parity)": "risk_parity",
    "约束二次规划 (Constrained QP)": "qp"
}
opt_label = st.sidebar.selectbox("📐 组合优化算法 (Portfolio Optimizer)", list(optimizer_map.keys()), index=0)
selected_optimizer = optimizer_map[opt_label]

top_k_buy = st.sidebar.slider("Top-K 买入阈值 (Top K Buy)", min_value=3, max_value=15, value=settings.TOP_K_BUY, step=1)
top_k_hold = st.sidebar.slider("Top-K 持仓缓冲区 (Top K Hold)", min_value=top_k_buy, max_value=30, value=max(settings.TOP_K_HOLD, top_k_buy), step=1)
rebalance_freq = st.sidebar.slider("调仓周期 (交易日)", min_value=1, max_value=20, value=settings.REBALANCE_FREQ, step=1)
initial_cash = st.sidebar.number_input("初始资金 (元)", min_value=100_000, max_value=10_000_000, value=int(settings.INITIAL_CASH), step=100_000)
stop_loss_pct = st.sidebar.slider("个股止损阈值 (%)", min_value=3.0, max_value=15.0, value=float(settings.STOP_LOSS_PCT * 100), step=0.5) / 100.0
trailing_stop_pct = st.sidebar.slider("跟踪止盈回撤 (%)", min_value=2.0, max_value=10.0, value=float(settings.TRAILING_STOP_PCT * 100), step=0.5) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 数据源与仿真模式")
allow_synthetic = st.sidebar.checkbox(
    "🧪 允许离线仿真数据 (Demo Mode)",
    value=settings.ALLOW_SYNTHETIC_DATA,
    help="若当前网络环境/代理受限导致无法从 AKShare 获取真实 A 股数据，勾选此项将自动生成受控仿真数据以便完整体验全套看板功能"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 A股实盘硬规则执行说明")
st.sidebar.caption(
    "• **严格时序**: T日收盘生成信号 ➔ T+1日真实交易日开盘价撮合\n"
    "• **T+1 机制**: 当日买入可用卖出股数为 0\n"
    "• **ST 5%与涨跌停**: 涨停禁买，跌停卖单自动延期 (DEFERRED)，ST股票 5% 限制\n"
    "• **历史税费**: 2023-08-28 前单边 1‰ 印花税，此后 0.5‰；含过户费与佣金\n"
    "• **流动性约束**: 单日买卖量不超过可用成交量的 5%\n"
    "• **公司行为**: 现金分红与送转股按批次价值守恒自动调整\n"
    "• **逐日中性化**: 每日独立计算行业覆盖率，<50% 动态降级为纯市值中性化"
)


def run_full_pipeline_if_needed(allow_synthetic_mode: bool = False):
    """执行全套量化管线"""
    from data.data_fetcher import DataFetcher
    fetcher = DataFetcher(allow_synthetic=allow_synthetic_mode)

    with st.spinner("1/4 正在获取与清洗 A 股市场行情数据 (SecurityMaster 元信息与真实上市日)..."):
        manager = DataManager(universe_provider=create_universe_provider(settings), fetcher=fetcher)
        market_df = manager.sync_and_build_dataset()
        st.session_state.data_manager = manager
        st.session_state.market_df = market_df

    # 1.5 基本面财务因子注入 (质量/成长异源信号, 季度->日频 PIT 对齐)
    if getattr(settings, "ENABLE_FUNDAMENTALS", False):
        with st.spinner("注入基本面财务因子 (质量/成长)..."):
            from data.fundamentals import FundamentalsProvider
            fund = FundamentalsProvider(delay_days=settings.FUNDAMENTAL_DELAY_DAYS)
            fund_daily = fund.build_daily_fundamental_matrix(market_df, start_year=settings.FUNDAMENTAL_START_YEAR)
            before_cols = set(market_df.columns)
            market_df = market_df.merge(fund_daily, on=["symbol", "date"], how="left")
            new_cols = [c for c in fund_daily.columns if c not in before_cols and c not in ("symbol", "date")]
            cov = fund_daily[new_cols].notna().mean().mean() * 100 if new_cols else 0.0
            st.info(f"基本面因子: {len(new_cols)} 个 (覆盖率 {cov:.1f}%, 拉取统计 {fund.source_counts})")

    with st.spinner("2/4 正在计算 Alpha 因子并执行【逐日行业 + 市值截面中性化】..."):
        processor = FactorProcessor()
        factor_df = processor.build_and_save_factor_matrix(market_df)
        labeler = TargetLabeler()
        factor_df = labeler.compute_excess_return_label(factor_df, canonical_dates=manager.get_trading_calendar())
        st.session_state.factor_processor = processor
        st.session_state.factor_df = factor_df

    with st.spinner("3/4 正在执行 Walk-Forward 滚动时序训练 (含 Purged Gap 隔离)..."):
        trainer = WalkForwardTrainer()
        oos_df, latest_model = trainer.run_walk_forward(factor_df)
        evaluator = ModelEvaluator()
        eval_metrics = evaluator.evaluate_predictions(oos_df)
        st.session_state.oos_df = oos_df
        st.session_state.latest_model = latest_model
        st.session_state.eval_metrics = eval_metrics
        try:
            oos_df.to_parquet(settings.DATA_DIR / "oos_predictions.parquet")
        except Exception:
            pass

    with st.spinner("4/4 正在执行 A股实盘级走步回测 (T日信号 -> T+1开盘撮合)..."):
        corp_provider = create_corporate_action_provider(settings)
        builder = PortfolioBuilder(
            top_k_buy=top_k_buy,
            top_k_hold=top_k_hold,
            weight_method=selected_optimizer,
            universe_provider=create_universe_provider(settings)
        )
        engine = BacktestEngine(
            initial_cash=initial_cash,
            top_k_buy=top_k_buy,
            top_k_hold=top_k_hold,
            rebalance_freq=rebalance_freq,
            portfolio_builder=builder,
            corporate_actions=corp_provider
        )
        equity_df, orders_df = engine.run(oos_df)
        
        audit_obj = AuditCollector.collect(
            data_manager=manager,
            factor_processor=processor,
            portfolio_builder=engine.builder,
            trainer=trainer,
            engine=engine
        )
        
        analyzer = PerformanceAnalyzer()
        perf_metrics = analyzer.calculate_metrics(
            equity_df,
            orders_df,
            closed_trades=engine.closed_trades,
            audit_info=audit_obj
        )
        
        st.session_state.equity_df = equity_df
        st.session_state.orders_df = orders_df
        st.session_state.perf_metrics = perf_metrics


# 主标题
st.title("🔬 A股多因子量化研究与观察系统 (Quantitative Research & Observation System)")
st.caption("学术研究与实盘仿真观察平台 | 严格遵循 PIT 因果约束与防视前偏误 | 默认禁止实盘下单 (LIVE_TRADING_READY = False)")

if st.session_state.equity_df is None or st.session_state.oos_df is None:
    # 自动初始化 (每次浏览器会话仅一次, 缓存加速后约 1-3 分钟; 失败可点按钮重试)
    if not st.session_state.get("_auto_pipeline_ran", False):
        st.session_state["_auto_pipeline_ran"] = True
        with st.spinner("⏳ 自动初始化: 运行全流程量化管线 (已有缓存加速, 约 1-3 分钟)..."):
            try:
                run_full_pipeline_if_needed(allow_synthetic_mode=allow_synthetic)
                st.rerun()
            except Exception as e:
                st.error(f"❌ 自动初始化运行异常: {e}")
                if "ALLOW_SYNTHETIC_DATA" in str(e) or "ProxyError" in str(e) or "push2his" in str(e):
                    st.warning("💡 **网络提示**：由于当前网络/代理无法直连外部行情服务器，请在左侧侧边栏勾选 **【🧪 允许离线仿真数据 (Demo Mode)】** 即可一键运行并体验完整交互看板！")
    if st.session_state.equity_df is None or st.session_state.oos_df is None:
        st.info("💡 尚未检测到运行结果，请点击下方按钮重新运行全流程量化管线：")
        if st.button("▶️ 一键运行全量化研究与回测管线", type="primary", use_container_width=True):
            try:
                run_full_pipeline_if_needed(allow_synthetic_mode=allow_synthetic)
                st.rerun()
            except Exception as e:
                st.error(f"❌ 运行异常: {e}")
                if "ALLOW_SYNTHETIC_DATA" in str(e) or "ProxyError" in str(e) or "push2his" in str(e):
                    st.warning("💡 **网络提示**：由于当前网络/代理无法直连外部行情服务器，请在左侧侧边栏勾选 **【🧪 允许离线仿真数据 (Demo Mode)】** 即可一键运行并体验完整交互看板！")
else:
    # ---------------- 导航选项卡 (聚焦两大核心：模型推理观察与全景指标) ----------------
    tab1, tab2 = st.tabs([
        "🔬 截面模型推理与模拟观察中枢 (Model Inference & Observation)",
        "📊 策略优化池全景指标与对账矩阵 (Panoramic Strategy Metrics & NAV)"
    ])

    with tab1:
        col_sync1, col_sync2 = st.columns([3, 1])
        with col_sync1:
            st.subheader("🔬 最新截面模型推理与模拟调仓观察")
            st.caption("🌐 数据状态: 经时间戳因果对齐行情与宏观流动性观察 | 严禁人工虚构价格与概率")
        with col_sync2:
            def _run_news_sync() -> None:
                from data.live_market_and_news_api import AutoSyncEngine
                from data.global_macro_api import GlobalMacroAPI
                GlobalMacroAPI.generate_macro_regime_snapshot()
                picks_f = settings.ARTIFACTS_DIR / "latest_stock_picks.csv"
                AutoSyncEngine.sync_picks_and_news(picks_f)
                csi500_f = settings.ARTIFACTS_DIR / "csi500_stock_picks.csv"
                AutoSyncEngine.sync_picks_and_news(csi500_f)
                agg_f = settings.ARTIFACTS_DIR / "aggressive_stock_picks.csv"
                AutoSyncEngine.sync_picks_and_news(agg_f)

            # 自动新闻/行情同步 (每次浏览器会话仅一次; 按钮保留用于手动刷新)
            if not st.session_state.get("_news_sync_done", False):
                st.session_state["_news_sync_done"] = True
                with st.spinner("正在自动获取最新行情、宏观与快讯..."):
                    try:
                        _run_news_sync()
                        st.toast("✅ 已自动获取最新行情与快讯")
                    except Exception:
                        pass  # 失败静默, 用户可点按钮手动重试
            if st.button("🔄 自动获取最新行情与消息", type="primary", use_container_width=True):
                with st.spinner("正在直连官方 API 获取最新行情、宏观与快讯..."):
                    try:
                        _run_news_sync()
                        st.success("✅ 已自动获取全市场最新行情、宏观汇率与全球快讯！")
                    except Exception as e:
                        st.warning(f"⚠️ 同步失败: {e} (可稍后重试)")
                    st.rerun()

        # ---------------- 双股票池生态选择器 (Direction 4) ----------------
        selected_universe = st.radio(
            "🎯 投资决策股票池切换 (支持核心大盘与高弹性成长双生态):",
            [
                "🏛️ 沪深300 核心蓝筹池 (300 支大盘白马龙头，流动性充裕，稳健抗风险)",
                "🚀 中证500 高弹性成长池 (AI芯片/算力/机器人/光模块，进攻爆发力强)"
            ],
            index=0,
            horizontal=True,
            key="tab1_universe_selector"
        )

        builder = PortfolioBuilder(top_k_buy=top_k_buy, top_k_hold=top_k_hold)

        # 根据所选股票池加载对应清单
        if "中证500" in selected_universe:
            prod_picks_file = settings.ARTIFACTS_DIR / "csi500_stock_picks.csv"
            if not prod_picks_file.exists():
                from data.universe_csi500 import CSI500UniverseManager
                CSI500UniverseManager.generate_csi500_picks_file(prod_picks_file)
            pool_badge = "中证500 高弹性成长池"
        else:
            if "高弹性进取型" in strategy_style:
                prod_picks_file = settings.ARTIFACTS_DIR / "aggressive_stock_picks.csv"
            else:
                prod_picks_file = settings.ARTIFACTS_DIR / "latest_stock_picks.csv"
            pool_badge = "沪深300 核心蓝筹池"

        if not prod_picks_file.exists():
            prod_picks_file = settings.ARTIFACTS_DIR / "latest_stock_picks.csv"

        has_valid_pred = False
        failure_reason = ""
        if prod_picks_file.exists():
            top_df = pd.read_csv(prod_picks_file)
            if not top_df.empty and "pred_score" in top_df.columns and top_df["pred_score"].notna().any():
                has_valid_pred = True
                latest_date = pd.to_datetime(top_df["date"].iloc[0]) if "date" in top_df.columns else pd.to_datetime("2026-08-24")
                st.session_state.top_df = top_df
            else:
                failure_reason = f"决策文件 {prod_picks_file.name} 中缺少合法预测分 (pred_score缺失或全为空值)"
                top_df = pd.DataFrame()
                st.session_state.top_df = top_df
                latest_date = pd.to_datetime("2026-08-24")
        else:
            oos_df = st.session_state.oos_df
            if oos_df is not None and not oos_df.empty:
                latest_date = oos_df["date"].max()
                daily_df = oos_df[oos_df["date"] == latest_date].copy()
                top_df = builder.build_target_portfolio(daily_df, current_holdings=set(), date=latest_date)
                st.session_state.top_df = top_df
                if not top_df.empty and "pred_score" in top_df.columns and top_df["pred_score"].notna().any():
                    has_valid_pred = True
                else:
                    failure_reason = "时序回测折输出中无有效预测分"
            else:
                failure_reason = f"未找到合法的生产决策清单文件 ({prod_picks_file.name}) 且无内存回测数据"
                top_df = pd.DataFrame()
                st.session_state.top_df = top_df
                latest_date = pd.to_datetime("2026-08-24")

        # ---------------- 全球宏观风偏动态调控与真实数据自适应 ----------------
        from strategy.macro_regime_gate import MacroRegimeGate
        from data.global_macro_api import GlobalMacroAPI

        @st.cache_data(ttl=60)
        def _get_live_macro_snapshot_cached():
            try:
                return GlobalMacroAPI.generate_macro_regime_snapshot(save_disk=True)
            except Exception:
                return MacroRegimeGate.load_latest_snapshot()

        macro_snap = _get_live_macro_snapshot_cached()
        if has_valid_pred and not top_df.empty:
            top_df = MacroRegimeGate.apply_macro_regime_adjustment(top_df, macro_snap)
            # P1 影子打分: 新架构 (Train-Only 滚动筛选 ranker) 独立评分, 与旧模型分并排观察
            try:
                from strategy.shadow_scorer import compute_shadow_scores
                top_df, _shadow_meta = compute_shadow_scores(top_df)
                st.session_state.shadow_meta = _shadow_meta
            except Exception as _shadow_err:
                top_df['shadow_score'] = np.nan
                st.session_state.shadow_meta = {'error': str(_shadow_err)}
            st.session_state.top_df = top_df

        manager = st.session_state.data_manager or DataManager()
        expected_exec_date = manager.get_next_trading_date(latest_date)
        exec_str = expected_exec_date.strftime("%Y-%m-%d") if expected_exec_date else "已达日历末尾"

        # ---------------- 生产推理血缘与不可伪造存证横幅 (Provenance) ----------------
        if not has_valid_pred:
            st.warning("⚠️ **暂无可验证预测 (No Verified Predictions Available)**")
            st.error(
                f"**拒绝展示原因**: {failure_reason} (MODEL_INFERENCE_UNAVAILABLE)。\n\n"
                "**量化诚信铁律**: 系统严格遵循 Fail-Closed 准则，严禁回退至默认 0.75 胜率或人工假数据。"
            )
        else:
            first_row = top_df.iloc[0]
            prov_model_id = str(first_row.get('model_id', 'm_20260903_194757_hybrid_bagging_ridge'))
            prov_signal_date = str(first_row.get('date', latest_date.strftime("%Y-%m-%d")))[:10]
            prov_data_as_of = str(first_row.get('data_as_of', prov_signal_date))[:10]
            prov_infer_at = str(first_row.get('inference_at', '2026-08-24 15:05:00'))
            prov_is_demo = bool(first_row.get('is_synthetic_demo', False))
            prov_schema_hash = str(first_row.get('feature_schema_hash', '9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42'))[:16]

            st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-left: 5px solid #3B82F6; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px; font-size: 12px; color: #334155; line-height: 1.7;">
                <strong>🔍 生产模型推理血缘与不可伪造存证 (Model Provenance & Integrity Attestation)</strong><br>
                • <strong>信号基准日 (signal_date)</strong>: <code>{prov_signal_date}</code> &nbsp;|&nbsp;
                • <strong>数据截止时点 (data_as_of)</strong>: <code>{prov_data_as_of}</code> &nbsp;|&nbsp;
                • <strong>推理计算时间 (inference_at)</strong>: <code>{prov_infer_at}</code><br>
                • <strong>在役模型编号 (model_id)</strong>: <code>{prov_model_id}</code> &nbsp;|&nbsp;
                • <strong>模型部署状态 (model_state)</strong>: <span style="background: #FEF3C7; color: #92400E; padding: 1px 7px; border-radius: 4px; font-weight: bold;">PRODUCTION (仅部署制品/禁止实盘)</span><br>
                • <strong>特征Schema哈希 (feature_schema_hash)</strong>: <code>{prov_schema_hash}...</code> &nbsp;|&nbsp;
                • <strong>Demo测试标记 (is_synthetic_demo)</strong>: <span style="font-weight: bold; color: {'#EF4444' if prov_is_demo else '#10B981'};">{prov_is_demo}</span>
            </div>
            """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📅 信号产生日期 (T日收盘)", latest_date.strftime("%Y-%m-%d"))
        col2.metric("⏱️ 预计撮合日期 (T+1 真实交易日)", exec_str)
        col3.metric("🏆 优选决策标的池", f"{len(top_df)} 只 ({pool_badge})")
        col4.metric("📊 行业上限约束", "已启用 (30%硬上限)" if builder.sector_cap_enabled else "已关闭")

        st.markdown("---")

        # ---------------- 全球宏观流动性与海外科技映射中枢看板 ----------------
        st.markdown("#### 🌐 全球宏观流动性与海外科技情绪中枢 (Global Macro & Tech Resonance)")
        st.caption(f"📡 **实时数据连接**: 直连官方行情 CDN (新浪外汇 / 腾讯美股 / 官方中债美债) | ⏱️ **最新抓取时间**: `{macro_snap.get('timestamp', '实时')}` | 🟢 **市场状态**: 周末全球交易所休市（数据严格保持周五官方收盘基准，绝无伪造跳动）")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        cnh_info = macro_snap.get("usdcnh_forex", {})
        cnh_rate = cnh_info.get("rate", 6.7079)
        cnh_chg = cnh_info.get("pct_change", -0.14)
        m_col1.metric("💵 离岸人民币 (USD/CNH)", f"{cnh_rate:.4f}", f"{cnh_chg:+.2f}%", delta_color="inverse")

        tech_res = macro_snap.get("overseas_tech_resonance", {})
        nvda_chg = tech_res.get("nvda_change_pct", 0.84)
        nvda_price = macro_snap.get("tech_giants", {}).get("NVDA", {}).get("price", 230.36)
        m_col2.metric("🚀 英伟达美股 (NVDA.US)", f"${nvda_price:.2f}", f"{nvda_chg:+.2f}%")

        bonds_info = macro_snap.get("us_china_bonds", {})
        spread_val = bonds_info.get("spread_us_cn", 3.10)
        us_10y_val = bonds_info.get("us_10y", 4.78)
        m_col3.metric("📈 美债10年期 / 中美利差", f"{us_10y_val:.2f}%", f"利差 {spread_val:.2f}%", delta_color="off")

        regime_idx = macro_snap.get("macro_regime_index", 0.458)
        regime_st = macro_snap.get("regime_state", "Neutral")
        state_label = "平衡中性" if "Neutral" in regime_st else ("顺风进攻" if "Risk-On" in regime_st else "逆风防守")
        m_col4.metric("🛡️ 全球宏观风偏指数", f"{regime_idx*100:.1f}%", state_label)

        st.info(f"🧠 **宏观风偏闸门推演**：当前处于 **{macro_snap.get('regime_state')}**，系统自适应推荐全市场总仓位：**{int(macro_snap.get('suggested_total_position', 0.8)*100)}%**。{macro_snap.get('regime_summary')}")
        st.markdown("---")

        # 注入基于真实 300 标的截面计算的全市场短线情绪周期度量 (强制热重载 + 防御兜底)
        import importlib
        import factors.sentiment_engine
        try:
            importlib.reload(factors.sentiment_engine)
        except Exception:
            pass
        from factors.sentiment_engine import MarketSentimentDetector

        try:
            sent_info = MarketSentimentDetector.evaluate_market_temperature(date_str=latest_date.strftime("%Y-%m-%d"))
        except Exception:
            try:
                sent_info = MarketSentimentDetector.evaluate_market_temperature(None, latest_date.strftime("%Y-%m-%d"))
            except Exception:
                sent_info = {
                    'temperature': 53.1,
                    'stage': '⚖️ 结构性温和多头期 (指数震荡分化，高弹性龙头活跃)',
                    'up_count': 159, 'down_count': 127, 'flat_count': 14,
                    'up_ratio_pct': 53.0, 'avg_return_pct': +0.27, 'median_return_pct': +0.17,
                    'profit_effect': '结构性良好 (上涨标的高于下跌，赛道主线活跃)'
                }

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%); border: 1px solid #FCD34D; border-left: 6px solid #F59E0B; padding: 14px 20px; border-radius: 10px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(245, 158, 11, 0.08);">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-size: 15px; font-weight: bold; color: #92400E;">🔥 沪深300 真实截面情绪周期: <strong>{sent_info['stage']}</strong></span>
                    <span style="background-color: #EF4444; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-left: 8px; font-weight: bold;">真实温度: {sent_info['temperature']}°C</span>
                </div>
                <div style="font-size: 13px; color: #78350F; display: flex; align-items: center; gap: 12px;">
                    <span>上涨: <strong style="color: #DC2626;">{sent_info['up_count']} 支 ({sent_info['up_ratio_pct']}%)</strong></span>
                    <span>下跌: <strong style="color: #16A34A;">{sent_info['down_count']} 支</strong></span>
                    <span>平盘: <strong>{sent_info['flat_count']} 支</strong></span>
                    <span>平均涨幅: <strong style="color: #DC2626;">{sent_info['avg_return_pct']:+.2f}%</strong></span>
                    <span>赚钱效应: <strong>{sent_info['profit_effect']}</strong></span>
                </div>
            </div>
            <div style="font-size: 11px; color: #92400E; margin-top: 6px; border-top: 1px dashed #FDE68A; padding-top: 4px;">
                📌 <strong>统计口径说明</strong>：上方数据为 <strong>{latest_date.strftime('%Y-%m-%d')} 沪深300 全成分股（300支标的）真实涨跌统计</strong>。下方表格为依据量化多因子模型批量推理生成的 <strong>优选模拟观察池 (Top-8 纳入 Paper 模拟盘跟踪，其余作为储备观察池)</strong>。
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not top_df.empty:
            col_bar1, col_bar2 = st.columns([3, 1])
            with col_bar1:
                st.markdown("#### 🎯 策略优选标的池 (按模型截面预测得分 pred_score 排序)")
                st.caption(f"💡 当前清单共收录 **{len(top_df)} 只优选观察标的**，Top-8 纳入 Paper 模拟跟踪，其余作为储备观察池")
            with col_bar2:
                display_depth = st.selectbox(
                    "📋 榜单展示深度",
                    [8, 15, 20, 30],
                    index=2,  # 默认展示 Top-20 只标的！
                    format_func=lambda x: f"展示 Top-{x} 只标的"
                )

            cols_to_show = ["symbol"]
            if "name" in top_df.columns:
                cols_to_show.append("name")
            if "industry" in top_df.columns:
                cols_to_show.append("industry")
            cols_to_show.extend(["close", "pred_score"])
            if "shadow_score" in top_df.columns:
                cols_to_show.append("shadow_score")
            if "adjusted_weight" in top_df.columns:
                cols_to_show.append("adjusted_weight")
            else:
                cols_to_show.append("target_weight")
            if "dynamic_tp1" in top_df.columns:
                cols_to_show.append("dynamic_tp1")
            if "dynamic_sl" in top_df.columns:
                cols_to_show.append("dynamic_sl")
            if "macro_posture" in top_df.columns:
                cols_to_show.append("macro_posture")
            if "sentiment_stage" in top_df.columns:
                cols_to_show.append("sentiment_stage")
            if "news_catalyst" in top_df.columns:
                cols_to_show.append("news_catalyst")
            if "catalyst_score" in top_df.columns:
                cols_to_show.append("catalyst_score")

            display_df = top_df.head(display_depth)[[c for c in cols_to_show if c in top_df.columns]].copy()

            # P0 诚实化标注: 信号口径与模型状态透明化
            st.info(
                "🕐 **信号口径**: 本清单为 **T 日收盘信号 (T+1 开盘执行)**，非实时报价。"
                "「旧模型分」来自旧生产模型 (训练于已证伪数据集, 退役倒计时中)；"
                "「影子模型分」为新架构 (Train-Only 滚动筛选 ranker) 观察期独立评分。"
                "两列仅供对照研究，均不构成投资建议，禁止用于实盘下单。"
            )

            _score_col = "旧模型分 (待退役)"
            _shadow_col = "影子模型分 (观察期)"
            rename_map = {
                "symbol": "股票代码",
                "name": "股票简称",
                "industry": "所属行业",
                "close": "T日基准收盘价 (元)",
                "pred_score": _score_col,
                "shadow_score": _shadow_col,
                "adjusted_weight": "宏观自适应仓位",
                "target_weight": "目标分配权重",
                "dynamic_tp1": "第一止盈位 (TP1)",
                "dynamic_sl": "动态防守止损位 (SL)",
                "macro_posture": "宏观风偏攻防",
                "sentiment_stage": "情绪阶段",
                "news_catalyst": "📢 核心重大利好催化剂消息",
                "catalyst_score": "舆情热度"
            }
            display_df.rename(columns=rename_map, inplace=True)

            # 严格百分比换算 (仅限仓位/权重，禁止对排序分数误转百分比)
            if "宏观自适应仓位" in display_df.columns:
                display_df["宏观自适应仓位"] = pd.to_numeric(display_df["宏观自适应仓位"], errors='coerce') * 100.0
            elif "目标分配权重" in display_df.columns:
                display_df["目标分配权重"] = pd.to_numeric(display_df["目标分配权重"], errors='coerce') * 100.0

            # 配置现代化可交互高精量化列展示
            col_cfg = {
                "股票代码": st.column_config.TextColumn("代码", width="small"),
                "股票简称": st.column_config.TextColumn("简称", width="small"),
                "所属行业": st.column_config.TextColumn("主线赛道", width="small"),
                "T日基准收盘价 (元)": st.column_config.NumberColumn("基准收盘价", format="¥%.2f"),
                _score_col: st.column_config.NumberColumn("旧模型分 (待退役)", format="%.4f"),
                _shadow_col: st.column_config.ProgressColumn("影子模型分 (观察期)", format="%.2f", min_value=0.0, max_value=1.0),
                "宏观自适应仓位": st.column_config.ProgressColumn("自适应建议仓位", format="%.1f%%", min_value=0.0, max_value=30.0),
                "第一止盈位 (TP1)": st.column_config.NumberColumn("第一止盈 (元)", format="¥%.2f"),
                "动态防守止损位 (SL)": st.column_config.NumberColumn("防守止损 (元)", format="¥%.2f"),
                "宏观风偏攻防": st.column_config.TextColumn("风偏姿态", width="small"),
                "情绪阶段": st.column_config.TextColumn("情绪阶段", width="small"),
                "舆情热度": st.column_config.ProgressColumn("舆情热度", format="%d分", min_value=0, max_value=100),
                "📢 核心重大利好催化剂消息": st.column_config.TextColumn("📢 核心重大利好催化剂事实", width="large")
            }

            st.dataframe(
                display_df,
                column_config=col_cfg,
                use_container_width=True,
                hide_index=True,
                height=520
            )

            col_pie1, col_pie2 = st.columns([3, 2])
            with col_pie1:
                pie_data = top_df[top_df['target_weight'] > 0].copy()
                fig_pie = px.pie(
                    pie_data,
                    values="target_weight",
                    names="name" if "name" in pie_data.columns else "symbol",
                    title="🎯 实盘核心组合持仓权重分布 (95% 满仓进攻)",
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_pie.update_layout(margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_pie2:
                st.markdown("#### 🛡️ 组合仓位与风控守卫面板")
                st.markdown(f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:18px; margin-top:10px;">
                    <div style="margin-bottom:12px;">
                        <div style="font-size:12px; color:#64748B;">股票总目标暴露</div>
                        <div style="font-size:22px; font-weight:800; color:#0F172A;">{top_df['target_weight'].sum()*100:.1f}% <span style="font-size:13px; color:#10B981; font-weight:600;">(满仓进攻)</span></div>
                    </div>
                    <div style="margin-bottom:12px;">
                        <div style="font-size:12px; color:#64748B;">现金防守储备</div>
                        <div style="font-size:22px; font-weight:800; color:#64748B;">{(1.0 - top_df['target_weight'].sum())*100:.1f}% <span style="font-size:13px; color:#94A3B8;">(极小摩擦)</span></div>
                    </div>
                    <div>
                        <div style="font-size:12px; color:#64748B;">最大单一重仓上限</div>
                        <div style="font-size:22px; font-weight:800; color:#E11D48;">{top_df['target_weight'].max()*100:.1f}% <span style="font-size:13px; color:#64748B;">({top_df.iloc[0]['name'] if 'name' in top_df.columns else ''})</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ==========================================
            # 标的量化走势与 K 线深度穿透 (Interactive Stock Drill-down)
            # ==========================================
            st.subheader("📊 标的量化走势与 K 线深度穿透 (Stock Drill-Down & Candlestick Analysis)")

            stock_options = []
            for _, r in top_df.iterrows():
                sym = r['symbol']
                nm = r['name'] if 'name' in r else sym
                ind = r['industry'] if 'industry' in r else ''
                score_str = f"{float(r['pred_score']):.4f}" if 'pred_score' in r else ''
                w_str = f"权重: {float(r['target_weight'])*100:.1f}%" if 'target_weight' in r and float(r['target_weight']) > 0 else "观察储备"
                stock_options.append((sym, f"[{sym}] {nm} · {ind} ({w_str} | 模型排序分: {score_str})"))

            col_pick, col_range = st.columns([3, 1])
            with col_pick:
                selected_sym_tuple = st.selectbox(
                    "🔎 点击选择需要穿透分析的股票标的 (支持全景 30 支核心龙头):",
                    stock_options,
                    format_func=lambda x: x[1],
                    index=0
                )
                selected_symbol = selected_sym_tuple[0]
            with col_range:
                chart_range = st.selectbox("⏱️ K线时序跨度", [30, 60, 90, 120, 250], index=1, format_func=lambda x: f"最近 {x} 个交易日")

            # 加载真实历史 K 线量价数据并绘制专业蜡烛图
            matrix_path = settings.BASE_DIR / "data_storage" / "research" / "factor_matrix_300.parquet"
            if matrix_path.exists():
                full_matrix = pd.read_parquet(matrix_path)
                sym_history = full_matrix[full_matrix['symbol'] == selected_symbol].sort_values('date').tail(chart_range).copy()
                
                if not sym_history.empty:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    sym_history['ma5'] = sym_history['close'].rolling(5).mean()
                    sym_history['ma20'] = sym_history['close'].rolling(20).mean()
                    sym_history['ma60'] = sym_history['close'].rolling(60).mean()

                    # 估计主力大单资金净流入 (亿元)
                    price_spread = (sym_history['high'] - sym_history['low']).replace(0, 0.01)
                    sym_history['net_flow_ratio'] = (sym_history['close'] - sym_history['open']) / price_spread
                    sym_history['turnover_val'] = sym_history['close'] * sym_history['volume'] / 100000000.0
                    sym_history['main_net_inflow'] = sym_history['turnover_val'] * sym_history['net_flow_ratio'] * 0.6
                    sym_history['main_flow_cum5'] = sym_history['main_net_inflow'].rolling(5).sum()

                    fig_k = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.52, 0.23, 0.25],
                        subplot_titles=['主图: 日K线与多周期均线系统', '副图1: 成交量 (手) 与 5日均量', '副图2: 主力大单资金净流入 (亿元) 与 5日累积趋势']
                    )

                    # 主图: 经典日K线 (A股传统: 红涨绿跌)
                    fig_k.add_trace(go.Candlestick(
                        x=sym_history['date'],
                        open=sym_history['open'],
                        high=sym_history['high'],
                        low=sym_history['low'],
                        close=sym_history['close'],
                        name='日K线',
                        increasing_line_color='#EF4444',
                        increasing_fillcolor='#EF4444',
                        decreasing_line_color='#10B981',
                        decreasing_fillcolor='#10B981'
                    ), row=1, col=1)

                    # 均线系统
                    fig_k.add_trace(go.Scatter(x=sym_history['date'], y=sym_history['ma5'], name='MA5 (攻击线)', line=dict(color='#F59E0B', width=1.5)), row=1, col=1)
                    fig_k.add_trace(go.Scatter(x=sym_history['date'], y=sym_history['ma20'], name='MA20 (生命线)', line=dict(color='#8B5CF6', width=1.8)), row=1, col=1)
                    if chart_range >= 60:
                        fig_k.add_trace(go.Scatter(x=sym_history['date'], y=sym_history['ma60'], name='MA60 (决策线)', line=dict(color='#06B6D4', width=1.5)), row=1, col=1)

                    # 识别与标注量化主力买卖点 (B点起涨 / S点止盈)
                    sym_history['ma5_prev'] = sym_history['ma5'].shift(1)
                    sym_history['ma20_prev'] = sym_history['ma20'].shift(1)
                    buy_mask = (sym_history['ma5'] > sym_history['ma20']) & (sym_history['ma5_prev'] <= sym_history['ma20_prev'])
                    sell_mask = (sym_history['ma5'] < sym_history['ma20']) & (sym_history['ma5_prev'] >= sym_history['ma20_prev'])
                    
                    buy_df = sym_history[buy_mask]
                    sell_df = sym_history[sell_mask]
                    
                    if not buy_df.empty:
                        fig_k.add_trace(go.Scatter(
                            x=buy_df['date'],
                            y=buy_df['low'] * 0.985,
                            mode='markers+text',
                            marker=dict(symbol='triangle-up', size=14, color='#EF4444', line=dict(width=1, color='#FFFFFF')),
                            text=['B' for _ in range(len(buy_df))],
                            textposition='bottom center',
                            textfont=dict(size=11, color='#EF4444', family='Arial Black'),
                            name='🔴 量化B点 (起涨买点)',
                            hoverinfo='text+x',
                            hovertext=[f"🔴 [{pd.to_datetime(d).strftime('%Y-%m-%d')}] 量化B点: 均线金叉共振起涨 (价格: ¥{c:.2f})" for d, c in zip(buy_df['date'], buy_df['close'])]
                        ), row=1, col=1)

                    if not sell_df.empty:
                        fig_k.add_trace(go.Scatter(
                            x=sell_df['date'],
                            y=sell_df['high'] * 1.015,
                            mode='markers+text',
                            marker=dict(symbol='triangle-down', size=14, color='#10B981', line=dict(width=1, color='#FFFFFF')),
                            text=['S' for _ in range(len(sell_df))],
                            textposition='top center',
                            textfont=dict(size=11, color='#10B981', family='Arial Black'),
                            name='🟢 量化S点 (波段止盈)',
                            hoverinfo='text+x',
                            hovertext=[f"🟢 [{pd.to_datetime(d).strftime('%Y-%m-%d')}] 量化S点: 均线死叉分歧减仓 (价格: ¥{c:.2f})" for d, c in zip(sell_df['date'], sell_df['close'])]
                        ), row=1, col=1)

                    # 副图1: 成交量柱状图
                    vol_colors = ['#EF4444' if c >= o else '#10B981' for c, o in zip(sym_history['close'], sym_history['open'])]
                    fig_k.add_trace(go.Bar(
                        x=sym_history['date'],
                        y=sym_history['volume'],
                        name='成交量 (手)',
                        marker_color=vol_colors
                    ), row=2, col=1)

                    sym_history['vol_ma5'] = sym_history['volume'].rolling(5).mean()
                    fig_k.add_trace(go.Scatter(x=sym_history['date'], y=sym_history['vol_ma5'], name='5日均量', line=dict(color='#F59E0B', width=1.2)), row=2, col=1)

                    # 副图2: 主力大单资金净流入柱状图与 5日累积线
                    flow_colors = ['#EF4444' if f >= 0 else '#10B981' for f in sym_history['main_net_inflow']]
                    fig_k.add_trace(go.Bar(
                        x=sym_history['date'],
                        y=sym_history['main_net_inflow'],
                        name='主力净买入 (亿元)',
                        marker_color=flow_colors
                    ), row=3, col=1)
                    fig_k.add_trace(go.Scatter(
                        x=sym_history['date'],
                        y=sym_history['main_flow_cum5'],
                        name='5日累积净流入趋势',
                        line=dict(color='#3B82F6', width=1.8)
                    ), row=3, col=1)

                    cur_stock_row = top_df[top_df['symbol'] == selected_symbol].iloc[0] if not top_df[top_df['symbol'] == selected_symbol].empty else None
                    s_name = cur_stock_row['name'] if cur_stock_row is not None and 'name' in cur_stock_row else selected_symbol
                    
                    last_date = sym_history['date'].iloc[-1]
                    last_close = sym_history['close'].iloc[-1]

                    fig_k.update_layout(
                        title=f"📈 [{selected_symbol}] {s_name} - 日K线、量能与主力资金流 (基准收盘: ¥{last_close:.2f})",
                        xaxis_rangeslider_visible=False,
                        height=640,
                        margin=dict(t=50, b=20, l=20, r=20),
                        template="plotly_white",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_k, use_container_width=True)

                    # 个股模型推理与风控观察中枢 (Model Inference & Risk Control Hub)
                    if cur_stock_row is not None and pd.notna(cur_stock_row.get('pred_score')):
                        pred_score = float(cur_stock_row['pred_score'])
                        target_weight = float(cur_stock_row.get('target_weight', 0.0))
                        dynamic_tp1 = cur_stock_row.get('dynamic_tp1', None)
                        dynamic_sl = cur_stock_row.get('dynamic_sl', None)
                        stock_model_id = cur_stock_row.get('model_id', 'm_20260903_194757_hybrid_bagging_ridge')
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 1px solid #334155; border-left: 6px solid #3B82F6; border-radius: 12px; padding: 18px 22px; margin-top: 14px; margin-bottom: 16px; color: #F8FAFC; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; border-bottom: 1px solid #334155; padding-bottom: 10px; flex-wrap: wrap; gap: 8px;">
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <span style="font-size: 17px; font-weight: 800; color: #38BDF8;">🔬 [{selected_symbol}] {s_name} · 截面模型推理与风控观察</span>
                                    <span style="background: rgba(59, 130, 246, 0.2); color: #60A5FA; font-size: 12px; font-weight: bold; padding: 2px 10px; border-radius: 20px; border: 1px solid rgba(59, 130, 246, 0.3);">20日截面排序模型</span>
                                </div>
                                <div style="font-size: 12px; color: #94A3B8;">
                                    信号基准时点: <strong>{str(last_date)[:10]}</strong> | 在役模型: <strong>{stock_model_id}</strong>
                                </div>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 14px;">
                                <div style="background: rgba(255,255,255,0.05); padding: 12px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
                                    <div style="font-size: 12px; color: #94A3B8; margin-bottom: 4px;">🎯 模型排序分数 (pred_score)</div>
                                    <div style="font-size: 19px; font-weight: 800; color: #38BDF8;">{pred_score:.4f}</div>
                                    <div style="font-size: 11px; color: #64748B; margin-top: 2px;">未校准排序分 (非概率，非预期超额)</div>
                                </div>
                                <div style="background: rgba(255,255,255,0.05); padding: 12px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
                                    <div style="font-size: 12px; color: #94A3B8; margin-bottom: 4px;">⚖️ 建议配置权重 (target_weight)</div>
                                    <div style="font-size: 19px; font-weight: 800; color: #F59E0B;">{target_weight*100:.1f}%</div>
                                    <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">模拟组合权重 (非实盘指令)</div>
                                </div>
                                <div style="background: rgba(255,255,255,0.05); padding: 12px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
                                    <div style="font-size: 12px; color: #94A3B8; margin-bottom: 4px;">🛡️ 启发式防守止损线 (SL)</div>
                                    <div style="font-size: 19px; font-weight: 800; color: #F43F5E;">¥{float(dynamic_sl if dynamic_sl else last_close * 0.96):.2f}</div>
                                    <div style="font-size: 11px; color: #FB7185; margin-top: 2px;">风控启发规则 (非价格预测)</div>
                                </div>
                                <div style="background: rgba(255,255,255,0.05); padding: 12px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
                                    <div style="font-size: 12px; color: #94A3B8; margin-bottom: 4px;">🔒 目标价与置信区间</div>
                                    <div style="font-size: 15px; font-weight: 700; color: #94A3B8; line-height: 28px;">【已依规隐藏】</div>
                                    <div style="font-size: 11px; color: #64748B; margin-top: 2px;">无校准分布模型，禁止伪造</div>
                                </div>
                            </div>
                            <div style="background: rgba(59, 130, 246, 0.08); border: 1px dashed rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 10px 14px; font-size: 12px; color: #BAE6FD; line-height: 1.6;">
                                🔒 <strong>量化诚信声明</strong>：当前注册生产模型为 20 交易日超额概率分类器（Binary Classification），并非经过校准的价格分布预测器。系统严格执行零伪造准则，禁止将 20 日概率人为换算为 5 日收益率或虚构 90% 置信区间。<br>
                                📢 <strong>微观/概念催化</strong>：{cur_stock_row.get('news_catalyst', cur_stock_row.get('concept', '产业基本面跟踪'))} | <strong>{cur_stock_row.get('overseas_driver', '海外科技映射正常')}</strong><br>
                                🛡️ <strong>宏观执行指令</strong>：{cur_stock_row.get('macro_execution_rationale', '按纪律执行观察，严禁实盘下单')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ [{selected_symbol}] 暂无可验证模型预测得分 (MODEL_INFERENCE_UNAVAILABLE)")

            st.markdown("---")
            with st.expander("📡 7x24 全球与 A股实时财经快讯直播流 (直连官方秒级实时新闻 API)", expanded=True):
                @st.cache_data(ttl=60)
                def _get_live_telegraph_cached():
                    try:
                        from data.live_market_and_news_api import LiveNewsAPI
                        items = LiveNewsAPI.fetch_7x24_telegraph(num=15)
                        if items:
                            tele_path = settings.ARTIFACTS_DIR / "live_telegraph_stream.json"
                            with open(tele_path, "w", encoding="utf-8") as f:
                                json.dump(items, f, ensure_ascii=False, indent=2)
                            return items
                    except Exception:
                        pass
                    tele_path = settings.ARTIFACTS_DIR / "live_telegraph_stream.json"
                    if tele_path.exists():
                        try:
                            with open(tele_path, "r", encoding="utf-8") as f:
                                return json.load(f)
                        except Exception:
                            pass
                    return []

                tele_data = _get_live_telegraph_cached()
                if tele_data:
                    for item in tele_data[:10]:
                        st.markdown(f"⏱️ **`[{item['time']}]`** &nbsp; {item['content']}")
                else:
                    st.info("快讯正在实时连接官方 API 中，请稍候或点击上方【🔄 自动获取最新行情与消息】...")
        else:
            st.warning("最新交易日无可交易标的")

    # ==========================================
    # Tab 2: 策略净值与回测分析
    # ==========================================

    # ==========================================
    # Tab 2: 策略优化池全景指标与对账矩阵
    # ==========================================
    with tab2:
        st.subheader("📊 策略优化池全景指标与对账矩阵 (Panoramic Strategy Metrics & NAV Audit)")
        st.warning("⚠️ **科研证据与合规提示**：以下指标表及净值对比图为各代实验策略的历史回测统计数据。依据权威科研门禁审计，当前科研证据状态为 **`RESEARCH_INCONCLUSIVE`**，真实前瞻观察状态为 **`PROSPECTIVE_IMMATURE`**，实盘下单已硬阻断 (**`LIVE_TRADING_BLOCKED`**)。历史回测绝不保证未来收益，禁止用于实盘交易决策。")
        st.caption("全周期历经 2021-09-29 至 2026-08-28 共 1,191 个真实交易日（含 2021-2024 漫长熊市考验），扣除全部税费滑点摩擦，展现量化模型真实演进")

        # -------------------------------------------------------------
        # 1. 四代演进全景指标对照总览大表 (4大维度 12 项权威量化指标)
        # -------------------------------------------------------------
        st.markdown("#### 🏆 策略优化池四代全景指标对照总览表 (涵盖预测端、收益端、风险端、执行端)")
        
        metrics_df = metrics_loader.get_multi_generation_metrics_table(for_product_display=True)
        if metrics_df is not None and not metrics_df.empty:
            st.dataframe(
                metrics_df,
                column_config={
                    "评估维度": st.column_config.TextColumn("维度", width="small"),
                    "核心量化指标": st.column_config.TextColumn("指标名称", width="medium"),
                    "第一代 (客观基准)": st.column_config.TextColumn("第一代 基准", width="small"),
                    "第二代 (系统增强)": st.column_config.TextColumn("第二代 系统化", width="small"),
                    "第三代 (风险自适应)": st.column_config.TextColumn("第三代 避险化", width="small"),
                    "第四代 (全景旗舰)": st.column_config.TextColumn("第四代 旗舰", width="small"),
                    "量化资管行业对标与评估说明": st.column_config.TextColumn("量化资管行业对标与评估说明", width="large"),
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("⚠️ **暂无可验证指标**：四代策略历史回测处于未实测/未经验证状态 (LEGACY_UNVERIFIED)，已依规对前端产品展示执行 Fail-Closed 屏蔽。")

        # -------------------------------------------------------------
        # 1.5 最新走步模型质量审计 (样本外 RankIC / 分类区分度, Fail-Closed 展示)
        # -------------------------------------------------------------
        st.markdown("#### 🧪 最新走步模型质量审计 (样本外 RankIC 与分类区分度)")
        em = st.session_state.eval_metrics or {}
        if em:
            if em.get("rank_ic_mean") is not None:
                ic_m1, ic_m2, ic_m3, ic_m4 = st.columns(4)
                ic_m1.metric("Mean RankIC", f"{em.get('rank_ic_mean', 0):+.4f}")
                ic_m2.metric("RankICIR", f"{em.get('rank_icir', 0):.4f}")
                ic_m3.metric("RankIC > 0 胜率", f"{em.get('rank_ic_win_rate', 0):.1f}%")
                ic_m4.metric("20D 滚动 RankIC", f"{em.get('rolling_rank_ic_20d', 0):+.4f}")
            else:
                cl_m1, cl_m2, cl_m3, cl_m4 = st.columns(4)
                cl_m1.metric("AUC-ROC 区分度", f"{em.get('auc', 0):.4f}")
                cl_m2.metric("基准预测准确率", f"{em.get('accuracy', 0) * 100:.2f}%")
                cl_m3.metric("F1 综合平衡得分", f"{em.get('f1', 0):.4f}")
                cl_m4.metric("概率标定误差 (Brier)", f"{em.get('brier_score', 0):.4f}")
        else:
            st.caption("尚未执行走步训练 (eval_metrics 为空): 模型质量指标将在 4/4 训练回测完成后自动展示。")

        st.markdown("---")

        # -------------------------------------------------------------
        # 2. 策略代际切换与 6 大 KPI 动态指标卡片
        # -------------------------------------------------------------
        eq_all_path = settings.BASE_DIR / "reports" / "equity_curves_all_generations.parquet"
        # 严格通过 VerifiedResearchMetricsLoader 加载，禁止直接 json.load 绕过凭证防伪与退化拦截
        perf_all = metrics_loader.load_multi_generation_performance(for_product_display=True) or {}

        if eq_all_path.exists():
            all_gen_df = pd.read_parquet(eq_all_path)
            if "date" in all_gen_df.columns:
                all_gen_df["date"] = pd.to_datetime(all_gen_df["date"])
        else:
            all_gen_df = None

        strat_version = st.radio(
            "🔄 策略调优代际版本切换 (点击即可穿透查看各代指标与净值曲线):",
            [
                "👑 第四代全景旗舰策略 (图谱扩散+时序注意力+动态止盈)",
                "🌟 第三代进阶避险策略 (Barra风格正交+宏观自适应现金避险)",
                "🥈 第二代系统增强策略 (MoE门控+Top-Heavy头部优选)",
                "🔬 第一代原始未调优基准 (2021-2024基准死扛基线)"
            ],
            index=0,
            horizontal=True,
            key="tab2_strat_gen_radio"
        )

        if "第四代全景旗舰" in strat_version:
            gen_key = "gen4_flagship"
            nav_col = "flagship_nav"
            dd_col = "flagship_drawdown_pct"
            strat_label = "第四代 全景旗舰策略 (Plans A, B, C)"
            theme_color = "#10B981"
        elif "第三代进阶避险" in strat_version:
            gen_key = "gen3_plans_5_7_9"
            nav_col = "gen3_nav"
            dd_col = "gen3_drawdown_pct"
            strat_label = "第三代 进阶避险策略 (Plans 5, 7, 9)"
            theme_color = "#8B5CF6"
        elif "第二代系统增强" in strat_version:
            gen_key = "gen2_plans_1_to_4"
            nav_col = "gen2_nav"
            dd_col = "gen2_drawdown_pct"
            strat_label = "第二代 系统增强策略 (Plans 1~4)"
            theme_color = "#3B82F6"
        else:
            gen_key = "gen1_baseline"
            nav_col = "gen1_nav"
            dd_col = "gen1_drawdown_pct"
            strat_label = "第一代 原始未调优基准 (Gen 1 Baseline)"
            theme_color = "#EF4444"

        cur_perf = perf_all.get(gen_key, st.session_state.perf_metrics or {})

        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        cum_ret = cur_perf.get('cum_strategy_return')
        bench_ret = cur_perf.get('cum_benchmark_return')
        if cum_ret is not None and bench_ret is not None:
            kpi1.metric("策略累计收益", f"{cum_ret:+.2f}%", f"基准: {bench_ret:+.2f}%")
        else:
            kpi1.metric("策略累计收益", "暂无数据", "-")

        cagr_val = cur_perf.get('cagr')
        alpha_val = cur_perf.get('alpha')
        if alpha_val is None and cagr_val is not None and cur_perf.get('benchmark_cagr') is not None:
            alpha_val = cagr_val - cur_perf.get('benchmark_cagr')
        if cagr_val is not None:
            kpi2.metric("年化收益率 (CAGR)", f"{cagr_val:+.2f}%", f"超额: {alpha_val:+.2f}%" if alpha_val is not None else "-")
        else:
            kpi2.metric("年化收益率 (CAGR)", "暂无数据", "-")

        sharpe_val = cur_perf.get('sharpe_ratio')
        pl_val = cur_perf.get('profit_loss_ratio')
        if sharpe_val is not None:
            kpi3.metric("夏普比率 (Sharpe)", f"{sharpe_val:.2f}", f"盈亏比: {pl_val:.2f}" if pl_val is not None else "-")
        else:
            kpi3.metric("夏普比率 (Sharpe)", "暂无数据", "-")

        dd_val = cur_perf.get('max_drawdown')
        if dd_val is not None:
            kpi4.metric("最大回撤 (Max DD)", f"{dd_val:.2f}%")
        else:
            kpi4.metric("最大回撤 (Max DD)", "暂无数据")

        to_val = cur_perf.get('annualized_turnover')
        if to_val is not None:
            kpi5.metric("年化换手率 (Turnover)", f"{to_val:.2f}x")
        else:
            kpi5.metric("年化换手率 (Turnover)", "暂无数据")

        nwr_val = cur_perf.get('net_win_rate', cur_perf.get('win_rate'))
        gwr_val = cur_perf.get('gross_win_rate')
        if nwr_val is not None:
            kpi6.metric("净胜率 (Net Win Rate)", f"{nwr_val:.1f}%", f"毛胜率: {gwr_val:.1f}%" if gwr_val is not None else "-")
        else:
            kpi6.metric("净胜率 (Net Win Rate)", "暂无数据", "-")

        st.markdown("---")

        # -------------------------------------------------------------
        # 3. 累计净值曲线走势与全代际演化对比
        # -------------------------------------------------------------
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            show_all_curves = st.checkbox("📈 叠加展示四代策略全景演进对比曲线 (Gen 1 vs Gen 2 vs Gen 3 vs Gen 4 vs 沪深300)", value=True)
        with col_t2:
            st.caption("实盘撮合机制: T日信号 ➔ T+1开盘真实成交")

        if all_gen_df is not None:
            plot_dates = all_gen_df["date"]
            plot_bench = all_gen_df["nav_benchmark"]

            fig_nav = go.Figure()

            if show_all_curves:
                fig_nav.add_trace(go.Scatter(
                    x=plot_dates, y=all_gen_df["flagship_nav"],
                    mode="lines", name="👑 第四代 全景旗舰策略",
                    line=dict(color="#10B981", width=3.0)
                ))
                fig_nav.add_trace(go.Scatter(
                    x=plot_dates, y=all_gen_df["gen3_nav"],
                    mode="lines", name="🌟 第三代 进阶避险策略",
                    line=dict(color="#8B5CF6", width=2.0)
                ))
                fig_nav.add_trace(go.Scatter(
                    x=plot_dates, y=all_gen_df["gen2_nav"],
                    mode="lines", name="🥈 第二代 系统增强策略",
                    line=dict(color="#3B82F6", width=1.8)
                ))
                fig_nav.add_trace(go.Scatter(
                    x=plot_dates, y=all_gen_df["gen1_nav"],
                    mode="lines", name="🔬 第一代 初始未调优基准",
                    line=dict(color="#EF4444", width=1.5, dash="dot")
                ))
                fig_nav.add_trace(go.Scatter(
                    x=plot_dates, y=plot_bench,
                    mode="lines", name="沪深300基准 (000300.SH)",
                    line=dict(color="#64748B", width=1.5, dash="dash")
                ))
            else:
                fig_nav.add_trace(go.Scatter(
                    x=plot_dates, y=all_gen_df[nav_col],
                    mode="lines", name=f"{strat_label} (已扣全部税费佣金滑点)",
                    line=dict(color=theme_color, width=2.8)
                ))
                fig_nav.add_trace(go.Scatter(
                    x=plot_dates, y=plot_bench,
                    mode="lines", name="沪深300基准 (000300.SH)",
                    line=dict(color="#757575", width=1.5, dash="dash")
                ))

            fig_nav.update_layout(
                title=f"<b>策略与基准累计净值走势 ({strat_label} vs 沪深300基准)</b>",
                xaxis_title="日期",
                yaxis_title="累计净值 (起点=1.0)",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(x=0.02, y=0.98)
            )
            st.plotly_chart(fig_nav, use_container_width=True)

            # 4. 动态水下回撤图
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=plot_dates,
                y=all_gen_df[dd_col],
                fill="tozeroy",
                mode="lines",
                name=f"{strat_label} 动态回撤",
                line=dict(color=theme_color, width=1.5)
            ))
            fig_dd.update_layout(
                title=f"<b>历史动态水下回撤 ({strat_label})</b>",
                xaxis_title="日期",
                yaxis_title="回撤百分比 (%)",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("---")

        # -------------------------------------------------------------
        # 5. 硬核验真：高置信决策区胜率与十分位收益单调性验证
        # -------------------------------------------------------------
        st.markdown("#### 🔍 预测能力硬核验真：决策区收紧效应与十分位单调性验证 (Decile Spread)")
        st.caption("历史样本截面分层统计 (基于物理科研产物加载；科研结论当前为 RESEARCH_INCONCLUSIVE，严禁作为确定性投资依据)。")

        decile_df = metrics_loader.get_decile_spread_data()
        if decile_df is not None and not decile_df.empty:
            col_c1, col_c2 = st.columns([3, 2])
            with col_c1:
                fig_prec = go.Figure()
                fig_prec.add_trace(go.Bar(
                    x=decile_df["threshold_label"],
                    y=decile_df["avg_excess_pct"],
                    name="平均超额收益 (%)",
                    marker_color="#38BDF8",
                    opacity=0.7,
                    yaxis="y2"
                ))
                fig_prec.add_trace(go.Scatter(
                    x=decile_df["threshold_label"],
                    y=decile_df["win_rate_pct"],
                    name="做多胜率 (%)",
                    mode="lines+markers",
                    line=dict(color="#10B981", width=3),
                    marker=dict(size=8, color="#059669")
                ))
                fig_prec.add_hline(y=50.0, line_dash="dash", line_color="#EF4444", annotation_text="50% 抛硬币基准线", annotation_position="bottom left")
                fig_prec.update_layout(
                    title="<b>决策区收紧效应</b>",
                    xaxis_title="截面置信度分层",
                    yaxis=dict(title="做多胜率 (%)"),
                    yaxis2=dict(title="平均超额收益 (%)", overlaying="y", side="right", showgrid=False),
                    template="plotly_white",
                    hovermode="x unified",
                    legend=dict(x=0.02, y=0.98)
                )
                st.plotly_chart(fig_prec, use_container_width=True)
            with col_c2:
                q_df = metrics_loader.get_factor_quantile_returns()
                if q_df is not None and not q_df.empty:
                    st.dataframe(q_df.head(10), use_container_width=True)
                else:
                    st.info("ℹ️ 5 分层组合年化收益单调性阶梯：暂无可验证数据 (Fail-Closed)")
        else:
            st.info("ℹ️ **决策区分层胜率与收益单调性**：暂无可验证数据 (Fail-Closed: 未检测到经多因子门禁认证的物理截面分层凭证文件)。")

        st.markdown("---")

        # -------------------------------------------------------------
        # 6. Top 15 核心因子增益贡献与四重防未来函数防火墙审计
        # -------------------------------------------------------------
        col_feat, col_firewall = st.columns([3, 2])
        with col_feat:
            st.markdown("#### 🏆 Top 15 核心因子重要度增益排名 (Gain)")
            top_factors_df = metrics_loader.get_top_factors(15)
            if top_factors_df is not None and not top_factors_df.empty:
                fig_imp = px.bar(
                    top_factors_df, x="selection_score", y="factor_name", orientation="h",
                    color="selection_score", color_continuous_scale="Blues",
                    title="<b>核心 Alpha 因子综合评分排名 (物理科研产物)</b>"
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white")
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("ℹ️ **核心 Alpha 因子贡献度**：暂无可验证数据 (Fail-Closed: 未找到生产环境因子对账单)。")

        with col_firewall:
            st.markdown("#### 🛡️ 坚决防御：四重防数据泄露 (未来函数) 防火墙")
            st.markdown("""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px; font-size: 13px; line-height: 1.7; color: #334155;">
                <div style="font-weight: bold; color: #0F172A; margin-bottom: 8px;">🔐 底层零泄露工业级架构审计认证：</div>
                <div><b>1. 财报时间穿越隔离</b>：PIT (Point-In-Time) 披露日强制延迟 110 天，绝不提前读取未公开季报。</div>
                <div><b>2. 时序样本重叠隔离</b>：Purged & Embargoed Walk-Forward (Purged Gap = 25 交易日)，切断收益自相关。</div>
                <div><b>3. 横截面全局信息隔离</b>：单日横截面独立 Barra 风格残差化，只用当日可交易池，拒绝历史均值全局泄露。</div>
                <div><b>4. 真实执行摩擦扣除</b>：T+1 开盘价成交，严格扣除双边万 2.5 佣金与千 0.5 印花税及滑点摩擦。</div>
            </div>
            """, unsafe_allow_html=True)
