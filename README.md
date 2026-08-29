# A股多因子机器学习量化交易与实盘决策闭环系统 (Enterprise A-Share AI Quant System)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 135+ Passed](https://img.shields.io/badge/Tests-135%2B%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![Data Integrity: Zero Leakage](https://img.shields.io/badge/Data%20Integrity-Zero%20Leakage%20%7C%20Point--in--Time-success.svg)](data/)

企业级端到端 A 股多因子 AI 量化投研与实盘决策系统，构建了从 **「多源数据工程 $\to$ 107 因子特征工厂 $\to$ 多模型 Walk-Forward 时序滚动训练 $\to$ 凸优化组合决策 $\to$ 7 重机构级资金安全防御 $\to$ 券商实盘/模拟撮合网关 $\to$ 盘后定时自动化推送 $\to$ 7 大主题 Streamlit 交互大屏」** 的全闭环链路。

---

## 🌟 核心技术架构与全流程闭环

```mermaid
flowchart TD
    subgraph 1. 数据工程与 Point-in-Time 基础
        D1["AKShare / TuShare 行情接口"] --> D2["SecurityMaster (真实上市日与动态行业映射)"]
        D2 --> D3["Point-in-Time 真实交易日历 (防幸存者偏差)"]
        D3 --> D4["Parquet 磁盘高速缓存 (462,844 行标准面板)"]
    end

    subgraph 2. 107 因子特征工厂 (Factor Zoo)
        F1["Qlib Alpha158 核心量价子集 (46)"]
        F2["A股特色微观流动性与影线博弈 (15)"]
        F3["遗传规划公式化 Alpha 自动挖掘 (GP_ALPHA)"]
        F4["财务报表基本面成长与质量因子 (9)"]
        F5["另类大单资金流与动量背离 (19)"]
        F1 & F2 & F3 & F4 & F5 --> F6["每日截面 MAD 去极值 + Z-Score + OLS 行业/市值中性化"]
        F6 --> F7["施密特 (Gram-Schmidt) 截面因子正交化"]
    end

    subgraph 3. 模型工坊与时序走步训练 (Model Zoo)
        M1["LightGBM 分类/回归 (Platt Scaling 连续单调校准)"]
        M2["微软 Qlib 级 DoubleEnsemble (样本损失重加权 + 特征子空间扰动)"]
        M3["TabularMLP 深度表征网络"]
        M4["行业产业链图关联 Lead-Lag 滞后传导网络"]
        M5["TPE 贝叶斯超参数自适应搜索"]
        M1 & M2 & M3 & M4 & M5 --> M6["严格 Walk-Forward 时序滚动训练 (Purged Gap 隔离，杜绝未来函数)"]
    end

    subgraph 4. 组合优化与宏观风控
        P1["现代凸优化器 (风险平价 Risk Parity / 约束二次规划 / 倒数波动率)"]
        P2["宏观市场状态识别 (牛市/震荡/高波/熊市) & 波动率目标仓位"]
        P3["动态阶梯回撤熔断 (6% / 10% / 15%)"]
        P4["迟滞换手缓冲区 (Hysteresis Buffer: Top 8 买入 / Top 24 持有)"]
    end

    subgraph 5. 7 重实盘资金安全防御中枢 (Capital Safety Guards)
        S1["95% 资金利用率硬顶 (强制保留 5% 现金防穿透)"]
        S2["20% 单股持仓硬上限 (防单股暴雷)"]
        S3["50% 日内总换手熔断 (防死循环刷单)"]
        S4["--live-confirm 显式二次确认开关 (默认永远 Dry-Run)"]
        S5["价格偏离 <=2% 与 +9.8% 涨停防追高拦截"]
        S6["调仓前全量自动撤销挂起未结订单"]
        S7["MiniQMT 脱机 10 秒自动熔断回退本地 Paper 仿真"]
    end

    subgraph 6. 执行中枢、自动化与交互大屏
        E1["本地 PaperBroker 仿真交易沙盒"]
        E2["迅投 MiniQMT 实盘直连网关"]
        E3["每日 15:05 Windows 自动化调度批处理"]
        E4["飞书 / 企业微信 / 钉钉 富文本决策卡片推送"]
        E5["Streamlit 7 大主题大屏 (localhost:8501)"]
    end

    D4 --> F6
    F7 --> M6
    M6 --> P1
    P1 & P2 & P3 & P4 --> S1
    S1 & S2 & S3 & S4 & S5 & S6 & S7 --> E1 & E2
    E1 & E2 --> E3 & E4 & E5
```

---

## 🔀 5 大核心业务路线一览

| 路线 | 模块定位 | 核心技术与交付成果 |
| :---: | :--- | :--- |
| **路线一** | **多模型时序 Walk-Forward 回测管线** | LightGBM、DoubleEnsemble、风险平价组合优化、T+1 开盘真实撮合、Point-in-Time 防未来函数 |
| **路线二** | **Streamlit 7 大主题交互决策大屏** | 今日选股、净值曲线、Barra 风格归因、实时持仓风控、因子工厂、实盘网关、真实性审计 |
| **路线三** | **盘后 15:05 全自动调度与多通道推送** | Windows Task Scheduler、每日收盘自动拉取、时序打分、飞书/企微/钉钉 决策卡片 |
| **路线四** | **自动化调仓执行引擎与 7 重安全防线** | 先卖后买、100股整手向下截断、95% 资金利用率硬顶、20% 单股持仓上限、50% 换手熔断 |
| **路线五** | **多股票池 Profile 与一键切换** | 沪深300核心蓝筹、中证500成长先锋、科创硬科技龙头、高股息红利稳健 |

---

## 📊 107 因子特征矩阵分类清单 (Factor Taxonomy)

全量 107 因子均执行每日截面 MAD 去极值、Z-Score 标准化与 OLS 行业+市值中性化：

```
107 因子完整体系
├── 1. Qlib Alpha158 经典量价核心子集 (46 个)
│   ├── K线形态特征 (KMID, KLEN, KMID2, KUP, KLOW, KSFT)
│   ├── 动量与收益率变化 (ROC5, ROC10, ROC20, ROC30, ROC60, ROC120, ROC250...)
│   ├── 极值比率与均线偏离 (MAX_RATIO_*, MIN_RATIO_*, MA_RATIO_*)
│   ├── 波动率序列 (STD5, STD10, STD20, STD30, STD60...)
│   └── 收益波动率比 (ROC_STD_*)
├── 2. A股本土微观结构与特色博弈因子 (15 个)
│   ├── 5日均量异动倍数 (TURNOVER_SURGE_5)
│   ├── 距离涨停价空间 (DISTANCE_TO_LIMIT_UP)
│   ├── 连板状态滞后标记 (CONSECUTIVE_LIMIT_UP_TAG)
│   ├── 日内上影线多空博弈 (SHADOW_UPPER_ASYM)
│   ├── 对数流通市值风格 (LOG_CIRC_MV)
│   ├── Amihud 非流动性冲击因子 (AMIHUD_ILLIQUIDITY_20D)
│   ├── Kyle's Lambda 微观价格冲击 (KYLE_LAMBDA_PROXY)
│   └── 下行波动率比率 (DOWNSIDE_VOL_RATIO_20D)
├── 3. 遗传规划 (GP) 符号 Alpha 挖掘因子 (GP_ALPHA)
│   ├── 时序算子库 (ts_rank, ts_corr, decay_linear, ts_std, ts_delta, ts_delay)
│   ├── 演化 Alpha 1 (GP_ALPHA_VOL_PRESSURE_10)
│   ├── 演化 Alpha 2 (GP_ALPHA_REVERSAL_STRENGTH_5)
│   └── 演化 Alpha 3 (GP_ALPHA_PRICE_RANGE_ACCEL_20)
├── 4. 另类特征与资金流动力学 (19 个)
│   ├── 5日主力净流入占比 (FLOW_NET_BUY_RATIO_5D)
│   ├── 20日布林带波动挤压 (VOLATILITY_SQUEEZE_20)
│   ├── 均线多头排列强度 (MA_BULL_ALIGNMENT)
│   └── 10日量价动量背离 (VP_DIVERGENCE_10D)
└── 5. 财务基本面成长与质量因子 (9 个, PIT 延迟 110 天对齐)
    ├── 净资产收益率 (ROE)
    ├── 销售毛利率 (GROSS_MARGIN)
    ├── 每股收益 (EPS)
    ├── 营业收入同比增速 (REVENUE_YOY)
    └── 净利润同比增速 (NET_PROFIT_YOY)
```

---

## 🛡️ 7 重机构级实盘资金安全防御中枢

为杜绝实盘接入券商真实资金（MiniQMT / QMT）时的极端风险，系统内置了 7 重安全防火墙：

| 防线 | 安全防护维度 | 实施机制与技术细节 | 触发场景 |
| :---: | :--- | :--- | :--- |
| **1** | **资金透支防御** | **资金使用率 95% 硬顶**：保留 5% 缓冲现金，绝不打满，防佣金滑点导致透支 | 目标买单市值超限时等比缩减 |
| **2** | **单股暴雷防御** | **单股 20% 仓位硬上限**：单标的分配金额不得超过总权益 20%，超限自动截断 | 单股权重过大时强制向下取整 |
| **3** | **死循环刷单防御**| **日内 50% 总换手熔断**：单日累计委托金额超限自动启动安全降频等比裁剪 | 异常高频循环调仓时锁死 |
| **4** | **误触实盘防御** | **双重显式确认 Flag**：未带 `--live-confirm` 时自动强制降级为 Dry-Run 演练模式 | 任何常规运行与调试 |
| **5** | **滑点失控防御** | **价格偏离限价保护**：委托限价 $\le \text{市价} \times 1.02$，严禁 $+9.8\%$ 涨停价挂买单 | 异常跳空追高时拦截 |
| **6** | **重复成交防御** | **前置清场撤单**：调仓生成前先扫描并全量撤销所有挂起未结订单 | 每次执行再平衡开始前 |
| **7** | **通道脱机防御** | **自动熔断与降级**：MiniQMT 失联 10 秒即刻切断实盘通道并回退本地 Paper 仿真 | 网络波动或客户端崩溃 |

---

## 📁 完整工程代码目录结构

```text
junl53582-oss/ai
├── config/                      # 全局配置中心
│   ├── settings.py              # 系统核心超参数、费率、风控阈值与路径单例
│   └── universe_profiles.py     # 4 大选股股票池 Profile 管理器
├── data/                        # 数据工程层
│   ├── data_fetcher.py          # AKShare / TuShare 多源行情拉取与异常捕获
│   ├── data_manager.py          # Parquet 缓存管理、上市天数过滤与状态标记
│   ├── fundamentals.py          # 业绩报表 PIT 对齐拉取与基本面因子计算
│   ├── security_master.py       # 证券主数据 (上市日、动态行业、ST历史)
│   └── universe_provider.py     # 股票池提供器 (STATIC / POINT_IN_TIME)
├── factors/                     # 因子特征工程层
│   ├── alpha158.py              # Qlib Alpha158 向量化计算引擎
│   ├── custom_ashare.py         # A股本土微观结构因子
│   ├── microstructure_advanced.py# Amihud、Kyle's Lambda、影线不对称高阶因子
│   ├── genetic_miner.py         # 遗传规划公式化 Alpha 自动挖掘器
│   ├── orthogonalizer.py        # 施密特 (Gram-Schmidt) 因子正交化引擎
│   ├── attribution.py           # Barra 6 大风格暴露归因与 CAPM 收益分解
│   ├── registry.py              # 动态因子注册表元数据仓库
│   └── processor.py             # 截面 MAD 去极值 + Z-Score + 逐日行业市值中性化
├── models/                      # 机器学习与深度模型层
│   ├── base_model.py            # 量化预测模型统一抽象基类
│   ├── lightgbm_model.py        # LightGBM 封装 (含 Platt Scaling 连续概率校准)
│   ├── double_ensemble.py       # 微软 Qlib 级 DoubleEnsemble 样本重加权集成模型
│   ├── deep_tabular.py          # TabularMLP 深度表征网络
│   ├── graph_network.py         # 行业产业链图关联 Lead-Lag 滞后传导模型
│   ├── hyper_tuner.py           # TPE 贝叶斯超参数自适应寻优
│   ├── asymmetric_loss.py       # 非对称下行风险惩罚损失函数 (2.5x 假突破惩罚)
│   ├── tradability_mask.py      # A股可交易性掩码 Mask-First 训练引擎
│   ├── labeler.py               # 连续超额收益与二分类极端分位标签生成器
│   ├── walk_forward.py          # 严格时序滚动训练器 (Purged Gap 隔离)
│   └── evaluator.py             # 分类 (AUC/Brier) 与排序 (RankIC/ICIR) 综合评估器
├── strategy/                    # 策略组合决策与风控层
│   ├── trading_rules.py         # A股交易制度 (T+1/整手/涨跌停/印花税)
│   ├── optimizer.py             # 现代组合优化器 (Equal/InvVol/Score/RiskParity/QP)
│   ├── portfolio.py             # Top-K 选股与迟滞缓冲区 (Hysteresis Buffer) 组合构建
│   ├── risk_manager.py          # 宏观市场状态识别、波动率目标仓位与阶梯熔断
│   └── corporate_actions.py     # 分红送配送股 PIT 还原处理器
├── execution/                   # 券商交易网关与实盘执行层
│   ├── broker_base.py           # 券商网关统一接口与订单/账户/持仓数据模型
│   ├── paper_broker.py          # 本地 Paper 仿真交易沙盒
│   ├── miniqmt_broker.py        # 迅投 MiniQMT (xtquant) 实盘直连网关
│   ├── safety_guard.py          # 7 重机构级实盘资金安全防御中枢
│   └── run_trader.py            # 自动化调仓执行器 (先卖后买、资金校验、订单导出)
├── scheduler/                   # 盘后自动化与消息推送层
│   ├── daily_runner.py          # 每日 15:05 全自动流水线执行器
│   └── notifier.py              # 飞书卡片 / 企业微信 / 钉钉 富文本报告推送
├── server/                      # 后端服务层
│   └── app.py                   # FastAPI RESTful API 服务 (/api/v1/predict, /api/v1/backtest)
├── dashboard/                   # 前端交互看板层
│   └── app.py                   # Streamlit 7 大主题量化决策大屏 (秒开缓存优化)
├── tests/                       # 135+ 个全维度自动化测试套件
│   ├── smoke_test.py            # 小型端到端冒烟测试
│   ├── test_no_leakage.py       # 无未来函数与时序隔离测试
│   ├── test_trading_rules.py    # 43 项 A 股实盘交易制度全覆盖测试
│   ├── test_pit_state_integrity.py # 28 项 Point-in-Time 历史时点状态测试
│   ├── test_evidence_integrity.py  # 18 项证据链与防篡改真实性测试
│   ├── test_v8_features.py      # 优化器、网关与另类因子测试
│   ├── test_advanced_optimizations.py # 微观因子、正交化与非对称损失测试
│   ├── test_genetic_and_graph.py# 遗传规划算子、图网络与贝叶斯调参测试
│   ├── test_risk_and_attribution.py # Barra 风格归因与宏观状态风控测试
│   ├── test_execution_safety.py # 7 重实盘资金安全防线单元测试
│   └── test_chaos_stress_pipeline.py # 极端黑天鹅与混沌压力测试
├── scripts/                     # 运维与批处理脚本
│   ├── start_dashboard.bat      # Windows 一键秒开 Streamlit 看板
│   ├── run_daily_post_market.bat# Windows 定时任务每日盘后流水线
│   ├── schedule_daily_job.bat   # Windows 自动注册定时任务脚本
│   ├── compare_models_107_factors.py # 107 因子多模型实证对比评估脚本
│   ├── run_chaos_stress_test.py # 全链路真实性暴力压测与审计脚本
│   └── verify_full_pipeline_e2e.py # 路线一至路线五端到端综合校验
├── requirements.txt             # 完整 Python 依赖清单
└── README.md                    # 项目说明文档
```

---

## 🚀 快速上手与操作指南

### 1. 环境准备与依赖安装

```powershell
# 推荐使用 Python 3.11 环境
git clone https://github.com/junl53582-oss/ai.git
cd ai
pip install -r requirements.txt
```

### 2. 运行全链路自动化测试与真实性审计

```powershell
# 运行 135+ 个单元与集成测试 (耗时约 1~2 分钟，100% 通过)
pytest tests/ -v

# 运行全链路真实性暴力压测与深度审计
python scripts/run_chaos_stress_test.py
```

### 3. 一键跑通全流程量化回测管线

```powershell
# 运行默认 LightGBM + 风险平价策略
python run_pipeline.py --optimizer risk_parity

# 切换至微软 Qlib DoubleEnsemble 样本重加权模型
python run_pipeline.py --model-type double_ensemble --optimizer risk_parity
```

### 4. 启动 Streamlit 7 大主题交互看板

```powershell
# 方式 1: 直接运行
streamlit run dashboard/app.py

# 方式 2: 双击 scripts/start_dashboard.bat 脚本
```
浏览器访问：`http://localhost:8501`

### 5. 启动券商实盘/模拟调仓执行器 (含 7 重资金安全防护)

```powershell
# 本地 Paper 仿真调仓
python execution/run_trader.py --broker paper

# 迅投 MiniQMT 实盘演练 (Dry-Run，仅计算并审查订单，不下单)
python execution/run_trader.py --broker miniqmt --dry-run

# 迅投 MiniQMT 真实下单 (必须显式带 --live-confirm 二次确认)
python execution/run_trader.py --broker miniqmt --live-confirm
```

---

## 📜 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。
