# A股工业级 AI 量化投研、未来价格前瞻推演与自动化交易中台

> **A-Share Institutional AI Quantitative Research, Forward Price Forecasting & Trading Platform**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/License-GPL--3.0-blue.svg?style=flat-square" alt="License GPL-3.0">
  <img src="https://img.shields.io/badge/CI%20Status-Passing%20(Fast%20%2B%20Audit)-10B981.svg?style=flat-square&logo=github-actions&logoColor=white" alt="CI Status">
  <img src="https://img.shields.io/badge/Research%20Certification-FAILED%20(Insufficient%20Evidence)-crimson.svg?style=flat-square" alt="Research Certification FAILED">
  <img src="https://img.shields.io/badge/Live%20Trading-FALSE%20(Paper%20Only)-orange.svg?style=flat-square" alt="Live Trading False">
  <img src="https://img.shields.io/badge/Dashboard-Port%208501-8B5CF6.svg?style=flat-square&logo=streamlit&logoColor=white" alt="Dashboard Port 8501">
</p>

---

> [!IMPORTANT]
> **【零伪造科学诚信准则 (Zero-Mock Provenance)】**
> 本项目所有量化指标、超额收益、RankIC 与回测曲线均来自真实物理计算与不可篡改密码学凭证（SHA-256 Manifest）。杜绝任何常量伪造与过度拟合欺骗。
> 严禁实盘直接使用，当前实盘就绪门禁严格保持：`LIVE_TRADING_READY = FALSE`（仅支持模拟与 Paper Trading 验证）。

---

## 📌 目录导航 (Table of Contents)

1. [🚦 系统权威状态矩阵 (Current Status)](#1-🚦-系统权威状态矩阵-current-status)
2. [💡 项目定位与核心技术亮点 (Project Overview)](#2-💡-项目定位与核心技术亮点-project-overview)
3. [🏛️ 八层量化系统工程架构 (Architecture)](#3-🏛️-八层量化系统工程架构-architecture)
4. [🔬 生产模型与研究候选模型体系 (Model Registry)](#4-🔬-生产模型与研究候选模型体系-model-registry)
5. [🛡️ 七重实盘资金安全风控守卫 (Safety Guards)](#5-🛡️-七重实盘资金安全风控守卫-safety-guards)
6. [📈 交互看板与未来 5 日价格前瞻 (Dashboard & Forecast)](#6-📈-交互看板与未来-5-日价格前瞻-dashboard--forecast)
7. [💻 快速克隆、安装与运行全流程 (Quickstart)](#7-💻-快速克隆安装与运行全流程-quickstart)
8. [📦 数据集获取与防篡改哈希核验 (Data & Artifacts)](#8-📦-数据集获取与防篡改哈希核验-data--artifacts)
9. [📁 项目完整工程目录结构 (Directory Layout)](#9-📁-项目完整工程目录结构-directory-layout)
10. [❓ 常见问题排查与 FAQ (Troubleshooting & FAQ)](#10-❓-常见问题排查与-faq-troubleshooting--faq)
11. [📜 开源许可证与免责声明 (License & Disclaimer)](#11-📜-开源许可证与免责声明-license--disclaimer)

---

## 1. 🚦 系统权威状态矩阵 (Current Status)

本项目严格坚持单一事实源（Single Source of Truth），将系统状态严格划分为三大维度：**软件工程状态**、**正式科研认证状态**与**部署/实盘状态**。

> [!WARNING]
> **【核心概念边界 Clarification】**
> - `ModelRegistry` 中的 `PRODUCTION` 状态仅代表**已打包的部署工程制品（`DEPLOYMENT_ARTIFACT` / `PAPER_PRODUCTION_MODEL`）**，可在模拟盘或交互看盘中运行批处理推理。
> - **`PRODUCTION` 状态绝对不等于科学实证认证通过（`scientific VERIFIED`），亦绝对不等于实盘交易批准（`live trading approved`）。**
> - 本项目实盘交易门禁永久硬阻断：`LIVE_TRADING_READY = FALSE`。

### 1.1 软件工程与 CI 状态 (Software Engineering Status)
| 检查项 | 状态判定 | 执行环境 | 实际测试记录 (Authoritative Logs) |
| :--- | :---: | :---: | :--- |
| **Local Full Pytest** | `PASS` | Windows 11 / Py 3.11 | **568 passed**, 0 failed, 164 warnings in 222.05s |
| **CI: Fast CI** | `PASS` | Ubuntu-latest / Py 3.11 | **178 passed**, 2 deselected, 2 warnings in 162.58s (Run #33842051349) |
| **CI: Audit Hardening** | `PASS` | Ubuntu-latest / Py 3.11 | **33 passed**, 2 warnings in 2.94s (Run #33842051439) |
| **CI: r3.2 Adversarial** | `PASS` | Ubuntu-latest / Py 3.11 | **16 passed**, 1 warning in 1.16s (Run #33842051439) |
| **CI: Formal E2E Smoke** | `PASS` | Ubuntu-latest / Py 3.11 | **3 passed**, 7 warnings in 9.42s (Run #33842051439) |
| **CI: Linux Full Pytest** | `PASS` | Ubuntu-latest / Py 3.11 | **567 passed**, 1 skipped, 52 warnings in 225.43s (Run #33842051439) |
| **Production Research Audit** | `VALID` | GitHub Actions Workflow | Workflow 语法与 schema 100% 合法，支持 `workflow_dispatch` 启动与 fail-closed 门禁拦截 |

### 1.2 科研认证门禁状态 (Research Certification Status)
依据官方冻结规范与 `FINAL_RUN_POINTER.json`（指向权威运行 `research_8dbf062_20260831_155701`），科研认证结论严格保持如下：
| 科研认证门禁维度 | 状态值 | 判定依据与说明 |
| :--- | :---: | :--- |
| **INFRASTRUCTURE_STATUS** | `INSUFFICIENT_EVIDENCE` | 缺少官方财报披露日逐笔因果存证，STRICT_FUNDAMENTAL_PIT 证据不足 |
| **MODEL_EVIDENCE_STATUS** | `MIXED_EVIDENCE_NOT_ROBUST` | Bootstrap 95% 置信区间下界 <= 0，且跨种子稳定性方差未达极致阈值 |
| **GOVERNANCE_STATUS** | `PASS` | 32 项代码防篡改、无未来函数与反欺诈门禁全量通过 |
| **OVERALL_RESEARCH_STATUS**| `FAILED` | 依据科研门禁体系，因上述关键项未达标，综合科研认证结论判定为未通过 |
| **FINAL_HOLDOUT_AVAILABLE**| `FALSE` | 严守时序盲测准则，终极样本外盲测集未到解封时点 |
| **PRODUCTION_MODEL_PROMOTION**| `FALSE` | 正式科研模型晋升已被治理网关否决，禁止自动转正 |

### 1.3 部署与实盘状态 (Deployment / Trading Status)
| 运行时维度 | 状态值 | 说明与风控边界 |
| :--- | :---: | :--- |
| **🚨 实盘交易就绪状态** | `FALSE` | **`LIVE_TRADING_READY = FALSE`**。绝对禁止实盘真实资金交易，仅限模拟盘沙盒 |
| **🏭 在役部署模型制品** | `DEPLOYMENT_ARTIFACT` | `m_20260903_194757_hybrid_bagging_ridge` (仅供 Paper Trading 与前瞻推演) |
| **🧪 前沿研究候选模型** | `CANDIDATE` | Gen 4 (DRL 智能体) / Gen 5 (双塔排序)，仅供离线实验 |
| **📅 盘后调度工作流** | `OPERATIONAL` | 每日 15:05:00 自动执行，默认 `--mode inference`，生产环境严禁重训 (fail-closed) |
| **🌐 可视化投研大屏** | `ONLINE` | Streamlit Port 8501 (`streamlit run dashboard/app.py`)，7 大主题看板正常运作 |

---

## 2. 💡 项目定位与核心技术亮点 (Project Overview)

本项目是一套专为 A 股市场研发的**工业级多模态 AI 量化投研与生产交易闭环中台**，解决传统开源量化“未来函数泛滥、简单分类无法反映真实排序、风控形同虚设、实盘无法闭环”的根本痛点：

* 🛡️ **严格零幸存者偏差 (Point-In-Time Universe)**：接入 2021~2026 年沪深 300 真实时点调仓事件流，历史回测严格对齐当时成分股。
* 🔒 **因果隔离与防视前泄漏**：财报按公告日延后 110 天披露窗口对齐，时序走步训练严格设置 25 天 Purge 隔离期。
* 🧩 **统一模型适配器层 (Model Adapter Pattern)**：解耦推理引擎与底层模型结构，无缝兼容 LightGBM、浅树袋装集成、DRL 及 PyTorch 深度模型。
* ⚖️ **A 股实盘交易硬约束**：内置 T+1 卖出限制、100 股整手向下取整、一字涨跌停禁买禁卖、ST 标的 5% 限价与双边滑点摩擦。
* 🤖 **盘后全自动巡航闭环**：每日收盘后自动同步全市场行情、批量推理截面得分、触发风控组合优化并多通道推送决策卡片。

---

## 3. 🏛️ 八层量化系统工程架构 (Architecture)

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 1: 数据基座 (Point-In-Time 行情专线直连 + 季度财报披露时点对齐)   │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 2: 特征工厂 (126维量价微结构 + 主力资金流背离 + 截面正交中性化)   │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 3: 投研引擎 (Purged Walk-Forward 滚动折叠 + 25天因果隔离 Gap)     │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 4: 防伪存证 (SHA-256 物理文件防篡改 + Git Commit 密码学证据链)    │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 5: 模型仓库 (ModelRegistry 状态机 + Model Adapter 统一推理适配层) │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 6: 前瞻推演 (未来 5 日价格预期走势 + 90% 置信区间 + 建议挂单卡)   │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 7: 执行风控 (7重实盘资金熔断守卫 + PaperBroker 仿真撮合账本)      │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 8: 交互看板 (Streamlit 7大主题大屏 + 每日 15:05 自动推送中枢)     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 🔬 生产模型与研究候选模型体系 (Model Registry)

### 4.1 现役生产模型 (Official Production)
* **Model ID**: `m_20260903_194757_hybrid_bagging_ridge`
* **模型架构**: 第二代深度混合异构集成（多种子浅树 Bagging + L2 线性正则化底仓）
* **特征输入**: 32 维精选微观特征（残差动量、换手率突变、质量动量交叉、特质低波惩罚等）
* **样本外指标**: Mean Rank IC = `0.0258`, Rank ICIR = `0.1916`
* **存储路径**: `saved_models/production/m_20260903_194757_hybrid_bagging_ridge/`（含 `model.pkl`、`metadata.json`、`manifest.json`）

### 4.2 前沿研究模型 (Research Candidates)
* **Gen 4 (DRL Policy Agent)**: 深度强化学习动态调仓智能体（策略网络根据波动率与换手自适应分配仓位）。
* **Gen 5 (Dual-Tower DeepRank)**: 基于 Pairwise Ranking 损失函数的双塔排序模型（时序注意力塔 + 截面强弱塔）。
* **Temporal Transformer**: 因果掩码时序注意力模型（捕捉微观资金大单沉淀）。

> **模型治理规范**: 任何新模型必须经由 `ModelRegistry.promote()` 完成指标检验与多级审批，严禁私自替换生产模型文件。

---

## 5. 🛡️ 七重实盘资金安全风控守卫 (Safety Guards)

| 守卫层级 | 风控守卫规则 | 触发阈值 | 保护动作 | 机制解读 |
| :---: | :--- | :---: | :--- | :--- |
| **L1** | **单日换手率熔断** | `> 50.0%` | 强制按比例缩放指令，阻断异常大换手 | 防止极端市场下模型频繁换仓产生高额滑点手续费 |
| **L2** | **单股最大持仓暴露** | `> 18.0%` | 强制截断单一标的权重，分散非系统性风险 | 杜绝单票黑天鹅导致组合净值大幅下挫 |
| **L3** | **单一行业集中度硬顶** | `> 30.0%` | 行业总敞口超限时强制等比例平滑重分配 | 严防单一板块政策突发冲击引发系统性亏损 |
| **L4** | **流动性冲击成本保护** | `> 2.0% 日成交` | 拆分多日分批委托队列，严禁市价砸盘冲击 | 依据目标标的日均成交金额智能测算冲击成本 |
| **L5** | **价格偏离与涨跌停防夹** | `> 2.5% 偏离` | 涨跌停板严禁追单，偏离基准价超限时撤单 | 杜绝涨停板买不进、跌停板卖不出的实操踩踏 |
| **L6** | **大盘短线情绪避险** | `温度 < 35°C` | 市场极端冰点时自动将总股票仓位下调至 30% | 联动 300 标的短线情绪温度计，智能规避退潮期 |
| **L7** | **手续费与现金底仓垫** | `≥ 5.0% 现金` | 强制预留 5% 现金头寸作为手续费摩擦缓冲垫 | 确保极端调仓手续费充沛，杜绝透支穿仓风险 |

---

## 6. 📈 交互看板与未来 5 日价格前瞻 (Dashboard & Forecast)

系统在 Streamlit 看板中提供全功能量化投研视图：
* **3 栏金融 K 线穿透**: 日 K 蜡烛图 + MA5/MA20/MA60 均线 + 🔴 量化 B 点金叉 / 🟢 S 点止盈 + 成交量 + 主力大单净流入趋势。
* **未来 5 日价格走势前瞻**: K 线前瞻预测虚线 + 90% 置信区间光晕 + 明日建议挂单区间与止盈止损决策卡。
* **短线情绪温度计**: 基于 300 标的涨跌分布计算全市场短线投机情绪。

---

## 7. 💻 快速克隆、安装与运行全流程 (Quickstart)

### 7.1 克隆代码仓库
```bash
git clone https://github.com/junl53582-oss/ai.git
cd ai
```

### 7.2 创建与激活 Python 虚拟环境
推荐使用 Python 3.11 环境：
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 激活虚拟环境 (Linux / macOS)
source .venv/bin/activate
```

### 7.3 安装系统依赖
```bash
# 升级 pip
python -m pip install --upgrade pip

# 安装轻量 CPU 版 PyTorch (若具备 GPU 可安装对应 CUDA 版本)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 安装项目全部依赖
pip install -r requirements.txt
```

### 7.4 运行全量自动化测试套件
在启动系统前，建议运行测试以确认本地环境完整性（全仓库 568 项测试均应通过）：
```bash
# 运行审计门禁与生产加固回归测试
python -m pytest -v tests/test_audit_hardening.py
python -m pytest -v tests/test_r3_2_evidence_integrity.py
python -m pytest -v tests/test_production_hardening.py

# 运行全量测试套件 (约 3~4 分钟)
python -m pytest -q
```

### 7.5 启动系统
#### 启动可视化交互看板
```bash
streamlit run dashboard/app.py --server.port 8501
```
打开浏览器访问 **`http://localhost:8501`** 即可进入大屏。

#### 运行盘后自动化批量推理
```bash
# Windows 批处理一键执行
scripts\run_daily_post_market.bat

# 或通过 Python 命令行调用 (生产推荐模式)
python scheduler/daily_runner.py --mode inference --optimizer risk_parity
```

---

## 8. 📦 数据集获取与防篡改哈希核验 (Data & Artifacts)

### 8.1 仓库内数据 vs 仓库外数据
为保证 GitHub 代码仓库体积轻巧（< 100MB 快速 clone），本项目采用分层策略：

| 数据文件 | 体积 | 状态 | 说明与获取方式 |
| :--- | :---: | :---: | :--- |
| `data_storage/fundamentals/yjbb_*.parquet` | ~29 MB | **已入库** | 覆盖 2018~2026 季度财报披露日，用于 PIT 基本面因果溯源 |
| `data_storage/research/market_daily_300_v2.parquet` | ~20 MB | **已入库** | 覆盖 2021~2026 沪深 300 官方复核行情，修复市值血缘污染 |
| `data_storage/universe_pit_events.parquet` | ~20 KB | **已入库** | 时点指数成分股变动事件流，杜绝幸存者偏差 |
| `data_storage/parquet/market_daily.parquet` | ~300 KB | **已入库** | CI 冒烟测试专用快照 |
| `data_storage/factors/factor_matrix.parquet` | ~800 KB | **已入库** | CI 冒烟测试专用因子矩阵 |
| `data_storage/research/factor_matrix_300.parquet` | ~311 MB | **Git Ignored** | 全量科研级因子矩阵。可由入库数据一键确定性重建 |

### 8.2 本地确定性重建全量因子矩阵
无需额外下载 311MB 大文件，在安装依赖后运行以下脚本即可在本地确定性复现：
```bash
python tools/build_dataset_300_v2.py
```

### 8.3 数据完整性与防篡改门禁校验
运行以下命令校验本地数据与官方清单哈希是否自洽：
```bash
python -u tools/check_committed_dataset_schema.py
python -u tools/validate_certification_artifacts.py
```

详细数据与模型获取说明请参阅：[docs/DATA_AND_MODELS.md](docs/DATA_AND_MODELS.md)。

---

## 9. 📁 项目完整工程目录结构 (Directory Layout)

```text
ai/
├── .github/                    # GitHub CI/CD 工作流与 PR/Issue 模版
│   ├── workflows/              # 自动化测试流水线 (fast_ci, audit_hardening)
│   ├── ISSUE_TEMPLATE/         # Bug 报告与功能建议模版
│   └── pull_request_template.md # PR 提交自查清单
├── config/                     # 全局参数与风控阈值
│   ├── settings.py             # 系统核心配置中枢
│   └── universe_profiles.py    # 多股票池定义管理器
├── dashboard/                  # Streamlit 交互大屏
│   └── app.py                  # 7 大主题大屏入口
├── data/                       # 数据同步与 PIT 股票池提供者
│   ├── data_manager.py         # 权威数据同步与清洗中枢
│   └── universe_provider.py    # 动态时点成分股提供器
├── data_storage/               # 数据存储目录 (含财报、行情快照与清单)
├── docs/                       # 核心文档与技术白皮书
│   └── DATA_AND_MODELS.md      # 数据与模型获取详细说明
├── execution/                  # 券商撮合与实盘接口
│   ├── miniqmt_broker.py       # 迅投 MiniQMT 实盘交易网关 (可选)
│   └── paper_broker.py         # 仿真交易沙盒与账户账本
├── factors/                    # 因子计算与截面特征工程
│   ├── alpha158.py             # Alpha158 算子库
│   └── processor.py            # 截面中性化与正交化处理器
├── models/                     # 机器学习模型与适配层
│   ├── adapters/               # 统一模型适配器 (LightGBM, Bagging, DRL)
│   ├── bagging_ensemble.py     # 多随机种子袋装浅树集成模型
│   ├── drl_strengthened_model.py # 强化学习增强模型
│   ├── inference.py            # 生产截面批量推理引擎 (BatchInference)
│   └── registry.py             # 模型生命周期注册表 (ModelRegistry)
├── reports/                    # 历史认证证据链与审计报告
├── saved_models/               # 模型制品存储
│   ├── production/             # 官方生产模型制品目录 (含 model.pkl, manifest.json)
│   └── legacy/                 # 历史模型归档目录 (含防混淆声明)
├── scheduler/                  # 调度器与通知
│   ├── daily_runner.py         # 每日盘后自动化调度主程序
│   └── notifier.py             # 飞书/企业微信/钉钉消息卡片渲染
├── scripts/                    # 运维与自动化执行脚本
│   └── run_daily_post_market.bat # Windows 盘后批处理脚本
├── tests/                      # 自动化测试套件 (47 个文件，568 项测试)
├── tools/                      # 科研重现与加固验证工具集
├── .env.example                # 环境变量配置模版 (安全示例)
├── CONTRIBUTING.md             # 开发者贡献与合规指南
├── LICENSE                     # GNU General Public License v3.0
├── requirements.txt            # 项目运行依赖清单
└── README.md                   # 项目主技术与部署文档
```

---

## 10. ❓ 常见问题排查与 FAQ (Troubleshooting & FAQ)

<details>
<summary><b>Q1: 为什么在生产环境下运行 --mode research 会报错？</b></summary>

这是系统内置的 **Fail-Closed 生产安全熔断机制**。当检测到环境变量 `PRODUCTION_RUNTIME=1` 时，严禁使用 research 现场重训模式直接输出交易信号，以防止模型口径漂移与无法归因。生产运行必须使用 `--mode inference` 加载经委员会审批的生产模型。
</details>

<details>
<summary><b>Q2: clone 仓库后缺少 factor_matrix_300.parquet，回测是否会报错？</b></summary>

该文件体积为 311MB，因超过 GitHub 单文件限制未存放在普通 Git 中。只需在安装依赖后运行 `python tools/build_dataset_300_v2.py`，系统即可直接从已入库的 20MB 行情基线中确定性重建该文件并校验哈希。
</details>

<details>
<summary><b>Q3: 如何配置飞书或企业微信每日自动推送？</b></summary>

将根目录下的 `.env.example` 复制为 `.env`，并在其中填写：
```bash
QUANT_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/你的机器人TOKEN"
QUANT_NOTIFIER_CHANNEL="feishu"
```
系统在每日 15:05 批量推理与调仓完成后，将自动生成排版美观的次日买卖决策卡片并推送。
</details>

<details>
<summary><b>Q4: PyTorch 是否为必须安装项？</b></summary>

如果仅运行官方生产模型（`hybrid_bagging_ridge`）的批量推理与每日选股，CPU 环境下安装的基础库即可完全满足需求；系统测试用例中涉及 PyTorch 的部分已配备 `importorskip` 容错机制。若需训练或推演前沿研究模型（Gen 5 DeepRank / Temporal Transformer），建议安装 PyTorch。
</details>

---

## 11. 📜 开源许可证与免责声明 (License & Disclaimer)

本项目基于 **[GNU General Public License v3.0 (GPL-3.0)](LICENSE)** 协议开源。

### ⚠️ 金融投资风险免责声明 (Financial Disclaimer)
1. 本项目所包含的代码、算法、模型、预测分值及相关文档**仅供量化投研学习、学术交流与算法验证**使用，不构成任何实质性投资建议或证券买卖推荐。
2. 证券市场存在极高风险，量化策略历史回测表现不预示其未来收益，模型前瞻推演结论具有不确定性。投资者据此操作所产生的任何盈利或亏损均由个人独立承担，本项目研发者及贡献者不承担任何直接或间接法律责任。
3. **严禁在未经过充分模拟盘检验（Paper Trading）与专业合规审查的前提下，将本系统直接用于真实资金交易**。
