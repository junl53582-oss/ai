# A股多因子股票涨跌预测与量化回测决策系统 (Enterprise A-Share Quant System)

基于 **Qlib Alpha158Subset (46个) + A股定制因子 (13个) + 截面中性化 + LightGBM 走步回测 (Walk-Forward + Purged Gap) + A股实盘交易硬约束 (T+1 / 一字板校验 / 订单状态机 / 熔断降仓)** 的企业级量化投研与决策系统。

系统默认采用 **涨跌二分类 (Classification)** 模式：预测每只股票未来 2 个交易日是否「跑赢基准 (上涨)」，输出上涨概率 (0~1)。同时保留回归模式 (Regression) 用于 A/B 对比与排序研究。

---

## 🔀 双任务模式 (涨跌分类 vs 连续收益回归)

通过 `config/settings.py` 中的 `TASK_TYPE` 一键切换：

| 配置项 | 分类模式 (默认) | 回归模式 |
| :--- | :--- | :--- |
| `TASK_TYPE` | `"classification"` | `"regression"` |
| 标签列 | `label_up_down_2d` (0=跌/跑输, 1=涨/跑赢) | `label_excess_2d` (连续超额收益) |
| LightGBM 目标 | `objective=binary` | `objective=regression` |
| 模型输出 (`pred_score`) | 上涨概率 ∈ [0, 1] | 预期超额收益 (连续值) |
| 评估指标 | AUC / Accuracy / Precision / Recall / F1 / Brier / 混淆矩阵 | IC / RankIC / ICIR / RankICIR / 滚动 RankIC |
| 二分类阈值 | `LABEL_THRESHOLD = 0.0` (超额收益 > 0 判定为涨) | - |

分类模式下，`pred_score` 即「未来 2 日跑赢沪深300基准的概率」，策略按概率降序选股 (Top-K)，概率越高越优先买入。

---

## 🌟 核心架构与严格实盘交易时间轴

整个系统严格遵循“无未来函数”原则，任何时刻 $T$ 的特征、预测与决策只能依赖 $\le T$ 的历史数据，交易撮合严格在 $T+1$ 开盘执行：

```text
T 日交易结束 (15:00)
        ↓
获取 T 日完整量价 (包含未复权 raw OHLCV 与前复权 adj OHLCV)
        ↓
计算 T 日 Alpha158Subset + A股特色因子 (基于 adj 价格)
        ↓
每日截面 MAD 去极值 + Z-Score 标准化 + 截面回归中性化 (对数市值/行业)
        ↓
LightGBM 模型预测未来 2 日涨跌方向 (上涨概率 pred_score ∈ [0,1])
        ↓
股票横截面打分排序 (Top_K_Buy 与 Top_K_Hold 缓冲区)
        ↓
T 日收盘后生成交易决策订单 (Order: PENDING, signal_date = T, signal_price = T Close)
        ↓
T+1 日开盘集合竞价 (09:30)
        ↓
A股交易规则校验 (T+1可用股数 / 停牌 / 一字涨停禁买 / 一字跌停禁卖)
        ↓
以 T+1 日 Open 价格撮合成交 (execution_date = T+1, execution_price = T+1 Open ± 滑点)
        ↓
更新账户现金、持仓、真实摩擦成本 (印花税0.05% + 佣金万2.5 + 滑点0.1%) 与 NAV
        ↓
多层动态风控监控 (个股硬止损-8% / 跟踪止盈高点回撤5% / 组合最大回撤12%熔断降仓至30%)
```

---

## 🔬 核心技术要点与无未来函数设计

1. **严格消除未来函数 (Zero Lookahead / Leakage)**：
   - **T日信号 $\to$ T+1日开盘执行**：绝对禁止使用 T 日 Close 预测又在 T 日 Close 成交；回测末尾无下一交易日时信号保持 PENDING，绝不伪造成交。
   - **Walk-Forward 滚动训练 + Purged Gap 隔离**：由于预测目标为未来 2 日超额收益，训练集末尾与验证集末尾各应用 `PURGE_GAP_DAYS = 5` 天隔离（恒大于标签视野），彻底杜绝训练标签渗透入验证集与测试集。
   - **杜绝未来填充**：全面移除 `bfill()`，缺失值仅允许使用历史已知价格前向填充 (`ffill()`) 或补 0。
2. **拆分原始未复权价格与前复权价格**：
   - **未复权价格 (`open, high, low, close`)**：用于真实回测撮合、100股整手计算、成交金额、手续费、滑点与账户市值估值；
   - **前复权价格 (`adj_open, adj_high, adj_low, adj_close`)**：用于 Alpha 因子计算、技术指标与跨期超额收益率标签生成，杜绝分红除权产生的虚假阶跃。
3. **因子库体系与截面中性化**：
   - **Qlib Alpha158Subset (46个核心因子)**：K线形态(6)、动量收益(20)、波动率极差(6)、成交量衍生(8)、量价相关与量能加权(6)；
   - **A股本土化定制因子 (13个特色因子)**：换手率异动(TURNOVER_SURGE)、距离涨停空间、连板状态滞后标记、日内影线多空博弈、对数流通市值风格因子；
   - **截面中性化 (`neutralize_cross_section`)**：每日截面执行 OLS 回归剥离对数市值与行业风格偏差，取回归残差作为纯净 Alpha。
4. **A股实盘交易硬约束与订单状态机**：
   - **订单状态机**：`PENDING`（待执行）、`FILLED`（已成交）、`REJECTED`（拒绝）、`DEFERRED`（延期顺延）、`CANCELLED`（撤销）；
   - **T+1 机制**：买入股票当日锁定可用卖出股数为 0，次日开盘自动解锁；
   - **涨跌停精细化判断**：一字涨停禁止买入；触发止损但若遇一字跌停或停牌，订单标记为 `DEFERRED`，在后续开板交易日继续尝试平仓；
   - **整手交易**：买入按 100 股向下取整；
   - **持仓缓冲区 (Hysteresis Buffer)**：`Top_K_Buy = 8`（未持仓需排名前 8 才能买入），`Top_K_Hold = 16`（已持仓在排名前 16 内继续持有，排名 > 16 才卖出），大幅降低无效换手。
5. **真实摩擦成本与动态风控**：
   - 印花税：卖出单边 $0.05\%$
   - 券商佣金：双边万分之 $2.5$（最低 5 元）
   - 滑点成本：双边 $0.1\%$
   - 个股硬止损（-8%）、跟踪止盈（基于 High 价格高点回撤 5% 锁定利润）、组合最大回撤熔断（回撤超 12% 目标压降持仓至 30%）。
6. **数据真实性保障 (No Silent Synthetic Data)**：
   - `ALLOW_SYNTHETIC_DATA = False`（默认）：数据接口失败时直接抛出 `DataFetchError`，杜绝静默生成随机假数据；
   - 显式开启 `ALLOW_SYNTHETIC_DATA = True` 时，终端输出明确警告，并在结果中标记 `data_source = synthetic`。

---

## 📁 项目工程目录

```text
股票预测/
├── config/
│   ├── __init__.py
│   └── settings.py              # 全局配置参数 (双K缓冲区、成本、风控阈值、PurgedGap、超参数、并行度)
├── data/
│   ├── __init__.py
│   ├── data_fetcher.py          # AKShare 原始/复权双价格拉取、本地 CSV 导入与 DataFetchError 保护
│   ├── data_manager.py          # Parquet 缓存管理、上市天数过滤 (>=60日)、A股状态标记
│   ├── security_master.py       # 证券主数据 (上市日期/行业映射) 与真实性校验
│   └── universe_provider.py     # 股票池提供器 (STATIC / INDEX_CONSTITUENTS / 时点成分 PIT 验证)
├── factors/
│   ├── __init__.py
│   ├── alpha158.py              # Alpha158Subset (46个核心因子) 向量化计算引擎
│   ├── custom_ashare.py         # A股定制因子 (13个特色因子)
│   └── processor.py             # 截面 MAD 去极值 + Z-Score 标准化 + 截面回归中性化 (支持多进程并行)
├── models/
│   ├── __init__.py
│   ├── labeler.py               # 未来 N 日标签生成器 (连续超额收益 + 涨跌二分类双标签)
│   ├── lightgbm_model.py        # LightGBM 模型封装 (分类/回归双模式、早停、Isotonic 概率校准)
│   ├── walk_forward.py          # 严格 Walk-Forward 走步滚动训练器 (含 Purged Gap 隔离)
│   └── evaluator.py             # AUC / IC / RankIC / ICIR / 滚动 RankIC / Q1~Q5 单调性
├── strategy/
│   ├── __init__.py
│   ├── trading_rules.py         # A股实盘硬规则、涨跌停判定与 Order 状态机
│   ├── corporate_actions.py     # 公司行为处理 (分红送配回测还原)
│   └── portfolio.py             # Top-K 缓冲区 (Hysteresis) 选股与再平衡权重计算 (含行业硬上限)
├── backtest/
│   ├── __init__.py
│   ├── engine.py                # 走步事件驱动回测引擎 (T日收盘决策 -> T+1日开盘撮合)
│   ├── risk_control.py          # 动态多层风控引擎 (个股止损、跟踪止盈、回撤熔断降仓)
│   ├── performance.py           # 绩效评价 + reports/ 产物落盘 (指标JSON/净值CSV/订单CSV/热力图PNG)
│   └── audit.py                 # Fail-Closed 运行时审计元数据收集 (AuditMetadata)
├── server/
│   ├── __init__.py
│   └── app.py                   # FastAPI RESTful API 后端服务 (提供 expected_execution_date)
├── dashboard/
│   ├── __init__.py
│   └── app.py                   # Streamlit 交互式量化决策看板
├── tests/
│   ├── __init__.py
│   ├── test_no_leakage.py       # 无未来函数与时序隔离自动化单元测试套件
│   ├── test_trading_rules.py    # A股实盘交易硬规则全覆盖单元测试套件
│   ├── test_pit_state_integrity.py  # 时点状态完整性测试
│   ├── test_evidence_integrity.py   # 证据链完整性测试
│   └── test_neutralization_parity.py # 中性化串行/并行数值一致性测试
├── tools/
│   └── generate_audit_report.py # CAPABILITY / RUNTIME_ATTESTATION / MASTER 三报告生成器
├── run_pipeline.py              # 一键自动化运行与调度入口脚本
├── requirements.txt             # 项目依赖清单
└── README.md                    # 项目说明文档
```

---

## 📂 运行产物说明

每次执行 `python run_pipeline.py` 会自动生成：

| 产物 | 路径 | 内容 |
| :--- | :--- | :--- |
| 绩效指标 JSON | `reports/performance_*.json` | 全套指标 (含 monthly_table) + 审计元数据摘要 |
| 净值曲线 CSV | `reports/equity_curve_*.csv` | 逐日策略/基准 NAV 归一化与回撤列 |
| 订单流水 CSV | `reports/orders_*.csv` | 全部订单明细 (含税费拆解) |
| 月度热力图 | `reports/monthly_heatmap_*.png` | 年×月 收益率热力图 |
| 运行时审计 | `artifacts/runtime_audit.json` | 数据血缘、来源明细、可信度评级 (默认导出) |
| 认证报告 | `RUNTIME_ATTESTATION.md` 等 | 基于真实运行产物自动刷新评级 |

以上文件均带时间戳与 `latest` 双副本，看板与 API 可直接读取 `latest` 版本。

---

## 🚀 快速上手与操作指南

### 1. 安装环境依赖

```powershell
pip install -r requirements.txt
```

### 2. 运行自动化单元测试 (验证无未来函数与交易规则)

```powershell
pytest tests/ -v
```

### 3. 一键跑通全流程量化管线 (命令行)

```powershell
python run_pipeline.py
```

### 4. 启动 Streamlit 交互式决策看板

```powershell
streamlit run dashboard/app.py
```
或：
```powershell
python run_pipeline.py --serve-dashboard
```

### 5. 启动 FastAPI RESTful 后端服务

```powershell
python run_pipeline.py --serve-api --port 8000
```
访问 `http://localhost:8000/docs` 查看 Swagger 交互式文档。
