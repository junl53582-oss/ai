# A股量化研究与观察系统架构规范 (Release v9.0.0)

## 🏛️ 八层量化系统工程架构

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 8: 仿真撮合与实盘安全硬阻断层                                     │
│ Execution Sandbox & Hard Live Gate                                      │
│                                                                         │
│ Paper / Shadow 独立物理核算 | 严禁伪造订单与成交                       │
│ LIVE_TRADING_READY = False 物理阻断所有真实资金委托                    │
│ MiniQMT 实盘网关断线与前置门禁：一律 Fail-Closed 拦截并报错             │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 7: 组合策略与执行层                                                │
│ Portfolio Policy & Execution                                            │
│                                                                         │
│ Top-K / Hold Buffer / Equal Weight / Inverse Vol 独立版本化              │
│ T+1制度 | 涨跌停 | 停牌延期 | 滑点 | 印花税 | 订单守恒                   │
│ Q_req = Q_filled + Q_rem + Q_canc                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 6: DAILY PIT 正式生产预测运行层                                   │
│ Canonical Production Runtime                                            │
│                                                                         │
│ 15:05 Preliminary Check                                                 │
│       ↓                                                                 │
│ 18:30 PIT Data Window                                                   │
│       ↓                                                                 │
│ seal-inputs → preflight → predict → rank → canonical ledger             │
│       ↓                                                                 │
│ settlement / monitoring                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 5: 不可变生产模型注册与晋级层                                      │
│ Production Model Registry & Promotion                                   │
│                                                                         │
│ Candidate Model → Promotion Gate → Production Model                     │
│ Model SHA | Feature Schema | Label Version | Code Freeze SHA            │
│ Research 禁止直接写入 Production Registry                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 4: 科学验证与防伪治理层                                            │
│ Scientific Governance & Certification                                   │
│                                                                         │
│ Walk-Forward | Purged Gap | Bootstrap | Multi-Seed                      │
│ Cost Robustness | Regime Robustness | Exposure Audit                    │
│ Untouched Holdout                                                       │
│                                                                         │
│ INFRASTRUCTURE_STATUS                                                   │
│ MODEL_EVIDENCE_STATUS                                                   │
│ CERTIFICATION_STATUS                                                    │
│ LIVE_TRADING_STATUS                                                     │
│                                                                         │
│ CAPABILITY_REPORT ≠ RUNTIME_ATTESTATION                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Alpha 研究与模型实验层                                          │
│ Research Engine                                                         │
│                                                                         │
│ Alpha Discovery | Feature Ablation | Label Research                     │
│ Ridge / LightGBM / Ensemble Candidate                                   │
│ Walk-Forward Research                                                   │
│                                                                         │
│ DoubleEnsemble 仅作为候选实验，不默认视为 Alpha 改进                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 2: PIT 特征与标签工厂                                              │
│ Feature & Label Factory                                                 │
│                                                                         │
│ Alpha158 | 微观结构 | PIT 财务 | 另类交互 Alpha                         │
│ MAD/Winsorize | Cross-sectional Normalize                              │
│ Industry Neutralization | Size Neutralization                          │
│                                                                         │
│ Execution-Aligned 20D Label:                                            │
│ T Close Signal → T+1 Executable Entry → 20-session Horizon              │
│ → Executable Exit → Benchmark/Cost Adjustment                           │
│                                                                         │
│ T+2 仅代表买入后的最低合法卖出约束，不代表 Label Horizon                │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 1: PIT 数据与不可篡改血缘层                                        │
│ PIT Data & Provenance Lake                                              │
│                                                                         │
│ Exchange Calendar | PIT Universe | Corporate Actions                    │
│ Announcement Timestamp | Point-in-Time Fundamentals                     │
│ Raw Snapshot | Parquet | SHA256 | Dataset Manifest                      │
│                                                                         │
│ 数据覆盖率由每次 Runtime Audit 实测，不在架构中写死百分比                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# ⚡ 下一阶段执行路线

## Stage 1 — Production Baseline Freeze
保持当前已经稳定的 DAILY PIT、canonical runtime、sandbox、production model 与执行基础设施冻结。
必须持续满足：
- Research Write Access → Production Model = DENIED
- Production Snapshot Before == Production Snapshot After
- Working Tree = CLEAN
- Code Freeze SHA = VERIFIED

## Stage 2 — Alpha Discovery
重点探索具有不同经济来源、尽量低相关的新信号（基本面质量、成长加速度、业绩超预期度、残差动量、流动性冲击、非线性交互等）。
必须分别报告 Standalone RankIC, ICIR, Positive IC Ratio, Turnover, Cost-adjusted Portfolio Alpha，严禁以因子堆叠代替真 Alpha。

## Stage 3 — Label Redesign
建立版本化 Label Registry (LABEL_V1 ~ LABEL_V5)，在完全相同的 Walk-Forward 样本上进行公平比较。

## Stage 4 — Seed Variance Root-Cause Diagnosis
系统诊断当前 Seed RankIC STD = 0.007994 > 0.0050 的根因究竟来自 Data, Feature Selection, Specific Fold, 还是 Sampling。输出 seed × fold, seed × year, seed × regime 等诊断矩阵。

## Stage 5 — Ensemble Candidate Experiment
DoubleEnsemble 作为候选方案之一，与 Regularized LightGBM, Ridge, Seed Averaging, Feature Bagging 等横向比对，严禁在 OOS 收益未增的情况下仅因通过方差门禁而晋级。

## Stage 6 — Certified Research
Certified 模式必须 Fail-Closed。0 因子通过时判定 INSUFFICIENT_SIGNAL -> FAIL，严禁强制补足 Top-K 充当科研证据。

## Stage 7 — Candidate Promotion
严格按 Candidate Registry -> Promotion Review -> Immutable Production Registry 流水线晋级。

## Stage 8 — Shadow → Live
MiniQMT 网络异常时停止下单、冻结状态并与 Broker 对账，严禁自动静默切换至 PaperBroker。

---

# 🎯 系统权威状态矩阵 (Four Canonical System Statuses)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. 软件工程完整性: ENGINEERING_VALIDATED                                    │
│    - 本地与 CI 自动化测试 100% 通过 (Pass)                                  │
│    - 数据集哈希校验一致 (tools/check_committed_dataset_schema.py PASS)      │
│    - 认证与研究元数据防篡改校验 100% 通过 (PASS)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. 科研实证结论:   RESEARCH_INCONCLUSIVE                                    │
│    - 缺少财报公告披露时点逐笔因果存证 (INSUFFICIENT_EVIDENCE)               │
│    - 多随机种子方差未收敛至严苛阈值，Bootstrap 95% CI 下界 <= 0            │
│    - 综合科研认证结论为未通过 (NOT_CERTIFIED)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. 前瞻观察成熟度: PROSPECTIVE_IMMATURE                                     │
│    - 真实样本外前瞻观察累计天数 < 20 个交易日 (PROSPECTIVE_IMMATURE)        │
│    - 严禁回填历史或虚构未来观察，必须经受真实时序考验                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. 实盘交易状态:   LIVE_TRADING_BLOCKED                                     │
│    - 核心配置硬门禁: LIVE_TRADING_READY = False                             │
│    - Broker 接口与调度器 Fail-Closed 物理阻断任何真实资金下单               │
└─────────────────────────────────────────────────────────────────────────────┘
```

> [!WARNING]
> **【PRODUCTION 模型概念边界澄清】**
> - `ModelRegistry` 中的 `PRODUCTION` 标签仅代表**已打包规范化的部署工程制品 (`DEPLOYMENT_ARTIFACT`)**，用于驱动每日 Paper Trading 仿真与看板观察。
> - **`PRODUCTION` 状态绝不等于科研认证通过 (`RESEARCH_CERTIFIED`)，亦绝不等于实盘交易批准 (`LIVE_APPROVED`)**。
> - 本项目永久默认切断实盘下单能力：`LIVE_TRADING_READY = False`。
