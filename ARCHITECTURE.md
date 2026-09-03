# A股多因子 AI 量化投研、生产预测与实盘闭环架构 (Release v9.0.0)

## 🏛️ 八层量化闭环全景架构

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 8: 真实资金安全与券商网关层                                       │
│ Capital Safety & MiniQMT Gateway                                        │
│                                                                         │
│ Paper → Shadow → Live Promotion                                         │
│ 资金利用率硬顶 | 单股仓位上限 | 日换手限制 | 回撤熔断 | Kill Switch      │
│ MiniQMT 断线：停止下单 → 状态冻结 → Broker 对账 → 人工/规则恢复          │
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

# 🎯 当前系统正确状态

```text
INFRASTRUCTURE_STATUS       = VERIFIED
RESEARCH_RUNTIME_STATUS     = OPERATIONAL
PRODUCTION_ISOLATION        = PASS
WALKFORWARD_PURGE           = PASS
ARTIFACT_INTEGRITY          = PASS
MULTI_SEED_ROBUSTNESS       = FAIL
ROBUST_MODEL_IMPROVEMENT    = MIXED_EVIDENCE_NOT_ROBUST
FINAL_HOLDOUT_AVAILABLE     = FALSE
CERTIFICATION_STATUS        = NOT_CERTIFIED
LIVE_TRADING_STATUS         = LOCKED
```

在全部严苛证据满足前：
LIVE_TRADING_READY = FALSE
