# 🔬 第二轮【机构级量化系统深度复核审计报告】
## (SECOND-LEVEL DEEP QUANT SYSTEM AUDIT REPORT)

> **审计标的仓库**: `https://github.com/junl53582-oss/ai`  
> **审计师团队**: Senior Quant Engineer + ML Systems Engineer + Quantitative Software Auditor  
> **审计基准**: 零盲信已有报告、全源码穿透分析、测试有效性甄别、实盘交易与模型状态机闭环  
> **审计时间**: 2026-09-04  

---

## 📊 总评分数卡 (System Scorecard)

| 审计维度 | 得分 (0-100) | 评级 | 核心诊断结论 |
| :--- | :---: | :---: | :--- |
| **Engineering (工程架构与质量)** | **92 / 100** | A | 架构分层高度清晰，全平台类型标注完备，Windows OpenMP 崩溃防御到位；存在少量重名定义代码异味。 |
| **Data Integrity (数据无泄漏与PIT安全)** | **95 / 100** | A+ | 财报严格基于公告日与 backward asof 对齐，后复权事件累计因子历史不变，动态成分股消除幸存者偏差。 |
| **Model Research (科研可信度与防过拟合)** | **88 / 100** | B+ | 25天 Purged 时序隔离严格拦截标签渗透；DoubleEnsemble 表现稳健，但 DRL 为纯手写策略梯度实验性质。 |
| **Backtest (回测仿真真实度)** | **92 / 100** | A | T+1、涨跌停、停牌与 5% 成交量容量限制 100% 进入主流程；滑点为线性 0.1%，未引入非线性平方根冲击。 |
| **Production Safety (生产安全与实盘隔离)** | **89 / 100** | B+ | 7 重资金防御中枢可靠；但发现生产推理引擎存在模型类型不兼容与定时任务默认研发重训的架构漏洞。 |

---

## 第一阶段：测试体系质量审查 (Test Suite Quality Audit)

针对 `tests/` 目录下全部 46 个测试文件、556 项测试函数进行 AST 语法树解析与断言深度剖析。

### 1. 556 项测试的本质结构诊断：A 还是 B？

**诊断判定: 混合结构 (Core Quant ~51.3% + Cryptographic/Governance ~48.7%)**

- **并不是纯粹的接口测试数量虚高**: 核心量化逻辑包含 **285 个实质性测试**，对微观撮合、T+1 锁定、涨跌停挂单、动态税费切换、送转股最高价除权、Purged Walk-Forward 边界等进行了大量真实的 DataFrame 计算与极端边界断言。
- **但测试总数确实存在“存证认证层膨胀”**: 556 个测试中有 **271 个测试 (占比 48.7%)** 集中在红队对抗渗透测试（`test_adversarial_certification.py` 单文件即有 124 项测试）、Ed25519 签名有效性、Manifest SHA-256 哈希防篡改与 Git 工作区干净度检查。这部分属于高规格软件安全审计，而非金融量化 Alpha 逻辑测试。

### 2. 测试模块分布与质量覆盖矩阵

| 测试模块 | 对应测试文件数 | 测试数量 | 覆盖能力评价 | 潜在风险与测试盲区 |
| :--- | :---: | :---: | :--- | :--- |
| **PIT 数据** | 5 个文件 | **50** | **深度覆盖**: 覆盖后复权累计因子、公告日延迟、ST 历史 Forward-As-Of、除息事件流。 | 仅覆盖单股与 300 标的历史，缺少全市场 5000 只股票极端缺失值压力。 |
| **Label 标签** | 3 个文件 | **24** | **精准覆盖**: 覆盖交易日历对齐、超额基准扣除、持有期边界扰动不变性。 | 缺少极端熊市熔断日连续停牌导致的标签截断样本测试。 |
| **Feature 因子** | 7 个文件 | **65** | **广泛覆盖**: 覆盖 Alpha158、A股定制特征、另类因子、逐日截面 OLS 行业市值中性化。 | 部分另类因子（如舆情特征）在离线测试中为 mock 生成，未跑全历史清洗。 |
| **Model 模型** | 11 个文件 | **76** | **核心覆盖**: 覆盖时序滚动边界、LightGBM 随机种子传播、双塔注意力网络前向与因果掩码。 | 缺少模型在线持续学习与跨不同牛熊体制迁移泛化能力的压力测试。 |
| **Backtest 回测** | 4 个文件 | **57** | **深度覆盖**: 覆盖 T+1 锁定、一字涨停禁买、跌停顺延、动态税费、5% 流动性成交量上限。 | 滑点测试仅验证了线性扣除，未测试订单超日成交 20% 时的非线性冲击。 |
| **Execution 执行** | 4 个文件 | **13** | **扎实覆盖**: 覆盖 7 重资金风控防线、先卖后买、价格偏离、MiniQMT 离线断线冻结。 | 缺少实盘委托柜台回报延迟（如网络丢包、部分成交超时撤单）的异步并发测试。 |
| **Registry 与治理** | 12 个文件 | **271** | **极度重度**: 覆盖 Ed25519 签名、Manifest 链、Git 干净度、对抗性红队攻击（70+种伪造）。 | 治理规则极严苛，导致正常数据轻微重构即可触发断言阻断（如当前 P1 报错）。 |

---

## 第二阶段：核心量化逻辑重新审计

### 1. PIT 数据安全审计 (`data/`)

- **财务公告日期真实使用**: [data/fundamentals.py:190-205](file:///c:/Users/lin/Documents/股票预测/data/fundamentals.py#L190-L205) 优先使用东方财富 `stock_yjbb_em` 的「最新公告日期」（`announcement_date`），缺失时 fallback 到保守的季度后 110 天估计。
- **报告期 (report_date) 穿越检查**: [data/fundamentals.py:259-266](file:///c:/Users/lin/Documents/股票预测/data/fundamentals.py#L259-L266) 使用 `pd.merge_asof(daily, eff_keep, left_on="date", right_on="effective_date", by="symbol", direction="backward")`。严格采用向后匹配（`direction="backward"`），$t$ 日行情只能匹配到披露日 $\le t$ 的财报，**无 report_date 穿越**。
- **股票池每日复原与幸存者偏差**: [data/universe_provider.py:131-260](file:///c:/Users/lin/Documents/股票预测/data/universe_provider.py#L131-L260) `PointInTimeUniverseProvider` 以 2021-09-29 官方 300 标的为基线，逐日根据调入/调出事件流更新成分股。全周期回测提取的是各期成分股全集（UNION 349,379 行），不在成分股期间的行通过 `in_universe == False` 标记隔离，**无幸存者偏差**。
- **复权数据未来污染**: [data/adjustment_factors.py:15-22, 91-114](file:///c:/Users/lin/Documents/股票预测/data/adjustment_factors.py#L15-L22) 采用新浪后复权事件表累计因子。历史 $t$ 日价格由 $t$ 日及之前的除权事件决定，未来新增除权事件不改变历史已算出的复权价，**无未来污染**。

#### **PIT 综合判定**: **PASS (绿色通过)**

---

### 2. Feature Engineering 特征工程审计 (`factors/`)

- **数据来源与时间截断**: [factors/alpha158.py](file:///c:/Users/lin/Documents/股票预测/factors/alpha158.py) 与 [factors/custom_ashare.py](file:///c:/Users/lin/Documents/股票预测/factors/custom_ashare.py)。
- **Rolling 窗口与未来数据排查**:
  - 全文检索 `factors/` 目录，**零处使用 `shift(-k)`**；
  - 全文检索 `rolling(` 窗口，**零处使用 `center=True`**；
  - 所有滚动统计量（如 MA, STD, ROC, ATR）均基于严格向前滑动窗口。
- **全市场统计泄漏审查**:
  - 截面中性化 [factors/processor.py:34-143](file:///c:/Users/lin/Documents/股票预测/factors/processor.py#L34-L143) 函数 `_neutralize_one_day` 仅在单个交易日内对截面样本执行 OLS 回归。
  - 回归自变量的行业 Dummy 与对数流通市值（`LOG_CIRC_MV`）仅在当日成分股样本内计算，残差映射到全集。
  - 因子标准化不使用跨日全局均值方差，**无时序截面统计泄漏**。

---

### 3. Label 标签工程审计 (`models/labeler.py`)

- **Label Horizon**: 预测未来 20 个交易日（`LABEL_HORIZON = 20`）。
- **Trading Calendar**: [models/labeler.py:59-82](file:///c:/Users/lin/Documents/股票预测/models/labeler.py#L59-L82) 强制要求全市场标准真实交易日历（Canonical Exchange Calendar），以交易所实际开市日滚动对齐，停牌缺行通过真实日历映射，杜绝了单纯 `shift(-20)` 跨越数月停牌的时间弹性失真。
- **Benchmark Neutral**: [models/labeler.py:105-120](file:///c:/Users/lin/Documents/股票预测/models/labeler.py#L105-L120) 强制依赖 `benchmark_close` 计算超额收益率：
  $$\text{Excess Return} = \left(\frac{P_{t+20}}{P_t} - 1\right) - \left(\frac{B_{t+20}}{B_t} - 1\right)$$
  若数据表缺失 `benchmark_close`，直接抛出 `ValueError` (Fail-Closed)，禁止将未对冲的大盘绝对收益伪装成 Alpha。
- **Execution Alignment**: $t$ 日闭市生成的标签预测的是从 $t$ 日到 $t+20$ 日的价格变动，回测与实盘在 $t+1$ 日开盘以执行价撮合，逻辑闭环。

#### **Label 可信度评分**: **95 / 100**

---

## 第三阶段：模型科研真实性审查 (`models/`)

### 1. 时序隔离与防泄漏链路 (Train $\to$ Validation $\to$ OOS Test)
- **代码位置**: [models/walk_forward.py:220-278](file:///c:/Users/lin/Documents/股票预测/models/walk_forward.py#L220-L278)
- **Purge 隔离执行**:
  - 训练集取 `[train_start, val_start - 25]`（剔除末尾 25 天）；
  - 验证集取 `[val_start, test_start - 25]`（剔除末尾 25 天）；
  - 测试集取 `[test_start, test_end]`。
  - 由于标签持有期为 20 天，25 天的 Purged Gap 彻底阻断了训练样本向验证集、验证样本向测试集的未来标签跨时段渗透。
- **运行时物理防御**: [models/walk_forward.py:261-278](file:///c:/Users/lin/Documents/股票预测/models/walk_forward.py#L261-L278) 设置了显式 Invariant 运行时断言，一旦 `train_max_date >= val_min_date` 或 `val_max_date >= test_min_date` 直接抛出 `RuntimeError` 熔断。

### 2. 模型矩阵科研可信度深度剖析

| 模型名称 | 代码路径 | 代码实现质量 | 科研可信度 | 生产就绪建议 |
| :--- | :--- | :---: | :---: | :--- |
| **LightGBM** | `models/lightgbm_model.py` | **高** | **高** | 正则化参数强（`lambda_l1=10, lambda_l2=20`），早停 80 轮，适合作为稳健 Baseline。 |
| **Double Ensemble** | `models/double_ensemble.py` | **高** | **高** | 论文对齐规范。5 子模型抽取 75% 特征，基于样本预测残差做 Loss 动态加权（高残差样本平滑聚焦），验证集 Softmax 融合。 |
| **Gen5 DeepRank** | `models/gen5_deep_rank_model.py` | **中高** | **中高** | 逐日截面百分位均匀排序归一化，双塔门控交叉注意力，对看涨但实际暴跌样本赋予 3.0x 惩罚，思路合理。 |
| **DRL Agent** | `models/reinforcement_agent.py` | **中** | **中低 (实验级)** | 基于纯手写单层 numpy MLP 的 REINFORCE with Baseline 算法，金融时序噪声极大导致策略梯度方差极高，且缺乏 Target Network 稳定性机制。**不宜直接接管百万级实盘资金**。 |

---

## 第四阶段：回测可信度压力测试 (`backtest/`)

审计核查了极端交易场景在回测主链路中的真实拦截情况：

1. **涨停无法买入**: [backtest/engine.py:520-534](file:///c:/Users/lin/Documents/股票预测/backtest/engine.py#L520-L534) 在开盘撮合买单时，调用 `rules.can_buy(row, exec_price)`，若一字涨停或涨停封死，订单直接判定为 `REJECTED` 并计入废单统计。**（已进入主流程）**
2. **跌停无法卖出**: [backtest/engine.py:356-368](file:///c:/Users/lin/Documents/股票预测/backtest/engine.py#L356-L368) 在开盘撮合卖单时，调用 `rules.can_sell(row, pos, exec_price)`，若一字跌停，订单状态置为 `DEFERRED` 并重新放回 `pending_orders` 顺延至次日撮合。**（已进入主流程）**
3. **停牌拦截与估值保持**: [backtest/engine.py:158-175, 349, 367](file:///c:/Users/lin/Documents/股票预测/backtest/engine.py#L158-L175) 停牌股票买卖均被拦截；盯市估值维持停牌前收盘价且不刷新日期，停牌超过 60 天触发 Stale Price 预警指标。**（已进入主流程）**
4. **流动性容量上限 (5% 日成交量)**: [backtest/engine.py:323-330, 381-395, 539-550](file:///c:/Users/lin/Documents/股票预测/backtest/engine.py#L323-L330) 全系统共享单日标的成交量上限（`day_vol * 0.05`），超量部分截断成交并保留余量顺延，杜绝了无限流动性作弊。**（已进入主流程）**
5. **资金冲击成本**: 滑点采用双边 0.1% 线性扣除。对于日常换手 20% 以内的大中盘蓝筹股足够保守，但在极端小微盘股单日建仓超 500 万元时未体现平方根冲击上升。

#### **Backtest Reality Score**: **92 / 100**

---

## 第五阶段：生产链路与治理审查 (`execution/`, `scheduler/`)

经过深入源代码排查，发现了三处**关键生产链路断层与架构不一致**：

### 1. 致命断层：生产推理引擎不支持已审批的生产模型类型 (P0)
- **源码实测**:
  - 在 `saved_models/registry/index.json` 中，当前唯一处于 `PRODUCTION` 状态的模型记录为:
    `m_20260903_194757_hybrid_bagging_ridge` (model_type: `hybrid_bagging_ridge`)。
  - 但在生产推理引擎 [models/inference.py:71-76](file:///c:/Users/lin/Documents/股票预测/models/inference.py#L71-L76) 中：
    ```python
    if self.record.model_type in ("lightgbm", "lightgbm_ranker", "lightgbm_reg", "regression", "ranking", "classification"):
        from models.lightgbm_model import LightGBMQuantModel
        task_type = self.record.task_type or "classification"
        m = LightGBMQuantModel(task_type=task_type)
        return m.load(artifact)
    raise InferenceError(f"暂不支持推理的模型类型: {self.record.model_type}")
    ```
  - **实机运行验证**: 导入 `BatchInference()` 即刻崩溃并抛出：
    `InferenceError: 暂不支持推理的模型类型: hybrid_bagging_ridge`。
  - **影响**: 生产推理路径在当前状态下**完全不可用**。

### 2. 生产调度绕过注册表，退化为每日重跑 11 折研发走步 (P1)
- **源码审查**:
  - 调度器主函数 [scheduler/daily_runner.py:207-208](file:///c:/Users/lin/Documents/股票预测/scheduler/daily_runner.py#L207-L208) 参数默认值为 `--mode research`。
  - Windows 定时任务批处理脚本 [scripts/run_daily_post_market.bat:21](file:///c:/Users/lin/Documents/股票预测/scripts/run_daily_post_market.bat#L21) 执行命令为:
    `"%PY311%" -u scheduler/daily_runner.py --optimizer risk_parity %WEBHOOK_ARG%`
    **未携带 `--mode inference` 参数**。
  - **影响**: 每日 15:05 定时触发时，系统实际是在现场重跑整个 11 折 Walk-Forward 滚动训练，然后取最后一折的样本外打分充当信号（即 Research Replay），完全绕过了精心构建的 ModelRegistry 治理体系。

### 3. 模型制品物理文件命名混淆 (P1)
- **源码实测**: 检查磁盘根目录下的生产模型文件 `saved_models/latest_lightgbm.pkl`。
- **Python 类型透视**:
  ```python
  obj = joblib.load('saved_models/latest_lightgbm.pkl')
  print(type(obj)) # 输出: <class 'models.drl_strengthened_model.DRLStrengthenedQuantModel'>
  ```
- **影响**: 文件名标明为 `latest_lightgbm.pkl`，但其实体对象为 `DRLStrengthenedQuantModel`。依赖此文件名的历史工具（如 `tools/predict_stocks.py`）存在隐性类型错位风险。

---

## 第六阶段：Shadow Trading 就绪评估

| 评估项 | 现状与证据 | 判定 |
| :--- | :--- | :---: |
| **Production Model** | 注册表中生产模型为 `hybrid_bagging_ridge`，但推理引擎无法加载抛错。 | **FAIL** |
| **Forward Validation** | 历史 Walk-Forward 已固化，但缺少注册表生产模型的实时前瞻跟踪序列。 | **WARNING** |
| **Paper Trading** | `PaperBroker` 沙盒已就绪，支持先卖后买与 T+1，代码测试通过。 | **PASS** |
| **Monitoring 看板** | `dashboard/app.py` 7 大主题大屏已在后台 8501 端口稳定常驻运行 (`HTTP 200 OK`)。 | **PASS** |
| **Drift Detection** | 数据集哈希与 Git commit 漂移检测函数已编写，但受制于推理引擎断层未能每日激活。 | **WARNING** |

### **Shadow Trading 最终就绪判定**: **NOT READY (有条件未就绪)**
*必须先修复推理引擎对生产模型类型的加载适配，并将自动化定时脚本切换为真实的 `--mode inference`。*

---

## 第七阶段：发现的问题清单 (P0 / P1 / P2)

### P0 (阻断实盘与生产推理，必须立即修复)

1. **生产推理引擎模型类型不兼容崩溃**
   - **所在文件与行号**: [models/inference.py:71-76](file:///c:/Users/lin/Documents/股票预测/models/inference.py#L71-L76)
   - **缺陷描述**: 注册表中已批准的生产模型类型为 `hybrid_bagging_ridge`，但 `BatchInference._load_model()` 仅硬编码支持 `lightgbm` 系列，导致生产推理实例化直接抛出 `InferenceError` 崩溃。
   - **修复建议代码**:
     ```python
     # models/inference.py 中扩展支持 hybrid_bagging_ridge 与 通用 joblib 加载
     if self.record.model_type in ("lightgbm", "lightgbm_ranker", "lightgbm_reg", "regression", "ranking", "classification"):
         from models.lightgbm_model import LightGBMQuantModel
         task_type = self.record.task_type or "classification"
         m = LightGBMQuantModel(task_type=task_type)
         return m.load(artifact)
     elif self.record.model_type in ("hybrid_bagging_ridge", "ensemble", "drl_strengthened"):
         import joblib
         return joblib.load(artifact)
     ```

---

### P1 (高风险，上线实盘前必须修复)

1. **生产因子矩阵数据清单指纹未同步**
   - **所在文件与行号**: [data_storage/research/factor_matrix_300.manifest.json:17](file:///c:/Users/lin/Documents/股票预测/data_storage/research/factor_matrix_300.manifest.json#L17)
   - **缺陷描述**: 物理文件 `factor_matrix_300.parquet` 的实际 SHA-256 为 `ad672d9cf58597566024f83ade5e5dfc9e4efde8604a7268f66bd3c7c16b4bb1`，但清单中记录的是旧值 `9a882c4568d662ab15220992989b6bd2d2042222469d9059ab33a68c882a4a42`，导致自动化测试第 600 行报错。
   - **修复建议**: 将 `manifest.json` 中 `file_sha256` 更新为物理文件的真实哈希。

2. **每日盘后自动化调度脚本未配置生产推理模式**
   - **所在文件与行号**: [scripts/run_daily_post_market.bat:21](file:///c:/Users/lin/Documents/股票预测/scripts/run_daily_post_market.bat#L21) 与 [scheduler/daily_runner.py:207](file:///c:/Users/lin/Documents/股票预测/scheduler/daily_runner.py#L207)
   - **缺陷描述**: 定时任务默认运行在 `research` 模式下，每日重跑 11 折走步训练冒充生产推理，不仅耗时长，且每日信号来自不同的模型快照，无法归因。
   - **修复建议**: 在 `run_daily_post_market.bat` 第 21 行加上 `--mode inference`，并将 `daily_runner.py` 中的默认参数更改为 `default="inference"`。

3. **模型制品文件命名与实际加载对象类型冲突**
   - **所在文件**: `saved_models/latest_lightgbm.pkl`
   - **缺陷描述**: 文件名包含 `lightgbm`，但实际 pickle 内容为 `DRLStrengthenedQuantModel` 对象，导致依赖该文件名的历史预测工具产生隐性类型混乱。
   - **修复建议**: 规范命名为 `latest_production_model.pkl` 或完全通过 `ModelRegistry` 解析路径，淘汰物理文件名硬编码。

---

### P2 (中低风险，后续优化项)

1. **`factors/processor.py` 中的重复函数头代码异味**
   - **所在文件与行号**: [factors/processor.py:403-418](file:///c:/Users/lin/Documents/股票预测/factors/processor.py#L403-L418)
   - **缺陷描述**: 存在一段未写完的 `def build_and_save_factor_matrix` 重复定义头部，虽在第 436 行被完整函数覆盖，但属于无用死代码。
   - **修复建议**: 移除第 403-418 行的重复无用函数头。

2. **回测冲击成本模型升级**
   - **所在文件与行号**: [config/settings.py:206](file:///c:/Users/lin/Documents/股票预测/config/settings.py#L206)
   - **优化建议**: 引入非线性平方根滑点模型 $\Delta P \propto \sigma \sqrt{Q / V}$，以更精准地评估大资金调仓冲击。

---

## 🏁 最终审计结论与评审决策

### 综合评审结论:
**【选项 B: 工程成熟，但需要完成生产模型推理适配与继续 Alpha 实盘仿真验证】**

### 深度定性总结:
1. **量化底层极为硬核**: 本系统的底层金融工程质量堪称开源界的标杆——无论是 PIT 公告日向后对齐、新浪 HFQ 事件表、动态成分股 Union、交易所日历对齐，还是 25 天 Purged 隔离窗口与回测中对 T+1、涨跌停、5% 成交量上限的严防死守，**均经过了代码级的严格落实，不存在低级作弊或未来函数**。
2. **测试并非全为业务虚高**: 285 个量化业务测试真实覆盖了所有交易与数据微观细节，另 271 个测试构建了一套近乎金融军工级的数字签名防篡改屏障。
3. **关键阻断点在于“生产最后一公里”**: 现阶段无法直接进入 Shadow Trading 的核心原因，是模型注册表与推理引擎之间在模型类型解析上的不兼容（`InferenceError: hybrid_bagging_ridge`），导致每日调度任务仍然被迫留在 `research` 走步重训模式。只需完成上述 P0 与 P1 修复，系统即可完全具备进入真实 Shadow Trading 的完备条件。
