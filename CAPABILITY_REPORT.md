# A股量化系统 · 代码静态能力认证报告 (CAPABILITY_REPORT)

> **报告版本**: Release v8.0.0-ProvenanceHarden  
> **生成时间**: 2026-08-29 18:34:32  
> **测试环境**: Python 3.11.9 (win32)  
> **数据血缘架构**: 严格分离【代码静态能力证明】与【生产运行时真实性认证】

---

## 1. 核心量化与数据真实性能力检验矩阵

| 序号 | 代码能力类别 | 静态验证状态 | 证明测试节点 (evidence_test nodeid) | 核心组件 (runtime_component) | 能力实现与证明逻辑 |
| :--- | :--- | :---: | :--- | :--- | :--- |
| 1 | **Point-In-Time 股票池回放与幸存者偏差消除** | `PROVEN_BY_TEST` | `tests/test_evidence_integrity.py::test_four_year_pit_pipeline_e2e_integration` | `PointInTimeUniverseProvider, DataManager` | 支持时点成分股逐日回放与基线快照校验，非成分股样本禁止参与截面排名与买入 |
| 2 | **生产级 PIT E2E 全链路集成能力 (DataManager Fixture 验证)** | `PROVEN_BY_TEST` | `tests/test_evidence_integrity.py::test_true_production_pit_e2e` | `DataManager.sync_and_build_dataset` | DataManager E2E fixture/integration capability proof: 证明流水线集成与时点数据构建能力 |
| 3 | **前复权因果时序与双快照时不变性 (QFQ Safety)** | `PROVEN_BY_TEST` | `tests/test_evidence_integrity.py::test_qfq_pit_safety_dual_snapshot_invariance` | `FactorProcessor, AlphaCalculator` | 证明在除权前后拉取的截面特征在历史日期具有因果不变性，杜绝未来调整因子泄露 |
| 4 | **逐日截面行业与对数流通市值中性化** | `PROVEN_BY_TEST` | `tests/test_trading_rules.py::test_neutralization_switches_per_date` | `FactorProcessor.neutralize_cross_section` | 仅在成分股样本上 OLS 剥离行业哑变量与市值，支持动态降级与空股票池隔离 |
| 5 | **停牌股维持 last_price 估值与超期标记** | `PROVEN_BY_TEST` | `tests/test_trading_rules.py::test_stale_price_warning_metrics` | `BacktestEngine.mark_to_market` | 停牌期间持仓维持上一次有效成交价估值，不刷新 last_price_date 并审计超期事件 |
| 6 | **全市场统一涨跌幅规则引擎 (PriceLimitRuleEngine)** | `PROVEN_BY_TEST` | `tests/test_evidence_integrity.py::test_price_limit_rule_engine_comprehensive` | `PriceLimitRuleEngine` | 涵盖主板 10%、创业板 2020-08-24 注册制 20%、科创板 20%、北交所 30% 及 ST 5% 唯一规则源 |
| 7 | **公司行为覆盖证明与分批委托订单守恒** | `PROVEN_BY_TEST` | `tests/test_evidence_integrity.py::test_corporate_action_split_partial_fill_conservation` | `CorporateActionProvider, BacktestEngine` | 送转股等比缩放挂单与持仓，严格保证 Q_req = Q_filled + Q_rem + Q_canc 数量守恒 |
| 8 | **现金分红移动止损基准动态保护** | `PROVEN_BY_TEST` | `tests/test_evidence_integrity.py::test_cash_dividend_trailing_stop_protection` | `BacktestEngine._apply_corporate_actions` | 除息日按分红金额同步下调持仓最高价基准，杜绝除息跳空缺口虚假触发移动止损 |
| 9 | **基准指数多股票跨截面未来信息泄漏防御** | `PROVEN_BY_TEST` | `tests/test_evidence_integrity.py::test_benchmark_cross_symbol_leakage_rejected` | `DataManager.sync_and_build_dataset` | 禁止在多股票截面 merge 后进行全局 ffill，杜绝跨标的基准行情未来污染 |
| 10 | **特征缓存完整流式 SHA256 指纹校验** | `PROVEN_BY_TEST` | `tests/test_evidence_integrity.py::test_factor_cache_streaming_sha256_rejection` | `FactorProcessor, DataManager` | 采用 pd.util.hash_pandas_object 流式计算全表指纹，中间数据微小篡改立即失效 |
| 11 | **审计元数据防篡改拦截与评级门禁 (CertificationPolicy)** | `PROVEN_BY_TEST` | `tests/test_evidence_integrity.py::test_certification_policy_truth_table` | `CertificationPolicy, AuditCollector` | 拦截针对 CERTIFICATION_FIELDS 的外部 override，评级严格由全要素门禁推导 |
| 12 | **数据血缘与 Raw CSV 防冒充官方校验 (Anti-Impersonation)** | `PROVEN_BY_TEST` | `tests/test_adversarial_certification.py::test_arbitrary_raw_csv_not_automatically_official` | `ProvenanceVerifier, SourceClass` | 非官方 CSV 即使具有正确哈希与格式，缺少受信任注册表登记一律拒绝官方认证 |
| 13 | **测试 Fixture 严禁进入生产认证 (Test Fixture Demotion)** | `PROVEN_BY_TEST` | `tests/test_adversarial_certification.py::test_test_fixture_provider_never_production_verified` | `ProvenanceVerifier, PointInTimeUniverseProvider` | 标记为 TEST_FIXTURE 的数据源绝对禁止进入生产 VERIFIED 认证 |
| 14 | **Manifest 自我声明布尔值无效与反伪造防御 (Anti-Self-Certification)** | `PROVEN_BY_TEST` | `tests/test_adversarial_certification.py::test_fake_source_metadata_boolean_cannot_self_certify` | `ProvenanceVerifier, CertificationPolicy` | 元数据自行写死 verified=True 无法绕过检查，必须通过 Trust Anchor 与物理证据闭环 |
| 15 | **ST 缺失显式判定为 UNKNOWN 门禁 (ST Explicit Unknown Gate)** | `PROVEN_BY_TEST` | `tests/test_trading_rules.py::test_missing_historical_st_is_explicitly_unknown` | `DataManager, CertificationPolicy` | 缺失历史 ST 状态时严格标记为 UNKNOWN 并触发偏差风险门禁，拒绝乐观假定非 ST |

---

## 2. 测试套件质量与覆盖率结论

- **自动化测试套件**: 全量 pytest 测试执行完毕，所有核心约束均具备正例与反例自动化用例。
- **能力覆盖率**: 15/15 项核心量化架构能力已通过自动化测试严格证明。
- **测试分类标识**:
  - `@pytest.mark.unit`: 纯组件单元测试（撮合、计算、规则）
  - `@pytest.mark.integration`: 多组件集成与数据流测试
  - `@pytest.mark.external`: 外部数据接口与离线回放测试
