# A股量化系统 · 本次回测真实性认证证书 (RUNTIME_ATTESTATION)

> **运行时实例 ID**: `DIRTY_WORKTREE_run_be94b0f8`  
> **认证类型**: `CURRENT_RUNTIME_ATTESTATION`  
> **执行代码 Commit (CODE_COMMIT_SHA)**: `5d2c96c7bc826be2ae841944ca919eeca5c3c075`  
> **构建产物归档类型 (ARTIFACT_STORAGE)**: `BUILD_ARTIFACT / REPOSITORY_GENERATED_OUTPUT`  
> **启动时源码纯净状态 (RUNTIME_START_SOURCE_DIRTY)**: `True`  
> **外部信任根锚定状态 (TRUST_ROOT_VERIFIED)**: `False`  
> **认证评估时间**: 2026-08-29 16:50:35  
> **本次回测可信度总评级**: **`HIGH_RISK`**  
> **认证判定机制**: `backtest.audit.CertificationPolicy` (全要素 Fail-Closed 判定)

---

## 1. 运行时全要素真实性检验清单

| 序号 | 运行时要素 | 实际运行状态 | 认证门禁 | 运行时具体证据与度量 |
| :--- | :--- | :---: | :---: | :--- |
| 1 | **外部信任根锚定 (External Trust Root)** | `UNPINNED_OR_TAMPERED` | ⚠️ ATTENTION / FAIL | 来源: UNPINNED_SELF_TRUST, 注册表哈希: c67a05171e4ab5f9..., 外部锚定: none... |
| 2 | **股票池时点覆盖 (PIT Universe)** | `COMPLETE` | ⚠️ ATTENTION / FAIL | 模式: PIT_INCOMPLETE, 来源类别: UNKNOWN, 原始证据校验: False, 幸存者风险: True |
| 3 | **真实数据源 (No Synthetic)** | `REAL_DATA_VERIFIED` | ✅ PASS | 数据源: akshare, 分布: {'akshare': 3, 'local_csv': 0, 'synthetic': 0} |
| 4 | **交易所官方交易日历** | `THIRD_PARTY/FALLBACK` | ⚠️ ATTENTION / FAIL | 日历来源: akshare_sina, 质量评级: third_party |
| 5 | **历史逐日 ST 时间线** | `LIMITED_STATIC` | ⚠️ ATTENTION / FAIL | 标的覆盖: 0.0%, 未知行数: 0, 偏差风险: True |
| 6 | **公司行为除权除息覆盖** | `LIMITED_COVERAGE` | ⚠️ ATTENTION / FAIL | 覆盖率: 0.0%, 调整可用: False, 数据源: custom_corporate_actions |
| 7 | **前复权因果安全性 (PIT Safe)** | `UNKNOWN_OR_UNVERIFIED` | ⚠️ ATTENTION / FAIL | 复权模式: unknown |
| 8 | **特征与行情缓存指纹校验** | `VERIFIED` | ✅ PASS | 原始血缘保持: True, 版本: 3.1 |
| 9 | **基准指数时间轴完整性** | `0.0%` | ⚠️ ATTENTION / FAIL | 缺失日历天数: 0, 数据源: unknown |
| 10 | **委托订单数量守恒** | `CONSERVED` | ✅ PASS | 部分成交: 9, 撤单: 0, 延期: 0 |
| 11 | **运行时防伪数字信封与配置哈希** | `UNTRUSTED_OR_TAMPERED` | ⚠️ ATTENTION / FAIL | 信封校验: False, 配置指纹: a6cf893fe1f64c33275911e35e2656e5808182642e38898bc218720d60182a76 |

---

## 2. 门禁评估与缺失证据诊断

### ⚠️ 未完全满足的认证门禁项:
- `external_trust_root_unverified`
- `runtime_config_hash_unverified`
- `universe_manifest_hash_unverified`
- `factor_manifest_hash_missing`
- `market_manifest_hash_missing`
- `manifest_chain_unverified`
- `universe_provenance_unverified`
- `universe_raw_evidence_unverified`
- `universe_dataset_hash_unverified`
- `universe_source_class_ineligible_for_production_UNKNOWN`
- `survivorship_bias_risk_present`
- `historical_st_coverage_incomplete`
- `historical_st_bias_risk_present`
- `corporate_action_missing_adjustment_or_zero_event_proof`
- `corporate_action_coverage_incomplete`
- `corporate_action_bias_risk_present`
- `corporate_action_provenance_unverified`
- `adjustment_not_point_in_time_safe`
- `future_adjustment_leakage_test_not_passed`
- `benchmark_source_unverified`
- `benchmark_coverage_ratio_less_than_100pct`
- `calendar_not_exchange_official`
- `unregistered_runtime_signing_key_DEV_UNTRUSTED_KEY`
- `runtime_started_from_dirty_source_code`
- `current_source_code_dirty`

---

## 3. 评级定义与解释说明

- **`VERIFIED` (最高真实性等级)**: 
  - 必须具备完整的官方/持牌 Raw Evidence 原始证据、Trust Anchor Ed25519 签名与 SHA256 密码级哈希链；
  - 必须具备合法的 RuntimeAttestationEnvelope 数字信封且通过当前 Commit 强绑定校验；
  - 必须 100% 通过无未来函数、订单数量守恒与交易所官方日历校验；
  - 幸存者偏差完全清零。
- **`CONTROLLED_WITH_LIMITATIONS` (受限受控等级)**: 
  - 代码具备完整量化与风控逻辑，但运行时使用了部分第三方公开接口或静态成分股子集。
- **`HIGH_RISK` (高风险提示)**: 
  - 缺少可信的 PIT 原始证据、数字信封无效、Commit 不匹配或存在幸存者偏差风险。系统严格遵循科学诚信，绝不粉饰。
