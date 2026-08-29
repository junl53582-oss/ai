# A股量化系统 · 本次回测真实性认证证书 (RUNTIME_ATTESTATION)

> **运行时实例 ID**: `run_96cc62c1`  
> **认证评估时间**: 2026-08-29 14:45:44  
> **本次回测可信度总评级**: **`HIGH_RISK`**  
> **认证判定机制**: `backtest.audit.CertificationPolicy` (全要素 Fail-Closed 判定)

---

## 1. 运行时全要素真实性检验清单

| 序号 | 运行时要素 | 实际运行状态 | 认证门禁 | 运行时具体证据与度量 |
| :--- | :--- | :---: | :---: | :--- |
| 1 | **股票池时点覆盖 (PIT Universe)** | `COMPLETE` | ⚠️ ATTENTION / FAIL | 模式: POINT_IN_TIME, 来源类别: UNKNOWN, 原始证据校验: False, 幸存者风险: False |
| 2 | **真实数据源 (No Synthetic)** | `REAL_DATA` | ✅ PASS | 数据源: akshare, 分布: {'akshare': 301, 'local_csv': 0, 'synthetic': 0} |
| 3 | **交易所官方交易日历** | `OFFICIAL` | ✅ PASS | 日历来源: sse_szse_official, 质量评级: official |
| 4 | **历史逐日 ST 时间线** | `LIMITED_STATIC` | ⚠️ ATTENTION / FAIL | 标的覆盖: 100.0%, 未知行数: 462844, 偏差风险: False |
| 5 | **公司行为除权除息覆盖** | `LIMITED_COVERAGE` | ⚠️ ATTENTION / FAIL | 覆盖率: 100.0%, 调整可用: False, 数据源: official_financial_announcements |
| 6 | **前复权因果安全性 (PIT Safe)** | `PIT_SAFE` | ✅ PASS | 复权模式: point_in_time_forward_adjusted |
| 7 | **特征与行情缓存指纹校验** | `VERIFIED` | ✅ PASS | 原始血缘保持: True, 版本: 3.0 |
| 8 | **基准指数时间轴完整性** | `100% COVERED` | ✅ PASS | 缺失日历天数: 0, 数据源: csi_official_index |
| 9 | **委托订单数量守恒** | `CONSERVED` | ✅ PASS | 部分成交: 79, 撤单: 0, 延期: 0 |
| 10 | **逐日截面行业中性化** | `FULL` | ✅ PASS | 中性化天数比例: 100.0%, 均值覆盖率: 73.4% |

---

## 2. 门禁评估与缺失证据诊断

### ⚠️ 未完全满足的认证门禁项:
- `universe_manifest_hash_missing`
- `factor_manifest_hash_missing`
- `market_manifest_hash_missing`
- `universe_raw_evidence_unverified`
- `universe_dataset_hash_unverified`
- `universe_source_class_ineligible_for_production_UNKNOWN`
- `st_unknown_rows_462844_inconsistent_with_complete_coverage`
- `corporate_action_missing_adjustment_or_zero_event_proof`

---

## 3. 评级定义与解释说明

- **`VERIFIED` (最高真实性等级)**: 
  - 必须具备完整的官方/持牌 Raw Evidence 原始证据与 SHA256 密码级哈希链；
  - 必须 100% 通过无未来函数、订单数量守恒与交易所官方日历校验；
  - 幸存者偏差完全清零。
- **`CONTROLLED_WITH_LIMITATIONS` (受限受控等级)**: 
  - 代码具备完整量化与风控逻辑，但运行时使用了部分第三方公开接口或静态成分股子集。
- **`HIGH_RISK` (高风险提示)**: 
  - 缺少可信的 PIT 原始证据或存在幸存者偏差风险。系统严格遵循科学诚信，绝不粉饰。
