# A股量化系统 · 本次回测真实性认证证书 (RUNTIME_ATTESTATION)

> **运行时实例 ID**: `run_96cc62c1`  
> **认证评估时间**: 2026-08-29 14:07:52  
> **本次回测可信度总评级**: **`HIGH_RISK`**  
> **认证判定机制**: `backtest.audit.CertificationPolicy` (全要素 Fail-Closed 判定)

---

## 1. 运行时全要素真实性检验清单

| 序号 | 运行时要素 | 实际运行状态 | 认证门禁 | 运行时具体证据与度量 |
| :--- | :--- | :---: | :---: | :--- |
| 1 | **股票池时点覆盖 (PIT Universe)** | `INCOMPLETE` | ⚠️ ATTENTION / FAIL | 模式: STATIC, 幸存者风险: True, 证明源: False |
| 2 | **真实数据源 (No Synthetic)** | `REAL_DATA` | ✅ PASS | 数据源: akshare, 分布: {'akshare': 301, 'local_csv': 0, 'synthetic': 0} |
| 3 | **交易所官方交易日历** | `THIRD_PARTY/FALLBACK` | ⚠️ ATTENTION / FAIL | 日历来源: akshare_sina, 质量评级: third_party |
| 4 | **历史逐日 ST 时间线** | `LIMITED_STATIC` | ⚠️ ATTENTION / FAIL | 标的覆盖: 0.0%, 偏差风险: True |
| 5 | **公司行为除权除息覆盖** | `LIMITED_COVERAGE` | ⚠️ ATTENTION / FAIL | 覆盖率: 0.0%, 数据源: custom_corporate_actions |
| 6 | **前复权因果安全性 (PIT Safe)** | `UNKNOWN_OR_UNVERIFIED` | ⚠️ ATTENTION / FAIL | 复权模式: unknown |
| 7 | **特征与行情缓存指纹校验** | `VERIFIED` | ✅ PASS | 原始血缘保持: True, 版本: 3.0 |
| 8 | **基准指数时间轴完整性** | `0.0%` | ⚠️ ATTENTION / FAIL | 缺失日历天数: 0, 数据源: unknown |
| 9 | **委托订单数量守恒** | `CONSERVED` | ✅ PASS | 部分成交: 79, 撤单: 0, 延期: 0 |
| 10 | **逐日截面行业中性化** | `FULL` | ✅ PASS | 中性化天数比例: 100.0%, 均值覆盖率: 73.4% |

---

## 2. 门禁诊断与可信度结论

### ⚠️ 未完全满足的认证门禁项:
- `universe_coverage_incomplete`
- `universe_provenance_unverified`
- `survivorship_bias_risk_present`
- `historical_st_coverage_incomplete`
- `historical_st_bias_risk_present`
- `corporate_action_coverage_incomplete`
- `corporate_action_bias_risk_present`
- `adjustment_not_point_in_time_safe`
- `future_adjustment_leakage_test_not_passed`
- `benchmark_coverage_ratio_less_than_100pct`
- `calendar_not_exchange_official`

---

## 3. 评级使用与投产指导

- **`VERIFIED`**: 本次回测所用数据为真实官方数据、日历经过交易所认证、股票池具备完整时点无幸存者偏差、前复权安全且订单完全守恒，回测收益与胜率可作为真实投资决策依据。
- **`CONTROLLED_WITH_LIMITATIONS`**: 本次回测在受控环境下完成，但存在明确局限（如使用了离线静态股票池或第三方交易日历），收益率仅供策略研发对比参考。
- **`HIGH_RISK`**: 本次回测存在重大偏差风险（如仿真数据、未来信息泄露或状态不守恒），严禁用于实盘投资参考。
