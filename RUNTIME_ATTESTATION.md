# A股量化系统 · 本次回测真实性认证证书 (RUNTIME_ATTESTATION)

> **运行时实例 ID**: `run_96cc62c1`  
> **认证评估时间**: 2026-08-29 14:08:55  
> **本次回测可信度总评级**: **`VERIFIED`**  
> **认证判定机制**: `backtest.audit.CertificationPolicy` (全要素 Fail-Closed 判定)

---

## 1. 运行时全要素真实性检验清单

| 序号 | 运行时要素 | 实际运行状态 | 认证门禁 | 运行时具体证据与度量 |
| :--- | :--- | :---: | :---: | :--- |
| 1 | **股票池时点覆盖 (PIT Universe)** | `COMPLETE` | ✅ PASS | 模式: POINT_IN_TIME, 幸存者风险: False, 证明源: True |
| 2 | **真实数据源 (No Synthetic)** | `REAL_DATA` | ✅ PASS | 数据源: akshare, 分布: {'akshare': 301, 'local_csv': 0, 'synthetic': 0} |
| 3 | **交易所官方交易日历** | `OFFICIAL` | ✅ PASS | 日历来源: sse_szse_official, 质量评级: official |
| 4 | **历史逐日 ST 时间线** | `COVERED` | ✅ PASS | 标的覆盖: 100.0%, 偏差风险: False |
| 5 | **公司行为除权除息覆盖** | `COVERED` | ✅ PASS | 覆盖率: 100.0%, 数据源: official_financial_announcements |
| 6 | **前复权因果安全性 (PIT Safe)** | `PIT_SAFE` | ✅ PASS | 复权模式: point_in_time_forward_adjusted |
| 7 | **特征与行情缓存指纹校验** | `VERIFIED` | ✅ PASS | 原始血缘保持: True, 版本: 3.0 |
| 8 | **基准指数时间轴完整性** | `100% COVERED` | ✅ PASS | 缺失日历天数: 0, 数据源: csi_official_index |
| 9 | **委托订单数量守恒** | `CONSERVED` | ✅ PASS | 部分成交: 79, 撤单: 0, 延期: 0 |
| 10 | **逐日截面行业中性化** | `FULL` | ✅ PASS | 中性化天数比例: 100.0%, 均值覆盖率: 73.4% |

---

## 2. 门禁诊断与可信度结论

### ✅ 全要素认证门禁检验全部通过！

---

## 3. 评级使用与投产指导

- **`VERIFIED`**: 本次回测所用数据为真实官方数据、日历经过交易所认证、股票池具备完整时点无幸存者偏差、前复权安全且订单完全守恒，回测收益与胜率可作为真实投资决策依据。
- **`CONTROLLED_WITH_LIMITATIONS`**: 本次回测在受控环境下完成，但存在明确局限（如使用了离线静态股票池或第三方交易日历），收益率仅供策略研发对比参考。
- **`HIGH_RISK`**: 本次回测存在重大偏差风险（如仿真数据、未来信息泄露或状态不守恒），严禁用于实盘投资参考。
