# A股多因子量化选股与实盘级回测系统 · 终极认证与能力总览 (Release v8.0.0)

> **生成时间**: 2026-08-29 15:29:42  
> **架构设计**: 严格分离【代码静态能力证明】与【生产运行时真实性认证】

本系统采用双报告认证体系：

1. 📖 **[CAPABILITY_REPORT.md](./CAPABILITY_REPORT.md)**
   - **核心职责**: 回答“代码具备什么能力”。
   - **证据来源**: 基于全量自动化测试套件 (`artifacts/pytest.xml`) 动态推导。
   - **当前状态**: 已收集 **182** 个测试用例，通过 **182** 个。

2. 🛡️ **[RUNTIME_ATTESTATION.md](./RUNTIME_ATTESTATION.md)**
   - **核心职责**: 回答“本次具体回测实际上证明了什么”。
   - **证据来源**: 基于本次回测运行的 AuditMetadata (`artifacts/runtime_audit.json`)、RuntimeAttestationEnvelope 防伪数字信封与数据血缘推导。
   - **当前评级**: **`HIGH_RISK`**

---
*注：任何 VERIFIED 认证必须来自真实运行产物与 Raw Evidence，严禁任何自我声明与硬编码结论。*
