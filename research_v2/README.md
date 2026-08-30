# Phase 2.1+ V2 Research Architecture & Roadmap
# A股新一代科学投研、单自变量因果优化与组合对齐路线图

本命名空间 `research_v2/` 承载 Phase 2.1 及后续所有量化预测与组合架构研发。

---

## 0. 核心科学准则 (Core Scientific Principles)

1. **`LEGACY_BASELINE_V1` 永恒不变**: 任何新实验必须与已冻结的 Legacy V1 进行 OOS 对照。
2. **`ONE EXPERIMENT = ONE PRIMARY CHANGE`**: 严格单变量因果实验，禁止同时修改多维要素。
3. **`EXECUTION-ALIGNED GROUND TRUTH`**: 标签计算与撮合执行必须在时序与价格维度 100% 对齐。

---

## 1. 投研推进路线图 (Research Roadmap)

| 阶段代号 | 阶段名称 | 核心自变量 (Primary Change) | 预期交付物 |
| :--- | :--- | :--- | :--- |
| **Phase 2.1-0** | **Legacy Truth Freeze + Registry** | 冻结历史基准，搭建 V2 基础设施 | `LEGACY_BASELINE_V1`, `research_v2/` 骨架 |
| **Phase 2.1-A** | **Execution-Aligned Labels** | 修正标签时序脱节 (`T+1 Open -> T+21 Open`) | 实盘执行对齐新标签矩阵与 OOS 基准对比 |
| **Phase 2.1-B** | **True LambdaRank** | 真正实现 NDCG 排序损失目标函数 | `true_lambdarank_certified = True` 实证对比 |
| **Phase 2.1-C** | **Versioned Feature Sets** | 结构化拆分量价、波动、流动性与基本面 | 版本化特征集 A/B 实验 |
| **Phase 2.1-D** | **Fundamental A/B Testing** | PIT 财报因子因果贡献独立实证 | 验证财报真实 Alpha 增益与稳健性 |
| **Phase 2.1-E** | **Market / Industry Context** | 截面市场环境与行业轮动条件特征 | 宏观/截面 Regime 特征矩阵 |
| **Phase 2.1-F** | **Stable Feature Selection** | 消除 Fold 间特征漂移，稳健特征选择 | 跨 Fold 稳定特征子集 |
| **Phase 2.2-A** | **Multi-Task Model Experts** | 多任务分类/回归/排序专家混合 | MoE 混合专家预测框架 |
| **Phase 2.2-B** | **Confidence / No-View Filter** | 预测置信度与低胜率自适应弃权 | 动态自适应开仓过滤器 |
| **Phase 2.2-C** | **Alpha-to-Portfolio Optimization** | 风险平价、换手惩罚与组合优化 | 交易端最终 Alpha 实盘就绪认证 |
