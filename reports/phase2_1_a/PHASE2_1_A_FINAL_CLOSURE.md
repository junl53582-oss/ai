# Phase 2.1-A — Final Closure Evidence Document
# 实盘执行对齐标签受控 A/B 研究 — 最终封箱证据入库文档

> **最终科学判定 (Certified Scientific Verdict)**: **`NO_IMPROVEMENT`**  
> **研究阶段状态 (Phase Lifecycle Status)**: **`CLOSED` (正式封箱)**  
> **实盘许可与模型准入守卫**: `LIVE_TRADING_READY = FALSE`, `PRODUCTION_MODEL_PROMOTION = FALSE`

---

## 1. 完整 Git 提交血缘链 (4-Tier Commit Provenance)

| 提交层级 | Commit SHA | 类型与说明 |
| :--- | :--- | :--- |
| **Initial r2 Code Commit** | [`8c9a2a84191945a7e7a91b99f840d392fdfae567`](https://github.com/junl53582-oss/ai/commit/8c9a2a84191945a7e7a91b99f840d392fdfae567) | 代码级基准：固定参数 `scale_pos_weight=1.0`，修复生产模型路径与隔离 |
| **Initial r2 Evidence Commit** | [`bfda1a7e404521d21f8801fa05d8b53c433ceb57`](https://github.com/junl53582-oss/ai/commit/bfda1a7e404521d21f8801fa05d8b53c433ceb57) | 认证实验证据：从 clean code commit `8c9a2a8` 生成的正式实证数据与 Manifest |
| **Final Guard Hotfix Commit** | [`04ab30a1044edbfa44c8cb12fa7ee7576e402c4f`](https://github.com/junl53582-oss/ai/commit/04ab30a1044edbfa44c8cb12fa7ee7576e402c4f) | 防护加固：生产模型隔离测试全面迁移至 100% `tmp_path` 模拟，绝不触碰真实目录 |
| **Final Evidence Micro-Patch Code** | [`135cdf016b2d74aef0346f9a25595b8489d5c21a`](https://github.com/junl53582-oss/ai/commit/135cdf016b2d74aef0346f9a25595b8489d5c21a) | 证据微补丁：`git ls-remote` 真实远程 SHA 强校验，修复 porcelain 前导空格解析，真实临时 Git Repo 单测闭环 |

---

## 2. 冻结的已认证核心科研实证指标 (Frozen Certified Scientific Metrics)

> **科研真实性声明**: 本实验在严格受控的单变量条件下执行（唯一自变量为训练标签：Legacy vs Execution-Aligned，特征集、模型参数、种子 42、时序划分完全冻结）。以下指标保持历史不可逆冻结，未在后续补丁中重新训练或修改。

| 核心评估指标 | Arm A (Legacy Label) | Arm B (Execution-Aligned Label) | 差异 (Delta B - A) |
| :--- | :---: | :---: | :---: |
| **Mean Daily OOS RankIC** | **0.045205** | **0.043682** | **-0.001524** |
| **NW20 RankICIR (年化)** | **0.348391** | **0.352015** | **+0.003624** |
| **RankIC > 0 交易日占比** | 60.30% | 60.03% | -0.27% |
| **Q5-Q1 年化超额收益差 (pct points)** | 3.31 pts | 5.16 pts | +1.85 pts |
| **Top 10% 20日平均真实执行净超额** | 0.3900% | 0.5054% | +0.1154% |
| **分组单调性得分** | 0.7000 | 0.4000 | -0.3000 |

### 统计检验与 Fold 胜率分布：
- **20-Day Paired Block Bootstrap (2,000 Resamples)**:
  - Mean RankIC Delta: **`-0.001524`**
  - 95% 置信区间 (95% CI): `[-0.011169, +0.009657]`
  - 提升概率 $P(	ext{Delta} > 0)$: **`33.25%`**
  - 统计显著提升 (`ci_lower > 0`): **`False`**
- **Fold-Level 胜率 (已排除 0 样本无效折)**:
  - 滚动折总数: `20` 折
  - 有效对比折数: `19` 折
  - 排除无效折数: `1` 折 (Fold 20: 数据集尾部前瞻持有期不足，共同评估样本为 0)
  - **Execution Arm 胜出折 (9 折)**: Fold 1, Fold 4, Fold 6, Fold 8, Fold 10, Fold 14, Fold 15, Fold 16, Fold 19
  - **Legacy Arm 胜出折 (10 折)**: Fold 2, Fold 3, Fold 5, Fold 7, Fold 9, Fold 11, Fold 12, Fold 13, Fold 17, Fold 18
  - **有效 Fold 胜率 (Fold Win Ratio)**: **`47.37%`** (9 / 19)

---

## 3. 客观科学结论与局限性 (Scientific Conclusion & Limitations)

1. **科学结论**: **`NO_IMPROVEMENT`**
   - 在 LightGBM 二分类模型与当前 79 个特征因子体系下，将训练标签由 Legacy (收盘价-收盘价未扣费) 切换为 Execution-Aligned (T+1 开盘买入、T+21 开盘卖出、扣除费率与税费并处理停牌/涨跌停延期退出) 后，样本外平均日频 RankIC 从 `0.045205` 微降至 `0.043682`（Delta = `-0.001524`）。
   - 实证表明：**“更符合实盘交易规则的标签”并不必然等同于“更容易被树分类模型拟合出更高的全截面秩相关排序能力”**。
2. **局限性与风险提示**:
   - **未证明实盘盈利**: 本实验仅评估标签对 OOS 预测能力的影响，`LIVE_TRADING_READY = FALSE`，严禁投入实盘。
   - **生产模型未晋升**: 真实生产模型 `saved_models/latest_lightgbm.pkl` 在全流程中保持物理隔离与 SHA256 不变，`PRODUCTION_MODEL_PROMOTION = FALSE`。
   - **测试与 CI 状态明确区分**: 本地自动化测试全量通过 (`LOCAL_TEST_STATUS = PASS (409 passed)`)；远程 GitHub 未挂载 Actions 流程 (`GITHUB_CI_STATUS = NO_STATUS / NOT_RUN`)。

---

## 4. 大文件证据与 Git 治理声明 (Large Artifact Governance)

- `NO_NEW_LARGE_BINARY_COMMITTED_IN_R2 = TRUE`: r2 及后续补丁中未向 Git 仓库提交任何大型二进制文件。
- `CURRENT_TREE_COMMON_OOS_TRACKED = FALSE`: `common_execution_oos.parquet` (~10.3MB) 处于 `local_not_git_tracked` 状态，由 `.gitignore` 规则有效拦截并由 Manifest 记录哈希。
- `CURRENT_TREE_EXPERIMENT_PKLS_TRACKED = FALSE`: 实验训练的模型 `.pkl` 文件同样由 `.gitignore` 排除。
- `HISTORICAL_DB6D_LARGE_ARTIFACT_REMAINS = TRUE`: 恪守科研与工程诚信，不使用 `filter-repo`/`rebase` 重写 Git 历史，历史 commit `db6d8ec` 中的大文件作为不可逆历史事实保留。
- `HISTORY_REWRITE_PERFORMED = FALSE`。

---

## 5. 下一阶段研发建议 (Next Phase Recommendation)

- **Phase 2.1-A 状态**: **`PHASE_2_1_A_FINAL_CLOSURE = CLOSED`**
- **建议开启研发路线**:
  - **`Phase 2.1-B — Model Objective Study`**
  - 在维持执行对齐标签作为统一真实评价基准的前提下，重点探索排序学习损失函数（如 LightGBM LambdaRank）、回归目标或非对称损失函数对实盘收益特征的拟合能力。
