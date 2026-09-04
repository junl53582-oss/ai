# 数据与模型制品获取说明 (Data & Model Artifacts Guide)

本文档详细说明本项目中涉及的数据集、特征矩阵以及模型制品的物理分布、获取方式与可复现构建方法。

---

## 1. 数据集架构与入库策略

为了兼顾代码仓库的轻量性（< 100MB 快速 clone）与量化因果溯源的严谨性，项目采取分层存储策略：

| 数据集路径 | 数据类型 | 体积 | Git 状态 | 作用与来源说明 |
| :--- | :--- | :---: | :---: | :--- |
| `data_storage/fundamentals/yjbb_*.parquet` | PIT 季度财务报表 | ~29 MB | **已入库 (Git Tracked)** | 覆盖 2018~2026 季度财报，包含真实公告日期，用于基本面因子 PIT 严格因果对齐 |
| `data_storage/research/market_daily_300_v2.parquet` | 沪深300 日线行情 | ~20 MB | **已入库 (Git Tracked)** | 覆盖 2021~2026 官方复核行情，修复对数市值血缘污染，作为核心测试基线 |
| `data_storage/universe_pit_events.parquet` | 历史成分股事件流 | ~20 KB | **已入库 (Git Tracked)** | 沪深300 每半年指数成分股调入/调出时点事件流，彻底消除幸存者偏差 |
| `data_storage/parquet/market_daily.parquet` | CI 极简行情快照 | ~300 KB | **已入库 (Git Tracked)** | 5 标的快速测试专用快照，供 GitHub Actions 快速冒烟测试 |
| `data_storage/factors/factor_matrix.parquet` | CI 极简因子矩阵 | ~800 KB | **已入库 (Git Tracked)** | 5 标的快速测试专用矩阵，供 CI Smoke Research 验证 |
| `data_storage/research/factor_matrix_300.parquet` | 全量科研级因子矩阵 | ~311 MB | **本地缓存 (Git Ignored)** | 包含 300 标的、126 维特征的完整面板。因体积超限不在普通 Git 仓库中直接保存 |

### 1.1 全量因子矩阵的本地确定性重建
若需要在本地重现完整的 300 标的科研回测，无需下载 311MB 文件，可在安装依赖后直接执行：
```bash
# 从已入库的 market_daily_300_v2.parquet 确定性构建 factor_matrix_300.parquet
python tools/build_dataset_300_v2.py
```
构建完成后，系统会自动生成并校验 `data_storage/research/factor_matrix_300.manifest.json` 中的 SHA-256 指纹。

---

## 2. 生产模型制品体系

### 2.1 生产模型标准目录 (`saved_models/production/`)
经量化委员会签署并上线到生产环境的模型存放在独立目录中：
```text
saved_models/production/m_20260903_194757_hybrid_bagging_ridge/
├── model.pkl        # 二进制模型制品 (~94 KB)
├── metadata.json    # 生产元数据快照 (含训练提交哈希、数据集指纹、审批证据)
└── manifest.json    # 防篡改清单 (包含 SHA-256 与 32 维特征列表)
```
- **查看现役生产模型**:
  ```python
  from models.registry import ModelRegistry
  reg = ModelRegistry()
  prod = reg.get_production()
  print(f"当前生产模型: {prod.model_id} (类型: {prod.model_type})")
  ```

### 2.2 历史过渡模型归档 (`saved_models/legacy/`)
根目录历史遗留的 `.pkl` 文件已归档至 `saved_models/legacy/`：
- **历史混淆提示**: `saved_models/latest_lightgbm.pkl` 实为 Gen 4 深度强化学习模型 (`DRLStrengthenedQuantModel`)，仅保留以保证已有回归测试通过。
- 生产推理必须使用 `BatchInference()` 统一加载，严禁直接依赖根目录下的硬编码文件名。

---

## 3. 防篡改哈希与合规门禁校验

本项目所有数据集与模型均配备配套的 `.manifest.json` 密码学凭证文件。运行以下命令可一键验证全库数据与模型完整性：
```bash
# 验证数据集 Schema 与分布完整性
python -u tools/check_committed_dataset_schema.py

# 验证认证报告与物理制品指纹自洽
python -u tools/validate_certification_artifacts.py

# 运行生产加固回归测试
python -m pytest -v tests/test_production_hardening.py
```
若出现任何文件被意外篡改或字节漂移，系统将自动触发 fail-closed 阻断运行。
