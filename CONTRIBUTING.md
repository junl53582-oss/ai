# 贡献指南 (Contributing Guide)

感谢您对本项目（A股工业级量化投研与自动化交易中台）的关注与贡献。本项目涉及严肃的金融时序预测、风险控制与资金执行，所有代码变更均须遵循最高标准的金融因果性与软件工程规范。

---

## 1. 核心研发与合规准则 (Zero-Tolerance Gates)

在提交任何代码或数据前，请务必遵守以下铁律：

1. **绝对禁止未来函数与前视偏差 (No Lookahead Bias)**:
   - 所有横截面特征必须严格按 `T` 日收盘点位对齐，不得引用 `T` 日之后的行情或公告。
   - 财务基本面因子必须采用 Point-In-Time (PIT) 实际披露日对齐（`FUNDAMENTAL_DELAY_DAYS >= 110`）。
   - 走步训练必须设置 `PURGE_GAP_DAYS >= LABEL_HORIZON`，严禁样本外标签泄露。
2. **绝对禁止伪造或篡改科研证据 (Immutable Evidence)**:
   - 严禁手工伪造回测胜率、IC/ICIR 指标或夏普比率。
   - 严禁篡改已封签的历史认证报告与 Manifest 哈希。
   - 生产模型必须通过完整的五级生命周期状态机晋升（`RESEARCH → CANDIDATE → APPROVED → PRODUCTION`）。
3. **真实可交易性约束 (Realistic Trading Constraints)**:
   - 严格执行 A股 `T+1` 卖出限制、100股整手向下取整、一字涨停禁买、一字跌停禁卖、ST 5% 涨跌幅限制与双边滑点及印花税扣除。
4. **敏感信息与安全合规 (Security First)**:
   - 严禁提交任何 `.env` 文件、真实 API 密钥、群机器人 Webhook URL、真实券商账户信息或用户私人路径。

---

## 2. 本地开发与复现工作流

### 2.1 环境准备
```bash
# 推荐使用 Python 3.11
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装轻量 CPU 版 PyTorch (若有 GPU 可自行安装 CUDA 版)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 安装全量依赖
pip install -r requirements.txt
```

### 2.2 运行测试套件
在提交任何修改前，必须确保本地所有测试全量通过：
```bash
# 运行生产加固与密码学审计门禁
python -m pytest -v tests/test_audit_hardening.py
python -m pytest -v tests/test_r3_2_evidence_integrity.py
python -m pytest -v tests/test_formal_research_runner_e2e.py

# 运行生产模型适配器回归测试
python -m pytest -v tests/test_production_hardening.py

# 运行全量测试套件 (568 项)
python -m pytest -q
```

---

## 3. 分支与 Pull Request 规范

1. **分支命名**:
   - 特性分支：`feature/your-feature-name`
   - 修复分支：`fix/your-fix-name`
   - 架构/加固：`chore/your-task-name` 或 `codex/your-task-name`
   - **禁止直接向 `main` 分支 push**。
2. **Commit 规范**:
   遵循 Conventional Commits：
   - `feat(...)`: 新功能或新因子
   - `fix(...)`: 修复缺陷
   - `test(...)`: 新增或优化测试用例
   - `docs(...)`: 文档更新
   - `chore(...)`: 依赖或工程脚本维护
3. **Pull Request 提交**:
   - 填写 PR 模板中的变更说明与测试结果。
   - 确保 GitHub Actions CI（`Fast CI` 与 `Audit Hardening Certification`）全部绿标通过。
