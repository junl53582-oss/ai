"""
Stage 5 Verification Test Suite: System Rebranding & Live Readiness Declaration
Validates:
1. System title and branding across README.md, ARCHITECTURE.md, and dashboard/app.py.
2. The 4 canonical system statuses:
   - ENGINEERING_VALIDATED
   - RESEARCH_INCONCLUSIVE
   - PROSPECTIVE_IMMATURE
   - LIVE_TRADING_BLOCKED
3. Complete eradication of unproven marketing terms.
4. Unambiguous declaration that PRODUCTION in ModelRegistry is a DEPLOYMENT_ARTIFACT and not live trading clearance.
5. Absolute enforcement of LIVE_TRADING_READY = False.
"""
from pathlib import Path
import pytest
from config.settings import settings


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parent.parent


def test_stage5_readme_rebranding(repo_root):
    readme_path = repo_root / "README.md"
    assert readme_path.exists(), "README.md must exist"
    content = readme_path.read_text(encoding="utf-8")

    # 1. Main Title
    assert "# A股量化研究与观察系统" in content, "README title must be 'A股量化研究与观察系统'"
    assert "A-Share Quantitative Research & Observation System" in content

    # 2. Four Canonical Statuses
    assert "ENGINEERING_VALIDATED" in content
    assert "RESEARCH_INCONCLUSIVE" in content
    assert "PROSPECTIVE_IMMATURE" in content
    assert "LIVE_TRADING_BLOCKED" in content

    # 3. Live Trading Hard Block Declaration
    assert "LIVE_TRADING_READY = FALSE" in content or "LIVE_TRADING_READY = False" in content

    # 4. Production Artifact Clarification
    assert "DEPLOYMENT_ARTIFACT" in content

    # 5. Eradication of 5-day price projection marketing claims
    assert "未来 5 日价格走势前瞻: K 线前瞻预测虚线" not in content
    assert "未来 5 日价格预期走势 + 90% 置信区间" not in content


def test_stage5_architecture_rebranding(repo_root):
    arch_path = repo_root / "ARCHITECTURE.md"
    assert arch_path.exists(), "ARCHITECTURE.md must exist"
    content = arch_path.read_text(encoding="utf-8")

    # 1. Title
    assert "# A股量化研究与观察系统架构规范" in content

    # 2. Four Canonical Statuses
    assert "ENGINEERING_VALIDATED" in content
    assert "RESEARCH_INCONCLUSIVE" in content
    assert "PROSPECTIVE_IMMATURE" in content
    assert "LIVE_TRADING_BLOCKED" in content

    # 3. Production Model Concept Clarification
    assert "DEPLOYMENT_ARTIFACT" in content
    assert "LIVE_TRADING_READY = False" in content


def test_stage5_dashboard_rebranding_and_warnings(repo_root):
    dashboard_path = repo_root / "dashboard" / "app.py"
    assert dashboard_path.exists(), "dashboard/app.py must exist"
    content = dashboard_path.read_text(encoding="utf-8")

    # 1. Page title and header
    assert 'page_title="A股量化研究与观察系统"' in content
    assert "🔬 A股量化研究与观察系统 (Quantitative Research & Observation System)" in content

    # 2. Four Canonical Statuses
    assert "ENGINEERING_VALIDATED" in content
    assert "RESEARCH_INCONCLUSIVE" in content
    assert "PROSPECTIVE_IMMATURE" in content
    assert "LIVE_TRADING_BLOCKED" in content

    # 3. Live trading blocked declaration
    assert "LIVE_TRADING_READY = False" in content or "LIVE_TRADING_READY=False" in content

    # 4. Unproven marketing terms removed
    assert "实战顶级确定性单" not in content
    assert "95.0% 满仓进攻" not in content


def test_stage5_live_trading_gate_hard_pinned():
    assert settings.LIVE_TRADING_READY is False, "LIVE_TRADING_READY must be strictly False"
    assert settings.LIVE_TRADING_STATUS == "LOCKED", "LIVE_TRADING_STATUS must be 'LOCKED'"
