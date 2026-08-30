"""
Targeted Git Provenance and Working Tree Tests for Phase 2.1-A (tests/test_phase2_1_a_git_provenance.py)
Covering:
1. Real temporary Git repository end-to-end integration tests (clean, dirty, untracked, ignored, staged, leading space).
2. True remote SHA (git ls-remote) fail-closed parsing and validation.
3. Provenance gate fail-closed tests.
"""
import subprocess
import pytest
from pathlib import Path

from tools.run_phase2_1_a_label_ab import (
    _git_working_tree_clean,
    _git_true_remote_sha,
    _validate_source_provenance,
)


def _init_test_git_repo(repo_dir: Path) -> Path:
    """初始化一个用于测试的临时本地 Git 仓库"""
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "TestUser"], cwd=repo_dir, capture_output=True, check=True)
    
    # 提交基础文件
    readme = repo_dir / "README.md"
    readme.write_text("# Test Repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=repo_dir, capture_output=True, check=True)
    return repo_dir


# Case A: Clean Repo
def test_tmp_git_repo_clean_working_tree_passes(tmp_path):
    repo = _init_test_git_repo(tmp_path / "repo_clean")
    is_clean, dirty = _git_working_tree_clean(repo)
    assert is_clean is True
    assert dirty == []


# Case B: Dirty Tracked File (Unstaged)
def test_tmp_git_repo_dirty_tracked_file_fails(tmp_path):
    repo = _init_test_git_repo(tmp_path / "repo_dirty")
    readme = repo / "README.md"
    readme.write_text("# Modified Content\n", encoding="utf-8")
    
    is_clean, dirty = _git_working_tree_clean(repo)
    assert is_clean is False
    assert len(dirty) == 1
    assert any("README.md" in item for item in dirty)


# Case C: Untracked Source File
def test_tmp_git_repo_untracked_source_file_fails(tmp_path):
    repo = _init_test_git_repo(tmp_path / "repo_untracked")
    src_dir = repo / "research_v2"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "new_algo.py").write_text("print('test')\n", encoding="utf-8")
    
    is_clean, dirty = _git_working_tree_clean(repo)
    assert is_clean is False
    assert len(dirty) == 1
    assert any("UNTRACKED: research_v2/new_algo.py" in item for item in dirty)


# Case D: Ignored Runtime Artifact (via .gitignore)
def test_tmp_git_repo_ignored_runtime_artifact_passes(tmp_path):
    repo = _init_test_git_repo(tmp_path / "repo_ignored")
    gitignore = repo / ".gitignore"
    gitignore.write_text("runtime/\n*.parquet\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "add gitignore"], cwd=repo, capture_output=True, check=True)
    
    runtime_dir = repo / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "common_execution_oos.parquet").write_bytes(b"PARQUET_TEST_DATA")
    (repo / "data.parquet").write_bytes(b"PARQUET_TEST_DATA")
    
    is_clean, dirty = _git_working_tree_clean(repo)
    assert is_clean is True
    assert dirty == []


# Case E: Staged File
def test_tmp_git_repo_staged_file_fails(tmp_path):
    repo = _init_test_git_repo(tmp_path / "repo_staged")
    readme = repo / "README.md"
    readme.write_text("# Staged Change\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, capture_output=True, check=True)
    
    is_clean, dirty = _git_working_tree_clean(repo)
    assert is_clean is False
    assert any("MODIFIED/STAGED" in item for item in dirty)


# Case F: Porcelain Leading Space Preserved (Unstaged modification produces ' M')
def test_tmp_git_repo_porcelain_leading_space_preserved(tmp_path):
    repo = _init_test_git_repo(tmp_path / "repo_space")
    readme = repo / "README.md"
    readme.write_text("# Unstaged Space Test\n", encoding="utf-8")
    
    # 验证 git status --porcelain 输出首字符为空格
    raw_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    assert raw_status.startswith(" M ") or raw_status.startswith("M  ")
    
    is_clean, dirty = _git_working_tree_clean(repo)
    assert is_clean is False
    assert any("README.md" in item for item in dirty)
    # 确保没有因错误 strip 导致路径首字母被吞噬为 "EADME.md"
    assert not any("EADME.md" in item and "README.md" not in item for item in dirty)


# True Remote SHA Unit Tests
def test_git_true_remote_sha_parses_valid_ls_remote(monkeypatch):
    sample_sha = "0123456789abcdef0123456789abcdef01234567"
    mock_output = f"{sample_sha}\trefs/heads/phase2.1-a-exec-labels\n"
    
    class MockCompletedProcess:
        stdout = mock_output
        returncode = 0
        
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: MockCompletedProcess())
    sha = _git_true_remote_sha("phase2.1-a-exec-labels")
    assert sha == sample_sha


def test_git_true_remote_sha_fails_closed_on_empty(monkeypatch):
    class MockCompletedProcess:
        stdout = ""
        returncode = 0
        
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: MockCompletedProcess())
    sha = _git_true_remote_sha("phase2.1-a-exec-labels")
    assert sha == "UNKNOWN"


def test_git_true_remote_sha_fails_closed_on_subprocess_error(monkeypatch):
    def mock_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd="git ls-remote")
        
    monkeypatch.setattr(subprocess, "run", mock_run)
    sha = _git_true_remote_sha("phase2.1-a-exec-labels")
    assert sha == "UNKNOWN"


def test_git_true_remote_sha_fails_closed_on_invalid_sha(monkeypatch):
    class MockCompletedProcess:
        stdout = "not_a_valid_40_hex_sha\trefs/heads/phase2.1-a-exec-labels\n"
        returncode = 0
        
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: MockCompletedProcess())
    sha = _git_true_remote_sha("phase2.1-a-exec-labels")
    assert sha == "UNKNOWN"


# Provenance Gate Validation Tests
def test_source_provenance_clean_and_true_remote_match_passes(monkeypatch):
    valid_sha = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_sha", lambda *args: valid_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_branch", lambda *args: "phase2.1-a-exec-labels")
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_remote_sha", lambda *args: valid_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_true_remote_sha", lambda *args: valid_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_working_tree_clean", lambda *args: (True, []))

    prov = _validate_source_provenance(enforce_clean=True)
    assert prov["source_commit_sha"] == valid_sha
    assert prov["true_remote_sha"] == valid_sha
    assert prov["source_commit_tree_clean"] is True
    assert prov["source_commit_remote_match"] is True


def test_source_provenance_fails_on_dirty_tree(monkeypatch):
    valid_sha = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_sha", lambda *args: valid_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_branch", lambda *args: "phase2.1-a-exec-labels")
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_remote_sha", lambda *args: valid_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_true_remote_sha", lambda *args: valid_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_working_tree_clean", lambda *args: (False, ["MODIFIED: models/walk_forward.py"]))

    with pytest.raises(RuntimeError, match="clean tracked source tree"):
        _validate_source_provenance(enforce_clean=True)


def test_source_provenance_fails_on_unknown_head(monkeypatch):
    valid_sha = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_sha", lambda *args: "UNKNOWN")
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_branch", lambda *args: "phase2.1-a-exec-labels")
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_remote_sha", lambda *args: valid_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_true_remote_sha", lambda *args: valid_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_working_tree_clean", lambda *args: (True, []))

    with pytest.raises(RuntimeError, match="HEAD source commit SHA"):
        _validate_source_provenance(enforce_clean=True)


def test_source_provenance_fails_on_unknown_branch(monkeypatch):
    valid_sha = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_sha", lambda *args: valid_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_branch", lambda *args: "UNKNOWN")
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_remote_sha", lambda *args: "UNKNOWN")
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_true_remote_sha", lambda *args: "UNKNOWN")
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_working_tree_clean", lambda *args: (True, []))

    with pytest.raises(RuntimeError, match="current Git branch name"):
        _validate_source_provenance(enforce_clean=True)


def test_source_provenance_fails_on_unknown_true_remote(monkeypatch):
    valid_sha = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_sha", lambda *args: valid_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_branch", lambda *args: "phase2.1-a-exec-labels")
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_remote_sha", lambda *args: valid_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_true_remote_sha", lambda *args: "UNKNOWN")
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_working_tree_clean", lambda *args: (True, []))

    with pytest.raises(RuntimeError, match="true remote branch SHA"):
        _validate_source_provenance(enforce_clean=True)


def test_source_provenance_fails_on_head_true_remote_mismatch(monkeypatch):
    local_sha = "0123456789abcdef0123456789abcdef01234567"
    remote_sha = "fedcba9876543210fedcba9876543210fedcba98"
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_sha", lambda *args: local_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_branch", lambda *args: "phase2.1-a-exec-labels")
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_remote_sha", lambda *args: remote_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_true_remote_sha", lambda *args: remote_sha)
    monkeypatch.setattr("tools.run_phase2_1_a_label_ab._git_working_tree_clean", lambda *args: (True, []))

    with pytest.raises(RuntimeError, match="does not match true remote origin"):
        _validate_source_provenance(enforce_clean=True)
