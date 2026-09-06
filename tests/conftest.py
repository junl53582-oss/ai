import os
import tempfile
import pytest
from pathlib import Path
from config.settings import settings
from data.network_policy import install_network_guard

# pytest 启动阶段初始化全局禁网守卫与测试环境标记
os.environ["QUANT_TEST_MODE"] = "1"
os.environ["QUANT_DISABLE_NETWORK"] = "1"
default_test_art = (Path(tempfile.gettempdir()) / "quant_pytest_artifacts").resolve()
default_test_art.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("QUANT_ARTIFACTS_DIR", str(default_test_art))

install_network_guard()


@pytest.fixture(autouse=True)
def isolate_paper_broker_state(tmp_path, monkeypatch):
    """自动隔离 PaperBroker 状态落盘路径，确保测试永远不会污染或读取生产账本"""
    from execution.paper_broker import PaperBroker
    test_state_file = tmp_path / "test_paper_broker_state.json"
    monkeypatch.setattr(PaperBroker, "_get_state_file", lambda self: test_state_file)
    yield


@pytest.fixture(autouse=True)
def configure_hermetic_test_environment(tmp_path, monkeypatch):
    """
    Stage S4 密闭测试隔离 (显式配置 + 底层禁网):
    - 注入环境变量 QUANT_TEST_MODE=1
    - 注入环境变量 QUANT_DISABLE_NETWORK=1
    - 注入环境变量 QUANT_ARTIFACTS_DIR=<tmp_path>/artifacts
    - settings.ARTIFACTS_DIR 动态切换至该临时目录
    - 激活底层网络拦截守卫 (requests, urllib, socket)
    - 彻底移除任何 open/replace 的隐式静默重定向黑魔法，直接写入工作区 artifacts 的行为将直接暴露
    """
    temp_artifacts = (tmp_path / "artifacts").resolve()
    temp_artifacts.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("QUANT_TEST_MODE", "1")
    monkeypatch.setenv("QUANT_DISABLE_NETWORK", "1")
    monkeypatch.setenv("QUANT_ARTIFACTS_DIR", str(temp_artifacts))
    monkeypatch.setattr(settings, "ARTIFACTS_DIR", temp_artifacts, raising=False)

    install_network_guard()
    yield

