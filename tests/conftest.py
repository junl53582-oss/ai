import pytest
from pathlib import Path

@pytest.fixture(autouse=True)
def isolate_paper_broker_state(tmp_path, monkeypatch):
    """自动隔离 PaperBroker 状态落盘路径，确保测试永远不会污染或读取生产账本"""
    from execution.paper_broker import PaperBroker
    test_state_file = tmp_path / "test_paper_broker_state.json"
    monkeypatch.setattr(PaperBroker, "_get_state_file", lambda self: test_state_file)
    yield
