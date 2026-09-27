"""Test fixtures.

Every test runs against its own freshly built copy of both databases, built
once per session by executing the code cells of notebooks 01 and 02 (so the
setup notebooks are tested too) and then copied per test. The environment is
pinned offline: scripted FakeLLM, BM25 retrieval, in-process tools and
in-memory checkpoints. MCP and live-LLM tests opt in explicitly.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

SOLUTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION_DIR))

# Must be set before agentic.workflow is imported (it builds `orchestrator`).
os.environ.update({
    "UDAHUB_OFFLINE": "1",
    "UDAHUB_TOOL_TRANSPORT": "local",
    "UDAHUB_CHECKPOINTER": "memory",
    "UDAHUB_ROUTING_STRATEGY": "rules_first",
    "UDAHUB_VERBOSE": "0",
})


@pytest.fixture(scope="session")
def pristine_data(tmp_path_factory) -> Path:
    from scripts.setup_databases import setup
    root = tmp_path_factory.mktemp("pristine")
    return setup(root)


@pytest.fixture
def data_env(pristine_data, tmp_path, monkeypatch) -> Path:
    data = tmp_path / "data"
    for sub in ("core", "external"):
        (data / sub).mkdir(parents=True)
    shutil.copy2(pristine_data / "core" / "udahub.db", data / "core" / "udahub.db")
    shutil.copy2(pristine_data / "external" / "cultpass.db", data / "external" / "cultpass.db")
    monkeypatch.setenv("UDAHUB_DATA_DIR", str(data))
    monkeypatch.setenv("UDAHUB_LOG_DIR", str(tmp_path / "logs"))
    return data


@pytest.fixture
def fake_llm():
    from tests.fakes import FakeLLM
    return FakeLLM()


@pytest.fixture
def make_graph(data_env):
    """Factory: a compiled graph wired to a FakeLLM, local tools and MemorySaver."""
    from langgraph.checkpoint.memory import MemorySaver

    from agentic.services import Services
    from agentic.tools.registry import ToolRegistry
    from agentic.workflow import build_workflow
    from tests.fakes import FakeLLM

    built = []

    def _make(llm=None, embedder=None, transport="local"):
        services = Services(llm=llm or FakeLLM(), embedder=embedder, tools=ToolRegistry(transport))
        graph = build_workflow(services, checkpointer=MemorySaver())
        built.append(services)
        return graph

    yield _make
    for s in built:
        s.close()


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_LIVE") != "1":
        skip = pytest.mark.skip(reason="live LLM tests need RUN_LIVE=1 and an API key")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip)
