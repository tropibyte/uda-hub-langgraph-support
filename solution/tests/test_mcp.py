"""The FastMCP server: tool discovery, calls, error envelopes and fallback."""
import pytest

from agentic.tools import cultpass_ops as ops
from agentic.tools import registry as reg_mod
from agentic.tools.registry import ToolRegistry

pytestmark = pytest.mark.mcp


@pytest.fixture
def mcp_registry(data_env):
    reg = ToolRegistry("mcp").start()
    yield reg
    reg.close()


def test_mcp_server_exposes_every_operation(mcp_registry):
    assert mcp_registry.transport == "mcp", mcp_registry.fallback_reason
    assert set(mcp_registry._bridge.tool_names) == set(ops.OPERATIONS)


def test_mcp_calls_match_local_results(mcp_registry):
    via_mcp = mcp_registry.call("get_customer_profile", {}, user_id="f556c0")
    assert via_mcp["_meta"]["transport"] == "mcp"
    via_mcp.pop("_meta")
    assert via_mcp == ops.get_customer_profile("f556c0")


def test_mcp_write_and_error_envelopes(mcp_registry):
    listing = mcp_registry.call("list_reservations", {}, user_id="f556c0")["data"]
    samba = next(r for r in listing if r["title"] == "Samba Night at Lapa")
    ok = mcp_registry.call("cancel_reservation", {"reservation_id": samba["reservation_id"]}, user_id="f556c0")
    assert ok["ok"] and ok["data"]["credit_returned"] is True
    err = mcp_registry.call("cancel_subscription", {}, user_id="f556c0")
    assert err["ok"] is False and err["error"]["code"] == "confirmation_required"
    assert mcp_registry.call("get_customer_profile", {}, user_id="nope")["error"]["code"] == "not_found"


def test_falls_back_to_local_when_server_cannot_start(data_env, monkeypatch):
    monkeypatch.setattr(reg_mod, "SERVER_PATH", "does/not/exist.py")
    monkeypatch.setattr(reg_mod.MCPBridge.__init__, "__defaults__", (10.0, 30.0))
    reg = ToolRegistry("mcp").start()
    try:
        assert reg.transport == "local" and reg.fallback_reason
        assert reg.call("get_customer_profile", {}, user_id="f556c0")["ok"]
    finally:
        reg.close()
