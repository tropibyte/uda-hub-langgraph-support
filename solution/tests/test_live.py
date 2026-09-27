"""Live tests against the real LLM, embeddings and MCP server.

    RUN_LIVE=1 pytest tests/test_live.py -v        (PowerShell: $env:RUN_LIVE="1"; pytest tests/test_live.py -v)

They assert on outcomes and side effects, not on exact wording.
"""

import pytest

from agentic.db import cultpass, cultpass_engine, session_scope

pytestmark = pytest.mark.live


@pytest.fixture
def live_graph(data_env, monkeypatch):
    monkeypatch.setenv("UDAHUB_OFFLINE", "0")
    monkeypatch.setenv("UDAHUB_TOOL_TRANSPORT", "mcp")
    from langgraph.checkpoint.memory import MemorySaver

    from agentic.config import api_key
    from agentic.services import Services
    from agentic.workflow import build_workflow
    if not api_key():
        pytest.skip("no OPENAI_API_KEY")
    services = Services()
    graph = build_workflow(services, checkpointer=MemorySaver())
    yield graph
    services.close()


def test_live_kb_answer_is_grounded(live_graph):
    from agentic.workflow import send_message, submit_ticket
    out = send_message(live_graph, submit_ticket("f556c0"), "The QR code won't scan at the venue, what should I do?")
    f = out["final"]
    assert f["status"] == "resolved"
    assert "QR Code Not Scanning at the Venue" in [c["title"] for c in f["citations"]]
    assert out["retrieval"]["method"].startswith("hybrid")
    assert f["confidence"] >= 0.6


def test_live_two_turn_cancellation_over_mcp(live_graph):
    from agentic.workflow import send_message, submit_ticket
    tid = submit_ticket("f556c0")
    first = send_message(live_graph, tid, "I need to cancel one of my reservations")
    assert first["final"]["status"] == "needs_customer_input"
    second = send_message(live_graph, tid, "The samba one please")
    assert second["final"]["status"] == "resolved"
    cancels = [t for t in second["tool_calls"] if t["tool"] == "cancel_reservation" and t["ok"]]
    assert cancels and cancels[0]["transport"] == "mcp"
    with session_scope(cultpass_engine()) as s:
        statuses = {r.experience.title: r.status for r in s.query(cultpass.Reservation).filter_by(user_id="f556c0")}
    assert statuses["Samba Night at Lapa"] == "cancelled"
    assert statuses["Carnival History Tour in Olinda"] == "reserved"


def test_live_refund_escalates_to_billing_lead(live_graph):
    from agentic.workflow import send_message, submit_ticket
    out = send_message(live_graph, submit_ticket("88382b", channel="email"),
                       "I was charged twice this month. Please refund the duplicate.")
    assert out["final"]["status"] == "escalated" and out["escalation"]["team"] == "billing_lead"
    assert any(t["tool"] == "submit_refund_request" and t["ok"] for t in out["tool_calls"])
    assert "approved" not in out["final"]["response"].lower() or "if approved" in out["final"]["response"].lower()


def test_live_off_topic_escalates(live_graph):
    from agentic.workflow import send_message, submit_ticket
    out = send_message(live_graph, submit_ticket("e6376d"), "Can you recommend a good pizza recipe?")
    assert out["final"]["status"] == "escalated"
    assert out["routing_history"][0]["rule"] == "K1_no_article"
