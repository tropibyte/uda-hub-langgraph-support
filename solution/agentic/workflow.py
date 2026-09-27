"""UDA-Hub orchestration graph (Supervisor pattern), built from scratch with StateGraph.

    START -> intake -> classifier -> knowledge_retriever -> supervisor
    supervisor -> knowledge_resolver | account_specialist | billing_specialist   (each returns to supervisor)
    supervisor -> qa_reviewer | escalation
    qa_reviewer -> responder | escalation | <same specialist for one revision>
    escalation -> responder -> memory_curator -> END

See agentic/design/ARCHITECTURE.md for the diagram and the decision rules.

Usage:
    from agentic.workflow import orchestrator, submit_ticket, send_message
    tid = submit_ticket("f556c0", channel="chat")
    result = send_message(orchestrator, tid, "How do I cancel my Samba reservation?")
    print(result["final"]["response"])
"""
from __future__ import annotations

import sqlite3
from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic.agents import (make_classifier, make_escalation, make_intake, make_knowledge_retriever,
                            make_memory_curator, make_qa_reviewer, make_responder, make_specialist,
                            make_supervisor)
from agentic.agents.common import emit
from agentic.config import get_settings
from agentic.services import Services
from agentic.state import UDAHubState
from agentic.tools import udahub_ops

SPECIALISTS = ("knowledge_resolver", "account_specialist", "billing_specialist")


def make_checkpointer(kind: str | None = None):
    """Short-term memory store. SQLite keeps sessions across process restarts."""
    kind = (kind or get_settings().checkpointer).lower()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            path = get_settings().checkpoint_db
            path.parent.mkdir(parents=True, exist_ok=True)
            return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
        except ImportError:
            pass  # langgraph-checkpoint-sqlite not installed -> in-memory sessions
    return MemorySaver()


def _guard(fn, agent: str):
    """Last-resort isolation for the nodes that must always produce output."""
    def wrapped(state: UDAHubState) -> dict:
        try:
            return fn(state)
        except Exception as exc:  # noqa: BLE001
            ev = emit(state, agent, "error", {"error": f"{exc.__class__.__name__}: {exc}"[:300]}, level="ERROR")
            if agent == "escalation":
                return {"events": [ev], "final": {
                    "status": "escalated", "handled_by": "escalation", "citations": [], "confidence": None,
                    "team": "tier2_support", "priority": "P2",
                    "response": "I've passed your ticket to our support team, who will reply within 1 business day."}}
            return {"events": [ev]}
    wrapped.__name__ = agent
    return wrapped


def build_workflow(services: Optional[Services] = None, checkpointer=None) -> CompiledStateGraph:
    services = services or Services()
    g = StateGraph(UDAHubState)

    g.add_node("intake", make_intake(services))
    g.add_node("classifier", make_classifier(services))
    g.add_node("knowledge_retriever", make_knowledge_retriever(services))
    g.add_node("supervisor", make_supervisor(services))
    for name in SPECIALISTS:
        g.add_node(name, make_specialist(services, name))
    g.add_node("qa_reviewer", make_qa_reviewer(services))
    g.add_node("escalation", _guard(make_escalation(services), "escalation"))
    g.add_node("responder", _guard(make_responder(), "responder"))
    g.add_node("memory_curator", _guard(make_memory_curator(services), "memory_curator"))

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", lambda s: "responder" if s.get("final") else
                            ("escalation" if s.get("error") else "classifier"),
                            ["classifier", "responder", "escalation"])
    g.add_edge("classifier", "knowledge_retriever")
    g.add_edge("knowledge_retriever", "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["route"],
                            [*SPECIALISTS, "qa_reviewer", "escalation"])
    for name in SPECIALISTS:
        g.add_edge(name, "supervisor")
    g.add_conditional_edges("qa_reviewer", lambda s: s["route"],
                            ["responder", "escalation", *SPECIALISTS])
    g.add_edge("escalation", "responder")
    g.add_edge("responder", "memory_curator")
    g.add_edge("memory_curator", END)

    graph = g.compile(checkpointer=checkpointer if checkpointer is not None else make_checkpointer(),
                      name="uda_hub")
    graph.services = services  # handy for notebooks/tests (tool transport, llm call count)
    return graph


# ---------------------------------------------------------------------------
# Convenience API used by the notebook, the CLI, the evaluation and the tests
# ---------------------------------------------------------------------------

def submit_ticket(external_user_id: str, content: str | None = None, *, channel: str = "chat",
                  urgency: str | None = None, subject: str | None = None, ticket_id: str | None = None) -> str:
    """Create a ticket in UDA-Hub (what a Zendesk/Intercom connector would call).

    ``content`` is optional: the first message is normally sent with
    ``send_message`` so it flows through the graph (and is persisted there).
    """
    tid = udahub_ops.create_ticket(external_user_id, channel=channel, urgency=urgency, subject=subject,
                                   ticket_id=ticket_id)
    if content:
        udahub_ops.add_message(tid, "user", content)
    return tid


def send_message(graph: CompiledStateGraph, ticket_id: str, text: str) -> dict:
    """Run one customer turn on a ticket's thread and return the final state."""
    return graph.invoke({"messages": [HumanMessage(content=text)], "ticket_id": ticket_id},
                        config=thread_config(ticket_id))


def process_ticket(graph: CompiledStateGraph, ticket_id: str) -> dict:
    """Run the graph on a ticket that arrived with its first message already stored
    (e.g. the seeded ticket from 02_core_db_setup). The stored message id is reused,
    so intake does not store the message twice."""
    ticket = udahub_ops.get_ticket(ticket_id)
    if ticket is None:
        raise ValueError(f"unknown ticket {ticket_id}")
    last = next((m for m in reversed(ticket["messages"]) if m["role"] == "user"), None)
    if last is None:
        raise ValueError(f"ticket {ticket_id} has no customer message")
    return graph.invoke({"messages": [HumanMessage(content=last["content"], id=last["message_id"])],
                         "ticket_id": ticket_id}, config=thread_config(ticket_id))


def thread_config(ticket_id: str) -> dict:
    return {"configurable": {"thread_id": ticket_id}}


orchestrator = build_workflow()
