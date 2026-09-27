"""Demo scenarios shared by 03_agentic_app.ipynb and 03_agentic_app.py.

Each scenario is one ticket (one thread) with one or more customer turns and
states what it demonstrates. ``run_scenario`` prints the decision trail of
each turn (classification, retrieval, routing, tools, QA, outcome) and
returns the final state for inspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from utils import print_turn

from agentic.workflow import send_message, submit_ticket


@dataclass
class Scenario:
    key: str
    title: str
    user: str
    messages: list[str]
    channel: str = "chat"
    urgency: str | None = None
    shows: list[str] = field(default_factory=list)


SCENARIOS = [
    Scenario("kb_resolution", "Knowledge-base answer", "f556c0",
             ["How do I reserve a spot for an event?"],
             shows=["classification", "RAG retrieval", "knowledge_resolver", "QA", "citations", "resolved"]),
    Scenario("multi_turn_tools", "Two-turn reservation cancellation (session memory + tools)", "f556c0",
             ["I need to cancel one of my reservations", "The samba one please"],
             shows=["account_specialist", "MCP tools list/cancel", "clarifying question", "short-term memory"]),
    Scenario("reserve", "Book an experience through tools", "e6376d",
             ["Can you book the Samba Night at Lapa for me?"],
             shows=["search_experiences", "reserve_experience", "quota rules"]),
    Scenario("sold_out", "Sold-out event: tool error + waitlist policy", "88382b",
             ["Please book the Ibirapuera Park Bike Ride for me"],
             shows=["tool error handling", "follow-up KB search", "waitlist article"]),
    Scenario("refund", "Refund request (email) -> billing lead", "88382b",
             ["Hi,\nI was charged twice this month for my plan. Please refund the duplicate charge.\n\nThanks,\nCathy\n\n"
              "On Mon, CultPass <no-reply@cultpass.com> wrote:\n> Your receipt"],
             channel="email",
             shows=["email normalisation", "billing_specialist", "submit_refund_request", "escalation by policy", "email formatting"]),
    Scenario("blocked", "Blocked account -> Trust & Safety guardrail", "a4ab87",
             ["I can't log in, it says my account is suspended. I have an event tomorrow!"], urgency="high",
             shows=["metadata-aware priority", "guardrail G3", "escalation handoff"]),
    Scenario("safety", "Safety incident (keyword backstop)", "e6376d",
             ["A staff member at the samba club harassed me last night. I felt unsafe."],
             shows=["guardrail G1", "P1 priority", "Trust & Safety"]),
    Scenario("no_article", "Nothing in the knowledge base -> escalation", "f556c0",
             ["Can you recommend a good pizza recipe?"],
             shows=["knowledge gate K1", "low retrieval confidence", "escalation"]),
    Scenario("portuguese_social", "Portuguese question on social media", "88382b",
             ["Olá! Como faço para reservar uma experiência?"], channel="social",
             shows=["language detection", "social channel formatting"]),
]


def run_scenario(graph, s: Scenario, show_trace: bool = True) -> dict:
    tid = submit_ticket(s.user, channel=s.channel, urgency=s.urgency)
    print(f"=== {s.title}  [user {s.user}, {s.channel}, ticket {tid[:8]}]")
    print(f"    demonstrates: {', '.join(s.shows)}\n")
    out = None
    for m in s.messages:
        print("User:", m.replace("\n", " ")[:160])
        out = send_message(graph, tid, m)
        print_turn(out, show_trace)
    return out
