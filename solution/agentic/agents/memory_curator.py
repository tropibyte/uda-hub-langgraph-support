"""Memory Curator agent: decides what is worth remembering after each turn.

* Preferences/facts: an LLM extraction, run only when the customer's message
  contains preference cues ("I prefer", "please always", "in Portuguese",
  "call me"...). Most turns have none, so this saves an LLM call per turn;
  skipped turns are logged.
* Issue outcomes: every resolved or escalated turn upserts one
  ``resolved_issue`` / ``escalated_issue`` memory for the ticket, so the next
  session knows what happened without re-reading the transcript.
"""
from __future__ import annotations

import re
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from agentic.agents.common import emit
from agentic.services import Services
from agentic.state import MemoryExtraction, UDAHubState

CUES = re.compile(
    r"\b(prefer\w*|always|never|from now on|going forward|in future|instead of|call me|my name is|i (like|love|enjoy|am into)|"
    r"interested in|favou?rite|in (portuguese|english|spanish)|speak (portuguese|english|spanish)|"
    r"language|email me|text me|contact me|wheelchair|accessib\w*|hearing|vegetarian|i live in|i'?m based in)\b", re.I)

SYSTEM = """You maintain long-term memory for CultPass customer support. From the customer's messages,
extract durable preferences and facts about the customer that will help future support conversations
(contact channel, language, city, interests, accessibility needs, how to address them). Ignore the
current problem itself, one-off details, and anything the customer did not state about themselves."""


def make_memory_curator(services: Services):
    def memory_curator(state: UDAHubState) -> dict:
        events = []
        ticket = state.get("ticket") or {}
        user_id = ticket.get("user_id")
        final = state.get("final") or {}
        cls = state.get("classification") or {}
        if not user_id or ticket.get("user_name") == "Guest":
            return {}
        written = []
        try:
            text = state.get("user_message", "")
            if CUES.search(text):
                ex = services.structured(MemoryExtraction, [SystemMessage(SYSTEM), HumanMessage(f"Customer message:\n{text}")])
                for p in ex.preferences:
                    written.append(services.memory.remember(user_id, "preference", p.value, key=p.key,
                                                            source_ticket_id=state.get("ticket_id")))
                for f in ex.facts:
                    written.append(services.memory.remember(user_id, "fact", f, source_ticket_id=state.get("ticket_id")))
            else:
                events.append(emit(state, "memory_curator", "memory_write", {"skipped": "no preference cues in message"}))
            if final.get("status") in ("resolved", "escalated"):
                kind = "resolved_issue" if final["status"] == "resolved" else "escalated_issue"
                how = (f"escalated to {final.get('team')}" if kind == "escalated_issue"
                       else f"resolved by {final.get('handled_by')}")
                content = (f"{date.today().isoformat()} [{cls.get('category')}] {cls.get('intent') or state.get('user_message', '')[:80]}"
                           f" -> {how}. Reply: {final.get('response', '')[:160]}")
                written.append(services.memory.remember(user_id, kind, content, source_ticket_id=state.get("ticket_id")))
            if written:
                events.append(emit(state, "memory_curator", "memory_write", {
                    "items": [f"{w['action']} {w['kind']}{'/' + w['key'] if w.get('key') else ''}: {w['content'][:70]}" for w in written]}))
        except Exception as exc:  # noqa: BLE001 - memory must never block the reply
            events.append(emit(state, "memory_curator", "error", {"error": f"{exc.__class__.__name__}: {exc}"[:300]}, level="ERROR"))
        return {"events": events}

    return memory_curator
