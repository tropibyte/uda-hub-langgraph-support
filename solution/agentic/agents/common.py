"""Helpers shared by the agent nodes: logging, error isolation, context blocks."""
from __future__ import annotations

import functools
import json
import traceback
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agentic.logging_utils import log_event
from agentic.state import UDAHubState


def emit(state: UDAHubState, agent: str, event: str, payload: dict | None = None, level: str = "INFO") -> dict:
    """Log an event with the ticket/run correlation ids taken from state.

    Returns a compact copy for the state's ``events`` trail so the decision
    history is inspectable from the checkpoint as well as from the log.
    """
    rec = log_event(event, agent, payload or {}, ticket_id=state.get("ticket_id"),
                    thread_id=state.get("ticket_id"), run_id=state.get("run_id"), level=level)
    return {"ts": rec["ts"], "turn": state.get("turn"), "agent": agent, "event": event,
            "level": level, "payload": rec["payload"]}


def safe_node(agent: str):
    """Isolate failures: a crashing node hands the ticket to escalation instead of raising."""
    def deco(fn: Callable[[UDAHubState], dict]):
        @functools.wraps(fn)
        def wrapper(state: UDAHubState, *args, **kwargs) -> dict:
            try:
                return fn(state, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                err = f"{agent}: {exc.__class__.__name__}: {exc}"[:400]
                ev = emit(state, agent, "error", {"error": err, "trace": traceback.format_exc()[-1200:]}, level="ERROR")
                return {"error": err, "route": "escalation", "events": [ev]}
        return wrapper
    return deco


def conversation(state: UDAHubState, limit: int = 12) -> list[BaseMessage]:
    """The customer-facing conversation of this thread (short-term memory)."""
    msgs = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
    return msgs[-limit:]


def transcript(state: UDAHubState, limit: int = 12) -> str:
    lines = []
    for m in conversation(state, limit):
        who = "Customer" if isinstance(m, HumanMessage) else "Support"
        lines.append(f"{who}: {m.content}")
    return "\n".join(lines)


def articles_block(articles: list[dict], max_chars: int = 1400) -> str:
    if not articles:
        return "(no knowledge-base article matched this request)"
    parts = []
    for a in articles:
        rel = f" relevance={a['relevance']}" if a.get("relevance") is not None else ""
        parts.append(f"[{a['ref']}] {a['title']}{rel}\n{a['content'][:max_chars]}")
    return "\n\n".join(parts)


def customer_block(state: UDAHubState) -> str:
    c = state.get("customer") or {}
    if not c:
        return "Customer: not linked to a CultPass account (guest). Account tools will not work."
    sub = c.get("subscription") or {}
    quota = c.get("quota") or {}
    return (f"Customer: {c.get('full_name')} (CultPass id {c.get('user_id')}), blocked={c.get('is_blocked')}\n"
            f"Plan: tier={sub.get('tier')} status={sub.get('status')} monthly_quota={sub.get('monthly_quota')} "
            f"used_this_cycle={quota.get('used_this_cycle')} remaining={quota.get('remaining')} "
            f"resets={quota.get('cycle_resets_at')}; upcoming reservations={c.get('upcoming_reservations')}")


def memory_block(state: UDAHubState) -> str:
    mems = state.get("memories") or []
    hist = state.get("history") or []
    lines = []
    prefs = [m for m in mems if m["kind"] == "preference"]
    if prefs:
        lines.append("Known preferences: " + "; ".join(f"{m['key']}={m['content']}" for m in prefs))
    for m in mems:
        if m["kind"] != "preference":
            lines.append(f"Past ({m['kind']}, {m.get('updated_at')}): {m['content']}")
    if hist:
        lines.append(f"Previous tickets: {len(hist)}. Most recent: " + "; ".join(
            f"{h['created_at']} {h['issue_type'] or 'unclassified'} ({h['status']}): {h['first_message'][:80]}" for h in hist[:3]))
    return "\n".join(lines) if lines else "No previous interactions on record (first contact)."


def ticket_block(state: UDAHubState) -> str:
    t = state.get("ticket") or {}
    c = state.get("classification") or {}
    p = state.get("priority") or {}
    from agentic.agents.responder import CHANNEL_GUIDANCE
    return (f"Ticket {state.get('ticket_id')} via {t.get('channel')} | category={c.get('category')} "
            f"intent={c.get('intent')!r} urgency={c.get('urgency')} sentiment={c.get('sentiment')} "
            f"priority={p.get('level')} language={c.get('language')}\n"
            f"Channel rules: {CHANNEL_GUIDANCE.get(t.get('channel'), '')}\n"
            f"REPLY LANGUAGE: {reply_language(state)}")


def reply_language(state: UDAHubState) -> str:
    """A stored language preference (long-term memory) beats the language of the message."""
    pref = next((m["content"] for m in state.get("memories") or []
                 if m["kind"] == "preference" and (m.get("key") or "").lower() in ("language", "preferred_language")), None)
    if pref:
        return f"{pref} (the customer's stored preference; mandatory even if they write in another language)"
    return f"the language of the customer's message ({(state.get('classification') or {}).get('language') or 'unknown'})"


def compact(obj: Any, limit: int = 600) -> str:
    s = json.dumps(obj, default=str, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "...}"
