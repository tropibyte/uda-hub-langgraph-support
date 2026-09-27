"""Intake agent: turns an incoming message into a fully-contextualised ticket.

Deterministic (no LLM). Each turn it:
1. resolves the ticket (creating it if the thread is new) and the customer,
2. normalises the text for its channel (strips quoted email replies/signatures),
3. persists the customer's message to ticket_messages,
4. loads the CultPass profile through the same tool layer the agents use,
5. loads previous tickets and recalls long-term memories,
6. assigns the A/B routing variant, and
7. resets the per-turn working fields of the state.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage

from agentic.agents.common import emit, safe_node
from agentic.config import get_settings
from agentic.services import Services
from agentic.state import UDAHubState
from agentic.tools import udahub_ops

_QUOTED_REPLY = re.compile(r"\n(On .{5,120} wrote:|-----Original Message-----|From: .+\nSent: ).*", re.S)
_SIGNATURE = re.compile(r"\n(--\s*\n|Sent from my \w+).*", re.S)


def normalise(text: str, channel: str) -> str:
    text = (text or "").replace("\r\n", "\n").strip()
    if channel == "email":
        text = _QUOTED_REPLY.sub("", text)
        text = _SIGNATURE.sub("", text)
        text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(">")).strip()
    return re.sub(r"[ \t]+", " ", text)


def assign_variant(ticket_id: str) -> str:
    strategy = get_settings().routing_strategy
    if strategy in ("rules_first", "llm_supervisor"):
        return strategy
    return "rules_first" if int(hashlib.sha256(ticket_id.encode()).hexdigest(), 16) % 2 == 0 else "llm_supervisor"


def make_intake(services: Services):
    @safe_node("intake")
    def intake(state: UDAHubState, config=None) -> dict:
        cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
        ticket_id = state.get("ticket_id") or cfg.get("ticket_id") or cfg.get("thread_id")
        if not ticket_id:
            raise ValueError("no ticket_id: invoke with config={'configurable': {'thread_id': <ticket_id>}}")
        turn = int(state.get("turn") or 0) + 1
        run_id = uuid.uuid4().hex[:12]
        base = {"ticket_id": ticket_id, "turn": turn, "run_id": run_id}
        st = {**state, **base}

        ticket = udahub_ops.get_ticket(ticket_id)
        if ticket is None:  # new thread started from chat_interface / an API call
            ext = state.get("external_user_id") or cfg.get("external_user_id") or f"guest-{ticket_id[:8]}"
            udahub_ops.create_ticket(ext, channel=cfg.get("channel", "chat"), ticket_id=ticket_id,
                                     urgency=cfg.get("urgency"), subject=cfg.get("subject"))
            ticket = udahub_ops.get_ticket(ticket_id)
        channel = ticket["channel"] or "chat"

        last = state["messages"][-1] if state.get("messages") else None
        raw = last.content if isinstance(last, HumanMessage) else ""
        text = normalise(raw if isinstance(raw, str) else str(raw), channel)
        if isinstance(last, HumanMessage):
            # message id doubles as TicketMessage.message_id -> idempotent persistence
            udahub_ops.add_message(ticket_id, "user", text, message_id=last.id)
        events = [emit(st, "intake", "ticket_received", {"channel": channel, "chars": len(text), "turn": turn})]

        # CultPass profile through the tool layer (logged like any tool call)
        customer, tool_calls = {}, []
        ext_id = ticket["external_user_id"]
        if not ext_id.startswith("guest-"):
            res = services.tools.call("get_customer_profile", {}, user_id=ext_id, ticket_id=ticket_id)
            tool_calls.append({"turn": turn, "agent": "intake", "tool": "get_customer_profile", "args": {},
                               "ok": res.get("ok"), "result": res.get("data") or res.get("error"),
                               **res.get("_meta", {})})
            events.append(emit(st, "intake", "tool_call", {"tool": "get_customer_profile", "args": {},
                                                            "transport": res.get("_meta", {}).get("transport")}))
            events.append(emit(st, "intake", "tool_result", {"tool": "get_customer_profile", "ok": res.get("ok"),
                                                              "result": res.get("data") or res.get("error")}))
            if res.get("ok"):
                customer = res["data"]

        history = udahub_ops.get_user_ticket_history(ticket["user_id"], exclude_ticket_id=ticket_id)
        memories = services.memory.recall(ticket["user_id"], text) if text else []
        variant = state.get("ab_variant") or assign_variant(ticket_id)

        created = ticket["created_at"] or datetime.now()
        age_h = round((datetime.now() - created).total_seconds() / 3600, 1)
        two_weeks_ago = (datetime.now() - timedelta(days=14)).isoformat(timespec="minutes")
        recent_unresolved = [h for h in history if h["status"] in ("open", "escalated", "pending_customer")
                             and (h["created_at"] or "") >= two_weeks_ago]
        ticket_ctx = {
            "channel": channel, "status": ticket["status"], "created_at": created.isoformat(timespec="minutes"),
            "age_hours": age_h, "user_id": ticket["user_id"], "user_name": ticket["user_name"],
            "provided_urgency": udahub_ops.tag_value(ticket["tags"], "urgency"),
            "subject": udahub_ops.tag_value(ticket["tags"], "subject"),
            "prior_tickets": len(history), "prior_unresolved": len(recent_unresolved),
        }
        if ticket["status"] in ("resolved", "pending_customer") and turn > 1:
            udahub_ops.update_ticket_metadata(ticket_id, status="in_progress")
        if turn == 1:
            udahub_ops.update_ticket_metadata(ticket_id, add_tags=[f"variant:{variant}"])
            events.append(emit(st, "intake", "ab_assignment", {"variant": variant}))
        events.append(emit(st, "intake", "context_loaded", {
            "customer_found": bool(customer), "blocked": customer.get("is_blocked"),
            "tier": (customer.get("subscription") or {}).get("tier"), "prior_tickets": len(history),
            "ticket_age_hours": age_h}))
        if memories:
            events.append(emit(st, "intake", "memory_recalled", {
                "count": len(memories), "items": [f"{m['kind']}:{m['content'][:60]}" for m in memories]}))

        update = {
            **base, "external_user_id": ext_id, "ticket": ticket_ctx, "customer": customer,
            "history": history, "memories": memories, "ab_variant": variant, "user_message": text,
            # per-turn working state
            "classification": {}, "priority": {}, "retrieval": {}, "route": "", "hops": 0,
            "specialist_result": {}, "last_specialist": None, "qa": {}, "revision_count": 0,
            "escalation": {}, "final": {}, "error": None,
            "tool_calls": tool_calls, "events": events,
        }
        if not text:
            update["final"] = {
                "status": "needs_customer_input", "handled_by": "intake", "confidence": 1.0, "citations": [],
                "response": "Thanks for reaching out to CultPass support! Could you tell me a bit more about what you need help with?",
            }
        return update

    return intake
