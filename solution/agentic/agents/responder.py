"""Responder agent: formats the final reply for its channel and records the outcome.

Deterministic. It adapts the approved text to the channel (email greeting and
sign-off, short public-safe social replies, markdown-free phone/voice
scripts), appends it to the conversation, persists it as a TicketMessage,
sets the ticket status and logs the ``resolution`` event.
"""
from __future__ import annotations

import re
import time
from datetime import datetime

from langchain_core.messages import AIMessage

from agentic.agents.common import emit
from agentic.state import UDAHubState
from agentic.tools import udahub_ops

STATUS = {"resolved": "resolved", "needs_customer_input": "pending_customer", "escalated": "escalated"}
SOCIAL_LIMIT = 280
CHANNEL_GUIDANCE = {
    "social": "Public social-media reply: at most 250 characters, no account details (ids, emails, "
              "reservations); invite them to send a private message for account-specific help.",
    "email": "Email: complete sentences; the greeting line and sign-off are added automatically, so do not add a sign-off.",
    "phone": "Phone/voice script: short spoken sentences, no markdown, no lists.",
    "chat": "Live chat: short and conversational.",
    "web_form": "Web form reply: complete sentences.",
}


def format_for_channel(text: str, channel: str, first_name: str | None) -> str:
    text = (text or "").strip()
    if channel == "email":
        greeting = f"Hi {first_name}," if first_name and first_name.lower() not in text[:40].lower() else ""
        body = f"{greeting}\n\n{text}" if greeting else text
        return f"{body}\n\nBest regards,\nCultPass Support"
    if channel == "social":
        # public channel: short, no personal details. The agent is told the limit up
        # front (CHANNEL_GUIDANCE); this is only the safety net, cutting at a sentence.
        text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email hidden]", text)
        if len(text) > SOCIAL_LIMIT:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            kept = ""
            for sent in sentences:
                if len(kept) + len(sent) + 1 > SOCIAL_LIMIT:
                    break
                kept = f"{kept} {sent}".strip()
            text = kept or text[:SOCIAL_LIMIT - 3].rsplit(" ", 1)[0] + "..."
        return text
    if channel == "phone":
        return re.sub(r"[*_#`]", "", text)
    return text


def make_responder():
    def responder(state: UDAHubState) -> dict:
        final = dict(state.get("final") or {})
        if not final:  # defensive: nothing produced a reply
            final = {"status": "escalated", "handled_by": "responder", "citations": [], "confidence": None,
                     "response": "Thanks for your patience - I've passed your ticket to our support team and they'll reply within 1 business day."}
        channel = (state.get("ticket") or {}).get("channel", "chat")
        name = ((state.get("customer") or {}).get("full_name") or (state.get("ticket") or {}).get("user_name") or "")
        first = name.split(" ")[0] if name and name != "Guest" else None
        text = format_for_channel(final["response"], channel, first)
        final.update(response=text, channel=channel)

        msg_id = udahub_ops.add_message(state["ticket_id"], "ai", text)
        cls = state.get("classification") or {}
        udahub_ops.update_ticket_metadata(state["ticket_id"], status=STATUS.get(final["status"], "open"),
                                          main_issue_type=cls.get("category"),
                                          add_tags=[f"handled_by:{final.get('handled_by')}"])
        started = next((e["ts"] for e in state.get("events", []) if e.get("turn") == state.get("turn")
                        and e["event"] == "ticket_received"), None)
        latency = None
        if started:
            latency = round(time.time() - datetime.fromisoformat(started).timestamp(), 2)
        route_path = [r["route"] for r in state.get("routing_history", []) if r.get("turn") == state.get("turn")]
        ev = emit(state, "responder", "resolution", {
            "status": final["status"], "handled_by": final.get("handled_by"), "confidence": final.get("confidence"),
            "citations": [c["ref"] for c in final.get("citations", [])], "channel": channel,
            "variant": state.get("ab_variant"), "category": cls.get("category"), "priority": (state.get("priority") or {}).get("level"),
            "route_path": route_path, "hops": state.get("hops"), "revisions": state.get("revision_count"),
            "tools": [t["tool"] for t in state.get("tool_calls", []) if t.get("turn") == state.get("turn") and t.get("agent") != "intake"],
            "latency_s": latency})
        return {"final": final, "messages": [AIMessage(content=text, id=msg_id, name=final.get("handled_by"))],
                "events": [ev]}

    return responder
