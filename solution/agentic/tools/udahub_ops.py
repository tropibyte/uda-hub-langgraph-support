"""UDA-Hub core operations: tickets, messages and metadata.

These persist the customer conversation (every user and AI message becomes a
TicketMessage row) and the ticket outcome (TicketMetadata status, issue type
and tags), which is what makes history available across sessions.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Iterable

from agentic.config import ACCOUNT_ID
from agentic.db import core_engine, cultpass_engine, cultpass, session_scope, udahub

VALID_CHANNELS = ("chat", "email", "social", "phone", "web_form")
VALID_STATUSES = ("open", "in_progress", "pending_customer", "resolved", "escalated", "closed")


# ---------------------------------------------------------------------------
# Tags are "free, tags, key:value" strings; key:value tags are upserted by key.
# ---------------------------------------------------------------------------

def parse_tags(tags: str | None) -> list[str]:
    return [t.strip() for t in (tags or "").split(",") if t.strip()]


def merge_tags(tags: str | None, add: Iterable[str]) -> str:
    current = parse_tags(tags)
    for tag in add:
        if ":" in tag:
            key = tag.split(":", 1)[0] + ":"
            current = [t for t in current if not t.startswith(key)]
        if tag not in current:
            current.append(tag)
    return ", ".join(current)


def tag_value(tags: str | None, key: str) -> str | None:
    for t in parse_tags(tags):
        if t.startswith(key + ":"):
            return t.split(":", 1)[1]
    return None


# ---------------------------------------------------------------------------

def get_or_create_user(external_user_id: str, user_name: str | None = None) -> dict:
    """Map a CultPass user id to a UDA-Hub user, creating the row on first contact."""
    with session_scope(core_engine()) as s:
        user = s.query(udahub.User).filter_by(account_id=ACCOUNT_ID, external_user_id=external_user_id).first()
        if user is None:
            if not user_name:
                try:
                    with session_scope(cultpass_engine()) as cs:
                        cp = cs.get(cultpass.User, external_user_id)
                        user_name = cp.full_name if cp else None
                except Exception:  # noqa: BLE001 - external DB unavailable -> still create the user
                    user_name = None
            user = udahub.User(user_id=str(uuid.uuid4()), account_id=ACCOUNT_ID,
                               external_user_id=external_user_id, user_name=user_name or "Guest")
            s.add(user)
        return {"user_id": user.user_id, "external_user_id": user.external_user_id, "user_name": user.user_name}


def create_ticket(external_user_id: str, *, channel: str = "chat", content: str | None = None,
                  urgency: str | None = None, subject: str | None = None, tags: Iterable[str] = (),
                  ticket_id: str | None = None, created_at: datetime | None = None) -> str:
    """Open a ticket (Ticket + TicketMetadata, and the first message when given)."""
    if channel not in VALID_CHANNELS:
        raise ValueError(f"channel must be one of {VALID_CHANNELS}")
    user = get_or_create_user(external_user_id)
    ticket_id = ticket_id or str(uuid.uuid4())
    extra = list(tags)
    if urgency:
        extra.append(f"urgency:{urgency}")
    if subject:
        extra.append(f"subject:{subject.replace(',', ' ')}")
    with session_scope(core_engine()) as s:
        s.add(udahub.Ticket(ticket_id=ticket_id, account_id=ACCOUNT_ID, user_id=user["user_id"],
                            channel=channel, created_at=created_at or datetime.now()))
        s.add(udahub.TicketMetadata(ticket_id=ticket_id, status="open", main_issue_type=None,
                                    tags=merge_tags("", extra)))
    if content:
        add_message(ticket_id, "user", content)
    return ticket_id


def add_message(ticket_id: str, role: str, content: str, message_id: str | None = None) -> str:
    """Append a message; idempotent when the same message_id is written twice."""
    message_id = message_id or str(uuid.uuid4())
    with session_scope(core_engine()) as s:
        if s.get(udahub.TicketMessage, message_id) is None:
            s.add(udahub.TicketMessage(message_id=message_id, ticket_id=ticket_id,
                                       role=udahub.RoleEnum(role), content=content))
    return message_id


def update_ticket_metadata(ticket_id: str, *, status: str | None = None, main_issue_type: str | None = None,
                           add_tags: Iterable[str] = ()) -> None:
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}")
    with session_scope(core_engine()) as s:
        md = s.get(udahub.TicketMetadata, ticket_id)
        if md is None:
            md = udahub.TicketMetadata(ticket_id=ticket_id, status=status or "open")
            s.add(md)
        if status:
            md.status = status
        if main_issue_type:
            md.main_issue_type = main_issue_type
        md.tags = merge_tags(md.tags, add_tags)


def get_ticket(ticket_id: str) -> dict | None:
    with session_scope(core_engine()) as s:
        t = s.get(udahub.Ticket, ticket_id)
        if t is None:
            return None
        md = t.ticket_metadata
        msgs = sorted(t.messages, key=lambda m: m.created_at or datetime.min)
        return {
            "ticket_id": t.ticket_id, "account_id": t.account_id, "channel": t.channel,
            "created_at": t.created_at, "user_id": t.user_id,
            "external_user_id": t.user.external_user_id, "user_name": t.user.user_name,
            "status": md.status if md else "open", "main_issue_type": md.main_issue_type if md else None,
            "tags": md.tags if md else "",
            "messages": [{"message_id": m.message_id, "role": m.role.value, "content": m.content,
                          "created_at": m.created_at} for m in msgs],
        }


def get_user_ticket_history(user_id: str, exclude_ticket_id: str | None = None, limit: int = 10) -> list[dict]:
    """Previous tickets of a UDA-Hub user, newest first, with first ask and last reply."""
    with session_scope(core_engine()) as s:
        q = s.query(udahub.Ticket).filter(udahub.Ticket.user_id == user_id)
        if exclude_ticket_id:
            q = q.filter(udahub.Ticket.ticket_id != exclude_ticket_id)
        out = []
        for t in q.order_by(udahub.Ticket.created_at.desc()).limit(limit).all():
            msgs = sorted(t.messages, key=lambda m: m.created_at or datetime.min)
            first_user = next((m.content for m in msgs if m.role.value == "user"), None)
            last_reply = next((m.content for m in reversed(msgs) if m.role.value in ("ai", "agent")), None)
            md = t.ticket_metadata
            out.append({
                "ticket_id": t.ticket_id, "created_at": t.created_at.isoformat(timespec="minutes") if t.created_at else None,
                "channel": t.channel, "status": md.status if md else None,
                "issue_type": md.main_issue_type if md else None,
                "first_message": (first_user or "")[:200], "last_reply": (last_reply or "")[:200],
            })
        return out


def list_tickets(status: str | None = None) -> list[dict]:
    with session_scope(core_engine()) as s:
        q = s.query(udahub.Ticket)
        rows = []
        for t in q.order_by(udahub.Ticket.created_at).all():
            md = t.ticket_metadata
            if status and (md is None or md.status != status):
                continue
            rows.append({"ticket_id": t.ticket_id, "user_name": t.user.user_name, "channel": t.channel,
                         "status": md.status if md else None, "issue_type": md.main_issue_type if md else None,
                         "tags": md.tags if md else None})
        return rows
