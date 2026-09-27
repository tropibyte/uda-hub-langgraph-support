"""Structured, searchable event logging.

Every agent decision, routing choice, retrieval, tool call and outcome is
written twice:

* one JSON object per line in ``logs/udahub_events.jsonl`` (grep/jq friendly)
* one row in the ``agent_events`` table of udahub.db (SQL friendly)

Both carry the same fields: ts, event_id, run_id, thread_id, ticket_id,
agent, event, level, payload. ``search_events`` queries the table and
``python -m agentic.logging_utils --help`` exposes the same search on the
command line.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from agentic.config import get_settings

_py_logger = logging.getLogger("udahub")

# Canonical event names, so searches do not depend on free text.
EVENTS = (
    "ticket_received", "context_loaded", "memory_recalled", "classification",
    "retrieval", "routing_decision", "agent_started", "tool_call", "tool_result",
    "agent_finished", "qa_review", "escalation", "resolution", "memory_write",
    "ab_assignment", "error",
)


def _json_default(o: Any):
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


def log_event(event: str, agent: str, payload: dict | None = None, *, ticket_id: str | None = None,
              thread_id: str | None = None, run_id: str | None = None, level: str = "INFO") -> dict:
    """Write one event to the JSONL log and the agent_events table; return it."""
    settings = get_settings()
    record = {
        "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "event_id": uuid.uuid4().hex[:12],
        "run_id": run_id,
        "thread_id": thread_id,
        "ticket_id": ticket_id,
        "agent": agent,
        "event": event,
        "level": level,
        "payload": payload or {},
    }
    line = json.dumps(record, default=_json_default, ensure_ascii=False)
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        with open(settings.event_log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:  # logging must never break ticket handling
        _py_logger.warning("could not write event log: %s", exc)
    try:
        from agentic.db import core_engine, session_scope, udahub
        with session_scope(core_engine()) as s:
            s.add(udahub.AgentEvent(
                event_id=record["event_id"],
                ts=datetime.fromisoformat(record["ts"]).replace(tzinfo=None),
                run_id=run_id, thread_id=thread_id, ticket_id=ticket_id,
                agent=agent, event=event, level=level,
                payload=json.dumps(record["payload"], default=_json_default, ensure_ascii=False),
            ))
    except Exception as exc:  # noqa: BLE001
        _py_logger.warning("could not write agent_events row: %s", exc)
    if os.getenv("UDAHUB_VERBOSE", "0") == "1":
        print(format_event(record))
    return record


def format_event(rec: dict) -> str:
    """One readable line per event, used by the notebook trace and the CLI."""
    p = rec.get("payload") or {}
    summary = {
        "classification": lambda: f"category={p.get('category')} urgency={p.get('urgency')} sentiment={p.get('sentiment')} priority={p.get('priority')}",
        "retrieval": lambda: f"confidence={p.get('confidence')} top={[a.get('title') for a in p.get('articles', [])[:3]]}",
        "routing_decision": lambda: f"-> {p.get('route')} ({p.get('rule')}: {p.get('reason')})",
        "tool_call": lambda: f"{p.get('tool')}({p.get('args')}) via {p.get('transport')}",
        "tool_result": lambda: f"{p.get('tool')} ok={p.get('ok')} {str(p.get('result'))[:90]}",
        "qa_review": lambda: f"passed={p.get('passed')} confidence={p.get('confidence')} issues={p.get('issues')}",
        "resolution": lambda: f"status={p.get('status')} confidence={p.get('confidence')} handled_by={p.get('handled_by')}",
        "escalation": lambda: f"team={p.get('team')} priority={p.get('priority')} reason={p.get('reason')}",
    }.get(rec["event"], lambda: json.dumps(p, default=_json_default)[:140])
    return f"  [{rec['agent']:<20}] {rec['event']:<17} {summary()}"


def search_events(*, ticket_id: str | None = None, thread_id: str | None = None, run_id: str | None = None,
                  agent: str | None = None, event: str | None = None, level: str | None = None,
                  text: str | None = None, limit: int = 200) -> list[dict]:
    """Query the agent_events table. All filters are optional and ANDed."""
    from agentic.db import core_engine, session_scope, udahub
    E = udahub.AgentEvent
    with session_scope(core_engine()) as s:
        q = s.query(E)
        for col, val in ((E.ticket_id, ticket_id), (E.thread_id, thread_id), (E.run_id, run_id),
                         (E.agent, agent), (E.event, event), (E.level, level)):
            if val is not None:
                q = q.filter(col == val)
        if text:
            q = q.filter(E.payload.contains(text))
        rows = q.order_by(E.ts.asc()).limit(limit).all()
        return [{
            "ts": r.ts.isoformat() if r.ts else None, "event_id": r.event_id, "run_id": r.run_id,
            "thread_id": r.thread_id, "ticket_id": r.ticket_id, "agent": r.agent,
            "event": r.event, "level": r.level, "payload": json.loads(r.payload or "{}"),
        } for r in rows]


def _cli():
    ap = argparse.ArgumentParser(description="Search UDA-Hub agent events")
    for f in ("ticket_id", "thread_id", "run_id", "agent", "event", "level", "text"):
        ap.add_argument(f"--{f.replace('_', '-')}", dest=f)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--json", action="store_true", help="print raw JSON instead of one line per event")
    a = vars(ap.parse_args())
    as_json = a.pop("json")
    for rec in search_events(**a):
        print(json.dumps(rec, ensure_ascii=False) if as_json else f"{rec['ts']} {rec['ticket_id'] or '-':>36} " + format_event(rec).strip())


if __name__ == "__main__":
    _cli()
