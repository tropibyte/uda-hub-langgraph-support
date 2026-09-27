"""Rubric: database infrastructure and knowledge base."""
import json

from sqlalchemy import inspect

from agentic.db import core_engine, cultpass, cultpass_engine, session_scope, udahub
from agentic.tools import udahub_ops

REQUIRED = {"accounts", "users", "tickets", "ticket_metadata", "ticket_messages", "knowledge"}
ADDED = {"customer_memories", "knowledge_embeddings", "agent_events", "support_actions"}


def test_core_db_has_required_and_added_tables(data_env):
    tables = set(inspect(core_engine()).get_table_names())
    assert REQUIRED <= tables
    assert ADDED <= tables


def test_knowledge_base_has_at_least_14_articles_and_10_new(data_env):
    with session_scope(core_engine()) as s:
        titles = [k.title for k in s.query(udahub.Knowledge).all()]
    original = {"How to Reserve a Spot for an Event", "What's Included in a CultPass Subscription",
                "How to Cancel or Pause a Subscription", "How to Handle Login Issues?"}
    assert len(titles) >= 14
    assert original <= set(titles)
    assert len(set(titles) - original) >= 10
    assert len(titles) == len(set(titles)), "duplicate article titles"


def test_articles_cover_diverse_categories_and_are_well_formed(data_env):
    from agentic.config import SOLUTION_DIR
    rows = [json.loads(l) for l in open(SOLUTION_DIR / "data/external/cultpass_articles.jsonl", encoding="utf-8")]
    categories = {r["tags"].split(",")[0].strip() for r in rows}
    for needed in ("billing", "subscription", "reservation", "technical", "account", "safety", "escalation"):
        assert needed in categories, needed
    for r in rows:
        assert r["title"] and r["tags"] and len(r["content"]) > 150
        assert "Suggested phrasing" in r["content"]


def test_cultpass_seed_data_is_deterministic(data_env):
    with session_scope(cultpass_engine()) as s:
        users = {u.user_id: u for u in s.query(cultpass.User).all()}
        assert len(users) == 6
        assert users["a4ab87"].is_blocked and not users["f556c0"].is_blocked
        assert users["f556c0"].subscription.tier == "basic"
        assert users["f1f10d"].subscription.status == "paused"
        owners = {r.user_id for r in s.query(cultpass.Reservation).all()}
        assert {"a4ab87", "f556c0", "88382b", "e6376d"} <= owners
        sold_out = s.query(cultpass.Experience).filter_by(title="Ibirapuera Park Bike Ride").one()
        assert sold_out.slots_available == 0


def test_seeded_tickets_and_history_are_retrievable(data_env):
    tickets = udahub_ops.list_tickets()
    assert len(tickets) == 3
    open_ticket = next(t for t in tickets if t["status"] == "open")
    full = udahub_ops.get_ticket(open_ticket["ticket_id"])
    assert full["messages"][0]["content"] == "I can't log in to my Cultpass account."
    assert full["external_user_id"] == "a4ab87"
    bob = next(t for t in tickets if t["user_name"] == "Bob Stone")
    history = udahub_ops.get_user_ticket_history(udahub_ops.get_ticket(bob["ticket_id"])["user_id"])
    assert history and history[0]["issue_type"] == "technical_issue"


def test_ticket_crud_roundtrip(data_env):
    tid = udahub_ops.create_ticket("f556c0", channel="email", content="hello", urgency="high", subject="Help, please")
    udahub_ops.add_message(tid, "ai", "hi!", message_id="m1")
    udahub_ops.add_message(tid, "ai", "hi!", message_id="m1")  # idempotent
    udahub_ops.update_ticket_metadata(tid, status="resolved", main_issue_type="general_inquiry", add_tags=["urgency:low"])
    t = udahub_ops.get_ticket(tid)
    assert [m["role"] for m in t["messages"]] == ["user", "ai"]
    assert t["status"] == "resolved" and t["main_issue_type"] == "general_inquiry"
    assert udahub_ops.tag_value(t["tags"], "urgency") == "low"  # key:value tags are upserted
    assert udahub_ops.tag_value(t["tags"], "subject") == "Help  please"
