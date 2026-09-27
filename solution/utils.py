# reset_udahub.py
import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph


Base = declarative_base()

def reset_db(db_path: str, echo: bool = True):
    """Drops the existing udahub.db file and recreates all tables."""

    # Remove the file if it exists, plus the WAL sidecars the agent runtime
    # creates (a stale -wal file must never be replayed into a fresh database)
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing {db_path}")
    for sidecar in (db_path + "-wal", db_path + "-shm", db_path + "-journal"):
        if os.path.exists(sidecar):
            os.remove(sidecar)

    # Create a new engine and recreate tables
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    engine.dispose()  # release the file handle (Windows cannot delete an open SQLite file)
    print(f"✅ Recreated {db_path} with fresh schema")


@contextmanager
def get_session(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }


def print_turn(result: dict, show_trace: bool = True) -> None:
    """Pretty-print one graph turn: the decision trail, then the reply."""
    final = result.get("final") or {}
    turn = result.get("turn")
    if show_trace:
        c = result.get("classification") or {}
        if c:
            print(f"  · classified   {c.get('category')} | urgency={c.get('urgency')} | "
                  f"sentiment={c.get('sentiment')} | priority={(result.get('priority') or {}).get('level')}")
        r = result.get("retrieval") or {}
        if r:
            print(f"  · retrieved    confidence={r.get('confidence')} | "
                  f"{[a['title'] for a in r.get('articles', [])[:3]]}")
        for h in result.get("routing_history", []):
            if h.get("turn") == turn:
                print(f"  · routed       -> {h['route']:<19} [{h['rule']}] {h['reason']}")
        for t in result.get("tool_calls", []):
            if t.get("turn") == turn and t.get("agent") != "intake":
                print(f"  · tool         {t['agent']}: {t['tool']}({t.get('args')}) ok={t.get('ok')} via {t.get('transport')}")
        q = result.get("qa") or {}
        if q:
            print(f"  · QA           passed={q.get('passed')} confidence={q.get('confidence')} {q.get('issues') or ''}")
        e = result.get("escalation") or {}
        if final.get("status") == "escalated" and e:
            print(f"  · escalated    team={e.get('team')} priority={e.get('priority')} reason={e.get('reason')}")
        cites = [c["title"] for c in final.get("citations", [])]
        print(f"  · outcome      {final.get('status')} by {final.get('handled_by')} | sources: {cites}")
    print(f"Assistant: {final.get('response')}\n")


def chat_interface(agent: CompiledStateGraph, ticket_id: str, external_user_id: str | None = None,
                   channel: str = "chat", scripted_inputs: list[str] | None = None, show_trace: bool = True):
    """Chat with UDA-Hub on one ticket thread.

    ticket_id is the LangGraph thread_id, so the conversation (short-term
    memory) continues across calls with the same id. A ticket that does not
    exist yet is created for ``external_user_id`` (a CultPass user id) on the
    first message. Pass ``scripted_inputs`` to run a conversation
    non-interactively (used in the notebook). Type 'quit' to leave.
    """
    config = {"configurable": {"thread_id": ticket_id, "external_user_id": external_user_id, "channel": channel}}
    inputs = iter(scripted_inputs) if scripted_inputs is not None else None
    while True:
        if inputs is not None:
            user_input = next(inputs, "quit")
        else:
            user_input = input("User: ")
        print("User:", user_input)
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Assistant: Goodbye!")
            break
        trigger = {
            "messages": [HumanMessage(content=user_input)],
            "ticket_id": ticket_id,
        }
        if external_user_id:
            trigger["external_user_id"] = external_user_id
        result = agent.invoke(input=trigger, config=config)
        print_turn(result, show_trace)
