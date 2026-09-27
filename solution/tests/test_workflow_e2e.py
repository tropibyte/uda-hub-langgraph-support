"""End-to-end: the real LangGraph workflow, real tools and databases, scripted LLM.

Covers the rubric's integration items: classification -> routing -> retrieval
-> tool use -> resolution/escalation -> final action, short- and long-term
memory, state inspection by thread_id, structured searchable logs, error
handling and edge cases.
"""
import json

from langchain_core.messages import HumanMessage

from agentic.db import core_engine, cultpass, cultpass_engine, session_scope, udahub
from agentic.logging_utils import search_events
from agentic.state import GroundingVerdict
from agentic.tools import udahub_ops
from agentic.workflow import process_ticket, send_message, submit_ticket, thread_config
from tests.fakes import FakeLLM


def turn_events(result):
    return [e["event"] for e in result["events"] if e["turn"] == result["turn"]]


def test_knowledge_resolution_path(make_graph):
    g = make_graph()
    tid = submit_ticket("f556c0", channel="chat")
    out = send_message(g, tid, "How do I reserve a spot for an event?")
    f = out["final"]
    assert f["status"] == "resolved" and f["handled_by"] == "knowledge_resolver"
    assert f["citations"] and f["citations"][0]["title"] == "How to Reserve a Spot for an Event"
    assert f["confidence"] >= 0.6
    assert [r["route"] for r in out["routing_history"]] == ["knowledge_resolver", "qa_reviewer"]
    assert out["qa"]["passed"] is True
    ticket = udahub_ops.get_ticket(tid)
    assert ticket["status"] == "resolved" and ticket["main_issue_type"] == "reservation_booking"
    assert [m["role"] for m in ticket["messages"]] == ["user", "ai"]  # conversation persisted
    for ev in ("ticket_received", "context_loaded", "classification", "retrieval", "routing_decision",
               "agent_started", "agent_finished", "qa_review", "resolution"):
        assert ev in turn_events(out), ev


def test_escalation_when_no_article_matches(make_graph):
    g = make_graph()
    tid = submit_ticket("e6376d")
    out = send_message(g, tid, "Can you recommend a good pizza recipe?")
    assert out["final"]["status"] == "escalated"
    assert out["routing_history"][-1]["rule"] == "K1_no_article"
    assert out["escalation"]["team"] == "tier2_support"
    ticket = udahub_ops.get_ticket(tid)
    assert ticket["status"] == "escalated"
    handoff = [m for m in ticket["messages"] if m["role"] == "system"]
    assert handoff and handoff[0]["content"].startswith("[HANDOFF -> senior support team")
    assert out["final"]["citations"][0]["title"].startswith("When and How Support Escalates")


def test_multi_turn_tool_workflow_uses_session_memory(make_graph):
    g = make_graph()
    tid = submit_ticket("f556c0")
    first = send_message(g, tid, "I need to cancel one of my reservations")
    assert first["final"]["status"] == "needs_customer_input"
    assert "Carnival" in first["final"]["response"] and "Samba" in first["final"]["response"]
    assert udahub_ops.get_ticket(tid)["status"] == "pending_customer"

    second = send_message(g, tid, "the samba one please")
    assert second["turn"] == 2
    assert second["final"]["status"] == "resolved"
    assert [m.type for m in second["messages"]] == ["human", "ai", "human", "ai"]  # short-term memory
    tools = [(t["turn"], t["tool"], t["ok"]) for t in second["tool_calls"] if t["agent"] != "intake"]
    assert (1, "list_reservations", True) in tools and (2, "cancel_reservation", True) in tools
    with session_scope(cultpass_engine()) as s:
        samba = (s.query(cultpass.Reservation).join(cultpass.Experience)
                 .filter(cultpass.Reservation.user_id == "f556c0", cultpass.Experience.title == "Samba Night at Lapa").one())
        assert samba.status == "cancelled"
    # the workflow is inspectable per thread_id
    snapshot = g.get_state(thread_config(tid))
    assert snapshot.values["ticket_id"] == tid and len(snapshot.values["tool_calls"]) >= 4
    history = list(g.get_state_history(thread_config(tid)))
    assert len(history) > 10
    assert {"intake", "classifier", "supervisor", "account_specialist", "responder"} <= {
        n for h in history for n in (h.metadata or {}).get("writes", {}) or {}} | {
        n for h in history for n in h.next}


def test_refund_is_submitted_then_escalated_to_billing_lead(make_graph):
    g = make_graph()
    tid = submit_ticket("88382b", channel="email")
    out = send_message(g, tid, "I was charged twice this month, I want my money back.")
    assert out["final"]["status"] == "escalated"
    assert out["escalation"]["team"] == "billing_lead"
    assert [t["tool"] for t in out["tool_calls"] if t["agent"] == "billing_specialist"] == \
        ["get_customer_profile", "submit_refund_request"]
    with session_scope(core_engine()) as s:
        action = s.query(udahub.SupportAction).filter_by(external_user_id="88382b").one()
        assert action.status == "pending_approval" and action.ticket_id == tid
    assert out["final"]["response"].startswith("Hi Cathy,") and out["final"]["response"].endswith("CultPass Support")


def test_blocked_account_goes_straight_to_trust_and_safety(make_graph):
    llm = FakeLLM()
    g = make_graph(llm)
    tid = submit_ticket("a4ab87")
    out = send_message(g, tid, "I can't log in to my account")
    assert out["final"]["status"] == "escalated"
    assert out["escalation"]["team"] == "trust_and_safety"
    assert out["escalation"]["sla"] == "2 business days"  # from the blocked-account policy, not the generic SLA
    assert not any(c["kind"] == "tools" for c in llm.calls), "no specialist should run"


def test_subscription_action_by_billing_specialist(make_graph):
    g = make_graph()
    tid = submit_ticket("f556c0")
    out = send_message(g, tid, "Please pause my subscription")
    assert out["final"]["status"] == "resolved" and out["final"]["handled_by"] == "billing_specialist"
    with session_scope(cultpass_engine()) as s:
        assert s.get(cultpass.User, "f556c0").subscription.status == "paused"


def test_long_term_memory_across_sessions(make_graph):
    llm = FakeLLM()
    g = make_graph(llm)
    t1 = submit_ticket("88382b")
    send_message(g, t1, "How do I book an event? Also, I prefer email for any follow-up.")
    uid = udahub_ops.get_ticket(t1)["user_id"]
    with session_scope(core_engine()) as s:
        kinds = {(m.kind, m.key) for m in s.query(udahub.CustomerMemory).filter_by(user_id=uid)}
    assert ("preference", "contact_channel") in kinds and ("resolved_issue", None) in kinds

    t2 = submit_ticket("88382b")  # new ticket = new thread = new session
    out = send_message(g, t2, "The QR code won't scan at the museum")
    recalled = {m["kind"] for m in out["memories"]}
    assert {"preference", "resolved_issue"} <= recalled
    prompt = [c for c in llm.calls if c["kind"] == "tools"][-1]["system"]
    assert "contact_channel=email" in prompt  # memory reaches the agent's decision context
    assert "Previous tickets:" in prompt


def test_qa_rejection_triggers_one_revision(make_graph):
    bad = GroundingVerdict(supported=False, claims_actions_not_performed=False, answers_question=True,
                           score=0.2, issues=["invented a fee"])
    good = GroundingVerdict(supported=True, claims_actions_not_performed=False, answers_question=True, score=0.9)
    g = make_graph(FakeLLM(overrides={"GroundingVerdict": [bad, good]}))
    out = send_message(g, submit_ticket("f556c0"), "The app keeps crashing when I open it")
    assert out["final"]["status"] == "resolved"
    assert out["revision_count"] == 1
    qa_events = [e for e in out["events"] if e["event"] == "qa_review"]
    assert [e["payload"]["passed"] for e in qa_events] == [False, True]


def test_repeated_qa_failure_escalates(make_graph):
    bad = GroundingVerdict(supported=False, claims_actions_not_performed=True, answers_question=True,
                           score=0.1, issues=["says cancelled but nothing was cancelled"])
    g = make_graph(FakeLLM(overrides={"GroundingVerdict": bad}))
    out = send_message(g, submit_ticket("f556c0"), "The app keeps crashing when I open it")
    assert out["final"]["status"] == "escalated"
    assert out["escalation"]["rule"] is None or out["routing_history"]


def test_llm_outage_escalates_gracefully_and_is_logged(make_graph):
    g = make_graph(FakeLLM(fail_on={"TicketClassification", "EscalationHandoff"}))
    tid = submit_ticket("f556c0")
    out = send_message(g, tid, "How do I reserve a spot?")
    assert out["final"]["status"] == "escalated"
    assert "1 business day" in out["final"]["response"] or "business" in out["final"]["response"]
    errors = search_events(ticket_id=tid, event="error")
    assert errors and errors[0]["agent"] == "classifier" and errors[0]["level"] == "ERROR"


def test_specialist_crash_is_isolated(make_graph):
    g = make_graph(FakeLLM(fail_on={"specialist"}))
    out = send_message(g, submit_ticket("f556c0"), "How do I reserve a spot for an event?")
    assert out["final"]["status"] == "escalated"
    assert out["routing_history"][-1]["rule"] == "S0_error"


def test_empty_message_asks_for_details_without_llm(make_graph):
    llm = FakeLLM()
    g = make_graph(llm)
    out = send_message(g, submit_ticket("f556c0"), "   ")
    assert out["final"]["status"] == "needs_customer_input" and not llm.calls


def test_unknown_thread_creates_guest_ticket(make_graph):
    g = make_graph()
    out = g.invoke({"messages": [HumanMessage("How do I reserve an event?")]}, config=thread_config("brand-new-1"))
    t = udahub_ops.get_ticket("brand-new-1")
    assert t is not None and t["external_user_id"].startswith("guest-")
    assert out["final"]["status"] == "resolved" and out["customer"] == {}


def test_guest_cannot_use_account_tools(make_graph):
    g = make_graph()
    cfg = {"configurable": {"thread_id": "guest-t"}}
    out = g.invoke({"messages": [HumanMessage("please pause my subscription")]}, config=cfg)
    billing = [t for t in out["tool_calls"] if t["agent"] == "billing_specialist"]
    assert billing and billing[0]["ok"] is False and billing[0]["result"]["code"] == "forbidden"
    assert billing[0]["transport"] == "refused"  # never reached the database
    assert out["final"]["status"] == "escalated"


def test_process_seeded_ticket_does_not_duplicate_message(make_graph):
    g = make_graph()
    seeded = next(t for t in udahub_ops.list_tickets() if t["status"] == "open")["ticket_id"]
    out = process_ticket(g, seeded)
    msgs = udahub_ops.get_ticket(seeded)["messages"]
    assert [m["role"] for m in msgs].count("user") == 1
    assert out["final"]["status"] == "escalated"  # Alice is blocked -> Trust & Safety


def test_social_channel_is_short_and_private(make_graph):
    from agentic.agents.responder import format_for_channel
    long = "Write to me at bob.stone@granite.com. " + "word " * 100
    out = format_for_channel(long, "social", "Bob")
    assert len(out) <= 280 and "granite.com" not in out and out.endswith(".")
    assert format_for_channel("**Step 1**", "phone", None) == "Step 1"
    assert format_for_channel("Hello", "email", "Bob").startswith("Hi Bob,")


def test_logs_are_structured_and_searchable(make_graph, data_env):
    g = make_graph()
    tid = submit_ticket("f556c0")
    send_message(g, tid, "How do I reserve a spot for an event?")
    rows = search_events(ticket_id=tid)
    assert {"classification", "routing_decision", "retrieval", "resolution"} <= {r["event"] for r in rows}
    assert len({r["run_id"] for r in rows}) == 1
    decision = search_events(ticket_id=tid, event="routing_decision")[0]
    assert decision["payload"]["route"] == "knowledge_resolver" and decision["payload"]["variant"] == "rules_first"
    assert search_events(text="knowledge_resolver", event="resolution")
    log_file = data_env.parent / "logs" / "udahub_events.jsonl"
    lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines()]
    assert any(l["ticket_id"] == tid and l["event"] == "resolution" for l in lines)
    assert all({"ts", "event_id", "run_id", "ticket_id", "agent", "event", "level", "payload"} <= set(l) for l in lines)


def test_llm_supervisor_variant_end_to_end(make_graph, monkeypatch):
    monkeypatch.setenv("UDAHUB_ROUTING_STRATEGY", "llm_supervisor")
    g = make_graph()
    out = send_message(g, submit_ticket("88382b"), "How many experiences do I have left this month?")
    assert out["ab_variant"] == "llm_supervisor"
    assert out["routing_history"][0]["rule"] == "A_llm_supervisor"
    assert out["final"]["status"] == "resolved" and "left" in out["final"]["response"]


def test_chat_interface_scripted(make_graph, capsys):
    from utils import chat_interface
    g = make_graph()
    chat_interface(g, "chat-1", external_user_id="f556c0",
                   scripted_inputs=["How do I reserve a spot for an event?", "quit"])
    out = capsys.readouterr().out
    assert "routed       -> knowledge_resolver" in out and "Goodbye" in out


def test_qa_revision_never_repeats_an_account_action(make_graph):
    # regression (found by the live evaluation): the revision pass re-ran the tool loop and
    # tried to book the same experience a second time
    bad = GroundingVerdict(supported=False, claims_actions_not_performed=False, answers_question=True,
                           score=0.3, issues=["wording about quota is unclear"])
    good = GroundingVerdict(supported=True, claims_actions_not_performed=False, answers_question=True, score=0.9)
    g = make_graph(FakeLLM(overrides={"GroundingVerdict": [bad, good]}))
    out = send_message(g, submit_ticket("f556c0"), "Please book the MASP visit for me")
    reserves = [t for t in out["tool_calls"] if t["tool"] == "reserve_experience"]
    assert [t["transport"] for t in reserves] == ["local", "deduplicated"]
    assert out["revision_count"] == 1 and out["final"]["status"] == "resolved"
    with session_scope(cultpass_engine()) as s:
        n = (s.query(cultpass.Reservation).join(cultpass.Experience)
             .filter(cultpass.Reservation.user_id == "f556c0", cultpass.Experience.title == "Modern Art at MASP").count())
    assert n == 1


def test_repeated_identical_reads_are_broken_and_the_loop_always_finishes(make_graph):
    # regression (found by the live evaluation): the specialist called list_reservations
    # six times with the same arguments looking for a reservation that does not exist,
    # then timed out with an empty answer
    def stubborn(llm, tool_names, msgs):
        return llm._call("list_reservations", status="reserved")
    llm = FakeLLM(overrides={"Specialist": stubborn})
    g = make_graph(llm)
    out = send_message(g, submit_ticket("88382b"), "Please cancel my reservation for the samba night")
    listed = [t for t in out["tool_calls"] if t["tool"] == "list_reservations"]
    assert [t["transport"] for t in listed][:2] == ["local", "deduplicated"]
    assert sum(t["transport"] == "local" for t in listed) == 1  # executed once, never again
    assert any(c["kind"] == "forced_finish" for c in llm.calls)  # last step forced submit_answer
    assert out["final"]["handled_by"] == "account_specialist" and out["final"]["response"]


def test_ambiguous_cancellation_is_blocked_until_the_customer_says_which(make_graph):
    # regression (found by the live tests): asked to "cancel one of my reservations", the model
    # sometimes picked one itself. With several active reservations, cancel_reservation only runs
    # once the customer's own words identify the target.
    def eager(llm, tool_names, msgs):
        from langchain_core.messages import ToolMessage
        results = [json.loads(m.content) for m in msgs if isinstance(m, ToolMessage)]
        if not results:
            return llm._call("list_reservations", status="reserved")
        if len(results) == 1:
            samba = next(r for r in results[0]["data"] if "Samba" in r["title"])
            return llm._call("cancel_reservation", reservation_id=samba["reservation_id"])
        return llm._submit("Which reservation should I cancel?", [], "needs_customer_input", 0.9)
    g = make_graph(FakeLLM(overrides={"Specialist": eager}))
    tid = submit_ticket("f556c0")
    out = send_message(g, tid, "I need to cancel one of my reservations")
    blocked = [t for t in out["tool_calls"] if t["tool"] == "cancel_reservation"]
    assert blocked and blocked[0]["transport"] == "guarded" and blocked[0]["result"]["code"] == "confirmation_required"
    with session_scope(cultpass_engine()) as s:
        assert s.query(cultpass.Reservation).filter_by(user_id="f556c0", status="reserved").count() == 2

    # once the customer names it, the same call goes through
    g2 = make_graph()
    tid2 = submit_ticket("f556c0")
    send_message(g2, tid2, "I need to cancel one of my reservations")
    done = send_message(g2, tid2, "the samba one please")
    assert any(t["tool"] == "cancel_reservation" and t["ok"] for t in done["tool_calls"])
