"""Rubric: classification, metadata-aware routing, guardrails and the A/B split."""
import pytest

from agentic.agents.classifier import make_classifier, priority_score
from agentic.agents.intake import assign_variant, normalise
from agentic.agents.supervisor import make_supervisor, rules_first
from agentic.services import Services
from agentic.state import TicketClassification
from tests.fakes import FakeLLM, classify


def _days_ago(n):
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(days=n)).isoformat(timespec="minutes")


def cls(**kw):
    base = classify("hello")
    base.update(kw)
    return base


def decide(data_env, *, classification, customer=None, ticket=None, retrieval_conf=0.9, variant="rules_first",
           specialist_result=None, hops=0, error=None, llm=None, history=None):
    services = Services(llm=llm or FakeLLM(), embedder=None)
    node = make_supervisor(services)
    from langchain_core.messages import HumanMessage
    state = {"ticket_id": "t", "turn": 1, "classification": classification, "customer": customer or {"is_blocked": False},
             "ticket": ticket or {"prior_unresolved": 0}, "retrieval": {"confidence": retrieval_conf, "articles": []},
             "ab_variant": variant, "specialist_result": specialist_result or {}, "hops": hops, "error": error,
             "history": history or [],
             "messages": [HumanMessage("hi")]}
    return node(state)


@pytest.mark.parametrize("c, route", [
    (cls(category="login_access"), "knowledge_resolver"),
    (cls(category="technical_issue"), "knowledge_resolver"),
    (cls(category="general_inquiry"), "knowledge_resolver"),
    (cls(category="reservation_booking"), "knowledge_resolver"),
    (cls(category="reservation_booking", needs_account_data=True), "account_specialist"),
    (cls(category="reservation_change", requested_action="cancel_reservation"), "account_specialist"),
    (cls(category="subscription_management", requested_action="check_account_status"), "account_specialist"),
    (cls(category="subscription_management", needs_account_data=True), "billing_specialist"),
    (cls(category="subscription_management", requested_action="pause_subscription"), "billing_specialist"),
    (cls(category="billing_payment"), "billing_specialist"),
    (cls(category="refund_request", requested_action="refund"), "billing_specialist"),
])
def test_rules_first_table(c, route):
    assert rules_first(c)[0] == route


@pytest.mark.parametrize("kwargs, rule, team", [
    (dict(classification=cls(category="safety_incident")), "G1_safety", "trust_and_safety"),
    (dict(classification=cls(category="account_security")), "G2_security", "trust_and_safety"),
    (dict(classification=cls(category="login_access"), customer={"is_blocked": True}), "G3_blocked", "trust_and_safety"),
    (dict(classification=cls(category="privacy_request")), "G4_privacy", "privacy_team"),
    (dict(classification=cls(legal_or_chargeback_threat=True)), "G5_legal", "billing_lead"),
    (dict(classification=cls(wants_human=True)), "G6_human", "tier2_support"),
    (dict(classification=cls(urgency="critical")), "G7_critical", "tier2_support"),
    (dict(classification=cls(category="login_access"), history=[
        {"status": "escalated", "issue_type": "login_access", "created_at": _days_ago(2)},
        {"status": "pending_customer", "issue_type": "login_access", "created_at": _days_ago(5)}]), "G8_repeat", "tier2_support"),
    (dict(classification=cls(category="login_access"), retrieval_conf=0.1), "K1_no_article", "tier2_support"),
    (dict(classification=cls(), error="boom"), "S0_error", "tier2_support"),
    (dict(classification=cls(), hops=4), "S1_hops", "tier2_support"),
])
def test_guardrails_escalate(data_env, kwargs, rule, team):
    out = decide(data_env, **kwargs)
    assert out["route"] == "escalation"
    assert out["routing_history"][0]["rule"] == rule
    assert out["escalation"]["team"] == team


def test_unrelated_or_old_unresolved_tickets_do_not_force_escalation(data_env):
    # regression: two unrelated escalations used to send every later question to a human
    history = [{"status": "escalated", "issue_type": "privacy_request", "created_at": _days_ago(1)},
               {"status": "escalated", "issue_type": "other", "created_at": _days_ago(1)},
               {"status": "escalated", "issue_type": "login_access", "created_at": _days_ago(40)},
               {"status": "escalated", "issue_type": "login_access", "created_at": _days_ago(30)}]
    out = decide(data_env, classification=cls(category="login_access"), history=history)
    assert out["route"] == "knowledge_resolver"


def test_blocked_customer_can_still_ask_general_questions(data_env):
    out = decide(data_env, classification=cls(category="general_inquiry"), customer={"is_blocked": True})
    assert out["route"] == "knowledge_resolver"


def test_knowledge_gate_does_not_block_account_work(data_env):
    out = decide(data_env, classification=cls(category="reservation_change", requested_action="cancel_reservation"),
                 retrieval_conf=0.05)
    assert out["route"] == "account_specialist"


def test_after_specialist(data_env):
    assert decide(data_env, classification=cls(), specialist_result={"agent": "knowledge_resolver", "outcome": "resolved"})["route"] == "qa_reviewer"
    esc = decide(data_env, classification=cls(), specialist_result={"agent": "billing_specialist", "outcome": "escalate",
                                                                    "escalation_reason": "refund awaiting approval"})
    assert esc["route"] == "escalation" and esc["escalation"]["reason"] == "refund awaiting approval"
    hand = decide(data_env, classification=cls(), specialist_result={"agent": "knowledge_resolver", "outcome": "resolved",
                                                                     "handoff_to": "account_specialist"})
    assert hand["route"] == "account_specialist" and hand["specialist_result"] == {}


def test_llm_supervisor_variant_uses_structured_decision(data_env):
    llm = FakeLLM()
    out = decide(data_env, classification=cls(category="billing_payment"), variant="llm_supervisor", llm=llm)
    assert out["route"] == "billing_specialist"
    assert out["routing_history"][0]["rule"] == "A_llm_supervisor"
    assert any(c["schema"] == "RoutingDecision" for c in llm.calls)


def test_ab_assignment_is_deterministic_and_balanced(monkeypatch):
    monkeypatch.setenv("UDAHUB_ROUTING_STRATEGY", "ab")
    ids = [f"ticket-{i}" for i in range(200)]
    variants = [assign_variant(i) for i in ids]
    assert variants == [assign_variant(i) for i in ids]
    share = variants.count("rules_first") / len(ids)
    assert 0.4 < share < 0.6
    monkeypatch.setenv("UDAHUB_ROUTING_STRATEGY", "llm_supervisor")
    assert assign_variant("x") == "llm_supervisor"


# ------------------------------------------------------------ classifier ---

def test_priority_uses_metadata():
    calm = priority_score(cls(urgency="low"), {}, {"prior_unresolved": 0, "age_hours": 0})
    assert calm["level"] == "P4"
    hot = priority_score(cls(urgency="high", sentiment="very_negative"), {"subscription": {"tier": "premium"}},
                         {"prior_unresolved": 1, "age_hours": 30, "provided_urgency": "high"})
    assert hot["level"] == "P1"
    assert any("premium" in r for r in hot["reasons"]) and any("open 30" in r for r in hot["reasons"])


def _run_classifier(data_env, text, llm):
    from langchain_core.messages import HumanMessage
    from agentic.tools.udahub_ops import create_ticket
    tid = create_ticket("f556c0")
    node = make_classifier(Services(llm=llm, embedder=None))
    return node({"ticket_id": tid, "turn": 1, "user_message": text, "ticket": {"channel": "chat"},
                 "customer": {}, "messages": [HumanMessage(text)]})


def test_backstops_override_a_wrong_classification(data_env):
    wrong = TicketClassification(**cls(category="reservation_booking", urgency="low"))
    out = _run_classifier(data_env, "A staff member harassed me at the samba night. I want to speak to a manager, or I'll call my lawyer.",
                          FakeLLM(overrides={"TicketClassification": wrong}))
    c = out["classification"]
    assert c["category"] == "safety_incident" and c["urgency"] == "critical"
    assert c["wants_human"] and c["legal_or_chargeback_threat"]
    assert len(c["overrides"]) == 3


def test_classifier_records_metadata_on_ticket(data_env):
    from agentic.tools.udahub_ops import get_ticket, tag_value
    out = _run_classifier(data_env, "This is ridiculous, my card was declined URGENTLY", FakeLLM())
    assert out["classification"]["category"] == "billing_payment"
    t = get_ticket(_last_ticket())
    assert t["main_issue_type"] == "billing_payment"
    assert tag_value(t["tags"], "sentiment") == "very_negative"
    assert tag_value(t["tags"], "priority") in ("P1", "P2")


def _last_ticket():
    from agentic.tools.udahub_ops import list_tickets
    return list_tickets()[-1]["ticket_id"]


def test_email_normalisation_strips_quotes_and_signature():
    raw = "Hi team,\nMy QR code fails.\n\n--\nBob\nSent from my iPhone\n\nOn Mon, 1 Sep 2026 CultPass wrote:\n> old reply"
    assert normalise(raw, "email") == "Hi team,\nMy QR code fails."
    assert normalise("  keep   this  ", "chat") == "keep this"
