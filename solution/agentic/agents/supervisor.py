"""Supervisor agent: decides which agent works on the ticket next.

The supervisor is the hub of the Supervisor pattern: every specialist
returns to it, and it decides what happens next. Its policy has three
layers, evaluated in order (design doc section 4):

1. **Guardrails** (always rule-based, both A/B variants): system errors,
   safety incidents, account compromise, blocked accounts, privacy requests,
   legal/chargeback threats, explicit requests for a human, critical urgency,
   repeat unresolved contacts.
2. **Knowledge gate**: if the request would be answered from the knowledge
   base but retrieval confidence is below the threshold, escalate - the
   system never answers without a relevant article.
3. **Assignment**: pick the specialist. This is the A/B-tested step:
   ``rules_first`` uses a category x action routing table, while
   ``llm_supervisor`` asks the LLM for a ``RoutingDecision``.

After a specialist reports back, the supervisor sends the answer to QA,
honours a hand-off to another specialist, or escalates.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agentic.agents.common import emit, safe_node, transcript
from agentic.config import get_settings
from agentic.services import Services
from agentic.state import RoutingDecision, UDAHubState

SPECIALISTS = ("knowledge_resolver", "account_specialist", "billing_specialist")
BLOCKED_SENSITIVE = {"login_access", "account_management", "reservation_booking", "reservation_change",
                     "subscription_management", "billing_payment", "refund_request", "account_security"}

# Guardrails: (rule id, predicate(cls, state) -> bool, team, reason)
GUARDRAILS = [
    ("G1_safety", lambda c, s: c["category"] == "safety_incident", "trust_and_safety", "safety incident reported"),
    ("G2_security", lambda c, s: c["category"] == "account_security", "trust_and_safety", "possible account compromise"),
    ("G3_blocked", lambda c, s: (s.get("customer") or {}).get("is_blocked") and c["category"] in BLOCKED_SENSITIVE,
     "trust_and_safety", "account is blocked; only Trust & Safety can review it"),
    ("G4_privacy", lambda c, s: c["category"] == "privacy_request", "privacy_team", "personal-data request must be handled by the privacy team"),
    ("G5_legal", lambda c, s: c.get("legal_or_chargeback_threat"), "billing_lead", "legal or chargeback threat"),
    ("G6_human", lambda c, s: c.get("wants_human"), "tier2_support", "customer asked for a human agent"),
    ("G7_critical", lambda c, s: c.get("urgency") == "critical", "tier2_support", "critical urgency"),
    ("G8_repeat", lambda c, s: repeat_contacts(c, s) >= 2,
     "tier2_support", "repeat contact: the same issue is still unresolved on earlier tickets"),
]


def repeat_contacts(cls: dict, state: dict, days: int = 14) -> int:
    """Earlier unresolved tickets about the SAME issue in the last ``days`` days.

    Unrelated open escalations (a privacy request, an off-topic question) must
    not force every new question to a human, so only matching issue types count.
    """
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(days=days)
    n = 0
    for h in state.get("history") or []:
        created = datetime.fromisoformat(h["created_at"]) if h.get("created_at") else None
        if (h.get("status") in ("open", "escalated", "pending_customer") and h.get("issue_type") == cls.get("category")
                and (created is None or created >= cutoff)):
            n += 1
    return n

ACCOUNT_ACTIONS = {"reserve_experience", "cancel_reservation", "check_account_status"}
BILLING_ACTIONS = {"pause_subscription", "resume_subscription", "cancel_subscription", "refund"}


def rules_first(cls: dict) -> tuple[str, str]:
    """Category x action routing table (variant A)."""
    cat, action, needs = cls["category"], cls.get("requested_action", "none"), cls.get("needs_account_data")
    if cat in ("billing_payment", "refund_request") or action in BILLING_ACTIONS:
        return "billing_specialist", f"{cat}/{action} is a billing or subscription operation"
    if action in ACCOUNT_ACTIONS:
        return "account_specialist", f"requested action {action} needs the reservation tools"
    if cat == "subscription_management" and needs:
        return "billing_specialist", "subscription question about this customer's own plan"
    if cat in ("reservation_booking", "reservation_change", "account_management") and needs:
        return "account_specialist", f"{cat} needs this customer's account data"
    return "knowledge_resolver", f"{cat} can be answered from the knowledge base"


LLM_SUPERVISOR_PROMPT = """You are the Supervisor agent of UDA-Hub (CultPass customer support). Choose the ONE agent
that should handle the customer's latest request:
- knowledge_resolver: answers how-to / policy / troubleshooting questions from the knowledge base. No account access.
- account_specialist: looks up and changes THIS customer's reservations (list, reserve, cancel) and checks quota/profile.
- billing_specialist: subscription status and changes (pause, resume, cancel), payments, charges, refund requests.
- escalation: a human must handle it (policy exception, dispute, nothing in the knowledge base covers it).
Prefer the least-privileged agent that can fully resolve the request."""


def make_supervisor(services: Services):
    settings = get_settings()

    def decide(state: UDAHubState) -> dict:
        cls = state.get("classification") or {}
        retrieval = state.get("retrieval") or {}
        result = state.get("specialist_result") or {}
        hops = int(state.get("hops") or 0)
        variant = state.get("ab_variant", "rules_first")

        if state.get("error"):
            return {"route": "escalation", "rule": "S0_error", "reason": f"system error: {state['error'][:120]}", "team": "tier2_support"}
        if hops >= settings.max_supervisor_hops:
            return {"route": "escalation", "rule": "S1_hops", "reason": "too many hand-offs between agents", "team": "tier2_support"}

        # --- a specialist has reported back ---------------------------------
        if result:
            if result.get("outcome") == "escalate":
                return {"route": "escalation", "rule": "S2_specialist", "team": None,
                        "reason": result.get("escalation_reason") or f"{result.get('agent')} could not resolve"}
            handoff = result.get("handoff_to")
            if handoff in SPECIALISTS and handoff != result.get("agent"):
                return {"route": handoff, "rule": "S3_handoff", "reason": f"{result.get('agent')} handed off to {handoff}"}
            return {"route": "qa_reviewer", "rule": "S4_review", "reason": f"{result.get('agent')} answered; verify before sending"}

        # --- first assignment for this turn ---------------------------------
        for rule, pred, team, reason in GUARDRAILS:
            if pred(cls, state):
                return {"route": "escalation", "rule": rule, "reason": reason, "team": team}

        if variant == "llm_supervisor":
            d = services.structured(RoutingDecision, [
                SystemMessage(LLM_SUPERVISOR_PROMPT),
                HumanMessage(
                    f"Classification: {cls}\n"
                    f"Top knowledge articles (confidence {retrieval.get('confidence')}): "
                    f"{[a['title'] for a in retrieval.get('articles', [])[:3]]}\n"
                    f"Customer blocked: {(state.get('customer') or {}).get('is_blocked')}\n\n"
                    f"Conversation:\n{transcript(state, 6)}")])
            route, reason, rule = d.next_agent, d.reason, "A_llm_supervisor"
        else:
            route, reason = rules_first(cls)
            rule = "A_rules_first"

        # Knowledge gate: never answer from the KB without a relevant article.
        if route == "knowledge_resolver" and retrieval.get("confidence", 0) < settings.min_retrieval_relevance:
            return {"route": "escalation", "rule": "K1_no_article", "team": "tier2_support",
                    "reason": f"no relevant knowledge article (retrieval confidence {retrieval.get('confidence', 0):.2f} "
                              f"< {settings.min_retrieval_relevance})"}
        if route == "escalation":
            return {"route": "escalation", "rule": rule, "reason": reason, "team": "tier2_support"}
        return {"route": route, "rule": rule, "reason": reason}

    @safe_node("supervisor")
    def supervisor(state: UDAHubState) -> dict:
        d = decide(state)
        hops = int(state.get("hops") or 0) + (1 if d["route"] in SPECIALISTS else 0)
        entry = {"turn": state.get("turn"), "variant": state.get("ab_variant"), "hop": hops, **d}
        ev = emit(state, "supervisor", "routing_decision", {**d, "variant": state.get("ab_variant"), "hop": hops})
        update = {"route": d["route"], "hops": hops, "routing_history": [entry], "events": [ev]}
        if d["route"] in SPECIALISTS and state.get("specialist_result"):
            update["specialist_result"] = {}  # hand-off: the next specialist starts clean
        if d["route"] == "escalation":
            update["escalation"] = {"reason": d["reason"], "rule": d["rule"], "team": d.get("team")}
        return update

    return supervisor
