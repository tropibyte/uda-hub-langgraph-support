"""Classifier agent: category, urgency, sentiment, complexity and priority.

The LLM produces a ``TicketClassification``. Two deterministic layers sit on
top of it because some mistakes are too costly to leave to a model:

* **Safety backstops** - regexes for safety incidents, account compromise,
  legal/chargeback threats and "I want a human". A hit the model missed
  overrides the classification and is logged as an override.
* **Priority scoring** - combines urgency, sentiment, customer tier, repeat
  contacts and ticket age (metadata) into P1-P4, used for escalation
  priority and recorded on the ticket.
"""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agentic.agents.common import emit, memory_block, safe_node, transcript
from agentic.services import Services
from agentic.state import CATEGORIES, TicketClassification, UDAHubState
from agentic.tools import udahub_ops

BACKSTOPS = [
    ("safety_incident", re.compile(r"\b(harass\w*|assault\w*|injur\w*|attack\w*|unsafe|groped|threaten\w* me|hurt me|got hurt|was hurt|discriminat\w*)\b", re.I)),
    ("account_security", re.compile(r"\b(hack\w*|compromised|someone (else )?(logged|is using|used|accessed)|unauthori[sz]ed|didn'?t make (this|these|that) (reservation|booking)s?|password (was )?changed without)\b", re.I)),
]
LEGAL = re.compile(r"\b(lawyer|attorney|sue|suing|legal action|procon|chargeback|small claims|consumer protection)\b", re.I)
HUMAN = re.compile(r"\b(human|real person|live agent|speak to (someone|a person|an agent)|talk to (someone|a person|an agent)|manager|supervisor)\b", re.I)

SYSTEM = """You are the Classifier agent of UDA-Hub, the support brain for CultPass, a cultural-experiences
subscription in Brazil (members reserve museum visits, concerts, tours, etc. within a monthly quota).
Classify the customer's LATEST message, using the conversation only to resolve references
("yes, the second one", "same problem again").

Category guide:
- login_access: cannot log in, password reset, reset email not arriving
- account_management: profile, email change, notification settings
- account_security: hacked account, reservations or charges the customer did not make
- billing_payment: failed payment, card declined, double or unexpected charge, payment method
- refund_request: the customer explicitly asks for money back
- subscription_management: plans/tiers, upgrade/downgrade, pause, cancel, resume, monthly quota
- reservation_booking: how to reserve, availability, sold out/waitlist, premium fees, booking a specific event
- reservation_change: cancel or transfer an existing reservation, partner cancelled/rescheduled event
- technical_issue: app crash/freeze, QR code not scanning, emails/notifications not arriving
- safety_incident: injury, harassment, discrimination or unsafe conditions at a venue
- privacy_request: delete my account/data, copy of my personal data (LGPD/GDPR)
- general_inquiry: what is included, where CultPass works, accessibility, gift cards, corporate plans
- other: unrelated to CultPass support
needs_account_data is true when the answer depends on THIS customer's plan, quota, reservations or
subscription status (e.g. "how many experiences do I have left", "cancel my Samba reservation").
"""


def priority_score(cls: dict, customer: dict, ticket: dict) -> dict:
    """Metadata-aware priority (design doc section 4.2)."""
    score = {"low": 1, "normal": 2, "high": 3, "critical": 4}[cls["urgency"]]
    reasons = [f"urgency={cls['urgency']}"]
    if cls["sentiment"] == "very_negative":
        score += 1; reasons.append("very negative sentiment")
    elif cls["sentiment"] == "negative":
        score += 0.5; reasons.append("negative sentiment")
    if ((customer or {}).get("subscription") or {}).get("tier") == "premium":
        score += 0.5; reasons.append("premium member")
    if ticket.get("prior_unresolved", 0) >= 1:
        score += 1; reasons.append(f"{ticket['prior_unresolved']} other unresolved ticket(s)")
    if ticket.get("age_hours", 0) >= 24:
        score += 0.5; reasons.append(f"ticket open {ticket['age_hours']}h")
    if ticket.get("provided_urgency") in ("high", "urgent", "critical"):
        score += 0.5; reasons.append(f"flagged {ticket['provided_urgency']} by source system")
    level = "P1" if score >= 4.5 else "P2" if score >= 3.5 else "P3" if score >= 2 else "P4"
    return {"level": level, "score": score, "reasons": reasons}


def make_classifier(services: Services):
    @safe_node("classifier")
    def classifier(state: UDAHubState) -> dict:
        if state.get("error"):
            return {}
        text = state.get("user_message", "")
        t = state.get("ticket") or {}
        prompt = [
            SystemMessage(SYSTEM),
            HumanMessage(
                f"Channel: {t.get('channel')} | urgency flag from source system: {t.get('provided_urgency') or 'none'} | "
                f"subject: {t.get('subject') or 'none'}\n"
                f"Customer context: {memory_block(state)}\n\n"
                f"Conversation so far:\n{transcript(state, 8)}\n\n"
                f"LATEST customer message to classify:\n{text}"),
        ]
        cls = services.structured(TicketClassification, prompt).model_dump()

        overrides = []
        for category, rx in BACKSTOPS:
            if rx.search(text) and cls["category"] != category:
                overrides.append(f"category {cls['category']} -> {category} (keyword backstop)")
                cls["secondary_category"], cls["category"] = cls["category"], category
                cls["urgency"] = "critical"
                break
        if LEGAL.search(text) and not cls["legal_or_chargeback_threat"]:
            cls["legal_or_chargeback_threat"] = True
            overrides.append("legal_or_chargeback_threat -> True (keyword backstop)")
        if HUMAN.search(text) and not cls["wants_human"]:
            cls["wants_human"] = True
            overrides.append("wants_human -> True (keyword backstop)")
        if cls["category"] not in CATEGORIES:
            cls["category"] = "other"
        cls["overrides"] = overrides

        prio = priority_score(cls, state.get("customer") or {}, t)
        udahub_ops.update_ticket_metadata(state["ticket_id"], main_issue_type=cls["category"], status="in_progress",
                                          add_tags=[f"priority:{prio['level']}", f"sentiment:{cls['sentiment']}",
                                                    f"urgency:{cls['urgency']}"])
        ev = emit(state, "classifier", "classification", {
            "category": cls["category"], "secondary": cls.get("secondary_category"), "intent": cls["intent"],
            "urgency": cls["urgency"], "complexity": cls["complexity"], "sentiment": cls["sentiment"],
            "needs_account_data": cls["needs_account_data"], "requested_action": cls["requested_action"],
            "wants_human": cls["wants_human"], "legal": cls["legal_or_chargeback_threat"],
            "confidence": cls["confidence"], "priority": prio["level"], "priority_reasons": prio["reasons"],
            "overrides": overrides})
        return {"classification": cls, "priority": prio, "events": [ev]}

    return classifier
