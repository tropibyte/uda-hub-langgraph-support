"""Graph state and the structured messages agents pass to each other."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ticket taxonomy (design doc section 3)
# ---------------------------------------------------------------------------
Category = Literal[
    "login_access",             # can't log in, password reset
    "account_management",       # profile, email change, notifications settings
    "account_security",         # hacked, suspicious activity, unrecognised reservations
    "billing_payment",          # failed payment, double charge, payment method
    "refund_request",           # explicit ask for money back
    "subscription_management",  # plans, upgrade/downgrade, pause, cancel, resume, quota
    "reservation_booking",      # how to reserve, availability, waitlist, premium fees
    "reservation_change",       # cancel/transfer an existing reservation, partner cancelled
    "technical_issue",          # app crash, QR code, emails not arriving
    "safety_incident",          # injury, harassment, unsafe venue
    "privacy_request",          # delete account, data copy (LGPD/GDPR)
    "general_inquiry",          # coverage, accessibility, gift cards, what is included
    "other",                    # anything outside CultPass support
]
CATEGORIES: tuple[str, ...] = Category.__args__  # type: ignore[attr-defined]

RequestedAction = Literal[
    "none", "reserve_experience", "cancel_reservation", "pause_subscription", "resume_subscription",
    "cancel_subscription", "refund", "check_account_status", "other",
]


class TicketClassification(BaseModel):
    """Classifier agent output."""
    category: Category = Field(description="Primary support category of the customer's latest request.")
    secondary_category: Optional[Category] = Field(default=None, description="Second category if the ticket clearly mixes two issues, else null.")
    intent: str = Field(description="One short sentence: what the customer wants.")
    urgency: Literal["low", "normal", "high", "critical"] = Field(
        description="critical: safety, security breach, fraud. high: locked out, event today, money taken wrongly. normal: most requests. low: general curiosity.")
    complexity: Literal["simple", "moderate", "complex"] = Field(
        description="simple: one FAQ answer. moderate: needs account data or one action. complex: several issues, disputes, or policy exceptions.")
    sentiment: Literal["positive", "neutral", "negative", "very_negative"] = Field(description="Customer's emotional tone.")
    needs_account_data: bool = Field(description="True if answering requires looking at THIS customer's account, subscription or reservations.")
    requested_action: RequestedAction = Field(description="Concrete account action the customer asks us to perform, or 'none'.")
    wants_human: bool = Field(description="True if the customer explicitly asks for a human/person/manager.")
    legal_or_chargeback_threat: bool = Field(description="True if the customer threatens legal action, a chargeback, or a regulator complaint.")
    language: str = Field(description="ISO 639-1 language code of the customer's message, e.g. 'en', 'pt'.")
    confidence: float = Field(ge=0, le=1, description="Confidence in the category, 0-1.")
    rationale: str = Field(description="One sentence explaining the classification.")


class RoutingDecision(BaseModel):
    """LLM supervisor output (llm_supervisor strategy)."""
    next_agent: Literal["knowledge_resolver", "account_specialist", "billing_specialist", "escalation"]
    reason: str = Field(description="One sentence justification.")
    confidence: float = Field(ge=0, le=1)


class SpecialistAnswer(BaseModel):
    """What every specialist hands back to the supervisor (via the submit_answer tool)."""
    response: str = Field(description="The reply to send to the customer. Plain text, no markdown headings.")
    cited_articles: list[str] = Field(default_factory=list, description="KB refs (e.g. 'KB-1a2b3c') whose content the reply is based on.")
    outcome: Literal["resolved", "needs_customer_input", "escalate"] = Field(
        description="resolved: the request is answered/done. needs_customer_input: you asked the customer a question. escalate: a human must take over.")
    confidence: float = Field(ge=0, le=1, description="How sure you are the reply is correct, complete and supported by the cited articles/tool results.")
    escalation_reason: Optional[str] = Field(default=None, description="Why a human is needed (only when outcome='escalate').")
    handoff_to: Optional[Literal["knowledge_resolver", "account_specialist", "billing_specialist"]] = Field(
        default=None, description="Set only if another specialist is clearly better placed to finish this request.")


class GroundingVerdict(BaseModel):
    """QA reviewer LLM-judge output."""
    supported: bool = Field(description="True if every factual claim and instruction in the reply is supported by the cited articles or tool results.")
    claims_actions_not_performed: bool = Field(description="True if the reply says an action was done (cancelled, reserved, refunded, paused...) that no successful tool result shows.")
    answers_question: bool = Field(description="True if the reply actually addresses the customer's latest request.")
    score: float = Field(ge=0, le=1, description="Overall groundedness and helpfulness, 0-1.")
    issues: list[str] = Field(default_factory=list, description="Specific problems to fix, empty if none.")


class EscalationHandoff(BaseModel):
    """Escalation agent output. Team and priority are decided by policy code, not the LLM."""
    customer_message: str = Field(description="Message to the customer: acknowledge the issue, say which team will handle it and the expected response time from the policy article.")
    summary: str = Field(description="Internal handoff summary for the human agent: the issue, what was already tried, relevant account facts, customer sentiment.")
    cited_articles: list[str] = Field(default_factory=list, description="KB refs the customer message is based on.")


class Preference(BaseModel):
    key: str = Field(description="snake_case key, e.g. contact_channel, language, preferred_city, interests, accessibility_needs, name_to_use")
    value: str = Field(description="The preference in a few words, as the customer stated it.")


class MemoryExtraction(BaseModel):
    """Memory curator output."""
    preferences: list[Preference] = Field(
        default_factory=list,
        description="Durable preferences the customer stated about themselves in their messages. Never guesses; empty if none.")
    facts: list[str] = Field(default_factory=list, description="Other stable facts worth remembering for future tickets; usually empty.")


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class UDAHubState(TypedDict, total=False):
    # Conversation (short-term memory): customer messages and final replies only.
    messages: Annotated[list[AnyMessage], add_messages]

    # Ticket and customer context, loaded by intake each turn
    ticket_id: str
    external_user_id: Optional[str]
    ticket: dict[str, Any]
    customer: dict[str, Any]
    history: list[dict[str, Any]]
    memories: list[dict[str, Any]]
    ab_variant: str

    # Per-turn working state (reset by intake at the start of every turn)
    turn: int
    run_id: str
    user_message: str
    classification: dict[str, Any]
    priority: dict[str, Any]
    retrieval: dict[str, Any]
    route: str
    hops: int
    specialist_result: dict[str, Any]
    last_specialist: Optional[str]
    qa: dict[str, Any]
    revision_count: int
    escalation: dict[str, Any]
    final: dict[str, Any]
    error: Optional[str]

    # Append-only audit trails (whole session; each entry carries its turn)
    routing_history: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]
