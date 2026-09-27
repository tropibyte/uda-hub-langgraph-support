"""Escalation agent: hands the ticket to a human with everything they need.

Team and priority are policy decisions made in code (the supervisor's rule
or the reason text); the LLM writes the customer message, grounded on the
escalation-policy article plus the article for the topic, and the internal
handoff summary. If the LLM is unavailable a template built from the
policy article's suggested phrasing is used, so escalation cannot fail.
"""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agentic.agents.common import (articles_block, compact, customer_block, emit, memory_block, ticket_block,
                                   transcript)
from agentic.services import Services
from agentic.state import EscalationHandoff, UDAHubState
from agentic.tools import udahub_ops

TEAM_BY_CATEGORY = {
    "safety_incident": "trust_and_safety", "account_security": "trust_and_safety", "privacy_request": "privacy_team",
    "refund_request": "billing_lead", "billing_payment": "billing_lead", "technical_issue": "technical_support",
}
TEAM_LABEL = {
    "trust_and_safety": "Trust & Safety team", "privacy_team": "privacy team", "billing_lead": "billing lead",
    "technical_support": "technical support team", "partnerships": "partnerships team", "tier2_support": "senior support team",
}
POLICY_TITLES = {
    "trust_and_safety": ["Account Blocked or Suspended", "Suspicious Activity", "Reporting a Problem or Safety Incident"],
    "privacy_team": ["Deleting Your Account"],
    "billing_lead": ["Refund Policy"],
}

SYSTEM = """You are the Escalation agent of UDA-Hub (CultPass support). A human colleague will take over this
ticket. Write (1) a short, warm message to the customer and (2) an internal handoff summary.

Customer message rules:
- Acknowledge the issue with empathy and use the customer's first name. Reply in the language of a stored language preference if there is one, otherwise in the customer's language.
- Say which team will handle it and when they will hear back, using EXACTLY the "response time to
  state" given below (policy code has already chosen it from the right article); never another time.
- Only give "meanwhile" advice that is written in the articles provided (e.g. reset the password, call
  emergency services). Never invent advice, workarounds, links, forums or other resources.
- Never promise an outcome (no "your refund is approved", "we will unblock you") and never reveal
  internal escalation reasons or rules.
- If a specialist draft is given, keep its grounded facts (e.g. a refund request was submitted and is
  pending review, and the timeline once approved).
- If the request is unrelated to CultPass, say kindly that you can only help with CultPass memberships,
  reservations and experiences, and invite them to share any CultPass question. Do not answer it.
- At most ~90 words, plain text.
Summary rules: 3-6 lines for the human agent: the request, what was tried (tools, answers), account
facts, sentiment/priority, why it was escalated.
Put the refs of the articles your customer message relies on in cited_articles."""


def pick_team(esc: dict, cls: dict) -> str:
    if esc.get("team"):
        return esc["team"]
    reason = (esc.get("reason") or "").lower()
    if re.search(r"refund|charge|payment", reason):
        return "billing_lead"
    if re.search(r"accessib|corporate|partner", reason):
        return "partnerships"
    return TEAM_BY_CATEGORY.get(cls.get("category"), "tier2_support")


# Response times stated in team-specific policy articles override the generic
# escalation SLA (e.g. "Account Blocked or Suspended": Trust & Safety, 2 business days).
SLA_BY_RULE = {"G3_blocked": "2 business days"}


def sla_for(priority: str, rule: str | None = None) -> str:
    if rule in SLA_BY_RULE:
        return SLA_BY_RULE[rule]
    return "4 business hours" if priority in ("P1", "P2") else "1 business day"


def make_escalation(services: Services):
    def escalation(state: UDAHubState) -> dict:
        cls = state.get("classification") or {}
        esc = dict(state.get("escalation") or {})
        if not esc.get("reason"):
            esc["reason"] = state.get("error") or "escalated"
        team = pick_team(esc, cls)
        rule = esc.get("rule")
        priority = (state.get("priority") or {}).get("level") or "P3"
        if cls.get("urgency") == "critical" or team == "trust_and_safety":
            priority = "P1" if cls.get("category") in ("safety_incident", "account_security") else min(priority, "P2")

        policy = services.retriever.get_by_title("When and How Support Escalates")
        articles = [policy] if policy else []
        for title in POLICY_TITLES.get(team, []):
            a = services.retriever.get_by_title(title)
            if a and a["ref"] not in {x["ref"] for x in articles}:
                articles.append(a)
        for a in (state.get("retrieval") or {}).get("articles", [])[:2]:
            if a["ref"] not in {x["ref"] for x in articles} and (a.get("relevance") or 0) >= 0.35:
                articles.append(a)

        turn = state.get("turn")
        tools_this_turn = [f"{t['tool']} ok={t['ok']}" for t in state.get("tool_calls", []) if t.get("turn") == turn]
        draft = (state.get("specialist_result") or {}).get("response") or (state.get("qa") or {}).get("draft") or ""
        off_topic = ("\nNOTE: this request is unrelated to CultPass. Do not say the request itself was passed on; "
                     "say kindly that you can only help with CultPass and invite a CultPass question.\n"
                     if cls.get("category") == "other" else "")
        try:
            h = services.structured(EscalationHandoff, [
                SystemMessage(SYSTEM),
                HumanMessage(
                    f"{off_topic}Escalation reason (internal): {esc['reason']}\nAssigned team: {TEAM_LABEL[team]} | priority {priority} "
                    f"| response time to state: {sla_for(priority, rule)}\n"
                    f"{ticket_block(state)}\n{customer_block(state)}\n{memory_block(state)}\n"
                    f"Tools used this turn: {tools_this_turn or 'none'}\n"
                    f"Specialist draft (may contain useful grounded facts, e.g. a refund request id): {draft or 'none'}\n\n"
                    f"Policy / topic articles:\n{articles_block(articles, 1200)}\n\n"
                    f"Conversation:\n{transcript(state, 10)}")])
            message, summary, cited = h.customer_message, h.summary, h.cited_articles
            # The promised response time is policy, never optional. Checked by its number so a
            # reply in another language ("4 horas úteis") counts as stated.
            if not re.search(rf"\b{sla_for(priority, rule).split()[0]}\b", message):
                message = message.rstrip() + f" You'll hear back within {sla_for(priority, rule)}."
        except Exception as exc:  # noqa: BLE001 - template fallback
            name = ((state.get("customer") or {}).get("full_name") or "").split(" ")[0]
            message = (f"{'Hi ' + name + ', ' if name else ''}I've passed your ticket to our {TEAM_LABEL[team]} with a "
                       f"summary of everything so far, so you won't need to repeat yourself. "
                       f"You'll hear back within {sla_for(priority, rule)}.")
            summary = f"Auto-summary (LLM unavailable: {exc.__class__.__name__}). Reason: {esc['reason']}. {ticket_block(state)}"
            cited = [policy["ref"]] if policy else []

        valid_refs = {a["ref"] for a in articles}
        used = [r for r in cited if r in valid_refs] or [a["ref"] for a in articles[:1 + len(POLICY_TITLES.get(team, []))]]
        citations = [{"ref": a["ref"], "title": a["title"]} for a in articles if a["ref"] in used]
        handoff = {"team": team, "priority": priority, "reason": esc["reason"], "rule": esc.get("rule"),
                   "summary": summary, "sla": sla_for(priority, rule)}
        udahub_ops.add_message(state["ticket_id"], "system",
                               f"[HANDOFF -> {TEAM_LABEL[team]} | {priority} | SLA {sla_for(priority, rule)}]\n"
                               f"Reason: {esc['reason']}\n{summary}")
        ev = emit(state, "escalation", "escalation", {"team": team, "priority": priority, "reason": esc["reason"],
                                                      "rule": esc.get("rule"), "summary": compact(summary, 400)})
        return {
            "escalation": handoff, "events": [ev],
            "final": {"status": "escalated", "response": message, "confidence": (state.get("qa") or {}).get("confidence"),
                      "citations": citations, "handled_by": "escalation", "team": team, "priority": priority},
        }

    return escalation
