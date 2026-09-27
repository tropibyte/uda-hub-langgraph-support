"""Deterministic stand-ins for the LLM and the embeddings API.

``FakeLLM`` implements the three methods the agents use (``invoke`` via
``bind_tools`` / ``with_structured_output``) with keyword rules, so the real
graph, real tools and real databases run end to end without an API key.
Every call is recorded in ``llm.calls`` so tests can assert what each agent
was shown. ``overrides`` replaces the answer for one schema, e.g.
``{"GroundingVerdict": GroundingVerdict(supported=False, ...)}``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentic.agents.supervisor import rules_first
from agentic.state import (EscalationHandoff, GroundingVerdict, MemoryExtraction, Preference, RoutingDecision,
                           TicketClassification)

REF = re.compile(r"\[(KB-[0-9a-f]{6})\]")


def latest_text(msgs) -> str:
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            c = m.content
            if "LATEST customer message to classify:\n" in c:
                return c.split("LATEST customer message to classify:\n", 1)[1]
            if "Customer message:\n" in c:
                return c.split("Customer message:\n", 1)[1]
            return c
    return ""


def classify(text: str) -> dict:
    t = text.lower()
    c = dict(category="general_inquiry", secondary_category=None, intent=text[:60], urgency="normal",
             complexity="simple", sentiment="neutral", needs_account_data=False, requested_action="none",
             wants_human=False, legal_or_chargeback_threat=False, language="en", confidence=0.9,
             rationale="keyword rules (fake)")
    rules: list[tuple[str, dict]] = [
        (r"pizza|weather|poem|tax return", dict(category="other", urgency="low")),
        (r"harass|injur", dict(category="safety_incident", urgency="critical")),
        (r"hacked|someone else", dict(category="account_security", urgency="critical")),
        (r"delete my (account|data)", dict(category="privacy_request")),
        (r"refund|money back", dict(category="refund_request", requested_action="refund", needs_account_data=True)),
        (r"charged twice|double charge|declined", dict(category="billing_payment", needs_account_data=True)),
        (r"cancel my (subscription|plan)", dict(category="subscription_management", requested_action="cancel_subscription", needs_account_data=True)),
        (r"\bpause\b", dict(category="subscription_management", requested_action="pause_subscription", needs_account_data=True)),
        (r"resume|reactivate", dict(category="subscription_management", requested_action="resume_subscription", needs_account_data=True)),
        (r"cancel .*reservation|cancel my booking|the samba one|the carnival one",
         dict(category="reservation_change", requested_action="cancel_reservation", needs_account_data=True)),
        (r"book .* for me|reserve .* for me", dict(category="reservation_booking", requested_action="reserve_experience", needs_account_data=True)),
        (r"how do i (reserve|book)|reserve a spot", dict(category="reservation_booking")),
        (r"log ?in|password", dict(category="login_access")),
        (r"crash|qr code", dict(category="technical_issue")),
        (r"how many .* left|my quota", dict(category="subscription_management", requested_action="check_account_status", needs_account_data=True)),
    ]
    for pattern, upd in rules:
        if re.search(pattern, t):
            c.update(upd)
            break
    if re.search(r"urgent|asap|immediately", t):
        c["urgency"] = "high"
    if re.search(r"terrible|furious|angry|worst|ridiculous", t):
        c["sentiment"] = "very_negative"
    return c


class _Structured:
    def __init__(self, llm: "FakeLLM", schema):
        self.llm, self.schema = llm, schema

    def invoke(self, msgs, *_, **__):
        return self.llm._structured(self.schema, msgs)


class _Bound:
    def __init__(self, llm: "FakeLLM", tools, tool_choice=None):
        self.llm, self.tool_names, self.tool_choice = llm, [t.name for t in tools], tool_choice

    def invoke(self, msgs, *_, **__):
        if self.tool_choice == "submit_answer":  # forced finish
            refs = REF.findall(next(m.content for m in msgs if isinstance(m, SystemMessage)))
            self.llm.calls.append({"kind": "forced_finish", "messages": msgs})
            return self.llm._submit("Here is what I found so far.", refs, "resolved", 0.7)
        return self.llm._tool_step(self.tool_names, msgs)


class FakeLLM:
    def __init__(self, overrides: dict[str, Any] | None = None, fail_on: set[str] | None = None):
        self.overrides = overrides or {}
        self.fail_on = fail_on or set()
        self.calls: list[dict] = []
        self._n = 0

    # LangChain-compatible surface ------------------------------------------
    def with_structured_output(self, schema, method=None, **_):
        return _Structured(self, schema)

    def bind_tools(self, tools, tool_choice=None, **_):
        return _Bound(self, tools, tool_choice)

    # structured outputs ------------------------------------------------------
    def _structured(self, schema, msgs):
        name = schema.__name__
        self.calls.append({"kind": "structured", "schema": name, "messages": msgs})
        if name in self.fail_on:
            raise RuntimeError(f"simulated LLM outage for {name}")
        if name in self.overrides:
            o = self.overrides[name]
            if isinstance(o, list):  # a sequence of answers, one per call
                o = o.pop(0) if len(o) > 1 else o[0]
            return o(msgs) if callable(o) else o
        text = latest_text(msgs)
        if name == "TicketClassification":
            return TicketClassification(**classify(text))
        if name == "RoutingDecision":
            human = msgs[-1].content
            m = re.search(r"'category': '(\w+)'.*?'needs_account_data': (True|False).*?'requested_action': '(\w+)'", human, re.S)
            cls = {"category": m.group(1), "needs_account_data": m.group(2) == "True", "requested_action": m.group(3)} if m else {"category": "general_inquiry"}
            route, reason = rules_first(cls)
            return RoutingDecision(next_agent=route, reason="(fake llm) " + reason, confidence=0.8)
        if name == "GroundingVerdict":
            return GroundingVerdict(supported=True, claims_actions_not_performed=False, answers_question=True, score=0.9, issues=[])
        if name == "EscalationHandoff":
            refs = REF.findall(msgs[-1].content)
            return EscalationHandoff(customer_message="I've passed your ticket to our specialist team; you'll hear back soon.",
                                     summary="fake summary", cited_articles=refs[:1])
        if name == "MemoryExtraction":
            prefs = []
            if "email" in text.lower() and "prefer" in text.lower():
                prefs.append(Preference(key="contact_channel", value="email"))
            if "portuguese" in text.lower():
                prefs.append(Preference(key="language", value="Portuguese"))
            return MemoryExtraction(preferences=prefs, facts=[])
        raise NotImplementedError(name)

    # tool-calling specialists --------------------------------------------------
    def _call(self, name, **args) -> AIMessage:
        self._n += 1
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{self._n}", "type": "tool_call"}])

    def _submit(self, response, refs, outcome="resolved", confidence=0.9, **extra) -> AIMessage:
        return self._call("submit_answer", response=response, cited_articles=refs[:1], outcome=outcome,
                          confidence=confidence, **extra)

    def _tool_step(self, tool_names, msgs) -> AIMessage:
        system = next(m.content for m in msgs if isinstance(m, SystemMessage))
        self.calls.append({"kind": "tools", "tools": tool_names, "system": system, "messages": msgs})
        if "specialist" in self.fail_on:
            raise RuntimeError("simulated LLM outage in specialist")
        if "Specialist" in self.overrides:
            return self.overrides["Specialist"](self, tool_names, msgs)
        refs = REF.findall(system)
        text = latest_text(msgs).lower()
        results: dict[str, Any] = {}
        call_names = {}
        for m in msgs:
            if isinstance(m, AIMessage):
                for tc in m.tool_calls:
                    call_names[tc["id"]] = tc["name"]
            if isinstance(m, ToolMessage):
                try:
                    results[call_names.get(m.tool_call_id)] = json.loads(m.content)
                except json.JSONDecodeError:
                    results[call_names.get(m.tool_call_id)] = m.content

        if "Account Specialist" in system:
            if re.search(r"cancel|samba|carnival", text):
                listing = results.get("list_reservations")
                if listing is None:
                    return self._call("list_reservations")
                if "cancel_reservation" in results:
                    r = results["cancel_reservation"]
                    if r.get("ok"):
                        return self._submit(f"Your reservation for {r['data']['title']} is cancelled.", refs)
                    return self._submit("I could not cancel that reservation.", refs, "escalate", 0.4,
                                        escalation_reason=r["error"]["message"])
                active = [r for r in listing.get("data", []) if r["status"] == "reserved"]
                chosen = [r for r in active if any(w in text for w in r["title"].lower().split() if len(w) > 4)]
                if len(chosen) == 1:
                    return self._call("cancel_reservation", reservation_id=chosen[0]["reservation_id"])
                titles = ", ".join(r["title"] for r in active)
                return self._submit(f"You have these reservations: {titles}. Which one should I cancel?", refs,
                                    "needs_customer_input", 0.9)
            if "for me" in text:
                if "search_experiences" not in results:
                    word = next((w for w in ("masp", "samba", "paddleboarding", "ibirapuera", "carnival") if w in text), "")
                    return self._call("search_experiences", query=word)
                found = results["search_experiences"].get("data", [])
                if found and "reserve_experience" not in results:
                    return self._call("reserve_experience", experience_id=found[0]["experience_id"])
                r = results.get("reserve_experience", {})
                if r.get("ok"):
                    return self._submit(f"You're booked for {r['data']['title']}.", refs)
                return self._submit(f"That could not be booked: {r.get('error', {}).get('message')}", refs,
                                    "needs_customer_input", 0.8)
            if "get_customer_profile" not in results:
                return self._call("get_customer_profile")
            q = results["get_customer_profile"]["data"]["quota"]
            return self._submit(f"You have {q['remaining']} experiences left this cycle.", refs)

        if "Billing & Subscription Specialist" in system:
            if re.search(r"refund|money back", text):
                if "get_customer_profile" not in results:
                    return self._call("get_customer_profile")
                if "submit_refund_request" not in results:
                    return self._call("submit_refund_request", reason="duplicate charge reported by customer")
                return self._submit("I've submitted your refund request for review.", refs, "escalate", 0.9,
                                    escalation_reason="refund awaiting approval")
            for word, tool, args in (("cancel", "cancel_subscription", {"customer_confirmed": True}),
                                     ("pause", "pause_subscription", {}), ("resume", "resume_subscription", {})):
                if word in text:
                    if tool not in results:
                        return self._call(tool, **args)
                    ok = results[tool].get("ok")
                    return self._submit(f"{tool} {'done' if ok else 'failed'}.", refs, "resolved" if ok else "escalate",
                                        0.9 if ok else 0.3, escalation_reason=None if ok else "tool failed")
            if "get_customer_profile" not in results:
                return self._call("get_customer_profile")
            return self._submit("Here is your billing status.", refs)

        # Knowledge Resolver
        if not refs:
            return self._submit("", [], "escalate", 0.1, escalation_reason="no article")
        first = re.search(r"\[KB-[0-9a-f]{6}\] .*?\n(.+?)\n", system)
        return self._submit(f"Based on our guide: {first.group(1) if first else 'see the steps'}", refs)


class FakeEmbedder:
    """Hashed bag-of-words vectors: similar texts -> similar vectors, no API."""
    def __init__(self, dim: int = 256, fail: bool = False):
        self.dim, self.fail, self.document_calls, self.query_calls = dim, fail, 0, 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        if self.fail:
            raise ConnectionError("embeddings API unreachable")
        self.document_calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        if self.fail:
            raise ConnectionError("embeddings API unreachable")
        self.query_calls += 1
        return self._vec(text)
