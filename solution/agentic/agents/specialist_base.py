"""The tool-calling loop every specialist runs (built by hand, no prebuilt agent).

    context + conversation
            |
            v
     LLM (tools bound, a tool call is required every step)
            |-- support tool ----> ToolRegistry.call (MCP) --> ToolMessage --+
            |-- search_knowledge_base -> retriever ----------> ToolMessage --+--> next step
            '-- submit_answer ---> SpecialistAnswer (loop ends)

Requiring a tool call on every step means the loop can only end through
``submit_answer``, so every specialist returns the same structured object
(reply, cited KB refs, outcome, self-rated confidence, optional hand-off).
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agentic.agents.common import (articles_block, compact, conversation, customer_block, emit,
                                   memory_block, ticket_block)
from agentic.config import get_settings
from agentic.services import Services
from agentic.state import SpecialistAnswer, UDAHubState
from agentic.tools.cultpass_ops import MUTATING
from agentic.tools.registry import llm_tools

_GENERIC_TITLE_WORDS = {"tour", "night", "experience", "walk", "ride", "park", "history", "modern", "sunset", "with"}


def unambiguous_target(services: Services, state: UDAHubState, reservation_id: str) -> str | None:
    """Destructive-action guard for cancel_reservation (policy in code, not prompt).

    Returns None when the cancellation may proceed: the customer has a single active
    reservation, or their own messages name this one (a distinctive title word, its
    date or its id). Otherwise returns the reason to ask the customer first.
    """
    user_id = state.get("external_user_id") if state.get("customer") else None
    res = services.tools.call("list_reservations", {"status": "reserved"}, user_id=user_id)
    active = res.get("data") or [] if res.get("ok") else []
    target = next((r for r in active if r["reservation_id"] == reservation_id), None)
    if target is None or len(active) <= 1:
        return None  # nothing to disambiguate; the tool itself validates ownership/status
    said = " ".join(str(m.content) for m in state.get("messages", []) if isinstance(m, HumanMessage)).lower()
    words = {w for w in re.findall(r"[a-zà-ú]+", (target["title"] or "").lower())
             if len(w) >= 4 and w not in _GENERIC_TITLE_WORDS}
    day = (target.get("when") or "")[:10]
    if reservation_id.lower() in said or any(w in said for w in words) or (day and day in said):
        return None
    return (f"The customer has {len(active)} active reservations and has not said which one to cancel. "
            "Do not guess: list them and ask which one (outcome needs_customer_input).")


# Tool -> escalation reason. A successful call opens a request a human must approve.
REQUIRES_HUMAN_DECISION = {"submit_refund_request": "refund awaiting approval"}

COMMON_RULES = """
Rules you must follow:
1. Ground everything. Every policy, step, time frame, price or rule you state must come from a knowledge
   article in the context or returned by search_knowledge_base, or from a tool result. Put the refs of
   the articles you used in cited_articles. Never invent policies, fees, dates or promises.
2. If the articles and tools do not cover the request, or policy says a human must decide, finish
   with outcome="escalate" and explain why in escalation_reason. Do not guess.
3. Never claim an action (reserved, cancelled, paused, refunded...) unless a tool result in this
   conversation shows it succeeded. Refund requests are only *submitted*; never say a refund is approved.
4. Personalise: use the customer's first name, honour known preferences (language, contact channel,
   interests) and acknowledge relevant past issues when helpful. Reply in the customer's
   stored preferred language if one is on record, otherwise in the language they wrote in.
5. Be concise and warm: at most ~120 words, plain text, numbered steps only when giving instructions.
   Do not mention knowledge-base refs, internal tools, agents or confidence scores to the customer.
6. If you need information only the customer can give (e.g. which reservation), ask one clear question
   and finish with outcome="needs_customer_input".
7. The customer's identity is already verified by the ticket; tools act on their account automatically.
8. When a tool returns an error code (sold_out, quota_exceeded, forbidden, conflict, ...), look up the
   matching policy with search_knowledge_base (e.g. "waitlist", "monthly quota") and base your reply on it.
9. Always finish by calling submit_answer exactly once.
"""


def _call_key(tool: str, args: dict | None) -> str:
    return tool + json.dumps(args or {}, sort_keys=True, default=str)


class _KBQuery(BaseModel):
    query: str = Field(description="What to look up in the CultPass knowledge base.")


def _kb_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda query: query, name="search_knowledge_base", args_schema=_KBQuery,
        description="Search the CultPass knowledge base when the provided articles do not cover the request. Returns articles with refs.")


def _submit_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda **kw: kw, name="submit_answer", args_schema=SpecialistAnswer,
        description="Finish: submit the reply to the customer with cited article refs, outcome and confidence.")


def run_specialist(services: Services, state: UDAHubState, *, agent: str, role_prompt: str,
                   tool_names: list[str]) -> dict:
    settings = get_settings()
    retrieval = state.get("retrieval") or {}
    articles = list(retrieval.get("articles", []))
    qa = state.get("qa") or {}
    revision = ""
    if qa.get("issues") and state.get("last_specialist") == agent:
        revision = ("\n\nYOUR PREVIOUS DRAFT WAS REJECTED BY QA. Fix these issues:\n- " + "\n- ".join(qa["issues"]) +
                    f"\nPrevious draft: {qa.get('draft', '')}")
    earlier = [t for t in state.get("tool_calls", []) if t.get("turn") != state.get("turn") and t.get("agent") != "intake"]
    session_tools = ""
    if earlier:
        session_tools = "\n--- Tool results earlier in this session (ids are exact; reuse them) ---\n" + "\n".join(
            f"turn {t['turn']} {t['tool']}({t.get('args')}): {compact(t.get('result'), 700)}" for t in earlier[-4:])
    # Tool work already done in THIS turn (e.g. before a QA revision or a hand-off):
    # shown to the model, and mutating calls are never executed twice (see guard below).
    this_turn = [t for t in state.get("tool_calls", []) if t.get("turn") == state.get("turn") and t.get("agent") != "intake"]
    done_actions = {_call_key(t["tool"], t.get("args")): t for t in this_turn if t.get("ok") and t["tool"] in MUTATING}
    if this_turn:
        session_tools += ("\n--- Already done in THIS turn: do NOT repeat these actions, reuse their results ---\n" +
                          "\n".join(f"{t['tool']}({t.get('args')}) ok={t.get('ok')}: {compact(t.get('result'), 700)}"
                                    for t in this_turn[-6:]))
    handoff = ""
    hist = state.get("routing_history") or []
    if hist and hist[-1].get("rule") == "S3_handoff":
        handoff = f"\n\nHand-off note: {hist[-1]['reason']}."

    system = SystemMessage(
        f"{role_prompt}\n{COMMON_RULES}\n"
        f"--- Ticket ---\n{ticket_block(state)}\n"
        f"--- Customer ---\n{customer_block(state)}\n"
        f"--- Customer history & memory ---\n{memory_block(state)}\n"
        f"--- Knowledge articles (retrieved for this request) ---\n{articles_block(articles)}"
        f"{session_tools}{handoff}{revision}")
    msgs: list = [system, *conversation(state)]
    if not msgs[-1:] or not isinstance(msgs[-1], HumanMessage):
        msgs.append(HumanMessage(state.get("user_message", "")))

    tools = llm_tools(tool_names) + [_kb_tool(), _submit_tool()]
    bound = services.llm.bind_tools(tools, tool_choice="any")
    turn = state.get("turn")
    tool_calls: list[dict] = []
    events = [emit(state, agent, "agent_started", {"tools": [t.name for t in tools], "revision": bool(revision)})]
    extra_articles: list[dict] = []
    answer: SpecialistAnswer | None = None

    finish = services.llm.bind_tools(tools, tool_choice="submit_answer")
    seen_calls: set[str] = set()  # identical calls already made in this loop

    for step in range(settings.max_tool_iterations):
        services.llm_calls += 1
        last_step = step == settings.max_tool_iterations - 1
        if last_step:  # out of steps: the model must answer with what it has, never time out empty-handed
            msgs.append(HumanMessage("Step limit reached: call submit_answer now, based on the evidence you already have."))
        ai: AIMessage = (finish if last_step else bound).invoke(msgs)
        msgs.append(ai)
        if not ai.tool_calls:
            # model ignored tool_choice: accept its text, but with low confidence
            answer = SpecialistAnswer(response=str(ai.content), cited_articles=[], outcome="resolved", confidence=0.3)
            break
        for tc in ai.tool_calls:
            name, args = tc["name"], tc.get("args") or {}
            if name == "submit_answer":
                try:
                    answer = SpecialistAnswer.model_validate(args)
                    content = "accepted"
                except Exception as exc:  # noqa: BLE001
                    content = f"invalid submit_answer arguments: {exc}"[:300]
                msgs.append(ToolMessage(content=content, tool_call_id=tc["id"]))
                continue
            if name == "search_knowledge_base":
                res = services.retriever.search(args.get("query", ""), k=3).as_dict()
                known = {a["ref"] for a in articles + extra_articles}
                extra_articles += [a for a in res["articles"] if a["ref"] not in known]
                payload: Any = [{"ref": a["ref"], "title": a["title"], "relevance": a["relevance"], "content": a["content"]}
                                for a in res["articles"]]
                events.append(emit(state, agent, "retrieval", {"source": "search_knowledge_base", "query": args.get("query"),
                                                                "confidence": res["confidence"],
                                                                "articles": [{"ref": a["ref"], "title": a["title"]} for a in res["articles"]]}))
                record = {"turn": turn, "agent": agent, "tool": name, "args": args, "ok": True,
                          "result": [a["title"] for a in res["articles"]], "transport": "in-process"}
            elif _call_key(name, args) in seen_calls and name not in MUTATING:
                payload = {"ok": False, "error": {"code": "duplicate_call", "message":
                           "You already made this exact call in this turn and have its result above. "
                           "Do not repeat it: answer with submit_answer (escalate if the evidence is insufficient)."}}
                events.append(emit(state, agent, "tool_result", {"tool": name, "ok": False, "deduplicated": True,
                                                                  "result": "repeat-call breaker: identical read already made"}))
                record = {"turn": turn, "agent": agent, "tool": name, "args": args, "ok": False,
                          "result": {"code": "duplicate_call"}, "transport": "deduplicated"}
            elif _call_key(name, args) in done_actions:
                prev = done_actions[_call_key(name, args)]
                payload = {"ok": True, "data": prev.get("result"),
                           "note": "already performed earlier in this turn; not executed again"}
                events.append(emit(state, agent, "tool_result", {"tool": name, "ok": True, "deduplicated": True,
                                                                  "result": "idempotency guard: identical action already succeeded this turn"}))
                record = {"turn": turn, "agent": agent, "tool": name, "args": args, "ok": True,
                          "result": prev.get("result"), "transport": "deduplicated"}
            elif name == "cancel_reservation" and (why := unambiguous_target(services, state, str(args.get("reservation_id", "")))):
                payload = {"ok": False, "error": {"code": "confirmation_required", "message": why}}
                events.append(emit(state, agent, "tool_result", {"tool": name, "ok": False, "blocked_by": "ambiguity_guard",
                                                                  "result": why}, level="WARNING"))
                record = {"turn": turn, "agent": agent, "tool": name, "args": args, "ok": False,
                          "result": {"code": "confirmation_required"}, "transport": "guarded"}
            else:
                events.append(emit(state, agent, "tool_call", {"tool": name, "args": args, "transport": services.tools.transport,
                                                                "mutating": name in MUTATING}))
                res = services.tools.call(name, args, user_id=state.get("external_user_id")
                                          if (state.get("customer") or {}) else None, ticket_id=state.get("ticket_id"))
                meta = res.pop("_meta", {})
                payload = res
                events.append(emit(state, agent, "tool_result", {"tool": name, "ok": res.get("ok"),
                                                                  "result": res.get("data") if res.get("ok") else res.get("error"),
                                                                  **meta}, level="INFO" if res.get("ok") else "WARNING"))
                record = {"turn": turn, "agent": agent, "tool": name, "args": args, "ok": res.get("ok"),
                          "result": res.get("data") if res.get("ok") else res.get("error"), **meta}
            seen_calls.add(_call_key(name, args))
            tool_calls.append(record)
            msgs.append(ToolMessage(content=json.dumps(payload, default=str, ensure_ascii=False)[:6000], tool_call_id=tc["id"]))
        if answer is not None:
            break

    if answer is None:
        answer = SpecialistAnswer(response="", outcome="escalate", confidence=0.0,
                                  escalation_reason=f"{agent} did not finish within {settings.max_tool_iterations} steps")
    # Policy, not prompt: some successful actions are only requests that a human must decide.
    pending = [t["tool"] for t in this_turn + tool_calls if t.get("ok") and t["tool"] in REQUIRES_HUMAN_DECISION]
    if pending and answer.outcome != "escalate":
        answer = answer.model_copy(update={"outcome": "escalate",
                                           "escalation_reason": REQUIRES_HUMAN_DECISION[pending[0]]})
    result = {**answer.model_dump(), "agent": agent, "extra_articles": extra_articles}
    events.append(emit(state, agent, "agent_finished", {
        "outcome": answer.outcome, "confidence": answer.confidence, "cited": answer.cited_articles,
        "handoff_to": answer.handoff_to, "tools_used": [t["tool"] for t in tool_calls],
        "response_preview": compact(answer.response, 160)}))
    return {"specialist_result": result, "last_specialist": agent, "tool_calls": tool_calls, "events": events}
