"""QA Reviewer agent: verifies a specialist's reply before the customer sees it.

Checks (design doc section 6):
* deterministic - a resolved answer must cite at least one article, and every
  cited ref must be an article that was actually retrieved in this turn;
* LLM judge - is every claim supported by the cited articles / tool results,
  does the reply claim an action no successful tool call performed, does it
  answer the question;
* confidence - ``0.4 * evidence + 0.3 * self-rating + 0.3 * judge score``,
  where evidence is the best relevance among the cited articles.

Pass -> responder. Fail -> one revision round with the issues as feedback,
then escalation if it still fails or confidence stays below the threshold.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agentic.agents.common import articles_block, compact, emit, safe_node
from agentic.config import get_settings
from agentic.services import Services
from agentic.state import GroundingVerdict, UDAHubState

MAX_REVISIONS = 1

JUDGE = """You are the QA Reviewer of UDA-Hub customer support. Judge a draft reply strictly against the
evidence: the knowledge-base articles and the tool results. A claim is supported only if the evidence
states it. Friendly wording, greetings and restating the customer's question need no support. Asking
the customer a clarifying question is fine. Be strict about policies, time frames, fees and about
actions the reply says were performed."""


def confidence_score(evidence: float, self_rating: float, judge: float) -> float:
    return round(0.4 * evidence + 0.3 * self_rating + 0.3 * judge, 3)


def make_qa_reviewer(services: Services):
    settings = get_settings()

    @safe_node("qa_reviewer")
    def qa_reviewer(state: UDAHubState) -> dict:
        res = state.get("specialist_result") or {}
        retrieval = state.get("retrieval") or {}
        articles = {a["ref"]: a for a in retrieval.get("articles", []) + res.get("extra_articles", [])}
        cited = [r for r in res.get("cited_articles", []) if r]
        outcome = res.get("outcome")
        turn = state.get("turn")
        tool_results = [t for t in state.get("tool_calls", []) if t.get("turn") == turn and t.get("agent") != "intake"]

        issues = []
        unknown = [r for r in cited if r not in articles]
        if unknown:
            issues.append(f"cited refs {unknown} were not retrieved; cite only refs from the provided articles")
        valid = [r for r in cited if r in articles]
        if outcome == "resolved" and not valid:
            issues.append("the reply is not based on any cited knowledge-base article")
        if not (res.get("response") or "").strip():
            issues.append("empty reply")

        evidence_articles = [articles[r] for r in valid] or list(articles.values())[:3]
        verdict = services.structured(GroundingVerdict, [
            SystemMessage(JUDGE),
            HumanMessage(
                f"Customer's latest message:\n{state.get('user_message')}\n\n"
                f"Knowledge articles:\n{articles_block(evidence_articles, 1800)}\n\n"
                f"Tool results this turn:\n{compact([{k: t.get(k) for k in ('tool', 'args', 'ok', 'result')} for t in tool_results], 3500)}\n\n"
                f"Draft reply:\n{res.get('response')}")])
        if not verdict.supported:
            issues += [f"unsupported: {i}" for i in verdict.issues] or ["reply contains claims not supported by the evidence"]
        if verdict.claims_actions_not_performed:
            issues.append("reply claims an account action that no successful tool call performed")
        if not verdict.answers_question:
            issues.append("reply does not address the customer's latest request")

        evidence = max([articles[r].get("relevance") or 0 for r in valid], default=0.0)
        if outcome == "needs_customer_input":
            evidence = max(evidence, 0.8)  # a clarifying question needs no article support
        if any(t.get("ok") for t in tool_results):
            evidence = max(evidence, 0.75)  # facts came from the customer's own records
        confidence = confidence_score(evidence, float(res.get("confidence") or 0), verdict.score)
        low = confidence < settings.escalation_confidence and outcome == "resolved"
        if low:
            issues.append(f"confidence {confidence} below threshold {settings.escalation_confidence}")
        passed = not issues
        revisions = int(state.get("revision_count") or 0)

        fixable = len(issues) > (1 if low else 0)  # something besides low evidence went wrong
        if passed:
            route = "responder"
        elif revisions < MAX_REVISIONS and fixable:
            route = res.get("agent") or "knowledge_resolver"
        else:
            route = "escalation"
        qa = {"passed": passed, "confidence": confidence, "issues": issues, "verdict": verdict.model_dump(),
              "evidence": evidence, "draft": res.get("response", "")[:600], "route": route}
        ev = emit(state, "qa_reviewer", "qa_review", {"passed": passed, "confidence": confidence, "issues": issues,
                                                      "route": route, "judge_score": verdict.score})
        update = {"qa": qa, "route": route, "events": [ev]}
        if route == "responder":
            update["final"] = {"status": "resolved" if outcome == "resolved" else "needs_customer_input",
                               "response": res["response"], "confidence": confidence,
                               "citations": [{"ref": r, "title": articles[r]["title"]} for r in valid],
                               "handled_by": res.get("agent")}
        elif route == "escalation":
            update["escalation"] = {"reason": "; ".join(issues)[:300] or "QA rejected the reply",
                                    "rule": "Q1_qa_failed", "team": "tier2_support"}
        else:
            update["revision_count"] = revisions + 1
            update["specialist_result"] = {}
        return update

    return qa_reviewer
