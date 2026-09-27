"""Knowledge Retriever agent: runs RAG for every ticket before routing.

Retrieval happens once, up front, so the supervisor can use the retrieval
confidence as a routing signal ("no relevant article -> escalate") and every
specialist starts from the same grounded evidence.

The query combines the latest message with the classifier's one-line intent;
short follow-ups ("yes please", "the second one") also include the previous
customer message so the topic is not lost.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from agentic.agents.common import emit, safe_node
from agentic.services import Services
from agentic.state import UDAHubState


def build_query(state: UDAHubState) -> str:
    text = state.get("user_message", "")
    intent = (state.get("classification") or {}).get("intent") or ""
    humans = [m.content for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    parts = []
    if len(text.split()) < 8 and len(humans) >= 2:
        parts.append(str(humans[-2]))
    parts += [text, intent]
    return "\n".join(p for p in parts if p)


def make_knowledge_retriever(services: Services):
    @safe_node("knowledge_retriever")
    def knowledge_retriever(state: UDAHubState) -> dict:
        if state.get("error"):
            return {}
        category = (state.get("classification") or {}).get("category")
        result = services.retriever.search(build_query(state), k=4, category=category).as_dict()
        if services.retriever.dense_error:
            result["dense_error"] = services.retriever.dense_error
        ev = emit(state, "knowledge_retriever", "retrieval", {
            "method": result["method"], "confidence": result["confidence"],
            "articles": [{"ref": a["ref"], "title": a["title"], "relevance": a["relevance"]} for a in result["articles"]],
            "dense_error": result.get("dense_error")})
        return {"retrieval": result, "events": [ev]}

    return knowledge_retriever
