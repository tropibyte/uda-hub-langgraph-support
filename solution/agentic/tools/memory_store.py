"""Long-term customer memory with semantic recall.

Short-term memory is the LangGraph checkpoint for a thread (thread_id =
ticket_id). This module is the *long-term* layer that survives across
tickets and sessions, stored in the ``customer_memories`` table:

* ``preference``      - upserted by key ("contact_channel", "language", ...)
* ``resolved_issue``  - one line per resolved ticket
* ``escalated_issue`` - one line per escalated ticket (so the next agent knows)
* ``fact``            - stable facts the customer told us ("travels for work")

Recall = every preference (they are few and always relevant) + the top-k
issue/fact memories ranked by cosine similarity to the current ticket, or by
BM25 when embeddings are unavailable.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime

from agentic.config import ACCOUNT_ID
from agentic.db import core_engine, session_scope, udahub
from agentic.tools.knowledge_retriever import BM25, cosine, tokenize

KINDS = ("preference", "resolved_issue", "escalated_issue", "fact")


class MemoryStore:
    def __init__(self, embedder=None):
        self.embedder = embedder

    def _embed(self, text: str) -> list[float] | None:
        if self.embedder is None:
            return None
        try:
            return self.embedder.embed_query(text)
        except Exception:  # noqa: BLE001 - memory still works lexically
            return None

    def remember(self, user_id: str, kind: str, content: str, *, key: str | None = None,
                 source_ticket_id: str | None = None) -> dict:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        content = content.strip()
        if not content:
            raise ValueError("content must not be empty")
        emb = self._embed(content)
        with session_scope(core_engine()) as s:
            row = None
            if kind == "preference" and key:
                row = s.query(udahub.CustomerMemory).filter_by(user_id=user_id, kind="preference", key=key).first()
            if row is None and kind != "preference":
                row = s.query(udahub.CustomerMemory).filter_by(user_id=user_id, kind=kind,
                                                               source_ticket_id=source_ticket_id).first() \
                    if source_ticket_id and kind in ("resolved_issue", "escalated_issue") else None
            action = "updated" if row is not None else "created"
            if row is None:
                row = udahub.CustomerMemory(memory_id=str(uuid.uuid4()), account_id=ACCOUNT_ID,
                                            user_id=user_id, kind=kind, key=key)
                s.add(row)
            row.content = content
            row.embedding = json.dumps([round(x, 6) for x in emb]) if emb else None
            row.source_ticket_id = source_ticket_id
            row.updated_at = datetime.now()
            return {"memory_id": row.memory_id, "kind": kind, "key": key, "content": content, "action": action}

    def all_for_user(self, user_id: str) -> list[dict]:
        with session_scope(core_engine()) as s:
            rows = (s.query(udahub.CustomerMemory).filter_by(user_id=user_id)
                    .order_by(udahub.CustomerMemory.updated_at.desc()).all())
            return [self._row(r) for r in rows]

    @staticmethod
    def _row(r) -> dict:
        return {"memory_id": r.memory_id, "kind": r.kind, "key": r.key, "content": r.content,
                "source_ticket_id": r.source_ticket_id,
                "updated_at": r.updated_at.isoformat(timespec="minutes") if r.updated_at else None,
                "_embedding": json.loads(r.embedding) if r.embedding else None}

    def recall(self, user_id: str, query: str, k: int = 4) -> list[dict]:
        """All preferences + the k most relevant other memories for this user."""
        rows = self.all_for_user(user_id)
        prefs = [r for r in rows if r["kind"] == "preference"]
        others = [r for r in rows if r["kind"] != "preference"]
        if others and query.strip():
            qv = self._embed(query) if any(r["_embedding"] for r in others) else None
            if qv is not None:
                scored = [(cosine(qv, r["_embedding"]) if r["_embedding"] else 0.0, r) for r in others]
                method = "cosine"
            else:
                bm = BM25([tokenize(r["content"]) for r in others]).scores(tokenize(query))
                scored = list(zip(bm, others))
                method = "bm25"
            scored.sort(key=lambda x: -x[0])
            # recency still matters: keep the latest issue even when it is not similar
            chosen = [dict(r, relevance=round(sc, 3), recall_method=method) for sc, r in scored[:k]]
            latest = others[0]
            if latest["memory_id"] not in {c["memory_id"] for c in chosen}:
                chosen.append(dict(latest, relevance=None, recall_method="most_recent"))
        else:
            chosen = []
        out = [dict(p, relevance=None, recall_method="preference") for p in prefs] + chosen
        for r in out:
            r.pop("_embedding", None)
        return out
