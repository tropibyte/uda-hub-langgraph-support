"""Hybrid retrieval over the UDA-Hub knowledge base (RAG).

How it works (full write-up in agentic/design/RAG.md):

1. Articles for the account are read from the ``knowledge`` table.
2. Dense: each article (title + tags + content) is embedded once with
   ``text-embedding-3-small``; vectors are cached in ``knowledge_embeddings``
   keyed by a content hash, so edits re-embed only what changed.
3. Sparse: a small BM25 index over the same text (title and tags weighted x2)
   catches exact terms such as "QR", "LGPD" or "double charge".
4. Fusion: weighted Reciprocal Rank Fusion (dense 1.0, BM25 0.5).
5. Confidence: the top article's dense cosine similarity mapped to [0, 1]
   between two calibrated anchors, plus a small bonus when BM25 agrees. The
   supervisor escalates when it falls below ``min_retrieval_relevance``.

With no API key (or UDAHUB_OFFLINE=1) the retriever runs BM25-only and maps
the BM25 score to confidence with a saturating curve instead.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from agentic.config import ACCOUNT_ID, get_settings
from agentic.db import core_engine, session_scope, udahub

# Calibration anchors (see evaluation/calibration.md): cosine at or below LOW
# means unrelated, at or above HIGH means clearly on topic.
DENSE_LOW, DENSE_HIGH = 0.22, 0.58
BM25_SATURATION = 6.0
RRF_K = 60
DENSE_WEIGHT, BM25_WEIGHT = 1.0, 0.5

_STOP = set("""a an and are as at be but by can do does for from has have how i if in into is it its
me my no not of on or our so that the their them then there these they this to too was we were what
when where which who why will with you your i'm im i've ive cant can't dont don't please hi hello thanks
thank just get got would could should also""".split())


# Multi-word phrasings normalised to the vocabulary the articles use.
_PHRASES = [
    (re.compile(r"\b(log|logged|logging|sign|signed|signing)\s+(in|into|on)\b"), "login"),
    (re.compile(r"\bmoney\s+back\b"), "refund"),
    (re.compile(r"\bcharged\s+twice\b"), "double charge"),
]

# Ticket category -> article tags that belong to it. The classifier runs before
# retrieval, so its category is used as a metadata boost (design doc RAG.md s.4).
CATEGORY_TAGS: dict[str, set[str]] = {
    "login_access": {"login", "password", "access"},
    "account_management": {"account", "profile", "change email", "update details", "notifications"},
    "account_security": {"security", "hacked", "compromised", "suspicious activity", "fraud"},
    "billing_payment": {"billing", "payment", "card declined", "double charge", "payment method"},
    "refund_request": {"refund", "money back", "double charge"},
    "subscription_management": {"subscription", "plans", "quota", "cancelation", "pause", "reactivate", "upgrade"},
    "reservation_booking": {"reservation", "booking", "sold out", "waitlist", "premium"},
    "reservation_change": {"cancel reservation", "transfer", "event cancelled", "rescheduled", "no-show"},
    "technical_issue": {"technical", "app crash", "qr code", "notifications"},
    "safety_incident": {"safety", "incident"},
    "privacy_request": {"privacy", "delete account", "personal data"},
    "general_inquiry": {"general", "availability", "accessibility", "gift card", "benefits"},
}
CATEGORY_BOOST = 0.12


def tokenize(text: str) -> list[str]:
    text = text.lower()
    for rx, repl in _PHRASES:
        text = rx.sub(repl, text)
    toks = re.findall(r"[a-z0-9]+", text)
    out = []
    for t in toks:
        if t in _STOP or len(t) < 2:
            continue
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def article_ref(article_id: str) -> str:
    return "KB-" + article_id.replace("-", "")[:6]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = (sum(map(len, docs)) / len(docs)) if docs else 0.0
        self.tf = [Counter(d) for d in docs]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for q in query:
                if q in tf:
                    f = tf[q]
                    s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def dense_relevance(cos: float) -> float:
    return max(0.0, min(1.0, (cos - DENSE_LOW) / (DENSE_HIGH - DENSE_LOW)))


def bm25_relevance(score: float) -> float:
    return score / (score + BM25_SATURATION) if score > 0 else 0.0


@dataclass
class RetrievalResult:
    query: str
    method: str
    articles: list[dict] = field(default_factory=list)
    confidence: float = 0.0

    def as_dict(self) -> dict:
        return {"query": self.query, "method": self.method, "confidence": round(self.confidence, 3),
                "articles": self.articles}


class KnowledgeRetriever:
    def __init__(self, embedder=None, account_id: str = ACCOUNT_ID):
        self.embedder = embedder
        self.account_id = account_id
        self._signature = None
        self._articles: list[dict] = []
        self._bm25: BM25 | None = None
        self._vectors: dict[str, list[float]] = {}
        self.dense_error: str | None = None

    # -- index -----------------------------------------------------------
    def _current_signature(self, s) -> tuple:
        rows = s.query(udahub.Knowledge.article_id, udahub.Knowledge.updated_at).filter_by(account_id=self.account_id).all()
        return tuple(sorted((r[0], str(r[1])) for r in rows))

    def refresh(self, force: bool = False) -> None:
        with session_scope(core_engine()) as s:
            sig = self._current_signature(s)
            if sig == self._signature and not force:
                return
            rows = s.query(udahub.Knowledge).filter_by(account_id=self.account_id).all()
            self._articles = [{"article_id": r.article_id, "ref": article_ref(r.article_id), "title": r.title,
                               "tags": r.tags or "", "content": r.content} for r in rows]
        self._bm25 = BM25([tokenize(f"{a['title']} {a['title']} {a['tags']} {a['tags']} {a['content']}")
                           for a in self._articles])
        self._vectors = {}
        if self.embedder is not None:
            try:
                self._vectors = self._load_or_embed()
                self.dense_error = None
            except Exception as exc:  # noqa: BLE001 - degrade to BM25, never fail retrieval
                self.dense_error = f"{exc.__class__.__name__}: {exc}"[:200]
        self._signature = sig

    @staticmethod
    def _doc_text(a: dict) -> str:
        return f"{a['title']}\nTags: {a['tags']}\n{a['content']}"

    def _load_or_embed(self) -> dict[str, list[float]]:
        model = get_settings().embedding_model
        hashes = {a["article_id"]: hashlib.sha1(self._doc_text(a).encode()).hexdigest() for a in self._articles}
        vectors: dict[str, list[float]] = {}
        with session_scope(core_engine()) as s:
            for row in s.query(udahub.KnowledgeEmbedding).filter_by(model=model).all():
                if hashes.get(row.article_id) == row.content_hash:
                    vectors[row.article_id] = json.loads(row.embedding)
        missing = [a for a in self._articles if a["article_id"] not in vectors]
        if missing:
            embs = self.embedder.embed_documents([self._doc_text(a) for a in missing])
            with session_scope(core_engine()) as s:
                for a, e in zip(missing, embs):
                    vectors[a["article_id"]] = e
                    s.merge(udahub.KnowledgeEmbedding(article_id=a["article_id"], model=model,
                                                      content_hash=hashes[a["article_id"]],
                                                      embedding=json.dumps([round(x, 6) for x in e])))
        return vectors

    # -- search ----------------------------------------------------------
    def get_by_title(self, title_substring: str) -> dict | None:
        self.refresh()
        needle = title_substring.lower()
        return next((dict(a) for a in self._articles if needle in a["title"].lower()), None)

    def search(self, query: str, k: int = 4, min_relevance: float = 0.12,
               category: str | None = None) -> RetrievalResult:
        """Hybrid search. ``category`` (the classifier's label) boosts articles whose tags
        belong to that category: one extra RRF list plus CATEGORY_BOOST on relevance."""
        self.refresh()
        if not self._articles or not query.strip():
            return RetrievalResult(query=query, method="none")
        bm = self._bm25.scores(tokenize(query))
        bm_rank = sorted(range(len(bm)), key=lambda i: -bm[i])
        dense = None
        if self._vectors:
            try:
                qv = self.embedder.embed_query(query)
                dense = [cosine(qv, self._vectors.get(a["article_id"], [])) for a in self._articles]
            except Exception as exc:  # noqa: BLE001
                self.dense_error = f"{exc.__class__.__name__}: {exc}"[:200]
        fused = [0.0] * len(self._articles)
        for rank, i in enumerate(bm_rank):
            if bm[i] > 0:
                fused[i] += BM25_WEIGHT / (RRF_K + rank + 1)
        if dense is not None:
            for rank, i in enumerate(sorted(range(len(dense)), key=lambda i: -dense[i])):
                fused[i] += DENSE_WEIGHT / (RRF_K + rank + 1)
            method = "hybrid(dense+bm25, weighted RRF)"
        else:
            method = "bm25" + (" (dense unavailable)" if self.embedder is not None else "")
        wanted = CATEGORY_TAGS.get(category or "", set())
        matches = {i for i, a in enumerate(self._articles)
                   if wanted & {t.strip() for t in a["tags"].lower().split(",")}}
        for i in matches:
            fused[i] += DENSE_WEIGHT / (RRF_K + 1)
        if matches:
            method += f" + category boost ({category})"
        order = sorted(range(len(fused)), key=lambda i: -fused[i])
        bm_top3 = set(bm_rank[:3])
        results = []
        for i in order[: max(k, 1) * 2]:
            rel = dense_relevance(dense[i]) if dense is not None else bm25_relevance(bm[i])
            if dense is not None and i in bm_top3 and bm[i] > 0:
                rel = min(1.0, rel + 0.08)  # lexical agreement bonus
            if i in matches:
                rel = min(1.0, rel + CATEGORY_BOOST)  # metadata agreement bonus
            if rel < min_relevance and results:
                continue
            a = self._articles[i]
            results.append({**a, "relevance": round(rel, 3), "category_match": i in matches, "dense_cosine": round(dense[i], 4) if dense else None,
                            "bm25": round(bm[i], 3), "rrf": round(fused[i], 5)})
            if len(results) >= k:
                break
        # keep the fused (RRF) order: it already combines dense, lexical and category evidence
        confidence = max((r["relevance"] for r in results), default=0.0)
        return RetrievalResult(query=query, method=method, articles=results, confidence=confidence)
