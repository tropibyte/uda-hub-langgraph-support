"""RAG retrieval (BM25 + dense + fusion + confidence) and long-term memory."""
import pytest

from agentic.db import core_engine, session_scope, udahub
from agentic.tools.knowledge_retriever import KnowledgeRetriever, dense_relevance, tokenize
from agentic.tools.memory_store import MemoryStore
from tests.fakes import FakeEmbedder


@pytest.mark.parametrize("query, expected", [
    ("I can't log in and the password reset email never arrives", "How to Handle Login Issues?"),
    ("my card was declined when renewing", "Payment Failed or Card Declined"),
    ("do unused experiences roll over to next month", "How the Monthly Experience Quota Works"),
    ("QR code won't scan at the venue entrance", "QR Code Not Scanning at the Venue"),
    ("please delete my account and personal data", "Deleting Your Account and Personal Data Requests"),
])
def test_bm25_finds_the_right_article(data_env, query, expected):
    res = KnowledgeRetriever(embedder=None).search(query, k=3)
    assert res.method == "bm25"
    assert expected in [a["title"] for a in res.articles]
    assert res.confidence > 0.35


def test_unrelated_query_has_no_confident_match(data_env):
    res = KnowledgeRetriever(embedder=None).search("Can you recommend a good pizza recipe?")
    assert res.confidence < 0.35


def test_articles_carry_refs_and_scores(data_env):
    a = KnowledgeRetriever().search("how do I reserve an event")
    top = a.articles[0]
    assert top["ref"].startswith("KB-") and len(top["ref"]) == 9
    assert {"relevance", "bm25", "rrf", "content", "title", "tags"} <= set(top)


def test_dense_path_caches_embeddings_and_reuses_them(data_env):
    emb = FakeEmbedder()
    r = KnowledgeRetriever(embedder=emb)
    res = r.search("my card was declined")
    assert res.method.startswith("hybrid")
    assert res.articles[0]["dense_cosine"] is not None
    with session_scope(core_engine()) as s:
        assert s.query(udahub.KnowledgeEmbedding).count() == 29
    # a new retriever instance reuses the cached vectors: no new document embedding call
    emb2 = FakeEmbedder()
    KnowledgeRetriever(embedder=emb2).search("refund")
    assert emb2.document_calls == 0 and emb2.query_calls == 1


def test_edited_article_is_re_embedded(data_env):
    emb = FakeEmbedder()
    r = KnowledgeRetriever(embedder=emb)
    r.search("refund")
    with session_scope(core_engine()) as s:
        k = s.query(udahub.Knowledge).filter_by(title="Refund Policy and Refund Requests").one()
        k.content = k.content + "\nNew rule: refunds for gift cards are never possible."
    r.search("gift card refund")
    assert emb.document_calls == 2  # initial batch + the one edited article


def test_dense_failure_degrades_to_bm25(data_env):
    r = KnowledgeRetriever(embedder=FakeEmbedder(fail=True))
    res = r.search("my card was declined")
    assert res.method == "bm25 (dense unavailable)" and r.dense_error
    assert res.articles[0]["title"] == "Payment Failed or Card Declined"


def test_confidence_mapping_and_tokenizer():
    assert dense_relevance(0.10) == 0 and dense_relevance(0.9) == 1
    assert 0 < dense_relevance(0.40) < 1
    assert tokenize("Queries about the QR codes!") == ["query", "about", "qr", "code"]


# ---------------------------------------------------------------- memory ---

def _uid(ext="f556c0"):
    from agentic.tools.udahub_ops import get_or_create_user
    return get_or_create_user(ext)["user_id"]


def test_preferences_are_upserted_by_key(data_env):
    m, uid = MemoryStore(), _uid()
    first = m.remember(uid, "preference", "email", key="contact_channel")
    second = m.remember(uid, "preference", "WhatsApp", key="contact_channel")
    assert first["action"] == "created" and second["action"] == "updated"
    prefs = [x for x in m.all_for_user(uid) if x["kind"] == "preference"]
    assert len(prefs) == 1 and prefs[0]["content"] == "WhatsApp"


def test_issue_memory_is_one_row_per_ticket(data_env):
    from agentic.tools.udahub_ops import create_ticket
    m, uid = MemoryStore(), _uid()
    tid = create_ticket("f556c0")
    m.remember(uid, "resolved_issue", "QR code fixed", source_ticket_id=tid)
    m.remember(uid, "resolved_issue", "QR code fixed, confirmed", source_ticket_id=tid)
    assert len([x for x in m.all_for_user(uid) if x["kind"] == "resolved_issue"]) == 1


def test_recall_returns_preferences_and_relevant_issues(data_env):
    m, uid = MemoryStore(), _uid()
    m.remember(uid, "preference", "Portuguese", key="language")
    m.remember(uid, "resolved_issue", "QR code would not scan at MASP; fixed by raising brightness")
    m.remember(uid, "resolved_issue", "asked how to upgrade to premium")
    m.remember(uid, "escalated_issue", "double charge refund request sent to billing lead")
    got = m.recall(uid, "my QR code does not scan again", k=1)
    kinds = [g["kind"] for g in got]
    assert kinds[0] == "preference"
    assert any("QR code" in g["content"] for g in got)
    assert any(g["recall_method"] == "most_recent" for g in got)  # latest issue always kept


def test_recall_with_embeddings_uses_cosine(data_env):
    m, uid = MemoryStore(FakeEmbedder()), _uid()
    m.remember(uid, "fact", "travels to Salvador often for work")
    m.remember(uid, "resolved_issue", "password reset email went to spam")
    got = m.recall(uid, "reset password email spam", k=1)
    assert got[0]["recall_method"] == "cosine" and "password" in got[0]["content"]


def test_memory_validation(data_env):
    m, uid = MemoryStore(), _uid()
    with pytest.raises(ValueError):
        m.remember(uid, "gossip", "x")
    with pytest.raises(ValueError):
        m.remember(uid, "fact", "   ")
    assert m.recall(_uid("e6376d"), "anything") == []


def test_phrase_normalisation_for_lexical_search():
    assert "login" in tokenize("I can't log in to my account")
    assert "login" in tokenize("signed into the app")
    assert "refund" in tokenize("I want my money back")


def test_category_boost_ranks_the_matching_article_first(data_env):
    r = KnowledgeRetriever(embedder=FakeEmbedder())
    plain = r.search("I can't access my CultPass account at all", k=4)
    boosted = r.search("I can't access my CultPass account at all", k=4, category="login_access")
    assert boosted.articles[0]["title"] == "How to Handle Login Issues?"
    assert boosted.articles[0]["category_match"] is True
    assert "category boost (login_access)" in boosted.method and "category" not in plain.method
    # an off-topic question gains at most the boost, so it still stays below the gate
    assert r.search("Can you recommend a good pizza recipe?", category="general_inquiry").confidence < 0.35
