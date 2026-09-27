"""Retrieval calibration: hit@k and the confidence split between in-domain and
off-topic questions, for hybrid (dense + BM25) and BM25-only retrieval.

    python evaluation/calibrate_retrieval.py      # writes evaluation/calibration.md

The 0.35 knowledge-gate threshold and the DENSE_LOW/DENSE_HIGH anchors in
agentic/tools/knowledge_retriever.py were chosen from this table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SOLUTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION_DIR))

from agentic.config import get_settings  # noqa: E402
from agentic.services import Services  # noqa: E402
from agentic.tools.knowledge_retriever import DENSE_HIGH, DENSE_LOW, KnowledgeRetriever  # noqa: E402


def evaluate(retriever: KnowledgeRetriever, rows: list[dict], use_category: bool = False) -> tuple[dict, list[dict]]:
    out = []
    for q in rows:
        res = retriever.search(q["query"], k=3, category=q.get("category") if use_category else None)
        titles = [a["title"] for a in res.articles]
        out.append({"query": q["query"], "expected": q["expected"], "top": titles[0] if titles else None,
                    "hit1": bool(q["expected"]) and titles[:1] == [q["expected"]],
                    "hit3": bool(q["expected"]) and q["expected"] in titles,
                    "confidence": round(res.confidence, 3),
                    "cosine": res.articles[0]["dense_cosine"] if res.articles else None})
    rel = [r for r in out if r["expected"]]
    ood = [r for r in out if not r["expected"]]
    gate = get_settings().min_retrieval_relevance
    summary = {
        "method": retriever.search("test").method + (" + category boost" if use_category else ""),
        "in_domain": len(rel), "off_topic": len(ood),
        "hit@1": f"{sum(r['hit1'] for r in rel)}/{len(rel)}",
        "hit@3": f"{sum(r['hit3'] for r in rel)}/{len(rel)}",
        "min_in_domain_confidence": min(r["confidence"] for r in rel),
        "max_off_topic_confidence": max(r["confidence"] for r in ood),
        f"in_domain_above_gate({gate})": f"{sum(r['confidence'] >= gate for r in rel)}/{len(rel)}",
        f"off_topic_below_gate({gate})": f"{sum(r['confidence'] < gate for r in ood)}/{len(ood)}",
    }
    return summary, out


def main():
    rows = [json.loads(l) for l in open(SOLUTION_DIR / "evaluation" / "retrieval_queries.jsonl", encoding="utf-8")]
    services = Services()
    results = {}
    for label, emb, use_cat in (("hybrid + category (production)", services.embedder, True),
                                ("hybrid", services.embedder, False), ("bm25_only", None, False)):
        if emb is None and label != "bm25_only":
            print("no API key / offline: skipping", label)
            continue
        results[label] = evaluate(KnowledgeRetriever(emb), rows, use_cat)
    lines = ["# Retrieval calibration", "",
             f"{len(rows)} labelled queries (`evaluation/retrieval_queries.jsonl`): in-domain questions with the article "
             "that should answer them, plus off-topic questions that must not match anything.",
             f"Dense anchors: cosine {DENSE_LOW} -> confidence 0, cosine {DENSE_HIGH} -> confidence 1.", "",
             "| metric | " + " | ".join(results) + " |", "|---" * (len(results) + 1) + "|"]
    for k in next(iter(results.values()))[0]:
        lines.append(f"| {k} | " + " | ".join(str(results[m][0][k]) for m in results) + " |")
    for label, (_, per) in results.items():
        lines += ["", f"## {label}", "", "| query | expected | top-1 | hit@3 | confidence | cosine |", "|---|---|---|---|---|---|"]
        for r in per:
            lines.append(f"| {r['query']} | {r['expected'] or '(off-topic)'} | {r['top']} | "
                         f"{'✓' if r['hit3'] else ('-' if not r['expected'] else '✗')} | {r['confidence']} | {r['cosine']} |")
    lines += ["", "Columns: *hybrid + category* is what the workflow runs (the classifier's category boosts articles "
              "tagged for it; each query carries the category a correct classification would give). *hybrid* is "
              "pure retrieval with no metadata. *bm25_only* is the offline fallback.",
              "", "Reading: the category boost is what makes hard paraphrases such as \"I can't log in to my Cultpass "
              "account\" (the embedding model ranks the login article low for it) land on the right article. "
              "'I want to talk to a real person' is escalated by the wants-human guardrail before the gate is "
              "consulted, so its low score is harmless. BM25-only ranks reasonably but cannot separate off-topic "
              "questions by score, which is why it is documented as a degraded mode."]
    out = SOLUTION_DIR / "evaluation" / "calibration.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for label, (s, _) in results.items():
        print(label, json.dumps(s))
    print("wrote", out)


if __name__ == "__main__":
    main()
