"""A/B testing of routing strategies.

Every ticket is assigned a routing variant at intake (a stable hash of the
ticket id, so a ticket never switches variant mid-conversation):

* ``rules_first``    - category x action routing table (deterministic, free)
* ``llm_supervisor`` - the LLM picks the specialist (one extra call per turn)

Guardrails and the knowledge gate apply to both, so the experiment only
varies the assignment step. The variant is stored on the ticket
(``variant:<name>`` tag) and on every ``routing_decision``/``resolution``
event, so outcomes can be compared from production logs with ``ab_report``,
or offline against labelled tickets with ``evaluation/run_eval.py``.

    python -m agentic.ab_testing          # report over everything logged so far
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import mean

from agentic.logging_utils import search_events


def ab_report(limit: int = 100000) -> dict[str, dict]:
    """Aggregate the last resolution of each ticket turn, per variant."""
    rows = search_events(event="resolution", limit=limit)
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        p = r["payload"]
        by_variant[p.get("variant") or "unknown"].append(p)
    report = {}
    for variant, items in sorted(by_variant.items()):
        n = len(items)
        conf = [i["confidence"] for i in items if isinstance(i.get("confidence"), (int, float))]
        lat = [i["latency_s"] for i in items if isinstance(i.get("latency_s"), (int, float))]
        report[variant] = {
            "turns": n,
            "resolved_rate": round(sum(i["status"] == "resolved" for i in items) / n, 3),
            "escalated_rate": round(sum(i["status"] == "escalated" for i in items) / n, 3),
            "needs_input_rate": round(sum(i["status"] == "needs_customer_input" for i in items) / n, 3),
            "avg_confidence": round(mean(conf), 3) if conf else None,
            "avg_hops": round(mean(i.get("hops") or 0 for i in items), 2),
            "revision_rate": round(sum((i.get("revisions") or 0) > 0 for i in items) / n, 3),
            "avg_latency_s": round(mean(lat), 2) if lat else None,
        }
    return report


def format_report(report: dict[str, dict]) -> str:
    if not report:
        return "no resolutions logged yet"
    cols = list(next(iter(report.values())).keys())
    lines = ["| variant | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for v, m in report.items():
        lines.append(f"| {v} | " + " | ".join(str(m[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    argparse.ArgumentParser(description="A/B report of routing strategies from the agent_events log").parse_args()
    print(format_report(ab_report()))
