"""Offline evaluation + A/B comparison of the two routing strategies.

Runs every labelled ticket in evaluation/tickets.jsonl through the real graph
(live LLM, MCP tools), once per routing strategy, each strategy on its own
freshly built copy of the databases, and scores:

  category accuracy, first-route accuracy, outcome accuracy, escalation
  precision/recall, expected-article hit, required/forbidden tool use,
  grounded-answer rate, confidence, QA judge score, latency, LLM calls.

    python evaluation/run_eval.py                    # both variants
    python evaluation/run_eval.py --variants rules_first --only T01 T17

Writes evaluation/results/{summary.json, per_ticket.csv, report.md}.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean

SOLUTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION_DIR))
OUT = SOLUTION_DIR / "evaluation" / "results"


def load_tickets(only=None):
    rows = [json.loads(l) for l in open(SOLUTION_DIR / "evaluation" / "tickets.jsonl", encoding="utf-8") if l.strip()]
    return [r for r in rows if not only or r["id"] in only]


def score(t: dict, out: dict, elapsed: float, llm_calls: int) -> dict:
    final = out.get("final") or {}
    cls = out.get("classification") or {}
    first_route = next((r["route"] for r in out.get("routing_history", []) if r.get("turn") == out.get("turn")), None)
    turn_tools = [x for x in out.get("tool_calls", []) if x.get("agent") != "intake"]
    called_ok = {x["tool"] for x in turn_tools if x.get("ok")}
    called = {x["tool"] for x in turn_tools}
    retrieved = [a["title"] for a in (out.get("retrieval") or {}).get("articles", [])[:3]]
    cited = [c["title"] for c in final.get("citations", [])]
    status = final.get("status")
    status = {"needs_customer_input": "needs_customer_input"}.get(status, status)
    r = {
        "id": t["id"], "variant": out.get("ab_variant"), "message": t["messages"][-1][:70],
        "category": cls.get("category"), "category_ok": cls.get("category") in t["category"],
        "first_route": first_route, "route_ok": first_route in t["route"],
        "status": status, "outcome_ok": status in t["outcome"],
        "escalation_required": t["outcome"] == ["escalated"], "escalation_allowed": "escalated" in t["outcome"],
        "escalated": status == "escalated",
        "team": (out.get("escalation") or {}).get("team") if status == "escalated" else None,
        "team_ok": (t.get("team") is None) or ((out.get("escalation") or {}).get("team") == t["team"]),
        "article": t.get("article"), "article_ok": None if not t.get("article") else (t["article"] in cited or t["article"] in retrieved),
        "article_cited": None if not t.get("article") else t["article"] in cited,
        "tools_used": sorted(called),
        "tools_ok": all(x in called_ok for x in t.get("tools", [])) and not any(
            (f[:-8] in called_ok) if f.endswith("_success") else (f in called) for f in t.get("forbidden_tools", [])),
        "grounded": bool(cited) if status == "resolved" else None,
        "confidence": final.get("confidence"),
        "judge_score": ((out.get("qa") or {}).get("verdict") or {}).get("score"),
        "latency_s": round(elapsed, 2), "llm_calls": llm_calls,
        "response": final.get("response", "").replace("\n", " ")[:400],
    }
    return r


def run_variant(variant: str, tickets: list[dict]) -> list[dict]:
    from scripts.setup_databases import setup
    work = Path(tempfile.mkdtemp(prefix=f"udahub-eval-{variant}-"))
    data = setup(work / "root")
    os.environ.update({"UDAHUB_DATA_DIR": str(data), "UDAHUB_LOG_DIR": str(work / "logs"),
                       "UDAHUB_ROUTING_STRATEGY": variant, "UDAHUB_CHECKPOINTER": "memory"})
    from langgraph.checkpoint.memory import MemorySaver

    from agentic.services import Services
    from agentic.workflow import build_workflow, send_message, submit_ticket
    services = Services()
    graph = build_workflow(services, checkpointer=MemorySaver())
    results = []
    try:
        for t in tickets:
            tid = submit_ticket(t["user"], channel=t["channel"])
            calls0, t0 = services.llm_calls, time.time()
            out = None
            for m in t["messages"]:
                out = send_message(graph, tid, m)
            r = score(t, out, time.time() - t0, services.llm_calls - calls0)
            r["transport"] = services.tools.transport
            results.append(r)
            flag = "OK " if r["category_ok"] and r["route_ok"] and r["outcome_ok"] and r["tools_ok"] and r["article_ok"] is not False else "!! "
            print(f"{flag}{variant:<15}{t['id']} cat={r['category']:<24} route={str(r['first_route']):<19} "
                  f"status={r['status']:<21} conf={r['confidence']} {r['latency_s']}s", flush=True)
    finally:
        services.close()
        shutil.copy2(work / "logs" / "udahub_events.jsonl", OUT / f"events_{variant}.jsonl")
    return results


def summarise(rows: list[dict]) -> dict:
    def rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(bool(v) for v in vals) / len(vals), 3) if vals else None
    esc_true = [r for r in rows if r["escalation_required"]]
    esc_pred = [r for r in rows if r["escalated"]]
    conf = [r["confidence"] for r in rows if isinstance(r["confidence"], (int, float))]
    judge = [r["judge_score"] for r in rows if isinstance(r["judge_score"], (int, float))]
    return {
        "tickets": len(rows),
        "category_accuracy": rate("category_ok"), "route_accuracy": rate("route_ok"),
        "outcome_accuracy": rate("outcome_ok"), "team_accuracy": rate("team_ok"),
        # precision: escalations that the label allows; recall: mandatory escalations that happened
        "escalation_precision": round(sum(r["escalation_allowed"] for r in esc_pred) / len(esc_pred), 3) if esc_pred else None,
        "escalation_recall": round(sum(r["escalated"] for r in esc_true) / len(esc_true), 3) if esc_true else None,
        "article_hit_rate": rate("article_ok"), "article_cited_rate": rate("article_cited"),
        "tool_accuracy": rate("tools_ok"), "grounded_resolution_rate": rate("grounded"),
        "avg_confidence": round(mean(conf), 3) if conf else None,
        "avg_judge_score": round(mean(judge), 3) if judge else None,
        "avg_latency_s": round(mean(r["latency_s"] for r in rows), 2),
        "avg_llm_calls": round(mean(r["llm_calls"] for r in rows), 2),
        "all_checks_passed": sum(r["category_ok"] and r["route_ok"] and r["outcome_ok"] and r["tools_ok"]
                                 and r["team_ok"] and r["article_ok"] is not False for r in rows),
    }


def write_report(summary: dict, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "per_ticket.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(OUT / "per_ticket.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()})
    metrics = list(next(iter(summary.values())).keys())
    lines = ["# UDA-Hub evaluation report", "",
             f"Labelled tickets: {len(load_tickets())} (evaluation/tickets.jsonl). Each routing strategy ran on its own "
             "fresh copy of the databases with the live LLM and MCP tools.", "",
             "| metric | " + " | ".join(summary) + " |", "|---" * (len(summary) + 1) + "|"]
    for m in metrics:
        lines.append(f"| {m} | " + " | ".join(str(summary[v][m]) for v in summary) + " |")
    lines += ["", "## Per ticket", "",
              "| id | variant | category | route | status | conf | checks | tools |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        checks = "".join("✓" if r[k] or r[k] is None else "✗" for k in ("category_ok", "route_ok", "outcome_ok", "tools_ok", "team_ok", "article_ok"))
        lines.append(f"| {r['id']} | {r['variant']} | {r['category']} | {r['first_route']} | {r['status']} | "
                     f"{r['confidence']} | {checks} | {', '.join(r['tools_used'])} |")
    lines += ["", "checks = category, first route, outcome, tools, escalation team, expected article (✓ = pass or n/a)."]
    failures = [r for r in rows if not (r["category_ok"] and r["route_ok"] and r["outcome_ok"] and r["tools_ok"]
                                        and r["team_ok"] and r["article_ok"] is not False)]
    if failures:
        lines += ["", "## Misses", ""]
        for r in failures:
            lines.append(f"- **{r['id']} ({r['variant']})** `{r['message']}` -> category={r['category']}, "
                         f"route={r['first_route']}, status={r['status']}, tools={r['tools_used']}. Reply: {r['response'][:200]}")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["rules_first", "llm_supervisor"])
    ap.add_argument("--only", nargs="*", help="ticket ids to run")
    ap.add_argument("--rescore", action="store_true", help="recompute summary/report from results/per_ticket.json without running")
    args = ap.parse_args()
    if args.rescore:
        rows = json.loads((OUT / "per_ticket.json").read_text(encoding="utf-8"))
        labels = {t["id"]: t for t in load_tickets()}
        for r in rows:  # re-apply the current labels to the recorded behaviour
            t = labels[r["id"]]
            r.update(category_ok=r["category"] in t["category"], route_ok=r["first_route"] in t["route"],
                     outcome_ok=r["status"] in t["outcome"], escalation_required=t["outcome"] == ["escalated"],
                     escalation_allowed="escalated" in t["outcome"])
        summary = {v: summarise([r for r in rows if r["variant"] == v]) for v in dict.fromkeys(r["variant"] for r in rows)}
        write_report(summary, rows)
        return print(json.dumps(summary, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    tickets = load_tickets(args.only)
    all_rows, summary = [], {}
    for v in args.variants:
        rows = run_variant(v, tickets)
        all_rows += rows
        summary[v] = summarise(rows)
    write_report(summary, all_rows)
    print(json.dumps(summary, indent=2))
    print(f"report: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
