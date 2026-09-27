"""UDA-Hub command-line app (the .py alternative to 03_agentic_app.ipynb).

Run from the solution folder after notebooks 01 and 02 (or
`python scripts/setup_databases.py`):

    python 03_agentic_app.py demo                         # all demo scenarios with decision traces
    python 03_agentic_app.py demo --only refund blocked   # selected scenarios
    python 03_agentic_app.py chat --user f556c0           # interactive chat as Bob (new ticket)
    python 03_agentic_app.py chat --user f556c0 --ticket <id>   # continue a ticket (session memory)
    python 03_agentic_app.py ticket --user 88382b --channel email "I was charged twice"
    python 03_agentic_app.py process <ticket_id>          # run a ticket already stored in the DB
    python 03_agentic_app.py state <ticket_id>            # inspect a thread: messages, routing, tool usage
    python 03_agentic_app.py memory --user 88382b         # long-term memories of a customer
    python 03_agentic_app.py tickets                      # list tickets and their status
    python 03_agentic_app.py logs --ticket-id <id>        # structured decision log (see --help)
    python 03_agentic_app.py ab-report                    # A/B comparison of routing strategies

CultPass demo users: a4ab87 Alice (blocked), f556c0 Bob (basic), 88382b Cathy (premium),
888fb2 David (blocked, cancelled), f1f10d Eva (paused), e6376d Frank (premium).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description="UDA-Hub multi-agent support app",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run the demo scenarios")
    d.add_argument("--only", nargs="*")
    d.add_argument("--quiet", action="store_true", help="replies only, no decision trace")
    c = sub.add_parser("chat", help="interactive chat")
    c.add_argument("--user", required=True, help="CultPass user id")
    c.add_argument("--ticket", help="existing ticket id to continue")
    c.add_argument("--channel", default="chat")
    t = sub.add_parser("ticket", help="submit one message as a new ticket")
    t.add_argument("--user", required=True)
    t.add_argument("--channel", default="chat")
    t.add_argument("--urgency")
    t.add_argument("text")
    p = sub.add_parser("process", help="process a stored ticket")
    p.add_argument("ticket_id")
    s = sub.add_parser("state", help="inspect a ticket thread")
    s.add_argument("ticket_id")
    m = sub.add_parser("memory", help="show a customer's long-term memory")
    m.add_argument("--user", required=True)
    sub.add_parser("tickets", help="list tickets")
    sub.add_parser("ab-report", help="A/B report from the event log")
    sub.add_parser("logs", help="search the event log", add_help=False)
    args, rest = ap.parse_known_args(argv)

    if args.cmd == "logs":
        from agentic import logging_utils
        sys.argv = ["logs", *rest]
        return logging_utils._cli()
    if args.cmd == "ab-report":
        from agentic.ab_testing import ab_report, format_report
        return print(format_report(ab_report()))
    if args.cmd == "tickets":
        from agentic.tools.udahub_ops import list_tickets
        for row in list_tickets():
            print(f"{row['ticket_id']}  {row['user_name']:<15} {row['channel']:<7} {row['status']:<17} {row['issue_type']}")
        return
    if args.cmd == "memory":
        from agentic.tools.memory_store import MemoryStore
        from agentic.tools.udahub_ops import get_or_create_user
        for mem in MemoryStore().all_for_user(get_or_create_user(args.user)["user_id"]):
            print(f"{mem['updated_at']}  {mem['kind']:<16} {mem['key'] or '':<16} {mem['content']}")
        return

    from utils import chat_interface, print_turn
    from agentic.workflow import orchestrator, process_ticket, send_message, submit_ticket, thread_config
    try:
        if args.cmd == "demo":
            from agentic.demo import SCENARIOS, run_scenario
            for sc in SCENARIOS:
                if not args.only or sc.key in args.only:
                    run_scenario(orchestrator, sc, show_trace=not args.quiet)
        elif args.cmd == "chat":
            tid = args.ticket or submit_ticket(args.user, channel=args.channel)
            print(f"ticket/thread id: {tid}  (type 'quit' to leave; reuse --ticket {tid} to continue later)")
            chat_interface(orchestrator, tid, external_user_id=args.user, channel=args.channel)
        elif args.cmd == "ticket":
            tid = submit_ticket(args.user, channel=args.channel, urgency=args.urgency)
            print(f"ticket {tid}\nUser: {args.text}")
            print_turn(send_message(orchestrator, tid, args.text))
        elif args.cmd == "process":
            print_turn(process_ticket(orchestrator, args.ticket_id))
        elif args.cmd == "state":
            snap = orchestrator.get_state(thread_config(args.ticket_id))
            v = snap.values
            if not v:
                return print("no checkpoint for this thread (sessions persist with UDAHUB_CHECKPOINTER=sqlite, the default)")
            print(f"turns: {v.get('turn')}  variant: {v.get('ab_variant')}  checkpoints: {len(list(orchestrator.get_state_history(thread_config(args.ticket_id))))}")
            print("\nmessages:")
            for msg in v.get("messages", []):
                print(f"  {msg.type:>5}: {msg.content[:150]}")
            print("\nrouting:")
            for r in v.get("routing_history", []):
                print(f"  turn {r['turn']}: -> {r['route']:<19} [{r['rule']}] {r['reason']}")
            print("\ntool usage:")
            for tc in v.get("tool_calls", []):
                print(f"  turn {tc['turn']}: {tc['agent']:<19} {tc['tool']}({json.dumps(tc.get('args'))}) ok={tc.get('ok')} via {tc.get('transport')}")
    finally:
        orchestrator.services.close()


if __name__ == "__main__":
    main()
