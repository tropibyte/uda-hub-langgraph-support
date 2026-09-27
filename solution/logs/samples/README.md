# Sample logs

`notebook_run_events.jsonl` is the structured event log written while `03_agentic_app.ipynb` was executed: one JSON object per line with `ts, event_id, run_id, thread_id, ticket_id, agent, event, level, payload`. The same events are in the `agent_events` table. Search them with `python -m agentic.logging_utils --help`.

The evaluation runs write their own logs to `evaluation/results/events_<variant>.jsonl`.
