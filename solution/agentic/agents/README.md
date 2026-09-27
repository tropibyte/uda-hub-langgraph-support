# Agents

Each module builds one LangGraph node (`make_*`) bound to the shared `Services`
(LLM, retriever, memory, tools). Roles, inputs and outputs are specified in
`../design/ARCHITECTURE.md` section 2.

| Module | Agent(s) | Uses LLM |
|---|---|---|
| `intake.py` | Intake: ticket/customer context, channel normalisation, memory recall, A/B variant | no |
| `classifier.py` | Classifier: category, urgency, sentiment, action + safety backstops + P1-P4 priority | structured output |
| `knowledge_retriever.py` | Knowledge Retriever: hybrid RAG before routing | embeddings |
| `supervisor.py` | Supervisor: guardrails, knowledge gate, assignment (rules_first / llm_supervisor), hand-offs | variant B only |
| `specialists.py` + `specialist_base.py` | Knowledge Resolver, Account Specialist, Billing Specialist: hand-written tool-calling loop ending in `submit_answer` | tool calling |
| `qa_reviewer.py` | QA Reviewer: citation checks, LLM judge, confidence, one revision | structured output |
| `escalation.py` | Escalation: team/priority by policy, grounded customer message, handoff summary | structured output |
| `responder.py` | Responder: channel formatting, persistence, ticket status, resolution event | no |
| `memory_curator.py` | Memory Curator: preferences/facts (gated LLM) + issue summaries | structured output |
| `common.py` | shared helpers: event logging, failure isolation, context blocks | - |
