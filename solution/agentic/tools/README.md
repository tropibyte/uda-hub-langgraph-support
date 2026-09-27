# Tools

| Module | What |
|---|---|
| `cultpass_ops.py` | The 9 CultPass support operations (profile, reservations, experiences, reserve, cancel, pause/resume/cancel subscription, refund request). Validated inputs, `{"ok", "data" / "error"}` envelopes, no exceptions for expected failures. |
| `mcp_server.py` | FastMCP server exposing those operations over stdio (`python agentic/tools/mcp_server.py`). |
| `registry.py` | What agents see (LLM schemas without `user_id`) and how calls run (persistent MCP client session, identity injection, local fallback). |
| `udahub_ops.py` | UDA-Hub core: tickets, messages, metadata/tags, ticket history. |
| `knowledge_retriever.py` | Hybrid RAG (embeddings + BM25, weighted RRF, calibrated confidence). See `../design/RAG.md`. |
| `memory_store.py` | Long-term customer memory with semantic recall. |

All database paths are absolute (`agentic/config.py`), so the tools, the MCP
server subprocess and the notebooks resolve the same files from any working directory.
