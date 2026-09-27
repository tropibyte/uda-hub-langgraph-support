"""Shared services injected into every agent node.

Keeping the LLM, embeddings, retriever, memory store and tool registry in
one object lets the tests swap in a scripted chat model and an offline
retriever without touching agent code, and makes every external dependency
lazy: importing ``agentic.workflow`` never opens a network connection.
"""
from __future__ import annotations

import time
from typing import Any, Type, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agentic.config import get_settings, openai_kwargs
from agentic.tools.knowledge_retriever import KnowledgeRetriever
from agentic.tools.memory_store import MemoryStore
from agentic.tools.registry import ToolRegistry

T = TypeVar("T", bound=BaseModel)
_UNSET: Any = object()


class LLMError(RuntimeError):
    pass


class Services:
    def __init__(self, llm=None, embedder=_UNSET, tools: ToolRegistry | None = None):
        self._llm = llm
        self._embedder = embedder
        self._tools = tools
        self._retriever: KnowledgeRetriever | None = None
        self._memory: MemoryStore | None = None
        self.llm_calls = 0

    # -- lazily created dependencies ----------------------------------------
    @property
    def llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            s = get_settings()
            self._llm = ChatOpenAI(model=s.chat_model, temperature=0, timeout=60, max_retries=2, **openai_kwargs())
        return self._llm

    @property
    def embedder(self):
        if self._embedder is _UNSET:
            if get_settings().offline:
                self._embedder = None
            else:
                from langchain_openai import OpenAIEmbeddings
                # check_embedding_ctx_length=False sends plain strings; the
                # Vocareum gateway rejects the token-id arrays sent otherwise.
                self._embedder = OpenAIEmbeddings(model=get_settings().embedding_model,
                                                  check_embedding_ctx_length=False, **openai_kwargs())
        return self._embedder

    @property
    def retriever(self) -> KnowledgeRetriever:
        if self._retriever is None:
            self._retriever = KnowledgeRetriever(self.embedder)
        return self._retriever

    @property
    def memory(self) -> MemoryStore:
        if self._memory is None:
            self._memory = MemoryStore(self.embedder)
        return self._memory

    @property
    def tools(self) -> ToolRegistry:
        if self._tools is None:
            self._tools = ToolRegistry()
        return self._tools

    # -- LLM helpers -----------------------------------------------------------
    def structured(self, schema: Type[T], messages: list[BaseMessage], retries: int = 1) -> T:
        """Call the LLM for a pydantic object; retry once on a malformed reply."""
        runnable = self.llm.with_structured_output(schema, method="function_calling")
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                self.llm_calls += 1
                out = runnable.invoke(messages)
                if isinstance(out, dict):
                    out = schema.model_validate(out)
                if out is None:
                    raise LLMError(f"model returned no {schema.__name__}")
                return out
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.5 * (attempt + 1))
        raise LLMError(f"{schema.__name__} failed: {last.__class__.__name__}: {last}") from last

    def close(self):
        if self._tools is not None:
            self._tools.close()
