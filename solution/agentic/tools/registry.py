"""Tool registry: how agents see and call the support tools.

Two concerns are split on purpose:

* **What the LLM sees** - ``llm_tools(names)`` returns LangChain tool specs
  built from the operation signatures *minus* ``user_id`` and ``ticket_id``.
  The model therefore cannot act on another customer's account; the identity
  comes from the ticket, not from the prompt.
* **How a call is executed** - ``ToolRegistry.call`` injects the ticket
  owner's ``user_id``/``ticket_id`` and dispatches either to the FastMCP server
  over stdio (default, ``UDAHUB_TOOL_TRANSPORT=mcp``) or in-process
  (``local``). If the MCP server cannot start, the registry falls back to
  local and records why, so a ticket never fails because of transport.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import threading
import time
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from agentic.config import get_settings
from agentic.tools import cultpass_ops as ops

INJECTED_ARGS = ("user_id", "ticket_id")
SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")


# ---------------------------------------------------------------------------
# LLM-facing tool specs
# ---------------------------------------------------------------------------

def _spec_for(name: str) -> StructuredTool:
    fn = ops.OPERATIONS[name]
    fields: dict[str, Any] = {}
    for p in inspect.signature(fn).parameters.values():
        if p.name in INJECTED_ARGS:
            continue
        ann = p.annotation if p.annotation is not inspect.Parameter.empty else str
        if isinstance(ann, str):  # postponed annotations
            ann = eval(ann, vars(ops))  # noqa: S307 - our own module's annotations
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[p.name] = (ann, Field(default))
    schema = create_model(f"{name}_args", **fields)

    def _not_direct(**_):
        raise RuntimeError("support tools are executed through ToolRegistry.call")

    return StructuredTool.from_function(func=_not_direct, name=name, description=inspect.getdoc(fn),
                                        args_schema=schema)


_SPECS: dict[str, StructuredTool] = {}


def llm_tools(names: list[str]) -> list[StructuredTool]:
    for n in names:
        if n not in _SPECS:
            _SPECS[n] = _spec_for(n)
    return [_SPECS[n] for n in names]


# ---------------------------------------------------------------------------
# MCP transport: one persistent stdio session on a background event loop
# ---------------------------------------------------------------------------

class MCPBridge:
    """Keeps an MCP ClientSession open on its own thread so sync code can call it."""

    def __init__(self, startup_timeout: float = 60.0, call_timeout: float = 30.0):
        self.call_timeout = call_timeout
        self.tool_names: list[str] = []
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session = None
        self._stop: Optional[asyncio.Event] = None
        self._thread = threading.Thread(target=self._run, name="udahub-mcp", daemon=True)
        self._thread.start()
        if not self._ready.wait(startup_timeout):
            raise TimeoutError("MCP server did not start in time")
        if self._error:
            raise RuntimeError(f"MCP server failed to start: {self._error!r}") from self._error

    def _run(self):
        # ipykernel installs a selector policy on Windows, which cannot spawn
        # subprocesses; the MCP stdio client needs the proactor loop.
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()
        finally:
            loop.close()

    async def _main(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stop = asyncio.Event()
        env = {**os.environ, "PYTHONUTF8": "1"}
        params = StdioServerParameters(command=sys.executable, args=[SERVER_PATH], env=env)
        with open(os.devnull, "w") as devnull:
            async with stdio_client(params, errlog=devnull) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.tool_names = [t.name for t in (await session.list_tools()).tools]
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()

    def call(self, name: str, args: dict) -> dict:
        fut = asyncio.run_coroutine_threadsafe(self._session.call_tool(name, args), self._loop)
        res = fut.result(timeout=self.call_timeout)
        data = getattr(res, "structuredContent", None)
        if not data:
            text = "".join(getattr(c, "text", "") for c in res.content)
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"ok": not res.isError, "data" if not res.isError else "error": text}
        if isinstance(data, dict) and "ok" not in data and "result" in data:
            data = data["result"]
        if getattr(res, "isError", False) and isinstance(data, dict) and data.get("ok") is not False:
            data = {"ok": False, "error": {"code": "tool_error", "message": str(data)[:300]}}
        return data

    def close(self):
        if self._loop and self._stop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(timeout=10)


class ToolRegistry:
    def __init__(self, transport: str | None = None):
        self.requested_transport = (transport or get_settings().tool_transport).lower()
        self.transport = "local"
        self.fallback_reason: str | None = None
        self._bridge: MCPBridge | None = None
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> "ToolRegistry":
        with self._lock:
            if self._started:
                return self
            if self.requested_transport == "mcp":
                try:
                    self._bridge = MCPBridge()
                    missing = set(ops.OPERATIONS) - set(self._bridge.tool_names)
                    if missing:
                        raise RuntimeError(f"MCP server is missing tools: {sorted(missing)}")
                    self.transport = "mcp"
                except Exception as exc:  # noqa: BLE001
                    self.fallback_reason = f"{exc.__class__.__name__}: {exc}"[:300]
                    self.transport = "local"
            self._started = True
            return self

    @property
    def tool_names(self) -> list[str]:
        return list(ops.OPERATIONS)

    def call(self, name: str, args: dict | None, *, user_id: str | None = None,
             ticket_id: str | None = None) -> dict:
        """Execute a tool; returns the envelope plus transport/latency metadata."""
        self.start()
        refused = self._validate(name, args, user_id)
        if refused:
            refused["_meta"] = {"transport": "refused", "latency_ms": 0.0}
            return refused
        args = {k: v for k, v in (args or {}).items() if k not in INJECTED_ARGS}
        params = inspect.signature(ops.OPERATIONS[name]).parameters
        if "user_id" in params:
            args["user_id"] = user_id
        if "ticket_id" in params and ticket_id:
            args["ticket_id"] = ticket_id
        return self._dispatch(name, args)

    @staticmethod
    def _validate(name: str, args: dict | None, user_id: str | None) -> dict | None:
        """Refuse calls that must never reach the database; None when the call is fine."""
        if name not in ops.OPERATIONS:
            return {"ok": False, "error": {"code": "invalid_argument", "message": f"unknown tool '{name}'"}}
        params = inspect.signature(ops.OPERATIONS[name]).parameters
        unknown = {k for k in (args or {}) if k not in INJECTED_ARGS} - set(params)
        if unknown:
            return {"ok": False, "error": {"code": "invalid_argument", "message": f"unexpected arguments {sorted(unknown)}"}}
        if "user_id" in params and not user_id:
            return {"ok": False, "error": {"code": "forbidden", "message": "this ticket is not linked to a CultPass customer"}}
        return None

    def _dispatch(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        try:
            if self.transport == "mcp" and self._bridge is not None:
                result = self._bridge.call(name, args)
            else:
                result = ops.OPERATIONS[name](**args)
        except Exception as exc:  # noqa: BLE001 - transport errors become tool errors
            result = {"ok": False, "error": {"code": "tool_error", "message": f"{exc.__class__.__name__}: {exc}"[:300]}}
        if not isinstance(result, dict) or "ok" not in result:
            result = {"ok": True, "data": result}
        result["_meta"] = {"transport": self.transport, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        return result

    def close(self):
        if self._bridge:
            self._bridge.close()
            self._bridge = None
        self._started = False
