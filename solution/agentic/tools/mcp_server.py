"""FastMCP server exposing the CultPass support operations.

Run standalone (stdio transport, which is what the workflow uses):

    python agentic/tools/mcp_server.py

or inspect it with the MCP inspector:  fastmcp dev agentic/tools/mcp_server.py

Tools are generated from ``cultpass_ops.OPERATIONS`` so the MCP schema, the
in-process fallback tools and the implementation share one signature and one
docstring. Every tool returns the envelope {"ok": bool, "data" | "error"}.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

SOLUTION_DIR = Path(__file__).resolve().parents[2]
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from fastmcp import FastMCP  # noqa: E402

from agentic.tools import cultpass_ops as ops  # noqa: E402

mcp = FastMCP(
    name="cultpass-support-ops",
    instructions="Support operations over the CultPass customer database for UDA-Hub agents.",
)

for _name, _fn in ops.OPERATIONS.items():
    mcp.tool(_fn, name=_name, description=inspect.getdoc(_fn))


if __name__ == "__main__":
    if "show_banner" in inspect.signature(mcp.run).parameters:
        mcp.run(show_banner=False)
    else:
        mcp.run()
