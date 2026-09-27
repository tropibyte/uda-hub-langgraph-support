"""Central configuration for UDA-Hub.

Every path is absolute and derived from this file's location, so the agents,
the MCP server subprocess and the notebooks all resolve the same databases no
matter which directory they are launched from. Environment variables are read
at call time (not import time) so tests can point the system at a temporary
copy of the data.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

SOLUTION_DIR = Path(__file__).resolve().parents[1]
load_dotenv(SOLUTION_DIR / ".env")

ACCOUNT_ID = "cultpass"
VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    log_dir: Path
    chat_model: str
    embedding_model: str
    tool_transport: str          # "mcp" (default) or "local"
    checkpointer: str            # "sqlite" (default) or "memory"
    routing_strategy: str        # "ab" (default), "rules_first" or "llm_supervisor"
    offline: bool                # True -> no embeddings API, BM25-only retrieval
    # Decision thresholds (calibrated in evaluation/, see design doc section 6)
    min_retrieval_relevance: float
    escalation_confidence: float
    max_supervisor_hops: int
    max_tool_iterations: int

    @property
    def core_db(self) -> Path:
        return self.data_dir / "core" / "udahub.db"

    @property
    def cultpass_db(self) -> Path:
        return self.data_dir / "external" / "cultpass.db"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "core" / "udahub_checkpoints.db"

    @property
    def event_log(self) -> Path:
        return self.log_dir / "udahub_events.jsonl"


def get_settings() -> Settings:
    return Settings(
        data_dir=Path(os.getenv("UDAHUB_DATA_DIR", SOLUTION_DIR / "data")).resolve(),
        log_dir=Path(os.getenv("UDAHUB_LOG_DIR", SOLUTION_DIR / "logs")).resolve(),
        chat_model=os.getenv("UDAHUB_CHAT_MODEL", "gpt-4.1-mini"),
        embedding_model=os.getenv("UDAHUB_EMBEDDING_MODEL", "text-embedding-3-small"),
        tool_transport=os.getenv("UDAHUB_TOOL_TRANSPORT", "mcp").lower(),
        checkpointer=os.getenv("UDAHUB_CHECKPOINTER", "sqlite").lower(),
        routing_strategy=os.getenv("UDAHUB_ROUTING_STRATEGY", "ab").lower(),
        offline=os.getenv("UDAHUB_OFFLINE", "0") == "1" or not api_key(),
        min_retrieval_relevance=_float("UDAHUB_MIN_RETRIEVAL_RELEVANCE", 0.35),
        escalation_confidence=_float("UDAHUB_ESCALATION_CONFIDENCE", 0.60),
        max_supervisor_hops=int(os.getenv("UDAHUB_MAX_HOPS", "4")),
        max_tool_iterations=int(os.getenv("UDAHUB_MAX_TOOL_ITERATIONS", "6")),
    )


def api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY") or os.getenv("VOCAREUM_API_KEY")


def openai_kwargs() -> dict:
    """Keyword arguments for ChatOpenAI / OpenAIEmbeddings.

    A Vocareum key (``voc-...``) only works against the Vocareum gateway, so the
    base URL defaults to it for those keys unless OPENAI_BASE_URL overrides it.
    """
    key = api_key()
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url and key and key.startswith("voc-"):
        base_url = VOCAREUM_BASE_URL
    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs
