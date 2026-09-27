"""Database access shared by tools, agents and the MCP server."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from agentic.config import SOLUTION_DIR, get_settings

if str(SOLUTION_DIR) not in sys.path:  # data.models lives beside agentic/
    sys.path.insert(0, str(SOLUTION_DIR))

from data.models import cultpass, udahub  # noqa: E402


class DatabaseNotInitialised(RuntimeError):
    pass


@lru_cache(maxsize=None)
def _engine(path: str) -> Engine:
    engine = create_engine(f"sqlite:///{path}", echo=False, connect_args={"check_same_thread": False, "timeout": 15})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _):
        # WAL + NORMAL sync: one fsync per checkpoint instead of per commit
        # (~160 ms -> ~2 ms per event row on Windows) and concurrent readers
        # while the MCP server process writes.
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    return engine


def _require(path: Path, notebook: str) -> str:
    if not path.exists():
        raise DatabaseNotInitialised(
            f"{path} does not exist. Run {notebook} first (or `python scripts/setup_databases.py`)."
        )
    return str(path)


def core_engine() -> Engine:
    engine = _engine(_require(get_settings().core_db, "02_core_db_setup.ipynb"))
    ensure_core_schema(engine)
    return engine


def cultpass_engine() -> Engine:
    return _engine(_require(get_settings().cultpass_db, "01_external_db_setup.ipynb"))


_schema_checked: set[str] = set()


def ensure_core_schema(engine: Engine) -> None:
    """Create the agent-support tables if the DB came from an older setup run."""
    key = str(engine.url)
    if key not in _schema_checked:
        udahub.Base.metadata.create_all(engine)
        _schema_checked.add(key)


@contextmanager
def session_scope(engine: Engine):
    session: Session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "cultpass", "udahub", "core_engine", "cultpass_engine", "session_scope",
    "DatabaseNotInitialised", "ensure_core_schema",
]
