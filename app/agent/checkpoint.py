from __future__ import annotations

import sqlite3
import threading
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import CHECKPOINT_DB, ensure_directories

_CHECKPOINT_LOCK = threading.RLock()


def create_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteSaver:
    with _CHECKPOINT_LOCK:
        ensure_directories()
        return create_checkpointer(CHECKPOINT_DB)


def thread_config(project_id: str) -> dict:
    if not project_id:
        raise ValueError("project_id 不能为空")
    return {"configurable": {"thread_id": project_id}}
