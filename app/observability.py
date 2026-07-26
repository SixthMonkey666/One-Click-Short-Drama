from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from app.config import LANGFUSE_ENABLED


def trace_context(name: str, project_id: str):
    """Return a Langfuse trace context when credentials exist, else a no-op context."""
    if not LANGFUSE_ENABLED:
        return nullcontext()
    try:
        from langfuse import propagate_attributes

        return propagate_attributes(
            trace_name=name,
            session_id=project_id,
            tags=["storyboard-flow", "local"],
            metadata={"project_id": project_id},
        )
    except Exception:
        return nullcontext()


def callbacks() -> list[Any]:
    """Return Langfuse callbacks only when local tracing is explicitly configured."""
    if not LANGFUSE_ENABLED:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:
        return []
