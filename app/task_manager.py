"""Process-wide generation task manager with SQLite-backed status snapshots.

Streamlit reruns must not own long-running model inference. This module owns one
background worker for the whole application process and enforces a single active
generation task across all users, projects, and model types.
"""

from __future__ import annotations

import gc
import json
import os
import threading
import time
import traceback
from collections.abc import Callable, Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import PROJECT_DB, ensure_directories

TaskRunner = Callable[[Callable[[dict[str, Any]], None]], Any]
ACTIVE_STATUSES = ("queued", "running")
TERMINAL_STATUSES = ("completed", "failed", "interrupted")
MAX_EVENTS = 120
TASK_HEARTBEAT_SECONDS = 5.0


class TaskBusyError(RuntimeError):
    def __init__(self, active_task: dict[str, Any]):
        self.active_task = active_task
        label = active_task.get("title") or active_task.get("kind") or "生成任务"
        super().__init__(f"已有任务正在运行：{label}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _connect():
    import sqlite3

    ensure_directories()
    connection = sqlite3.connect(PROJECT_DB, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS generation_tasks (
          id TEXT PRIMARY KEY,
          kind TEXT NOT NULL,
          project_id TEXT,
          title TEXT NOT NULL,
          status TEXT NOT NULL,
          payload TEXT NOT NULL,
          owner_pid INTEGER NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS generation_tasks_one_active
        ON generation_tasks ((1))
        WHERE status IN ('queued', 'running')
        """
    )
    return connection


def _row_payload(row) -> dict[str, Any] | None:
    return json.loads(row["payload"]) if row else None


class GenerationTaskManager:
    def __init__(self, database_path: Path = PROJECT_DB):
        self.database_path = database_path
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._recover_orphaned_tasks()

    def _recover_orphaned_tasks(self) -> None:
        """A worker cannot survive its Python process, so release locks from old PIDs."""
        now = _now()
        with _connect() as connection:
            rows = connection.execute(
                "SELECT id, payload FROM generation_tasks "
                "WHERE status IN ('queued', 'running') AND owner_pid != ?",
                (os.getpid(),),
            ).fetchall()
            for row in rows:
                payload = json.loads(row["payload"])
                payload.update({
                    "status": "interrupted",
                    "message": "应用进程已重启，原后台任务无法继续，请重新提交",
                    "finished_at": now,
                    "updated_at": now,
                })
                connection.execute(
                    "UPDATE generation_tasks SET status = 'interrupted', payload = ?, updated_at = ? "
                    "WHERE id = ?",
                    (_json(payload), now, row["id"]),
                )

    def submit(
        self,
        kind: str,
        title: str,
        runner: TaskRunner,
        project_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task_id = str(uuid4())
        now = _now()
        task = {
            "id": task_id,
            "kind": kind,
            "project_id": project_id,
            "context": dict(context or {}),
            "title": title,
            "status": "queued",
            "message": "任务已提交，正在启动后台线程",
            "phase": "queued",
            "events": [],
            "result": None,
            "error": None,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "updated_at": now,
            "heartbeat_at": now,
        }
        with self._lock:
            connection = _connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT payload FROM generation_tasks "
                    "WHERE status IN ('queued', 'running') LIMIT 1"
                ).fetchone()
                if row:
                    connection.rollback()
                    raise TaskBusyError(json.loads(row["payload"]))
                connection.execute(
                    "INSERT INTO generation_tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        task_id,
                        kind,
                        project_id,
                        title,
                        "queued",
                        _json(task),
                        os.getpid(),
                        now,
                        now,
                    ),
                )
                connection.commit()
            finally:
                connection.close()
            self._tasks[task_id] = task
            thread = threading.Thread(
                target=self._run,
                args=(task_id, runner),
                name=f"generation-task-{task_id[:8]}",
                daemon=True,
            )
            self._threads[task_id] = thread
            thread.start()
            heartbeat = threading.Thread(
                target=self._heartbeat_loop,
                args=(task_id,),
                name=f"generation-heartbeat-{task_id[:8]}",
                daemon=True,
            )
            heartbeat.start()
            return dict(task)

    def _heartbeat_loop(self, task_id: str) -> None:
        while True:
            time.sleep(TASK_HEARTBEAT_SECONDS)
            with self._lock:
                task = self._tasks.get(task_id)
                worker = self._threads.get(task_id)
                if not task or task.get("status") not in ACTIVE_STATUSES:
                    return
                worker_alive = bool(worker and worker.is_alive())
            self._append_event(task_id, {
                "type": "task_heartbeat",
                "worker_alive": worker_alive,
                "timestamp": _now(),
            }, {
                "heartbeat_at": _now(),
                "worker_alive": worker_alive,
            }, persist=True)

    def _run(self, task_id: str, runner: TaskRunner) -> None:
        self._mutate(task_id, {
            "status": "running",
            "phase": "starting",
            "message": "后台任务已启动",
            "started_at": _now(),
        }, persist=True)
        last_persisted = 0.0

        def emit(event: dict[str, Any]) -> None:
            nonlocal last_persisted
            normalized = dict(event)
            normalized.setdefault("timestamp", _now())
            update = self._event_update(normalized)
            now_monotonic = time.monotonic()
            important = normalized.get("type") not in {"text_delta", "generation_heartbeat"}
            persist = important or now_monotonic - last_persisted >= 1.0
            if persist:
                last_persisted = now_monotonic
            self._append_event(task_id, normalized, update, persist=persist)

        try:
            result = runner(emit)
            self._mutate(task_id, {
                "status": "completed",
                "phase": "completed",
                "message": "任务执行完成",
                "result": result,
                "finished_at": _now(),
            }, persist=True)
        except BaseException as exc:
            self._append_event(task_id, {
                "type": "task_failed",
                "message": str(exc),
                "error_type": type(exc).__name__,
                "timestamp": _now(),
            }, {}, persist=False)
            self._mutate(task_id, {
                "status": "failed",
                "phase": "failed",
                "message": f"任务失败：{exc}",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "finished_at": _now(),
            }, persist=True)
        finally:
            try:
                from app.providers import release_all

                release_all()
            except Exception:
                pass
            gc.collect()
            with self._lock:
                self._threads.pop(task_id, None)

    @staticmethod
    def _event_update(event: dict[str, Any]) -> dict[str, Any]:
        event_type = str(event.get("type") or "")
        update: dict[str, Any] = {"heartbeat_at": _now()}
        if event.get("message"):
            update["message"] = str(event["message"])
        if event_type == "generation_phase":
            update["phase"] = event.get("phase") or "generation"
        elif event_type == "generation_heartbeat":
            update["phase"] = event.get("phase") or "generation"
        elif event_type == "node_started":
            update["phase"] = event.get("node") or "agent"
            update["message"] = f"Agent 正在执行：{event.get('node', '')}"
        elif event_type in {"asset_start", "keyframe_start"}:
            update["phase"] = "asset_generation"
            update["current_asset"] = (
                event.get("asset_name") or event.get("entity_name") or event.get("shot_id")
            )
            update["current_view"] = event.get("view_label") or event.get("view")
        elif event_type in {"asset_progress", "keyframe_progress"}:
            update["step"] = int(event.get("step") or 0)
            update["total_steps"] = int(event.get("total_steps") or 1)
        elif event_type == "assets_start":
            update["overall_done"] = 0
            update["overall_total"] = int(event.get("total") or 0)
        elif event_type == "asset_progress_overall":
            update["overall_done"] = int(event.get("done") or 0)
            update["overall_total"] = int(event.get("total") or 0)
        elif event_type == "judge_batch_start":
            update["phase"] = "auto_judge"
            update["overall_done"] = 0
            update["overall_total"] = int(event.get("total") or 0)
        elif event_type == "judge_item_start":
            update["phase"] = "auto_judge"
            update["current_asset"] = event.get("asset_name")
            update["overall_done"] = max(int(event.get("index") or 1) - 1, 0)
            update["overall_total"] = int(event.get("total") or 0)
        elif event_type in {"judge_item_done", "judge_item_failed"}:
            update["phase"] = "auto_judge"
            update["overall_done"] = int(event.get("index") or 0)
            update["overall_total"] = int(event.get("total") or 0)
        elif event_type == "judge_batch_complete":
            update["phase"] = "auto_judge_complete"
            update["overall_done"] = int(event.get("total") or 0)
            update["overall_total"] = int(event.get("total") or 0)
        elif event_type == "text_delta":
            update["phase"] = "text_generation"
        return update

    def _append_event(
        self,
        task_id: str,
        event: dict[str, Any],
        update: dict[str, Any],
        *,
        persist: bool,
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id) or self.get(task_id)
            if not task:
                return
            events = list(task.get("events", []))
            if event.get("type") == "text_delta" and events and events[-1].get("type") == "text_delta":
                previous = dict(events[-1])
                previous["delta"] = (
                    str(previous.get("delta") or "") + str(event.get("delta") or "")
                )[-1000:]
                previous["timestamp"] = event["timestamp"]
                events[-1] = previous
            else:
                events.append(event)
            task.update(update)
            task["events"] = events[-MAX_EVENTS:]
            task["updated_at"] = _now()
            self._tasks[task_id] = task
            if persist:
                self._persist(task)

    def _mutate(self, task_id: str, update: dict[str, Any], *, persist: bool) -> None:
        with self._lock:
            task = self._tasks.get(task_id) or self.get(task_id)
            if not task:
                return
            task.update(update)
            task["updated_at"] = _now()
            task["heartbeat_at"] = task["updated_at"]
            self._tasks[task_id] = task
            if persist:
                self._persist(task)

    @staticmethod
    def _persist(task: dict[str, Any]) -> None:
        with _connect() as connection:
            connection.execute(
                "UPDATE generation_tasks SET status = ?, payload = ?, updated_at = ? WHERE id = ?",
                (task["status"], _json(task), task["updated_at"], task["id"]),
            )

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            if task_id in self._tasks:
                return json.loads(_json(self._tasks[task_id]))
        with _connect() as connection:
            row = connection.execute(
                "SELECT payload FROM generation_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        task = _row_payload(row)
        if task and task.get("status") in ACTIVE_STATUSES:
            with self._lock:
                self._tasks[task_id] = task
        return task

    def active(self) -> dict[str, Any] | None:
        with _connect() as connection:
            row = connection.execute(
                "SELECT id, payload FROM generation_tasks "
                "WHERE status IN ('queued', 'running') ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return self.get(row["id"]) or _row_payload(row)

    def latest(self, project_id: str | None = None, kind: str | None = None) -> dict[str, Any] | None:
        query = "SELECT payload FROM generation_tasks"
        conditions: list[str] = []
        values: list[Any] = []
        if project_id is not None:
            conditions.append("project_id = ?")
            values.append(project_id)
        if kind is not None:
            conditions.append("kind = ?")
            values.append(kind)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT 1"
        with _connect() as connection:
            row = connection.execute(query, values).fetchone()
        return _row_payload(row)

    def wait(self, task_id: str, timeout: float = 10.0) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = self.get(task_id)
            if not task or task.get("status") in TERMINAL_STATUSES:
                return task
            time.sleep(0.01)
        return self.get(task_id)


_MANAGER = GenerationTaskManager()


def submit_task(
    kind: str,
    title: str,
    runner: TaskRunner,
    project_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _MANAGER.submit(kind, title, runner, project_id, context)


def submit_generator(
    kind: str,
    title: str,
    generator_factory: Callable[[], Generator],
    project_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def runner(emit):
        generator = generator_factory()
        while True:
            try:
                event = next(generator)
                if isinstance(event, dict):
                    emit(event)
            except StopIteration as stop:
                return stop.value

    return submit_task(kind, title, runner, project_id, context)


def get_active_task() -> dict[str, Any] | None:
    return _MANAGER.active()


def get_task(task_id: str) -> dict[str, Any] | None:
    return _MANAGER.get(task_id)


def get_latest_task(
    project_id: str | None = None,
    kind: str | None = None,
) -> dict[str, Any] | None:
    return _MANAGER.latest(project_id, kind)


def wait_for_task(task_id: str, timeout: float = 10.0) -> dict[str, Any] | None:
    return _MANAGER.wait(task_id, timeout)
