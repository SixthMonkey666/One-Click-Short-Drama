from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import ASSETS_DIR, DEFAULT_IMAGE_RESOLUTION, PROJECT_DB, ensure_directories


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _connect() -> sqlite3.Connection:
    ensure_directories()
    connection = sqlite3.connect(PROJECT_DB)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          source_text TEXT NOT NULL,
          status TEXT NOT NULL,
          payload TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    return connection


def create_project(title: str, source_text: str, resolution_preset: str | None = None) -> dict[str, Any]:
    project_id = str(uuid4())
    timestamp = _now()
    preset = resolution_preset or DEFAULT_IMAGE_RESOLUTION
    project = {
        "id": project_id,
        "title": title.strip() or "未命名视频项目",
        "source_text": source_text.strip(),
        "status": "new",
        "analysis": None,
        "script": "",
        "characters": [],
        "shots": [],
        "review_history": [],
        "video": None,
        "resolution_preset": preset,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    with _connect() as connection:
        connection.execute(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                project_id,
                project["title"],
                project["source_text"],
                project["status"],
                json.dumps(project, ensure_ascii=False),
                timestamp,
                timestamp,
            ),
        )
    (ASSETS_DIR / project_id).mkdir(parents=True, exist_ok=True)
    return project


def save_project(project: dict[str, Any]) -> dict[str, Any]:
    project["updated_at"] = _now()
    with _connect() as connection:
        connection.execute(
            """UPDATE projects
               SET title = ?, source_text = ?, status = ?, payload = ?, updated_at = ?
               WHERE id = ?""",
            (
                project["title"],
                project["source_text"],
                project["status"],
                json.dumps(project, ensure_ascii=False),
                project["updated_at"],
                project["id"],
            ),
        )
    return project


def get_project(project_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute("SELECT payload FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not row:
        return None
    project = json.loads(row["payload"])
    if "analysis" not in project:
        project["analysis"] = None
    return project


def list_projects() -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute("SELECT payload FROM projects ORDER BY updated_at DESC").fetchall()
    projects = []
    for row in rows:
        p = json.loads(row["payload"])
        if "analysis" not in p:
            p["analysis"] = None
        projects.append(p)
    return projects


def _asset_subdirectory(*parts: str) -> Path:
    """Create an asset directory without allowing any path to escape ASSETS_DIR."""
    normalized_parts: list[str] = []
    for part in parts:
        value = str(part)
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("资产目录标识包含非法路径字符")
        normalized_parts.append(value)

    root = ASSETS_DIR.resolve()
    path = root.joinpath(*normalized_parts).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("资产目录不能位于项目数据目录之外")
    path.mkdir(parents=True, exist_ok=True)
    return path


def asset_directory(project_id: str, shot_id: str) -> Path:
    return _asset_subdirectory(project_id, shot_id)


def character_asset_directory(project_id: str, character_id: str) -> Path:
    return _asset_subdirectory(project_id, "characters", character_id)


def prop_asset_directory(project_id: str, prop_id: str) -> Path:
    return _asset_subdirectory(project_id, "props", prop_id)


def environment_asset_directory(project_id: str, env_id: str) -> Path:
    return _asset_subdirectory(project_id, "environments", env_id)


def delete_project(project_id: str) -> bool:
    project = get_project(project_id)
    if not project:
        return False
    with _connect() as connection:
        connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    asset_dir = _asset_subdirectory(project_id)
    if asset_dir.exists():
        shutil.rmtree(asset_dir, ignore_errors=True)
    return True


def rename_project(project_id: str, new_title: str) -> dict[str, Any] | None:
    project = get_project(project_id)
    if not project:
        return None
    project["title"] = new_title.strip() or "未命名视频项目"
    save_project(project)
    return project
