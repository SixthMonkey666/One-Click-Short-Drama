from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import ELO_INITIAL_RATING, ELO_K_FACTOR, EVALUATION_DB, ensure_directories
from app.evaluation.migrations import add_missing_columns
from app.evaluation.rubric import Rubric, load_default_rubric


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _connect() -> sqlite3.Connection:
    ensure_directories()
    conn = sqlite3.connect(EVALUATION_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rubric_versions (
            id TEXT PRIMARY KEY,
            version TEXT NOT NULL,
            name TEXT NOT NULL,
            task_type TEXT NOT NULL,
            dimensions_json TEXT NOT NULL,
            bad_case_tags_json TEXT NOT NULL,
            is_default INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            project_id TEXT,
            task_type TEXT NOT NULL,
            rubric_version_id TEXT NOT NULL,
            judge_type TEXT NOT NULL,
            judge_provider TEXT,
            judge_model_version TEXT,
            prompt_version TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (rubric_version_id) REFERENCES rubric_versions(id)
        );

        CREATE TABLE IF NOT EXISTS evaluation_items (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            project_id TEXT,
            shot_id TEXT,
            asset_id TEXT,
            asset_type TEXT NOT NULL,
            asset_path TEXT NOT NULL,
            prompt TEXT,
            reference_images_json TEXT,
            model_provider TEXT,
            model_version TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            order_index INTEGER DEFAULT 0,
            total_score REAL,
            note TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            submitted_at TEXT,
            FOREIGN KEY (run_id) REFERENCES evaluation_runs(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS dimension_scores (
            id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            dimension_key TEXT NOT NULL,
            score INTEGER NOT NULL,
            weighted_score REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (item_id) REFERENCES evaluation_items(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS bad_case_tags (
            id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (item_id) REFERENCES evaluation_items(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS pairwise_comparisons (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            task_type TEXT NOT NULL,
            prompt TEXT,
            asset_a_path TEXT NOT NULL,
            asset_b_path TEXT NOT NULL,
            model_a TEXT NOT NULL,
            model_b TEXT NOT NULL,
            model_a_version TEXT,
            model_b_version TEXT,
            winner TEXT,
            reason_tags_json TEXT,
            reason_note TEXT,
            judge_type TEXT NOT NULL DEFAULT 'human',
            judged_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_rubric_task_type ON rubric_versions(task_type)",
        "CREATE INDEX IF NOT EXISTS idx_rubric_default ON rubric_versions(is_default)",
        "CREATE INDEX IF NOT EXISTS idx_runs_project ON evaluation_runs(project_id)",
        "CREATE INDEX IF NOT EXISTS idx_runs_status ON evaluation_runs(status)",
        "CREATE INDEX IF NOT EXISTS idx_runs_judge_type ON evaluation_runs(judge_type)",
        "CREATE INDEX IF NOT EXISTS idx_items_run ON evaluation_items(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_items_project ON evaluation_items(project_id)",
        "CREATE INDEX IF NOT EXISTS idx_items_status ON evaluation_items(status)",
        "CREATE INDEX IF NOT EXISTS idx_items_asset_type ON evaluation_items(asset_type)",
        "CREATE INDEX IF NOT EXISTS idx_items_model ON evaluation_items(model_provider)",
        "CREATE INDEX IF NOT EXISTS idx_scores_item ON dimension_scores(item_id)",
        "CREATE INDEX IF NOT EXISTS idx_tags_item ON bad_case_tags(item_id)",
        "CREATE INDEX IF NOT EXISTS idx_pairwise_task ON pairwise_comparisons(task_type)",
        "CREATE INDEX IF NOT EXISTS idx_pairwise_judged ON pairwise_comparisons(judged_at)",
    ]
    for idx_sql in indexes:
        conn.execute(idx_sql)

    add_missing_columns(
        conn,
        "dimension_scores",
        {"judge_type": "TEXT NOT NULL DEFAULT 'human'"},
    )
    add_missing_columns(
        conn,
        "bad_case_tags",
        {"judge_type": "TEXT NOT NULL DEFAULT 'human'"},
    )
    add_missing_columns(
        conn,
        "evaluation_items",
        {
            "auto_total_score": "REAL",
            "auto_judge_provider": "TEXT",
            "auto_confidence": "REAL",
            "auto_explanation": "TEXT",
            "auto_judged_at": "TEXT",
            "auto_raw_response": "TEXT",
        },
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scores_judge "
        "ON dimension_scores(item_id, judge_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tags_judge "
        "ON bad_case_tags(item_id, judge_type)"
    )

    conn.commit()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def _seed_default_rubric(conn: sqlite3.Connection) -> str:
    existing = conn.execute(
        "SELECT id FROM rubric_versions WHERE is_default = 1 LIMIT 1"
    ).fetchone()
    if existing:
        return existing["id"]

    rubric = load_default_rubric()
    rubric_id = str(uuid4())
    ts = _now()
    conn.execute(
        "INSERT INTO rubric_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            rubric_id,
            rubric.version,
            rubric.name,
            rubric.task_type,
            rubric.to_json(),
            json.dumps(rubric.bad_case_tags, ensure_ascii=False),
            1,
            ts,
        ),
    )
    conn.commit()
    return rubric_id


def create_rubric_version(rubric: Rubric, is_default: bool = False) -> dict[str, Any]:
    """Persist an immutable rubric version, reusing an exact named version."""
    with _connect() as conn:
        existing = conn.execute(
            """SELECT * FROM rubric_versions
               WHERE version = ? AND name = ? AND task_type = ?
               LIMIT 1""",
            (rubric.version, rubric.name, rubric.task_type),
        ).fetchone()
        if existing:
            row = existing
        else:
            rubric_id = str(uuid4())
            conn.execute(
                """INSERT INTO rubric_versions
                   (id, version, name, task_type, dimensions_json,
                    bad_case_tags_json, is_default, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rubric_id,
                    rubric.version,
                    rubric.name,
                    rubric.task_type,
                    rubric.to_json(),
                    json.dumps(rubric.bad_case_tags, ensure_ascii=False),
                    int(is_default),
                    _now(),
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM rubric_versions WHERE id = ?", (rubric_id,)
            ).fetchone()

    result = dict(row)
    result["rubric"] = Rubric.from_json(result["dimensions_json"])
    result["bad_case_tags"] = json.loads(result["bad_case_tags_json"])
    return result


def _seed_file_rubrics() -> None:
    """Import rubric YAML files as immutable versions without replacing history."""
    from app.config import RUBRICS_DIR

    for path in sorted(RUBRICS_DIR.glob("*.yaml")):
        rubric = Rubric.from_yaml_file(path)
        create_rubric_version(rubric, is_default=False)


def get_or_create_default_rubric() -> dict[str, Any]:
    with _connect() as conn:
        rid = _seed_default_rubric(conn)
        row = conn.execute("SELECT * FROM rubric_versions WHERE id = ?", (rid,)).fetchone()
        d = _row_to_dict(row)
        if d:
            d["rubric"] = Rubric.from_json(d["dimensions_json"])
            d["bad_case_tags"] = json.loads(d["bad_case_tags_json"])
        return d


def list_rubrics() -> list[dict[str, Any]]:
    with _connect() as conn:
        _seed_default_rubric(conn)
    _seed_file_rubrics()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM rubric_versions ORDER BY is_default DESC, created_at DESC"
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["rubric"] = Rubric.from_json(d["dimensions_json"])
        d["bad_case_tags"] = json.loads(d["bad_case_tags_json"])
        results.append(d)
    return results


def create_evaluation_run(
    name: str,
    task_type: str,
    judge_type: str = "human",
    project_id: str | None = None,
    rubric_version_id: str | None = None,
    judge_provider: str | None = None,
    judge_model_version: str | None = None,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    run_id = str(uuid4())
    ts = _now()
    with _connect() as conn:
        if rubric_version_id is None:
            rubric_version_id = _seed_default_rubric(conn)
        rubric_exists = conn.execute(
            "SELECT 1 FROM rubric_versions WHERE id = ?", (rubric_version_id,)
        ).fetchone()
        if not rubric_exists:
            raise ValueError(f"Rubric version does not exist: {rubric_version_id}")
        conn.execute(
            "INSERT INTO evaluation_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, name, project_id, task_type, rubric_version_id,
                judge_type, judge_provider, judge_model_version, prompt_version,
                "pending", ts, ts,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM evaluation_runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row)


def add_evaluation_item(
    run_id: str,
    asset_path: str,
    asset_type: str,
    project_id: str | None = None,
    shot_id: str | None = None,
    asset_id: str | None = None,
    prompt: str | None = None,
    reference_images: list[str] | None = None,
    model_provider: str | None = None,
    model_version: str | None = None,
    order_index: int = 0,
) -> dict[str, Any]:
    item_id = str(uuid4())
    ts = _now()
    ref_json = json.dumps(reference_images or [], ensure_ascii=False)
    with _connect() as conn:
        conn.execute(
            """INSERT INTO evaluation_items
               (id, run_id, project_id, shot_id, asset_id, asset_type, asset_path,
                prompt, reference_images_json, model_provider, model_version,
                status, order_index, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
            (item_id, run_id, project_id, shot_id, asset_id, asset_type, asset_path,
             prompt, ref_json, model_provider, model_version, order_index, ts, ts),
        )
        conn.execute(
            "UPDATE evaluation_runs SET updated_at = ?, status = 'in_progress' WHERE id = ?",
            (ts, run_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM evaluation_items WHERE id = ?", (item_id,)).fetchone()
    return dict(row)


def list_runs(project_id: str | None = None, judge_type: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM evaluation_runs WHERE 1=1"
    params: list[Any] = []
    if project_id:
        sql += " AND project_id = ?"
        params.append(project_id)
    if judge_type:
        sql += " AND judge_type = ?"
        params.append(judge_type)
    sql += " ORDER BY updated_at DESC"
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM evaluation_runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        rubric_row = conn.execute(
            "SELECT * FROM rubric_versions WHERE id = ?", (d["rubric_version_id"],)
        ).fetchone()
        if rubric_row:
            d["rubric"] = Rubric.from_json(rubric_row["dimensions_json"])
            d["bad_case_tags"] = json.loads(rubric_row["bad_case_tags_json"])
        return d


def list_run_items(run_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM evaluation_items WHERE run_id = ? ORDER BY order_index ASC, created_at ASC",
            (run_id,),
        ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["reference_images"] = json.loads(item.get("reference_images_json") or "[]")
            scores = conn.execute(
                "SELECT dimension_key, score, weighted_score FROM dimension_scores WHERE item_id = ? AND judge_type = 'human'",
                (item["id"],),
            ).fetchall()
            item["scores"] = {s["dimension_key"]: dict(s) for s in scores}
            auto_scores = conn.execute(
                "SELECT dimension_key, score, weighted_score FROM dimension_scores WHERE item_id = ? AND judge_type != 'human'",
                (item["id"],),
            ).fetchall()
            item["auto_scores"] = {s["dimension_key"]: dict(s) for s in auto_scores}
            tags = conn.execute(
                "SELECT tag, note FROM bad_case_tags WHERE item_id = ? AND judge_type = 'human'", (item["id"],)
            ).fetchall()
            item["bad_case_tags"] = [dict(t) for t in tags]
            auto_tags = conn.execute(
                "SELECT tag, note FROM bad_case_tags WHERE item_id = ? AND judge_type != 'human'", (item["id"],)
            ).fetchall()
            item["auto_bad_case_tags"] = [dict(t) for t in auto_tags]
            results.append(item)
        return results


def get_item(item_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM evaluation_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["reference_images"] = json.loads(item.get("reference_images_json") or "[]")
        scores = conn.execute(
            "SELECT dimension_key, score, weighted_score FROM dimension_scores WHERE item_id = ? AND judge_type = 'human'",
            (item["id"],),
        ).fetchall()
        item["scores"] = {s["dimension_key"]: dict(s) for s in scores}
        auto_scores = conn.execute(
            "SELECT dimension_key, score, weighted_score FROM dimension_scores WHERE item_id = ? AND judge_type != 'human'",
            (item["id"],),
        ).fetchall()
        item["auto_scores"] = {s["dimension_key"]: dict(s) for s in auto_scores}
        tags = conn.execute(
            "SELECT tag, note FROM bad_case_tags WHERE item_id = ? AND judge_type = 'human'", (item["id"],)
        ).fetchall()
        item["bad_case_tags"] = [dict(t) for t in tags]
        auto_tags = conn.execute(
            "SELECT tag, note FROM bad_case_tags WHERE item_id = ? AND judge_type != 'human'", (item["id"],)
        ).fetchall()
        item["auto_bad_case_tags"] = [dict(t) for t in auto_tags]
        return item


def save_item_scores(
    item_id: str,
    scores: dict[str, int],
    bad_case_tags: list[dict[str, str]] | None = None,
    note: str | None = None,
    total_score: float | None = None,
    submit: bool = False,
) -> dict[str, Any] | None:
    ts = _now()
    with _connect() as conn:
        item = conn.execute("SELECT * FROM evaluation_items WHERE id = ?", (item_id,)).fetchone()
        if not item:
            return None

        run = dict(conn.execute("SELECT * FROM evaluation_runs WHERE id = ?", (item["run_id"],)).fetchone())
        rubric_row = conn.execute(
            "SELECT dimensions_json FROM rubric_versions WHERE id = ?", (run["rubric_version_id"],)
        ).fetchone()
        rubric = Rubric.from_json(rubric_row["dimensions_json"]) if rubric_row else None

        conn.execute("DELETE FROM dimension_scores WHERE item_id = ? AND judge_type = 'human'", (item_id,))
        conn.execute("DELETE FROM bad_case_tags WHERE item_id = ? AND judge_type = 'human'", (item_id,))

        for dim_key, score_val in scores.items():
            if score_val is None or score_val < 1 or score_val > 5:
                continue
            weighted = None
            if rubric:
                dim = rubric.get_dimension(dim_key)
                if dim:
                    weighted = score_val * dim.weight / rubric.total_weight * 5 if rubric.total_weight > 0 else score_val
            sid = str(uuid4())
            conn.execute(
                "INSERT INTO dimension_scores (id, item_id, dimension_key, score, weighted_score, judge_type, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'human', ?, ?)",
                (sid, item_id, dim_key, int(score_val), weighted, ts, ts),
            )

        if bad_case_tags:
            for tag_entry in bad_case_tags:
                tag = str(tag_entry.get("tag", "")).strip()
                if not tag:
                    continue
                tid = str(uuid4())
                conn.execute(
                    "INSERT INTO bad_case_tags (id, item_id, tag, note, judge_type, created_at) VALUES (?, ?, ?, ?, 'human', ?)",
                    (tid, item_id, tag, tag_entry.get("note"), ts),
                )

        submitted_at = item["submitted_at"]
        status = item["status"]
        if submit:
            status = "submitted"
            submitted_at = ts
        elif scores:
            status = "in_progress"

        conn.execute(
            """UPDATE evaluation_items
               SET status = ?, total_score = ?, note = ?, updated_at = ?, submitted_at = ?
               WHERE id = ?""",
            (status, total_score, note, ts, submitted_at, item_id),
        )

        remaining = conn.execute(
            "SELECT COUNT(*) as c FROM evaluation_items WHERE run_id = ? AND status != 'submitted'",
            (item["run_id"],),
        ).fetchone()["c"]
        run_status = "completed" if remaining == 0 else "in_progress"
        conn.execute(
            "UPDATE evaluation_runs SET status = ?, updated_at = ? WHERE id = ?",
            (run_status, ts, item["run_id"]),
        )
        conn.commit()

        return get_item(item_id)


def delete_run(run_id: str) -> bool:
    with _connect() as conn:
        conn.execute("DELETE FROM evaluation_runs WHERE id = ?", (run_id,))
        conn.commit()
    return True


def get_run_progress(run_id: str) -> dict[str, Any]:
    with _connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) as c FROM evaluation_items WHERE run_id = ?", (run_id,)
        ).fetchone()["c"]
        submitted = conn.execute(
            "SELECT COUNT(*) as c FROM evaluation_items WHERE run_id = ? AND status = 'submitted'",
            (run_id,),
        ).fetchone()["c"]
        in_progress = conn.execute(
            "SELECT COUNT(*) as c FROM evaluation_items WHERE run_id = ? AND status = 'in_progress'",
            (run_id,),
        ).fetchone()["c"]
    return {
        "total": total,
        "submitted": submitted,
        "in_progress": in_progress,
        "pending": total - submitted - in_progress,
        "progress_pct": (submitted / total * 100) if total > 0 else 0.0,
    }


def get_run_stats(run_id: str) -> dict[str, Any]:
    with _connect() as conn:
        run = conn.execute("SELECT * FROM evaluation_runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            return {}
        rubric_row = conn.execute(
            "SELECT dimensions_json FROM rubric_versions WHERE id = ?", (run["rubric_version_id"],)
        ).fetchone()
        rubric = Rubric.from_json(rubric_row["dimensions_json"]) if rubric_row else None

        items = conn.execute(
            "SELECT * FROM evaluation_items WHERE run_id = ?", (run_id,)
        ).fetchall()

        total_scores = [it["total_score"] for it in items if it["total_score"] is not None]
        dim_scores: dict[str, list[int]] = {d.key: [] for d in (rubric.dimensions if rubric else [])}
        tag_counts: dict[str, int] = {}
        model_scores: dict[str, list[float]] = {}

        for it in items:
            if it["model_provider"]:
                model_scores.setdefault(it["model_provider"], []).append(
                    it["total_score"] or 0.0
                )
            scores = conn.execute(
                "SELECT dimension_key, score FROM dimension_scores WHERE item_id = ? AND judge_type = 'human'",
                (it["id"],),
            ).fetchall()
            for s in scores:
                dim_scores.setdefault(s["dimension_key"], []).append(s["score"])
            tags = conn.execute("SELECT tag FROM bad_case_tags WHERE item_id = ? AND judge_type = 'human'", (it["id"],)).fetchall()
            for t in tags:
                tag_counts[t["tag"]] = tag_counts.get(t["tag"], 0) + 1

    avg_total = sum(total_scores) / len(total_scores) if total_scores else 0.0
    dim_avgs = {k: (sum(v) / len(v) if v else 0.0) for k, v in dim_scores.items()}
    model_avgs = {k: (sum(v) / len(v) if v else 0.0) for k, v in model_scores.items()}

    low_scores = []
    for it in items:
        if it["total_score"] is not None and it["total_score"] < 3.0:
            low_scores.append(dict(it))

    return {
        "total_items": len(items),
        "submitted_items": sum(1 for it in items if it["status"] == "submitted"),
        "avg_total_score": round(avg_total, 2),
        "dimension_averages": {k: round(v, 2) for k, v in dim_avgs.items()},
        "tag_counts": tag_counts,
        "model_averages": {k: round(v, 2) for k, v in model_avgs.items()},
        "low_score_items": low_scores[:20],
        "score_distribution": _score_distribution(total_scores),
    }


def _score_distribution(scores: list[float]) -> dict[str, int]:
    dist = {"1分": 0, "2分": 0, "3分": 0, "4分": 0, "5分": 0}
    for s in scores:
        rounded = max(1, min(5, round(s)))
        dist[f"{rounded}分"] += 1
    return dist


def discover_project_assets(project: dict[str, Any]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    pid = project.get("id")
    if not pid:
        return assets

    from app.repository import (
        character_asset_directory,
    )

    for shot in project.get("shots", []):
        shot_id = shot.get("id")
        for img in shot.get("image_versions", []):
            path = img.get("path")
            if path and Path(path).exists():
                assets.append({
                    "asset_type": "keyframe",
                    "project_id": pid,
                    "shot_id": shot_id,
                    "asset_id": shot_id,
                    "asset_path": path,
                    "prompt": shot.get("prompt", ""),
                    "model_provider": img.get("provider", "unknown"),
                    "model_version": f"v{img.get('version', 1)}",
                    "title": shot.get("title", f"镜头{shot.get('number', '?')}"),
                    "number": shot.get("number"),
                })

    analysis = project.get("analysis") or {}
    for char in analysis.get("characters", []) or project.get("characters", []):
        cid = char.get("id")
        _cdir = character_asset_directory(pid, cid)
        for img in char.get("reference_images", []):
            path = img.get("path")
            if path and Path(path).exists():
                assets.append({
                    "asset_type": "character",
                    "project_id": pid,
                    "asset_id": cid,
                    "asset_path": path,
                    "prompt": char.get("appearance") or char.get("prompt", ""),
                    "model_provider": img.get("provider", "unknown"),
                    "model_version": f"v{img.get('version', 1)}",
                    "title": f"{char.get('name', '角色')} · {img.get('view_label', img.get('view', ''))}",
                })

    for prop in analysis.get("props", []):
        pid_prop = prop.get("id")
        for img in prop.get("reference_images", []):
            path = img.get("path")
            if path and Path(path).exists():
                assets.append({
                    "asset_type": "prop",
                    "project_id": pid,
                    "asset_id": pid_prop,
                    "asset_path": path,
                    "prompt": prop.get("visual_prompt") or prop.get("prompt", ""),
                    "model_provider": img.get("provider", "unknown"),
                    "model_version": f"v{img.get('version', 1)}",
                    "title": f"道具: {prop.get('name', '')}",
                })

    for env in analysis.get("environments", []):
        eid = env.get("id")
        for img in env.get("reference_images", []):
            path = img.get("path")
            if path and Path(path).exists():
                assets.append({
                    "asset_type": "environment",
                    "project_id": pid,
                    "asset_id": eid,
                    "asset_path": path,
                    "prompt": env.get("visual_prompt") or env.get("prompt", ""),
                    "model_provider": img.get("provider", "unknown"),
                    "model_version": f"v{img.get('version', 1)}",
                    "title": f"场景: {env.get('name', '')}",
                })

    video = project.get("video")
    if video and video.get("path") and Path(video["path"]).exists():
        assets.append({
            "asset_type": "video",
            "project_id": pid,
            "asset_path": video["path"],
            "prompt": project.get("script", ""),
            "model_provider": video.get("provider", "unknown"),
            "model_version": video.get("version", "v1"),
            "title": f"最终视频: {project.get('title', '')}",
        })

    return assets


def create_pairwise_comparison(
    asset_a_path: str,
    asset_b_path: str,
    model_a: str,
    model_b: str,
    task_type: str = "keyframe",
    project_id: str | None = None,
    prompt: str | None = None,
    model_a_version: str | None = None,
    model_b_version: str | None = None,
) -> dict[str, Any]:
    pid = str(uuid4())
    ts = _now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO pairwise_comparisons
               (id, project_id, task_type, prompt, asset_a_path, asset_b_path,
                model_a, model_b, model_a_version, model_b_version,
                winner, reason_tags_json, reason_note, judge_type, judged_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 'human', NULL, ?, ?)""",
            (pid, project_id, task_type, prompt, asset_a_path, asset_b_path,
             model_a, model_b, model_a_version, model_b_version, ts, ts),
        )
        conn.commit()
    return get_pairwise(pid)


def create_pairwise_batch(
    assets: list[dict[str, Any]],
    task_type: str = "keyframe",
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    import random

    candidates = build_pairwise_candidates(assets)
    random.shuffle(candidates)
    with _connect() as conn:
        rows = conn.execute(
            """SELECT asset_a_path, asset_b_path FROM pairwise_comparisons
               WHERE project_id IS ? AND task_type = ?""",
            (project_id, task_type),
        ).fetchall()
    existing_pairs = {
        tuple(sorted((row["asset_a_path"], row["asset_b_path"])))
        for row in rows
    }

    created = []
    for a, b in candidates:
        key = tuple(sorted((a["asset_path"], b["asset_path"])))
        if key in existing_pairs:
            continue
        existing_pairs.add(key)
        swap = random.random() < 0.5
        left, right = (b, a) if swap else (a, b)
        rec = create_pairwise_comparison(
            asset_a_path=left["asset_path"],
            asset_b_path=right["asset_path"],
            model_a=left.get("model_provider", "unknown"),
            model_b=right.get("model_provider", "unknown"),
            model_a_version=left.get("model_version"),
            model_b_version=right.get("model_version"),
            task_type=task_type,
            project_id=project_id,
            prompt=left.get("prompt"),
        )
        created.append(rec)
    return created


def build_pairwise_candidates(
    assets: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Build fair comparisons from outputs for the same prompt and target."""
    from itertools import combinations

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for asset in assets:
        prompt = str(asset.get("prompt") or "").strip()
        if not prompt:
            continue
        target_id = str(
            asset.get("shot_id")
            or asset.get("asset_id")
            or asset.get("title")
            or ""
        )
        title = str(asset.get("title") or "") if asset.get("asset_id") else ""
        groups.setdefault((prompt, target_id, title), []).append(asset)

    candidates = []
    seen_paths: set[tuple[str, str]] = set()
    for grouped_assets in groups.values():
        for left, right in combinations(grouped_assets, 2):
            left_path = str(left.get("asset_path") or "")
            right_path = str(right.get("asset_path") or "")
            if not left_path or not right_path or left_path == right_path:
                continue
            if left.get("model_provider") == right.get("model_provider"):
                continue
            path_key = tuple(sorted((left_path, right_path)))
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            candidates.append((left, right))
    return candidates


def get_pairwise(pair_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM pairwise_comparisons WHERE id = ?", (pair_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("reason_tags_json"):
            try:
                d["reason_tags"] = json.loads(d["reason_tags_json"])
            except (json.JSONDecodeError, TypeError):
                d["reason_tags"] = []
        else:
            d["reason_tags"] = []
        return d


def get_next_pairwise(project_id: str | None = None, task_type: str | None = None) -> dict[str, Any] | None:
    with _connect() as conn:
        q = "SELECT * FROM pairwise_comparisons WHERE winner IS NULL"
        params: list[Any] = []
        if project_id:
            q += " AND project_id = ?"
            params.append(project_id)
        if task_type:
            q += " AND task_type = ?"
            params.append(task_type)
        q += " ORDER BY RANDOM() LIMIT 1"
        row = conn.execute(q, params).fetchone()
        if not row:
            return None
        d = dict(row)
        d["reason_tags"] = []
        return d


def vote_pairwise(
    pair_id: str,
    winner: str,
    reason_tags: list[str] | None = None,
    reason_note: str = "",
) -> dict[str, Any] | None:
    if winner not in ("A", "B", "tie", "both_unusable"):
        raise ValueError(
            "winner must be 'A', 'B', 'tie', or 'both_unusable', "
            f"got {winner!r}"
        )
    normalized_tags = [
        str(tag).strip() for tag in (reason_tags or []) if str(tag).strip()
    ]
    normalized_note = str(reason_note or "").strip()
    if not normalized_tags and not normalized_note:
        raise ValueError("A vote reason tag or note is required")
    ts = _now()
    with _connect() as conn:
        conn.execute(
            """UPDATE pairwise_comparisons
               SET winner = ?, reason_tags_json = ?, reason_note = ?, judged_at = ?, updated_at = ?
               WHERE id = ?""",
            (
                winner,
                json.dumps(normalized_tags, ensure_ascii=False),
                normalized_note,
                ts,
                ts,
                pair_id,
            ),
        )
        conn.commit()
    return get_pairwise(pair_id)


def list_pairwise_comparisons(
    project_id: str | None = None,
    task_type: str | None = None,
    judged_only: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    with _connect() as conn:
        q = "SELECT * FROM pairwise_comparisons WHERE 1=1"
        params: list[Any] = []
        if project_id:
            q += " AND project_id = ?"
            params.append(project_id)
        if task_type:
            q += " AND task_type = ?"
            params.append(task_type)
        if judged_only:
            q += " AND winner IS NOT NULL"
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(q, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("reason_tags_json"):
                try:
                    d["reason_tags"] = json.loads(d["reason_tags_json"])
                except (json.JSONDecodeError, TypeError):
                    d["reason_tags"] = []
            else:
                d["reason_tags"] = []
            results.append(d)
        return results


def get_pairwise_stats(project_id: str | None = None) -> dict[str, Any]:
    with _connect() as conn:
        q = "SELECT model_a, model_b, winner FROM pairwise_comparisons WHERE winner IS NOT NULL"
        params: list[Any] = []
        if project_id:
            q += " AND project_id = ?"
            params.append(project_id)
        rows = conn.execute(q, params).fetchall()

    model_wins: dict[str, int] = {}
    model_losses: dict[str, int] = {}
    model_ties: dict[str, int] = {}
    model_unusable: dict[str, int] = {}
    model_total: dict[str, int] = {}

    for r in rows:
        ma, mb, w = r["model_a"], r["model_b"], r["winner"]
        for m in (ma, mb):
            model_total[m] = model_total.get(m, 0) + 1
        if w == "A":
            model_wins[ma] = model_wins.get(ma, 0) + 1
            model_losses[mb] = model_losses.get(mb, 0) + 1
        elif w == "B":
            model_wins[mb] = model_wins.get(mb, 0) + 1
            model_losses[ma] = model_losses.get(ma, 0) + 1
        elif w == "tie":
            model_ties[ma] = model_ties.get(ma, 0) + 1
            model_ties[mb] = model_ties.get(mb, 0) + 1
        elif w == "both_unusable":
            model_unusable[ma] = model_unusable.get(ma, 0) + 1
            model_unusable[mb] = model_unusable.get(mb, 0) + 1

    elo = _compute_elo(rows)
    leaderboard = []
    for model, rating in sorted(elo.items(), key=lambda x: -x[1]):
        wins = model_wins.get(model, 0)
        losses = model_losses.get(model, 0)
        ties = model_ties.get(model, 0)
        unusable = model_unusable.get(model, 0)
        total = model_total.get(model, 0)
        win_rate = wins / total if total > 0 else 0.0
        leaderboard.append({
            "model": model,
            "elo": round(rating, 0),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "unusable": unusable,
            "total": total,
            "win_rate": round(win_rate, 3),
        })

    total_votes = len(rows)
    pending = 0
    with _connect() as conn:
        pq = "SELECT COUNT(*) as c FROM pairwise_comparisons WHERE winner IS NULL"
        pp: list[Any] = []
        if project_id:
            pq += " AND project_id = ?"
            pp.append(project_id)
        pending = conn.execute(pq, pp).fetchone()["c"]

    return {
        "total_votes": total_votes,
        "pending": pending,
        "leaderboard": leaderboard,
    }


def _auto_judged_count(run_id: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as c FROM evaluation_items WHERE run_id = ? AND auto_judged_at IS NOT NULL",
            (run_id,),
        ).fetchone()
        return row["c"] if row else 0


def _compute_elo(
    rows: list,
    k_factor: float = ELO_K_FACTOR,
    initial: float = ELO_INITIAL_RATING,
) -> dict[str, float]:
    ratings: dict[str, float] = {}

    def expected(ra: float, rb: float) -> float:
        import math
        return 1.0 / (1.0 + math.pow(10.0, (rb - ra) / 400.0))

    for r in rows:
        ma, mb, w = r["model_a"], r["model_b"], r["winner"]
        ra = ratings.get(ma, initial)
        rb = ratings.get(mb, initial)
        if w == "both_unusable":
            ratings.setdefault(ma, ra)
            ratings.setdefault(mb, rb)
            continue
        ea = expected(ra, rb)
        eb = 1.0 - ea
        if w == "A":
            sa, sb = 1.0, 0.0
        elif w == "B":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5
        ratings[ma] = ra + k_factor * (sa - ea)
        ratings[mb] = rb + k_factor * (sb - eb)

    for r in rows:
        for m in (r["model_a"], r["model_b"]):
            if m not in ratings:
                ratings[m] = initial
    return ratings


def save_auto_scores(
    item_id: str,
    dim_scores: dict[str, int],
    bad_case_tags: list[str],
    total_score: float,
    explanation: str,
    confidence: float,
    provider_name: str = "vlm",
    raw_response: str = "",
) -> bool:
    """Save VLM/auto-judge scores without overwriting human scores."""
    from app.providers.judge_base import JudgeResult, validate_judge_result

    ts = _now()
    normalized_provider = str(provider_name or "").strip()
    if not normalized_provider or normalized_provider == "human":
        raise ValueError("A non-human auto-judge provider name is required")
    judge_type = (
        "mock_auto" if normalized_provider == "mock" else normalized_provider
    )
    with _connect() as conn:
        item = conn.execute("SELECT * FROM evaluation_items WHERE id = ?", (item_id,)).fetchone()
        if not item:
            return False
        rubric_row = conn.execute(
            """SELECT rv.dimensions_json
               FROM evaluation_runs er
               JOIN rubric_versions rv ON rv.id = er.rubric_version_id
               WHERE er.id = ?""",
            (item["run_id"],),
        ).fetchone()
        if not rubric_row:
            raise ValueError("The evaluation item has no valid rubric")
        rubric = Rubric.from_json(rubric_row["dimensions_json"])
        validated = validate_judge_result(
            JudgeResult(
                dim_scores,
                total_score,
                bad_case_tags,
                explanation,
                confidence,
                raw_response,
            ),
            rubric.to_dict(),
        )

        conn.execute(
            "DELETE FROM dimension_scores WHERE item_id = ? AND judge_type != 'human'",
            (item_id,),
        )
        conn.execute(
            "DELETE FROM bad_case_tags WHERE item_id = ? AND judge_type != 'human'",
            (item_id,),
        )

        for dim_key, sv in validated.dimension_scores.items():
            sid = str(uuid4())
            conn.execute(
                "INSERT INTO dimension_scores (id, item_id, dimension_key, score, weighted_score, judge_type, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sid, item_id, dim_key, sv, None, judge_type, ts, ts),
            )

        for tag in validated.bad_case_tags:
            tid = str(uuid4())
            conn.execute(
                "INSERT INTO bad_case_tags (id, item_id, tag, note, judge_type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (tid, item_id, tag, None, judge_type, ts),
            )

        conn.execute(
            """UPDATE evaluation_items
               SET auto_total_score = ?, auto_judge_provider = ?, auto_confidence = ?,
                   auto_explanation = ?, auto_raw_response = ?, auto_judged_at = ?, updated_at = ?
               WHERE id = ?""",
            (
                validated.total_score,
                normalized_provider,
                validated.confidence,
                validated.explanation,
                validated.raw_response[:2000],
                ts,
                ts,
                item_id,
            ),
        )
        conn.commit()
    return True


def list_items_for_auto_judge(run_id: str, skip_judged: bool = True) -> list[dict[str, Any]]:
    """List items that need auto-judging."""
    with _connect() as conn:
        if skip_judged:
            rows = conn.execute(
                "SELECT * FROM evaluation_items WHERE run_id = ? AND auto_judged_at IS NULL ORDER BY order_index ASC",
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM evaluation_items WHERE run_id = ? ORDER BY order_index ASC",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]


def get_agreement_analysis(run_id: str) -> dict[str, Any]:
    """Compute human-machine agreement statistics for items that have both scores.

    Returns:
        - n_pairs: number of items with both human and auto scores
        - pearson_r: Pearson correlation between human and auto total scores
        - spearman_rho: Spearman rank correlation
        - mae: mean absolute error
        - rmse: root mean square error
        - exact_match_rate: rate of exact dimension-level match
        - within_one_rate: rate of scores within ±1 point
        - dimension_agreement: per-dimension agreement stats
        - confusion: confusion matrix-like analysis
    """
    return analyze_agreement(list_run_items(run_id))


def analyze_agreement(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute agreement metrics for an explicit, already-filtered item set."""
    pairs = []
    dim_pairs: dict[str, list[tuple[int, int]]] = {}

    for it in items:
        human_total = it.get("total_score")
        auto_total = it.get("auto_total_score")
        if human_total is not None and auto_total is not None:
            pairs.append((human_total, auto_total))
            human_scores = it.get("scores", {})
            auto_scores = it.get("auto_scores", {})
            for dk in set(list(human_scores.keys()) + list(auto_scores.keys())):
                hs = human_scores.get(dk, {}).get("score") if isinstance(human_scores.get(dk), dict) else None
                aus = auto_scores.get(dk, {}).get("score") if isinstance(auto_scores.get(dk), dict) else None
                if hs is not None and aus is not None:
                    try:
                        dim_pairs.setdefault(dk, []).append((int(hs), int(aus)))
                    except (TypeError, ValueError):
                        pass

    n = len(pairs)
    if n < 2:
        return {
            "n_pairs": n,
            "pearson_r": None,
            "spearman_rho": None,
            "mae": None,
            "rmse": None,
            "exact_match_rate": None,
            "within_one_rate": None,
            "dimension_agreement": {},
            "human_avg": None,
            "auto_avg": None,
            "note": "样本量不足（<2），无法计算一致性指标" if n < 2 else "尚无同时拥有人评和机评的样本",
        }

    human_scores_list = [p[0] for p in pairs]
    auto_scores_list = [p[1] for p in pairs]

    def pearson(x: list[float], y: list[float]) -> float:
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False))
        dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
        dy = sum((yi - my) ** 2 for yi in y) ** 0.5
        if dx == 0 or dy == 0:
            return 0.0
        return num / (dx * dy)

    def rankdata(a: list[float]) -> list[float]:
        sorted_pairs = sorted(enumerate(a), key=lambda x: x[1])
        ranks = [0.0] * len(a)
        i = 0
        while i < len(sorted_pairs):
            j = i
            while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[sorted_pairs[k][0]] = avg_rank
            i = j + 1
        return ranks

    def spearman(x: list[float], y: list[float]) -> float:
        rx = rankdata(x)
        ry = rankdata(y)
        return pearson(rx, ry)

    pr = pearson(human_scores_list, auto_scores_list)
    sr = spearman(human_scores_list, auto_scores_list)

    errors = [abs(h - a) for h, a in pairs]
    sq_errors = [(h - a) ** 2 for h, a in pairs]
    mae = sum(errors) / n
    rmse = (sum(sq_errors) / n) ** 0.5

    all_dim_pairs: list[tuple[int, int]] = []
    for dk, dlist in dim_pairs.items():
        all_dim_pairs.extend(dlist)

    exact = sum(1 for h, a in all_dim_pairs if h == a)
    within_one = sum(1 for h, a in all_dim_pairs if abs(h - a) <= 1)
    total_dim = len(all_dim_pairs)
    exact_rate = exact / total_dim if total_dim > 0 else 0.0
    within_one_rate = within_one / total_dim if total_dim > 0 else 0.0

    dim_agreement = {}
    for dk, dlist in dim_pairs.items():
        if len(dlist) < 2:
            dim_agreement[dk] = {"n": len(dlist), "pearson": None, "mae": None, "exact_rate": None}
            continue
        dh = [float(p[0]) for p in dlist]
        da = [float(p[1]) for p in dlist]
        dim_agreement[dk] = {
            "n": len(dlist),
            "pearson": round(pearson(dh, da), 3),
            "mae": round(sum(abs(h - a) for h, a in dlist) / len(dlist), 2),
            "exact_rate": round(sum(1 for h, a in dlist if h == a) / len(dlist), 2),
            "within_one_rate": round(sum(1 for h, a in dlist if abs(h - a) <= 1) / len(dlist), 2),
        }

    return {
        "n_pairs": n,
        "pearson_r": round(pr, 3),
        "spearman_rho": round(sr, 3),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "exact_match_rate": round(exact_rate, 3),
        "within_one_rate": round(within_one_rate, 3),
        "dimension_agreement": dim_agreement,
        "human_avg": round(sum(human_scores_list) / n, 2),
        "auto_avg": round(sum(auto_scores_list) / n, 2),
        "human_scores": human_scores_list,
        "auto_scores": auto_scores_list,
    }
