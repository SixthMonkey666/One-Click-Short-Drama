from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any

SENSITIVE_KEYWORDS = ("key", "secret", "token", "password", "apikey", "api_key", "credential", "auth")


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items() if not _is_sensitive_key(k)}
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, str) and len(value) > 5000:
        return value[:5000] + "...(truncated)"
    return value


def _is_sensitive_key(key: str) -> bool:
    k = key.lower().replace("_", "").replace("-", "")
    return any(kw in k for kw in SENSITIVE_KEYWORDS)


def _serialize_item(item: dict[str, Any], rubric_dimensions: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "item_id": item.get("id", ""),
        "project_id": item.get("project_id", ""),
        "asset_type": item.get("asset_type", ""),
        "shot_id": item.get("shot_id", ""),
        "asset_path": item.get("asset_path", ""),
        "prompt": item.get("prompt", ""),
        "model_provider": item.get("model_provider", ""),
        "model_version": item.get("model_version", ""),
        "status": item.get("status", ""),
        "total_score": item.get("total_score", ""),
        "auto_total_score": item.get("auto_total_score", ""),
        "auto_judge_provider": item.get("auto_judge_provider", ""),
        "auto_confidence": item.get("auto_confidence", ""),
        "auto_explanation": item.get("auto_explanation", ""),
        "note": item.get("note", ""),
        "created_at": item.get("created_at", ""),
        "submitted_at": item.get("submitted_at", ""),
    }
    scores = item.get("scores") or {}
    for dim in rubric_dimensions:
        sdata = scores.get(dim)
        if isinstance(sdata, dict):
            row[f"score_{dim}"] = sdata.get("score", "")
        else:
            row[f"score_{dim}"] = sdata if sdata is not None else ""
    tags = item.get("bad_case_tags") or []
    tag_names = []
    for t in tags:
        if isinstance(t, dict):
            tag_names.append(t.get("tag", ""))
        else:
            tag_names.append(str(t))
    row["bad_case_tags"] = "|".join(tag_names)
    auto_scores = item.get("auto_scores") or {}
    for dim in rubric_dimensions:
        score_data = auto_scores.get(dim)
        if isinstance(score_data, dict):
            row[f"auto_score_{dim}"] = score_data.get("score", "")
        else:
            row[f"auto_score_{dim}"] = score_data if score_data is not None else ""
    auto_tags = item.get("auto_bad_case_tags") or []
    row["auto_bad_case_tags"] = "|".join(
        tag.get("tag", "") if isinstance(tag, dict) else str(tag)
        for tag in auto_tags
    )
    return _sanitize_value(row)


def export_run_to_json(run: dict[str, Any], items: list[dict[str, Any]]) -> str:
    rubric = run.get("rubric")
    dim_keys = [d.key for d in rubric.dimensions] if rubric else []
    payload = {
        "exported_at": datetime.now().isoformat(),
        "run": {
            "id": run.get("id"),
            "name": run.get("name"),
            "task_type": run.get("task_type"),
            "judge_type": run.get("judge_type"),
            "judge_provider": run.get("judge_provider"),
            "status": run.get("status"),
            "created_at": run.get("created_at"),
            "source_run_ids": run.get("source_run_ids", []),
        },
        "rubric": rubric.to_dict() if rubric else None,
        "items": [_serialize_item(it, dim_keys) for it in items],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_run_to_csv(run: dict[str, Any], items: list[dict[str, Any]]) -> str:
    rubric = run.get("rubric")
    dim_keys = [d.key for d in rubric.dimensions] if rubric else []
    dim_labels = {d.key: d.label for d in rubric.dimensions} if rubric else {}

    fieldnames = [
        "item_id", "project_id", "asset_type", "shot_id", "asset_path",
        "prompt", "model_provider", "model_version", "status", "total_score",
        "auto_total_score", "auto_judge_provider", "auto_confidence",
        "auto_explanation",
    ]
    for dk in dim_keys:
        fieldnames.append(f"score_{dim_labels.get(dk, dk)}")
    for dk in dim_keys:
        fieldnames.append(f"auto_score_{dim_labels.get(dk, dk)}")
    fieldnames.extend([
        "bad_case_tags", "auto_bad_case_tags", "note", "created_at",
        "submitted_at",
    ])

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    label_map = {
        **{f"score_{dk}": f"score_{dim_labels.get(dk, dk)}" for dk in dim_keys},
        **{
            f"auto_score_{dk}": f"auto_score_{dim_labels.get(dk, dk)}"
            for dk in dim_keys
        },
    }
    for it in items:
        row = _serialize_item(it, dim_keys)
        renamed_row = {}
        for k, v in row.items():
            new_key = label_map.get(k, k)
            renamed_row[new_key] = v
        writer.writerow(renamed_row)

    return output.getvalue()


def export_runs_summary_json(runs: list[dict[str, Any]], run_stats: dict[str, dict[str, Any]]) -> str:
    payload = {
        "exported_at": datetime.now().isoformat(),
        "runs": [],
    }
    for run in runs:
        stats = run_stats.get(run.get("id"), {})
        payload["runs"].append({
            "id": run.get("id"),
            "name": run.get("name"),
            "task_type": run.get("task_type"),
            "judge_type": run.get("judge_type"),
            "judge_provider": run.get("judge_provider"),
            "status": run.get("status"),
            "created_at": run.get("created_at"),
            "stats": _sanitize_value(stats),
        })
    return json.dumps(payload, ensure_ascii=False, indent=2)
