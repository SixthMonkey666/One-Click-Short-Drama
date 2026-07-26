from __future__ import annotations

import json
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any

from langgraph.config import get_stream_writer
from pydantic import ValidationError

from app import adapters
from app.agent.schemas import ExecutionPlan, PlannedStep, ProviderSelection
from app.config import (
    AGENT_MAX_STORYBOARD_RETRIES,
    AGENT_STORYBOARD_PASS_SCORE,
    IMAGE_PROVIDER,
    JUDGE_PROVIDER,
    LLM_PROVIDER,
    VIDEO_PROVIDER,
)
from app.repository import get_project, save_project


def _drain(generator: Generator) -> tuple[Any, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    writer = _current_writer()
    while True:
        try:
            event = next(generator)
            if isinstance(event, dict):
                events.append(event)
                _write_pipeline_event(writer, event)
        except StopIteration as stop:
            return stop.value, events


def _current_writer():
    try:
        return get_stream_writer()
    except Exception:
        return None


def _write_pipeline_event(writer, event: dict[str, Any]) -> None:
    if writer is not None:
        writer({
            "type": "pipeline_event",
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
        })


def analyze_requirement(project_id: str) -> dict[str, Any]:
    project = _require_project(project_id)
    if project.get("analysis"):
        return project["analysis"]
    analysis, _ = _drain(adapters.generate_analysis_stream(project["source_text"]))
    project["analysis"] = analysis
    project["characters"] = analysis.get("characters", [])
    project["title"] = analysis.get("title", project["title"])
    project["status"] = "agent_planning"
    save_project(project)
    return analysis


def build_execution_plan(project_id: str, analysis: dict[str, Any]) -> ExecutionPlan:
    project = _require_project(project_id)
    steps = [
        PlannedStep(name="requirement_analysis", reason="理解用户目标和约束"),
        PlannedStep(name="visual_bible", reason="固化角色、道具、场景和视觉风格"),
        PlannedStep(name="storyboard_generation", reason="生成可执行分镜", requires_human_review=True),
    ]
    if analysis.get("characters"):
        steps.append(PlannedStep(name="character_generation", reason="故事包含角色"))
    if analysis.get("props"):
        steps.append(PlannedStep(name="prop_generation", reason="故事包含关键道具"))
    if analysis.get("environments"):
        steps.append(PlannedStep(name="environment_generation", reason="建立场景一致性"))
    steps.extend([
        PlannedStep(name="keyframe_generation", reason="为每个镜头生成关键帧"),
        PlannedStep(name="video_prompt_generation", reason="生成镜头运动提示"),
        PlannedStep(name="video_render", reason="合成最终视频", requires_human_review=True),
        PlannedStep(name="final_evaluation", reason="验证最终交付物"),
    ])
    providers = ProviderSelection(
        llm=LLM_PROVIDER, image=IMAGE_PROVIDER, video=VIDEO_PROVIDER, judge=JUDGE_PROVIDER,
    )
    try:
        plan = ExecutionPlan(
            video_type=str(analysis.get("genre") or "叙事短片"),
            duration_seconds=int(analysis.get("total_duration_seconds") or 20),
            steps=steps,
            providers=providers,
            evaluation_threshold=AGENT_STORYBOARD_PASS_SCORE,
            max_storyboard_retries=AGENT_MAX_STORYBOARD_RETRIES,
        )
    except (TypeError, ValueError, ValidationError):
        plan = ExecutionPlan(
            duration_seconds=20,
            steps=[
                PlannedStep(name="requirement_analysis", reason="安全回退：重新确认需求"),
                PlannedStep(name="visual_bible", reason="安全回退：固化视觉设定"),
                PlannedStep(
                    name="storyboard_generation",
                    reason="安全回退：生成基础分镜",
                    requires_human_review=True,
                ),
                PlannedStep(name="keyframe_generation", reason="安全回退：生成关键帧"),
                PlannedStep(name="video_prompt_generation", reason="安全回退：生成运动提示"),
                PlannedStep(
                    name="video_render",
                    reason="安全回退：人工确认后渲染",
                    requires_human_review=True,
                ),
                PlannedStep(name="final_evaluation", reason="安全回退：验证交付"),
            ],
            providers=providers,
            evaluation_threshold=AGENT_STORYBOARD_PASS_SCORE,
            max_storyboard_retries=AGENT_MAX_STORYBOARD_RETRIES,
        )
    project["agent_plan"] = plan.model_dump(mode="json")
    save_project(project)
    return plan


def generate_storyboard(
    project_id: str,
    analysis: dict[str, Any],
    feedback: dict[str, Any] | None,
    attempt: int,
) -> tuple[str, list[dict[str, Any]]]:
    project = _require_project(project_id)
    operation_key = f"storyboard:{attempt}"
    completed = project.setdefault("agent_operations", {}).get(operation_key)
    if completed and project.get("shots"):
        return project.get("script", ""), project["shots"]

    source = project["source_text"]
    if feedback:
        source += "\n\n上一轮分镜评测修复要求（只修复这些问题，保持其他设定稳定）：\n"
        source += json.dumps(feedback, ensure_ascii=False)
    result, _ = _drain(adapters.generate_storyboard_stream(source, analysis))
    script, shots = result
    project["script"] = script
    project["shots"] = shots
    project["status"] = "agent_storyboard_evaluation"
    project["agent_operations"][operation_key] = {"completed": True}
    save_project(project)
    return script, shots


def generate_missing_assets(project_id: str) -> dict[str, Any]:
    project = _require_project(project_id)
    analysis = project.get("analysis") or {}
    writer = _current_writer()
    character_jobs = sum(
        4
        for character in analysis.get("characters", [])
        if not {"front", "side", "back", "face"}.issubset(
            {image.get("view") for image in character.get("reference_images", [])}
        )
    )
    prop_jobs = sum(not item.get("reference_images") for item in analysis.get("props", []))
    environment_jobs = sum(
        not item.get("reference_images") for item in analysis.get("environments", [])
    )
    keyframe_jobs = sum(not shot.get("selected_image") for shot in project.get("shots", []))
    total_jobs = character_jobs + prop_jobs + environment_jobs + keyframe_jobs
    completed_jobs = 0
    _write_pipeline_event(writer, {
        "type": "assets_start",
        "total": total_jobs,
        "characters": character_jobs,
        "props": prop_jobs,
        "environments": environment_jobs,
        "keyframes": keyframe_jobs,
    })

    def mark_completed(count: int, message: str) -> None:
        nonlocal completed_jobs
        completed_jobs += count
        _write_pipeline_event(writer, {
            "type": "asset_progress_overall",
            "done": completed_jobs,
            "total": total_jobs,
            "message": message,
        })

    for character in analysis.get("characters", []):
        views = {image.get("view") for image in character.get("reference_images", [])}
        if not {"front", "side", "back", "face"}.issubset(views):
            images, _ = _drain(adapters.generate_character_images_stream(project_id, character))
            character["reference_images"] = images
            save_project(project)
            mark_completed(len(images), f"角色「{character.get('name', '未命名')}」参考图完成")
    for prop in analysis.get("props", []):
        prop.setdefault("reference_images", [])
        if not prop["reference_images"]:
            _drain(adapters.generate_prop_image_stream(project_id, prop))
            save_project(project)
            mark_completed(1, f"道具「{prop.get('name', '未命名')}」参考图完成")
    for environment in analysis.get("environments", []):
        environment.setdefault("reference_images", [])
        if not environment["reference_images"]:
            _drain(adapters.generate_environment_image_stream(project_id, environment))
            save_project(project)
            mark_completed(1, f"场景「{environment.get('name', '未命名')}」参考图完成")
    for shot in project.get("shots", []):
        shot.setdefault("image_versions", [])
        if not shot.get("selected_image"):
            _drain(adapters.generate_keyframe_image_stream(project_id, shot, analysis))
            save_project(project)
            mark_completed(1, f"镜头 {shot.get('number', '?')} 关键帧完成")
    _write_pipeline_event(writer, {
        "type": "assets_complete",
        "done": completed_jobs,
        "total": total_jobs,
    })
    project["status"] = "agent_asset_evaluation"
    save_project(project)
    return summarize_assets(project)


def repair_failed_assets(project_id: str, failed_items: list[dict[str, Any]]) -> dict[str, Any]:
    project = _require_project(project_id)
    analysis = project.get("analysis") or {}
    for failed in failed_items:
        if failed.get("type") != "keyframe":
            continue
        shot = next((item for item in project.get("shots", []) if item.get("id") == failed.get("id")), None)
        if shot:
            _drain(adapters.generate_keyframe_image_stream(project_id, shot, analysis))
            save_project(project)
    return summarize_assets(project)


def evaluate_assets(project_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    project = _require_project(project_id)
    failed: list[dict[str, Any]] = []
    for shot in project.get("shots", []):
        selected = shot.get("selected_image") or {}
        path = selected.get("path")
        if not path:
            failed.append({"type": "keyframe", "id": shot.get("id"), "reason": "缺少关键帧"})
    result = {
        "score": 5.0 if not failed else max(1.0, 5.0 - len(failed)),
        "passed": not failed,
        "issues": [item["reason"] for item in failed],
        "recommended_action": "continue" if not failed else "repair",
    }
    return result, failed


def generate_video_prompts(project_id: str, force: bool = False) -> list[dict[str, Any]]:
    project = _require_project(project_id)
    if not force and project.get("shots") and all(shot.get("video_prompt") for shot in project["shots"]):
        return [shot["video_prompt"] for shot in project["shots"]]
    prompts, _ = _drain(adapters.generate_video_prompts_stream(
        project_id, project["source_text"], project.get("script", ""),
        project.get("shots", []), project.get("characters"),
    ))
    by_number = {prompt.get("shot_number"): prompt for prompt in prompts}
    for shot in project.get("shots", []):
        shot["video_prompt"] = by_number.get(shot.get("number"))
    project["status"] = "awaiting_video_prompt_review"
    save_project(project)
    return prompts


def render_video(project_id: str) -> dict[str, Any]:
    project = _require_project(project_id)
    if (project.get("video") or {}).get("status") == "completed":
        return project["video"]
    result = adapters.generate_final_video(project_id, project.get("shots", []))
    project["video"] = result
    project["status"] = "completed"
    save_project(project)
    return result


def summarize_assets(project: dict[str, Any]) -> dict[str, Any]:
    analysis = project.get("analysis") or {}
    return {
        "characters": [
            {"id": item.get("id"), "paths": [image.get("path") for image in item.get("reference_images", [])]}
            for item in analysis.get("characters", [])
        ],
        "props": [
            {"id": item.get("id"), "paths": [image.get("path") for image in item.get("reference_images", [])]}
            for item in analysis.get("props", [])
        ],
        "environments": [
            {"id": item.get("id"), "paths": [image.get("path") for image in item.get("reference_images", [])]}
            for item in analysis.get("environments", [])
        ],
        "keyframes": [
            {"id": shot.get("id"), "path": (shot.get("selected_image") or {}).get("path")}
            for shot in project.get("shots", [])
        ],
    }


def _require_project(project_id: str) -> dict[str, Any]:
    project = get_project(project_id)
    if not project:
        raise ValueError(f"项目不存在：{project_id}")
    return project
