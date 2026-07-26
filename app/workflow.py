"""Workflow orchestration for the storyboard-to-video pipeline.

New state machine:
  new → analyzing_idea → awaiting_analysis_review →
  generating_storyboard → awaiting_storyboard_review →
  generating_assets (chars → props → envs → keyframes) →
  generating_video_prompts → awaiting_video_prompt_review →
  rendering_video → completed
"""

from __future__ import annotations

from typing import Any, Generator

from app.adapters import (
    generate_analysis_stream,
    generate_character_images_stream,
    generate_environment_image_stream,
    generate_final_video,
    generate_keyframe_image_stream,
    generate_prop_image_stream,
    generate_storyboard_stream,
)
from app.adapters import (
    generate_video_prompts_stream as _adapter_video_prompts_stream,
)
from app.config import CHARACTER_VIEWS
from app.providers import get_image_provider, get_llm_provider, release_all
from app.repository import get_project, save_project


def _release_models() -> None:
    try:
        get_llm_provider().release()
    except Exception:
        pass
    try:
        get_image_provider().release()
    except Exception:
        pass
    release_all()


def analyze_idea_stream(project_id: str) -> Generator[dict[str, Any], None, None]:
    project = get_project(project_id)
    if not project:
        yield {"type": "error", "message": "项目不存在"}
        return

    project["status"] = "analyzing_idea"
    save_project(project)
    yield {"type": "status", "message": "正在分析创意..."}

    analysis = None
    gen = generate_analysis_stream(project["source_text"])
    while True:
        try:
            event = next(gen)
            yield event
        except StopIteration as e:
            analysis = e.value
            break

    project["analysis"] = analysis
    project["title"] = analysis.get("title", project["title"])
    project["characters"] = analysis.get("characters", [])
    project["status"] = "awaiting_analysis_review"
    save_project(project)
    _release_models()
    yield {"type": "analysis_complete", "analysis": analysis}


def approve_analysis(project_id: str, analysis: dict[str, Any]) -> dict[str, Any]:
    project = get_project(project_id)
    if not project:
        return {"error": "项目不存在"}
    project["analysis"] = analysis
    project["title"] = analysis.get("title", project["title"])
    project["characters"] = analysis.get("characters", [])
    project["status"] = "generating_storyboard"
    save_project(project)
    return project


def generate_storyboard_from_analysis_stream(project_id: str) -> Generator[dict[str, Any], None, None]:
    project = get_project(project_id)
    if not project:
        yield {"type": "error", "message": "项目不存在"}
        return
    analysis = project.get("analysis")
    if not analysis:
        yield {"type": "error", "message": "缺少创意分析"}
        return

    project["status"] = "generating_storyboard"
    save_project(project)
    yield {"type": "status", "message": "正在生成分镜表..."}

    script = ""
    shots = []
    gen = generate_storyboard_stream(project["source_text"], analysis)
    while True:
        try:
            event = next(gen)
            yield event
        except StopIteration as e:
            script, shots = e.value
            break

    project["script"] = script
    project["shots"] = shots
    project["status"] = "awaiting_storyboard_review"
    save_project(project)
    _release_models()
    yield {"type": "storyboard_complete", "script": script, "shots": shots}


def approve_storyboard(project_id: str, shots: list[dict[str, Any]]) -> dict[str, Any]:
    project = get_project(project_id)
    if not project:
        return {"error": "项目不存在"}
    project["shots"] = shots
    project["status"] = "generating_assets"
    save_project(project)
    return project


def generate_assets_stream(project_id: str) -> Generator[dict[str, Any], None, None]:
    """Generate all assets in order: characters → props → environments → keyframes."""
    project = get_project(project_id)
    if not project:
        yield {"type": "error", "message": "项目不存在"}
        return
    analysis = project.get("analysis") or {}
    characters = analysis.get("characters", []) or project.get("characters", [])
    props = analysis.get("props", [])
    environments = analysis.get("environments", [])
    shots = project.get("shots", [])

    project["status"] = "generating_assets"
    save_project(project)

    total_assets = len(characters) * len(CHARACTER_VIEWS) + len(props) + len(environments) + len(shots)
    done_assets = 0

    def _progress(msg: str) -> dict[str, Any]:
        nonlocal done_assets
        done_assets += 1
        return {"type": "asset_progress_overall", "done": done_assets, "total": total_assets, "message": msg}

    yield {"type": "assets_start", "total": total_assets,
           "characters": len(characters), "props": len(props),
           "environments": len(environments), "keyframes": len(shots)}

    for char in characters:
        yield {"type": "phase_start", "phase": "characters", "name": char["name"]}
        char_gen = generate_character_images_stream(project_id, char)
        while True:
            try:
                event = next(char_gen)
                yield event
                if event.get("type") == "asset_done":
                    image = event.get("image", {})
                    view_label = image.get("view_label", image.get("view", "参考图"))
                    yield _progress(f"角色「{char['name']}」{view_label}完成")
            except StopIteration as e:
                images = e.value
                char["reference_images"] = images
                break
        save_project(project)
        _release_models()

    for prop in props:
        yield {"type": "phase_start", "phase": "props", "name": prop["name"]}
        prop_gen = generate_prop_image_stream(project_id, prop)
        while True:
            try:
                event = next(prop_gen)
                yield event
            except StopIteration:
                break
        save_project(project)
        yield _progress(f"道具「{prop['name']}」参考图完成")
        _release_models()

    for env in environments:
        yield {"type": "phase_start", "phase": "environments", "name": env["name"]}
        env_gen = generate_environment_image_stream(project_id, env)
        while True:
            try:
                event = next(env_gen)
                yield event
            except StopIteration:
                break
        save_project(project)
        yield _progress(f"场景「{env['name']}」参考图完成")
        _release_models()

    for shot in shots:
        yield {"type": "phase_start", "phase": "keyframes", "name": shot.get("title", f"镜头{shot.get('number')}"),
               "shot_number": shot.get("number")}
        kf_gen = generate_keyframe_image_stream(project_id, shot, analysis)
        while True:
            try:
                event = next(kf_gen)
                yield event
            except StopIteration:
                break
        save_project(project)
        yield _progress(f"镜头{shot.get('number')}关键帧完成")
        _release_models()

    project["status"] = "awaiting_video_prompt_review"
    save_project(project)
    _release_models()
    yield {"type": "assets_complete"}


def render_video_stream(project_id: str) -> Generator[dict[str, Any], None, None]:
    """One-click video generation: generate prompts then render."""
    project = get_project(project_id)
    if not project:
        yield {"type": "error", "message": "项目不存在"}
        return

    has_prompts = all(s.get("video_prompt") for s in project.get("shots", []))
    if not has_prompts:
        project["status"] = "generating_video_prompts"
        save_project(project)
        yield {"type": "status", "message": "正在生成视频运动提示词..."}
        gen = _adapter_video_prompts_stream(
            project_id, project["source_text"], project.get("script", ""),
            project.get("shots", []), project.get("characters"),
        )
        prompts: list[dict[str, Any]] = []
        while True:
            try:
                event = next(gen)
                yield event
            except StopIteration as e:
                prompts = e.value
                break
        for p in prompts:
            sn = p.get("shot_number")
            for shot in project["shots"]:
                if shot.get("number") == sn:
                    shot["video_prompt"] = p
                    break

    project["status"] = "rendering_video"
    save_project(project)
    yield {"type": "status", "message": "正在合成视频..."}
    try:
        result = generate_final_video(project_id, project["shots"])
        project["video"] = result
        project["status"] = "completed"
        yield {"type": "video_complete", "result": result}
    except Exception as e:
        project["video"] = {"provider": "error", "status": "failed", "error": str(e)}
        project["status"] = "awaiting_video_prompt_review"
        yield {"type": "error", "message": str(e)}
    save_project(project)
    _release_models()


def generate_video_prompts_stream(project_id: str) -> Generator[dict[str, Any], None, None]:
    project = get_project(project_id)
    if not project:
        yield {"type": "error", "message": "项目不存在"}
        return

    project["status"] = "generating_video_prompts"
    save_project(project)
    yield {"type": "status", "message": "正在生成视频运动提示词..."}

    gen = _adapter_video_prompts_stream(
        project_id, project["source_text"], project.get("script", ""),
        project.get("shots", []), project.get("characters"),
    )
    prompts: list[dict[str, Any]] = []
    while True:
        try:
            event = next(gen)
            yield event
        except StopIteration as e:
            prompts = e.value
            break

    for p in prompts:
        sn = p.get("shot_number")
        for shot in project["shots"]:
            if shot.get("number") == sn:
                shot["video_prompt"] = p
                break

    project["status"] = "awaiting_video_prompt_review"
    save_project(project)
    _release_models()
    yield {"type": "video_prompts_complete", "prompts": prompts}


def approve_video_prompts_and_render(project_id: str) -> dict[str, Any]:
    project = get_project(project_id)
    if not project:
        return {"error": "项目不存在"}
    project["status"] = "rendering_video"
    save_project(project)
    try:
        result = generate_final_video(project_id, project["shots"])
        project["video"] = result
        project["status"] = "completed"
    except Exception as e:
        project["video"] = {"provider": "error", "status": "failed", "error": str(e)}
        project["status"] = "awaiting_video_prompt_review"
    save_project(project)
    _release_models()
    return project


def regenerate_keyframe_stream(project_id: str, shot_id: str) -> Generator[dict[str, Any], None, None]:
    project = get_project(project_id)
    if not project:
        yield {"type": "error", "message": "项目不存在"}
        return
    shot = next((s for s in project["shots"] if s["id"] == shot_id), None)
    if not shot:
        yield {"type": "error", "message": "镜头不存在"}
        return
    analysis = project.get("analysis")
    gen = generate_keyframe_image_stream(project_id, shot, analysis)
    while True:
        try:
            event = next(gen)
            yield event
        except StopIteration:
            break
    save_project(project)
    _release_models()


def regenerate_character_images_stream(
    project_id: str, character_id: str
) -> Generator[dict[str, Any], None, None]:
    project = get_project(project_id)
    if not project:
        yield {"type": "error", "message": "项目不存在"}
        return
    analysis = project.get("analysis") or {}
    chars = analysis.get("characters", []) or project.get("characters", [])
    char = next((c for c in chars if c["id"] == character_id), None)
    if not char:
        yield {"type": "error", "message": "角色不存在"}
        return
    # Keep the previous canonical front view available while regenerating so
    # FLUX can preserve identity instead of inventing a different character.
    gen = generate_character_images_stream(project_id, char)
    while True:
        try:
            event = next(gen)
            yield event
        except StopIteration as e:
            char["reference_images"] = e.value
            break
    save_project(project)
    _release_models()
