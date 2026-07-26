from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Callable

from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from app.agent import tools
from app.agent.policies import build_reflection, evaluate_storyboard
from app.agent.schemas import HumanReviewDecision
from app.agent.state import VideoAgentState
from app.observability import trace_context
from app.repository import get_project, save_project


def _execute(name: str, state: VideoAgentState, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    project_id = state["project_id"]
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None
    if writer is not None:
        writer({
            "type": "node_started",
            "node": name,
            "timestamp": started_at,
        })
    try:
        with trace_context(f"agent.{name}", project_id):
            update = operation()
        log = {
            "project_id": project_id, "thread_id": state["thread_id"], "node": name,
            "started_at": started_at,
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
            "status": "completed",
            "providers": dict(state.get("selected_providers", {})),
            "input_summary": {
                "previous_stage": state.get("current_stage"),
                "retry_counts": dict(state.get("retry_counts", {})),
                "failed_item_count": len(state.get("failed_items", [])),
            },
            "output_summary": sorted(update),
            "evaluation_score": _latest_evaluation_score(update),
            "retry_reason": (state.get("feedback") or {}).get("summary"),
            "next_route": "resolved_by_graph_edge",
        }
        update.setdefault("current_stage", name)
        update.setdefault("completed_nodes", [name])
        update.setdefault("execution_logs", [log])
        if writer is not None:
            writer({
                "type": "node_completed",
                "node": name,
                "duration_seconds": round(time.monotonic() - started, 1),
                "timestamp": datetime.now(UTC).isoformat(),
            })
        return update
    except Exception as exc:
        if writer is not None:
            writer({
                "type": "node_failed",
                "node": name,
                "error": str(exc),
                "duration_seconds": round(time.monotonic() - started, 1),
                "timestamp": datetime.now(UTC).isoformat(),
            })
        return {
            "current_stage": "error",
            "errors": [{"node": name, "type": type(exc).__name__, "message": str(exc)}],
            "execution_logs": [{
                "project_id": project_id, "thread_id": state["thread_id"], "node": name,
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
                "status": "failed", "error_type": type(exc).__name__,
            }],
        }


def requirement_analysis_node(state: VideoAgentState) -> dict[str, Any]:
    return _execute("requirement_analysis", state, lambda: {
        "analysis": tools.analyze_requirement(state["project_id"]),
    })


def planner_node(state: VideoAgentState) -> dict[str, Any]:
    def operation():
        plan = tools.build_execution_plan(state["project_id"], state["analysis"])
        return {
            "execution_plan": plan.model_dump(mode="json"),
            "constraints": {
                "duration_seconds": plan.duration_seconds,
                "aspect_ratio": plan.aspect_ratio,
                "evaluation_threshold": plan.evaluation_threshold,
            },
            "selected_providers": plan.providers.model_dump(mode="json"),
        }
    return _execute("planner", state, operation)


def visual_bible_node(state: VideoAgentState) -> dict[str, Any]:
    analysis = state["analysis"]
    return _execute("visual_bible", state, lambda: {"visual_bible": {
        "visual_style": analysis.get("visual_style"),
        "characters": analysis.get("characters", []),
        "props": analysis.get("props", []),
        "environments": analysis.get("environments", []),
    }})


def storyboard_generation_node(state: VideoAgentState) -> dict[str, Any]:
    def operation():
        attempt = state.get("retry_counts", {}).get("storyboard", 0)
        script, shots = tools.generate_storyboard(
            state["project_id"], state["analysis"], state.get("feedback"), attempt,
        )
        return {"storyboard": shots, "assets": {"script": script}}
    return _execute("storyboard_generation", state, operation)


def storyboard_evaluation_node(state: VideoAgentState) -> dict[str, Any]:
    def operation():
        result = evaluate_storyboard(state.get("storyboard", []), state["analysis"], state.get("constraints"))
        evaluations = dict(state.get("evaluation_results", {}))
        evaluations["storyboard"] = result.model_dump(mode="json")
        return {"evaluation_results": evaluations}
    return _execute("storyboard_evaluation", state, operation)


def storyboard_reflection_node(state: VideoAgentState) -> dict[str, Any]:
    def operation():
        evaluation = state.get("evaluation_results", {}).get("storyboard", {})
        reflection = build_reflection(evaluation)
        retries = dict(state.get("retry_counts", {}))
        retries["storyboard"] = retries.get("storyboard", 0) + 1
        return {"feedback": reflection.model_dump(mode="json"), "retry_counts": retries}
    return _execute("storyboard_reflection", state, operation)


def storyboard_human_review_node(state: VideoAgentState) -> dict[str, Any]:
    project = get_project(state["project_id"])
    if project and project.get("status") != "awaiting_storyboard_review":
        project["status"] = "awaiting_storyboard_review"
        save_project(project)
    evaluation = state.get("evaluation_results", {}).get("storyboard", {})
    payload = {
        "review_type": "storyboard", "project_id": state["project_id"],
        "content": state.get("storyboard", []), "evaluation": evaluation,
        "feedback": state.get("feedback", {}),
        "allowed_actions": ["approve", "edit_and_continue", "regenerate", "terminate"],
    }
    raw_decision = interrupt(payload)
    decision = HumanReviewDecision.model_validate(raw_decision)
    shots = decision.edited_storyboard or state.get("storyboard", [])
    if decision.action == "edit_and_continue":
        project = get_project(state["project_id"])
        if project:
            project["shots"] = shots
            project["status"] = "generating_assets"
            save_project(project)
    return {
        "storyboard": shots,
        "human_decision": decision.model_dump(mode="json"),
        "human_review_type": "storyboard",
        "waiting_for_human": False,
        "current_stage": "storyboard_human_review",
        "completed_nodes": ["storyboard_human_review"],
        "execution_logs": [_human_log("storyboard_human_review", state, decision.action)],
    }


def asset_planning_node(state: VideoAgentState) -> dict[str, Any]:
    return _execute("asset_planning", state, lambda: {"failed_items": []})


def asset_generation_node(state: VideoAgentState) -> dict[str, Any]:
    return _execute("asset_generation", state, lambda: {
        "assets": tools.generate_missing_assets(state["project_id"]),
    })


def asset_evaluation_node(state: VideoAgentState) -> dict[str, Any]:
    def operation():
        result, failed = tools.evaluate_assets(state["project_id"])
        evaluations = dict(state.get("evaluation_results", {}))
        evaluations["assets"] = result
        return {"evaluation_results": evaluations, "failed_items": failed}
    return _execute("asset_evaluation", state, operation)


def asset_repair_node(state: VideoAgentState) -> dict[str, Any]:
    def operation():
        retries = dict(state.get("retry_counts", {}))
        retries["assets"] = retries.get("assets", 0) + 1
        return {
            "assets": tools.repair_failed_assets(state["project_id"], state.get("failed_items", [])),
            "retry_counts": retries,
        }
    return _execute("asset_repair", state, operation)


def video_prompt_generation_node(state: VideoAgentState) -> dict[str, Any]:
    def operation():
        retries = dict(state.get("retry_counts", {}))
        previous = state.get("evaluation_results", {}).get("video_prompts")
        force = bool(previous and not previous.get("passed"))
        if force:
            retries["video_prompts"] = retries.get("video_prompts", 0) + 1
        prompts = tools.generate_video_prompts(state["project_id"], force=force)
        return {"video_prompts": prompts, "retry_counts": retries}
    return _execute("video_prompt_generation", state, operation)


def asset_human_review_node(state: VideoAgentState) -> dict[str, Any]:
    raw_decision = interrupt({
        "review_type": "assets", "project_id": state["project_id"],
        "content": state.get("failed_items", []),
        "evaluation": state.get("evaluation_results", {}).get("assets", {}),
        "allowed_actions": ["approve", "regenerate", "terminate"],
    })
    decision = HumanReviewDecision.model_validate(raw_decision)
    return {
        "human_decision": decision.model_dump(mode="json"),
        "human_review_type": "assets", "waiting_for_human": False,
        "current_stage": "asset_human_review", "completed_nodes": ["asset_human_review"],
        "execution_logs": [_human_log("asset_human_review", state, decision.action)],
    }


def video_prompt_evaluation_node(state: VideoAgentState) -> dict[str, Any]:
    def operation():
        project = get_project(state["project_id"]) or {}
        shots = project.get("shots", [])
        missing = [shot.get("id") for shot in shots if not shot.get("video_prompt")]
        evaluations = dict(state.get("evaluation_results", {}))
        evaluations["video_prompts"] = {
            "score": 5.0 if not missing else 2.0, "passed": not missing,
            "issues": [f"缺少视频提示词：{item}" for item in missing],
            "recommended_action": "continue" if not missing else "repair",
        }
        return {"evaluation_results": evaluations}
    return _execute("video_prompt_evaluation", state, operation)


def render_approval_node(state: VideoAgentState) -> dict[str, Any]:
    project = get_project(state["project_id"])
    if project and project.get("status") != "awaiting_video_prompt_review":
        project["status"] = "awaiting_video_prompt_review"
        save_project(project)
    raw_decision = interrupt({
        "review_type": "render", "project_id": state["project_id"],
        "content": state.get("video_prompts", []),
        "evaluation": state.get("evaluation_results", {}).get("video_prompts", {}),
        "allowed_actions": ["approve", "terminate"],
    })
    decision = HumanReviewDecision.model_validate(raw_decision)
    return {
        "human_decision": decision.model_dump(mode="json"),
        "human_review_type": "render", "waiting_for_human": False,
        "current_stage": "render_approval", "completed_nodes": ["render_approval"],
        "execution_logs": [_human_log("render_approval", state, decision.action)],
    }


def video_render_node(state: VideoAgentState) -> dict[str, Any]:
    return _execute("video_render", state, lambda: {
        "final_video": tools.render_video(state["project_id"]),
    })


def final_evaluation_node(state: VideoAgentState) -> dict[str, Any]:
    def operation():
        video = state.get("final_video") or {}
        evaluations = dict(state.get("evaluation_results", {}))
        evaluations["final_video"] = {
            "score": 5.0 if video.get("status") == "completed" else 1.0,
            "passed": video.get("status") == "completed",
            "issues": [] if video.get("status") == "completed" else ["视频未完成"],
            "recommended_action": "continue" if video.get("status") == "completed" else "abort",
        }
        return {"evaluation_results": evaluations}
    return _execute("final_evaluation", state, operation)


def result_packaging_node(state: VideoAgentState) -> dict[str, Any]:
    return _execute("result_packaging", state, lambda: {"waiting_for_human": False})


def error_handler_node(state: VideoAgentState) -> dict[str, Any]:
    project = get_project(state["project_id"])
    if project:
        project["status"] = "agent_error"
        project["agent_error"] = (state.get("errors") or [{"message": "用户终止"}])[-1]
        save_project(project)
    return {
        "current_stage": "error_handler", "waiting_for_human": False,
        "completed_nodes": ["error_handler"],
        "execution_logs": [_human_log("error_handler", state, "end")],
    }


def _human_log(name: str, state: VideoAgentState, next_route: str) -> dict[str, Any]:
    return {
        "project_id": state["project_id"],
        "thread_id": state["thread_id"],
        "node": name,
        "started_at": datetime.now(UTC).isoformat(),
        "duration_ms": 0.0,
        "status": "completed",
        "providers": dict(state.get("selected_providers", {})),
        "input_summary": {"review_type": state.get("human_review_type")},
        "output_summary": ["human_decision"],
        "next_route": next_route,
    }


def _latest_evaluation_score(update: dict[str, Any]) -> float | None:
    evaluations = update.get("evaluation_results")
    if not isinstance(evaluations, dict):
        return None
    for result in reversed(list(evaluations.values())):
        if isinstance(result, dict) and isinstance(result.get("score"), (int, float)):
            return float(result["score"])
    return None
