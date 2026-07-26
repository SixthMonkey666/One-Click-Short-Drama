from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agent.checkpoint import get_checkpointer, thread_config
from app.agent.nodes import (
    asset_evaluation_node,
    asset_generation_node,
    asset_human_review_node,
    asset_planning_node,
    asset_repair_node,
    error_handler_node,
    final_evaluation_node,
    planner_node,
    render_approval_node,
    requirement_analysis_node,
    result_packaging_node,
    storyboard_evaluation_node,
    storyboard_generation_node,
    storyboard_human_review_node,
    storyboard_reflection_node,
    video_prompt_evaluation_node,
    video_prompt_generation_node,
    video_render_node,
    visual_bible_node,
)
from app.agent.routers import (
    route_asset_evaluation,
    route_asset_human_decision,
    route_render_decision,
    route_storyboard_evaluation,
    route_storyboard_human_decision,
    route_video_prompt_evaluation,
)
from app.agent.state import VideoAgentState
from app.repository import get_project


def _next_or_error(next_node: str):
    def route(state: VideoAgentState) -> str:
        return "error_handler" if state.get("errors") else next_node
    return route


def build_video_agent_graph(checkpointer=None):
    builder = StateGraph(VideoAgentState)
    nodes = {
        "requirement_analysis": requirement_analysis_node,
        "planner": planner_node,
        "visual_bible": visual_bible_node,
        "storyboard_generation": storyboard_generation_node,
        "storyboard_evaluation": storyboard_evaluation_node,
        "storyboard_reflection": storyboard_reflection_node,
        "storyboard_human_review": storyboard_human_review_node,
        "asset_planning": asset_planning_node,
        "asset_generation": asset_generation_node,
        "asset_evaluation": asset_evaluation_node,
        "asset_human_review": asset_human_review_node,
        "asset_repair": asset_repair_node,
        "video_prompt_generation": video_prompt_generation_node,
        "video_prompt_evaluation": video_prompt_evaluation_node,
        "render_approval": render_approval_node,
        "video_render": video_render_node,
        "final_evaluation": final_evaluation_node,
        "result_packaging": result_packaging_node,
        "error_handler": error_handler_node,
    }
    for name, node in nodes.items():
        builder.add_node(name, node)

    builder.add_edge(START, "requirement_analysis")
    for source, target in (
        ("requirement_analysis", "planner"),
        ("planner", "visual_bible"),
        ("visual_bible", "storyboard_generation"),
        ("storyboard_generation", "storyboard_evaluation"),
    ):
        builder.add_conditional_edges(source, _next_or_error(target), [target, "error_handler"])
    builder.add_conditional_edges(
        "storyboard_evaluation", route_storyboard_evaluation,
        ["storyboard_human_review", "storyboard_reflection", "error_handler"],
    )
    builder.add_edge("storyboard_reflection", "storyboard_generation")
    builder.add_conditional_edges(
        "storyboard_human_review", route_storyboard_human_decision,
        ["asset_planning", "storyboard_reflection", "error_handler"],
    )
    builder.add_edge("asset_planning", "asset_generation")
    builder.add_edge("asset_generation", "asset_evaluation")
    builder.add_conditional_edges(
        "asset_evaluation", route_asset_evaluation,
        ["video_prompt_generation", "asset_repair", "asset_human_review", "error_handler"],
    )
    builder.add_edge("asset_repair", "asset_evaluation")
    builder.add_conditional_edges(
        "asset_human_review", route_asset_human_decision,
        ["video_prompt_generation", "asset_repair", "error_handler"],
    )
    builder.add_edge("video_prompt_generation", "video_prompt_evaluation")
    builder.add_conditional_edges(
        "video_prompt_evaluation", route_video_prompt_evaluation,
        ["render_approval", "video_prompt_generation", "error_handler"],
    )
    builder.add_conditional_edges(
        "render_approval", route_render_decision, ["video_render", "error_handler"],
    )
    builder.add_conditional_edges(
        "video_render", _next_or_error("final_evaluation"),
        ["final_evaluation", "error_handler"],
    )
    builder.add_conditional_edges(
        "final_evaluation", _next_or_error("result_packaging"),
        ["result_packaging", "error_handler"],
    )
    builder.add_edge("result_packaging", END)
    builder.add_edge("error_handler", END)
    return builder.compile(checkpointer=checkpointer or get_checkpointer(), name="storyboard-video-agent")


@lru_cache(maxsize=1)
def get_video_agent_graph():
    return build_video_agent_graph()


def _initial_state(project_id: str, user_input: str | None = None) -> VideoAgentState:
    project = get_project(project_id)
    if not project:
        raise ValueError(f"项目不存在：{project_id}")
    return {
        "project_id": project_id,
        "thread_id": project_id,
        "user_goal": user_input or project.get("source_text", ""),
        "constraints": {},
        "retry_counts": {},
        "evaluation_results": {},
        "feedback": {},
        "failed_items": [],
        "errors": [],
        "completed_nodes": [],
        "execution_logs": [],
        "waiting_for_human": False,
        "estimated_cost": 0.0,
    }


def stream_agent(
    project_id: str,
    user_input: str | None = None,
    decision: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    graph = get_video_agent_graph()
    graph_input: VideoAgentState | Command
    graph_input = Command(resume=decision) if decision is not None else _initial_state(project_id, user_input)
    for mode, payload in graph.stream(
        graph_input,
        thread_config(project_id),
        stream_mode=["updates", "custom"],
    ):
        yield {"stream_mode": mode, "data": payload}


def start_agent(project_id: str, user_input: str | None = None) -> dict[str, Any]:
    for _ in stream_agent(project_id, user_input=user_input):
        pass
    return get_agent_state(project_id)


def resume_agent(project_id: str, human_decision: dict[str, Any]) -> dict[str, Any]:
    for _ in stream_agent(project_id, decision=human_decision):
        pass
    return get_agent_state(project_id)


def get_agent_state(project_id: str) -> dict[str, Any]:
    snapshot = get_video_agent_graph().get_state(thread_config(project_id))
    interrupts = [item.value for item in snapshot.interrupts]
    return {
        "values": dict(snapshot.values or {}),
        "next": list(snapshot.next),
        "interrupts": interrupts,
        "waiting_for_human": bool(interrupts),
        "created_at": snapshot.created_at,
    }


def get_agent_history(project_id: str, limit: int = 20) -> list[dict[str, Any]]:
    history = []
    for snapshot in get_video_agent_graph().get_state_history(thread_config(project_id), limit=limit):
        history.append({
            "values": dict(snapshot.values or {}),
            "next": list(snapshot.next),
            "created_at": snapshot.created_at,
        })
    return history


def clear_agent_thread(project_id: str) -> None:
    get_checkpointer().delete_thread(project_id)
