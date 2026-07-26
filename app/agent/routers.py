from __future__ import annotations

from app.agent.state import VideoAgentState
from app.config import (
    AGENT_MAX_ASSET_RETRIES,
    AGENT_MAX_STORYBOARD_RETRIES,
    AGENT_MAX_VIDEO_PROMPT_RETRIES,
)


def route_storyboard_evaluation(state: VideoAgentState) -> str:
    if state.get("errors"):
        return "error_handler"
    result = state.get("evaluation_results", {}).get("storyboard", {})
    retries = state.get("retry_counts", {}).get("storyboard", 0)
    if result.get("passed") or result.get("high_risk") or retries >= AGENT_MAX_STORYBOARD_RETRIES:
        return "storyboard_human_review"
    return "storyboard_reflection"


def route_storyboard_human_decision(state: VideoAgentState) -> str:
    action = (state.get("human_decision") or {}).get("action")
    if action in {"approve", "edit_and_continue"}:
        return "asset_planning"
    if action == "regenerate":
        return "storyboard_reflection"
    return "error_handler"


def route_asset_evaluation(state: VideoAgentState) -> str:
    if state.get("errors"):
        return "error_handler"
    result = state.get("evaluation_results", {}).get("assets", {})
    retries = state.get("retry_counts", {}).get("assets", 0)
    if result.get("passed"):
        return "video_prompt_generation"
    if retries < AGENT_MAX_ASSET_RETRIES:
        return "asset_repair"
    return "asset_human_review"


def route_asset_human_decision(state: VideoAgentState) -> str:
    action = (state.get("human_decision") or {}).get("action")
    if action == "approve":
        return "video_prompt_generation"
    if action == "regenerate":
        return "asset_repair"
    return "error_handler"


def route_video_prompt_evaluation(state: VideoAgentState) -> str:
    result = state.get("evaluation_results", {}).get("video_prompts", {})
    retries = state.get("retry_counts", {}).get("video_prompts", 0)
    if result.get("passed"):
        return "render_approval"
    if retries < AGENT_MAX_VIDEO_PROMPT_RETRIES:
        return "video_prompt_generation"
    return "render_approval"


def route_render_decision(state: VideoAgentState) -> str:
    action = (state.get("human_decision") or {}).get("action")
    return "video_render" if action in {"approve", "edit_and_continue"} else "error_handler"
