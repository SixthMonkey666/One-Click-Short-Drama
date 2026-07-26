from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class VideoAgentState(TypedDict, total=False):
    project_id: str
    thread_id: str
    user_goal: str
    constraints: dict[str, Any]
    execution_plan: dict[str, Any]
    current_stage: str
    analysis: dict[str, Any]
    visual_bible: dict[str, Any]
    storyboard: list[dict[str, Any]]
    assets: dict[str, Any]
    video_prompts: list[dict[str, Any]]
    final_video: dict[str, Any] | None
    evaluation_results: dict[str, Any]
    feedback: dict[str, Any]
    retry_counts: dict[str, int]
    failed_items: list[dict[str, Any]]
    selected_providers: dict[str, str]
    estimated_cost: float
    errors: Annotated[list[dict[str, Any]], operator.add]
    human_decision: dict[str, Any] | None
    human_review_type: str | None
    waiting_for_human: bool
    completed_nodes: Annotated[list[str], operator.add]
    execution_logs: Annotated[list[dict[str, Any]], operator.add]
