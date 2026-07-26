from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AllowedStep = Literal[
    "requirement_analysis", "visual_bible", "storyboard_generation",
    "character_generation", "prop_generation", "environment_generation",
    "keyframe_generation", "video_prompt_generation", "video_render",
    "final_evaluation",
]

ALLOWED_STEPS = [
    "requirement_analysis", "visual_bible", "storyboard_generation",
    "character_generation", "prop_generation", "environment_generation",
    "keyframe_generation", "video_prompt_generation", "video_render",
    "final_evaluation",
]


class PlannedStep(BaseModel):
    name: AllowedStep
    required: bool = True
    reason: str
    requires_human_review: bool = False


class ProviderSelection(BaseModel):
    llm: str
    image: str
    video: str
    judge: str


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_type: str = "叙事短片"
    target_platform: str = "通用"
    duration_seconds: int = Field(default=20, ge=1, le=600)
    aspect_ratio: str = "16:9"
    steps: list[PlannedStep]
    providers: ProviderSelection
    evaluation_threshold: float = Field(default=3.5, ge=1, le=5)
    max_storyboard_retries: int = Field(default=2, ge=0, le=5)

    @field_validator("steps")
    @classmethod
    def require_storyboard(cls, value: list[PlannedStep]) -> list[PlannedStep]:
        if not any(step.name == "storyboard_generation" for step in value):
            raise ValueError("计划必须包含 storyboard_generation")
        return value


class RepairAction(BaseModel):
    target: str
    action: Literal[
        "split_shot", "shorten_shot", "extend_shot", "restore_costume",
        "simplify_action", "complete_field", "regenerate", "remove_risk",
    ]
    reason: str


class EvaluationResult(BaseModel):
    score: float = Field(ge=1, le=5)
    passed: bool
    issues: list[str] = Field(default_factory=list)
    feedback: str = ""
    failed_dimensions: list[str] = Field(default_factory=list)
    recommended_action: Literal["continue", "repair", "human_review", "abort"]
    high_risk: bool = False


class ReflectionFeedback(BaseModel):
    failed_dimensions: list[str]
    repair_actions: list[RepairAction]
    summary: str


class HumanReviewDecision(BaseModel):
    action: Literal["approve", "edit_and_continue", "regenerate", "terminate"]
    edited_storyboard: list[dict[str, Any]] | None = None
    note: str = ""

    @field_validator("edited_storyboard")
    @classmethod
    def edits_required_for_edit_action(cls, value, info):
        if info.data.get("action") == "edit_and_continue" and not value:
            raise ValueError("edit_and_continue 必须提供 edited_storyboard")
        return value
