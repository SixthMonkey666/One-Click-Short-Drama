from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator


class JudgeValidationError(ValueError):
    pass


class JudgeResult:
    def __init__(
        self,
        dimension_scores: dict[str, int],
        total_score: float,
        bad_case_tags: list[str],
        explanation: str,
        confidence: float,
        raw_response: str = "",
    ):
        self.dimension_scores = dimension_scores
        self.total_score = total_score
        self.bad_case_tags = bad_case_tags
        self.explanation = explanation
        self.confidence = confidence
        self.raw_response = raw_response

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension_scores": dict(self.dimension_scores),
            "total_score": self.total_score,
            "bad_case_tags": list(self.bad_case_tags),
            "explanation": self.explanation,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
        }


class JudgeProvider(ABC):
    name: str = "base"
    display_name: str = "Base Judge"
    prompt_version: str = "v1.0"
    model_version: str = "unknown"

    @abstractmethod
    def evaluate_image(
        self,
        image_path: Path,
        prompt: str,
        task_type: str,
        rubric: dict[str, Any],
        reference_images: list[Path] | None = None,
    ) -> JudgeResult:
        ...

    def evaluate_image_stream(
        self,
        image_path: Path,
        prompt: str,
        task_type: str,
        rubric: dict[str, Any],
        reference_images: list[Path] | None = None,
    ) -> Generator[dict[str, Any], None, JudgeResult]:
        yield {"type": "status", "message": f"正在使用 {self.display_name} 评测..."}
        result = self.evaluate_image(image_path, prompt, task_type, rubric, reference_images)
        yield {"type": "judge_complete", "result": result.to_dict()}
        return result

    def evaluate_validated(
        self,
        image_path: Path,
        prompt: str,
        task_type: str,
        rubric: dict[str, Any],
        reference_images: list[Path] | None = None,
        max_retries: int = 0,
    ) -> JudgeResult:
        last_error: Exception | None = None
        for _attempt in range(max(0, max_retries) + 1):
            try:
                result = self.evaluate_image(
                    image_path,
                    prompt,
                    task_type,
                    rubric,
                    reference_images,
                )
                return validate_judge_result(result, rubric)
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def release(self) -> None:
        pass

    @classmethod
    def is_available(cls) -> bool:
        return True


def validate_judge_result(
    result: JudgeResult,
    rubric: dict[str, Any],
) -> JudgeResult:
    if not isinstance(result, JudgeResult):
        raise JudgeValidationError("Judge provider returned an invalid result type")

    dimensions = rubric.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise JudgeValidationError("Rubric must define at least one dimension")

    known_keys = {
        str(dimension.get("key"))
        for dimension in dimensions
        if dimension.get("key")
    }
    required_keys = {
        str(dimension.get("key"))
        for dimension in dimensions
        if dimension.get("key") and dimension.get("required", True)
    }
    if not isinstance(result.dimension_scores, dict):
        raise JudgeValidationError("dimension_scores must be an object")
    score_keys = set(result.dimension_scores)
    missing = required_keys - score_keys
    unknown = score_keys - known_keys
    if missing:
        raise JudgeValidationError(
            f"Missing required dimension scores: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise JudgeValidationError(
            f"Unknown dimension scores: {', '.join(sorted(unknown))}"
        )

    normalized_scores: dict[str, int] = {}
    for key, value in result.dimension_scores.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise JudgeValidationError(f"Dimension {key!r} must be an integer")
        if value < 1 or value > 5:
            raise JudgeValidationError(f"Dimension {key!r} must be between 1 and 5")
        normalized_scores[key] = value

    if not isinstance(result.bad_case_tags, list):
        raise JudgeValidationError("bad_case_tags must be an array")
    valid_tags = set(rubric.get("bad_case_tags") or [])
    invalid_tags = [
        tag
        for tag in result.bad_case_tags
        if not isinstance(tag, str) or tag not in valid_tags
    ]
    if invalid_tags:
        raise JudgeValidationError("Judge result contains invalid bad-case tags")

    if not isinstance(result.explanation, str):
        raise JudgeValidationError("Judge result explanation must be a string")
    explanation = result.explanation.strip()
    if not explanation:
        raise JudgeValidationError("Judge result explanation is required")
    if isinstance(result.confidence, bool) or not isinstance(result.confidence, (int, float)):
        raise JudgeValidationError("Judge confidence must be numeric")
    confidence = float(result.confidence)
    if confidence < 0.0 or confidence > 1.0:
        raise JudgeValidationError("Judge confidence must be between 0 and 1")

    weights = {
        str(dimension["key"]): float(dimension.get("weight", 1.0))
        for dimension in dimensions
        if dimension.get("key")
    }
    scored_weight = sum(weights[key] for key in normalized_scores)
    if scored_weight <= 0:
        raise JudgeValidationError("Judge result has no weighted scores")
    total_score = round(
        sum(score * weights[key] for key, score in normalized_scores.items())
        / scored_weight,
        2,
    )

    return JudgeResult(
        dimension_scores=normalized_scores,
        total_score=total_score,
        bad_case_tags=list(dict.fromkeys(result.bad_case_tags)),
        explanation=explanation,
        confidence=confidence,
        raw_response=str(result.raw_response or ""),
    )
