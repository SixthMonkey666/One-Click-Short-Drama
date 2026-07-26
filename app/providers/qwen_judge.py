from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Generator

from app.config import (
    JUDGE_MAX_VIDEO_FRAMES,
    QWEN_JUDGE_CPU_FALLBACK,
    QWEN_JUDGE_MAX_INPUT_IMAGES,
)
from app.local_models import (
    LocalModelError,
    _is_mps_oom_error,
    extract_video_frames,
    qwen_vlm_judge_stream,
    release_qwen,
)
from app.providers.judge_base import JudgeProvider, JudgeResult


class QwenVLJudgeProvider(JudgeProvider):
    """Qwen3-VL based automatic judge using local VLM model.

    Evaluates images (and extracted video frames) against a structured rubric,
    producing dimension scores, bad case tags, and explanations.
    Falls back gracefully when model is unavailable or vision processing fails.
    """

    name = "qwen3_vl"
    display_name = "Qwen3-VL-4B 自动机评"
    prompt_version = "v1.0-vlm"
    model_version = "Qwen3-VL-4B"

    def __init__(self) -> None:
        self._model_loaded = False
        self._device_override: str | None = None
        self._status_callback = None

    def set_status_callback(self, callback) -> None:
        self._status_callback = callback

    def _report_status(self, message: str, **details: Any) -> None:
        if self._status_callback is not None:
            self._status_callback({
                "type": "status",
                "message": message,
                **details,
            })

    def evaluate_image(
        self,
        image_path: Path,
        prompt: str,
        task_type: str,
        rubric: dict[str, Any],
        reference_images: list[Path] | None = None,
    ) -> JudgeResult:
        with tempfile.TemporaryDirectory(prefix="vlm_judge_frames_") as temp_dir:
            if task_type == "video" or image_path.suffix.lower() in (
                ".mp4",
                ".mov",
                ".avi",
                ".webm",
            ):
                images = extract_video_frames(
                    image_path,
                    Path(temp_dir),
                    max_frames=JUDGE_MAX_VIDEO_FRAMES,
                )
                if not images:
                    raise LocalModelError("视频抽帧失败，无法进行视觉评测")
            else:
                images = [image_path]

            if reference_images:
                images.extend(
                    reference
                    for reference in reference_images
                    if reference and reference.exists()
                )
            images = [path for path in images if path and path.exists()]
            images = images[:QWEN_JUDGE_MAX_INPUT_IMAGES]
            if not images:
                raise LocalModelError("无有效图片可供视觉评测")

            result_data = None
            last_error: Exception | None = None
            attempts = [self._device_override]
            if self._device_override is None and QWEN_JUDGE_CPU_FALLBACK:
                attempts.append("cpu")
            for attempt_device in attempts:
                try:
                    self._report_status(
                        f"正在使用 {(attempt_device or 'MPS').upper()} 评测当前素材",
                        device=attempt_device or "mps",
                    )
                    generator_kwargs = (
                        {"device_override": attempt_device}
                        if attempt_device is not None
                        else {}
                    )
                    generator = qwen_vlm_judge_stream(
                        images,
                        prompt,
                        task_type,
                        rubric,
                        **generator_kwargs,
                    )
                    while True:
                        try:
                            next(generator)
                        except StopIteration as stop:
                            result_data = stop.value
                            break
                    if result_data:
                        self._device_override = attempt_device
                        break
                except Exception as exc:
                    last_error = exc
                    should_fallback = (
                        attempt_device is None
                        and QWEN_JUDGE_CPU_FALLBACK
                        and _is_mps_oom_error(exc)
                    )
                    if not should_fallback:
                        raise
                    release_qwen()
                    self._device_override = "cpu"
                    self._report_status(
                        "MPS 内存不足，已卸载模型并切换到 CPU 保守模式重试；"
                        "速度会变慢，但不会降低评分维度和输出标准。",
                        device="cpu",
                        fallback_reason="mps_oom",
                    )

            if not result_data and last_error is not None:
                raise last_error

            if not result_data:
                raise LocalModelError("VLM 未返回有效评测结果")

            return JudgeResult(
                dimension_scores=result_data.get("dimension_scores", {}),
                total_score=0.0,
                bad_case_tags=result_data.get("bad_case_tags", []),
                explanation=result_data.get("explanation", ""),
                confidence=result_data.get("confidence", 0.5),
                raw_response=result_data.get("raw_response", ""),
            )

    def evaluate_image_stream(
        self,
        image_path: Path,
        prompt: str,
        task_type: str,
        rubric: dict[str, Any],
        reference_images: list[Path] | None = None,
    ) -> Generator[dict[str, Any], None, JudgeResult]:
        images: list[Path] = []
        temp_dir: Path | None = None

        if task_type == "video" or (isinstance(image_path, Path) and image_path.suffix.lower() in (".mp4", ".mov", ".avi", ".webm")):
            temp_dir = Path(tempfile.mkdtemp(prefix="vlm_judge_frames_"))
            yield {"type": "status", "message": "[VLM] 正在抽帧..."}
            frames = extract_video_frames(
                image_path, temp_dir, max_frames=JUDGE_MAX_VIDEO_FRAMES
            )
            images = frames if frames else []
            if images:
                yield {"type": "status", "message": f"[VLM] 已抽取{len(images)}帧，开始VLM推理..."}
        else:
            images = [image_path]

        if reference_images:
            images.extend([r for r in reference_images if r and r.exists()])

        images = [p for p in images if p and Path(p).exists()]
        images = images[:QWEN_JUDGE_MAX_INPUT_IMAGES]
        if not images:
            if temp_dir:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            result = JudgeResult(
                dimension_scores={}, total_score=0.0, bad_case_tags=[],
                explanation="[错误] 无有效图片可供评测", confidence=0.0, raw_response="no_valid_images",
            )
            yield {"type": "judge_complete", "result": result.to_dict()}
            return result

        raw_parts: list[str] = []
        try:
            generator_kwargs = (
                {"device_override": self._device_override}
                if self._device_override is not None
                else {}
            )
            gen = qwen_vlm_judge_stream(
                images,
                prompt,
                task_type,
                rubric,
                **generator_kwargs,
            )
            while True:
                try:
                    event = next(gen)
                    etype = event.get("type")
                    if etype == "status":
                        yield event
                    elif etype == "judge_delta":
                        raw_parts.append(event.get("text", ""))
                        yield {"type": "judge_delta", "text": event.get("text", "")}
                except StopIteration as e:
                    result_data = e.value
                    break
        except LocalModelError as err:
            if temp_dir:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            result = JudgeResult(
                dimension_scores={}, total_score=0.0, bad_case_tags=[],
                explanation=f"[模型错误] {err}", confidence=0.0, raw_response=str(err),
            )
            yield {"type": "judge_complete", "result": result.to_dict()}
            return result

        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

        dim_scores = result_data.get("dimension_scores", {})
        total_weight = 0.0
        weighted_sum = 0.0
        for d in rubric.get("dimensions", []):
            key = d["key"]
            if key in dim_scores:
                w = d.get("weight", 1.0)
                weighted_sum += dim_scores[key] * w
                total_weight += w
        total_score = round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0

        result = JudgeResult(
            dimension_scores=dim_scores,
            total_score=total_score,
            bad_case_tags=result_data.get("bad_case_tags", []),
            explanation=result_data.get("explanation", ""),
            confidence=result_data.get("confidence", 0.5),
            raw_response=result_data.get("raw_response", ""),
        )
        yield {"type": "judge_complete", "result": result.to_dict()}
        return result

    def release(self) -> None:
        release_qwen()

    @classmethod
    def is_available(cls) -> bool:
        try:
            from app.config import LOCAL_QWEN_MODEL_DIR
            return LOCAL_QWEN_MODEL_DIR.exists()
        except Exception:
            return False
