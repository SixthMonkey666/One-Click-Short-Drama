"""Qwen3-VL provider implementation for local LLM inference."""

from __future__ import annotations

from typing import Any, Generator

from app.local_models import (
    generate_qwen_analysis_stream,
    generate_qwen_characters_stream,
    generate_qwen_storyboard_stream,
    generate_qwen_video_prompts_stream,
    generate_random_idea_stream,
    release_qwen,
)
from app.providers.base import LLMProvider


class QwenVLProvider(LLMProvider):
    """Local Qwen-VL model provider (Qwen3-VL-4B).

    Uses HuggingFace transformers with streaming TextIteratorStreamer.
    Runs on MPS (Apple Silicon), CUDA, or CPU depending on hardware.
    """

    name = "qwen3_vl"
    display_name = "Qwen3-VL-4B (本地)"

    def generate_idea_stream(
        self,
    ) -> Generator[dict[str, Any] | str, None, list[dict[str, Any]]]:
        return generate_random_idea_stream()

    def generate_analysis_stream(
        self, source_text: str
    ) -> Generator[dict[str, Any], None, dict[str, Any]]:
        return generate_qwen_analysis_stream(source_text)

    def generate_characters_stream(
        self, source_text: str
    ) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
        gen = generate_qwen_characters_stream(source_text)
        while True:
            try:
                event = next(gen)
                yield event
            except StopIteration as e:
                return e.value

    def generate_storyboard_stream(
        self, source_text: str, analysis: dict[str, Any]
    ) -> Generator[dict[str, Any], None, tuple[str, list[dict[str, Any]]]]:
        gen = generate_qwen_storyboard_stream(source_text, analysis)
        while True:
            try:
                event = next(gen)
                yield event
            except StopIteration as e:
                return e.value

    def generate_video_prompts_stream(
        self,
        source_text: str,
        script: str,
        shots: list[dict[str, Any]],
        characters: list[dict[str, Any]] | None = None,
    ) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
        gen = generate_qwen_video_prompts_stream(source_text, script, shots, characters)
        while True:
            try:
                event = next(gen)
                yield event
            except StopIteration as e:
                return e.value

    def release(self) -> None:
        release_qwen()

    @staticmethod
    def is_available() -> bool:
        from app.config import LOCAL_QWEN_MODEL_DIR
        return LOCAL_QWEN_MODEL_DIR.exists()
