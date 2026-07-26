from __future__ import annotations

import base64
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generator
from unittest.mock import patch

import httpx
from PIL import Image

from app.providers.base import LLMProvider
from app.providers.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.openai_image import OpenAICompatibleImageProvider


def _consume(generator: Generator[Any, None, Any]) -> tuple[list[Any], Any]:
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as exc:
            return events, exc.value


class _FakeResponse:
    def __init__(
        self,
        *,
        lines: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._lines = lines or []
        self._payload = payload or {}
        self.status_code = 200
        self.text = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        yield from self._lines

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    last_path = ""
    analysis = {
        "title": "可替换模型测试",
        "genre": "科幻",
        "theme": "模型解耦",
        "tone": "明快",
        "visual_style": "cinematic",
        "total_duration_seconds": 20,
        "narrative_summary": "一个用于验证 Provider 的故事。",
        "characters": [
            {
                "name": "测试角色",
                "type": "人物",
                "description": "主角",
                "appearance": "blue coat",
            }
        ],
        "props": [],
        "environments": [
            {
                "name": "实验室",
                "description": "明亮实验室",
                "visual_prompt": "bright lab",
            }
        ],
    }
    image_bytes = b""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def stream(self, method: str, url: str, **kwargs: Any):
        del method, kwargs
        type(self).last_path = httpx.URL(url).path
        if url.endswith("/chat/completions"):
            content = json.dumps(self.analysis, ensure_ascii=False)
            chunks = [content[: len(content) // 2], content[len(content) // 2 :]]
            lines = [
                "data: "
                + json.dumps(
                    {"choices": [{"delta": {"content": chunk}}]},
                    ensure_ascii=False,
                )
                for chunk in chunks
            ]
            lines.append("data: [DONE]")
            return _FakeResponse(lines=lines)
        raise AssertionError(f"Unexpected stream URL: {url}")

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        del kwargs
        type(self).last_path = httpx.URL(url).path
        payload = {
            "data": [
                {
                    "b64_json": base64.b64encode(
                        self.image_bytes
                    ).decode()
                }
            ]
        }
        return _FakeResponse(payload=payload)


class DummyCustomLLMProvider(LLMProvider):
    name = "custom"
    display_name = "Test custom LLM"

    def generate_idea_stream(self):
        yield "idea"
        return [{"text": "idea", "topics": []}]

    def generate_analysis_stream(self, source_text):
        del source_text
        yield {}
        return {}

    def generate_characters_stream(self, source_text):
        del source_text
        yield {}
        return []

    def generate_storyboard_stream(self, source_text, analysis):
        del source_text, analysis
        yield {}
        return "", []

    def generate_video_prompts_stream(
        self, source_text, script, shots, characters=None
    ):
        del source_text, script, shots, characters
        yield {}
        return []


class TestProviderExtensibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), "blue").save(buffer, format="PNG")
        _FakeClient.image_bytes = buffer.getvalue()
        cls.base_url = "http://127.0.0.1:12345/v1"

    def test_openai_compatible_llm_runs_structured_workflow(self) -> None:
        with (
            patch(
                "app.providers.openai_compatible."
                "OPENAI_COMPATIBLE_LLM_BASE_URL",
                self.base_url,
            ),
            patch(
                "app.providers.openai_compatible."
                "OPENAI_COMPATIBLE_LLM_MODEL",
                "local-model",
            ),
            patch("httpx.Client", _FakeClient),
        ):
            events, analysis = _consume(
                OpenAICompatibleLLMProvider().generate_analysis_stream(
                    "测试创意"
                )
            )
        self.assertEqual(analysis["title"], "可替换模型测试")
        self.assertEqual(analysis["characters"][0]["id"], "char-01")
        self.assertTrue(
            any(event.get("type") == "text_delta" for event in events)
        )

    def test_openai_compatible_image_supports_generation_and_edits(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "generated.png"
            reference = root / "reference.png"
            Image.new("RGB", (8, 8), "red").save(reference)
            with (
                patch(
                    "app.providers.openai_image."
                    "OPENAI_COMPATIBLE_IMAGE_BASE_URL",
                    self.base_url,
                ),
                patch(
                    "app.providers.openai_image."
                    "OPENAI_COMPATIBLE_IMAGE_MODEL",
                    "local-image-model",
                ),
                patch("httpx.Client", _FakeClient),
            ):
                provider = OpenAICompatibleImageProvider()
                provider.generate_image(
                    "a test image", output, width=32, height=24
                )
                self.assertEqual(
                    _FakeClient.last_path,
                    "/v1/images/generations",
                )
                with Image.open(output) as image:
                    self.assertEqual(image.size, (32, 24))
                provider.generate_image(
                    "same character",
                    output,
                    width=32,
                    height=24,
                    reference_images=[reference],
                )
                self.assertEqual(
                    _FakeClient.last_path,
                    "/v1/images/edits",
                )

    def test_custom_provider_entrypoint_is_loaded_lazily(self) -> None:
        from app import providers

        entrypoint = f"{__name__}:DummyCustomLLMProvider"
        with (
            patch.object(
                providers, "CUSTOM_LLM_PROVIDER_CLASS", entrypoint
            ),
            patch.object(providers, "_LLM_PROVIDER_NAME", "custom"),
        ):
            registry = providers._llm_registry()
        self.assertIs(registry["custom"], DummyCustomLLMProvider)


if __name__ == "__main__":
    unittest.main()
